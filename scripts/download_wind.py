#!/usr/bin/env python3
"""Download NES-LTER underway wind data for every cruise in the CTD catalog.

Data source
-----------
NES-LTER API (https://nes-lter-api.whoi.edu), endpoint
``/api/underway/{cruise}.csv`` — the same endpoint the ``nes-lter-mcp`` MCP
server (``query_underway`` / ``list_dataset_rows``) uses. The per-vessel
column resolution below started from that server's
``UNDERWAY_VARIABLE_ALIASES`` table but now deliberately diverges from it: see
the Endeavor knots conversion and, more significantly, the Armstrong/Atlantis
true-vs-relative column choice, both documented inline in ``WIND_ALIASES``.
Column semantics are checked against ``/api/underway/column_definition/``
rather than guessed from column names.

Outputs
-------
``data/raw/{cruise}.parquet``      per-cruise wind readings (date, wind_speed_m_s,
                                   wind_dir_deg, lat, lon)
``data/processed/wind.parquet``    combined readings with cruise/vessel/season
``data/processed/cruises.csv``     cruise catalog with wind-availability status
``data/processed/provenance.json`` full provenance record

Data manipulation (assembling readings into tables, date parsing, filtering,
and both per-cruise and combined output) is done with polars. The two large,
per-reading tables are written as Parquet, which is both far smaller than CSV
for this data and preserves the parsed datetime type, so the notebook that
reads them back doesn't need to re-parse timestamps at all. The small
cruise-catalog metadata stays CSV since it's human-readable and not
performance-sensitive (74 rows).
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import polars as pl

API = "https://nes-lter-api.whoi.edu"
KT_TO_MS = 0.514444
SEASONS = {
    "winter": {12, 1, 2},
    "spring": {3, 4, 5},
    "summer": {6, 7, 8},
    "fall": {9, 10, 11},
}

VESSEL_FAMILIES = {
    "endeavor": "endeavor",
    "neil armstrong": "armstrong",
    "atlantis": "armstrong",
    "hugh r. sharp": "sharp",
    "atlantic explorer": "atlantic_explorer",
}


class VesselWindConfig(TypedDict, total=False):
    speed: list[tuple[str, float]]
    direction: list[tuple[str, float]]
    direction_reference: str
    heading: list[tuple[str, float]]
    lat: list[str]
    lon: list[str]


# friendly -> [(column, factor_to_m_s_or_deg), ...] in priority order
WIND_ALIASES: dict[str, VesselWindConfig] = {
    "endeavor": {
        # wind_truewindbow_speed is in knots, not m/s: verified by reconstructing
        # true wind vectorially from wind_gill_bow_windrelspd/windreldir +
        # gyro1_heading + speedlog_groundspeedfwd across ~4400 EN608 rows -- the
        # reconstruction only matches the reported true wind (median residual
        # 2.4%) if relative wind, true wind, and ship speed share one unit, and
        # the ship's speed log tops out at ~12.85, consistent with knots for a
        # vessel whose documented top speed is 10 kt (25 m/s would be impossible).
        "speed": [("wind_truewindbow_speed", KT_TO_MS)],
        "direction": [("wind_truewindbow_direction", 1.0)],
        "lat": ["gps_furuno_latitude"],
        "lon": ["gps_furuno_longitude"],
    },
    "armstrong": {
        # The Vaisala WXT units report BOTH relative and true wind, and this
        # pipeline previously published the relative pair (wxtp_sm/wxtp_dm) as
        # true wind. The API's own column_definition endpoint settles it:
        #
        #   WXTP_Sm | "Port Vaisala relative wind speed average" | m/s
        #   WXTP_Dm | "Port Vaisala relative wind direction average" | degrees
        #   WXTP_TS | "Port Vaisala True Wind Speed" | m/s
        #   WXTP_TD | "Port Vaisala True Wind Direction" | degrees
        #
        # so _ts/_td are the correct columns and need no heading correction.
        # Confirmed present on all 49 Armstrong/Atlantis cruises (including
        # AT46) via /api/underway/get_column_headers. On AR39B the old
        # published value (wxtp_sm) vs wxtp_ts has ratio p50 1.000 but p5 0.625
        # / p95 1.885 -- equal only while on station, diverging under way,
        # which is the signature of apparent wind. Units are m/s per the
        # definitions above, hence factor 1.0 (these are NOT knots).
        "speed": [("wxtp_ts", 1.0), ("wxts_ts", 1.0)],
        "direction": [("wxtp_td", 1.0), ("wxts_td", 1.0)],
        "lat": ["dec_lat"],
        "lon": ["dec_lon"],
    },
    "sharp": {
        # Two naming eras. HRS26xx use wind1/wind2_true_*; HRS2303 (2023) uses
        # true_wind_speed__knots (yes, a double underscore) and
        # true_wind_direction_deg, and was previously excluded outright because
        # neither spelling was listed here. Both eras report knots.
        "speed": [
            ("wind1_true_speed_kt", KT_TO_MS),
            ("wind2_true_speed_kt", KT_TO_MS),
            ("true_wind_speed__knots", KT_TO_MS),
            ("true_wind_speed_knots", KT_TO_MS),
        ],
        "direction": [
            ("wind1_true_dir_deg", 1.0),
            ("wind2_true_dir_deg", 1.0),
            ("true_wind_direction_deg", 1.0),
        ],
        # HRS2303 names its position columns latitude_deg/longitude_deg.
        "lat": ["dec_lat", "latitude_deg"],
        "lon": ["dec_lon", "longitude_deg"],
    },
    "atlantic_explorer": {
        "speed": [
            ("twindspdpri_kts", KT_TO_MS),
            ("twindspdsec_kts", KT_TO_MS),
            ("twindspdter_kts", KT_TO_MS),
        ],
        "direction": [
            ("twinddirpri_deg", 1.0),
            ("twinddirsec_deg", 1.0),
            ("twinddirter_deg", 1.0),
        ],
        "lat": ["latitude"],
        "lon": ["longitude"],
    },
}

MISSING = {
    "",
    "nan",
    "n/a",
    "nodata",
    "null",
    "none",
    "missing",
}

# Applied after float conversion rather than as strings, so -99, -99.0, -99.00
# and -9.9e1 all die by the same rule. HRS2303 uses -99 where other feeds use
# -999/-9999; matching on the string spelling alone missed it. None of these
# are physical for the five variables this script emits (speed, direction,
# lat, lon) -- this rule is scoped to those, not a general-purpose filter.
SENTINEL_NUMERICS = {-9999.0, -999.0, -99.0, 999.0, 9999.0}


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1} after {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def vessel_family(vessel_name: str | None) -> str | None:
    if not vessel_name:
        return None
    return next(
        (f for k, f in VESSEL_FAMILIES.items() if k in vessel_name.lower()), None
    )


def first_float(row: dict, candidates: list[str]) -> float | None:
    for col in candidates:
        v = row.get(col)
        if v is None:
            continue
        v = str(v).strip()
        if v.lower() in MISSING:
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if not math.isfinite(f) or f in SENTINEL_NUMERICS:
            continue
        return f
    return None


def valid_direction(wd: float | None) -> float | None:
    """Degrees from north, or None. Guards against sentinels reaching the output."""
    if wd is None or not (0 <= wd <= 360):
        return None
    return wd % 360


def valid_position(lat: float | None, lon: float | None) -> tuple[float | None, float | None]:
    """Drop implausible fixes as a pair -- half a position is not useful."""
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None, None
    return lat, lon


def resolve(
    candidates: list[tuple[str, float]], headers: set[str]
) -> tuple[str, float] | None:
    for col, factor in candidates:
        if col in headers:
            return col, factor
    return None


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    raw_dir = here / "data" / "raw"
    proc_dir = here / "data" / "processed"
    for d in (raw_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Fetching cruise catalog: {API}/api/ctd/cruises/all")
    cruises = json.loads(fetch(f"{API}/api/ctd/cruises/all"))

    wind_frames: list[pl.DataFrame] = []
    n_wind_readings = 0
    cruise_records: list[dict] = []
    for i, c in enumerate(cruises, 1):
        name, vessel = c["name"], c.get("vessel_name")
        family = vessel_family(vessel)
        record = {
            "name": name,
            "vessel": vessel,
            "vessel_family": family,
            "type": c.get("type"),
            "start_time": c.get("start_time"),
            "end_time": c.get("end_time"),
            "season": None,
            "n_feed_rows": 0,
            "n_wind_readings": 0,
            "n_duplicates_dropped": 0,
            "wind_speed_col": None,
            "wind_dir_col": None,
            "heading_col": None,
            "status": "ok",
            "notes": [],
        }
        start = c.get("start_time")
        if start:
            m = int(start[5:7])
            record["season"] = next(s for s, months in SEASONS.items() if m in months)
        print(f"[{i}/{len(cruises)}] {name} ({vessel}) ...", flush=True)

        if family is None:
            record["status"] = "skipped"
            record["notes"].append("unknown vessel; cannot resolve wind columns")
            cruise_records.append(record)
            continue

        try:
            body = fetch(f"{API}/api/underway/{name}.csv").decode("utf-8", "replace")
        except Exception as e:
            record["status"] = "error"
            record["notes"].append(f"fetch failed: {e}")
            cruise_records.append(record)
            continue

        reader = csv.DictReader(io.StringIO(body))
        headers = set(reader.fieldnames or [])
        speed = resolve(WIND_ALIASES[family]["speed"], headers)
        direction = resolve(WIND_ALIASES[family]["direction"], headers)
        if not speed or not direction:
            record["status"] = "no_wind_columns"
            rel = [h for h in headers if "relative_wind" in h]
            if rel:
                record["notes"].append(
                    f"only relative wind available ({rel}) — excluded to keep the dataset true-wind-only"
                )
            else:
                record["notes"].append(
                    f"no wind columns found; headers: {sorted(headers)[:15]}"
                )
            cruise_records.append(record)
            continue

        needs_heading = (
            WIND_ALIASES[family].get("direction_reference") == "relative_to_bow"
        )
        heading = (
            resolve(WIND_ALIASES[family].get("heading", []), headers)
            if needs_heading
            else None
        )
        if needs_heading and not heading:
            record["status"] = "no_wind_columns"
            record["notes"].append(
                f"direction column {direction[0]} is relative to the bow but no heading "
                f"column was found to correct it to true; excluded to avoid publishing "
                f"ship-relative angles as absolute direction"
            )
            cruise_records.append(record)
            continue

        record["wind_speed_col"] = speed[0]
        record["wind_dir_col"] = direction[0]
        if heading:
            record["heading_col"] = heading[0]

        per_cruise: list[dict] = []
        n_rows = 0
        for row in reader:
            n_rows += 1
            ws = first_float(row, [speed[0]])
            if ws is None:
                continue
            ws *= speed[1]
            if not (0 <= ws < 100):
                continue
            wd = valid_direction(first_float(row, [direction[0]]))
            if wd is not None and heading is not None:
                hdg = first_float(row, [heading[0]])
                wd = valid_direction(wd + hdg) if hdg is not None else None
            lat, lon = valid_position(
                first_float(row, WIND_ALIASES[family]["lat"]),
                first_float(row, WIND_ALIASES[family]["lon"]),
            )
            per_cruise.append(
                {
                    "date": (row.get("date") or "").strip(),
                    "cruise": name,
                    "vessel": vessel,
                    "season": record["season"],
                    "wind_speed_m_s": round(ws, 4),
                    # Re-wrap after rounding: 359.96 rounds to 360.0, which is
                    # outside the half-open [0, 360) the rest of the project
                    # (and the wind-rose sector binning) assumes.
                    "wind_dir_deg": round(wd, 1) % 360 if wd is not None else None,
                    "lat": round(lat, 4) if lat is not None else None,
                    "lon": round(lon, 4) if lon is not None else None,
                }
            )
        record["n_feed_rows"] = n_rows
        record["n_wind_readings"] = len(per_cruise)
        if not per_cruise:
            record["status"] = "no_wind_data"
            record["notes"].append(
                f"columns present ({speed[0]}, {direction[0]}) but {n_rows} rows had no valid values"
            )
        else:
            # Parse the date column once here (polars' auto-inference handles the
            # API's mix of second- and microsecond-precision timestamps within a
            # single cruise) so it's written to parquet as a real datetime -- the
            # notebook that reads it back needs no ISO8601 string parsing at all.
            cruise_df = pl.DataFrame(per_cruise).with_columns(
                pl.col("date")
                .str.to_datetime(strict=False, time_zone="UTC")
                .dt.replace_time_zone(None)
            )
            # Drop exact duplicate readings. The API has been observed serving a
            # doubled file (HRS2607 once came back as 4,726 distinct rows
            # repeated twice); it currently does not, so this is a standing
            # guard against recurrence rather than a one-time cleanup -- do not
            # remove it just because a run reports zero. Keyed on the whole
            # value tuple, never on 'date' alone: two rows sharing a timestamp
            # but carrying different values are real samples, and collapsing
            # them would be the same class of silent data loss this guards against.
            before = len(cruise_df)
            cruise_df = cruise_df.unique(keep="first", maintain_order=True)
            record["n_duplicates_dropped"] = before - len(cruise_df)
            record["n_wind_readings"] = len(cruise_df)
            if record["n_duplicates_dropped"]:
                record["notes"].append(
                    f"dropped {record['n_duplicates_dropped']} exact duplicate readings"
                )
            cruise_df.write_parquet(raw_dir / f"{name}.parquet")
            wind_frames.append(cruise_df)
            n_wind_readings += len(cruise_df)
        cruise_records.append(record)
        dupes = record["n_duplicates_dropped"]
        print(
            f"    rows={n_rows}, wind readings={record['n_wind_readings']}"
            + (f" (dropped {dupes} duplicates)" if dupes else "")
        )

    # combined processed file
    if wind_frames:
        pl.concat(wind_frames).write_parquet(proc_dir / "wind.parquet")

    cruises_df = pl.DataFrame(
        [{**r, "notes": "; ".join(r["notes"])} for r in cruise_records]
    )
    cruises_df.write_csv(proc_dir / "cruises.csv")

    provenance = {
        "project": "NES-LTER underway wind analysis",
        "fetched_at": fetched_at,
        "source": {
            "api_base": API,
            "cruise_catalog_endpoint": "/api/ctd/cruises/all",
            "underway_csv_endpoint": "/api/underway/{cruise}.csv",
            "column_definition_endpoint": "/api/underway/column_definition/{cruise}",
            "column_headers_endpoint": "/api/underway/get_column_headers/{cruise}",
            "mcp": "Columns and endpoints were originally located via the nes-lter-mcp MCP "
            "server (tools: find_cruises, query_underway, get_dataset_schema, "
            "resolve_variable). The vessel->column mapping no longer mirrors that server's "
            "UNDERWAY_VARIABLE_ALIASES: it now diverges on the Endeavor speed conversion "
            "factor and, more importantly, on which Armstrong/Atlantis columns are true "
            "wind at all (see variables below). Column choices are now checked against the "
            "API's own column_definition endpoint rather than inferred from column names.",
            "catalog_completeness": "The cruise catalog (/api/ctd/cruises/all, 74 cruises) was "
            "verified to be the complete underway universe by cross-checking the underway "
            "file store (/api/underway/find/{start}/{end}), the per-year cruise index pages, "
            "and the nutrient/chlorophyll datasets, plus negative-control probes of "
            "plausible-but-unlisted cruise IDs (all 404). No underway-only cruises exist "
            "outside this catalog.",
        },
        "variables": {
            "wind_speed_m_s": "True wind speed, m/s. Sharp, Atlantic Explorer, and Endeavor "
            "sensors report knots and are converted; Armstrong/Atlantis wxtp_ts/wxts_ts are "
            "already m/s per the API's column_definition. The Endeavor unit was "
            "independently confirmed by reconstructing true wind vectorially from relative "
            "wind + heading + ship speed log, since the raw column name carries no unit and "
            "the MCP server's own alias table assumes m/s incorrectly.",
            "wind_dir_deg": "True wind direction, degrees from north (0-360).",
            "armstrong_column_correction": "Armstrong/Atlantis previously used wxtp_sm/wxtp_dm "
            "with a (dm + hdt) % 360 heading correction. The API's column_definition "
            "endpoint documents those as 'Port Vaisala relative wind speed/direction "
            "average', while wxtp_ts/wxtp_td are 'Port Vaisala True Wind Speed/Direction' -- "
            "so the published speed was apparent, not true. This pipeline now uses "
            "wxtp_ts/wxtp_td (wxts_* as fallback), which are present on all 49 "
            "Armstrong/Atlantis cruises and need no heading correction. On AR39B the old "
            "and new speeds have ratio p50 1.000 but p5 0.625 / p95 1.885 -- identical only "
            "while on station, diverging under way, as apparent wind does. Note the old "
            "direction reconstruction was close to right: (dm + hdt) % 360 agrees with the "
            "vendor's wxtp_td to a 1.4 deg median on AR39B, so this change affects speed far "
            "more than direction.",
        },
        "quality": "Rows are dropped when speed is missing, non-finite, a sentinel, or "
        "non-physical (<0 or >=100 m/s). Missing values are recognized both as strings "
        "(NODATA/NAN/etc.) and numerically (-9999, -999, -99, 999, 9999) -- the numeric rule "
        "matters because HRS2303 uses -99 where other feeds use -999. Direction outside "
        "0-360 is nulled (the row is kept for its speed); positions outside physical "
        "lat/lon range are nulled as a pair. Exact duplicate readings are dropped per cruise "
        "(the API was once observed serving a doubled file for HRS2607); the dedup key is "
        "the whole value tuple, never the timestamp alone. Only true wind is used; cruises "
        "with only relative wind are excluded (see cruise notes). No gust de-spiking -- a "
        "small number of very large readings remain and are visible in the velocity "
        "distribution. "
        "BUOY VALIDATION (re-run against the corrected columns; see "
        "scripts/validate_buoys.py): matching ship readings to OOI Pioneer Array METBK "
        "buoys (CP01CNSM, CP03ISSM, CP04OSSM) within 5 km and 10 min gives, for "
        "Armstrong/Atlantis, median ship/buoy speed ratio 1.268 and median direction error "
        "7.2 deg over 72,263 matches -- versus 1.273 and 7.9 deg for the same comparison "
        "against the pre-correction (wxtp_sm/wxtp_dm) data. The Endeavor control is "
        "unchanged at 1.227 and 7.6 deg over 8,778 matches. "
        "IMPORTANT INTERPRETATION: the speed ratio barely moved, which does NOT vindicate "
        "the old columns. The buoy-matched readings are precisely the ones least affected "
        "by the correction -- near a mooring the correction shifts speed by a median of "
        "0.20 m/s, versus 0.40 m/s over the dataset as a whole -- so this comparison has "
        "roughly half the sensitivity to the true-vs-relative question and cannot "
        "adjudicate it in either direction. That is also why the original validation "
        "appeared to rule out apparent-wind contamination when it could not. The decisive "
        "evidence for the column change is the API's own column_definition metadata, not "
        "this comparison. What the re-run does show is that direction improved slightly "
        "(7.9 -> 7.2 deg), and that the residual 1.27-vs-1.23 gap against Endeavor is now a "
        "genuine inter-vessel difference (mast height and exposure) rather than "
        "apparent-wind inflation.",
        "season_definition": "Cruise start month: winter={12,1,2}, spring={3,4,5}, summer={6,7,8}, fall={9,10,11} "
        "(same convention as the NES-LTER API / nes-lter-mcp find_cruises tool).",
        "cruises": cruise_records,
        "totals": {
            "cruises_in_catalog": len(cruises),
            "cruises_with_wind": sum(1 for r in cruise_records if r["status"] == "ok"),
            "wind_readings": n_wind_readings,
            "feed_rows": sum(r["n_feed_rows"] for r in cruise_records),
            "duplicates_dropped": sum(
                r["n_duplicates_dropped"] for r in cruise_records
            ),
        },
    }
    with open(proc_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"\nDone: {provenance['totals']}")


if __name__ == "__main__":
    main()
