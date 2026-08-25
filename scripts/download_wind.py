#!/usr/bin/env python3
"""Download NES-LTER underway wind data for every cruise in the CTD catalog.

Data source
-----------
NES-LTER API (https://nes-lter-api.whoi.edu), endpoint
``/api/underway/{cruise}.csv`` — the same endpoint the ``nes-lter-mcp`` MCP
server (``query_underway`` / ``list_dataset_rows``) uses. The per-vessel
column resolution below mirrors the server's ``UNDERWAY_VARIABLE_ALIASES``
table, which was discovered/verified via the MCP tool ``resolve_variable``.

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
        "speed": [("wxtp_sm", 1.0), ("wxts_sm", 1.0)],
        # wxtp_dm/wxts_dm are relative to the bow, not true/absolute compass
        # bearings: verified across AR16/AR62/AT46 by comparing direction
        # stability during heading changes -- wxtp_dm alone shifts almost in
        # lockstep with heading (regression slope ~-0.7, median |delta| 31-42
        # deg during >10 deg heading changes), while (wxtp_dm + heading) is
        # far more stable (median |delta| 8-13 deg), confirming heading is the
        # correction needed to get true direction. See "direction_reference".
        "direction": [("wxtp_dm", 1.0), ("wxts_dm", 1.0)],
        "direction_reference": "relative_to_bow",
        "heading": [("hdt", 1.0)],
        "lat": ["dec_lat"],
        "lon": ["dec_lon"],
    },
    "sharp": {
        "speed": [("wind1_true_speed_kt", KT_TO_MS), ("wind2_true_speed_kt", KT_TO_MS)],
        "direction": [("wind1_true_dir_deg", 1.0), ("wind2_true_dir_deg", 1.0)],
        "lat": ["dec_lat"],
        "lon": ["dec_lon"],
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
    "-999",
    "-999.0",
    "-9999",
}


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
            return float(v)
        except ValueError:
            continue
    return None


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
            "n_wind_readings": 0,
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
            wd = first_float(row, [direction[0]])
            if wd is not None and heading is not None:
                hdg = first_float(row, [heading[0]])
                wd = (wd + hdg) % 360 if hdg is not None else None
            lat = first_float(row, WIND_ALIASES[family]["lat"])
            lon = first_float(row, WIND_ALIASES[family]["lon"])
            per_cruise.append(
                {
                    "date": (row.get("date") or "").strip(),
                    "cruise": name,
                    "vessel": vessel,
                    "season": record["season"],
                    "wind_speed_m_s": round(ws, 4),
                    "wind_dir_deg": round(wd, 1) if wd is not None else None,
                    "lat": round(lat, 4) if lat is not None else None,
                    "lon": round(lon, 4) if lon is not None else None,
                }
            )
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
            cruise_df.write_parquet(raw_dir / f"{name}.parquet")
            wind_frames.append(cruise_df)
            n_wind_readings += len(per_cruise)
        cruise_records.append(record)
        print(f"    rows={n_rows}, wind readings={len(per_cruise)}")

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
            "mcp": "Columns and endpoints were located via the nes-lter-mcp MCP server "
            "(tools: find_cruises, query_underway, get_dataset_schema, resolve_variable). "
            "Vessel->column mapping mirrors that server's UNDERWAY_VARIABLE_ALIASES, except "
            "two corrections that table does not have: the Endeavor speed conversion factor, "
            "and the Armstrong/Atlantis direction-to-heading correction (see notes below).",
        },
        "variables": {
            "wind_speed_m_s": "True wind speed at the bow anemometer, converted to m/s "
            "(Sharp, Atlantic Explorer, and Endeavor sensors report knots; "
            "the Endeavor unit was independently confirmed by reconstructing "
            "true wind vectorially from relative wind + heading + ship speed "
            "log, since the raw column name carries no unit and the MCP "
            "server's own alias table assumes m/s incorrectly).",
            "wind_dir_deg": "True wind direction, degrees from north (0-360). For Armstrong/"
            "Atlantis, wxtp_dm/wxts_dm are relative to the bow (0=dead ahead, "
            "clockwise), NOT true direction as their name suggests -- confirmed by "
            "comparing direction stability during heading changes across AR16/AR62/"
            "AT46 (raw value shifts almost in lockstep with heading, regression "
            "slope ~-0.7-0.8; adding heading back in, i.e. (dm + hdt) % 360, cuts "
            "the median instability from 31-42 deg to 8-13 deg during turns). The "
            "MCP server's alias table has this same gap (uses dm directly with no "
            "heading correction). This pipeline now corrects it using the true "
            "heading column (hdt); cruises where hdt isn't available are excluded "
            "rather than publishing uncorrected relative angles as absolute.",
        },
        "quality": "Rows with NODATA/NAN/missing sentinel values or non-physical speeds "
        "(<0 or >=100 m/s) are dropped; no other QA applied. Only true wind is used; "
        "cruises that only have relative wind are excluded (see cruise notes). "
        "No gust de-spiking — a small number of very large readings remain and are "
        "visible in the velocity distribution. "
        "Armstrong/Atlantis (wxtp_sm/wxts_sm) speed and direction were independently "
        "validated against OOI Pioneer Array METBK buoys (CP01CNSM, CP03ISSM, CP04OSSM -- "
        "moored directly in the NES-LTER sampling area), which report wind in m/s with no "
        "unit ambiguity. Matching ~72,600 Armstrong/Atlantis readings within 5 km and 10 "
        "min of a buoy gives median ship/buoy speed ratio 1.27 and median direction error "
        "7.7 deg; the same comparison for Endeavor (whose speed/units were separately "
        "confirmed correct) gives ratio 1.23 and error 7.6 deg over ~8,400 matches -- "
        "essentially identical, and consistent with ships' bow anemometers sitting well "
        "above a buoy's ~3-4 m sensor (higher wind speed with height is expected, not an "
        "error). This rules out both a m/s-vs-knots mislabeling and an uncorrected "
        "apparent-wind contamination for Armstrong/Atlantis speed: had either been true, "
        "its ratio would differ sharply from Endeavor's rather than closely tracking it.",
        "season_definition": "Cruise start month: winter={12,1,2}, spring={3,4,5}, summer={6,7,8}, fall={9,10,11} "
        "(same convention as the NES-LTER API / nes-lter-mcp find_cruises tool).",
        "cruises": cruise_records,
        "totals": {
            "cruises_in_catalog": len(cruises),
            "cruises_with_wind": sum(1 for r in cruise_records if r["status"] == "ok"),
            "wind_readings": n_wind_readings,
        },
    }
    with open(proc_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"\nDone: {provenance['totals']}")


if __name__ == "__main__":
    main()
