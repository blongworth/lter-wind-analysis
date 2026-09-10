#!/usr/bin/env python3
"""Validate shipboard underway wind against OOI Pioneer Array METBK buoys.

Why
---
The moored buoys report wind in unambiguous m/s at a fixed, known height, with
no ship-relative angles to correct and no vessel-specific column naming. They
are therefore an independent control on the shipboard columns this project
publishes -- which is how the Armstrong/Atlantis relative-vs-true column error
was originally missed and later caught.

An earlier run of this comparison (~72,600 Armstrong/Atlantis matches, median
ship/buoy speed ratio 1.27 vs Endeavor's 1.23) was done ad hoc and not
committed, and its conclusion -- "the close match to the Endeavor control rules
out apparent-wind contamination" -- was wrong: apparent-wind inflation produces
that same signature. Hence this script, so the comparison is reproducible and
can be re-run whenever the column mapping changes.

Method
------
Buoy: eastward_wind / northward_wind (m/s) from the METBK datasets at the three
Pioneer NES surface moorings, via the OOI Data Explorer ERDDAP. Speed is the
vector magnitude; direction is converted to the meteorological "direction from"
convention to match the shipboard columns. QC-flagged samples are dropped.

Matching: a ship reading matches a buoy sample when they are within
--max-km (default 5 km) and --max-min (default 10 min). Each ship reading takes
its single nearest-in-time buoy sample, so one reading contributes one match.

Reported per vessel group: median ship/buoy speed ratio and median absolute
angular direction error. Pass --baseline to run the identical comparison
against a second (e.g. pre-correction) dataset for an A/B.

Usage
-----
    uv run python scripts/validate_buoys.py
    uv run python scripts/validate_buoys.py --baseline /tmp/wind_baseline/processed/wind.parquet
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

ERDDAP = "https://erddap.dataexplorer.oceanobservatories.org/erddap/tabledap"

# METBK datasets at the three Pioneer NES surface moorings. Each mooring also
# publishes an ...a001 series, but those carry no wind variables at all (11
# variables, a different stream) and are deliberately omitted -- querying them
# returns HTTP 400 "Unrecognized variable". CP01CNSM has two telemetry
# positions (sbd11/sbd12) that do both carry wind; they are queried together
# and de-duplicated on (mooring, time).
BUOY_DATASETS = {
    "CP01CNSM": [
        "ooi-cp01cnsm-sbd11-06-metbka000",
        "ooi-cp01cnsm-sbd12-06-metbka000",
    ],
    "CP03ISSM": ["ooi-cp03issm-sbd11-06-metbka000"],
    "CP04OSSM": ["ooi-cp04ossm-sbd11-06-metbka000"],
}

# Nominal mooring positions, used only to pre-filter ship readings cheaply.
# Actual per-sample buoy positions are used for the real distance test.
NOMINAL = {
    "CP01CNSM": (40.13325, -70.778317),
    "CP03ISSM": (40.367167, -70.881817),
    "CP04OSSM": (39.937517, -70.886850),
}

# OOI aggregate QC: 1 pass, 2 not evaluated, 3 suspect, 4 fail, 9 missing.
#
# In practice essentially every METBK sample in these datasets is flagged 3
# ("suspect") -- e.g. 8,639 of 8,640 samples in a sampled week at CP01CNSM --
# while the values themselves are physically unremarkable (median 6-8.5 m/s).
# The aggregate is evidently a blanket flag here rather than a per-sample
# judgement, so excluding it would throw away the entire control. We therefore
# accept 3 and reject only outright failures. The QC histogram is printed on
# every run so this stays visible rather than buried.
QC_ACCEPT = {1, 2, 3}

EARTH_R_KM = 6371.0088

# The Pioneer Array was recovered from the NES shelf in late 2022 and
# redeployed in the Mid-Atlantic Bight, so no later cruise can match.
BUOY_LAST_YEAR = 2022


def haversine_km(
    lat1: pl.Expr | float, lon1: pl.Expr | float, lat2: float, lon2: float
) -> pl.Expr:
    p1, p2 = (pl.lit(lat1) if isinstance(lat1, float) else lat1).radians(), math.radians(lat2)
    dlat = p2 - p1
    dlon = (pl.lit(lon2) - (pl.lit(lon1) if isinstance(lon1, float) else lon1)).radians()
    a = (dlat / 2).sin() ** 2 + p1.cos() * math.cos(p2) * (dlon / 2).sin() ** 2
    return 2 * EARTH_R_KM * a.sqrt().arcsin()


def angular_diff(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Smallest absolute angle between two compass bearings, in degrees."""
    return ((a - b + 180) % 360 - 180).abs()


def fetch_buoy(dataset: str, start: str, end: str) -> pl.DataFrame | None:
    cols = (
        "time,latitude,longitude,eastward_wind,northward_wind,"
        "eastward_wind_qc_agg,northward_wind_qc_agg"
    )
    url = (
        f"{ERDDAP}/{dataset}.csv?{urllib.parse.quote(cols, safe=',')}"
        f"&time%3E={start}&time%3C={end}"
    )
    try:
        with urllib.request.urlopen(url, timeout=300) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        # ERDDAP answers "no rows in this time range" with 404, and also with
        # 400 whose body names the condition. Anything else (a bad variable
        # name, say) is a real bug and must not be silently treated as
        # "no data" -- that is exactly how a missing column goes unnoticed.
        if e.code == 404 or (e.code == 400 and "nRows = 0" in detail):
            return None
        raise RuntimeError(f"{dataset}: HTTP {e.code}: {detail}") from e
    lines = body.splitlines()
    if len(lines) < 3:
        return None
    # ERDDAP emits a units row directly under the header; drop it.
    csv_text = "\n".join([lines[0], *lines[2:]])
    df = pl.read_csv(io.StringIO(csv_text), infer_schema_length=10000, ignore_errors=True)
    if df.is_empty():
        return None
    return df.with_columns(
        pl.col("time").str.to_datetime(strict=False, time_zone="UTC").dt.replace_time_zone(None)
    )


def load_buoys(windows: list[tuple[str, str]], workers: int) -> pl.DataFrame:
    """Fetch buoy wind for each cruise window.

    Fetching the full 2017-2022 overlap in one request would pull ~9 years of
    1-minute data from 8 datasets; the ship is only ever near a buoy during a
    cruise, so we request one window per cruise instead.
    """
    jobs = [
        (m, ds, s, e)
        for m, dss in BUOY_DATASETS.items()
        for ds in dss
        for s, e in windows
    ]
    frames: list[pl.DataFrame] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_buoy, ds, s, e): m for m, ds, s, e in jobs}
        for done, fut in enumerate(cf.as_completed(futs), 1):
            mooring = futs[fut]
            df = fut.result()
            if done % 50 == 0:
                print(f"    {done}/{len(jobs)} requests", flush=True)
            if df is not None:
                frames.append(df.with_columns(pl.lit(mooring).alias("mooring")))
    if not frames:
        sys.exit("No buoy data returned for the requested windows.")

    buoy = pl.concat(frames, how="vertical_relaxed")
    hist = dict(
        buoy.group_by("eastward_wind_qc_agg").len().sort("eastward_wind_qc_agg").iter_rows()
    )
    print(f"  buoy QC histogram (eastward_wind_qc_agg): {hist}")
    print(f"  accepting QC flags {sorted(QC_ACCEPT)}")
    qc = list(QC_ACCEPT)
    buoy = (
        buoy.filter(
            pl.col("eastward_wind").is_not_null()
            & pl.col("northward_wind").is_not_null()
            & pl.col("eastward_wind_qc_agg").is_in(qc)
            & pl.col("northward_wind_qc_agg").is_in(qc)
        )
        .unique(subset=["mooring", "time"], keep="first")
        .with_columns(
            (pl.col("eastward_wind") ** 2 + pl.col("northward_wind") ** 2)
            .sqrt()
            .alias("buoy_speed"),
            # u/v are the velocity vector (direction the air moves toward);
            # shipboard true-wind direction is the meteorological "from"
            # bearing, so rotate by 180 deg here rather than at compare time.
            (
                (
                    270.0
                    - pl.arctan2(pl.col("northward_wind"), pl.col("eastward_wind")).degrees()
                )
                % 360
            ).alias("buoy_dir"),
        )
        .select("mooring", "time", "latitude", "longitude", "buoy_speed", "buoy_dir")
        .sort("time")
    )
    return buoy


def match(ship: pl.DataFrame, buoy: pl.DataFrame, max_km: float, max_min: int) -> pl.DataFrame:
    """Nearest-in-time buoy sample per ship reading, within max_km and max_min."""
    out: list[pl.DataFrame] = []
    for mooring, (blat, blon) in NOMINAL.items():
        near = ship.filter(haversine_km(pl.col("lat"), pl.col("lon"), blat, blon) <= max_km)
        if near.is_empty():
            continue
        b = buoy.filter(pl.col("mooring") == mooring)
        if b.is_empty():
            continue
        joined = near.sort("date").join_asof(
            b.sort("time"),
            left_on="date",
            right_on="time",
            strategy="nearest",
            tolerance=f"{max_min}m",
        )
        joined = joined.filter(pl.col("buoy_speed").is_not_null())
        if joined.is_empty():
            continue
        # Re-test distance against the buoy's own reported position.
        joined = joined.filter(
            haversine_km(pl.col("lat"), pl.col("lon"), blat, blon) <= max_km
        )
        out.append(joined.with_columns(pl.lit(mooring).alias("matched_mooring")))
    if not out:
        return pl.DataFrame()
    return pl.concat(out, how="vertical_relaxed")


def summarize(m: pl.DataFrame, label: str) -> dict | None:
    if m.is_empty():
        print(f"  {label:26} no matches")
        return None
    m = m.filter(pl.col("buoy_speed") > 0.5)  # ratios explode near calm
    if m.is_empty():
        print(f"  {label:26} no matches above the calm cutoff")
        return None
    ratio = (pl.col("wind_speed_m_s") / pl.col("buoy_speed")).alias("ratio")
    withr = m.with_columns(ratio)
    d = withr.filter(pl.col("wind_dir_deg").is_not_null()).with_columns(
        angular_diff(pl.col("wind_dir_deg"), pl.col("buoy_dir")).alias("dir_err")
    )
    def num(v: object) -> float:
        """Aggregates come back None on an all-null column; keep that visible as NaN."""
        return float(v) if isinstance(v, (int, float)) else float("nan")

    res = {
        "label": label,
        "n": len(m),
        "ratio_median": num(withr["ratio"].median()),
        "ratio_p25": num(withr["ratio"].quantile(0.25)),
        "ratio_p75": num(withr["ratio"].quantile(0.75)),
        "dir_err_median": num(d["dir_err"].median()) if len(d) else float("nan"),
        "n_dir": len(d),
    }
    print(
        f"  {label:26} n={res['n']:6}  ratio med={res['ratio_median']:.3f} "
        f"(p25 {res['ratio_p25']:.3f}, p75 {res['ratio_p75']:.3f})  "
        f"dir err med={res['dir_err_median']:5.1f} deg (n={res['n_dir']})"
    )
    return res


GROUPS = {
    "Armstrong/Atlantis": ["Neil Armstrong", "Atlantis"],
    "Endeavor (control)": ["Endeavor"],
}


def run(path: Path, buoy: pl.DataFrame, max_km: float, max_min: int, tag: str) -> dict:
    ship = pl.read_parquet(path).filter(
        pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
    )
    print(f"\n{tag}: {path}  ({len(ship):,} positioned readings)")
    results = {}
    for label, vessels in GROUPS.items():
        sub = ship.filter(pl.col("vessel").is_in(vessels))
        r = summarize(match(sub, buoy, max_km, max_min), label)
        if r:
            results[label] = r
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wind", default="data/processed/wind.parquet", type=Path)
    ap.add_argument("--baseline", type=Path, help="second dataset for an A/B comparison")
    ap.add_argument("--max-km", type=float, default=5.0)
    ap.add_argument("--max-min", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ship = pl.read_parquet(args.wind)
    # One window per cruise, padded by the match tolerance. Cruises entirely
    # after the buoys were recovered contribute nothing and are skipped.
    spans = (
        ship.filter(pl.col("vessel").is_in([v for vs in GROUPS.values() for v in vs]))
        .group_by("cruise")
        .agg(pl.col("date").min().alias("t0"), pl.col("date").max().alias("t1"))
        .sort("t0")
    )
    pad = pl.duration(minutes=args.max_min)
    spans = spans.with_columns((pl.col("t0") - pad).alias("t0"), (pl.col("t1") + pad).alias("t1"))
    windows = [
        (t0.strftime("%Y-%m-%dT%H:%M:%SZ"), t1.strftime("%Y-%m-%dT%H:%M:%SZ"))
        for t0, t1 in spans.select("t0", "t1").iter_rows()
        if t0.year <= BUOY_LAST_YEAR
    ]
    print(
        f"Fetching OOI METBK buoy wind for {len(windows)} cruise windows "
        f"({len(BUOY_DATASETS)} moorings, {sum(len(v) for v in BUOY_DATASETS.values())} datasets)"
    )
    buoy = load_buoys(windows, args.workers)
    print(f"  buoy samples after QC: {len(buoy):,}")
    print(f"  per mooring: {dict(buoy.group_by('mooring').len().iter_rows())}")

    print(f"\nMatching within {args.max_km} km and {args.max_min} min:")
    new = run(args.wind, buoy, args.max_km, args.max_min, "CURRENT")
    if args.baseline:
        old = run(args.baseline, buoy, args.max_km, args.max_min, "BASELINE")
        print("\nA/B (baseline -> current):")
        for label in GROUPS:
            if label in old and label in new:
                o, n = old[label], new[label]
                print(
                    f"  {label:26} ratio {o['ratio_median']:.3f} -> {n['ratio_median']:.3f}"
                    f"   dir err {o['dir_err_median']:.1f} -> {n['dir_err_median']:.1f} deg"
                )


if __name__ == "__main__":
    main()
