# NES-LTER wind analysis

Seasonal wind speed and direction statistics from NES-LTER underway shipboard data —
how often the wind exceeds a given threshold, by season and by cruise, plus a catalog
of sustained high-wind events.

**[View the notebook →](https://blongworth.github.io/lter-wind-analysis/)** (static
export, rebuilt from source data on every push to `main`)

## Quick start

```sh
uv sync
uv run python scripts/download_wind.py   # fetch + build data/ (~5 min)
uv run marimo edit notebooks/wind_analysis.py
```

`data/` is gitignored and fully reproducible from the download script, so it isn't
checked in.

## What's in the notebook

Two interactive controls — a threshold **x** (m/s) and a season filter — drive:

1. **Speed distribution by season** — density histograms with the threshold marked
1. **Survival curves** — S(v) = P(wind > v) per season, answering "what percentage of
   the time is wind above x?"
1. **Per-cruise fractions** — the same question broken out by cruise
1. **Wind roses** — 16 compass sectors per season, petal length = % of time from that
   sector, binned by speed
1. **High-wind event catalog** — sustained runs above the threshold, written to
   `data/processed/high_wind_events.csv`

All fractions are **time-weighted** (each reading weighted by seconds until the next
reading in the same cruise, capped at 1 h) rather than a naive reading count, since
underway sampling intervals vary between vessels and cruises.

## Layout

```
scripts/download_wind.py   fetch underway data, resolve per-vessel wind columns, build data/
notebooks/wind_analysis.py marimo notebook (holoviews/bokeh plots, polars)
data/raw/{cruise}.parquet  per-cruise wind readings
data/processed/            wind.parquet, cruises.csv, provenance.json, high_wind_events.csv
```

## Data source and caveats

Wind comes from the [NES-LTER API](https://nes-lter-api.whoi.edu) `/api/underway/`
endpoint, covering every cruise in the CTD catalog (73 of 74 cruises have usable true
wind, ~770k readings). That 74-cruise catalog was verified to be the complete underway
universe — cross-checked against the underway file store, the per-year cruise indexes,
and the nutrient/chlorophyll datasets, with negative-control probes for unlisted cruise
IDs. Column names differ per vessel, so the download script resolves them per vessel
family — Endeavor, Armstrong/Atlantis, Sharp, and Atlantic Explorer.

Corrections applied that the upstream alias tables don't have, all verified against the
data and documented in detail in `data/processed/provenance.json`:

- **Armstrong/Atlantis use `wxtp_ts`/`wxtp_td`, not `wxtp_sm`/`wxtp_dm`.** The API's
  `column_definition` endpoint documents the latter pair as *relative* wind speed and
  direction, and the former as True Wind Speed/Direction. This pipeline previously
  published the relative speed as true wind for 49 cruises (~72% of readings). The
  true-wind columns need no heading correction. This affects speed much more than
  direction: the old `(dm + hdt) % 360` reconstruction agrees with the vendor's own
  `wxtp_td` to a 1.4° median.
- **Endeavor speed is in knots**, not m/s, despite the column name carrying no unit —
  confirmed by reconstructing true wind vectorially from relative wind, heading, and
  ship speed.
- **Sharp has two column-naming eras.** HRS26xx use `wind1_true_speed_kt`; HRS2303 uses
  `true_wind_speed__knots` (double underscore) and was previously excluded outright
  because that spelling wasn't recognized.

Only true wind is used; cruises with only relative wind are excluded (HRS2601 is the
one remaining exclusion — it has no wind columns at all). Missing values are rejected
both as strings and numerically (`-9999`, `-999`, `-99`, …), since feeds disagree on the
sentinel. Non-physical speeds are dropped; out-of-range directions and positions are
nulled rather than dropping the reading. Exact duplicate readings are removed per cruise
— the API has been observed serving a doubled file. No gust de-spiking is applied, so a
few very large readings remain visible in the distributions. Per-cruise retention,
discard counts, and exclusion reasons are recorded in `data/processed/cruises.csv`.

### Buoy validation

`scripts/validate_buoys.py` checks the shipboard columns against OOI Pioneer Array METBK
buoys (CP01CNSM, CP03ISSM, CP04OSSM), which report wind in unambiguous m/s at a fixed
height. Matching within 5 km and 10 min:

| | matches | ship/buoy speed ratio | median direction error |
|---|---|---|---|
| Armstrong/Atlantis, corrected (`wxtp_ts`) | 72,263 | 1.268 | 7.2° |
| Armstrong/Atlantis, pre-correction (`wxtp_sm`) | 75,286 | 1.273 | 7.9° |
| Endeavor (control, unchanged) | 8,778 | 1.227 | 7.6° |

The speed ratio barely moved — but that does **not** vindicate the old columns. Readings
near a mooring are exactly the ones the correction affects least (median shift 0.20 m/s
near a buoy versus 0.40 m/s dataset-wide), so this comparison has roughly half the
sensitivity to the true-vs-relative question and can't settle it either way. That is also
why the original validation appeared to rule out apparent-wind contamination when it
structurally could not. The decisive evidence is the API's `column_definition` metadata.

What the re-run does establish: direction improved slightly, and the residual 1.27-vs-1.23
gap against Endeavor is now a genuine inter-vessel difference (mast height and exposure)
rather than apparent-wind inflation. Both ratios exceeding 1 is the expected effect of a
ship's mast-mounted anemometer sitting well above a buoy's ~3–4 m sensor.
