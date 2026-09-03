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
endpoint, covering every cruise in the CTD catalog (currently 72 of 74 cruises have
usable true wind, ~770k readings). Column names differ per vessel, so the download
script resolves them per vessel family — Endeavor, Armstrong/Atlantis, Sharp, and
Atlantic Explorer.

Two corrections are applied that the upstream alias tables don't have, both verified
against the data and documented in detail in `data/processed/provenance.json`:

- **Endeavor speed is in knots**, not m/s, despite the column name carrying no unit —
  confirmed by reconstructing true wind vectorially from relative wind, heading, and
  ship speed.
- **Armstrong/Atlantis direction (`wxtp_dm`/`wxts_dm`) is relative to the bow**, not a
  compass bearing — corrected to true as `(dm + hdt) % 360`. Cruises without a heading
  column are excluded rather than publishing ship-relative angles as absolute.

Armstrong/Atlantis speed and direction were independently validated against OOI Pioneer
Array METBK buoys moored in the sampling area. Only true wind is used; cruises with only
relative wind are excluded. Missing-value sentinels and non-physical speeds are dropped;
no gust de-spiking is applied, so a few very large readings remain visible in the
distributions. Per-cruise status and exclusion reasons are recorded in
`data/processed/cruises.csv`.
