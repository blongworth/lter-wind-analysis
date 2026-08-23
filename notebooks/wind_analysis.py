import marimo
__generated_with = "0.24.0"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo
    import json
    import math
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import holoviews as hv
    from holoviews.operation.datashader import rasterize, shade
    hv.extension('bokeh')
    return (marimo, json, math, Path, np, pd, hv, rasterize, shade)

@app.cell
def _(json, Path, pd):
    # Load combined wind readings, the cruise catalog, and the provenance record.
    #
    # dt_s is a per-reading time weight (seconds until the next reading in the
    # same cruise, capped at 1 h) so fractions below are time-weighted, not a
    # naive reading count. The first reading of a cruise and readings after a
    # >1 h gap inherit the cruise's median weight so they are not double- or
    # zero-counted.
    base = Path('data/processed')
    df = pd.read_csv(base / 'wind.csv').sort_values(['cruise', 'date']).reset_index(drop=True)
    # ISO8601: the API mixes second- and microsecond-precision timestamps (strict
    # first-row inference would drop ~70% of rows); utc=True + tz_localize(None)
    # normalizes any tz-offset rows to naive UTC so numpy datetime64 math works below.
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='ISO8601', utc=True).dt.tz_localize(None)
    dropped = df['date'].isna().sum()
    if dropped:
        df = df.dropna(subset=['date']).reset_index(drop=True)
    _dt = df.groupby('cruise')['date'].diff().dt.total_seconds()
    df['dt_s'] = _dt.where(_dt <= 3600, 0.0).to_numpy()
    df['dt_s'] = pd.Series(df['dt_s'].where(~(df['dt_s'] == 0), df.groupby('cruise')['dt_s'].transform('median')), index=df.index)
    df['wind_dir_deg'] = pd.to_numeric(df['wind_dir_deg'], errors='coerce')
    cruises = pd.read_csv(base / 'cruises.csv')
    prov = json.loads((base / 'provenance.json').read_text())
    return (df, cruises, prov)

@app.cell
def _(marimo, prov, df):
    _span = f"{df['date'].min():%Y}–{df['date'].max():%Y}"
    marimo.md(f"""
    ## Data provenance

    | | |
    |---|---|
    | **Source** | NES-LTER API, `https://nes-lter-api.whoi.edu` |
    | **Cruise catalog** | `/api/ctd/cruises/all` |
    | **Underway data** | `/api/underway/{{cruise}}.csv` (one CSV per cruise, stored in `data/raw/`) |
    | **Variables** | true wind speed at the bow anemometer, converted to m/s (Sharp, Atlantic Explorer, and Endeavor sensors report knots); true wind direction, degrees from north |
    | **Season** | by cruise start month: winter {{12,1,2}}, spring {{3,4,5}}, summer {{6,7,8}}, fall {{9,10,11}} |
    | **QA** | NODATA/NAN sentinels and non-physical speeds (<0 or ≥100 m/s) dropped. **True wind only** — cruises with only relative wind are excluded (see `cruises.csv` notes). No gust de-spiking, so a few large spikes remain and are visible in the distribution tail. |
    | **Discovery** | endpoints and vessel→column mapping located via the `nes-lter-mcp` MCP server (`find_cruises`, `query_underway`, `get_dataset_schema`, `resolve_variable`), except the Endeavor unit, which that server's `UNDERWAY_VARIABLE_ALIASES` table gets wrong — confirmed in knots by vectorially reconstructing true wind from relative wind + heading + ship speed (median residual 2.4% across 4,428 EN608 rows). Armstrong/Atlantis's unit could not be independently confirmed the same way (no paired relative-wind data) and is left as m/s on climatological plausibility only — see `provenance.json` for both. |
    | **Download** | `scripts/download_wind.py` (re-runnable; writes `data/raw/`, `data/processed/wind.csv`, `data/processed/cruises.csv`, `data/processed/provenance.json`) |

    **Coverage:** {prov['totals']['cruises_with_wind']} of {prov['totals']['cruises_in_catalog']} catalog cruises contributed wind · {prov['totals']['wind_readings']:,} readings · {_span}
    """)
    return

@app.cell
def _(marimo):
    marimo.md("""
    ## Controls

    * **x** — threshold for the "above x" questions (m/s).
    * **Season** — restrict the distributions/analyses to one season (or all).
    """)
    return

@app.cell
def _(df, marimo):
    x = marimo.ui.slider(start=0, stop=30, step=0.5, value=12, label='threshold x (m/s)')
    season = marimo.ui.dropdown(options=['all'] + sorted(df['season'].dropna().unique().tolist()), value='all', label='season')
    marimo.hstack([x, season])
    return (x, season)

@app.cell
def _(df, season):
    # Working frame restricted to the selected season.
    d = df if season.value == 'all' else df[df['season'] == season.value]
    return (d,)

@app.cell
def _():
    SEASON_ORDER = ['winter', 'spring', 'summer', 'fall']
    COLORS = {'winter': '#4c78a8', 'spring': '#54a24b', 'summer': '#e4a72c', 'fall': '#b8629b'}
    return (SEASON_ORDER, COLORS)

@app.cell
def _(marimo):
    marimo.md("### 1. Wind speed distribution by season\n\nNormalized (density) histograms overlaid by season, with the threshold **x** marked.")
    return

@app.cell
def _(np, hv, d, x, SEASON_ORDER, COLORS):
    # Pre-bin with numpy instead of handing raw per-reading arrays to the plot:
    # a density histogram only ever needs the bin counts, so this is the
    # "resampling" step for this chart and keeps the payload tiny regardless
    # of how many hundreds of thousands of readings are behind it.
    _edges = np.linspace(0, max(d['wind_speed_m_s'].max(), x.value), 81)
    _hists = []
    for _s in SEASON_ORDER:
        _v = d.loc[d['season'] == _s, 'wind_speed_m_s'].dropna().to_numpy()
        if len(_v):
            _counts, _ = np.histogram(_v, bins=_edges, density=True)
            _hists.append(hv.Histogram((_edges, _counts), label=_s).opts(fill_color=COLORS[_s], fill_alpha=0.65, line_alpha=0))
    _overlay = hv.Overlay(_hists) * hv.VLine(x.value).opts(color='red', line_dash='dashed')
    _overlay.opts(
        hv.opts.Overlay(width=700, height=400, title='Underway true-wind speed distribution by season (density)', xlabel='wind speed (m/s)', ylabel='density', legend_position='bottom'),
    )
    return

@app.cell
def _(marimo):
    marimo.md('### 1b. "What percentage of the time is wind above x?" — survival curves\n\nS(v) = time-weighted P(wind > v) per season. The red dashed line marks **x**; each annotation is the answer for that season at that x.')
    return

@app.cell
def _(np, hv, rasterize, shade, d, x, SEASON_ORDER, COLORS):
    # Each season's survival curve has one point per underway reading (up to
    # ~10^5-10^6). Sending that many points to the browser as a Curve/Scatter
    # is exactly what was too big to render, so each curve is rasterized to a
    # fixed-size image (datashader resampling) before being colored and
    # overlaid — full data fidelity, constant-size payload.
    _curves = []
    _labels = []
    for _s in SEASON_ORDER:
        _sub = d[d['season'] == _s].dropna(subset=['wind_speed_m_s'])
        (_v, _w) = (_sub['wind_speed_m_s'].to_numpy(), _sub['dt_s'].to_numpy())
        _total = _w.sum()
        if len(_v) < 10 or _total <= 0:
            continue
        _order = np.argsort(_v)
        (_v, _w) = (_v[_order], _w[_order])
        _above = np.cumsum(_w[::-1])[::-1] / _total * 100
        _curve = hv.Curve((_v, _above), 'wind speed (m/s)', '% of time with wind above v')
        _raster = rasterize(_curve, width=700, height=400, dynamic=False)
        _curves.append(shade(_raster, cmap=[COLORS[_s]], dynamic=False))
        _frac = _w[_v >= x.value].sum() / _total * 100
        _labels.append({'x': x.value, 'y': _frac, 'text': f'{_s}: {_frac:.1f}%'})
    _label_data = {'x': [_l['x'] for _l in _labels], 'y': [_l['y'] for _l in _labels], 'text': [_l['text'] for _l in _labels]}
    _overlay = hv.Overlay(_curves) * hv.VLine(x.value).opts(color='red', line_dash='dashed') * hv.Labels(_label_data, ['x', 'y'], 'text').opts(text_align='left')
    _overlay.opts(hv.opts.Overlay(width=700, height=400, title=f'Fraction of time wind is above v, by season (at x = {x.value:g} m/s, see labels)'))
    return

@app.cell
def _(marimo, x):
    marimo.md(f'### 1c. Answer table: % of time wind ≥ x, by season\n\n**x = {x.value:g} m/s** (time-weighted; each reading weighted by seconds until the next reading, capped at 1 h).')
    return

@app.cell
def _(marimo, pd, d, x):
    rows = []
    for (_s, _sub) in d.groupby('season'):
        _t = _sub['dt_s'].sum()
        if _t > 0:
            rows.append({'season': _s, 'hours': round(_t / 3600, 1), 'readings': len(_sub), '% time >= x': round(_sub.loc[_sub['wind_speed_m_s'] >= x.value, 'dt_s'].sum() / _t * 100, 2)})
    _total = d['dt_s'].sum()
    rows.append({'season': 'ALL (selected)', 'hours': round(_total / 3600, 1), 'readings': len(d), '% time >= x': round(d.loc[d['wind_speed_m_s'] >= x.value, 'dt_s'].sum() / _total * 100, 2)})
    _tab = pd.DataFrame(rows).sort_values('season').reset_index(drop=True)
    marimo.ui.table(_tab, selection=None, page_size=10)
    return

@app.cell
def _(marimo, x):
    marimo.md(f'### 2. What percentage of each cruise was wind above x?\n\nTime-weighted per-cruise fraction at **x = {x.value:g} m/s**, sorted; full table below the chart.')
    return

@app.cell
def _(marimo, pd, hv, d, x, COLORS):
    recs = []
    for (_c, _sub) in d.groupby('cruise'):
        _t = _sub['dt_s'].sum()
        if _t <= 0:
            continue
        recs.append({'cruise': _c, 'vessel': _sub['vessel'].iloc[0], 'season': _sub['season'].iloc[0], 'hours': round(_t / 3600, 1), '% above x': round(_sub.loc[_sub['wind_speed_m_s'] >= x.value, 'dt_s'].sum() / _t * 100, 2)})
    _tab = pd.DataFrame(recs).sort_values('% above x', ascending=False).reset_index(drop=True)
    # ~70 cruises is small enough to render directly, no resampling needed.
    _bars = hv.Bars(_tab, kdims=[hv.Dimension('cruise', values=_tab['cruise'].tolist())], vdims=['% above x', 'season'])
    _bars = _bars.opts(invert_axes=True, color='season', cmap=COLORS, width=700, height=max(400, 18 * len(_tab)), tools=['hover'], xlabel=f'% of cruise time with wind ≥ {x.value:g} m/s')
    marimo.vstack([_bars, marimo.ui.table(_tab, selection=None, page_size=15)])
    return

@app.cell
def _(marimo):
    marimo.md('### 3. Wind direction by season (wind roses)\n\n30° sectors; bar length is the time-weighted share of readings in each sector, relative to the busiest sector of that season.')
    return

@app.cell
def _(np, hv, d, SEASON_ORDER, COLORS):
    # Bokeh (holoviews' backend here) has no native polar coordinate system,
    # so each rose is built as Cartesian wedge polygons -- one per 30-degree
    # sector, with radius = the time-weighted share of that sector. Data is
    # tiny (12 sectors x 4 seasons) so no resampling is needed here.
    def _wind_rose(counts, color, title):
        _nsec = len(counts)
        _edges_deg = np.linspace(0, 360, _nsec + 1)
        _polys = []
        for _i in range(_nsec):
            _c0, _c1 = _edges_deg[_i], _edges_deg[_i + 1]
            _m0, _m1 = np.radians(90 - _c1), np.radians(90 - _c0)
            _thetas = np.linspace(_m0, _m1, 6)
            _r = counts[_i]
            _xs = np.concatenate([[0], _r * np.cos(_thetas), [0]])
            _ys = np.concatenate([[0], _r * np.sin(_thetas), [0]])
            _polys.append({'x': _xs, 'y': _ys, 'pct': _r, 'sector deg': f'{_c0:.0f}-{_c1:.0f}'})
        _wedges = hv.Polygons(_polys, vdims=['pct', 'sector deg']).opts(color=color, line_color='white', line_width=0.5, alpha=0.85, tools=['hover'])
        _rings = hv.Path([hv.Ellipse(0, 0, 2 * _rr).array() for _rr in (25, 50, 75, 100)]).opts(color='gray', line_width=0.5, line_dash='dotted')
        _compass = hv.Labels({'x': [0, 108, 0, -108], 'y': [108, 0, -108, 0], 'text': ['N', 'E', 'S', 'W']}, ['x', 'y'], 'text').opts(text_font_size='9pt', text_color='gray')
        return (_rings * _wedges * _compass).opts(width=320, height=320, xaxis=None, yaxis=None, show_grid=False, xlim=(-118, 118), ylim=(-118, 118), title=title)

    nsec = 12
    edges = np.linspace(0, 360, nsec + 1)
    _roses = []
    for _s in SEASON_ORDER:
        _sub = d[(d['season'] == _s) & d['wind_dir_deg'].notna()]
        if _sub.empty:
            continue
        theta = (360 - _sub['wind_dir_deg']) % 360
        sector = np.clip(np.digitize(theta.to_numpy(), edges) - 1, 0, nsec - 1)
        _w = _sub['dt_s'].to_numpy()
        counts = np.zeros(nsec)
        np.add.at(counts, sector, _w)
        if counts.sum() <= 0:
            continue
        counts = counts / counts.max() * 100
        _roses.append(_wind_rose(counts, COLORS[_s], _s))
    hv.Layout(_roses).cols(2).opts(title='Wind direction by season (time-weighted, 30° sectors; compass: 0°=N, clockwise)')
    return

@app.cell
def _(marimo):
    marimo.md("""
    ### 4. Cataloging high-wind events — proposed method

    A **high-wind event** for a cruise is a *contiguous run* of underway readings that satisfies all of:

    1. wind speed ≥ **event threshold** (default 15 m/s ≈ Beaufort 7, "near gale");
    2. the run persists ≥ **minimum duration** (default 10 min) — this discards
       single-reading sensor spikes, i.e. the non-de-spiked outliers visible in the
       histogram tail;
    3. readings count as contiguous only when the gap to the next reading is ≤ 10 min,
       so an event never spans a data outage or ship-time jump.

    Each event is one row in the catalog:
    `cruise, vessel, season, start, end, duration_min, peak_m_s, mean_m_s, mean_dir_deg`
    (direction is a circular mean). The catalog is written to `data/processed/high_wind_events.csv`
    and is fully re-derivable from `data/raw/`, so it stays reproducible.

    Suggested next steps once the catalog exists:
    * aggregate to a per-cruise summary (n events, max peak, total event-hours) for ranking cruises;
    * optionally join event windows to the CTD cast record (`list_casts`) to see whether high-wind periods coincide with profile collections;
    * extend to *gust* events by lowering the threshold and raising the persistence requirement.

    Adjust the two knobs below and the catalog regenerates reactively.
    """)
    return

@app.cell
def _(marimo):
    event_threshold = marimo.ui.slider(start=5, stop=30, step=0.5, value=15, label='event threshold (m/s)')
    min_duration = marimo.ui.slider(start=1, stop=120, step=1, value=10, label='min event duration (min)')
    marimo.hstack([event_threshold, min_duration])
    return (event_threshold, min_duration)

@app.cell
def _(math, np, pd, Path, df, marimo, event_threshold, min_duration):
    # Detect high-wind events and write the catalog.
    threshold_val = event_threshold.value
    duration_val = min_duration.value
    gap_s = 10 * 60

    def circular_mean(degs):
        r = np.radians(degs)
        return math.degrees(math.atan2(np.sin(r).sum(), np.cos(r).sum())) % 360

    events = []
    for (_c, _sub) in df.sort_values(['cruise', 'date']).groupby('cruise'):
        _v = _sub['wind_speed_m_s'].to_numpy()
        if np.isnan(_v).all():
            continue
        _t = _sub['date'].to_numpy()
        dirg = _sub['wind_dir_deg'].to_numpy()
        _dt = np.zeros(len(_v))
        if len(_v) > 1:
            _dt[:-1] = np.diff(_t) / 1_000_000_000.0
        _dt = np.where(np.isfinite(_dt) & (_dt <= gap_s), _dt, np.inf)
        hot = _v >= threshold_val
        (_i, n) = (0, len(_v))
        while _i < n:
            if not hot[_i]:
                _i += 1
                continue
            j = _i
            while j + 1 < n and hot[j + 1] and (_dt[j] != np.inf):
                j += 1
            dur_min = (pd.Timestamp(_t[j]) - pd.Timestamp(_t[_i])).total_seconds() / 60
            if dur_min >= float(duration_val):
                seg_dir = dirg[_i:j + 1]
                events.append({'cruise': _c, 'vessel': _sub['vessel'].iloc[0], 'season': _sub['season'].iloc[0], 'start': str(pd.Timestamp(_t[_i])), 'end': str(pd.Timestamp(_t[j])), 'duration_min': round(float(dur_min), 1), 'peak_m_s': round(float(np.nanmax(_v[_i:j + 1])), 2), 'mean_m_s': round(float(np.nanmean(_v[_i:j + 1])), 2), 'mean_dir_deg': round(circular_mean(seg_dir), 1) if np.isfinite(seg_dir).any() else None})
            _i = j + 1

    ev = pd.DataFrame(events).sort_values(['cruise', 'start']).reset_index(drop=True) if events else pd.DataFrame(columns=['cruise', 'vessel', 'season', 'start', 'end', 'duration_min', 'peak_m_s', 'mean_m_s', 'mean_dir_deg'])
    if len(ev):
        ev.to_csv(Path('data/processed/high_wind_events.csv'), index=False)
    _summary = marimo.md(f"**{len(ev)} high-wind events** (≥ {threshold_val:g} m/s, lasting ≥ {duration_val:g} min) across {ev['cruise'].nunique() if len(ev) else 0} cruises — written to `data/processed/high_wind_events.csv`.")
    marimo.vstack([_summary] if not len(ev) else [_summary, marimo.ui.table(ev, selection=None, page_size=15)])
    return

if __name__ == '__main__':
    app.run()
