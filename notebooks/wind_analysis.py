import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    from pathlib import Path

    import holoviews as hv
    import marimo
    import numpy as np
    import polars as pl

    hv.extension("bokeh")

    def _no_active_tools(plot, element):
        plot.state.toolbar.active_drag = None
        plot.state.toolbar.active_scroll = None
        plot.state.toolbar.active_tap = None

    hv.opts.defaults(
        hv.opts.Overlay(hooks=[_no_active_tools]),
        hv.opts.Curve(hooks=[_no_active_tools]),
        hv.opts.Histogram(hooks=[_no_active_tools]),
        hv.opts.Bars(hooks=[_no_active_tools]),
        hv.opts.Polygons(hooks=[_no_active_tools]),
        hv.opts.Rectangles(hooks=[_no_active_tools]),
    )
    return Path, hv, json, marimo, math, np, pl


@app.cell
def _(Path, json, pl):
    # Load combined wind readings, the cruise catalog, and the provenance record.
    #
    # wind.parquet stores 'date' as a real datetime already (parsed once, at
    # download time) and is far smaller than the equivalent CSV, which is why
    # this is the one file in the project that isn't CSV.
    #
    # dt_s is a per-reading time weight (seconds until the next reading in the
    # same cruise, capped at 1 h) so fractions below are time-weighted, not a
    # naive reading count. The first reading of a cruise and readings after a
    # >1 h gap inherit the cruise's median weight so they are not double- or
    # zero-counted.
    base = Path("data/processed")
    df = pl.read_parquet(base / "wind.parquet").sort(["cruise", "date"])
    _dt_raw = pl.col("date").diff().over("cruise").dt.total_seconds()
    df = df.with_columns(
        pl.when(_dt_raw.is_not_null() & (_dt_raw <= 3600))
        .then(_dt_raw)
        .otherwise(0.0)
        .alias("dt_s")
    )
    df = df.with_columns(
        pl.when(pl.col("dt_s") == 0)
        .then(pl.col("dt_s").median().over("cruise"))
        .otherwise(pl.col("dt_s"))
        .alias("dt_s")
    )
    cruises = pl.read_csv(base / "cruises.csv")
    prov = json.loads((base / "provenance.json").read_text())
    return df, prov


@app.cell
def _(df, marimo, prov):
    _span = f"{df['date'].min():%Y}–{df['date'].max():%Y}"
    marimo.md(f"""
    ## Data provenance

    | | |
    |---|---|
    | **Source** | NES-LTER API, `https://nes-lter-api.whoi.edu` |
    | **Cruise catalog** | `/api/ctd/cruises/all` |
    | **Underway data** | `/api/underway/{{cruise}}.csv` (one CSV per cruise; parsed and stored as `data/raw/{{cruise}}.parquet`) |
    | **Variables** | true wind speed, m/s (Sharp, Atlantic Explorer, and Endeavor sensors report knots and are converted; Armstrong/Atlantis `wxtp_ts` is already m/s); true wind direction, degrees from north. Armstrong/Atlantis now use the vendor's true-wind columns `wxtp_ts`/`wxtp_td` rather than the relative pair — see the note below the wind roses |
    | **Season** | by cruise start month: winter {{12,1,2}}, spring {{3,4,5}}, summer {{6,7,8}}, fall {{9,10,11}} |
    | **QA** | Missing values dropped as strings (NODATA/NAN/…) *and* numerically (−9999, −999, −99, 999, 9999), since feeds disagree on the sentinel. Non-physical speeds (<0 or ≥100 m/s) dropped; out-of-range directions and positions nulled rather than dropping the reading. Exact duplicate readings removed per cruise. **True wind only** — cruises with only relative wind are excluded (see `cruises.csv` notes). No gust de-spiking, so a few large spikes remain and are visible in the distribution tail. |
    | **Discovery** | endpoints and the vessel→column mapping started from the `nes-lter-mcp` MCP server (`find_cruises`, `query_underway`, `get_dataset_schema`, `resolve_variable`) but now diverge from its `UNDERWAY_VARIABLE_ALIASES` table on the Endeavor unit and on *which Armstrong/Atlantis columns are true wind at all*. Column semantics are now checked against the API's `/api/underway/column_definition/` rather than inferred from names — see the note below the wind roses. |
    | **Download** | `scripts/download_wind.py` (re-runnable; writes `data/raw/`, `data/processed/wind.parquet`, `data/processed/cruises.csv`, `data/processed/provenance.json`; data manipulation throughout uses polars, with parquet for the two large per-reading tables) |

    **Coverage:** {prov["totals"]["cruises_with_wind"]} of {prov["totals"]["cruises_in_catalog"]} catalog cruises contributed wind · {prov["totals"]["wind_readings"]:,} readings · {_span}
    """)
    return


@app.cell
def _(marimo):
    marimo.md("""
    ## Controls

    * **x** — threshold for the "above x" questions (m/s).
    * **Season** — restrict the distributions/analyses to one season (or all).
    * **Cruise** — the single cruise highlighted alongside the seasonal series
      in §1 and §1b, or "none" for a seasons-only view. Not affected by the
      season filter.
    """)
    return


@app.cell
def _(df, marimo):
    x = marimo.ui.slider(
        start=0, stop=30, step=0.5, value=13, label="threshold x (m/s)"
    )
    season = marimo.ui.dropdown(
        options=["all"] + sorted(df["season"].drop_nulls().unique().to_list()),
        value="all",
        label="season",
    )
    # Highlighted cruise for sections 1 and 1b; "none" gives a seasons-only
    # view. Defaults to HRS2609 when it's present, otherwise "none", so the
    # notebook still runs against a dataset that predates (or postdates) that
    # cruise.
    _cruise_options = sorted(df["cruise"].drop_nulls().unique().to_list())
    cruise = marimo.ui.dropdown(
        options=["none"] + _cruise_options,
        value="HRS2609" if "HRS2609" in _cruise_options else "none",
        searchable=True,
        label="highlight cruise",
    )
    marimo.hstack([x, season, cruise])
    return cruise, season, x


@app.cell
def _(df, pl, season):
    # Working frame restricted to the selected season.
    d = df if season.value == "all" else df.filter(pl.col("season") == season.value)
    return (d,)


@app.cell
def _():
    SEASON_ORDER = ["winter", "spring", "summer", "fall"]
    COLORS = {
        "winter": "#4c78a8",
        "spring": "#54a24b",
        "summer": "#e4a72c",
        "fall": "#b8629b",
    }
    HIGHLIGHT_COLOR = "#666666"
    return COLORS, HIGHLIGHT_COLOR, SEASON_ORDER


@app.cell
def _(cruise, marimo):
    _extra = (
        ""
        if cruise.value == "none"
        else f" A final panel isolates cruise **{cruise.value}**."
    )
    marimo.md(
        f"### 1. Wind speed distribution by season\n\nNormalized (density) histograms overlaid by season, with the threshold **x** marked.{_extra}"
    )
    return


@app.cell
def _(COLORS, HIGHLIGHT_COLOR, SEASON_ORDER, cruise, d, df, hv, np, pl, x):
    # Pre-bin with numpy instead of handing raw per-reading arrays to the plot:
    # a density histogram only ever needs the bin counts, so this is the
    # "resampling" step for this chart and keeps the payload tiny regardless
    # of how many hundreds of thousands of readings are behind it.
    _edges = np.linspace(0, max(d["wind_speed_m_s"].max(), x.value), 81)

    def _panel(_v, _label, _color):
        _counts, _ = np.histogram(_v, bins=_edges, density=True)
        _hist = hv.Histogram((_edges, _counts), label=_label).opts(
            fill_color=_color, fill_alpha=0.45, line_alpha=0
        )
        return (_hist * hv.VLine(x.value).opts(color="red", line_dash="dashed")).opts(
            hv.opts.Overlay(
                width=700,
                height=150,
                xlabel="wind speed (m/s)",
                ylabel="density",
                show_legend=True,
                legend_position="top_right",
            )
        )

    _panels = []
    for _s in SEASON_ORDER:
        _v = d.filter(pl.col("season") == _s)["wind_speed_m_s"].drop_nulls().to_numpy()
        if len(_v):
            _panels.append(_panel(_v, _s, COLORS[_s]))
    # The highlighted cruise is pulled from the full (season-unfiltered)
    # dataset, since it's a single cruise and the season dropdown shouldn't
    # hide it.
    if cruise.value != "none":
        _cruise_v = (
            df.filter(pl.col("cruise") == cruise.value)["wind_speed_m_s"]
            .drop_nulls()
            .to_numpy()
        )
        if len(_cruise_v):
            _panels.append(_panel(_cruise_v, cruise.value, HIGHLIGHT_COLOR))
    hv.Layout(_panels).cols(1).opts(
        title="Underway true-wind speed distribution by season (density)"
    )
    return


@app.cell
def _(cruise, marimo):
    _extra = (
        ""
        if cruise.value == "none"
        else f", plus a separate line for cruise **{cruise.value}**"
    )
    marimo.md(
        f'### 1b. "What percentage of the time is wind above x?" — survival curves\n\nS(v) = time-weighted P(wind > v) per season{_extra}. The red dashed line marks **x**; the text next to the legend gives each series\' answer at that x.'
    )
    return


@app.cell
def _(COLORS, HIGHLIGHT_COLOR, SEASON_ORDER, cruise, d, df, hv, np, pl, x):
    # Each season's exact survival function has one point per underway reading
    # (up to ~10^5-10^6) and looks jagged at that resolution. Interpolating it
    # onto a shared 300-point grid is this chart's resampling step: tiny
    # payload, a smooth line, and (unlike a datashaded image) a plain Curve
    # that can carry a real legend entry.
    _grid = np.linspace(0, max(d["wind_speed_m_s"].max(), x.value), 300)
    _curves = []
    _fracs = {}

    def _survival(_sub, _label, _color):
        (_v, _w) = (
            _sub["wind_speed_m_s"].to_numpy(),
            _sub["dt_s"].to_numpy(),
        )
        _total = _w.sum()
        if len(_v) < 10 or _total <= 0:
            return
        _order = np.argsort(_v)
        (_v, _w) = (_v[_order], _w[_order])
        _above = np.cumsum(_w[::-1])[::-1] / _total * 100
        _smooth = np.interp(_grid, _v, _above)
        _curves.append(
            hv.Curve(
                (_grid, _smooth),
                "wind speed (m/s)",
                "% of time with wind above v",
                label=_label,
            ).opts(color=_color, line_width=2)
        )
        _fracs[_label] = _w[_v >= x.value].sum() / _total * 100

    for _s in SEASON_ORDER:
        _survival(
            d.filter(pl.col("season") == _s).drop_nulls("wind_speed_m_s"),
            _s,
            COLORS[_s],
        )
    # The highlighted cruise is pulled from the full (season-unfiltered)
    # dataset, since it's a single cruise and the season dropdown shouldn't
    # hide it.
    if cruise.value != "none":
        _survival(
            df.filter(pl.col("cruise") == cruise.value).drop_nulls("wind_speed_m_s"),
            cruise.value,
            HIGHLIGHT_COLOR,
        )
    # Threshold answers as a text block in the bottom-right corner, where the
    # curves themselves converge toward 0% (v is near its max) and the
    # top-right legend never reaches -- unlike a fixed y anchored near the
    # top, this stays clear of the legend regardless of how many series
    # (seasons + the highlighted cruise) are listed.
    _labels_order = SEASON_ORDER + ([] if cruise.value == "none" else [cruise.value])
    _note = "\n".join(
        f"{_l}: {_fracs[_l]:.1f}%" for _l in _labels_order if _l in _fracs
    )
    _note_label = hv.Labels(
        {"x": [_grid[-1] * 0.97], "y": [3], "text": [_note]},
        ["x", "y"],
        "text",
    ).opts(
        text_align="right",
        text_baseline="bottom",
        text_font_size="9pt",
        text_color="dimgray",
    )
    _overlay = (
        hv.Overlay(_curves)
        * hv.VLine(x.value).opts(color="red", line_dash="dashed")
        * _note_label
    )
    _overlay.opts(
        hv.opts.Overlay(
            width=700,
            height=400,
            title=f"Fraction of time wind is above v, by season (at x = {x.value:g} m/s)",
            legend_position="top_right",
        )
    )
    return


@app.cell
def _(marimo, x):
    marimo.md(
        f"### 1c. Answer table: % of time wind ≥ x, by season\n\n**x = {x.value:g} m/s** (time-weighted; each reading weighted by seconds until the next reading, capped at 1 h)."
    )
    return


@app.cell
def _(d, marimo, pl, x):
    rows = []
    for (_s,), _sub in d.group_by("season"):
        _t = _sub["dt_s"].sum()
        if _t > 0:
            rows.append(
                {
                    "season": _s,
                    "hours": round(_t / 3600, 1),
                    "readings": len(_sub),
                    "% time >= x": round(
                        _sub.filter(pl.col("wind_speed_m_s") >= x.value)["dt_s"].sum()
                        / _t
                        * 100,
                        2,
                    ),
                }
            )
    _total = d["dt_s"].sum()
    rows.append(
        {
            "season": "ALL (selected)",
            "hours": round(_total / 3600, 1),
            "readings": len(d),
            "% time >= x": round(
                d.filter(pl.col("wind_speed_m_s") >= x.value)["dt_s"].sum()
                / _total
                * 100,
                2,
            ),
        }
    )
    _tab = pl.DataFrame(rows).sort("season")
    marimo.ui.table(_tab, selection=None, page_size=10)
    return


@app.cell
def _(marimo, x):
    marimo.md(
        f"### 2. What percentage of each cruise was wind above x?\n\nTime-weighted per-cruise fraction at **x = {x.value:g} m/s**, sorted; full table below the chart."
    )
    return


@app.cell
def _(COLORS, d, hv, marimo, pl, x):
    recs = []
    for (_c,), _sub in d.group_by("cruise"):
        _t = _sub["dt_s"].sum()
        if _t <= 0:
            continue
        recs.append(
            {
                "cruise": _c,
                "vessel": _sub["vessel"][0],
                "season": _sub["season"][0],
                "hours": round(_t / 3600, 1),
                "% above x": round(
                    _sub.filter(pl.col("wind_speed_m_s") >= x.value)["dt_s"].sum()
                    / _t
                    * 100,
                    2,
                ),
            }
        )
    _tab = pl.DataFrame(recs).sort("% above x", descending=True)
    # ~70 cruises is small enough to render directly, no resampling needed.
    _bars = hv.Bars(
        _tab,
        kdims=[hv.Dimension("cruise", values=_tab["cruise"].to_list())],
        vdims=["% above x", "season"],
    )
    _bars = _bars.opts(
        invert_axes=True,
        color="season",
        cmap=COLORS,
        width=900,
        height=max(400, 18 * len(_tab)),
        tools=["hover"],
        xlabel=f"% of cruise time with wind ≥ {x.value:g} m/s",
        title=f"% of cruise time with wind ≥ {x.value:g} m/s, by cruise",
    )
    marimo.vstack([_bars, marimo.ui.table(_tab, selection=None, page_size=15)])
    return


@app.cell
def _(marimo):
    marimo.md("""
    ### 3. Wind direction by season (wind roses)

    Classic wind-rose layout: 16 compass sectors (0°=N, clockwise), petal
    length = % of time the wind blew from that sector (time-weighted), and
    each petal is stacked by wind speed bin -- weakest nearest the center,
    strongest at the tip -- so both prevailing direction and how hard it
    typically blows from that direction are visible together. The % of time
    below 1 m/s ("calm", direction undefined at near-zero wind) is shown in
    the center of each rose rather than assigned to a direction. Direction is
    true/absolute (not relative to the ship's heading; see provenance note
    below the roses).

    The four **seasonal** roses share one radial scale, so their petal lengths
    are directly comparable, and one legend (upper right). The fifth rose is
    the **selected cruise**, which is drawn on its own radial scale: a single
    cruise is a few days of one weather pattern rather than a climatology, so
    its strongest sector typically runs about 20% of the time and can reach
    50%, against 10-17% for a full season. Putting it on the shared scale
    would shrink all four seasonal roses by up to 3x depending on which cruise
    was picked. Instead, a **dashed ring** on the cruise rose marks the radius
    the seasonal peak would reach on that rose's scale: petals crossing it mean
    the cruise concentrated wind into one sector more strongly than any full
    season does. Read the ring labels, not the raw petal lengths, when
    comparing across that boundary.
    """)
    return


@app.cell
def _(hv, np):
    # Bokeh (holoviews' backend here) has no native polar coordinate system,
    # so each rose is built from Cartesian annular-wedge polygons -- one per
    # 16-sector x speed-bin combination. Data is tiny per rose (16 sectors x
    # 8 speed bins x 4 seasons) so no resampling is needed here.
    CALM_THRESHOLD = 1.0  # m/s; shown as a single number in the center of each rose
    # Bin edges chosen from the data's own distribution (99% of readings fall under
    # 19 m/s, max ~29 m/s), colored with the blue -> green -> yellow -> orange -> red
    # -> magenta progression used by windy.com / earth.nullschool-style wind speed
    # scales, rather than this project's usual single-hue sequential ramp -- a
    # deliberate exception since matching that specific, widely recognized
    # convention was requested.
    SPEED_BINS = [CALM_THRESHOLD, 4, 7, 10, 13, 16, 19, 22, np.inf]
    SPEED_LABELS = [
        "1–4",
        "4–7",
        "7–10",
        "10–13",
        "13–16",
        "16–19",
        "19–22",
        "22+",
    ]
    SPEED_COLORS = {
        "1–4": "#4a7bd4",
        "4–7": "#35a6d9",
        "7–10": "#35c48d",
        "10–13": "#7fc93c",
        "13–16": "#f2d43d",
        "16–19": "#f2a13d",
        "19–22": "#e8562f",
        "22+": "#c22b6e",
    }
    NSEC = 16
    COMPASS8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    def _annular_wedge(theta0, theta1, r_lo, r_hi, n=6):
        outer = np.linspace(theta0, theta1, n)
        inner = outer[::-1]
        xs = np.concatenate([r_hi * np.cos(outer), r_lo * np.cos(inner)])
        ys = np.concatenate([r_hi * np.sin(outer), r_lo * np.sin(inner)])
        return xs, ys

    def wind_rose(dirs, speeds, weights, title, max_r, note=None, ref_r=None):
        total_w = weights.sum()
        calm = speeds < CALM_THRESHOLD
        calm_pct = weights[calm].sum() / total_w * 100

        edges_deg = np.linspace(0, 360, NSEC + 1)
        sector = np.clip(np.digitize(dirs[~calm], edges_deg) - 1, 0, NSEC - 1)
        speed_bin = np.clip(
            np.digitize(speeds[~calm], SPEED_BINS) - 1,
            0,
            len(SPEED_LABELS) - 1,
        )
        freq = np.zeros((NSEC, len(SPEED_LABELS)))
        np.add.at(freq, (sector, speed_bin), weights[~calm])
        freq = freq / total_w * 100

        # petals start outside a small blank center "hole" (standard wind-rose
        # convention) that displays the calm % without overlapping any wedge
        hole_r = max_r * 0.14
        # A sector whose petals total f reaches radius hole_r + f, so the
        # reference ring for a frequency of ref_r goes at hole_r + ref_r. When
        # ref_r exceeds this rose's own peak the ring sits outside every petal,
        # so the viewport has to grow to keep it visible -- which also reads
        # correctly: the rose visibly falls short of the reference.
        outer_r = max(max_r, ref_r) if ref_r else max_r
        polys = []
        for i in range(NSEC):
            c0, c1 = edges_deg[i], edges_deg[i + 1]
            m0, m1 = np.radians(90 - c1), np.radians(90 - c0)
            r_lo = hole_r
            for k, label in enumerate(SPEED_LABELS):
                r_hi = r_lo + freq[i, k]
                if freq[i, k] > 0:
                    xs, ys = _annular_wedge(m0, m1, r_lo, r_hi)
                    polys.append(
                        {
                            "x": xs,
                            "y": ys,
                            "speed (m/s)": label,
                            "pct": round(float(freq[i, k]), 2),
                            "dir": f"{c0:.0f}-{c1:.0f}",
                        }
                    )
                r_lo = r_hi
        ring_step = max(1, round(max_r / 4))
        rings_r = np.arange(ring_step, max_r + hole_r + ring_step, ring_step)
        wedges = hv.Polygons(polys, vdims=["speed (m/s)", "pct", "dir"]).opts(
            color="speed (m/s)",
            cmap=SPEED_COLORS,
            line_color="white",
            line_width=0.5,
            tools=["hover"],
            show_legend=False,
            colorbar=False,
        )
        rings = hv.Path([hv.Ellipse(0, 0, 2 * r).array() for r in rings_r]).opts(
            color="gray", line_width=0.5, line_dash="dotted"
        )
        hole = hv.Ellipse(0, 0, 2 * hole_r).opts(
            color="white", line_color="gray", line_width=0.75
        )
        # offset labels off the N gridline (and away from the center label) at a fixed compass bearing
        _label_bearing = np.radians(90 - 22)
        ring_labels = hv.Labels(
            {
                "x": rings_r * np.cos(_label_bearing),
                "y": rings_r * np.sin(_label_bearing),
                "text": [f"{r:g}%" for r in rings_r],
            },
            ["x", "y"],
            "text",
        ).opts(text_font_size="7pt", text_color="gray", text_align="left")
        compass_r = (outer_r + hole_r) * 1.1
        cxs = compass_r * np.cos(np.radians(90 - np.arange(0, 360, 45)))
        cys = compass_r * np.sin(np.radians(90 - np.arange(0, 360, 45)))
        compass = hv.Labels(
            {"x": cxs, "y": cys, "text": COMPASS8}, ["x", "y"], "text"
        ).opts(text_font_size="9pt", text_color="gray")
        center = hv.Labels(
            {"x": [0], "y": [0], "text": [f"{calm_pct:.1f}% calm"]},
            ["x", "y"],
            "text",
        ).opts(
            text_font_size="7pt",
            text_color="dimgray",
            text_align="center",
            text_baseline="middle",
        )
        lim = (outer_r + hole_r) * 1.22
        rose = rings * wedges * hole * ring_labels * compass * center
        if ref_r:
            # Dashed and darker than the dotted gray gridlines so it reads as a
            # reference value rather than another ring label.
            rose = rose * hv.Ellipse(0, 0, 2 * (hole_r + ref_r)).opts(
                color="#555555", line_width=1.25, line_dash="dashed"
            )
        if note:
            # Caption under the rose rather than in the title: a rose on its own
            # radial scale has to say so, and the title would truncate at 340px.
            rose = rose * hv.Labels(
                {"x": [0], "y": [-lim * 0.94], "text": [note]}, ["x", "y"], "text"
            ).opts(text_font_size="7pt", text_color="dimgray", text_align="center")
        return rose.opts(
            width=340,
            height=340,
            xaxis=None,
            yaxis=None,
            show_grid=False,
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            title=title,
            data_aspect=1,
        )

    def wind_rose_legend():
        n = len(SPEED_LABELS)
        swatch_data = [
            (0, n - 1 - i - 0.35, 0.6, n - 1 - i + 0.35, label)
            for i, label in enumerate(SPEED_LABELS)
        ]
        swatches = hv.Rectangles(swatch_data, vdims=["speed (m/s)"]).opts(
            color="speed (m/s)",
            cmap=SPEED_COLORS,
            line_color=None,
            show_legend=False,
        )
        labels = hv.Labels(
            {
                "x": [0.9] * n,
                "y": [n - 1 - i for i in range(n)],
                "text": SPEED_LABELS,
            },
            ["x", "y"],
            "text",
        ).opts(
            text_font_size="9pt",
            text_align="left",
            text_baseline="middle",
        )
        title = hv.Text(0, n + 0.3, "Wind speed (m/s)").opts(
            text_font_size="9pt", text_align="left", text_font_style="bold"
        )
        return (swatches * labels * title).opts(
            width=230,
            height=340,
            xaxis=None,
            yaxis=None,
            show_grid=False,
            toolbar=None,
            xlim=(-0.4, 4.0),
            ylim=(-0.8, n + 0.9),
        )

    return wind_rose, wind_rose_legend


@app.cell
def _(SEASON_ORDER, cruise, d, df, hv, np, pl, wind_rose, wind_rose_legend):
    _season_data = {}
    for _s in SEASON_ORDER:
        _sub = d.filter(
            (pl.col("season") == _s)
            & pl.col("wind_dir_deg").is_not_null()
            & pl.col("wind_speed_m_s").is_not_null()
        )
        if len(_sub):
            _season_data[_s] = (
                _sub["wind_dir_deg"].to_numpy(),
                _sub["wind_speed_m_s"].to_numpy(),
                _sub["dt_s"].to_numpy(),
            )

    # shared radial scale across all four roses so petal lengths are comparable season to season
    def _max_freq(dirs, speeds, weights):
        edges_deg = np.linspace(0, 360, 16 + 1)
        sector = np.clip(np.digitize(dirs, edges_deg) - 1, 0, 15)
        totals = np.zeros(16)
        np.add.at(totals, sector, weights)
        return (totals / weights.sum() * 100).max()

    _max_r = max(_max_freq(*v) for v in _season_data.values()) if _season_data else 10.0
    _roses = [
        wind_rose(*_season_data[_s], _s, _max_r)
        for _s in SEASON_ORDER
        if _s in _season_data
    ]
    # 3 columns so the legend lands in the upper right, which leaves the lower
    # right slot free for the selected cruise:
    #   [winter, spring, legend] / [summer, fall, cruise]
    _panels = _roses[:2] + [wind_rose_legend()] + _roses[2:]

    # Fifth rose: the selected cruise. Taken from the season-unfiltered frame,
    # same as sections 1 and 1b, so the season dropdown can't hide it.
    #
    # It gets its OWN radial scale rather than joining the shared seasonal one:
    # a single cruise is a few days of one weather pattern, not a climatology,
    # so its strongest sector runs ~20% of the time and can reach 50%, against
    # 10-17% for a whole season. Folding it into the shared scale would shrink
    # all four seasonal roses by up to 3x depending on which cruise is picked,
    # making the season-to-season comparison depend on an unrelated control.
    # The caption states the seasonal peak so the two can still be compared.
    if cruise.value != "none":
        _csub = df.filter(
            (pl.col("cruise") == cruise.value)
            & pl.col("wind_dir_deg").is_not_null()
            & pl.col("wind_speed_m_s").is_not_null()
        )
        if len(_csub):
            _cruise_data = (
                _csub["wind_dir_deg"].to_numpy(),
                _csub["wind_speed_m_s"].to_numpy(),
                _csub["dt_s"].to_numpy(),
            )
            _cruise_max_r = _max_freq(*_cruise_data)
            _note = (
                f"own scale · dashed ring = seasonal peak ({_max_r:.1f}%)"
                if _season_data
                else "own scale"
            )
            _panels = _panels + [
                wind_rose(
                    *_cruise_data,
                    cruise.value,
                    _cruise_max_r,
                    note=_note,
                    ref_r=_max_r if _season_data else None,
                )
            ]

    hv.Layout(_panels).cols(3).opts(
        title="Wind direction by season, plus the selected cruise (time-weighted; 0°=N, clockwise)"
    )
    return


@app.cell
def _(marimo):
    marimo.md("""
    **Armstrong/Atlantis column correction:** these cruises previously used
    `wxtp_sm`/`wxtp_dm` with a `(dm + hdt) % 360` heading correction. The
    API's own `column_definition` endpoint documents that pair as *"Port
    Vaisala **relative** wind speed / direction average"*, while
    `wxtp_ts`/`wxtp_td` are *"Port Vaisala **True** Wind Speed / Direction"* —
    so the published speed was apparent wind, not true. The pipeline now uses
    `wxtp_ts`/`wxtp_td` (`wxts_*` as fallback), which are present on all 49
    Armstrong/Atlantis cruises and need no heading correction at all.

    This matters most for **speed**: on AR39B the old and new values have a
    median ratio of 1.000 but p5 0.625 / p95 1.885 — identical while the ship
    is on station, diverging under way, exactly as apparent wind behaves.
    The old **direction** reconstruction was close to right: `(dm + hdt) % 360`
    agrees with the vendor's own `wxtp_td` to a 1.4° median, so the wind roses
    change far less than the speed distributions. Endeavor, Sharp, and
    Atlantic Explorer already reported true direction directly and are
    unaffected.

    **Buoy validation** (`scripts/validate_buoys.py`, re-run against the
    corrected columns): matching to OOI Pioneer Array METBK buoys within 5 km
    and 10 minutes gives Armstrong/Atlantis a median ship/buoy speed ratio of
    **1.268** and direction error **7.2°** over 72,263 matches, versus 1.273
    and 7.9° for the same comparison against the old `wxtp_sm`/`wxtp_dm` data.
    The Endeavor control is unchanged at 1.227 and 7.6°.

    The speed ratio barely moved — but that does *not* vindicate the old
    columns. Buoy-matched readings are precisely the ones the correction
    affects least: near a mooring it shifts speed by a median of 0.20 m/s,
    against 0.40 m/s dataset-wide. This comparison therefore has roughly half
    the sensitivity to the true-vs-relative question and cannot settle it in
    either direction — which is also why the original validation appeared to
    rule out apparent-wind contamination when it structurally could not. The
    decisive evidence is the API's `column_definition` metadata. What the
    re-run does show is a slight direction improvement, and that the residual
    1.27-vs-1.23 gap against Endeavor is a genuine inter-vessel difference
    (mast height and exposure), not apparent wind. See `provenance.json`.
    """)
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
    event_threshold = marimo.ui.slider(
        start=5, stop=30, step=0.5, value=13, label="event threshold (m/s)"
    )
    min_duration = marimo.ui.slider(
        start=1, stop=240, step=1, value=90, label="min event duration (min)"
    )
    marimo.hstack([event_threshold, min_duration])
    return event_threshold, min_duration


@app.cell
def _(Path, df, event_threshold, marimo, math, min_duration, np, pl):
    # Detect high-wind events and write the catalog.
    threshold_val = event_threshold.value
    duration_val = min_duration.value
    gap_s = 10 * 60

    def circular_mean(degs):
        r = np.radians(degs)
        return math.degrees(math.atan2(np.sin(r).sum(), np.cos(r).sum())) % 360

    events = []
    for (_c,), _sub in df.sort(["cruise", "date"]).group_by(
        "cruise", maintain_order=True
    ):
        _v = _sub["wind_speed_m_s"].to_numpy()
        if np.isnan(_v).all():
            continue
        _t = _sub["date"].to_numpy()
        dirg = _sub["wind_dir_deg"].to_numpy()
        _dt = np.zeros(len(_v))
        if len(_v) > 1:
            # unit-independent: polars stores datetime64 in microsecond
            # resolution (unlike pandas' nanoseconds), so dividing by a fixed
            # power of 10 would silently be off by 1000x if that ever changes.
            _dt[:-1] = np.diff(_t) / np.timedelta64(1, "s")
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
            dur_min = (_t[j] - _t[_i]) / np.timedelta64(1, "m")
            if dur_min >= float(duration_val):
                seg_dir = dirg[_i : j + 1]
                events.append(
                    {
                        "cruise": _c,
                        "vessel": _sub["vessel"][0],
                        "season": _sub["season"][0],
                        "start": str(_t[_i]),
                        "end": str(_t[j]),
                        "duration_min": round(float(dur_min), 1),
                        "peak_m_s": round(float(np.nanmax(_v[_i : j + 1])), 2),
                        "mean_m_s": round(float(np.nanmean(_v[_i : j + 1])), 2),
                        "mean_dir_deg": round(circular_mean(seg_dir), 1)
                        if np.isfinite(seg_dir).any()
                        else None,
                    }
                )
            _i = j + 1

    ev = (
        pl.DataFrame(events).sort(["cruise", "start"])
        if events
        else pl.DataFrame(
            schema=[
                "cruise",
                "vessel",
                "season",
                "start",
                "end",
                "duration_min",
                "peak_m_s",
                "mean_m_s",
                "mean_dir_deg",
            ]
        )
    )
    if len(ev):
        ev.write_csv(Path("data/processed/high_wind_events.csv"))
    _summary = marimo.md(
        f"**{len(ev)} high-wind events** (≥ {threshold_val:g} m/s, lasting ≥ {duration_val:g} min) across {ev['cruise'].n_unique() if len(ev) else 0} cruises — written to `data/processed/high_wind_events.csv`."
    )
    marimo.vstack(
        [_summary]
        if not len(ev)
        else [_summary, marimo.ui.table(ev, selection=None, page_size=15)]
    )
    return


if __name__ == "__main__":
    app.run()
