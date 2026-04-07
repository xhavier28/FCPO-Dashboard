import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots
from FCPO_analysis import build_spread_table, load_combined_dataset
from mr_screener.data.loader import load_pair
from mr_screener.screener.pipeline import run_pair, autotune_delta

SPOT_DIR = "Raw Data"
YEAR_COLORS = {
    2020: "#9467bd", 2021: "#8c564b", 2022: "#e377c2",
    2023: "#1f77b4", 2024: "#ff7f0e", 2025: "#2ca02c", 2026: "#d62728",
}
TERM_DIR = "Raw Data/Term Structure"
SD_DIR   = "Raw Data/Stock and Production"
MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

SPREAD_PAIRS = [
    ("Spread3M4M",   "+3M",  "+4M"),
    ("Spread4M5M",   "+4M",  "+5M"),
    ("Spread5M6M",   "+5M",  "+6M"),
    ("Spread6M7M",   "+6M",  "+7M"),
    ("Spread7M8M",   "+7M",  "+8M"),
    ("Spread8M9M",   "+8M",  "+9M"),
    ("Spread9M10M",  "+9M",  "+10M"),
    ("Spread10M11M", "+10M", "+11M"),
]


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def color_with_alpha(color, alpha):
    """Add alpha to hex (#rrggbb) or plotly rgb(...) color strings."""
    if color.startswith("#"):
        return hex_to_rgba(color, alpha)
    if color.startswith("rgb("):
        inner = color[4:-1]
        return f"rgba({inner},{alpha})"
    return color

TICKVALS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
TICKTEXT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DARK_BG    = "#0e1117"
DARK_PLOT  = "#262730"
DARK_GRID  = "#3a3a4a"
DARK_TEXT  = "#fafafa"


@st.cache_data
def load_data(spot_dir):
    csvs = [
        f for f in os.listdir(spot_dir)
        if f.lower().endswith(".csv") and f.lower().startswith("myx_dly_fcpo")
    ]
    frames = []
    for fname in csvs:
        raw = pd.read_csv(f"{spot_dir}/{fname}")
        sample = str(raw["time"].iloc[0])
        if sample.replace(".", "").isdigit():
            raw["date"] = pd.to_datetime(raw["time"], unit="s", utc=True).dt.tz_convert("Asia/Kuala_Lumpur").dt.date
        else:
            raw["date"] = pd.to_datetime(raw["time"]).dt.date
        frames.append(raw)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").groupby("date", as_index=False).last()
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df = df[df["year"] >= 2020]
    return df[["date", "year", "doy", "open", "high", "low", "close", "Volume"]]


@st.cache_data
def load_contracts():
    contracts = {}
    if not os.path.exists(TERM_DIR):
        return contracts
    for year_dir in sorted(os.listdir(TERM_DIR)):
        if not year_dir.isdigit():
            continue
        year = int(year_dir)
        for m, abbr in enumerate(MONTH_ABBRS, 1):
            yy = str(year)[2:]
            path = f"{TERM_DIR}/{year_dir}/FCPO {abbr}{yy}_Daily.csv"
            if not os.path.exists(path):
                continue
            df_c = pd.read_csv(path)
            df_c["date"] = pd.to_datetime(
                df_c["Timestamp (UTC)"], infer_datetime_format=True
            ).dt.normalize()
            series = df_c.set_index("date")["Close"]
            contracts[(year, m)] = series[~series.index.duplicated(keep="last")]
    return contracts


def available_years(contracts):
    return sorted({y for y, m in contracts.keys()})


def front_month(date):
    if date.day <= 15:
        return (date.year, date.month)
    else:
        m = date.month + 1
        y = date.year + (1 if m > 12 else 0)
        return (y, m % 12 or 12)


def add_months(ym, n):
    y, m = ym
    m += n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return (y, m)


def build_term_table(contracts):
    all_dates = sorted(set(d for s in contracts.values() for d in s.index))
    today = pd.Timestamp.today().normalize()
    all_dates = [d for d in all_dates if pd.Timestamp("2020-01-01") <= d <= today]

    def week_label(d):
        w = ["W1", "W2", "W3", "W4"][(d.day - 1) // 7 if d.day <= 28 else 3]
        return f"{w} {d.strftime('%b %Y')}"

    rows = {}  # week_label → {col_index: [prices]}
    for date in all_dates:
        wl = week_label(date)
        if wl not in rows:
            rows[wl] = {i: [] for i in range(12)}
        fm = front_month(date)
        for i in range(12):
            ym = add_months(fm, i)
            if ym in contracts and date in contracts[ym].index:
                rows[wl][i].append(contracts[ym][date])

    col_names = ["Current"] + [f"+{i}M" for i in range(1, 12)]
    records = []
    for wl, cols in rows.items():
        row = {"Week": wl}
        for i, col in enumerate(col_names):
            vals = cols[i]
            row[col] = round(sum(vals) / len(vals), 0) if vals else None
        records.append(row)

    df = pd.DataFrame(records)
    df["_sort"] = (
        pd.to_datetime(df["Week"].str.extract(r'(\w+ \d{4})')[0], format="%b %Y")
        + pd.to_timedelta((df["Week"].str.extract(r'(W\d)')[0].str[1].astype(int) - 1) * 7, unit='d')
    )
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def build_daily_table(contracts):
    all_dates = sorted(set(d for s in contracts.values() for d in s.index))
    today = pd.Timestamp.today().normalize()
    all_dates = [d for d in all_dates if pd.Timestamp("2020-01-01") <= d <= today]

    col_names = ["Current"] + [f"+{i}M" for i in range(1, 12)]
    records = []
    for date in all_dates:
        fm = front_month(date)
        row = {"Date": date}
        for i, col in enumerate(col_names):
            ym = add_months(fm, i)
            row[col] = contracts[ym][date] if ym in contracts and date in contracts[ym].index else None
        records.append(row)

    df = pd.DataFrame(records)

    # Fill missing Current with +1M on roll days (contract expiry gap ~13th–15th)
    roll_mask = df["Current"].isnull() & df["+1M"].notnull()
    df.loc[roll_mask, "Current"] = df.loc[roll_mask, "+1M"]

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d %b %Y")
    return df


def build_delta_table(df):
    import numpy as np
    df = df.copy()
    df["date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
    df = df.sort_values("date").reset_index(drop=True)

    for name, near, far in SPREAD_PAIRS:
        df[name] = df[near] - df[far]

    df["_day"]      = df["date"].dt.day
    df["_prev_day"] = df["_day"].shift(1)
    df["_is_roll"]  = (df["_day"] > 15) & (df["_prev_day"] <= 15)

    spd_cols   = [p[0] for p in SPREAD_PAIRS]
    delta_cols = [f"DeltaS{p[0][6:]}" for p in SPREAD_PAIRS]
    spd_shift  = spd_cols[1:] + [None]

    for spd, next_spd, delta in zip(spd_cols, spd_shift, delta_cols):
        normal_dod = df[spd] - df[spd].shift(1)
        if next_spd is not None:
            roll_dod = df[spd] - df[next_spd].shift(1)
        else:
            roll_dod = pd.Series(np.nan, index=df.index)
        df[delta] = np.where(df["_is_roll"], roll_dod, normal_dod)

    df.drop(columns=["_day", "_prev_day", "_is_roll", "date"], inplace=True)
    return df


def build_combined_chart(df, year, df_term, compare_year=None):
    """
    Single figure, 2 rows, shared x-axis (doy), range-slider for scrolling.
    Row 1: 5-day smoothed spot price (actual MYR scale).
    Row 2: each week's term structure drawn as a normalised mini-curve
            positioned at the week's day-of-year window — shape only.
    Initial view: 6 months. Range slider scrolls both rows together.
    """
    col_names = ["Current"] + [f"+{i}M" for i in range(1, 12)]

    yr_df = df[df["year"] == year].sort_values("doy").copy()
    smoothed = yr_df["close"].rolling(window=5, min_periods=1, center=True).mean()

    year_rows = df_term[df_term["Week"].str.endswith(str(year))].reset_index(drop=True)
    n = len(year_rows)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.04,
    )

    # When comparing, most-recent year is bold (alpha=1); earlier year is dimmed (alpha=0.35)
    recent_year = max(year, compare_year) if compare_year is not None else year
    primary_alpha = 1.0 if year == recent_year else 0.35
    primary_width = 2.5 if year == recent_year else 1.5
    primary_color = color_with_alpha(YEAR_COLORS.get(year, "#636efa"), primary_alpha)

    # --- Row 1: spot price ---
    fig.add_trace(
        go.Scatter(
            x=yr_df["doy"], y=smoothed,
            mode="lines",
            line=dict(color=primary_color, width=primary_width),
            customdata=yr_df[["date", "close"]].assign(
                date_fmt=yr_df["date"].dt.strftime("%d %b %Y")
            )[["date_fmt", "close"]].values,
            hovertemplate="%{customdata[0]}<br>MYR %{customdata[1]:,.0f}<extra></extra>",
            showlegend=compare_year is not None,
            name=str(year),
        ),
        row=1, col=1,
    )

    if compare_year is not None:
        comp_alpha = 1.0 if compare_year == recent_year else 0.35
        comp_width = 2.5 if compare_year == recent_year else 1.5
        comp_color = color_with_alpha(YEAR_COLORS.get(compare_year, "#aaaaaa"), comp_alpha)
        comp_df = df[df["year"] == compare_year].sort_values("doy").copy()
        comp_smoothed = comp_df["close"].rolling(window=5, min_periods=1, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=comp_df["doy"], y=comp_smoothed,
                mode="lines",
                line=dict(color=comp_color, width=comp_width),
                customdata=comp_df[["date", "close"]].assign(
                    date_fmt=comp_df["date"].dt.strftime("%d %b %Y")
                )[["date_fmt", "close"]].values,
                hovertemplate="%{customdata[0]}<br>MYR %{customdata[1]:,.0f}<extra></extra>",
                showlegend=True,
                name=str(compare_year),
            ),
            row=1, col=1,
        )

    # --- Row 2: mini normalised term-structure curves ---
    sliver_alpha = 1.0 if year == recent_year else 0.35
    sliver_width = 1.5 if year == recent_year else 1.0
    if n > 0:
        # Turbo [0.05, 0.95] → vivid cyan→green→yellow→orange→red, pops on dark bg
        colorscale = plotly.colors.sample_colorscale(
            "Turbo", [0.05 + i / max(n - 1, 1) * 0.90 for i in range(n)]
        )

        for i, row_data in year_rows.iterrows():
            parts = row_data["Week"].split()        # ["W2", "Mar", "2025"]
            w_num = int(parts[0][1])                # 1-4
            m_num = MONTH_ABBRS.index(parts[1]) + 1  # 1-12

            # doy window for this week (6 days wide, 1-day gap between weeks)
            week_start = TICKVALS[m_num - 1] + (w_num - 1) * 7
            week_width = 6.0

            y_raw = [row_data[col] for col in col_names]
            pairs = [(j, y) for j, y in enumerate(y_raw) if y is not None]
            if not pairs:
                continue
            tenor_idx, prices = zip(*pairs)
            prices = list(prices)

            # Normalise to [0, 1] — shape only
            lo, hi = min(prices), max(prices)
            norm = [(p - lo) / (hi - lo) if hi > lo else 0.5 for p in prices]

            max_t = max(tenor_idx) or 1
            x_pos = [week_start + (t / max_t) * week_width for t in tenor_idx]

            hover = [f"{col_names[j]}: MYR {prices[k]:,.0f}"
                     for k, j in enumerate(tenor_idx)]

            fig.add_trace(
                go.Scatter(
                    x=x_pos, y=norm,
                    mode="lines",
                    line=dict(color=color_with_alpha(colorscale[i], sliver_alpha), width=sliver_width),
                    showlegend=False,
                    name=row_data["Week"],
                    customdata=hover,
                    hovertemplate=row_data["Week"] + "<br>%{customdata}<extra></extra>",
                ),
                row=2, col=1,
            )

    # --- Row 2: comparison year slivers ---
    if compare_year is not None:
        comp_sliver_alpha = 1.0 if compare_year == recent_year else 0.35
        comp_sliver_width = 1.5 if compare_year == recent_year else 1.0
        comp_rows = df_term[df_term["Week"].str.endswith(str(compare_year))].reset_index(drop=True)
        comp_n = len(comp_rows)
        if comp_n > 0:
            comp_colorscale = plotly.colors.sample_colorscale(
                "Turbo", [0.05 + i / max(comp_n - 1, 1) * 0.90 for i in range(comp_n)]
            )
            for i, row_data in comp_rows.iterrows():
                parts = row_data["Week"].split()
                w_num = int(parts[0][1])
                m_num = MONTH_ABBRS.index(parts[1]) + 1

                week_start = TICKVALS[m_num - 1] + (w_num - 1) * 7
                week_width = 6.0

                y_raw = [row_data[col] for col in col_names]
                pairs = [(j, y) for j, y in enumerate(y_raw) if y is not None]
                if not pairs:
                    continue
                tenor_idx, prices = zip(*pairs)
                prices = list(prices)

                lo, hi = min(prices), max(prices)
                norm = [(p - lo) / (hi - lo) if hi > lo else 0.5 for p in prices]

                max_t = max(tenor_idx) or 1
                x_pos = [week_start + (t / max_t) * week_width for t in tenor_idx]

                hover = [f"{col_names[j]}: MYR {prices[k]:,.0f}"
                         for k, j in enumerate(tenor_idx)]

                fig.add_trace(
                    go.Scatter(
                        x=x_pos, y=norm,
                        mode="lines",
                        line=dict(color=color_with_alpha(comp_colorscale[i], comp_sliver_alpha), width=comp_sliver_width),
                        showlegend=False,
                        name=row_data["Week"],
                        customdata=hover,
                        hovertemplate=row_data["Week"] + "<br>%{customdata}<extra></extra>",
                    ),
                    row=2, col=1,
                )

    # Build week-level tick labels: "W1 Jan", "W2", "W3", "W4", "W1 Feb", ...
    week_tickvals, week_ticktext = [], []
    for mi in range(12):
        for w in range(4):
            doy = TICKVALS[mi] + w * 7
            week_tickvals.append(doy)
            week_ticktext.append(f"W{w+1} {MONTH_ABBRS[mi]}" if w == 0 else f"W{w+1}")

    # Full-height vertical dividers spanning both subplots via yref='paper'
    shapes = []
    for mi in range(12):
        # Month boundary — solid, more visible
        shapes.append(dict(
            type='line', xref='x', yref='paper',
            x0=TICKVALS[mi], x1=TICKVALS[mi], y0=0, y1=1,
            line=dict(color='#6a7a8a', width=1),
        ))
        # Week sub-dividers (W2, W3, W4) — subtle dashed
        for w in range(1, 4):
            shapes.append(dict(
                type='line', xref='x', yref='paper',
                x0=TICKVALS[mi] + w * 7, x1=TICKVALS[mi] + w * 7, y0=0, y1=1,
                line=dict(color='#3e3e50', width=0.5, dash='dot'),
            ))

    # With shared_xaxes=True on 2 rows: xaxis = row1 (hidden), xaxis2 = row2 (visible bottom)
    # Tick labels and range must be set on xaxis2; dragmode='pan' locks zoom width
    fig.update_layout(
        hovermode="x",
        dragmode="pan",
        plot_bgcolor=DARK_PLOT, paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        height=540,
        margin=dict(l=60, r=30, t=40, b=90),
        showlegend=(compare_year is not None),
        legend=dict(orientation="h", x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(color=DARK_TEXT)),
        shapes=shapes,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            minallowed=1, maxallowed=366,
        ),
        xaxis2=dict(
            tickvals=week_tickvals, ticktext=week_ticktext,
            range=[1, 122],
            minallowed=1, maxallowed=366,
            showgrid=False,
            tickangle=-45,
            tickfont=dict(color=DARK_TEXT),
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="MYR", showgrid=True, gridcolor=DARK_GRID,
            tickfont=dict(color=DARK_TEXT),
            showline=False,
        ),
        yaxis2=dict(
            showticklabels=False, showgrid=False,
            zeroline=False, showline=False,
            fixedrange=True, range=[-0.15, 1.15],
        ),
    )
    return fig


@st.cache_data
def load_spread_data():
    return build_spread_table(load_combined_dataset())


@st.cache_data
def load_delta_data():
    contracts = load_contracts()
    df_raw = build_daily_table(contracts)
    return build_delta_table(df_raw)


@st.cache_data
def load_supply_demand():
    def _read(fname, col):
        df = pd.read_excel(f"{SD_DIR}/{fname}", sheet_name="Table Data")
        df = df.iloc[1:].reset_index(drop=True)   # drop the "Close" text row
        df.columns = ["Date", col]
        df["Date"] = pd.to_datetime(df["Date"])
        return df.dropna(subset=["Date"]).set_index("Date")[col]

    df = pd.DataFrame({
        "Stock":      _read("FCPO Stock 3Y.xlsx",       "Stock"),
        "Exports":    _read("MPOB Exports 3Y.xlsx",     "Exports"),
        "Production": _read("MPOB Production 3Y.xlsx",  "Production"),
    }).sort_index().reset_index().rename(columns={"index": "Date"})

    df["year"]        = df["Date"].dt.year
    df["month"]       = df["Date"].dt.month
    df["Delta_Stock"] = df["Stock"].diff()
    df["Consumption"] = df["Production"] - df["Exports"] - df["Delta_Stock"]
    return df


METRIC_LABELS = {
    "Stock":       "Stock Level (tonnes)",
    "Production":  "Production (tonnes)",
    "Exports":     "Exports (tonnes)",
    "Consumption": "Consumption (tonnes)",
}


def build_sd_chart(df_sd, metric, selected_years):
    most_recent = max(selected_years)
    fig = go.Figure()
    for year in sorted(selected_years):
        yr = df_sd[df_sd["year"] == year].sort_values("month")
        is_recent = (year == most_recent)
        color = YEAR_COLORS.get(year, "#aaaaaa")
        fig.add_trace(go.Scatter(
            x=yr["month"],
            y=yr[metric],
            mode="lines+markers",
            name=str(year),
            line=dict(
                color=color if is_recent else hex_to_rgba(color, 0.35),
                width=2.5 if is_recent else 1.5,
            ),
            marker=dict(size=5 if is_recent else 3),
            customdata=yr[["Date", metric]].assign(
                date_fmt=yr["Date"].dt.strftime("%b %Y")
            )[["date_fmt", metric]].values,
            hovertemplate="%{customdata[0]}<br>%{customdata[1]:,.0f} tonnes<extra></extra>",
        ))
    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor=DARK_PLOT, paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        xaxis=dict(
            title="Month",
            tickvals=list(range(1, 13)),
            ticktext=MONTH_ABBRS,
            showgrid=True, gridcolor=DARK_GRID,
            tickfont=dict(color=DARK_TEXT),
        ),
        yaxis=dict(
            title=METRIC_LABELS[metric],
            showgrid=True, gridcolor=DARK_GRID,
            tickformat=",",
            tickfont=dict(color=DARK_TEXT),
        ),
        legend=dict(title="Year", orientation="v", font=dict(color=DARK_TEXT)),
        height=480,
        margin=dict(l=60, r=30, t=30, b=60),
    )
    return fig


def build_sd_table(df_sd, year):
    metrics = ["Stock", "Production", "Exports", "Consumption"]
    yr = df_sd[df_sd["year"] == year].sort_values("month").copy()
    yr.index = [MONTH_ABBRS[int(m) - 1] for m in yr["month"]]

    result = {}
    for m in metrics:
        vals = yr[m]
        result[m] = vals.apply(lambda v: f"{int(v):,}" if pd.notna(v) else "–")
        pct = vals.pct_change() * 100
        result[f"{m} %MoM"] = pct.apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "–"
        )

    ordered = []
    for m in metrics:
        ordered += [m, f"{m} %MoM"]
    return pd.DataFrame(result)[ordered]


def _style_outlier_table(df_d, df_z):
    """df_d: display df (raw deltas + Prev Date).  df_z: z-score df (spread cols only)."""
    def z_color(z_val):
        if pd.isna(z_val):
            return ""
        az = abs(z_val)
        if az < 2.0:
            return ""
        if z_val > 0:
            if az < 2.25:  return "background-color: #c8e6c9; color: #1b5e20"
            elif az < 2.5: return "background-color: #43a047; color: #ffffff"
            else:          return "background-color: #1b5e20; color: #ffffff"
        else:
            if az < 2.25:  return "background-color: #ffcdd2; color: #b71c1c"
            elif az < 2.5: return "background-color: #ef5350; color: #ffffff"
            else:          return "background-color: #b71c1c; color: #ffffff"

    style_df = pd.DataFrame("", index=df_d.index, columns=df_d.columns)
    for col in df_z.columns:
        style_df[col] = df_z[col].map(z_color)

    def fmt(val):
        if isinstance(val, str):
            return val
        return "–" if pd.isna(val) else f"{val:+.1f}"

    return df_d.style.apply(lambda _: style_df, axis=None).format(fmt)


def build_outlier_table(df_delta, year, min_z=2.0):
    import numpy as np
    delta_cols  = [f"DeltaS{p[0][6:]}" for p in SPREAD_PAIRS]
    short_names = [c.replace("DeltaS", "S") for c in delta_cols]

    df = df_delta.copy()
    df["_date"]      = pd.to_datetime(df["Date"], format="%d %b %Y")
    df = df.sort_values("_date").reset_index(drop=True)
    df["_prev_date"] = df["_date"].shift(1)          # previous trading day (full dataset)
    df = df[df["_date"].dt.year == year].reset_index(drop=True)

    if df.empty:
        return None, None

    df["_month"] = df["_date"].dt.month
    stats = df.groupby("_month")[delta_cols].agg(["mean", "std"])

    df_z_vals = {}
    df_d_vals = {}
    for col, short in zip(delta_cols, short_names):
        mean_s = df["_month"].map(stats[col]["mean"])
        std_s  = df["_month"].map(stats[col]["std"])
        z = (df[col] - mean_s) / std_s
        z = z.where(std_s.notna() & (std_s != 0))
        df_z_vals[short] = z.values
        df_d_vals[short] = df[col].values            # raw delta (MYR) for display

    date_strs = df["_date"].dt.strftime("%d %b %Y")
    df_z = pd.DataFrame(df_z_vals, index=date_strs)
    df_d = pd.DataFrame(df_d_vals, index=date_strs)

    # Align prev_date by date-string index for easy lookup after masking
    prev_lookup = df.copy()
    prev_lookup.index = date_strs

    mask = df_z.abs().ge(min_z).any(axis=1)
    df_z = df_z[mask]
    df_d = df_d[mask]

    if df_z.empty:
        return None, None

    prev_vals = (
        prev_lookup.loc[df_z.index, "_prev_date"]
        .dt.strftime("%d %b %Y")
        .fillna("–")
        .values
    )
    df_d.insert(0, "Prev Date", prev_vals)

    return df_d, _style_outlier_table(df_d, df_z)


def show_login_page():
    st.markdown(
        f"""
        <style>
        .login-box {{
            max-width: 380px;
            margin: 80px auto 0 auto;
            padding: 2.5rem;
            background: {DARK_PLOT};
            border-radius: 12px;
            border: 1px solid {DARK_GRID};
        }}
        .login-title {{
            text-align: center;
            color: {DARK_TEXT};
            font-size: 1.15rem;
            margin-bottom: 0.3rem;
        }}
        .login-subtitle {{
            text-align: center;
            color: #aaaaaa;
            font-size: 0.85rem;
            margin-bottom: 1.8rem;
        }}
        </style>
        <div class="login-box">
          <div class="login-title"><strong>Investasi Sinergi Indonesia</strong></div>
          <div class="login-subtitle">MYX FCPO Futures Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, col, _ = st.columns([1, 2, 1])
    with col:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if (
                username == st.secrets["auth"]["username"]
                and password == st.secrets["auth"]["password"]
            ):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")


st.set_page_config(page_title="FCPO Dashboard", layout="wide")

if not st.session_state.get("authenticated"):
    show_login_page()
    st.stop()

st.title("Investasi Sinergi Indonesia - MYX FCPO Futures Dashboard")

with st.sidebar:
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

df = load_data(SPOT_DIR)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Year-over-Year", "Term Structure", "Event Log", "Supply & Demand", "Mean Reversion", "Pair Screener"])

with tab1:
    data_years = sorted(df["year"].unique().tolist())
    selected_years = st.multiselect(
        "Select Years",
        options=data_years,
        default=data_years,
    )
    if not selected_years:
        st.warning("Select at least one year.")
    else:
        fig = go.Figure()

        most_recent = max(selected_years)

        for year in selected_years:
            yr_df = df[df["year"] == year].sort_values("doy").copy()
            if yr_df.empty:
                continue
            is_recent = year == most_recent
            smoothed = yr_df["close"].rolling(window=5, min_periods=1, center=True).mean()
            color = YEAR_COLORS.get(year, "#aaaaaa") if is_recent else hex_to_rgba(YEAR_COLORS.get(year, "#aaaaaa"), 0.35)
            line_width = 2.5 if is_recent else 1.5
            fig.add_trace(
                go.Scatter(
                    x=yr_df["doy"],
                    y=smoothed,
                    mode="lines",
                    name=str(year),
                    line=dict(color=color, width=line_width),
                    customdata=yr_df[["date", "close"]].assign(
                        date_fmt=yr_df["date"].dt.strftime("%d %b %Y")
                    )[["date_fmt", "close"]].values,
                    hovertemplate="%{customdata[0]}<br>MYR %{customdata[1]:,.0f}<extra></extra>",
                )
            )

        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor=DARK_PLOT, paper_bgcolor=DARK_BG,
            font=dict(color=DARK_TEXT),
            xaxis=dict(
                title="Month",
                tickvals=TICKVALS,
                ticktext=TICKTEXT,
                range=[1, 366],
                showgrid=True,
                gridcolor=DARK_GRID,
            ),
            yaxis=dict(
                title="Close (MYR)",
                showgrid=True,
                gridcolor=DARK_GRID,
            ),
            legend=dict(title="Year", orientation="v"),
            height=520,
            margin=dict(l=60, r=30, t=30, b=60),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary Statistics")

        months_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        month_num_map = {i+1: m for i, m in enumerate(months_order)}

        cols = pd.MultiIndex.from_product([selected_years, ["Avg", "Min", "Max", "Range"]])
        stat_df = pd.DataFrame(index=months_order, columns=cols, dtype=float)
        stat_df.index.name = "Month"

        for year in selected_years:
            yr_df = df[df["year"] == year].copy()
            yr_df["_month"] = yr_df["date"].dt.month
            for m_num in range(1, 13):
                m_df = yr_df[yr_df["_month"] == m_num]
                if m_df.empty:
                    continue
                m_name = month_num_map[m_num]
                stat_df.loc[m_name, (year, "Avg")]   = round(m_df["close"].mean(), 0)
                stat_df.loc[m_name, (year, "Min")]   = m_df["close"].min()
                stat_df.loc[m_name, (year, "Max")]   = m_df["close"].max()
                stat_df.loc[m_name, (year, "Range")] = m_df["close"].max() - m_df["close"].min()

        stat_df = stat_df.dropna(how="all")
        st.dataframe(stat_df, use_container_width=True)

with tab2:
    contracts = load_contracts()
    df_term = build_term_table(contracts)

    # Limit selectable years to those with spot price data (2027 contracts are
    # used for forward tenors only, not as a standalone display year)
    spot_years = set(df["year"].unique())
    years = [y for y in available_years(contracts) if y in spot_years]

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        selected_ts_year = st.selectbox("Year", options=years[::-1], index=0)
    with col2:
        st.write("")
        compare_mode = st.checkbox("Comparison")
    with col3:
        if compare_mode:
            other_years = [y for y in years[::-1] if y != selected_ts_year]
            compare_year = st.selectbox("Compare Year", options=other_years, index=0) if other_years else None
        else:
            compare_year = None

    st.caption(
        "Top: 5-day smoothed spot price.  "
        "Bottom: each sliver = one week's forward curve shape (y not to scale). "
        "Solid = primary year, dashed = comparison year. "
        "Color: cyan (Jan) → red (Dec). Hover for week label and prices."
    )
    st.plotly_chart(build_combined_chart(df, selected_ts_year, df_term, compare_year=compare_year), use_container_width=True)

    st.subheader("Raw Data Table")
    _freq = st.radio("View", options=["Weekly", "Daily"], horizontal=True, key="ts_freq")
    if _freq == "Weekly":
        st.caption("Average daily Close per week per contract. Front month rolls on the 16th.")
        _display_df = df_term.set_index("Week")
        _fname = "fcpo_weekly_term_structure.xlsx"
    else:
        st.caption("Raw daily Close per contract. Front month rolls on the 16th.")
        _display_df = build_daily_table(contracts).set_index("Date")
        _fname = "fcpo_daily_term_structure.xlsx"

    st.dataframe(_display_df, use_container_width=True)

    import io
    _buf = io.BytesIO()
    _display_df.to_excel(_buf, engine="openpyxl")
    st.download_button(
        label="Download as Excel",
        data=_buf.getvalue(),
        file_name=_fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with tab4:
    df_sd = load_supply_demand()
    sd_years = sorted(df_sd["year"].unique().tolist())

    col1, col2 = st.columns([2, 3])
    with col1:
        sd_metric = st.selectbox(
            "Metric",
            options=["Production", "Exports", "Consumption", "Stock"],
            index=0,
        )
    with col2:
        sd_years_sel = st.multiselect(
            "Years",
            options=sd_years[::-1],
            default=sd_years[::-1],
        )

    if not sd_years_sel:
        st.warning("Select at least one year.")
    else:
        st.plotly_chart(build_sd_chart(df_sd, sd_metric, sd_years_sel), use_container_width=True)

    st.subheader("Monthly Breakdown — All Metrics")
    table_year = st.selectbox("Table Year", options=sd_years[::-1], index=0, key="sd_table_year")
    st.dataframe(build_sd_table(df_sd, table_year), use_container_width=True)

SPREAD_COLS = ["spd_4m", "spd_5m", "spd_6m", "spd_7m", "spd_8m", "spd_9m", "spd_10m", "spd_11m"]
SPREAD_LABELS = {
    "spd_4m":  "+3M / +4M",
    "spd_5m":  "+4M / +5M",
    "spd_6m":  "+5M / +6M",
    "spd_7m":  "+6M / +7M",
    "spd_8m":  "+7M / +8M",
    "spd_9m":  "+8M / +9M",
    "spd_10m": "+9M / +10M",
    "spd_11m": "+10M / +11M",
}
SPREAD_COLORS = [
    "#00b4d8", "#0077b6", "#48cae4", "#90e0ef",
    "#f4a261", "#e76f51", "#e9c46a", "#2a9d8f",
]

with tab5:
    df_spd = load_spread_data()
    df_spd["date"] = pd.to_datetime(df_spd["date"])
    df_spd_filt = df_spd[df_spd["date"].dt.year.isin([2025, 2026])].copy()
    df_spd_filt = df_spd_filt.dropna(subset=SPREAD_COLS, how="all").sort_values("date")

    all_dates = df_spd_filt["date"].dt.date.tolist()
    date_options = sorted(set(all_dates), reverse=True)

    col_ctrl1, col_ctrl2 = st.columns([2, 3])
    with col_ctrl1:
        end_date = st.selectbox(
            "End Date",
            options=date_options,
            index=0,
            format_func=lambda d: d.strftime("%d %b %Y"),
        )
    with col_ctrl2:
        lookback = st.radio(
            "Lookback",
            options=[7, 14, 21, 30],
            format_func=lambda x: f"{x} days",
            horizontal=True,
        )

    end_dt = pd.Timestamp(end_date)
    window_df = df_spd_filt[df_spd_filt["date"] <= end_dt].tail(lookback).copy()

    # ── Line chart ────────────────────────────────────────────────────────────
    fig_mr = go.Figure()

    # y=0 reference line
    fig_mr.add_hline(
        y=0,
        line=dict(color="#888888", width=1, dash="dash"),
    )

    for i, col in enumerate(SPREAD_COLS):
        label = SPREAD_LABELS[col]
        color = SPREAD_COLORS[i]
        valid = window_df[["date", col]].dropna()
        fig_mr.add_trace(go.Scatter(
            x=valid["date"],
            y=valid[col],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate="%{x|%d %b %Y}<br>" + label + ": MYR %{y:,.0f}<extra></extra>",
        ))

    fig_mr.update_layout(
        hovermode="x unified",
        plot_bgcolor=DARK_PLOT, paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        xaxis=dict(
            title="Date",
            showgrid=True, gridcolor=DARK_GRID,
            tickfont=dict(color=DARK_TEXT),
            tickformat="%d %b",
        ),
        yaxis=dict(
            title="Spread (MYR)",
            showgrid=True, gridcolor=DARK_GRID,
            tickfont=dict(color=DARK_TEXT),
            zeroline=False,
        ),
        legend=dict(orientation="v", font=dict(color=DARK_TEXT)),
        height=460,
        margin=dict(l=60, r=30, t=30, b=60),
    )
    st.plotly_chart(fig_mr, use_container_width=True)

    # ── Main table: rows = spreads, columns = dates ───────────────────────────
    date_cols = window_df["date"].dt.strftime("%d %b").tolist()
    table_data = {}
    for col, label in SPREAD_LABELS.items():
        row_vals = window_df[col].tolist()
        table_data[label] = [int(round(v)) if pd.notna(v) else None for v in row_vals]

    df_main = pd.DataFrame(table_data, index=date_cols).T
    df_main.index.name = "Spread"

    st.subheader("Spread Values by Date")
    st.dataframe(df_main, use_container_width=True)

    # ── Summary table: per-spread mean & std dev across the window ────────────
    summary_rows = {}
    for col, label in SPREAD_LABELS.items():
        vals = window_df[col].dropna()
        summary_rows[label] = {
            "Mean": round(vals.mean(), 1) if len(vals) else None,
            "Std Dev": round(vals.std(), 1) if len(vals) > 1 else None,
        }
    df_summary = pd.DataFrame(summary_rows).T
    df_summary.index.name = "Spread"

    st.subheader("Summary Statistics")
    st.dataframe(df_summary, use_container_width=True)

with tab3:
    df_delta = load_delta_data()

    # Parse dates for grouping (don't mutate cached df)
    _df = df_delta.copy()
    _df["_dt"]    = pd.to_datetime(_df["Date"], format="%d %b %Y")
    _df["_year"]  = _df["_dt"].dt.year
    _df["_month"] = _df["_dt"].dt.month

    delta_cols  = [f"DeltaS{p[0][6:]}" for p in SPREAD_PAIRS]
    short_names = [c.replace("DeltaS", "S") for c in delta_cols]

    analysis_mode = st.radio("Analysis", options=["Monthly", "Yearly"], horizontal=True, key="el_mode")

    if analysis_mode == "Monthly":
        stat_years = sorted(_df["_year"].unique().tolist(), reverse=True)
        stat_year  = st.selectbox("Year", options=stat_years, index=0, key="el_year")
        grp_df = _df[_df["_year"] == stat_year]
        grp = grp_df.groupby("_month")[delta_cols]
        mean_tbl = grp.mean().round(1)
        std_tbl  = grp.std().round(1)
        mean_tbl.index = [MONTH_ABBRS[i - 1] for i in mean_tbl.index]
        std_tbl.index  = [MONTH_ABBRS[i - 1] for i in std_tbl.index]
        row_label = "Month"
        title = f"DoD Spread Change Stats — {stat_year}"
    else:
        grp = _df.groupby("_year")[delta_cols]
        mean_tbl = grp.mean().round(1)
        std_tbl  = grp.std().round(1)
        mean_tbl.index = mean_tbl.index.astype(str)
        std_tbl.index  = std_tbl.index.astype(str)
        row_label = "Year"
        title = "DoD Spread Change Stats — All Years"

    # Build combined MultiIndex DataFrame: columns = (SpreadName, Mean/Std)
    col_data = {}
    for short, col in zip(short_names, delta_cols):
        col_data[(short, "Mean")] = mean_tbl[col]
        col_data[(short, "Std")]  = std_tbl[col]
    combined_df = pd.DataFrame(col_data)
    combined_df.columns = pd.MultiIndex.from_tuples(list(col_data.keys()))
    combined_df.index.name = row_label

    st.subheader(title)
    st.caption("Mean and Std Dev of daily spread DoD change (MYR). Roll days adjusted.")
    st.dataframe(combined_df, use_container_width=True)

    st.subheader("Outlier Detection")
    st.caption("Days where any spread DoD delta exceeds ±2.0σ from that month's mean. Values show raw delta (MYR); colour indicates σ tier.")
    st.markdown(
        '<span style="background-color:#c8e6c9;color:#1b5e20;padding:2px 6px;border-radius:3px;margin-right:4px">+2–2.25σ</span>'
        '<span style="background-color:#43a047;color:#ffffff;padding:2px 6px;border-radius:3px;margin-right:4px">+2.25–2.5σ</span>'
        '<span style="background-color:#1b5e20;color:#ffffff;padding:2px 6px;border-radius:3px;margin-right:12px">+≥2.5σ</span>'
        '<span style="background-color:#ffcdd2;color:#b71c1c;padding:2px 6px;border-radius:3px;margin-right:4px">−2–2.25σ</span>'
        '<span style="background-color:#ef5350;color:#ffffff;padding:2px 6px;border-radius:3px;margin-right:4px">−2.25–2.5σ</span>'
        '<span style="background-color:#b71c1c;color:#ffffff;padding:2px 6px;border-radius:3px">−≥2.5σ</span>',
        unsafe_allow_html=True,
    )
    _outlier_years = sorted(_df["_year"].unique().tolist(), reverse=True)
    _outlier_year  = st.selectbox("Year", options=_outlier_years, index=0, key="el_outlier_year")
    _tier_options = {2.0: "All (≥ 2.0σ)", 2.25: "≥ 2.25σ", 2.5: "≥ 2.5σ"}
    _tier_filter = st.radio(
        "Show tiers",
        options=list(_tier_options.keys()),
        format_func=lambda x: _tier_options[x],
        horizontal=True,
        key="el_tier_filter",
    )
    _df_z, _z_styler = build_outlier_table(df_delta, _outlier_year, min_z=_tier_filter)
    if _df_z is None:
        st.info(f"No outlier days found in {_outlier_year}.")
    else:
        _total_days = len(_df[_df["_year"] == _outlier_year])
        st.caption(f"{len(_df_z)} outlier days out of {_total_days} trading days.")
        st.dataframe(_z_styler, use_container_width=True)

    st.subheader("Raw Daily Term Delta")
    st.caption("Date, 12 tenor prices, 8 consecutive spreads, 8 roll-adjusted DoD deltas.")

    _raw_df = df_delta.copy()
    _raw_df["_dt"] = pd.to_datetime(_raw_df["Date"], format="%d %b %Y")
    _raw_years  = sorted(_raw_df["_dt"].dt.year.unique().tolist(), reverse=True)
    _raw_months = list(range(1, 13))

    _fc1, _fc2 = st.columns(2)
    with _fc1:
        _raw_year = st.selectbox("Year", options=_raw_years, index=0, key="raw_year")
    with _fc2:
        _raw_month = st.selectbox(
            "Month",
            options=[0] + _raw_months,
            format_func=lambda m: "All" if m == 0 else MONTH_ABBRS[m - 1],
            index=0,
            key="raw_month",
        )

    _mask = _raw_df["_dt"].dt.year == _raw_year
    if _raw_month != 0:
        _mask &= _raw_df["_dt"].dt.month == _raw_month
    _raw_filtered = _raw_df[_mask].drop(columns=["_dt"]).set_index("Date")

    st.dataframe(_raw_filtered, use_container_width=True)

    import io
    _buf = io.BytesIO()
    _raw_filtered.to_excel(_buf, engine="openpyxl")
    _fname = f"fcpo_daily_term_delta_{_raw_year}" + (f"_{MONTH_ABBRS[_raw_month-1]}" if _raw_month else "") + ".xlsx"
    st.download_button(
        label="Download as Excel",
        data=_buf.getvalue(),
        file_name=_fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ── Tab 6: Mean Reversion Pair Screener ──────────────────────────────────────
with tab6:
    st.header("Mean Reversion Pair Screener")

    col_a, col_b = st.columns(2)
    with col_a:
        label_y = st.text_input("Series A label", "Asset A", key="mr_label_y")
        file_y  = st.file_uploader("Upload Series A CSV", type="csv", key="mr_file_y")
        mult_y  = st.number_input("Price multiplier A", value=1.0, step=0.5, min_value=0.01, key="mr_mult_y")
    with col_b:
        label_x = st.text_input("Series B label", "Asset B", key="mr_label_x")
        file_x  = st.file_uploader("Upload Series B CSV", type="csv", key="mr_file_x")
        mult_x  = st.number_input("Price multiplier B", value=1.0, step=0.5, min_value=0.01, key="mr_mult_x")

    with st.expander("Advanced settings"):
        mr_auto_tune = st.checkbox(
            "Auto-tune delta",
            value=False,
            key="mr_auto_tune",
            help="Runs Kalman at [1e-5, 1e-4, 1e-3] and picks the delta giving beta_ar1 closest to 0.50",
        )
        mr_delta = st.number_input(
            "Kalman delta — how fast the hedge ratio adapts (higher = faster)",
            value=1e-4, format="%.1e", key="mr_delta",
            disabled=mr_auto_tune,
            help="Try 1e-3 (fast adaptation), 1e-4 (default), 1e-5 (slow / very smooth beta)",
        )
        mr_Ve = st.number_input("Kalman Ve", value=0.001, format="%.4f", key="mr_Ve")

    run_btn = st.button("Run Screener", disabled=(file_y is None or file_x is None), key="mr_run")

    if run_btn:
        with st.spinner("Running screener…"):
            data = load_pair(file_y.read(), file_x.read(), label_y, label_x, mult_y=mult_y, mult_x=mult_x)
            if mr_auto_tune:
                tune_result  = autotune_delta(data, Ve=mr_Ve)
                chosen_delta = tune_result["delta"]
            else:
                tune_result  = None
                chosen_delta = mr_delta
            results = run_pair(data, delta=chosen_delta, Ve=mr_Ve)

        if tune_result:
            st.info(f"**Auto-tune:** {tune_result['reason']}")
            tune_rows = [
                {
                    "Delta":        f"{r['delta']:.0e}",
                    "beta_ar1 Raw": f"{r['beta_ar1_raw']:.4f}" if r["beta_ar1_raw"] is not None else "—",
                    "beta_ar1 Log": f"{r['beta_ar1_log']:.4f}" if r["beta_ar1_log"] is not None else "—",
                    "Dist from 0.5": f"{r['dist_from_05']:.4f}" if r["dist_from_05"] != float('inf') else "—",
                    "Selected":     "✓" if r["delta"] == chosen_delta else "",
                }
                for r in tune_result["rows"]
            ]
            st.dataframe(pd.DataFrame(tune_rows), use_container_width=True, hide_index=True)

        raw_res = results["raw"]
        log_res = results["log"]
        dates   = results["dates"]

        # ── Section 1: Stationarity & Gating ─────────────────────────────────
        st.subheader("Stationarity & Cointegration")
        info = results.get("alignment", {})
        st.caption(
            f"Observations: {info.get('n_after', results['n_obs'])} daily bars "
            f"({info.get('freq_y','?')} → daily + {info.get('freq_x','?')} → daily) "
            f"| {info.get('date_start','?')} – {info.get('date_end','?')}"
        )

        gate_rows = []
        for space_label, r in [("Raw", raw_res), ("Log", log_res)]:
            gate_rows.append({
                "Test":    f"ADF {label_y}",
                "Space":   space_label,
                "Stat":    r["adf_y"]["adf_stat"],
                "p-value": r["adf_y"]["adf_pvalue"],
                "Verdict": r["adf_y"]["verdict"],
            })
            gate_rows.append({
                "Test":    f"ADF {label_x}",
                "Space":   space_label,
                "Stat":    r["adf_x"]["adf_stat"],
                "p-value": r["adf_x"]["adf_pvalue"],
                "Verdict": r["adf_x"]["verdict"],
            })
            gate_rows.append({
                "Test":    "EG Coint",
                "Space":   space_label,
                "Stat":    r["coint_eg"]["eg_stat"],
                "p-value": r["coint_eg"]["eg_pvalue"],
                "Verdict": r["coint_eg"]["verdict"],
            })
            gate_rows.append({
                "Test":    "Johansen",
                "Space":   space_label,
                "Stat":    r["johansen"]["trace_stat"],
                "p-value": "—",
                "Verdict": r["johansen"]["verdict"],
            })
            gate_rows.append({
                "Test":    "Hurst",
                "Space":   space_label,
                "Stat":    r["hurst"]["hurst"],
                "p-value": "—",
                "Verdict": r["hurst"]["verdict"],
            })

        gate_df = pd.DataFrame(gate_rows)

        def _color_verdict(val):
            if val in ("cointegrated", "I(1)", "strong_mr", "mild_mr", "tradeable"):
                return "background-color: #1a4a1a; color: #7fff7f"
            if val in ("borderline", "ambiguous", "random_walk"):
                return "background-color: #4a4a00; color: #ffff7f"
            if val in ("not_cointegrated", "stationary", "trending", "reject"):
                return "background-color: #4a1a1a; color: #ff7f7f"
            return ""

        styled_gate = gate_df.style.applymap(_color_verdict, subset=["Verdict"])
        st.dataframe(styled_gate, use_container_width=True, hide_index=True)

        tier     = results.get("gate_tier", "REJECT")
        warnings = results.get("gate_warnings", [])

        if tier == "STRONG":
            st.success("**STRONG** — passes all cointegration and mean-reversion criteria.")
        elif tier == "MARGINAL":
            st.warning("**MARGINAL** — passes at relaxed thresholds. Kalman & OU shown below — validate before trading.")
            for w in warnings:
                st.caption(f"⚠ {w}")
        else:
            st.error("**REJECT** — insufficient evidence of cointegration or mean reversion.")
            for w in warnings:
                st.caption(f"✗ {w}")
            st.caption("Consider fixing the price multiplier or uploading longer history.")

        # ── Sections 2–4 only if gate passed ─────────────────────────────────
        if results["gate_passed"]:

            if tier == "MARGINAL":
                st.info(
                    "**[!] Results below are for a MARGINAL pair.** "
                    "Do not trade without further validation: "
                    "(1) Verify price multiplier converts both series to comparable units. "
                    "(2) Check for structural breaks in the price history. "
                    "(3) Consider ratio-adjusted continuous contract instead of back-adjusted."
                )

            # ── Section 2: Kalman beta(t) chart ──────────────────────────────
            st.subheader("Kalman Beta Over Time")

            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(
                x=dates,
                y=raw_res["kalman"]["beta_t"],
                name="β (raw)",
                line=dict(color="#00b4d8", width=1.5),
            ))
            fig_beta.add_trace(go.Scatter(
                x=dates,
                y=log_res["kalman"]["beta_t"],
                name="β (log)",
                line=dict(color="#f4a261", width=1.5, dash="dash"),
                yaxis="y2",
            ))
            fig_beta.update_layout(
                paper_bgcolor=DARK_BG,
                plot_bgcolor=DARK_PLOT,
                font=dict(color=DARK_TEXT),
                xaxis=dict(gridcolor=DARK_GRID),
                yaxis=dict(
                    title=dict(text="β raw", font=dict(color="#00b4d8")),
                    gridcolor=DARK_GRID,
                    tickfont=dict(color="#00b4d8"),
                ),
                yaxis2=dict(
                    title=dict(text="β log", font=dict(color="#f4a261")),
                    overlaying="y",
                    side="right",
                    gridcolor=DARK_GRID,
                    tickfont=dict(color="#f4a261"),
                ),
                legend=dict(bgcolor=DARK_PLOT, bordercolor=DARK_GRID),
                height=300,
                margin=dict(l=60, r=60, t=30, b=30),
            )
            st.plotly_chart(fig_beta, use_container_width=True)

            # ── Section 3: Kalman spread + z-score ───────────────────────────
            st.subheader("Kalman Spread & Z-Score")

            for space_label, k_res in [("Raw", raw_res["kalman"]), ("Log", log_res["kalman"])]:
                spread_arr  = k_res["spread"]
                spread_mean = np.nanmean(spread_arr)
                spread_std  = np.nanstd(spread_arr)
                z_score = (spread_arr - spread_mean) / spread_std if spread_std > 0 else spread_arr * 0

                fig_spd = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.6, 0.4],
                    vertical_spacing=0.05,
                    subplot_titles=[f"Spread ({space_label})", "Z-Score"],
                )
                fig_spd.add_trace(go.Scatter(
                    x=dates, y=spread_arr,
                    name="Spread", line=dict(color="#2ca02c", width=1),
                ), row=1, col=1)
                fig_spd.add_trace(go.Scatter(
                    x=dates, y=z_score,
                    name="Z-Score", line=dict(color="#ff7f0e", width=1),
                ), row=2, col=1)
                for z_lv, col in [(2, "#ff4444"), (-2, "#ff4444"), (1, "#888888"), (-1, "#888888")]:
                    fig_spd.add_hline(y=z_lv, line=dict(color=col, dash="dot", width=1), row=2, col=1)

                fig_spd.update_layout(
                    paper_bgcolor=DARK_BG,
                    plot_bgcolor=DARK_PLOT,
                    font=dict(color=DARK_TEXT),
                    height=380,
                    margin=dict(l=60, r=30, t=40, b=30),
                    showlegend=False,
                )
                fig_spd.update_xaxes(gridcolor=DARK_GRID)
                fig_spd.update_yaxes(gridcolor=DARK_GRID)
                st.plotly_chart(fig_spd, use_container_width=True)

            # ── Section 4: OU parameters table ───────────────────────────────
            st.subheader("Ornstein-Uhlenbeck Parameters")

            ou_rows = []
            ou_reject_notes = []
            for space_label, ou_res in [("Raw", raw_res["ou"]), ("Log", log_res["ou"])]:
                if ou_res is None:
                    continue
                ou_rows.append({
                    "Space":            space_label,
                    "beta_ar1":         ou_res.get("beta_ar1"),
                    "κ (kappa)":        ou_res.get("kappa"),
                    "μ (mu)":           ou_res.get("mu"),
                    "σ_OU":             ou_res.get("ou_std"),
                    "Half-life (days)": ou_res.get("half_life"),
                    "LB p@5":           ou_res.get("lb_pval_5"),
                    "LB p@10":          ou_res.get("lb_pval_10"),
                    "JB p":             ou_res.get("jb_pvalue"),
                    "Verdict":          ou_res.get("verdict"),
                })
                if ou_res.get("reject_reason"):
                    ou_reject_notes.append(f"{space_label}: {ou_res['reject_reason']}")

            if ou_rows:
                ou_df     = pd.DataFrame(ou_rows)
                ou_styled = ou_df.style.applymap(_color_verdict, subset=["Verdict"])
                st.dataframe(ou_styled, use_container_width=True, hide_index=True)
                for note in ou_reject_notes:
                    st.caption(f"✗ {note}")

        # ── Section 5: Final verdict ──────────────────────────────────────────
        st.subheader("Final Verdict")

        if not results["gate_passed"]:
            st.error("REJECT — pair does not pass cointegration/mean-reversion gate.")
        else:
            ou_verdicts = [
                r.get("verdict")
                for r in [raw_res.get("ou"), log_res.get("ou")]
                if r is not None
            ]
            if "tradeable" in ou_verdicts:
                st.success("TRADEABLE — mean-reverting pair with sweet-spot half-life.")
            elif "borderline" in ou_verdicts:
                st.warning("BORDERLINE — some mean-reversion but half-life or residuals marginal.")
            else:
                st.error("REJECT — OU parameters outside acceptable range.")
