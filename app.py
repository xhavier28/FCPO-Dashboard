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
from fcpo_spread_engine import (
    get_active_curve, build_spread_history, build_butterfly_history,
    fair_spread_value, implied_s_backsolve, implied_c,
    conviction_score, scenario_interpretation, entry_conditions_checklist,
    load_spread_history_from_delta_files,
)
from fcpo_s_calculator import (
    load_mpob_history, estimate_capacity, build_regression_dataset,
    fit_s_regression, fit_seasonal_regression, build_seasonal_s_table,
    get_s_mpob, producer_s_composite, build_forward_s_curve, three_source_gaps,
    build_per_pair_regression, load_oni_history, load_enso_forecast,
)
from fcpo_tt_reader import read_all, get_outrights, is_available, get_last_update_time, compute_gaps
import datetime
import csv
from pathlib import Path

SPOT_DIR = "Raw Data"
YEAR_COLORS = {
    2015: "#17becf", 2016: "#bcbd22", 2017: "#7f7f7f", 2018: "#aec7e8", 2019: "#ffbb78",
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
    df = df[df["year"] >= 2015]
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
                df_c["Timestamp (UTC)"]
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


@st.cache_data(ttl=3600)
def load_mpob_data():
    return load_mpob_history("Raw Data/Stock and Production/FCPO Stock 3Y.xlsx")


@st.cache_data(ttl=3600)
def _cached_oni_history():
    return load_oni_history()


@st.cache_data(ttl=3600)
def _cached_enso_forecast():
    return load_enso_forecast()


@st.cache_resource
def get_s_regression_model():
    mpob_df   = load_mpob_data()
    contracts = load_contracts()
    capacity  = estimate_capacity(mpob_df['mpob_stocks'])['working_estimate']

    reg_df    = build_regression_dataset(mpob_df, contracts, capacity=capacity)
    reg_result      = fit_s_regression(reg_df)
    seasonal_result = fit_seasonal_regression(reg_df)
    seasonal_table  = build_seasonal_s_table(mpob_df, reg_result, capacity)

    spread_hist    = load_spread_history_from_delta_files()
    per_pair_reg   = build_per_pair_regression(spread_hist, mpob_df, capacity=capacity)

    return {
        'regression':          reg_result,
        'seasonal':            seasonal_result,
        'seasonal_table':      seasonal_table,
        'capacity':            capacity,
        'reg_df':              reg_df,
        'per_pair_regression': per_pair_reg,
        'spread_history':      spread_hist,
    }


def get_sharepoint_data():
    """Returns read_all() result or None. Called on each dashboard refresh."""
    return read_all()


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
        "Exports":    _read("MPOB Export 3Y.xlsx",      "Exports"),
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
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Spread & S Model**")
    r_annual = st.sidebar.number_input(
        "Risk-free rate — BNM OPR (%)",
        min_value=0.5, max_value=8.0, value=3.0, step=0.25,
        key="r_annual_input",
    ) / 100.0
    st.session_state['r_annual'] = r_annual

df = load_data(SPOT_DIR)

# ── Shared S computation — runs every rerun, propagates to both tabs ──────────
_r_sh       = st.session_state.get('r_annual', 0.03)
_contracts_sh = load_contracts()
_today_sh   = datetime.date.today()
_tt_xlsx_sh = str(Path(__file__).parent / "FCPO_Curve_Input.xlsx")

# 1. S Implied (avg) — 22-day trailing average M1/M2 from Term Structure CSVs
try:
    _fm_sh   = front_month(_today_sh)
    _fm2_sh  = add_months(_fm_sh, 1)
    _s1_sh   = _contracts_sh.get(_fm_sh)
    _s2_sh   = _contracts_sh.get(_fm2_sh)
    if _s1_sh is not None and _s2_sh is not None:
        _F_M1_avg = float(_s1_sh.dropna().tail(22).mean())
        _F_M2_avg = float(_s2_sh.dropna().tail(22).mean())
        if _F_M1_avg > 0 and _F_M2_avg > 0:
            _s_implied_avg = implied_s_backsolve(_F_M1_avg, _F_M2_avg, _r_sh)['s_implied_myr']
        else:
            _s_implied_avg = _F_M1_avg = _F_M2_avg = None
    else:
        _s_implied_avg = _F_M1_avg = _F_M2_avg = None
except Exception:
    _s_implied_avg = _F_M1_avg = _F_M2_avg = None

st.session_state['s_implied_avg'] = _s_implied_avg
st.session_state['F_M1_avg']      = _F_M1_avg
st.session_state['F_M2_avg']      = _F_M2_avg

# 2. S Implied (live) — from SharePoint Excel
try:
    _sp_sh     = read_all(_tt_xlsx_sh)
    _curve_sh  = _sp_sh["outrights"] if _sp_sh else get_active_curve(_contracts_sh, _today_sh)
    _F_M1_live = _curve_sh.get(1) if _curve_sh else None
    _F_M2_live = _curve_sh.get(2) if _curve_sh else None
    if _F_M1_live and _F_M2_live and _F_M1_live > 0:
        _s_implied_live = implied_s_backsolve(_F_M1_live, _F_M2_live, _r_sh)['s_implied_myr']
    else:
        _s_implied_live = None
except Exception:
    _sp_sh = _curve_sh = _F_M1_live = _F_M2_live = _s_implied_live = None

st.session_state['s_implied_live'] = _s_implied_live
st.session_state['F_M1_live']      = _F_M1_live
st.session_state['F_M2_live']      = _F_M2_live
st.session_state['sp_data_shared'] = _sp_sh
st.session_state['current_curve']  = _curve_sh

# 3. S MPOB — regression on MPOB stocks (use widget value from previous run if set)
try:
    _model_sh   = get_s_regression_model()
    _mpob_df_sh = load_mpob_data()
    _latest_stk = float(_mpob_df_sh['mpob_stocks'].iloc[-1]) if not _mpob_df_sh.empty else 1_800_000.0
    _stk_input  = float(st.session_state.get('t7_mpob_stocks', _latest_stk))
    _s_mpob_sh  = get_s_mpob(_stk_input, _model_sh['regression'], _model_sh['capacity'])
    st.session_state['s_mpob']  = _s_mpob_sh['s_mpob_myr']
    st.session_state['regime']  = _s_mpob_sh['regime']
except Exception:
    if 's_mpob' not in st.session_state:
        st.session_state['s_mpob']  = 15.0
        st.session_state['regime']  = 'UNKNOWN'

# 4. S Producer — only initialise if not yet set by producer form
if 's_producer_current' not in st.session_state:
    _s_prod_init = st.session_state.get('s_mpob', 15.0)
    _log_sh = Path("producer_log.csv")
    if _log_sh.exists():
        try:
            _log_df_sh = pd.read_csv(_log_sh)
            if not _log_df_sh.empty:
                _last_sh = _log_df_sh.sort_values('timestamp', ascending=False).iloc[0]
                _s_prod_init = float(_last_sh.get('s_current', _s_prod_init))
                st.session_state['s_producer_forward'] = float(_last_sh.get('s_forward', _s_prod_init))
        except Exception:
            pass
    st.session_state['s_producer_current']       = _s_prod_init
    st.session_state.setdefault('s_producer_forward',        _s_prod_init)
    st.session_state.setdefault('producer_conviction_bonus', 0)

# 5. Convenience yield — prefer live prices, fall back to avg
_F_c = _F_M1_live or _F_M1_avg
_G_c = _F_M2_live or _F_M2_avg
if _F_c and _G_c:
    st.session_state['c_current'] = implied_c(_F_c, _G_c, st.session_state['s_mpob'], _r_sh)
else:
    st.session_state.setdefault('c_current', 0.0)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab_enso = st.tabs([
    "Year-over-Year", "Term Structure", "Event Log",
    "Supply & Demand", "Mean Reversion", "Pair Screener",
    "S Calculator", "Spread & Butterfly", "ENSO Analysis",
])

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

    apply_fx = st.checkbox("Apply FX conversion to Asset B", value=False, key="mr_apply_fx")
    fx_table_file = None
    if apply_fx:
        fx_table_file = st.file_uploader(
            "Upload FX rate table (CSV)",
            type="csv", key="mr_fx_table",
            help="Format: month,rate  /  2022-01,4.19",
        )

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
            data = load_pair(
                file_y.read(), file_x.read(), label_y, label_x,
                mult_y=mult_y, mult_x=mult_x,
                apply_fx=apply_fx,
                fx_table_bytes=fx_table_file.read() if fx_table_file else None,
            )
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

        # ── Early-return: insufficient data ──────────────────────────────────
        if results.get("gate_tier") == "REJECT" and "raw" not in results:
            q = results.get("quality", {})
            sf = q.get("sufficiency", {})
            st.error(
                f"**INSUFFICIENT DATA** — not enough clean bars after break exclusions. "
                f"Daily bars available: {sf.get('n_daily', '?')}. "
                f"Upload longer history and re-run."
            )
            if q:
                br = q.get("break_report", {})
                st.caption(
                    f"Breaks excluded: {br.get('total', 0)}  |  "
                    f"Overall: {q.get('overall','?').upper()}"
                )
            st.stop()

        raw_res = results["raw"]
        log_res = results["log"]
        dates   = results["dates"]

        # ── FX info block ─────────────────────────────────────────────────────
        fx = data.get("fx") or {}
        if fx.get("fx_applied"):
            log = fx["conversion_log"]
            missing = log["n_missing_months"]
            missing_str = (
                "- Missing: 0 months"
                if missing == 0
                else f"- Missing: {missing} month(s) → fallback {log['fallback_rate']:.4f} used\n"
                     + "  " + " / ".join(log["missing_months"])
            )
            st.info(
                f"**FX conversion applied to Asset B**\n"
                f"- Table months: {log['n_table_months']}\n"
                f"- Rate range: {log['rates_min']:.4f} – {log['rates_max']:.4f}"
                f"  (mean {log['rates_mean']:.4f})\n"
                + missing_str
            )

        # ── Quality expander ──────────────────────────────────────────────────
        if "quality" in results:
            q  = results["quality"]
            br = q["break_report"]
            sf = q["sufficiency"]
            with st.expander(
                f"Data Quality  \u2022  {q['overall'].upper()}  \u2022  "
                f"{br['total']} break(s) excluded  \u2022  "
                f"{sf['n_daily']} clean daily bars",
                expanded=(q["overall"] != "good"),
            ):
                st.markdown("**Structural Breaks**")
                if br["total"] == 0:
                    st.success("No structural breaks detected")
                else:
                    rows = []
                    for p in br["periods"]:
                        rows.append({
                            "Date start": str(p.get("date_start", p["bar_start"]))[:10],
                            "Date end":   str(p.get("date_end",   p["bar_end"]))[:10],
                            "Bars":       p["n_bars"],
                            "Peak |z|":   f"{p['peak_z']:.2f}" if p["peak_z"] else "manual",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                    st.caption(
                        f"Mode: {br['mode']} — "
                        f"{br['auto_detected']} auto + {br['manual']} manual periods"
                    )

                st.markdown("**Data Sufficiency**")
                srows = []
                for test, s in sf["tests"].items():
                    icon = {"good": "\u2705", "marginal": "\u26a0\ufe0f", "blocked": "\U0001f6ab"}[s["status"]]
                    srows.append({
                        "Test":   test,
                        "Status": f"{icon} {s['status']}",
                        "Have":   s["n"],
                        "Min":    s["min"],
                        "Rec":    s["rec"],
                    })
                st.dataframe(pd.DataFrame(srows), hide_index=True, use_container_width=True)

                w = q["windows"]
                st.caption(
                    f"Adaptive windows — cointegration: {w['cointegration']}d "
                    f"({w['n_rolling_coint_points']} pts)  |  "
                    f"hurst: {w['hurst']}d ({w['n_rolling_hurst_points']} pts)"
                )

        # ── Section 1: Stationarity & Gating ─────────────────────────────────
        st.subheader("Stationarity & Cointegration")
        _info = results.get("alignment") or {}
        _n     = _info.get("n_after", results["n_obs"])
        _ndays = _info.get("n_common_days", "?")
        _avg   = _info.get("avg_bars_day")
        _avg_s = f"{_avg:.1f}" if isinstance(_avg, (int, float)) else "?"
        if _info.get("mode") == "intraday_positional":
            st.caption(
                f"Observations: {_n} hourly bars across {_ndays} trading days "
                f"(avg {_avg_s} bars/day, positional match within each day) "
                f"| {_info.get('date_start','?')} – {_info.get('date_end','?')}"
            )
        else:
            st.caption(
                f"Observations: {_n} daily bars "
                f"| {_info.get('date_start','?')} – {_info.get('date_end','?')}"
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
                spread_arr  = k_res["spread_reconstructed"]
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
                # Format half-life as "X.Xd (Y.Yh)" when both units available
                _hl_d = ou_res.get("half_life_days") or ou_res.get("half_life")
                _hl_h = ou_res.get("half_life_hours")
                if _hl_d is not None and _hl_h is not None:
                    _hl_str = f"{_hl_d:.1f}d  ({_hl_h:.1f}h)"
                elif _hl_d is not None:
                    _hl_str = f"{_hl_d:.1f}d"
                else:
                    _hl_str = "—"
                ou_rows.append({
                    "Space":       space_label,
                    "beta_ar1":    ou_res.get("beta_ar1"),
                    "κ (kappa)":   ou_res.get("kappa"),
                    "μ (mu)":      ou_res.get("mu"),
                    "σ_OU":        ou_res.get("ou_std"),
                    "Half-life":   _hl_str,
                    "LB p@5":      ou_res.get("lb_pval_5"),
                    "LB p@10":     ou_res.get("lb_pval_10"),
                    "JB p":        ou_res.get("jb_pvalue"),
                    "Verdict":     ou_res.get("verdict"),
                })
                if ou_res.get("reject_reason"):
                    ou_reject_notes.append(f"{space_label}: {ou_res['reject_reason']}")
                if ou_res.get("hl_reason"):
                    ou_reject_notes.append(f"{space_label}: {ou_res['hl_reason']}")

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

            # ── Recommended Hedge Ratio ───────────────────────────────────────
            st.subheader("Recommended Hedge Ratio")

            _raw_ou = raw_res.get("ou") or {}
            _log_ou = log_res.get("ou") or {}
            _p_y    = results.get("p_y_last", 1.0)
            _p_x    = results.get("p_x_last", 1.0)

            def _space_valid(ou):
                # Verdict already encodes whether half-life is in range (freq-agnostic)
                return bool(
                    ou.get("is_valid") and
                    ou.get("verdict") in ("tradeable", "borderline")
                )

            _log_ok = _space_valid(_log_ou)
            _raw_ok = _space_valid(_raw_ou)

            if _log_ok:
                _sel = "log"
            elif _raw_ok:
                _sel = "raw"
            else:
                _sel = "log_fallback"

            best_k       = log_res["kalman"] if _sel in ("log", "log_fallback") else raw_res["kalman"]
            best_ou      = _log_ou           if _sel in ("log", "log_fallback") else _raw_ou
            delta_used   = best_k["delta_used"]
            log_beta     = float(best_k["beta_t"][-1])
            log_alpha    = float(best_k["alpha_t"][-1])
            _hl_d        = best_ou.get("half_life_days") or best_ou.get("half_life")
            _hl_h        = best_ou.get("half_life_hours")
            _res_freq    = results.get("freq", "hourly")

            # Convert log-space beta → price-space hedge ratio
            # log(y) = β_log·log(x) + α  →  price HR = β_log · (P_y / P_x)
            price_hr = log_beta * (_p_y / _p_x) if _sel in ("log", "log_fallback") else log_beta

            if _hl_d is not None and _hl_h is not None:
                _hl_str = f"{_hl_d:.1f}d  ({_hl_h:.1f}h)"
            elif _hl_d is not None:
                _hl_str = f"{_hl_d:.1f}d"
            else:
                _hl_str = "—"
            _space_label = "Log" if _sel in ("log", "log_fallback") else "Raw"

            rec_cols = st.columns(4)
            rec_cols[0].metric("Signal Space", _space_label)
            rec_cols[1].metric("Log β (Kalman)", f"{log_beta:.4f}")
            rec_cols[2].metric("Price-space HR", f"{price_hr:.4f}")
            rec_cols[3].metric("Half-life", _hl_str)

            _freq_note = (
                "Signal frequency: hourly — update hedge ratio and z-score "
                "at each session open or when z-score crosses entry/exit threshold."
                if _res_freq == "hourly" else
                "Signal frequency: daily — update hedge ratio at market close."
            )

            if _sel == "log":
                st.caption(
                    f"**Trade instruction:** Long 1 unit {results['label_y']}, "
                    f"Short **{price_hr:.4f}** units {results['label_x']}. "
                    f"Signal from LOG space (scale-invariant). "
                    f"Enter when log z-score exceeds ±2σ; exit at 0σ. "
                    f"(Kalman δ={delta_used:.0e}, log prices)"
                )
                st.caption(_freq_note)
            elif _sel == "raw":
                st.caption(
                    f"**Trade instruction:** Long 1 unit {results['label_y']}, "
                    f"Short **{log_beta:.4f}** units {results['label_x']}. "
                    f"Signal from RAW price space. "
                    f"Enter when raw z-score exceeds ±2σ; exit at 0σ. "
                    f"(Kalman δ={delta_used:.0e}, raw prices)"
                )
                st.caption(_freq_note)
            else:  # log_fallback
                st.warning(
                    f"INDICATIVE ONLY — OU fit incomplete (half-life unavailable). "
                    f"Log β={log_beta:.4f}, price-space HR ≈ {price_hr:.4f}. "
                    f"Do not trade until half-life is confirmed. "
                    f"Verify the spread_reconstructed bug fix is applied and re-run."
                )

            if best_ou.get("hl_reason"):
                st.caption(f"⚠ {best_ou['hl_reason']}")


# ── Tab 7: S Calculator ───────────────────────────────────────────────────────
with tab7:
    st.header("Storage Cost (S) — Three Source Comparison")
    st.caption(
        "Three-source storage cost model: MPOB regression, seasonal baseline, and producer intelligence. "
        "S drives fair-value calendar spreads. Higher S = higher fair-value spread (backwardation expected)."
    )

    _r7 = st.session_state.get('r_annual', 0.03)

    # Load model once
    try:
        _s_model    = get_s_regression_model()
        _s_reg      = _s_model['regression']
        _s_seasonal = _s_model['seasonal']
        _s_seas_tbl = _s_model['seasonal_table']
        _s_capacity = _s_model['capacity']
        _s_reg_df   = _s_model['reg_df']
        _s_model_ok = True
    except Exception as _e:
        st.error(f"Could not build S regression model: {_e}")
        _s_model_ok = False

    if _s_model_ok:
        # ── Pull all three S sources from shared session state ─────────────────
        _s_implied_avg  = st.session_state.get('s_implied_avg')
        _s_implied_live = st.session_state.get('s_implied_live')
        _s_mpob_t7      = st.session_state.get('s_mpob', 15.0)
        _regime_t7      = st.session_state.get('regime', 'UNKNOWN')
        _s_producer_t7  = st.session_state.get('s_producer_current', _s_mpob_t7)
        _s_forward_t7   = st.session_state.get('s_producer_forward',  _s_mpob_t7)
        _current_month_t7 = _today_sh.month
        _log_path       = Path("producer_log.csv")

        # Still need MPOB data for the detail section below
        _mpob_df_t7 = load_mpob_data()
        if not _mpob_df_t7.empty:
            _latest_stocks_t7 = float(_mpob_df_t7['mpob_stocks'].iloc[-1])
            _latest_date_t7   = _mpob_df_t7['date'].iloc[-1]
        else:
            _latest_stocks_t7 = 1_800_000.0
            _latest_date_t7   = pd.Timestamp.today()

        # Reference F_M1 for producer log (prefer live, fall back to avg)
        _F_M1_t7 = st.session_state.get('F_M1_live') or st.session_state.get('F_M1_avg') or 4000.0

        # ── TOP: Three-source comparison ───────────────────────────────────────
        _tc1, _tc2, _tc3 = st.columns(3)
        with _tc1:
            st.metric(
                "S Implied — monthly avg",
                f"{_s_implied_avg:.1f} MYR/t" if _s_implied_avg is not None else "—",
                help="Back-solved from 22-day average M1/M2 prices (Term Structure CSVs)",
            )
            if _s_implied_live is not None and _s_implied_avg is not None:
                _live_dir = '↑ above' if _s_implied_live > _s_implied_avg else '↓ below'
                st.caption(f"Live (Excel): {_s_implied_live:.1f} MYR/t — "
                           f"{_live_dir} monthly avg by {abs(_s_implied_live - _s_implied_avg):.1f}")
            elif _s_implied_live is not None:
                st.caption(f"Live (Excel): {_s_implied_live:.1f} MYR/t")
        with _tc2:
            st.metric(
                "S MPOB — official",
                f"{_s_mpob_t7:.1f} MYR/t",
                delta=f"{_s_mpob_t7 - _s_implied_avg:+.1f} vs implied avg" if _s_implied_avg is not None else None,
                help=f"Regime: {_regime_t7}",
            )
        with _tc3:
            st.metric(
                "S Producer — physical",
                f"{_s_producer_t7:.1f} MYR/t",
                delta=f"{_s_producer_t7 - _s_implied_avg:+.1f} vs implied avg" if _s_implied_avg is not None else None,
                help="From producer intelligence form — updates when you click Compute & Log",
            )

        st.markdown("---")

        _ref_s = _s_implied_avg if _s_implied_avg is not None else _s_mpob_t7
        _gc1, _gc2, _gc3 = st.columns(3)
        _gc1.metric("Gap 1 — Implied vs MPOB",  f"{_ref_s - _s_mpob_t7:+.1f} MYR/t")
        _gc2.metric("Gap 2 — MPOB vs Producer", f"{_s_mpob_t7 - _s_producer_t7:+.1f} MYR/t")
        _alpha_gap_t7 = _ref_s - _s_producer_t7
        _gc3.metric(
            "Gap 3 — YOUR ALPHA (Implied vs Producer)",
            f"{_alpha_gap_t7:+.1f} MYR/t",
            delta="BUY SPREAD" if _alpha_gap_t7 < -8 else "SELL SPREAD" if _alpha_gap_t7 > 8 else "NEUTRAL",
        )

        st.markdown("---")

        # ── Market Tightness Proxy — Implied S vs Historical ────────────────────
        st.subheader("Market Tightness Proxy — Implied S vs Historical")
        st.caption(
            "Implied S back-solved from M1/M2 prices. Negative = backwardation (tight market). "
            "Positive = contango. Use as tightness direction indicator, not as absolute storage cost."
        )

        if not _s_reg_df.empty:
            _seas_df = _s_reg_df.copy()
            _seas_df['month'] = pd.to_datetime(_seas_df['date']).dt.month
            _seas_df['year']  = pd.to_datetime(_seas_df['date']).dt.year

            _cy = datetime.date.today().year
            _ly = _cy - 1

            _hist_seas = (
                _seas_df[_seas_df['year'] < _ly]
                .groupby('month')['s_implied']
                .agg(s_mean='mean', s_min='min', s_max='max')
                .reset_index()
            )
            _ly_seas = (
                _seas_df[_seas_df['year'] == _ly]
                .groupby('month')['s_implied'].mean()
                .reset_index()
            )
            _ly_seas.columns = ['month', 's_last_year']
            _cy_seas = (
                _seas_df[_seas_df['year'] == _cy]
                .groupby('month')['s_implied'].mean()
                .reset_index()
            )
            _cy_seas.columns = ['month', 's_current_year']

            _seasonal = (
                _hist_seas
                .merge(_ly_seas, on='month', how='left')
                .merge(_cy_seas, on='month', how='left')
            )
            _seasonal['gap_vs_hist'] = _seasonal['s_current_year'] - _seasonal['s_mean']
            _month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                             'Jul','Aug','Sep','Oct','Nov','Dec']
            _seasonal['month_label'] = _seasonal['month'].apply(lambda x: _month_labels[x - 1])

            # Line chart
            _fig_seas = go.Figure()
            _fig_seas.add_trace(go.Scatter(
                x=_seasonal['month_label'].tolist() + _seasonal['month_label'].tolist()[::-1],
                y=_seasonal['s_max'].tolist() + _seasonal['s_min'].tolist()[::-1],
                fill='toself', fillcolor='rgba(150,150,150,0.15)',
                line=dict(width=0), name='Historical range', showlegend=True,
            ))
            _fig_seas.add_trace(go.Scatter(
                x=_seasonal['month_label'], y=_seasonal['s_mean'],
                line=dict(color='#888888', width=1.5, dash='dot'),
                name='Historical mean',
            ))
            _fig_seas.add_trace(go.Scatter(
                x=_seasonal['month_label'], y=_seasonal['s_last_year'],
                line=dict(color='#4fc3f7', width=1.5),
                name=str(_ly),
            ))
            _fig_seas.add_trace(go.Scatter(
                x=_seasonal['month_label'], y=_seasonal['s_current_year'],
                line=dict(color='#ff6b35', width=2.5),
                mode='lines+markers', name=str(_cy),
            ))
            _fig_seas.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
                font_color=DARK_TEXT,
                xaxis=dict(gridcolor=DARK_GRID),
                yaxis=dict(gridcolor=DARK_GRID, title='S implied (MYR/t)'),
                height=320, margin=dict(l=40, r=20, t=30, b=30),
                legend=dict(orientation='h', y=1.02, font=dict(color=DARK_TEXT)),
            )
            st.plotly_chart(_fig_seas, use_container_width=True)

            # Gap bar chart
            _fig_gap = go.Figure()
            _fig_gap.add_trace(go.Bar(
                x=_seasonal['month_label'],
                y=_seasonal['gap_vs_hist'],
                marker_color=[
                    '#ef5350' if v > 0 else '#66bb6a'
                    for v in _seasonal['gap_vs_hist'].fillna(0)
                ],
                name='Gap vs historical mean',
            ))
            _fig_gap.add_hline(y=0, line_color='#888888', line_dash='dot', line_width=1)
            _fig_gap.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
                font_color=DARK_TEXT,
                xaxis=dict(gridcolor=DARK_GRID),
                yaxis=dict(gridcolor=DARK_GRID, title='Gap vs hist mean (MYR/t)'),
                height=200, margin=dict(l=40, r=20, t=10, b=30),
            )
            st.plotly_chart(_fig_gap, use_container_width=True)

            # One-line summary
            _cy_valid = _seasonal[_seasonal['s_current_year'].notna()]
            if not _cy_valid.empty:
                _latest_gap = _cy_valid['gap_vs_hist'].iloc[-1]
                _trend      = _cy_valid['gap_vs_hist'].diff().iloc[-1]
                if _latest_gap > 3 and _trend > 0:
                    _seas_msg = (f"Current year S running +{_latest_gap:.1f} MYR/t above historical "
                                 f"average and gap is widening — accelerating tightness signal.")
                elif _latest_gap > 3 and _trend <= 0:
                    _seas_msg = (f"Current year S running +{_latest_gap:.1f} MYR/t above historical "
                                 f"average but gap is narrowing — tightness easing.")
                elif _latest_gap < -3 and _trend < 0:
                    _seas_msg = (f"Current year S running {_latest_gap:.1f} MYR/t below historical "
                                 f"average and gap is widening — accelerating looseness.")
                elif _latest_gap < -3 and _trend >= 0:
                    _seas_msg = (f"Current year S running {_latest_gap:.1f} MYR/t below historical "
                                 f"average but gap is narrowing — conditions normalising.")
                else:
                    _seas_msg = (f"Current year S tracking close to historical average "
                                 f"({_latest_gap:+.1f} MYR/t). No strong seasonal deviation.")
                st.caption(_seas_msg)
        else:
            st.info("Regression dataset empty — seasonality chart unavailable.")

        st.markdown("---")

        # ── MPOB Storage Cost (S) — Utilisation Based vs Historical ──────────
        st.subheader("MPOB Storage Cost (S) — Utilisation Based vs Historical")
        st.caption(
            "S estimated from MPOB inventory utilisation via regression. "
            "Always positive — represents physical storage cost. "
            "Compare current year vs historical to see if storage pressure "
            "is above or below seasonal norm."
        )

        _mpob_util_fn = _s_reg.get('util_to_s_function') if _s_model_ok and _s_reg else None
        if _mpob_util_fn:
            _mpob_df_t7   = load_mpob_data()
            _mpob_records = []
            for _, _mrow in _mpob_df_t7.iterrows():
                _md = pd.to_datetime(_mrow['date'])
                _mu = _mrow['mpob_stocks'] / _s_capacity
                _ms = _mpob_util_fn(_mu)
                if 3.0 <= _ms <= 80.0:
                    _mpob_records.append({
                        'date': _md, 'year': _md.year, 'month': _md.month,
                        's_mpob': round(_ms, 2),
                    })

            _mpob_s_df = pd.DataFrame(_mpob_records).sort_values('date')
            _m_cy = datetime.date.today().year
            _m_ly = _m_cy - 1

            _m_hist = (
                _mpob_s_df[_mpob_s_df['year'] < _m_ly]
                .groupby('month')['s_mpob']
                .agg(s_mean='mean', s_min='min', s_max='max')
                .reset_index()
            )
            _m_ly_g = (
                _mpob_s_df[_mpob_s_df['year'] == _m_ly]
                .groupby('month')['s_mpob'].mean()
                .reset_index().rename(columns={'s_mpob': 's_last_year'})
            )
            _m_cy_g = (
                _mpob_s_df[_mpob_s_df['year'] == _m_cy]
                .groupby('month')['s_mpob'].mean()
                .reset_index().rename(columns={'s_mpob': 's_current_year'})
            )

            _df_mpob_seas = (
                _m_hist
                .merge(_m_ly_g, on='month', how='left')
                .merge(_m_cy_g, on='month', how='left')
            )
            _m_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                         'Jul','Aug','Sep','Oct','Nov','Dec']
            _df_mpob_seas['month_label'] = _df_mpob_seas['month'].apply(lambda x: _m_labels[x - 1])

            _fig_mpob = go.Figure()
            _fig_mpob.add_trace(go.Scatter(
                x=_df_mpob_seas['month_label'].tolist() + _df_mpob_seas['month_label'].tolist()[::-1],
                y=_df_mpob_seas['s_max'].tolist() + _df_mpob_seas['s_min'].tolist()[::-1],
                fill='toself', fillcolor='rgba(150,150,150,0.15)',
                line=dict(width=0), name='Historical range',
            ))
            _fig_mpob.add_trace(go.Scatter(
                x=_df_mpob_seas['month_label'], y=_df_mpob_seas['s_mean'],
                line=dict(color='#888888', width=1.5, dash='dot'), name='Historical mean',
            ))
            _fig_mpob.add_trace(go.Scatter(
                x=_df_mpob_seas['month_label'], y=_df_mpob_seas['s_last_year'],
                line=dict(color='#4fc3f7', width=1.5),
                mode='lines+markers', marker=dict(size=5), name=str(_m_ly),
            ))
            _fig_mpob.add_trace(go.Scatter(
                x=_df_mpob_seas['month_label'], y=_df_mpob_seas['s_current_year'],
                line=dict(color='#ff6b35', width=2.5),
                mode='lines+markers', marker=dict(size=7), name=str(_m_cy),
            ))
            _fig_mpob.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT, font_color=DARK_TEXT,
                xaxis_gridcolor=DARK_GRID, yaxis_gridcolor=DARK_GRID,
                height=320, yaxis_title='S MPOB (MYR/t)',
                margin=dict(l=40, r=40, t=30, b=30),
                legend=dict(orientation='h', y=1.02),
            )
            st.plotly_chart(_fig_mpob, use_container_width=True, key="mpob_s_seas")

            # Gap bar
            _df_mpob_gap = _df_mpob_seas.copy()
            _df_mpob_gap['gap'] = _df_mpob_gap['s_current_year'] - _df_mpob_gap['s_mean']
            _fig_mpob_gap = go.Figure(go.Bar(
                x=_df_mpob_gap['month_label'], y=_df_mpob_gap['gap'],
                marker_color=['#ef5350' if v > 0 else '#66bb6a' for v in _df_mpob_gap['gap'].fillna(0)],
            ))
            _fig_mpob_gap.add_hline(y=0, line_color='#888888', line_dash='dot', line_width=1)
            _fig_mpob_gap.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT, font_color=DARK_TEXT,
                height=180, yaxis_title='Gap vs hist mean (MYR/t)',
                margin=dict(l=40, r=40, t=10, b=30),
            )
            st.plotly_chart(_fig_mpob_gap, use_container_width=True, key="mpob_s_gap")

            if not _df_mpob_gap.empty:
                _ml = _df_mpob_gap.iloc[-1]
                _mg = _ml['gap']
                st.caption(
                    f"Latest MPOB S: {_ml['s_current_year']:.1f} MYR/t  ·  "
                    f"Historical mean: {_ml['s_mean']:.1f} MYR/t  ·  "
                    f"Gap: {_mg:+.1f} MYR/t  ·  "
                    f"{'Above normal — inventory tighter than usual' if _mg > 1 else 'Below normal — inventory looser than usual' if _mg < -1 else 'Within normal range'}"
                )
        else:
            st.warning("Regression model unavailable — extend MPOB history to enable this chart.")

        st.markdown("---")

        # ── M1/M2 Actual Spread — Seasonal Pattern ──────────────────────────
        st.subheader("M1/M2 Actual Spread — Seasonal Pattern")
        st.caption(
            "Actual M1/M2 spread (BMD convention: near minus far). "
            "Negative = contango. Positive = backwardation. "
            "Uses pre-calculated spread values from daily term delta files. "
            "Directly shows what the tradeable spread has done seasonally."
        )

        @st.cache_data(ttl=3600)
        def _cached_spread_hist():
            return load_spread_history_from_delta_files()

        _spread_hist = _cached_spread_hist()

        if not _spread_hist.empty and 'spread_m1m2' in _spread_hist.columns:
            _sp_monthly = (
                _spread_hist.groupby(['year', 'month'])['spread_m1m2']
                .agg(mean='mean', min='min', max='max', count='count')
                .reset_index()
            )
            _sp_monthly = _sp_monthly[_sp_monthly['count'] >= 5]

            _sp_cy = datetime.date.today().year
            _sp_ly = _sp_cy - 1

            _sp_hist = (
                _sp_monthly[_sp_monthly['year'] < _sp_ly]
                .groupby('month')
                .agg(s_mean=('mean', 'mean'), s_min=('min', 'min'), s_max=('max', 'max'))
                .reset_index()
            )
            _sp_ly_g = (
                _sp_monthly[_sp_monthly['year'] == _sp_ly][['month', 'mean']]
                .rename(columns={'mean': 's_last_year'})
            )
            _sp_cy_g = (
                _sp_monthly[_sp_monthly['year'] == _sp_cy][['month', 'mean']]
                .rename(columns={'mean': 's_current_year'})
            )
            _df_sp = (
                _sp_hist
                .merge(_sp_ly_g, on='month', how='left')
                .merge(_sp_cy_g, on='month', how='left')
            )
            _sp_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec']
            _df_sp['month_label'] = _df_sp['month'].apply(lambda x: _sp_labels[x - 1])

            _fig_sp = go.Figure()
            _fig_sp.add_trace(go.Scatter(
                x=_df_sp['month_label'].tolist() + _df_sp['month_label'].tolist()[::-1],
                y=_df_sp['s_max'].tolist() + _df_sp['s_min'].tolist()[::-1],
                fill='toself', fillcolor='rgba(150,150,150,0.15)',
                line=dict(width=0), name='Historical range',
            ))
            _fig_sp.add_trace(go.Scatter(
                x=_df_sp['month_label'], y=_df_sp['s_mean'],
                line=dict(color='#888888', width=1.5, dash='dot'), name='Historical mean',
            ))
            _fig_sp.add_trace(go.Scatter(
                x=_df_sp['month_label'], y=_df_sp['s_last_year'],
                line=dict(color='#4fc3f7', width=1.5),
                mode='lines+markers', marker=dict(size=5), name=str(_sp_ly),
            ))
            _fig_sp.add_trace(go.Scatter(
                x=_df_sp['month_label'], y=_df_sp['s_current_year'],
                line=dict(color='#ff6b35', width=2.5),
                mode='lines+markers', marker=dict(size=7), name=str(_sp_cy),
            ))
            _fig_sp.add_hline(
                y=0, line_color='#ffffff', line_dash='dot', line_width=0.8,
                annotation_text='0 — flat', annotation_position='right',
            )
            _fig_sp.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT, font_color=DARK_TEXT,
                xaxis_gridcolor=DARK_GRID, yaxis_gridcolor=DARK_GRID,
                height=320, yaxis_title='M1/M2 spread (MYR/t) — BMD: near minus far',
                margin=dict(l=40, r=80, t=30, b=30),
                legend=dict(orientation='h', y=1.02),
            )
            st.plotly_chart(_fig_sp, use_container_width=True, key="sp_m1m2_seas")

            # Gap bar
            _df_sp_gap = _df_sp.copy()
            _df_sp_gap['gap'] = _df_sp_gap['s_current_year'] - _df_sp_gap['s_mean']
            _fig_sp_gap = go.Figure(go.Bar(
                x=_df_sp_gap['month_label'], y=_df_sp_gap['gap'],
                marker_color=['#ef5350' if v > 0 else '#66bb6a' for v in _df_sp_gap['gap'].fillna(0)],
            ))
            _fig_sp_gap.add_hline(y=0, line_color='#888888', line_dash='dot', line_width=1)
            _fig_sp_gap.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT, font_color=DARK_TEXT,
                height=180, yaxis_title='Gap vs hist mean (MYR/t)',
                margin=dict(l=40, r=80, t=10, b=30),
            )
            st.plotly_chart(_fig_sp_gap, use_container_width=True, key="sp_m1m2_gap")

            if not _df_sp_gap.empty:
                _spl     = _df_sp_gap.iloc[-1]
                _sp_gap  = _spl['gap']
                _sp_act  = _spl['s_current_year']
                _sp_mean = _spl['s_mean']
                _sp_regime = "backwardation" if _sp_act > 0 else "contango"
                st.caption(
                    f"Latest M1/M2 spread: {_sp_act:+.1f} MYR/t ({_sp_regime})  ·  "
                    f"Historical mean: {_sp_mean:+.1f} MYR/t  ·  "
                    f"Gap: {_sp_gap:+.1f} MYR/t  ·  "
                    f"{'Richer than historical average — market tighter than usual' if _sp_gap > 5 else 'Cheaper than historical average — market looser than usual' if _sp_gap < -5 else 'Near historical average'}"
                )
        else:
            st.warning(
                "No pre-calculated spread data found. "
                "Ensure fcpo_daily_term_delta_*.xlsx files are in Raw Data/Daily Term/ folder."
            )

        st.markdown("---")

        # ── Model Diagnostics (collapsed) ─────────────────────────────────────
        with st.expander("Model Diagnostics", expanded=False):
            _diag_cols = st.columns(4)
            _diag_cols[0].metric("Best model",    _s_reg['best_model'].title())
            _diag_cols[1].metric("R²",            f"{_s_reg['best_r2']:.3f}")
            _diag_cols[2].metric("Observations",  _s_reg['n_observations'])
            _diag_cols[3].metric("Capacity est.", f"{_s_capacity/1e6:.2f}M t")
            if _s_reg.get('warning'):
                st.warning(_s_reg['warning'])
            st.caption(
                f"Util range: {_s_reg['util_range'][0]:.2f}–{_s_reg['util_range'][1]:.2f}  |  "
                f"S observed: {_s_reg['s_range_observed'][0]:.1f}–{_s_reg['s_range_observed'][1]:.1f} MYR/t  |  "
                f"Seasonal OLS R²: {_s_seasonal['r2']:.3f}"
            )
            if not _s_reg_df.empty:
                _u_range = np.linspace(
                    float(_s_reg_df['utilisation'].min()),
                    float(_s_reg_df['utilisation'].max()), 80
                )
                _s_fit = [_s_reg['util_to_s_function'](u) for u in _u_range]
                _fig_reg = go.Figure()
                _fig_reg.add_trace(go.Scatter(
                    x=_s_reg_df['utilisation'], y=_s_reg_df['s_implied'],
                    mode='markers', name='Observed',
                    marker=dict(color='#00b4d8', size=6, opacity=0.7),
                    hovertemplate='Util=%{x:.3f}<br>S=%{y:.1f} MYR/t<extra></extra>',
                ))
                _fig_reg.add_trace(go.Scatter(
                    x=_u_range, y=_s_fit,
                    mode='lines', name=f'Fit ({_s_reg["best_model"]})',
                    line=dict(color='#f4a261', width=2),
                ))
                _fig_reg.update_layout(
                    plot_bgcolor=DARK_PLOT, paper_bgcolor=DARK_BG,
                    font=dict(color=DARK_TEXT),
                    xaxis=dict(title='Utilisation', showgrid=True, gridcolor=DARK_GRID),
                    yaxis=dict(title='S implied (MYR/t)', showgrid=True, gridcolor=DARK_GRID),
                    height=320, margin=dict(l=60, r=30, t=30, b=50),
                    legend=dict(font=dict(color=DARK_TEXT)),
                )
                st.plotly_chart(_fig_reg, use_container_width=True)

        # ── Seasonal S table (collapsed) ──────────────────────────────────────
        with st.expander("Seasonal S Table (by month)", expanded=False):
            _seas_rows = []
            for _m in range(1, 13):
                _entry = _s_seas_tbl.get(_m, {})
                _seas_rows.append({
                    'Month':     _entry.get('month_name', MONTH_ABBRS[_m - 1]),
                    'Util mean': f"{_entry.get('util_mean', 0):.3f}",
                    'Util P25':  f"{_entry.get('util_p25', 0):.3f}",
                    'Util P75':  f"{_entry.get('util_p75', 0):.3f}",
                    'S mean':    f"{_entry.get('s_mean', 0):.1f}",
                    'S P25':     f"{_entry.get('s_low', 0):.1f}",
                    'S P75':     f"{_entry.get('s_high', 0):.1f}",
                    'N years':   _entry.get('n_years', 0),
                })
            st.dataframe(pd.DataFrame(_seas_rows).set_index('Month'), use_container_width=True)

        st.markdown("---")

        # ── MPOB stocks input ─────────────────────────────────────────────────
        st.subheader("MPOB-Based S (Current)")
        _t7c1, _t7c2 = st.columns([2, 3])
        with _t7c1:
            _current_stocks_input = st.number_input(
                "Current MPOB stocks (tonnes)",
                min_value=100_000, max_value=5_000_000,
                value=int(_latest_stocks_t7),
                step=10_000,
                key="t7_mpob_stocks",
            )
        with _t7c2:
            st.caption(f"Last MPOB data point: {_latest_date_t7.strftime('%b %Y')} — {_latest_stocks_t7:,.0f} t")

        _s_mpob_res = get_s_mpob(_current_stocks_input, _s_reg, _s_capacity)
        _m7c1, _m7c2, _m7c3 = st.columns(3)
        _m7c1.metric("Utilisation", f"{_s_mpob_res['utilisation']:.1%}")
        _m7c2.metric("S (MPOB)",    f"{_s_mpob_res['s_mpob_myr']:.1f} MYR/t")
        _m7c3.metric("Regime",      _s_mpob_res['regime'])

        st.markdown("---")

        # ── Producer Intelligence form ────────────────────────────────────────
        st.subheader("Producer Intelligence")
        recent_prod = pd.DataFrame()
        if _log_path.exists():
            try:
                recent_prod = pd.read_csv(_log_path)
                if not recent_prod.empty:
                    recent_prod['timestamp'] = pd.to_datetime(recent_prod['timestamp'])
                    recent_prod = recent_prod.sort_values('timestamp', ascending=False).head(10)
            except Exception:
                recent_prod = pd.DataFrame()

        _col_min, _col_cur, _col_max = st.columns(3)
        _hist_low  = _col_min.number_input("Historical low (%)",  min_value=0, max_value=100, value=35, step=1, key="t7_hist_low")
        _current   = _col_cur.number_input("Current (%)",         min_value=0, max_value=100, value=65, step=1, key="t7_tank_util")
        _hist_high = _col_max.number_input("Historical high (%)", min_value=0, max_value=100, value=88, step=1, key="t7_hist_high")

        _rel_pos = (_current - _hist_low) / (_hist_high - _hist_low) if _hist_high != _hist_low else 0.5
        _rel_label = (
            'very tight'  if _rel_pos > 0.75 else
            'tightening'  if _rel_pos > 0.50 else
            'comfortable' if _rel_pos > 0.25 else
            'very loose'
        )
        st.caption(f"Relative position: {_rel_pos*100:.0f}th percentile of their range  ({_rel_label})")

        _pf_col1, _pf_col2 = st.columns(2)
        with _pf_col1:
            _buyer_lifting = st.selectbox(
                "Buyer lifting pace",
                options=['rushing', 'on_time', 'slight_delay', 'major_delay'],
                index=1, key="t7_buyer_lifting",
            )
        with _pf_col2:
            _discount_pressure = st.selectbox(
                "Discount pressure",
                options=['none', 'small', 'large', 'distress'],
                index=0, key="t7_discount_pressure",
            )
        _production_outlook = st.selectbox(
            "Production outlook (next month)",
            options=['light', 'normal', 'heavy'],
            index=1, key="t7_production_outlook",
        )
        _submit_prod = st.button("Compute & Log", key="t7_compute_btn")

        if _submit_prod:
            _prod_res = producer_s_composite(
                _rel_pos, _buyer_lifting, _discount_pressure,
                _production_outlook, _s_seas_tbl, _current_month_t7,
            )
            # Push to session state — top panel reads these on next rerun
            st.session_state['s_producer_current']       = _prod_res['s_current']
            st.session_state['s_producer_forward']        = _prod_res['s_forward']
            st.session_state['producer_conviction_bonus'] = _prod_res['conviction_bonus']
            st.session_state['s_producer_rel_pos']        = _rel_pos
            # Also update forward curve override inputs
            st.session_state['t7_s_prod_cur'] = float(max(5.0, min(50.0, _prod_res['s_current'])))
            st.session_state['t7_s_prod_fwd'] = float(max(5.0, min(50.0, _prod_res['s_forward'])))

            _log_row = {
                'timestamp':          datetime.datetime.now().isoformat(),
                'hist_low_pct':       _hist_low,
                'current_pct':        _current,
                'hist_high_pct':      _hist_high,
                'rel_pos':            round(_rel_pos, 4),
                'buyer_lifting':      _buyer_lifting,
                'discount_pressure':  _discount_pressure,
                'production_outlook': _production_outlook,
                'F_M1':               round(_F_M1_t7, 1),
                's_current':          round(_prod_res['s_current'], 2),
                's_forward':          round(_prod_res['s_forward'], 2),
                'signal':             _prod_res['signal'],
            }
            _write_header = not _log_path.exists() or _log_path.stat().st_size == 0
            with open(_log_path, 'a', newline='') as _lf:
                _writer = csv.DictWriter(_lf, fieldnames=list(_log_row.keys()))
                if _write_header:
                    _writer.writeheader()
                _writer.writerow(_log_row)

            # Re-run immediately so the top panel reads the new session state values
            st.rerun()

        # Monthly context caption — always visible, updates after each submit
        _cm_data    = _s_seas_tbl.get(_current_month_t7, {})
        _cm_s_mean  = _cm_data.get('s_mean', 15.0)
        _cm_s_low   = _cm_data.get('s_low',  8.0)
        _cm_s_high  = _cm_data.get('s_high', 28.0)
        _cm_rel_pos = st.session_state.get('s_producer_rel_pos', _rel_pos)
        _cm_s_est   = st.session_state.get('s_producer_current', _s_producer_t7)
        st.caption(
            f"This month's S range (from MPOB history): "
            f"{_cm_s_low:.1f} – {_cm_s_high:.1f} MYR/t  ·  "
            f"Historical mean: {_cm_s_mean:.1f} MYR/t  ·  "
            f"Your producer at {_cm_rel_pos*100:.0f}th percentile → "
            f"S estimate: {_cm_s_est:.1f} MYR/t"
        )

        if not recent_prod.empty:
            st.subheader("Recent Producer Log (last 10 entries)")
            st.dataframe(recent_prod.drop(columns=['timestamp'], errors='ignore'), use_container_width=True)

        st.markdown("---")

        # ── Forward S Curve — Fair Value vs Actual ────────────────────────────

        # Default override boxes to log-derived values (only on first load)
        if 't7_s_prod_cur' not in st.session_state:
            st.session_state['t7_s_prod_cur'] = float(max(5.0, min(50.0, _s_producer_t7)))
        if 't7_s_prod_fwd' not in st.session_state:
            st.session_state['t7_s_prod_fwd'] = float(max(5.0, min(50.0, _s_forward_t7)))

        _fwd_inp_c1, _fwd_inp_c2 = st.columns(2)
        with _fwd_inp_c1:
            _enso_label = st.selectbox(
                "ENSO phase (back month context only)",
                options=["La Nina — above normal production expected",
                         "Neutral",
                         "El Nino — below normal production expected"],
                index=1, key="t7_enso_label",
            )
        with _fwd_inp_c2:
            _s_producer_current_input = st.number_input(
                "S producer current (MYR/t) — from form above",
                min_value=5.0, max_value=50.0,
                value=st.session_state['t7_s_prod_cur'],
                step=0.5, key="t7_s_prod_cur",
            )
            _s_producer_forward_input = st.number_input(
                "S producer forward (MYR/t)",
                min_value=5.0, max_value=50.0,
                value=st.session_state['t7_s_prod_fwd'],
                step=0.5, key="t7_s_prod_fwd",
            )

        _current_curve_t7  = st.session_state.get('current_curve', {})
        _per_pair_reg      = _s_model.get('per_pair_regression', {})
        _util_fn_t7        = _s_reg.get('util_to_s_function') if _s_reg else None
        _mpob_stk_t7       = _current_stocks_input
        _current_util_t7   = _mpob_stk_t7 / _s_capacity
        _current_month_t7  = datetime.date.today().month

        # Year ratio for M6+ adjustment
        _hist_util_cm   = _s_seas_tbl.get(_current_month_t7, {}).get('util_mean', _current_util_t7)
        _year_ratio     = _current_util_t7 / _hist_util_cm if _hist_util_cm > 0 else 1.0

        _fwd_rows = []
        for _i in range(1, 12):
            _near = _i; _far = _i + 1
            _pair = f"M{_near}/M{_far}"

            # Live prices
            _F_near = _current_curve_t7.get(_near)
            _F_far  = _current_curve_t7.get(_far)
            _actual_spread = round(_F_near - _F_far, 1) if _F_near and _F_far else None

            # Implied S for this pair
            _s_implied_pair = None
            if _F_near and _F_far and _F_near > 0 and _F_far > 0:
                try:
                    _res_impl = implied_s_backsolve(_F_near, _F_far, _r7)
                    _s_implied_pair = round(_res_impl['s_implied_myr'], 1)
                except Exception:
                    pass

            # Model S — source depends on pair
            _model_s = None; _model_spread = None; _source = "—"; _conf = "LOW"

            if _near == 1:
                _reg_pp = _per_pair_reg.get('M1/M2')
                if _reg_pp and _reg_pp.get('util_to_spread') and _reg_pp['r2'] >= 0.20:
                    _model_spread = round(_reg_pp['util_to_spread'](_current_util_t7), 1)
                    _source = f"regression (R²={_reg_pp['r2']})"
                    _conf = "HIGH" if _reg_pp['r2'] >= 0.4 else "MEDIUM"
                else:
                    _model_s = _s_producer_current_input
                    _source = "producer"
                    _conf = "HIGH"
            elif _near == 2:
                _reg_pp = _per_pair_reg.get('M2/M3')
                if _reg_pp and _reg_pp.get('util_to_spread') and _reg_pp['r2'] >= 0.20:
                    _model_spread = round(_reg_pp['util_to_spread'](_current_util_t7), 1)
                    _source = f"regression (R²={_reg_pp['r2']})"
                    _conf = "HIGH" if _reg_pp['r2'] >= 0.4 else "MEDIUM"
                else:
                    _model_s = _s_producer_forward_input
                    _source = "producer forward"
                    _conf = "MED-HIGH"
            elif _near <= 5:
                if _util_fn_t7:
                    _model_s = round(_util_fn_t7(_current_util_t7), 1)
                    _source = "MPOB current"
                    _conf = "MEDIUM"
                else:
                    _seas_entry = _s_seas_tbl.get(_current_month_t7, {})
                    _model_s = round(_seas_entry.get('s_mean', 15.0), 1)
                    _source = "seasonal"
                    _conf = "MEDIUM"
            else:
                _near_mo = ((_current_month_t7 - 1 + _near - 1) % 12) + 1
                _far_mo  = ((_current_month_t7 - 1 + _far - 1) % 12) + 1
                _s_near_s = _s_seas_tbl.get(_near_mo, {}).get('s_mean', 15.0)
                _s_far_s  = _s_seas_tbl.get(_far_mo, {}).get('s_mean', 15.0)
                _s_base   = (_s_near_s + _s_far_s) / 2
                _dampen   = max(0.0, 1.0 - (_near - 6) * 0.10)
                _ratio_adj = 1.0 + (_year_ratio - 1.0) * _dampen
                _model_s  = round(max(8.0, min(35.0, _s_base * _ratio_adj)), 1)
                _source   = "seasonal + year ratio"
                _conf     = "LOW" if _near <= 8 else "VERY LOW"

            # Fair spread
            if _model_spread is not None:
                _fair_spread = _model_spread
            elif _model_s is not None and _F_near and _F_near > 0:
                import math as _math
                _r_m   = _r7 / 12
                _s_rate = _model_s / _F_near
                _fair_spread = round(_F_near - _F_near * _math.exp((_r_m + _s_rate) * (1/12)), 1)
            else:
                _fair_spread = None

            # Edge
            _edge = round(_actual_spread - _fair_spread, 1) \
                    if _actual_spread is not None and _fair_spread is not None else None

            # Gap vs seasonal
            _near_mo2 = ((_current_month_t7 - 1 + _near - 1) % 12) + 1
            _seas_mean = _s_seas_tbl.get(_near_mo2, {}).get('s_mean')
            _gap_vs_seas = round(_model_s - _seas_mean, 1) if _model_s and _seas_mean else None

            # Trade role
            _trade_roles = {
                1: "Intel only — do not trade",
                2: "Direction — approach with care",
                3: "Primary trading zone",
                4: "Primary trading zone",
                5: "Primary trading zone — wider stops",
                6: "Selective trading",
                7: "Selective trading",
                8: "Selective trading",
                9: "Regime context only",
                10: "Regime context only",
                11: "Regime context only",
            }

            _fwd_rows.append({
                'Pair':             _pair,
                'Model S':          f"{_model_s:.1f}" if _model_s else "—",
                'Source':           _source,
                'Implied S (live)': f"{_s_implied_pair:.1f}" if _s_implied_pair is not None else "—",
                'Actual spread':    f"{_actual_spread:+.1f}" if _actual_spread is not None else "—",
                'Fair spread':      f"{_fair_spread:+.1f}" if _fair_spread is not None else "—",
                'Edge':             _edge,
                'Gap vs seasonal':  _gap_vs_seas,
                'Confidence':       _conf,
                'Trade role':       _trade_roles.get(_near, "—"),
                '_edge_raw':        _edge,
                '_gap_raw':         _gap_vs_seas,
            })

        _df_fwd = pd.DataFrame(_fwd_rows)

        def _style_forward_table(_df):
            _display = _df.drop(columns=['_edge_raw', '_gap_raw'])

            def _colour_edge(val):
                idx = _df.index[_display.index == val.name][0] if val.name in _display.index else None
                if idx is None:
                    return [''] * len(val)
                edge = _df.loc[idx, '_edge_raw']
                styles = [''] * len(val)
                edge_i = _display.columns.get_loc('Edge')
                if edge is not None:
                    if edge < -5:
                        styles[edge_i] = 'background-color: rgba(102,187,106,0.35); color: #c6efce'
                    elif edge > 5:
                        styles[edge_i] = 'background-color: rgba(239,83,80,0.35); color: #ffc7ce'
                    elif edge < -2:
                        styles[edge_i] = 'background-color: rgba(102,187,106,0.15)'
                    elif edge > 2:
                        styles[edge_i] = 'background-color: rgba(239,83,80,0.15)'
                gap = _df.loc[idx, '_gap_raw']
                gap_i = _display.columns.get_loc('Gap vs seasonal')
                if gap is not None:
                    if gap > 5:
                        styles[gap_i] = 'color: #ef5350; font-weight: bold'
                    elif gap < -5:
                        styles[gap_i] = 'color: #66bb6a; font-weight: bold'
                conf = val.iloc[_display.columns.get_loc('Confidence')]
                conf_i = _display.columns.get_loc('Confidence')
                _cmap = {'HIGH': 'color: #66bb6a', 'MED-HIGH': 'color: #aed581',
                         'MEDIUM': 'color: #ffd54f', 'LOW-MED': 'color: #ffb74d',
                         'LOW': 'color: #ff8a65', 'VERY LOW': 'color: #ef5350'}
                styles[conf_i] = _cmap.get(conf, '')
                return styles

            return _display.style.apply(_colour_edge, axis=1)

        st.subheader("Forward S Curve — Fair Value vs Actual")
        st.caption(
            "Model S = best estimate per pair based on data confidence. "
            "Edge = actual spread minus fair spread (BMD: near minus far). "
            "Green edge = spread cheap vs fair → BUY. "
            "Red edge = spread rich vs fair → SELL. "
            "Primary trading zone = M3/M4 through M5/M6."
        )
        st.dataframe(
            _style_forward_table(_df_fwd),
            use_container_width=True, hide_index=True,
        )

        # ENSO context footnote for M6+ rows
        if _enso_label != "Neutral":
            st.caption(f"M6+ context: {_enso_label}. Year ratio vs historical util: {_year_ratio:.2f}x.")

        # Edge bar chart
        _edge_rows = _df_fwd[_df_fwd['_edge_raw'].notna()].copy()

        if not _edge_rows.empty:
            _edge_vals   = _edge_rows['_edge_raw'].tolist()
            _pair_labels = _edge_rows['Pair'].tolist()

            _bar_colors = []
            for _v in _edge_vals:
                if _v < -5:    _bar_colors.append('#66bb6a')
                elif _v > 5:   _bar_colors.append('#ef5350')
                elif _v < -2:  _bar_colors.append('#aed581')
                elif _v > 2:   _bar_colors.append('#ff8a65')
                else:          _bar_colors.append('#888888')

            _fig_edge = go.Figure(go.Bar(
                x=_pair_labels, y=_edge_vals,
                marker_color=_bar_colors,
                text=[f"{v:+.1f}" for v in _edge_vals],
                textposition='outside',
            ))
            _fig_edge.add_hline(y=0, line_color='#ffffff', line_dash='dot', line_width=1)
            _fig_edge.add_hline(y=5, line_color='#ef5350', line_dash='dash', line_width=0.8,
                                annotation_text='sell threshold')
            _fig_edge.add_hline(y=-5, line_color='#66bb6a', line_dash='dash', line_width=0.8,
                                annotation_text='buy threshold')
            _fig_edge.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT, font_color=DARK_TEXT,
                xaxis_gridcolor=DARK_GRID, yaxis_gridcolor=DARK_GRID,
                height=320, yaxis_title='Edge: actual minus fair spread (MYR/t)',
                xaxis_title='Tenor pair',
                margin=dict(l=40, r=40, t=40, b=30), showlegend=False,
            )
            st.plotly_chart(_fig_edge, use_container_width=True)

            # Auto signal summary
            _primary_zone = _edge_rows[_edge_rows['Pair'].isin(['M3/M4', 'M4/M5', 'M5/M6'])]
            if not _primary_zone.empty:
                _buy_pairs  = _primary_zone[_primary_zone['_edge_raw'] < -5]['Pair'].tolist()
                _sell_pairs = _primary_zone[_primary_zone['_edge_raw'] > 5]['Pair'].tolist()
                if _buy_pairs:
                    st.success(f"BUY SPREAD signal in primary trading zone: {', '.join(_buy_pairs)}")
                elif _sell_pairs:
                    st.error(f"SELL SPREAD signal in primary trading zone: {', '.join(_sell_pairs)}")
                else:
                    st.caption("Primary trading zone (M3/M4–M5/M6): "
                               "no strong mispricing signal — within ±5 MYR/t of fair value.")


# ── Tab 8: Spread & Butterfly ─────────────────────────────────────────────────
with tab8:
    st.header("Spread & Butterfly Analysis")
    st.caption(
        "Calendar spread and butterfly mean-reversion signals with fair-value overlay. "
        "spread = F_near − F_far (positive = backwardation). "
        "butterfly = F_mid − 0.5·(F_front + F_back)."
    )

    # ── SharePoint / TT live curve — reuse shared block result ───────────────
    sp_data   = st.session_state.get('sp_data_shared')
    sp_gaps   = compute_gaps(sp_data)
    _sp_m1_ok = sp_data is not None and sp_data["outrights"].get(1) is not None
    if not _sp_m1_ok:
        st.warning(
            "No live prices — paste M1–M12 outrights into **FCPO_Curve_Input.xlsx** "
            "(sheet: *Curve Input*, column C, rows 7–18)."
        )
    else:
        _sp_filled_a = sum(1 for v in sp_data["outrights"].values() if v)
        _sp_filled_b = sum(1 for v in sp_data["spreads"].values() if v)
        _sp_filled_c = sum(1 for v in sp_data["butterflies"].values() if v)
        st.success(
            f"Live prices loaded — Outrights: {_sp_filled_a}/12 · "
            f"Spreads: {_sp_filled_b}/11 · Butterflies: {_sp_filled_c}/10"
        )

    _r8 = st.session_state.get('r_annual', 0.03)
    _contracts_t8 = load_contracts()

    _t8_col1, _t8_col2, _t8_col3 = st.columns(3)
    with _t8_col1:
        _near_off  = st.number_input("Near offset (M)", min_value=1, max_value=11, value=1, key="t8_near")
        _far_off   = st.number_input("Far offset (M)",  min_value=2, max_value=12, value=2, key="t8_far")
    with _t8_col2:
        _fr_off    = st.number_input("Front offset (M)", min_value=1, max_value=10, value=1, key="t8_fr")
        _mid_off   = st.number_input("Mid offset (M)",   min_value=2, max_value=11, value=2, key="t8_mid")
        _bk_off    = st.number_input("Back offset (M)",  min_value=3, max_value=12, value=3, key="t8_bk")
    with _t8_col3:
        _lookback8 = st.slider("Lookback (days)", min_value=30, max_value=365, value=180, key="t8_lb")
        _s_input_t8 = st.number_input(
            "S assumption (MYR/t) — for fair value",
            min_value=5.0, max_value=50.0, value=12.0, step=0.5, key="t8_s_input",
        )

    # ── Get current curve — reuse shared block result ─────────────────────────
    _today_t8 = _today_sh
    _curve_t8 = st.session_state.get('current_curve') or get_active_curve(_contracts_t8, _today_t8)
    F_M1 = float(_curve_t8.get(1) or 4000.0)
    F_M2 = float(_curve_t8.get(2) or 4000.0)

    # S values from shared session state
    s_implied_live = st.session_state.get('s_implied_live')
    s_implied_avg  = st.session_state.get('s_implied_avg')
    s_mpob         = st.session_state.get('s_mpob', _s_input_t8)
    s_producer     = st.session_state.get('s_producer_current', _s_input_t8)
    s_forward      = st.session_state.get('s_producer_forward',  _s_input_t8)

    # For fair value calc use live S implied if available, else MPOB
    _s_for_fv = s_implied_live if s_implied_live is not None else s_mpob
    c_current = implied_c(F_M1, F_M2, _s_input_t8, r_annual=_r8)
    _fv_current = fair_spread_value(F_M1, _r8, _s_input_t8, c_current)

    # ── Current curve metrics ─────────────────────────────────────────────────
    _c8r1, _c8r2, _c8r3, _c8r4 = st.columns(4)
    _c8r1.metric("M1 price",        f"MYR {F_M1:,.0f}")
    _c8r2.metric("M2 price",        f"MYR {F_M2:,.0f}")
    with _c8r3:
        st.metric(
            "S Implied (live)",
            f"{s_implied_live:.1f} MYR/t" if s_implied_live is not None else "No Excel data",
            help="Real-time: back-solved from current prices in FCPO_Curve_Input.xlsx",
        )
        if s_implied_avg is not None:
            st.caption(f"Monthly avg (CSV): {s_implied_avg:.1f} MYR/t")
    _c8r4.metric("c (conv. yield)",  f"{c_current:.1%}")

    _c8r5, _c8r6, _c8r7, _c8r8 = st.columns(4)
    _c8r5.metric("Fair spread (M1/M2)", f"{_fv_current:.1f} MYR/t")
    _c8r6.metric("Actual spread",       f"{F_M1 - F_M2:.1f} MYR/t")
    _c8r7.metric("S MPOB",              f"{s_mpob:.1f} MYR/t")
    _c8r8.metric("S Producer",          f"{s_producer:.1f} MYR/t")

    # Three-source gap analysis — use live S if available, else avg
    _s_implied_for_gaps = s_implied_live if s_implied_live is not None else (s_implied_avg or s_mpob)
    _gaps = three_source_gaps(_s_implied_for_gaps, s_mpob, s_producer)
    with st.expander("Three-Source Gap Analysis", expanded=True):
        _gc1, _gc2, _gc3 = st.columns(3)
        _gc1.metric("Gap1: Implied − MPOB",    f"{_gaps['gap1']:+.1f}", help="Market vs MPOB")
        _gc2.metric("Gap2: MPOB − Producer",   f"{_gaps['gap2']:+.1f}", help="MPOB vs physical")
        _gc3.metric("Gap3: Implied − Producer", f"{_gaps['gap3']:+.1f}", help="Market vs physical (key)")

        _sig_col = {
            'SELL SPREAD': 'error', 'BUY SPREAD': 'success', 'NEUTRAL': 'info'
        }
        getattr(st, _sig_col.get(_gaps['gap3_signal'], 'info'))(
            f"**{_gaps['gap3_signal']}** — {_gaps['story']}"
        )

    st.markdown("---")

    # ── Implied S Term Structure ───────────────────────────────────────────────
    st.subheader("Implied S Term Structure")
    st.caption(
        "Implied storage cost back-solved from each consecutive tenor pair, "
        "overlaid against the MPOB seasonal average. "
        "Red markers = pair pricing above seasonal norm (market sees tightness). "
        "Green markers = pair pricing below seasonal norm (market sees looseness)."
    )

    try:
        _seas_tbl_t8 = _s_model_t8['seasonal_table']
    except Exception:
        _seas_tbl_t8 = {}

    _implied_s_curve  = {}
    _seasonal_s_curve = {}

    for _i in range(1, 12):
        _near_p = _curve_t8.get(_i)
        _far_p  = _curve_t8.get(_i + 1)
        if _near_p and _far_p and _near_p > 0 and _far_p > 0:
            _res = implied_s_backsolve(_near_p, _far_p, _r8)
            _implied_s_curve[f"M{_i}/M{_i+1}"] = _res['s_implied_myr']

        _near_month = ((_today_t8.month - 1 + _i - 1) % 12) + 1
        _far_month  = ((_today_t8.month - 1 + _i)     % 12) + 1
        _near_seas  = _seas_tbl_t8.get(_near_month, {}).get('s_mean', 15.0)
        _far_seas   = _seas_tbl_t8.get(_far_month,  {}).get('s_mean', 15.0)
        _seasonal_s_curve[f"M{_i}/M{_i+1}"] = (_near_seas + _far_seas) / 2

    if _implied_s_curve:
        _st_pairs        = list(_implied_s_curve.keys())
        _st_implied_vals = [_implied_s_curve[p] for p in _st_pairs]
        _st_seas_vals    = [_seasonal_s_curve.get(p, 15.0) for p in _st_pairs]
        _st_gap_vals     = [_implied_s_curve[p] - _seasonal_s_curve.get(p, 15.0) for p in _st_pairs]

        _fig_s_term = go.Figure()

        # Shaded gap between implied and seasonal
        _fig_s_term.add_trace(go.Scatter(
            x=_st_pairs + _st_pairs[::-1],
            y=_st_implied_vals + _st_seas_vals[::-1],
            fill='toself', fillcolor='rgba(239,83,80,0.12)',
            line=dict(width=0), name='Gap vs seasonal', showlegend=True,
        ))
        # Seasonal baseline
        _fig_s_term.add_trace(go.Scatter(
            x=_st_pairs, y=_st_seas_vals,
            line=dict(color='#888888', width=1.5, dash='dot'),
            name='Seasonal average S',
        ))
        # Current implied S curve
        _fig_s_term.add_trace(go.Scatter(
            x=_st_pairs, y=_st_implied_vals,
            line=dict(color='#ff6b35', width=2.5),
            mode='lines+markers', marker=dict(size=8),
            name='Current implied S',
        ))
        # Colour each marker by gap direction
        _st_marker_colors = [
            '#ef5350' if g > 2 else '#66bb6a' if g < -2 else '#888888'
            for g in _st_gap_vals
        ]
        _fig_s_term.data[-1].marker.color = _st_marker_colors

        _fig_s_term.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
            font_color=DARK_TEXT,
            xaxis=dict(gridcolor=DARK_GRID),
            yaxis=dict(gridcolor=DARK_GRID, title='Implied S (MYR/t)'),
            height=300, margin=dict(l=40, r=20, t=30, b=30),
            legend=dict(orientation='h', y=1.02, font=dict(color=DARK_TEXT)),
        )
        st.plotly_chart(_fig_s_term, use_container_width=True)

        # Gap bar
        _fig_st_gap = go.Figure(go.Bar(
            x=_st_pairs, y=_st_gap_vals,
            marker_color=['#ef5350' if g > 0 else '#66bb6a' for g in _st_gap_vals],
            name='Implied vs seasonal gap',
        ))
        _fig_st_gap.add_hline(y=0, line_color='#888888', line_dash='dot')
        _fig_st_gap.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
            font_color=DARK_TEXT,
            xaxis=dict(gridcolor=DARK_GRID),
            yaxis=dict(gridcolor=DARK_GRID, title='Gap (MYR/t)'),
            height=180, margin=dict(l=40, r=20, t=10, b=30),
        )
        st.plotly_chart(_fig_st_gap, use_container_width=True)

        # Auto interpretation
        _max_gap_pair = _st_pairs[_st_gap_vals.index(max(_st_gap_vals))]
        _min_gap_pair = _st_pairs[_st_gap_vals.index(min(_st_gap_vals))]
        st.caption(
            f"Most expensive vs seasonal: {_max_gap_pair} "
            f"(+{max(_st_gap_vals):.1f} MYR/t above norm) — "
            f"market pricing extra tightness here.  "
            f"Most discounted: {_min_gap_pair} "
            f"({min(_st_gap_vals):.1f} MYR/t below norm)."
        )
    else:
        st.info("No consecutive pair prices available — fill in outrights in FCPO_Curve_Input.xlsx.")

    st.markdown("---")

    # ── Spread history ────────────────────────────────────────────────────────
    st.subheader(f"Spread M{int(_near_off)}/M{int(_far_off)} — {_lookback8}d history")

    try:
        _spd_hist = build_spread_history(
            _contracts_t8, int(_near_off), int(_far_off), lookback_days=_lookback8
        )
    except Exception as _ex:
        _spd_hist = pd.DataFrame()
        st.warning(f"Could not build spread history: {_ex}")

    if not _spd_hist.empty:
        # Z-score from latest row
        _last_spd  = _spd_hist.iloc[-1]
        _z20_spd   = float(_last_spd.get('z20', 0) or 0)
        _z45_spd   = float(_last_spd.get('z45', 0) or 0)
        _z90_spd   = float(_last_spd.get('z90', 0) or 0)
        _spd_now   = float(_last_spd.get('spread', 0) or 0)
        _ma20_spd  = float(_last_spd.get('ma20', _spd_now) or _spd_now)

        _z_vs_fv = (_spd_now - _fv_current) / float(_spd_hist['spread'].std() or 1)

        _conv = conviction_score(_z20_spd, _z_vs_fv, c_current, _z20_spd, _z45_spd)

        _sc1, _sc2, _sc3, _sc4, _sc5 = st.columns(5)
        _sc1.metric("Spread now",   f"{_spd_now:.1f}")
        _sc2.metric("Z (20d)",      f"{_z20_spd:.2f}")
        _sc3.metric("Z (45d)",      f"{_z45_spd:.2f}")
        _sc4.metric("Z (90d)",      f"{_z90_spd:.2f}")
        _sc5.metric("Conv. score",  f"{_conv['score']}/7")

        _conv_dir_colour = 'success' if _conv['direction'] == 'buy' else ('error' if _conv['direction'] == 'sell' else 'info')
        getattr(st, _conv_dir_colour)(
            f"**Direction: {_conv['direction'].upper()}** — Size: {_conv['size']}"
        )

        # Entry conditions checklist
        with st.expander("Entry Conditions Checklist"):
            _checklist = entry_conditions_checklist(
                _z20_spd, _z45_spd, _spd_now, _fv_current, c_current, _conv['direction']
            )
            for _chk in _checklist:
                _icon = "✅" if _chk['met'] else ("⚠️" if _chk['warning'] else "❌")
                st.markdown(f"{_icon} **{_chk['rule']}** — {_chk['condition']}")

        # Spread chart
        _fig_spd8 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.65, 0.35], vertical_spacing=0.04)

        _fig_spd8.add_trace(go.Scatter(
            x=_spd_hist['date'], y=_spd_hist['spread'],
            name='Spread', line=dict(color='#00b4d8', width=1.5),
            hovertemplate='%{x|%d %b %Y}<br>Spread: %{y:.1f}<extra></extra>',
        ), row=1, col=1)
        _fig_spd8.add_trace(go.Scatter(
            x=_spd_hist['date'], y=_spd_hist['ma20'],
            name='MA20', line=dict(color='#ff7f0e', width=1, dash='dash'),
        ), row=1, col=1)
        _fig_spd8.add_trace(go.Scatter(
            x=_spd_hist['date'], y=_spd_hist['ma45'],
            name='MA45', line=dict(color='#2ca02c', width=1, dash='dot'),
        ), row=1, col=1)

        # Fair value line
        _fig_spd8.add_hline(y=_fv_current, line=dict(color='#d62728', width=1, dash='dash'),
                             annotation_text=f"FV {_fv_current:.1f}", row=1, col=1)

        # Z-score
        _fig_spd8.add_trace(go.Scatter(
            x=_spd_hist['date'], y=_spd_hist['z20'],
            name='Z20', line=dict(color='#ff7f0e', width=1.5),
        ), row=2, col=1)
        _fig_spd8.add_hline(y=2.0,  line=dict(color='#ff4444', dash='dot', width=1), row=2, col=1)
        _fig_spd8.add_hline(y=-2.0, line=dict(color='#ff4444', dash='dot', width=1), row=2, col=1)
        _fig_spd8.add_hline(y=0,    line=dict(color='#888888', dash='dot', width=1), row=2, col=1)

        _fig_spd8.update_layout(
            plot_bgcolor=DARK_PLOT, paper_bgcolor=DARK_BG,
            font=dict(color=DARK_TEXT),
            height=480, margin=dict(l=60, r=30, t=30, b=50),
            legend=dict(font=dict(color=DARK_TEXT)),
        )
        _fig_spd8.update_xaxes(gridcolor=DARK_GRID)
        _fig_spd8.update_yaxes(gridcolor=DARK_GRID)
        _fig_spd8.update_yaxes(title_text='Spread (MYR)', row=1, col=1)
        _fig_spd8.update_yaxes(title_text='Z-score', row=2, col=1)
        st.plotly_chart(_fig_spd8, use_container_width=True)

    st.markdown("---")

    # ── Butterfly ─────────────────────────────────────────────────────────────
    st.subheader(f"Butterfly M{int(_fr_off)}/M{int(_mid_off)}/M{int(_bk_off)} — {_lookback8}d history")

    try:
        _bfly_hist = build_butterfly_history(
            _contracts_t8, int(_fr_off), int(_mid_off), int(_bk_off), lookback_days=_lookback8
        )
    except Exception as _ex2:
        _bfly_hist = pd.DataFrame()
        st.warning(f"Could not build butterfly history: {_ex2}")

    if not _bfly_hist.empty:
        _last_bfly = _bfly_hist.iloc[-1]
        _z20_bfly  = float(_last_bfly.get('z20', 0) or 0)
        _z45_bfly  = float(_last_bfly.get('z45', 0) or 0)
        _bfly_now  = float(_last_bfly.get('butterfly', 0) or 0)

        # Scenario interpretation
        _interp = scenario_interpretation(_z20_spd if not _spd_hist.empty else 0.0, _z20_bfly)

        _bc1, _bc2, _bc3, _bc4 = st.columns(4)
        _bc1.metric("Butterfly now", f"{_bfly_now:.1f}")
        _bc2.metric("Z (20d)",       f"{_z20_bfly:.2f}")
        _bc3.metric("Z (45d)",       f"{_z45_bfly:.2f}")
        _bc4.metric("MA20",          f"{float(_last_bfly.get('ma20', _bfly_now) or _bfly_now):.1f}")

        st.info(_interp)

        _fig_bfly = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.65, 0.35], vertical_spacing=0.04)
        _fig_bfly.add_trace(go.Scatter(
            x=_bfly_hist['date'], y=_bfly_hist['butterfly'],
            name='Butterfly', line=dict(color='#9467bd', width=1.5),
            hovertemplate='%{x|%d %b %Y}<br>Butterfly: %{y:.1f}<extra></extra>',
        ), row=1, col=1)
        _fig_bfly.add_trace(go.Scatter(
            x=_bfly_hist['date'], y=_bfly_hist['ma20'],
            name='MA20', line=dict(color='#ff7f0e', width=1, dash='dash'),
        ), row=1, col=1)
        _fig_bfly.add_hline(y=0, line=dict(color='#888888', dash='dot', width=1), row=1, col=1)

        _fig_bfly.add_trace(go.Scatter(
            x=_bfly_hist['date'], y=_bfly_hist['z20'],
            name='Z20', line=dict(color='#ff7f0e', width=1.5),
        ), row=2, col=1)
        _fig_bfly.add_hline(y=2.0,  line=dict(color='#ff4444', dash='dot', width=1), row=2, col=1)
        _fig_bfly.add_hline(y=-2.0, line=dict(color='#ff4444', dash='dot', width=1), row=2, col=1)
        _fig_bfly.add_hline(y=0,    line=dict(color='#888888', dash='dot', width=1), row=2, col=1)

        _fig_bfly.update_layout(
            plot_bgcolor=DARK_PLOT, paper_bgcolor=DARK_BG,
            font=dict(color=DARK_TEXT),
            height=460, margin=dict(l=60, r=30, t=30, b=50),
            legend=dict(font=dict(color=DARK_TEXT)),
        )
        _fig_bfly.update_xaxes(gridcolor=DARK_GRID)
        _fig_bfly.update_yaxes(gridcolor=DARK_GRID)
        _fig_bfly.update_yaxes(title_text='Butterfly (MYR)', row=1, col=1)
        _fig_bfly.update_yaxes(title_text='Z-score', row=2, col=1)
        st.plotly_chart(_fig_bfly, use_container_width=True)

    # ── Panel 5: TT Listed Contract Gaps (SharePoint) ─────────────────────────
    st.markdown("---")
    st.subheader("Panel 5 — TT Listed Contract Gaps")
    if sp_gaps:
        _lut = get_last_update_time(_tt_xlsx_sh)
        if _lut:
            st.caption(f"File last updated: {_lut.strftime('%d %b %Y %H:%M')}")

        _p5_left, _p5_right = st.columns(2)

        with _p5_left:
            st.markdown("**Calendar Spread Gaps**")
            _sg_rows = []
            for (n, f), v in sp_gaps['spread_gaps'].items():
                _sg_rows.append({
                    'Pair':   f"M{n}/M{f}",
                    'Listed': f"{v['listed']:+.1f}",
                    'Calc':   f"{v['calculated']:+.1f}",
                    'Gap':    f"{v['gap']:+.1f}",
                    'Signal': v['signal'],
                })
            if _sg_rows:
                _sg_df = pd.DataFrame(_sg_rows)
                def _style_spread_gap(val):
                    if val == 'RICH':  return 'background-color: #4a1a1a; color: #ff7f7f'
                    if val == 'CHEAP': return 'background-color: #1a4a1a; color: #7fff7f'
                    return ''
                st.dataframe(
                    _sg_df.style.applymap(_style_spread_gap, subset=['Signal']),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No spread gaps — fill listed spread prices in column C rows 24–34.")

        with _p5_right:
            st.markdown("**Butterfly Gaps**")
            _bg_rows = []
            for (fr, mi, bk), v in sp_gaps['butterfly_gaps'].items():
                _bg_rows.append({
                    'Butterfly': f"M{fr}/M{mi}/M{bk}",
                    'Listed':    f"{v['listed']:+.1f}",
                    'Calc':      f"{v['calculated']:+.1f}",
                    'Gap':       f"{v['gap']:+.1f}",
                    'Signal':    v['signal'],
                })
            if _bg_rows:
                _bg_df = pd.DataFrame(_bg_rows)
                def _style_bfly_gap(val):
                    if val == 'RICH':  return 'background-color: #4a1a1a; color: #ff7f7f'
                    if val == 'CHEAP': return 'background-color: #1a4a1a; color: #7fff7f'
                    return ''
                st.dataframe(
                    _bg_df.style.applymap(_style_bfly_gap, subset=['Signal']),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No butterfly gaps — fill listed butterfly prices in column C rows 39–48.")
    else:
        st.info("Paste outright prices (M1–M12) into FCPO_Curve_Input.xlsx to enable gap analysis.")

# ═══════════════════════════════════════════════════════════════
# TAB 9 — El Niño & La Niña Analysis
# ═══════════════════════════════════════════════════════════════
with tab_enso:
    st.title("El Niño & La Niña Analysis")
    st.caption(
        "Climate context for CPO back month spreads (M6+). "
        "El Niño → drier SE Asia → production risk 9-12 months later. "
        "La Niña → wetter → production builds → spread pressure. "
        "For directional bias on back months only — not front month trading."
    )

    oni_df  = _cached_oni_history()
    fc_df   = _cached_enso_forecast()

    if oni_df is None or oni_df.empty:
        st.error("ONI history file not found. Check Raw Data/ENSO/oni.ascii.txt")
        st.stop()

    # ── PANEL 1: Current state summary ──────────────────────
    st.subheader("Current ENSO State")

    latest          = oni_df.iloc[-1]
    latest_anom     = latest["oni_anomaly"]
    latest_phase    = latest["enso_phase"]
    latest_strength = latest["strength"]
    latest_season   = f"{latest['season']} {int(latest['year'])}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest ONI", f"{latest_anom:+.2f}°C",
                help=f"Season: {latest_season}")
    col2.metric("Current Phase", latest_phase,
                delta=latest_strength if latest_strength != "Neutral" else None)

    peak_el_pct = 0
    peak_season = ""
    if fc_df is not None and not fc_df.empty:
        latest_fc    = fc_df[fc_df["issue_date"] == fc_df["issue_date"].max()]
        peak_el_row  = latest_fc.loc[latest_fc["el_nino_pct"].idxmax()]
        peak_el_pct  = peak_el_row["el_nino_pct"]
        peak_season  = peak_el_row["season"]
        last_updated = fc_df["issue_date"].max()
        col3.metric("Peak El Niño Probability",
                    f"{peak_el_pct:.0f}%",
                    help=f"Peaks at {peak_season}")
        col4.metric("Forecast Last Updated",
                    pd.to_datetime(last_updated).strftime("%d %b %Y"))
    else:
        col3.metric("Peak El Niño Probability", "—")
        col4.metric("Forecast Last Updated", "No forecast loaded")

    # Alert banner
    if peak_el_pct >= 70:
        st.error(
            f"⚠️ **El Niño developing — {peak_el_pct:.0f}% probability at peak ({peak_season})**  ·  "
            f"Historical pattern: CPO production typically -5% to -18% below seasonal average "
            f"approximately 9-12 months after El Niño onset  ·  "
            f"Back month spreads M6+ for Q4 2026 – Q2 2027 delivery may be underpriced"
        )
    elif fc_df is not None and not fc_df.empty:
        peak_ln = fc_df[fc_df["issue_date"] == fc_df["issue_date"].max()]["la_nina_pct"].max()
        if peak_ln >= 70:
            st.info(
                f"ℹ️ **La Niña likely — {peak_ln:.0f}% probability**  ·  "
                f"Historical pattern: production above seasonal average  ·  "
                f"Back month spread pressure as stocks build"
            )
        else:
            st.warning(
                "Neutral conditions most likely — no strong ENSO-driven bias for back months. "
                "Monitor monthly ONI updates for phase change."
            )

    st.markdown("---")

    # ── PANEL 2: Forecast probability chart + evolution ─────
    st.subheader("ENSO Forecast — Probability & Evolution")

    if fc_df is not None and not fc_df.empty:
        latest_fc = fc_df[fc_df["issue_date"] == fc_df["issue_date"].max()].copy()
        last_upd  = fc_df["issue_date"].max()

        days_since = (pd.Timestamp.now() - last_upd).days
        if days_since > 35:
            st.warning(
                f"⚠️ Forecast is {days_since} days old — update ENSO_Data_Template.xlsx "
                f"with the latest IRI forecast and append new rows."
            )
        else:
            st.caption(
                f"Source: IRI Columbia University  ·  "
                f"Issued: {last_upd.strftime('%B %Y')}  ·  "
                f"Update monthly at iri.columbia.edu/our-expertise/climate/forecasts/"
            )

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Bar(
                name="La Niña", x=latest_fc["season"],
                y=latest_fc["la_nina_pct"], marker_color="#4fc3f7",
                text=latest_fc["la_nina_pct"].apply(lambda x: f"{x:.0f}%" if x >= 5 else ""),
                textposition="inside", textfont=dict(color="white", size=9),
            ))
            fig_fc.add_trace(go.Bar(
                name="Neutral", x=latest_fc["season"],
                y=latest_fc["neutral_pct"], marker_color="#888888",
                text=latest_fc["neutral_pct"].apply(lambda x: f"{x:.0f}%" if x >= 5 else ""),
                textposition="inside", textfont=dict(color="white", size=9),
            ))
            fig_fc.add_trace(go.Bar(
                name="El Niño", x=latest_fc["season"],
                y=latest_fc["el_nino_pct"], marker_color="#ef5350",
                text=latest_fc["el_nino_pct"].apply(lambda x: f"{x:.0f}%" if x >= 5 else ""),
                textposition="inside", textfont=dict(color="white", size=9),
            ))
            fig_fc.update_layout(
                barmode="stack",
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
                font_color=DARK_TEXT, height=280,
                yaxis=dict(title="Probability (%)", range=[0, 100]),
                margin=dict(l=40, r=20, t=20, b=30),
                legend=dict(orientation="h", y=1.05),
                xaxis_gridcolor=DARK_GRID,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        with col_table:
            st.caption("Forecast evolution — El Niño % by issue month")
            pivot = fc_df.pivot_table(
                index="season",
                columns="issue_date",
                values="el_nino_pct",
                aggfunc="first",
            )
            sm_map = fc_df[["season", "season_start_month"]].drop_duplicates()
            sm_map = sm_map.sort_values("season_start_month")
            ordered = [s for s in sm_map["season"].tolist() if s in pivot.index]
            pivot = pivot.reindex(ordered)
            pivot.columns = [pd.to_datetime(c).strftime("%b %Y")
                             for c in pivot.columns]

            def _enso_trend(row):
                vals = row.dropna().values
                if len(vals) < 2:
                    return "—"
                d = vals[-1] - vals[0]
                if d >= 10:   return f"↑↑ +{d:.0f}%"
                if d >= 3:    return f"↑ +{d:.0f}%"
                if d <= -10:  return f"↓↓ {d:.0f}%"
                if d <= -3:   return f"↓ {d:.0f}%"
                return "→ stable"

            pivot["Trend"] = pivot.apply(_enso_trend, axis=1)
            display_piv = pivot.copy()
            date_cols = [c for c in display_piv.columns if c != "Trend"]
            for col in date_cols:
                display_piv[col] = display_piv[col].apply(
                    lambda x: f"{x:.0f}%" if pd.notna(x) else "—")
            display_piv.index.name = "Season"
            display_piv = display_piv.reset_index()

            def _colour_pct(val):
                if val == "—":
                    return "color:#666666"
                try:
                    p = float(val.replace("%", "").strip())
                    if p >= 80:
                        return "background-color:rgba(183,28,28,0.4);color:white;font-weight:bold"
                    if p >= 60:
                        return "background-color:rgba(239,83,80,0.3);font-weight:bold"
                    if p >= 40:
                        return "background-color:rgba(239,83,80,0.15)"
                    return ""
                except Exception:
                    return ""

            def _colour_trend(val):
                if "↑" in str(val):
                    return "color:#ef5350;font-weight:bold"
                if "↓" in str(val):
                    return "color:#66bb6a;font-weight:bold"
                return "color:#888888"

            styled_piv = display_piv.style
            for col in date_cols:
                styled_piv = styled_piv.map(_colour_pct, subset=[col])
            styled_piv = styled_piv.map(_colour_trend, subset=["Trend"])
            st.dataframe(styled_piv, use_container_width=True, hide_index=True)
            st.caption(
                "Red = high El Niño probability. "
                "↑↑ = confidence rising fast. → = stable."
            )
    else:
        st.warning(
            "No forecast data. Add ENSO_Data_Template.xlsx to Raw Data/ENSO/ "
            "and populate the ENSO Forecast sheet."
        )

    st.markdown("---")

    # ── PANEL 3: ONI history bar chart ───────────────────────
    st.subheader("ONI History — Confirmed ENSO Phases")
    st.caption(
        "Historical ONI anomaly from 1950. "
        "Red = El Niño. Blue = La Niña. Grey = Neutral. "
        "Dashed lines = ±0.5°C phase thresholds."
    )

    recent_oni = oni_df[oni_df["year"] >= datetime.date.today().year - 10].copy()

    def _oni_color(row):
        if row["enso_phase"] == "El Nino":
            return {"Strong": "#b71c1c", "Moderate": "#ef5350", "Weak": "#ffcdd2"}.get(row["strength"], "#ef5350")
        if row["enso_phase"] == "La Nina":
            return {"Strong": "#0d47a1", "Moderate": "#4fc3f7", "Weak": "#b3e5fc"}.get(row["strength"], "#4fc3f7")
        return "#888888"

    bar_colors = [_oni_color(r) for _, r in recent_oni.iterrows()]

    fig_oni = go.Figure()
    fig_oni.add_trace(go.Bar(
        x=recent_oni["date"], y=recent_oni["oni_anomaly"],
        marker_color=bar_colors,
        hovertemplate="%{x|%b %Y}<br>ONI: %{y:+.2f}°C<extra></extra>",
    ))
    fig_oni.add_hline(y=0.5, line_color="#ef5350", line_dash="dash",
                      line_width=1, annotation_text="El Niño (+0.5)",
                      annotation_position="right")
    fig_oni.add_hline(y=-0.5, line_color="#4fc3f7", line_dash="dash",
                      line_width=1, annotation_text="La Niña (−0.5)",
                      annotation_position="right")
    fig_oni.add_hline(y=0, line_color="#ffffff", line_dash="dot", line_width=0.5)
    fig_oni.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
        font_color=DARK_TEXT, height=280,
        yaxis_title="ONI Anomaly (°C)",
        margin=dict(l=40, r=100, t=20, b=30),
        showlegend=False,
        xaxis_gridcolor=DARK_GRID, yaxis_gridcolor=DARK_GRID,
    )
    st.plotly_chart(fig_oni, use_container_width=True)

    with st.expander("Show full ONI history 1950–present"):
        bar_colors_all = [_oni_color(r) for _, r in oni_df.iterrows()]
        fig_oni_all = go.Figure()
        fig_oni_all.add_trace(go.Bar(
            x=oni_df["date"], y=oni_df["oni_anomaly"],
            marker_color=bar_colors_all,
            hovertemplate="%{x|%b %Y}<br>ONI: %{y:+.2f}°C<extra></extra>",
        ))
        fig_oni_all.add_hline(y=0.5, line_color="#ef5350", line_dash="dash", line_width=1)
        fig_oni_all.add_hline(y=-0.5, line_color="#4fc3f7", line_dash="dash", line_width=1)
        fig_oni_all.add_hline(y=0, line_color="#ffffff", line_dash="dot", line_width=0.5)
        fig_oni_all.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
            font_color=DARK_TEXT, height=280,
            yaxis_title="ONI Anomaly (°C)",
            margin=dict(l=40, r=40, t=20, b=30),
            showlegend=False,
            xaxis_gridcolor=DARK_GRID, yaxis_gridcolor=DARK_GRID,
        )
        st.plotly_chart(fig_oni_all, use_container_width=True)

    st.markdown("---")

    # ── PANEL 4: Historical pattern by ENSO phase ────────────
    def _assign_enso_to_mpob(mpob_in, oni_in):
        """
        Assigns ENSO phase to each MPOB monthly row.
        Uses the ONI season where the MPOB month is the MIDDLE month
        of the 3-month window — most representative assignment.
        """
        oni_tmp = oni_in.copy()
        oni_tmp["middle_month"] = oni_tmp["season_start_month"].apply(
            lambda m: (m % 12) + 1
        )
        oni_lookup = oni_tmp.set_index(["year", "middle_month"])[
            ["enso_phase", "strength", "oni_anomaly"]
        ].to_dict("index")

        result = mpob_in.copy()
        result["enso_phase"] = result.apply(
            lambda r: oni_lookup.get((r["year"], r["month"]), {}).get("enso_phase", "Unknown"), axis=1)
        result["strength"] = result.apply(
            lambda r: oni_lookup.get((r["year"], r["month"]), {}).get("strength", "Unknown"), axis=1)
        result["oni_anomaly"] = result.apply(
            lambda r: oni_lookup.get((r["year"], r["month"]), {}).get("oni_anomaly", None), axis=1)
        return result[result["enso_phase"] != "Unknown"]

    def _build_phase_bands(df_in, value_col, oni_in, cy):
        """
        Computes per-phase per-month: mean, min, max, std.
        Returns dict {phase: DataFrame} and the enso-tagged dataframe.
        """
        df_enso = _assign_enso_to_mpob(
            df_in.rename(columns={value_col: "mpob_stocks"}), oni_in
        )
        df_enso[value_col] = df_enso["mpob_stocks"]

        mn = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        result = {}
        for phase in ["El Nino", "La Nina", "Neutral"]:
            subset = df_enso[
                (df_enso["enso_phase"] == phase) &
                (df_enso["year"] < cy)
            ]
            if subset.empty:
                continue
            agg = subset.groupby("month")[value_col].agg(
                mean="mean", min="min", max="max",
                std="std", count="count",
            ).reset_index()
            agg = agg[agg["count"] >= 2]
            agg["month_name"] = agg["month"].apply(lambda x: mn[x - 1])
            result[phase] = agg
        return result, df_enso

    # Shared setup
    month_names  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    current_year = datetime.date.today().year
    phase_colors = {
        "El Nino": "#ef5350",
        "La Nina": "#4fc3f7",
        "Neutral": "#888888",
    }
    phase_labels = {
        "El Nino": "El Niño",
        "La Nina": "La Niña",
        "Neutral": "Neutral",
    }

    # Load all data
    mpob_raw = load_mpob_data().copy()
    mpob_raw["year"]  = pd.to_datetime(mpob_raw["date"]).dt.year
    mpob_raw["month"] = pd.to_datetime(mpob_raw["date"]).dt.month

    try:
        sd_df = load_supply_demand()
        sd_df["year"]  = pd.to_datetime(sd_df["Date"]).dt.year
        sd_df["month"] = pd.to_datetime(sd_df["Date"]).dt.month
        has_production = "Production" in sd_df.columns
    except Exception:
        has_production = False

    # Build monthly average price from existing spot data
    price_monthly = df.copy()
    price_monthly["month"] = price_monthly["date"].dt.month
    price_monthly = price_monthly.groupby(["year", "month"])["close"].mean().reset_index()
    price_monthly.columns = ["year", "month", "fcpo_price"]
    price_monthly["date"] = price_monthly.apply(
        lambda r: pd.Timestamp(year=int(r["year"]),
                               month=int(r["month"]), day=1), axis=1)
    has_price = not price_monthly.empty

    # Toggle
    st.subheader("Historical Pattern by ENSO Phase")
    toggle = st.radio(
        "Show phase:",
        options=["Both", "El Niño only", "La Niña only"],
        horizontal=True,
        key="enso_toggle",
    )
    show_el = toggle in ["Both", "El Niño only"]
    show_ln = toggle in ["Both", "La Niña only"]

    phases_to_show = []
    if show_el:
        phases_to_show.append("El Nino")
    if show_ln:
        phases_to_show.append("La Nina")

    # ── CHART 1: MPOB Stocks ─────────────────────────────────
    st.caption(
        "**MPOB Closing Stocks** — shaded band = historical min/max · "
        "solid line = phase average · orange = 2026 actual"
    )

    stocks_bands, stocks_enso = _build_phase_bands(
        mpob_raw, "mpob_stocks", oni_df, current_year)

    fig_stocks = go.Figure()

    for phase in phases_to_show:
        if phase not in stocks_bands:
            continue
        agg   = stocks_bands[phase].sort_values("month")
        color = phase_colors[phase]
        label = phase_labels[phase]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        fig_stocks.add_trace(go.Scatter(
            x=agg["month_name"].tolist() + agg["month_name"].tolist()[::-1],
            y=(agg["max"] / 1e6).tolist() + (agg["min"] / 1e6).tolist()[::-1],
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            line=dict(width=0),
            name=f"{label} range",
            showlegend=True,
        ))
        fig_stocks.add_trace(go.Scatter(
            x=agg["month_name"], y=agg["mean"] / 1e6,
            name=f"{label} avg",
            line=dict(color=color, width=2),
            mode="lines+markers", marker=dict(size=5),
        ))

    if "Neutral" in stocks_bands:
        neu = stocks_bands["Neutral"].sort_values("month")
        fig_stocks.add_trace(go.Scatter(
            x=neu["month_name"], y=neu["mean"] / 1e6,
            name="Neutral avg",
            line=dict(color="#888888", width=1.2, dash="dot"),
            mode="lines", opacity=0.6,
        ))

    cy_stocks = stocks_enso[stocks_enso["year"] == current_year].sort_values("month")
    if not cy_stocks.empty:
        fig_stocks.add_trace(go.Scatter(
            x=cy_stocks["month"].apply(lambda m: month_names[m - 1]),
            y=cy_stocks["mpob_stocks"] / 1e6,
            name=f"{current_year} actual",
            line=dict(color="#ff6b35", width=2.5),
            mode="lines+markers",
            marker=dict(size=9, symbol="star"),
        ))

    fig_stocks.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
        font_color=DARK_TEXT, height=320,
        yaxis_title="Million tonnes",
        xaxis=dict(categoryorder="array", categoryarray=month_names),
        xaxis_gridcolor=DARK_GRID, yaxis_gridcolor=DARK_GRID,
        margin=dict(l=40, r=20, t=20, b=30),
        legend=dict(orientation="h", y=1.05, font=dict(size=9)),
    )
    st.plotly_chart(fig_stocks, use_container_width=True)

    # ── % vs Neutral metric cards ─────────────────────────────
    if "Neutral" in stocks_bands:
        neu_avg = stocks_bands["Neutral"].set_index("month")["mean"]
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        for phase, col in [("El Nino", col_m1), ("La Nina", col_m2)]:
            if phase in stocks_bands:
                ph_avg = stocks_bands[phase].set_index("month")["mean"]
                common = neu_avg.index.intersection(ph_avg.index)
                if len(common) > 0:
                    pct_diff = ((ph_avg[common] - neu_avg[common]) /
                                neu_avg[common] * 100).mean()
                    implication = ("→ tighter supply, bullish spreads"
                                   if pct_diff < 0 else
                                   "→ looser supply, bearish spreads")
                    col.metric(
                        f"{phase_labels[phase]} vs Neutral (stocks)",
                        f"{pct_diff:+.1f}%",
                        delta=implication,
                        delta_color="inverse" if pct_diff < 0 else "normal",
                    )

        if not cy_stocks.empty:
            cy_vs_neu = []
            for _, row in cy_stocks.iterrows():
                m = row["month"]
                if m in neu_avg.index and neu_avg[m] > 0:
                    cy_vs_neu.append(
                        (row["mpob_stocks"] - neu_avg[m]) / neu_avg[m] * 100)
            if cy_vs_neu:
                avg_diff = sum(cy_vs_neu) / len(cy_vs_neu)
                col_m3.metric(
                    f"{current_year} vs Neutral (stocks)",
                    f"{avg_diff:+.1f}%",
                    delta="above neutral avg" if avg_diff > 0 else "below neutral avg",
                    delta_color="normal" if avg_diff > 0 else "inverse",
                )

        col_m4.metric(
            "Neutral avg stocks",
            f"{neu_avg.mean() / 1e6:.2f}M t",
            help="Average monthly MPOB stocks across all Neutral years",
        )

    st.markdown("---")

    # ── CHART 2: Production Anomaly ──────────────────────────
    if has_production:
        st.caption(
            "**CPO Production Anomaly vs Seasonal Average** — "
            "positive = above seasonal norm · negative = below"
        )

        sd_clean = sd_df[["year", "month", "Production"]].copy()
        seasonal_avg = sd_clean.groupby("month")["Production"].mean()
        sd_clean["prod_anomaly_pct"] = sd_clean.apply(
            lambda r: ((r["Production"] - seasonal_avg[r["month"]])
                       / seasonal_avg[r["month"]] * 100)
            if seasonal_avg.get(r["month"], 0) > 0 else None, axis=1,
        )
        sd_clean["date"] = sd_clean.apply(
            lambda r: pd.Timestamp(year=int(r["year"]),
                                   month=int(r["month"]), day=1), axis=1)

        prod_bands, prod_enso = _build_phase_bands(
            sd_clean.rename(columns={"prod_anomaly_pct": "mpob_stocks"}),
            "mpob_stocks", oni_df, current_year)

        fig_prod = go.Figure()
        fig_prod.add_hline(y=0, line_color="#ffffff",
                           line_dash="dot", line_width=1)

        for phase in phases_to_show:
            if phase not in prod_bands:
                continue
            agg   = prod_bands[phase].sort_values("month")
            color = phase_colors[phase]
            label = phase_labels[phase]
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

            fig_prod.add_trace(go.Scatter(
                x=agg["month_name"].tolist() + agg["month_name"].tolist()[::-1],
                y=agg["max"].tolist() + agg["min"].tolist()[::-1],
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.12)",
                line=dict(width=0),
                name=f"{label} range",
                showlegend=True,
            ))
            fig_prod.add_trace(go.Scatter(
                x=agg["month_name"], y=agg["mean"],
                name=f"{label} avg",
                line=dict(color=color, width=2),
                mode="lines+markers", marker=dict(size=5),
            ))

        if "Neutral" in prod_bands:
            neu_p = prod_bands["Neutral"].sort_values("month")
            fig_prod.add_trace(go.Scatter(
                x=neu_p["month_name"], y=neu_p["mean"],
                name="Neutral avg",
                line=dict(color="#888888", width=1.2, dash="dot"),
                mode="lines", opacity=0.6,
            ))

        cy_prod = prod_enso[prod_enso["year"] == current_year].sort_values("month")
        if not cy_prod.empty:
            fig_prod.add_trace(go.Scatter(
                x=cy_prod["month"].apply(lambda m: month_names[m - 1]),
                y=cy_prod["mpob_stocks"],
                name=f"{current_year} actual",
                line=dict(color="#ff6b35", width=2.5),
                mode="lines+markers",
                marker=dict(size=9, symbol="star"),
            ))

        fig_prod.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
            font_color=DARK_TEXT, height=320,
            yaxis_title="Production anomaly (%)",
            xaxis=dict(categoryorder="array", categoryarray=month_names),
            xaxis_gridcolor=DARK_GRID, yaxis_gridcolor=DARK_GRID,
            margin=dict(l=40, r=20, t=20, b=30),
            legend=dict(orientation="h", y=1.05, font=dict(size=9)),
        )
        st.plotly_chart(fig_prod, use_container_width=True)

        if "Neutral" in prod_bands:
            neu_prod_avg = prod_bands["Neutral"].set_index("month")["mean"]
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            for phase, col in [("El Nino", col_p1), ("La Nina", col_p2)]:
                if phase in prod_bands:
                    ph_avg = prod_bands[phase].set_index("month")["mean"]
                    common = neu_prod_avg.index.intersection(ph_avg.index)
                    if len(common) > 0:
                        pct = (ph_avg[common] - neu_prod_avg[common]).mean()
                        col.metric(
                            f"{phase_labels[phase]} vs Neutral (production)",
                            f"{pct:+.1f}%",
                            delta=("below seasonal avg → supply tightening"
                                   if pct < 0 else "above seasonal avg → supply building"),
                            delta_color="inverse" if pct < 0 else "normal",
                        )
            if not cy_prod.empty:
                cy_pct = cy_prod["mpob_stocks"].mean()
                col_p3.metric(
                    f"{current_year} vs seasonal avg",
                    f"{cy_pct:+.1f}%",
                    delta="above avg" if cy_pct > 0 else "below avg",
                    delta_color="normal" if cy_pct > 0 else "inverse",
                )

        st.markdown("---")

    # ── CHART 3: Lag Analysis ─────────────────────────────────
    st.subheader("Lag Analysis — ONI vs Production Anomaly")
    st.caption(
        "Tests how many months after an ONI reading the production impact arrives. "
        "Each dot = one month. Downward slope = El Niño reduces production after the lag. "
        "Select lag to find which offset shows the clearest relationship in your data."
    )

    if has_production:
        lag_months = st.select_slider(
            "Lag (months): ONI today → production impact in N months",
            options=[3, 6, 9, 12, 15],
            value=9,
            key="lag_slider",
        )

        oni_monthly = oni_df[["date", "oni_anomaly", "enso_phase"]].copy()
        oni_monthly["date"] = pd.to_datetime(oni_monthly["date"])
        oni_monthly["lag_date"] = oni_monthly["date"] + pd.DateOffset(months=lag_months)

        sd_monthly = sd_clean[["date", "prod_anomaly_pct"]].copy()
        sd_monthly["date"] = pd.to_datetime(sd_monthly["date"])

        lag_df = oni_monthly.merge(
            sd_monthly.rename(columns={"date": "lag_date",
                                       "prod_anomaly_pct": "prod_at_lag"}),
            on="lag_date", how="inner",
        ).dropna(subset=["oni_anomaly", "prod_at_lag"])

        if len(lag_df) >= 10:
            dot_colors = [phase_colors.get(p, "#888888")
                          for p in lag_df["enso_phase"]]

            fig_lag = go.Figure()
            fig_lag.add_trace(go.Scatter(
                x=lag_df["oni_anomaly"],
                y=lag_df["prod_at_lag"],
                mode="markers",
                marker=dict(color=dot_colors, size=8, opacity=0.8),
                text=lag_df["lag_date"].dt.strftime("%b %Y"),
                hovertemplate=(
                    "ONI: %{x:+.2f}°C<br>Prod anomaly: %{y:+.1f}%"
                    "<br>%{text}<extra></extra>"
                ),
                showlegend=False,
            ))

            z = np.polyfit(lag_df["oni_anomaly"], lag_df["prod_at_lag"], 1)
            x_line = np.linspace(lag_df["oni_anomaly"].min(),
                                 lag_df["oni_anomaly"].max(), 50)
            y_line = np.polyval(z, x_line)
            corr   = lag_df["oni_anomaly"].corr(lag_df["prod_at_lag"])

            fig_lag.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color="#ffd54f", width=1.5, dash="dash"),
                name=f"Trend (r={corr:.2f})",
            ))
            fig_lag.add_hline(y=0, line_color="#ffffff",
                              line_dash="dot", line_width=0.8)
            fig_lag.add_vline(x=0, line_color="#ffffff",
                              line_dash="dot", line_width=0.8)
            fig_lag.add_vline(x=0.5, line_color="#ef5350",
                              line_dash="dash", line_width=0.8,
                              annotation_text="El Niño threshold")
            fig_lag.add_vline(x=-0.5, line_color="#4fc3f7",
                              line_dash="dash", line_width=0.8,
                              annotation_text="La Niña threshold")

            fig_lag.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
                font_color=DARK_TEXT, height=320,
                xaxis_title="ONI anomaly (°C)",
                yaxis_title=f"Production anomaly {lag_months}mo later (%)",
                xaxis_gridcolor=DARK_GRID, yaxis_gridcolor=DARK_GRID,
                margin=dict(l=40, r=40, t=20, b=30),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_lag, use_container_width=True)

            strength_label = ("strong" if abs(corr) > 0.5
                              else ("moderate" if abs(corr) > 0.3 else "weak"))
            direction = "negative" if corr < 0 else "positive"
            st.caption(
                f"Lag = {lag_months} months · Correlation r = {corr:.2f} "
                f"({strength_label} {direction} relationship) · "
                f"n = {len(lag_df)} observations · "
                f"{'El Niño (positive ONI) associated with lower production after lag' if corr < -0.2 else 'Weak or unclear lag relationship at this offset — try adjusting the lag slider'}"
            )
        else:
            st.info("Insufficient overlapping data for lag analysis at this offset.")

    st.markdown("---")

    # ── CHART 4: FCPO Price Response ─────────────────────────
    if has_price:
        st.subheader("FCPO Price — Average by ENSO Phase")
        st.caption(
            "Average monthly FCPO price grouped by historical ENSO phase. "
            "Shaded band = historical min/max. Orange = 2026 actual. "
            "Shows whether El Niño years historically had higher or lower prices."
        )

        price_bands, price_enso = _build_phase_bands(
            price_monthly.rename(columns={"fcpo_price": "mpob_stocks"}),
            "mpob_stocks", oni_df, current_year)

        fig_price = go.Figure()

        for phase in phases_to_show:
            if phase not in price_bands:
                continue
            agg   = price_bands[phase].sort_values("month")
            color = phase_colors[phase]
            label = phase_labels[phase]
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

            fig_price.add_trace(go.Scatter(
                x=agg["month_name"].tolist() + agg["month_name"].tolist()[::-1],
                y=agg["max"].tolist() + agg["min"].tolist()[::-1],
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.12)",
                line=dict(width=0),
                name=f"{label} range", showlegend=True,
            ))
            fig_price.add_trace(go.Scatter(
                x=agg["month_name"], y=agg["mean"],
                name=f"{label} avg",
                line=dict(color=color, width=2),
                mode="lines+markers", marker=dict(size=5),
            ))

        if "Neutral" in price_bands:
            neu_pr = price_bands["Neutral"].sort_values("month")
            fig_price.add_trace(go.Scatter(
                x=neu_pr["month_name"], y=neu_pr["mean"],
                name="Neutral avg",
                line=dict(color="#888888", width=1.2, dash="dot"),
                mode="lines", opacity=0.6,
            ))

        cy_price = price_enso[price_enso["year"] == current_year].sort_values("month")
        if not cy_price.empty:
            fig_price.add_trace(go.Scatter(
                x=cy_price["month"].apply(lambda m: month_names[m - 1]),
                y=cy_price["mpob_stocks"],
                name=f"{current_year} actual",
                line=dict(color="#ff6b35", width=2.5),
                mode="lines+markers",
                marker=dict(size=9, symbol="star"),
            ))

        fig_price.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
            font_color=DARK_TEXT, height=320,
            yaxis_title="FCPO price (MYR/tonne)",
            xaxis=dict(categoryorder="array", categoryarray=month_names),
            xaxis_gridcolor=DARK_GRID, yaxis_gridcolor=DARK_GRID,
            margin=dict(l=40, r=20, t=20, b=30),
            legend=dict(orientation="h", y=1.05, font=dict(size=9)),
        )
        st.plotly_chart(fig_price, use_container_width=True)

        if "Neutral" in price_bands:
            neu_price_avg = price_bands["Neutral"].set_index("month")["mean"]
            col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
            for phase, col in [("El Nino", col_pr1), ("La Nina", col_pr2)]:
                if phase in price_bands:
                    ph_avg = price_bands[phase].set_index("month")["mean"]
                    common = neu_price_avg.index.intersection(ph_avg.index)
                    if len(common) > 0:
                        pct = ((ph_avg[common] - neu_price_avg[common]) /
                               neu_price_avg[common] * 100).mean()
                        col.metric(
                            f"{phase_labels[phase]} vs Neutral (price)",
                            f"{pct:+.1f}%",
                            delta=("higher price environment" if pct > 0
                                   else "lower price environment"),
                            delta_color="normal" if pct > 0 else "inverse",
                        )
            if not cy_price.empty:
                cy_pr_pct_list = []
                for _, row in cy_price.iterrows():
                    m = row["month"]
                    if m in neu_price_avg.index and neu_price_avg[m] > 0:
                        cy_pr_pct_list.append(
                            (row["mpob_stocks"] - neu_price_avg[m]) / neu_price_avg[m] * 100)
                if cy_pr_pct_list:
                    col_pr3.metric(
                        f"{current_year} vs Neutral (price)",
                        f"{sum(cy_pr_pct_list) / len(cy_pr_pct_list):+.1f}%",
                    )
            col_pr4.metric(
                "Neutral avg price",
                f"MYR {neu_price_avg.mean():,.0f}/t",
            )

    st.markdown("---")

    # ── PANEL 5: Update instructions ────────────────────────
    with st.expander("📋 How to update this tab monthly"):
        st.markdown(f"""
**Monthly update — 5 minutes total**

**Step 1 — ONI History (30 seconds):**
1. Go to [cpc.ncep.noaa.gov/data/indices/oni.ascii.txt](https://cpc.ncep.noaa.gov/data/indices/oni.ascii.txt)
2. Download the file
3. Replace `Raw Data/ENSO/oni.ascii.txt` with the new download
4. No manual editing needed — dashboard reads it automatically

**Step 2 — ENSO Forecast (4 minutes):**
1. Go to [iri.columbia.edu/our-expertise/climate/forecasts/](https://iri.columbia.edu/our-expertise/climate/forecasts/) → ENSO Forecast
2. Read the probability table (La Niña %, Neutral %, El Niño % per season)
3. Open `Raw Data/ENSO/ENSO_Data_Template.xlsx` → ENSO Forecast sheet
4. **ADD new rows at the bottom** with today's date as Issue Date
5. **Do NOT delete old rows** — history needed for the evolution table
6. Save the file

**Current data status:**
- ONI history: {oni_df['season'].iloc[-1]} {int(oni_df['year'].iloc[-1])} (latest entry)
- ONI coverage: {int(oni_df['year'].min())} – {int(oni_df['year'].max())} ({len(oni_df)} observations)
- Forecast: {'loaded — ' + fc_df['issue_date'].max().strftime('%d %b %Y') if fc_df is not None and not fc_df.empty else 'not loaded'}
        """)

    with st.expander("⚠️ Important caveats before trading on ENSO signals"):
        st.markdown("""
- **Back months only (M6+).** Front months M1-M3 use producer intelligence and MPOB data.
- **Lag varies 6-15 months.** Cannot predict the exact production impact month.
- **Demand can overwhelm supply.** 2022: El Niño supply pressure overwhelmed by Ukraine war demand shock.
- **Strength matters more than probability.** Confirmed Strong El Niño (ONI >1.5°C) is more actionable than 90% probability of Weak El Niño.
- **Directional bias only.** Confirm with Z-score and MPOB before entering any trade.
- **Update monthly.** Stale forecast data is worse than no data.
        """)
