import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots
from FCPO_analysis import build_spread_table, load_combined_dataset

DATA_PATH = "Raw Data/MYX_DLY_FCPO1!, D_59dbd.csv"
YEAR_COLORS = {
    2020: "#9467bd", 2021: "#8c564b", 2022: "#e377c2",
    2023: "#1f77b4", 2024: "#ff7f0e", 2025: "#2ca02c", 2026: "#d62728",
}
TERM_DIR = "Raw Data/Term Structure"
SD_DIR   = "Raw Data/Stock and Production"
MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


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
def load_data(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("Asia/Kuala_Lumpur")
    df["date"] = df["datetime"].dt.date
    df = df.sort_values("time").groupby("date", as_index=False).last()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df = df[df["year"].isin([2020, 2021, 2022, 2023, 2024, 2025, 2026])]
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
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d %b %Y")
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

df = load_data(DATA_PATH)

tab1, tab2, tab3, tab4 = st.tabs(["Year-over-Year", "Term Structure", "Supply & Demand", "Mean Reversion"])

with tab1:
    selected_years = st.multiselect(
        "Select Years",
        options=[2020, 2021, 2022, 2023, 2024, 2025, 2026],
        default=[2020, 2021, 2022, 2023, 2024, 2025, 2026],
    )
    if not selected_years:
        st.warning("Select at least one year.")
    else:
        fig = go.Figure()

        most_recent = max(selected_years)

        for year in selected_years:
            yr_df = df[df["year"] == year].sort_values("doy").copy()
            is_recent = year == most_recent
            smoothed = yr_df["close"].rolling(window=5, min_periods=1, center=True).mean()
            color = YEAR_COLORS[year] if is_recent else hex_to_rgba(YEAR_COLORS[year], 0.35)
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

        stat_rows = []
        for year in selected_years:
            yr_df = df[df["year"] == year]
            stat_rows.append(
                {
                    "Year": year,
                    "Trading Days": len(yr_df),
                    "Min Close": f"MYR {yr_df['close'].min():,.0f}",
                    "Max Close": f"MYR {yr_df['close'].max():,.0f}",
                    "Latest Close": f"MYR {yr_df.sort_values('date')['close'].iloc[-1]:,.0f}",
                    "Avg Volume": f"{yr_df['Volume'].mean():,.0f}",
                }
            )

        st.dataframe(pd.DataFrame(stat_rows).set_index("Year"), use_container_width=True)

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

with tab3:
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

with tab4:
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
