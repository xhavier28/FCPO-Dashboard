import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots

DATA_PATH = "Raw Data/MYX_DLY_FCPO1!, D_59dbd.csv"
YEAR_COLORS = {2023: "#1f77b4", 2024: "#ff7f0e", 2025: "#2ca02c", 2026: "#d62728"}
TERM_DIR = "Raw Data/Term Structure"
MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

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
    df = df[df["year"].isin([2023, 2024, 2025, 2026])]
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
    all_dates = [d for d in all_dates if d >= pd.Timestamp("2023-01-01")]

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

    # --- Row 1: spot price ---
    fig.add_trace(
        go.Scatter(
            x=yr_df["doy"], y=smoothed,
            mode="lines",
            line=dict(color=YEAR_COLORS.get(year, "#636efa"), width=2.5),
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
        comp_df = df[df["year"] == compare_year].sort_values("doy").copy()
        comp_smoothed = comp_df["close"].rolling(window=5, min_periods=1, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=comp_df["doy"], y=comp_smoothed,
                mode="lines",
                line=dict(color=YEAR_COLORS.get(compare_year, "#aaaaaa"), width=2, dash="dash"),
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
                    line=dict(color=colorscale[i], width=1.5),
                    showlegend=False,
                    name=row_data["Week"],
                    customdata=hover,
                    hovertemplate=row_data["Week"] + "<br>%{customdata}<extra></extra>",
                ),
                row=2, col=1,
            )

    # --- Row 2: comparison year slivers (dashed) ---
    if compare_year is not None:
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
                        line=dict(color=comp_colorscale[i], width=1.5, dash="dash"),
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


st.set_page_config(page_title="FCPO Dashboard", layout="wide")
st.title("MYX FCPO Futures")

df = load_data(DATA_PATH)

tab1, tab2 = st.tabs(["Year-over-Year", "Term Structure"])

with tab1:
    selected_years = st.multiselect(
        "Select Years",
        options=[2023, 2024, 2025, 2026],
        default=[2023, 2024, 2025, 2026],
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

    years = available_years(contracts)

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

    st.subheader("Weekly Data Table")
    st.caption(
        "Average daily Close per week per contract. Front month rolls on the 16th."
    )
    st.dataframe(df_term.set_index("Week"), use_container_width=True)
