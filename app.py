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


def build_spot_chart(df, year):
    yr_df = df[df["year"] == year].sort_values("doy").copy()
    smoothed = yr_df["close"].rolling(window=5, min_periods=1, center=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yr_df["doy"],
        y=smoothed,
        mode="lines",
        name=str(year),
        line=dict(color=YEAR_COLORS.get(year, "#636efa"), width=2.5),
        customdata=yr_df[["date", "close"]].assign(
            date_fmt=yr_df["date"].dt.strftime("%d %b %Y")
        )[["date_fmt", "close"]].values,
        hovertemplate="%{customdata[0]}<br>MYR %{customdata[1]:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{year} Spot Price (5-day smoothed)",
        hovermode="x unified",
        xaxis=dict(title="Month", tickvals=TICKVALS, ticktext=TICKTEXT,
                   range=[1, 366], showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Close (MYR)", showgrid=True, gridcolor="#e0e0e0"),
        plot_bgcolor="white", height=300,
        margin=dict(l=60, r=30, t=40, b=60),
        showlegend=False,
    )
    return fig


def build_term_grid(df_term, year):
    """1 row × N cols (one box per week, left→right = Jan W1 → Dec W4)."""
    col_names = ["Current"] + [f"+{i}M" for i in range(1, 12)]

    year_rows = df_term[df_term["Week"].str.endswith(str(year))].reset_index(drop=True)
    n = len(year_rows)
    if n == 0:
        return go.Figure()

    colorscale = plotly.colors.sample_colorscale(
        "Plasma", [i / max(n - 1, 1) for i in range(n)]
    )

    fig = make_subplots(
        rows=1, cols=n,
        horizontal_spacing=0.003,
    )

    for i, row_data in year_rows.iterrows():
        y_vals = [row_data[col] for col in col_names]
        pairs = [(j, y) for j, y in enumerate(y_vals) if y is not None]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        fig.add_trace(
            go.Scatter(
                x=list(xs), y=list(ys),
                mode="lines",
                line=dict(color=colorscale[i], width=1.5),
                showlegend=False,
                name=row_data["Week"],
                hovertemplate=row_data["Week"] + "<br>%{x}: MYR %{y:,.0f}<extra></extra>",
            ),
            row=1, col=i + 1,
        )

    # Add month label annotations every 4 boxes (start of each month)
    for m_idx, abbr in enumerate(MONTH_ABBRS):
        col_pos = m_idx * 4 + 1  # W1 of that month = subplot col index
        if col_pos <= n:
            x_frac = (col_pos - 1 + 0.5 * 4) / n  # centre of the 4 boxes
            fig.add_annotation(
                x=x_frac, y=1.08, xref="paper", yref="paper",
                text=abbr, showarrow=False,
                font=dict(size=9, color="#555555"),
            )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                     showline=True, linecolor="#cccccc", linewidth=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                     showline=False)
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=160,
        margin=dict(l=60, r=30, t=30, b=10),
        showlegend=False,
    )
    return fig


st.set_page_config(page_title="FCPO Dashboard", layout="wide")
st.title("MYX FCPO Futures")

df = load_data(DATA_PATH)

selected_years = st.sidebar.multiselect(
    "Select Years",
    options=[2023, 2024, 2025, 2026],
    default=[2023, 2024, 2025, 2026],
)

tab1, tab2 = st.tabs(["Year-over-Year", "Term Structure"])

with tab1:
    if not selected_years:
        st.warning("Select at least one year from the sidebar.")
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
            xaxis=dict(
                title="Month",
                tickvals=TICKVALS,
                ticktext=TICKTEXT,
                range=[1, 366],
                showgrid=True,
                gridcolor="#e0e0e0",
            ),
            yaxis=dict(
                title="Close (MYR)",
                showgrid=True,
                gridcolor="#e0e0e0",
            ),
            legend=dict(title="Year", orientation="v"),
            plot_bgcolor="white",
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
    selected_ts_year = st.selectbox("Year", options=years[::-1], index=0)

    st.subheader(f"Spot Price — {selected_ts_year}")
    st.plotly_chart(build_spot_chart(df, selected_ts_year), use_container_width=True)

    st.subheader(f"Term Structure Grid — {selected_ts_year}")
    st.caption(
        "48 boxes in one row, left → right = Jan W1 → Dec W4. "
        "Each box shows the forward curve shape for that week (y-axis not to scale across boxes). "
        "Color: Plasma (blue/purple Jan → yellow Dec). Hover a box for week label and prices."
    )
    st.plotly_chart(build_term_grid(df_term, selected_ts_year), use_container_width=True)

    st.subheader("Weekly Data Table")
    st.caption(
        "Average daily Close per week per contract. Front month rolls on the 16th."
    )
    st.dataframe(df_term.set_index("Week"), use_container_width=True)
