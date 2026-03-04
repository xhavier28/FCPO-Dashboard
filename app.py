import streamlit as st
import pandas as pd
import plotly.graph_objects as go

DATA_PATH = "Raw Data/MYX_DLY_FCPO1!, D_59dbd.csv"
YEAR_COLORS = {2023: "#1f77b4", 2024: "#ff7f0e", 2025: "#2ca02c", 2026: "#d62728"}


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


st.set_page_config(page_title="FCPO Year-over-Year", layout="wide")
st.title("MYX FCPO Futures — Year-over-Year Price Comparison")

df = load_data(DATA_PATH)

selected_years = st.sidebar.multiselect(
    "Select Years",
    options=[2023, 2024, 2025, 2026],
    default=[2023, 2024, 2025, 2026],
)

if not selected_years:
    st.warning("Select at least one year from the sidebar.")
    st.stop()

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

# Summary stats table
st.subheader("Summary Statistics")

rows = []
for year in selected_years:
    yr_df = df[df["year"] == year]
    rows.append(
        {
            "Year": year,
            "Trading Days": len(yr_df),
            "Min Close": f"MYR {yr_df['close'].min():,.0f}",
            "Max Close": f"MYR {yr_df['close'].max():,.0f}",
            "Latest Close": f"MYR {yr_df.sort_values('date')['close'].iloc[-1]:,.0f}",
            "Avg Volume": f"{yr_df['Volume'].mean():,.0f}",
        }
    )

st.dataframe(pd.DataFrame(rows).set_index("Year"), use_container_width=True)
