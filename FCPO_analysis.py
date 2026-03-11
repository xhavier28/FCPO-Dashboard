"""
FCPO_analysis.py
================
Standalone data analysis script for MYX FCPO Futures.
Separate from the Streamlit dashboard (app.py).

Usage:
    python FCPO_analysis.py

Each section can also be imported individually:
    from FCPO_analysis import load_spot_prices, spot_summary, plot_spot_yoy
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH      = "Raw Data/MYX_DLY_FCPO1!, D_59dbd.csv"
DAILY_TERM_PATH = "Raw Data/Daily Term/fcpo_daily_term_structure.xlsx"
TERM_DIR       = "Raw Data/Term Structure"
SD_DIR         = "Raw Data/Stock and Production"

MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

YEAR_COLORS = {
    2020: "#9467bd", 2021: "#8c564b", 2022: "#e377c2",
    2023: "#1f77b4", 2024: "#ff7f0e", 2025: "#2ca02c", 2026: "#d62728",
}

TENOR_COLS = ["Current"] + [f"+{i}M" for i in range(1, 12)]

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_spot_prices():
    """
    Load daily FCPO spot (front-month continuous) prices.
    Returns DataFrame with columns: date, year, month, doy, open, high, low, close, volume.
    """
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("Asia/Kuala_Lumpur")
    df["date"]     = df["datetime"].dt.date
    df = df.sort_values("time").groupby("date", as_index=False).last()
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["doy"]   = df["date"].dt.dayofyear
    df = df[df["year"] >= 2020].copy()
    return df[["date","year","month","doy","open","high","low","close","Volume"]].rename(columns={"Volume":"volume"})


def _front_month(date):
    if date.day <= 15:
        return (date.year, date.month)
    m = date.month + 1
    y = date.year + (1 if m > 12 else 0)
    return (y, m % 12 or 12)


def _add_months(ym, n):
    y, m = ym
    m += n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return (y, m)


def load_contracts():
    """
    Load all FCPO forward contracts from CSV files.
    Returns dict: {(year, month): pd.Series indexed by date}.
    """
    contracts = {}
    if not os.path.exists(TERM_DIR):
        return contracts
    for year_dir in sorted(os.listdir(TERM_DIR)):
        if not year_dir.isdigit():
            continue
        year = int(year_dir)
        for m, abbr in enumerate(MONTH_ABBRS, 1):
            path = f"{TERM_DIR}/{year_dir}/FCPO {abbr}{str(year)[2:]}_Daily.csv"
            if not os.path.exists(path):
                continue
            df_c = pd.read_csv(path)
            df_c["date"] = pd.to_datetime(df_c["Timestamp (UTC)"], infer_datetime_format=True).dt.normalize()
            series = df_c.set_index("date")["Close"]
            contracts[(year, m)] = series[~series.index.duplicated(keep="last")]
    return contracts


def build_daily_curves(contracts):
    """
    Build a daily forward curve table.
    Returns DataFrame: rows = trading date, columns = Date + 12 tenors (Current, +1M … +11M).
    """
    all_dates = sorted(set(d for s in contracts.values() for d in s.index))
    today     = pd.Timestamp.today().normalize()
    all_dates = [d for d in all_dates if pd.Timestamp("2020-01-01") <= d <= today]

    records = []
    for date in all_dates:
        fm  = _front_month(date)
        row = {"Date": date}
        for i, col in enumerate(TENOR_COLS):
            ym = _add_months(fm, i)
            row[col] = contracts[ym][date] if (ym in contracts and date in contracts[ym].index) else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index("Date")


def load_supply_demand():
    """
    Load MPOB supply & demand data from Excel files.
    Returns DataFrame with columns: Date, year, month, Stock, Exports, Production, Consumption.
    """
    def _read(fname, col):
        df = pd.read_excel(f"{SD_DIR}/{fname}", sheet_name="Table Data")
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = ["Date", col]
        df["Date"] = pd.to_datetime(df["Date"])
        return df.dropna(subset=["Date"]).set_index("Date")[col]

    df = pd.DataFrame({
        "Stock":      _read("FCPO Stock 3Y.xlsx",      "Stock"),
        "Exports":    _read("MPOB Exports 3Y.xlsx",    "Exports"),
        "Production": _read("MPOB Production 3Y.xlsx", "Production"),
    }).sort_index().reset_index().rename(columns={"index": "Date"})

    df["year"]        = df["Date"].dt.year
    df["month"]       = df["Date"].dt.month
    df["Delta_Stock"] = df["Stock"].diff()
    df["Consumption"] = df["Production"] - df["Exports"] - df["Delta_Stock"]
    return df


def load_combined_dataset():
    """
    Merge daily term structure with spot OHLCV into one clean DataFrame.

    Data quality fixes applied:
      1. Missing 'Current' (~45 days, contract roll 13th–15th): filled with +1M,
         which always has data on those days — the +1M is effectively the new
         front month when the current contract expires.
      2. Missing spot OHLCV on Fridays (~300 days): the continuous contract CSV
         has almost no Friday data. Forward-filled from the previous trading day.

    Returns DataFrame indexed by date with columns:
        open, high, low, close, volume,
        Current, +1M, +2M, ... +11M,
        year, month, doy, weekday,
        current_filled (bool — True on days where Current was imputed from +1M)
    """
    # ── Term structure ────────────────────────────────────────────────────────
    df_term = pd.read_excel(DAILY_TERM_PATH, index_col=0)
    df_term.index = pd.to_datetime(df_term.index, format="%d %b %Y")
    df_term.index.name = "date"
    df_term = df_term[df_term.index.year >= 2020].sort_index()

    # Fix 1: fill missing Current with +1M on roll days
    roll_mask = df_term["Current"].isnull() & df_term["+1M"].notnull()
    df_term.loc[roll_mask, "Current"] = df_term.loc[roll_mask, "+1M"]
    df_term["current_filled"] = roll_mask

    # ── Spot OHLCV ───────────────────────────────────────────────────────────
    df_spot = pd.read_csv(DATA_PATH)
    df_spot["datetime"] = pd.to_datetime(df_spot["time"], unit="s", utc=True).dt.tz_convert("Asia/Kuala_Lumpur")
    df_spot["date"]     = pd.to_datetime(df_spot["datetime"].dt.date)
    df_spot = df_spot.sort_values("time").groupby("date", as_index=False).last()
    df_spot = df_spot[df_spot["date"].dt.year >= 2020].set_index("date")
    df_spot = df_spot[["open", "high", "low", "close", "Volume"]].rename(columns={"Volume": "volume"})

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = df_term.join(df_spot, how="left")

    # Fix 2: forward-fill missing spot OHLCV (mostly Fridays — no data in source)
    spot_cols = ["open", "high", "low", "close", "volume"]
    df[spot_cols] = df[spot_cols].ffill()

    # ── Add calendar columns ─────────────────────────────────────────────────
    df["year"]    = df.index.year
    df["month"]   = df.index.month
    df["doy"]     = df.index.dayofyear
    df["weekday"] = df.index.day_name()

    return df.reset_index()


def enrich_dataset(df):
    """
    Add derived analysis columns to the combined dataset.

    New columns added:
      em_avg      — average of +3M to +7M (Early Month)
      fm_avg      — average of +8M to +11M (Far Month)
      em_fm_spread— EM minus FM (positive = Backwardation, negative = Contango)
      term_shape  — 'Backwardation' if EM > FM, else 'Contango'

      dod_close_pct  — day-over-day % change in spot close: (today-yesterday)/yesterday*100
      dod_<tenor>_pct— same for each of the 12 tenor columns

      open_close     — open minus close (positive = day closed lower than open)
      high_low       — high minus low (day's price range)
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Term shape ────────────────────────────────────────────────────────────
    # Early Month (EM): average of +3M to +7M
    # Far Month  (FM): average of +8M to +11M
    # EM > FM → Backwardation  |  FM > EM → Contango
    em_cols = ["+3M", "+4M", "+5M", "+6M", "+7M"]
    fm_cols = ["+8M", "+9M", "+10M", "+11M"]
    df["em_avg"]       = df[em_cols].mean(axis=1)
    df["fm_avg"]       = df[fm_cols].mean(axis=1)
    df["em_fm_spread"] = df["em_avg"] - df["fm_avg"]
    df["term_shape"]   = df["em_fm_spread"].apply(
        lambda x: "Backwardation" if pd.notna(x) and x > 0 else "Contango"
    )

    # ── Day-over-day % changes ────────────────────────────────────────────────
    # Formula: (today - yesterday) / yesterday * 100
    df["dod_close_pct"] = df["close"].pct_change() * 100
    for col in TENOR_COLS:
        df[f"dod_{col.replace('+','').replace('M','m')}_pct"] = df[col].pct_change() * 100

    # ── Intraday range columns ────────────────────────────────────────────────
    df["open_close"] = df["open"] - df["close"]   # >0: closed lower than open
    df["high_low"]   = df["high"] - df["low"]     # day's trading range

    return df


# ──────────────────────────────────────────────────────────────────────────────
# SPOT PRICE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def spot_summary(df):
    """
    Per-year summary statistics for spot close prices.
    Returns DataFrame.
    """
    rows = []
    for year, g in df.groupby("year"):
        g = g.sort_values("date")
        daily_ret = g["close"].pct_change().dropna()
        rows.append({
            "Year":           year,
            "Trading Days":   len(g),
            "Open (YTD)":     round(g["close"].iloc[0], 0),
            "Latest Close":   round(g["close"].iloc[-1], 0),
            "Min":            round(g["close"].min(), 0),
            "Max":            round(g["close"].max(), 0),
            "Mean":           round(g["close"].mean(), 0),
            "Std Dev":        round(g["close"].std(), 0),
            "Ann. Volatility":f"{daily_ret.std() * np.sqrt(252) * 100:.1f}%",
            "YTD Return":     f"{(g['close'].iloc[-1] / g['close'].iloc[0] - 1) * 100:+.1f}%",
        })
    return pd.DataFrame(rows).set_index("Year")


def rolling_volatility(df, window=30):
    """
    Compute rolling annualised volatility from daily log returns.
    Returns DataFrame with columns: date, year, rolling_vol.
    """
    df = df.sort_values("date").copy()
    df["log_ret"]     = np.log(df["close"] / df["close"].shift(1))
    df["rolling_vol"] = df["log_ret"].rolling(window).std() * np.sqrt(252) * 100
    return df[["date","year","close","rolling_vol"]].dropna()


def plot_spot_yoy(df):
    """Year-over-year close price on day-of-year axis."""
    fig, ax = plt.subplots(figsize=(14, 5))
    years = sorted(df["year"].unique())
    most_recent = max(years)

    for year in years:
        g     = df[df["year"] == year].sort_values("doy")
        color = YEAR_COLORS.get(year, "#aaaaaa")
        alpha = 1.0 if year == most_recent else 0.4
        lw    = 2.2 if year == most_recent else 1.2
        smoothed = g["close"].rolling(5, min_periods=1, center=True).mean()
        ax.plot(g["doy"], smoothed, color=color, alpha=alpha, lw=lw, label=str(year))

    ax.set_xticks([1,32,60,91,121,152,182,213,244,274,305,335])
    ax.set_xticklabels(MONTH_ABBRS)
    ax.set_xlim(1, 366)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"MYR {x:,.0f}"))
    ax.set_title("FCPO Spot Price — Year-over-Year", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Close (MYR)")
    ax.legend(title="Year", loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_rolling_volatility(df, window=30):
    """Rolling annualised volatility by year."""
    df_vol = rolling_volatility(df, window)
    fig, ax = plt.subplots(figsize=(14, 4))
    years = sorted(df_vol["year"].unique())
    most_recent = max(years)

    for year in years:
        g     = df_vol[df_vol["year"] == year].sort_values("date")
        color = YEAR_COLORS.get(year, "#aaaaaa")
        alpha = 1.0 if year == most_recent else 0.4
        lw    = 2.0 if year == most_recent else 1.2
        ax.plot(g["doy"] if "doy" in g else range(len(g)), g["rolling_vol"],
                color=color, alpha=alpha, lw=lw, label=str(year))

    # rebuild doy from rolling_volatility output
    df_vol["doy"] = pd.to_datetime(df_vol["date"]).dt.dayofyear
    ax.cla()
    for year in years:
        g     = df_vol[df_vol["year"] == year].sort_values("doy")
        color = YEAR_COLORS.get(year, "#aaaaaa")
        alpha = 1.0 if year == most_recent else 0.4
        lw    = 2.0 if year == most_recent else 1.2
        ax.plot(g["doy"], g["rolling_vol"], color=color, alpha=alpha, lw=lw, label=str(year))

    ax.set_xticks([1,32,60,91,121,152,182,213,244,274,305,335])
    ax.set_xticklabels(MONTH_ABBRS)
    ax.set_xlim(1, 366)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title(f"FCPO {window}-Day Rolling Annualised Volatility", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Volatility (%)")
    ax.legend(title="Year", loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# TERM STRUCTURE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def curve_shape_stats(df_curves):
    """
    Classify each day's forward curve as contango or backwardation.
    Compares Current vs +3M tenor.
    Returns DataFrame with: Date, Current, +3M, spread_3m, shape.
    """
    df = df_curves[["Current", "+3M"]].dropna().copy()
    df["spread_3m"] = df["+3M"] - df["Current"]
    df["shape"]     = df["spread_3m"].apply(lambda x: "Contango" if x > 0 else "Backwardation")
    return df.reset_index()


def tenor_spreads(df_curves):
    """
    Compute daily spreads between front month and each forward tenor.
    Returns DataFrame: Date + columns [+1M … +11M spread vs Current].
    """
    spreads = pd.DataFrame(index=df_curves.index)
    for col in TENOR_COLS[1:]:
        if col in df_curves.columns:
            spreads[f"{col} spread"] = df_curves[col] - df_curves["Current"]
    return spreads.dropna(how="all").reset_index()


def plot_curve_snapshot(df_curves, date=None):
    """
    Plot the forward curve for a single date (defaults to latest available).
    """
    if date is None:
        date = df_curves.index.max()
    else:
        date = pd.Timestamp(date)

    row = df_curves.loc[date] if date in df_curves.index else df_curves.iloc[-1]
    date_used = df_curves.index[-1] if date not in df_curves.index else date

    tenors = [i for i, col in enumerate(TENOR_COLS) if not np.isnan(row.get(col, np.nan))]
    prices = [row[TENOR_COLS[i]] for i in tenors]
    labels = [TENOR_COLS[i] for i in tenors]

    fig, ax = plt.subplots(figsize=(10, 4))
    color = "#d62728"
    ax.plot(tenors, prices, "o-", color=color, lw=2, ms=6)
    ax.set_xticks(tenors)
    ax.set_xticklabels(labels, rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"MYR {x:,.0f}"))
    ax.set_title(f"FCPO Forward Curve Snapshot — {date_used.strftime('%d %b %Y')}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Price (MYR)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_spread_history(df_curves, tenor="+3M"):
    """
    Plot historical spread between Current and a given tenor.
    Colours positive (contango) and negative (backwardation) areas differently.
    """
    df = df_curves[["Current", tenor]].dropna().copy()
    df["spread"] = df[tenor] - df["Current"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(df.index, df["spread"],
                    where=df["spread"] >= 0, color="#2ca02c", alpha=0.4, label="Contango")
    ax.fill_between(df.index, df["spread"],
                    where=df["spread"] <  0, color="#d62728", alpha=0.4, label="Backwardation")
    ax.plot(df.index, df["spread"], color="#444444", lw=0.8)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"MYR {x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_title(f"FCPO Spread: Current vs {tenor} (+ = Contango, − = Backwardation)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Spread (MYR)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SEASONALITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def monthly_returns(df):
    """
    Monthly average return for each calendar month, aggregated across years.
    Returns DataFrame: month × [avg_return, std_return, count].
    """
    df = df.sort_values("date").copy()
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month_start").agg(
        close_end=("close", "last"),
        close_start=("close", "first"),
        month=("month", "first"),
        year=("year", "first"),
    ).reset_index()
    monthly["ret"] = (monthly["close_end"] / monthly["close_start"] - 1) * 100

    agg = monthly.groupby("month")["ret"].agg(
        avg_return="mean", std_return="std", count="count"
    ).reset_index()
    agg["month_name"] = agg["month"].apply(lambda m: MONTH_ABBRS[m - 1])
    return agg.set_index("month_name")


def plot_monthly_seasonality(df):
    """Bar chart of average monthly return with ±1 std dev error bars."""
    agg = monthly_returns(df)
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in agg["avg_return"]]
    ax.bar(agg.index, agg["avg_return"], color=colors, alpha=0.8,
           yerr=agg["std_return"], capsize=4, error_kw=dict(elinewidth=1, ecolor="#555"))
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_title("FCPO Monthly Seasonality — Average Return by Calendar Month (2020–present)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Monthly Return (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_monthly_close_heatmap(df):
    """Heatmap of average monthly close price: rows = year, columns = month."""
    pivot = df.groupby(["year", "month"])["close"].mean().unstack()
    pivot.columns = [MONTH_ABBRS[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Avg Close (MYR)")
    ax.set_title("FCPO Average Monthly Close Price Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SUPPLY & DEMAND ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def sd_summary(df_sd):
    """
    Latest-year summary of S&D metrics with YoY change vs prior year.
    Returns DataFrame.
    """
    metrics = ["Production", "Exports", "Stock", "Consumption"]
    years   = sorted(df_sd["year"].unique())
    if len(years) < 2:
        return df_sd.groupby("year")[metrics].sum()

    latest, prior = years[-1], years[-2]
    rows = []
    for m in metrics:
        cur = df_sd[df_sd["year"] == latest][m].sum()
        prv = df_sd[df_sd["year"] == prior][m].sum()
        rows.append({
            "Metric":      m,
            f"{latest} YTD": f"{cur:,.0f}",
            f"{prior} YTD": f"{prv:,.0f}",
            "YoY Change":   f"{(cur/prv - 1)*100:+.1f}%" if prv else "–",
        })
    return pd.DataFrame(rows).set_index("Metric")


def sd_price_correlation(df_spot, df_sd):
    """
    Pearson correlation between monthly S&D metrics and monthly average spot price.
    Returns correlation DataFrame.
    """
    monthly_price = (
        df_spot.assign(period=df_spot["date"].dt.to_period("M"))
        .groupby("period")["close"].mean()
        .reset_index()
    )
    monthly_price["Date"] = monthly_price["period"].dt.to_timestamp()

    merged = pd.merge(
        df_sd[["Date", "Production", "Exports", "Stock", "Consumption"]],
        monthly_price[["Date", "close"]].rename(columns={"close": "Spot Price"}),
        on="Date", how="inner"
    )
    return merged[["Production", "Exports", "Stock", "Consumption", "Spot Price"]].corr()


def plot_sd_vs_price(df_spot, df_sd, metric="Stock"):
    """Dual-axis plot: monthly S&D metric vs monthly average spot price."""
    monthly_price = (
        df_spot.assign(period=df_spot["date"].dt.to_period("M"))
        .groupby("period")["close"].mean()
        .reset_index()
    )
    monthly_price["Date"] = monthly_price["period"].dt.to_timestamp()

    merged = pd.merge(
        df_sd[["Date", metric]],
        monthly_price[["Date", "close"]],
        on="Date", how="inner"
    ).sort_values("Date")

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()

    ax1.bar(merged["Date"], merged[metric], width=20, color="#1f77b4", alpha=0.5, label=metric)
    ax2.plot(merged["Date"], merged["close"], color="#d62728", lw=2, label="Spot Price")

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"MYR {x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax1.set_title(f"FCPO {metric} vs Spot Price", fontsize=13, fontweight="bold")
    ax1.set_ylabel(f"{metric} (tonnes)", color="#1f77b4")
    ax2.set_ylabel("Spot Close (MYR)", color="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — run all analyses
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading data...")
    df_combined = load_combined_dataset()
    df_spot     = load_spot_prices()
    contracts   = load_contracts()
    df_curves   = build_daily_curves(contracts)
    df_sd       = load_supply_demand()

    # ── Combined Dataset Preview ──────────────────────────────────────────────
    print("\n=== COMBINED DATASET (spot + term structure) ===")
    print(f"Rows: {len(df_combined)} | Date range: {df_combined['date'].min().date()} → {df_combined['date'].max().date()}")
    print(f"Columns: {df_combined.columns.tolist()}")
    print(f"\nCurrent filled from +1M (roll days): {df_combined['current_filled'].sum()} days")
    print(f"Spot forward-filled (Fridays/gaps):   {(df_combined['weekday'] == 'Friday').sum()} Fridays in dataset")
    print()
    display_cols = ["date", "open", "high", "low", "close", "volume", "Current", "+1M", "+3M", "+6M", "+11M"]
    print("Sample (first 5 rows):")
    print(df_combined[display_cols].head().to_string(index=False))
    print()
    print("Sample (last 5 rows):")
    print(df_combined[display_cols].tail().to_string(index=False))
    print()
    print("Null counts after cleaning:")
    print(df_combined[["close", "Current", "+1M", "+10M", "+11M"]].isnull().sum().to_string())

    # ── Spot Price Summary ────────────────────────────────────────────────────
    print("\n=== SPOT PRICE SUMMARY ===")
    print(spot_summary(df_spot).to_string())

    # ── Curve Shape Stats ─────────────────────────────────────────────────────
    print("\n=== CURVE SHAPE (Current vs +3M) ===")
    shape = curve_shape_stats(df_curves)
    counts = shape["shape"].value_counts()
    total  = len(shape)
    for s, c in counts.items():
        print(f"  {s}: {c} days ({c/total*100:.1f}%)")
    print(f"  Avg +3M spread: MYR {shape['spread_3m'].mean():+,.0f}")

    # ── Monthly Seasonality ───────────────────────────────────────────────────
    print("\n=== MONTHLY SEASONALITY ===")
    print(monthly_returns(df_spot)[["avg_return","std_return"]].round(2).to_string())

    # ── S&D Summary ───────────────────────────────────────────────────────────
    print("\n=== SUPPLY & DEMAND SUMMARY ===")
    print(sd_summary(df_sd).to_string())

    print("\n=== S&D vs SPOT CORRELATION ===")
    print(sd_price_correlation(df_spot, df_sd).round(3).to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating charts...")

    fig1 = plot_spot_yoy(df_spot)
    fig2 = plot_rolling_volatility(df_spot, window=30)
    fig3 = plot_curve_snapshot(df_curves)
    fig4 = plot_spread_history(df_curves, tenor="+3M")
    fig5 = plot_monthly_seasonality(df_spot)
    fig6 = plot_monthly_close_heatmap(df_spot)
    fig7 = plot_sd_vs_price(df_spot, df_sd, metric="Stock")
    fig8 = plot_sd_vs_price(df_spot, df_sd, metric="Production")

    plt.show()
    print("\nDone.")
