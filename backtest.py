import os
import re
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# SECTION 1 — Constants
# ---------------------------------------------------------------------------
DAILY_PATH = "Raw Data/Daily Term/fcpo_combined_daily.xlsx"
MIN_DIR    = "Raw Data/Minutes Data"

MONTH_CODES = {1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}
CODE_TO_MONTH = {v: k for k, v in MONTH_CODES.items()}

SPREAD_OPTIONS = [
    "+3M/+4M", "+4M/+5M", "+5M/+6M", "+6M/+7M",
    "+7M/+8M", "+8M/+9M", "+9M/+10M", "+10M/+11M",
]

# Tenor column name → offset from front month
TENOR_OFFSETS = {
    "Current": 0, "+1M": 1, "+2M": 2, "+3M": 3, "+4M": 4,
    "+5M": 5, "+6M": 6, "+7M": 7, "+8M": 8, "+9M": 9,
    "+10M": 10, "+11M": 11,
}

DARK_BG   = "#0e1117"
DARK_PLOT = "#262730"
DARK_GRID = "#3a3a4a"
DARK_TEXT = "#fafafa"


# ---------------------------------------------------------------------------
# SECTION 2 — Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_daily_data() -> pd.DataFrame:
    df = pd.read_excel(DAILY_PATH)
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y")
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def load_all_minute_data() -> dict:
    """Returns dict keyed by (year_int, month_int) → DataFrame with columns
    [time, open, high, low, close, date] where time is tz-aware UTC."""
    pattern = re.compile(r"MYX_DLY_FCPO([A-Z])(\d{4}),.*\.csv", re.IGNORECASE)
    result = {}

    for year_folder in sorted(os.listdir(MIN_DIR)):
        folder_path = os.path.join(MIN_DIR, year_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            m = pattern.match(fname)
            if not m:
                continue
            code, year_str = m.group(1).upper(), m.group(2)
            month_int = CODE_TO_MONTH.get(code)
            if month_int is None:
                continue
            year_int = int(year_str)
            fpath = os.path.join(folder_path, fname)
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df["date"] = df["time"].dt.date
            keep = ["time", "open", "high", "low", "close", "date"]
            df = df[[c for c in keep if c in df.columns]]
            result[(year_int, month_int)] = df

    return result


def build_minute_lookup(minute_data: dict) -> dict:
    """Group each contract's DataFrame by date for O(1) lookup.
    Stored in session_state to avoid cache serialisation overhead."""
    lookup = {}
    for key, df in minute_data.items():
        lookup[key] = {d: grp for d, grp in df.groupby("date")}
    return lookup


# ---------------------------------------------------------------------------
# SECTION 3 — Contract Resolution Helpers
# ---------------------------------------------------------------------------
def front_month(date: pd.Timestamp) -> tuple:
    if date.day <= 15:
        return (date.year, date.month)
    else:
        m = date.month + 1
        y = date.year + (1 if m > 12 else 0)
        return (y, m % 12 or 12)


def add_months(ym: tuple, n: int) -> tuple:
    y, m = ym
    m += n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return (y, m)


def resolve_contract(date: pd.Timestamp, tenor_offset: int) -> tuple:
    return add_months(front_month(date), tenor_offset)


# ---------------------------------------------------------------------------
# SECTION 4 — Signal Computation
# ---------------------------------------------------------------------------
def compute_signals(df_daily: pd.DataFrame, col_a: str, col_b: str, params: dict) -> pd.DataFrame:
    z_lookback  = params["z_lookback"]
    ma_short    = params["ma_short"]
    ma_long     = params["ma_long"]
    z_threshold = params["z_threshold"]
    stop_mult   = params["stop_mult"]

    df = df_daily[["date", col_a, col_b]].copy()
    df["spread"] = df[col_a] - df[col_b]

    # Rolling stats
    roll = df["spread"].rolling(z_lookback)
    df["rolling_mean"] = roll.mean()
    df["rolling_std"]  = roll.std()
    df["z_score"]      = (df["spread"] - df["rolling_mean"]) / df["rolling_std"]

    df["MA_short"]  = df["spread"].rolling(ma_short).mean()
    df["MA_long"]   = df["spread"].rolling(ma_long).mean()
    df["momentum"]  = df["spread"] - df["MA_short"]

    df["high20"] = df["spread"].rolling(20).max().shift(1)
    df["low20"]  = df["spread"].rolling(20).min().shift(1)
    df["breakout_up"]   = df["spread"] > df["high20"]
    df["breakout_down"] = df["spread"] < df["low20"]

    df["vol_short"] = df["rolling_std"]
    df["vol_long"]  = df["spread"].rolling(ma_long).std()
    df["vol_expansion"] = df["vol_short"] > df["vol_long"]

    # Regime + signal (vectorised-friendly but easier as apply for clarity)
    regimes = []
    signals = []
    stops   = []

    for _, row in df.iterrows():
        sp  = row["spread"]
        z   = row["z_score"]
        mom = row["momentum"]
        vs  = row["vol_short"]
        ve  = row["vol_expansion"]
        bu  = row["breakout_up"]
        bd  = row["breakout_down"]
        ma_s = row["MA_short"]
        ma_l = row["MA_long"]

        if pd.isna(sp) or pd.isna(z) or pd.isna(vs):
            regimes.append("NEUTRAL")
            signals.append(None)
            stops.append(np.nan)
            continue

        extreme    = abs(z) > z_threshold
        strong_mom = abs(mom) > vs if vs > 0 else False

        if extreme and not strong_mom:
            regime = "MR"
        elif strong_mom and ve:
            regime = "TREND"
        else:
            regime = "NEUTRAL"

        # Entry signal
        signal = None
        stop   = np.nan
        if regime == "MR":
            if z > z_threshold:
                signal = "SHORT"
                stop   = sp + stop_mult * vs
            elif z < -z_threshold:
                signal = "LONG"
                stop   = sp - stop_mult * vs
        elif regime == "TREND":
            if ma_s > ma_l and bu and abs(z) < z_threshold:
                signal = "LONG"
                stop   = sp - stop_mult * vs
            elif ma_s < ma_l and bd and abs(z) < z_threshold:
                signal = "SHORT"
                stop   = sp + stop_mult * vs

        regimes.append(regime)
        signals.append(signal)
        stops.append(stop)

    df["regime"]     = regimes
    df["signal"]     = signals
    df["stop_level"] = stops

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SECTION 5 — Execution Engine
# ---------------------------------------------------------------------------
def get_day_minute_spread(trade_date, contract_a: tuple, contract_b: tuple,
                          minute_lookup: dict):
    """Return intraday spread bars or None if either leg is missing."""
    bars_a = minute_lookup.get(contract_a, {}).get(trade_date)
    bars_b = minute_lookup.get(contract_b, {}).get(trade_date)
    if bars_a is None or bars_b is None:
        return None

    df_a = bars_a.set_index("time")
    df_b = bars_b.set_index("time")
    common = df_a.index.intersection(df_b.index)
    if len(common) == 0:
        return None

    df_a = df_a.loc[common]
    df_b = df_b.loc[common]

    out = pd.DataFrame(index=common)
    out["spread_close"] = df_a["close"] - df_b["close"]
    out["spread_low_worst"]  = df_a["low"]  - df_b["high"]  # worst-case for long
    out["spread_hi_worst"]   = df_a["high"] - df_b["low"]   # worst-case for short
    return out


def run_backtest(df_signals: pd.DataFrame, minute_lookup: dict, params: dict,
                 col_a: str, col_b: str) -> tuple:
    """Core backtest loop. Returns (trades list, equity DataFrame)."""
    max_hold    = params["max_hold"]
    tenor_a     = TENOR_OFFSETS[col_a]
    tenor_b     = TENOR_OFFSETS[col_b]

    trades      = []
    equity_rows = []

    position     = 0      # 0=flat, +1=long, -1=short
    entry_price  = None
    stop_level   = None
    entry_date   = None
    holding_days = 0
    entry_regime = None
    cum_pnl      = 0.0

    rows = df_signals.reset_index(drop=True)
    n    = len(rows)

    for i in range(n - 1):
        row_t  = rows.iloc[i]
        row_t1 = rows.iloc[i + 1]

        exec_date  = row_t1["date"].date() if hasattr(row_t1["date"], "date") else row_t1["date"]
        exec_spread = row_t1["spread"]

        # ---- ENTRY ----
        if position == 0 and row_t["signal"] is not None and not pd.isna(exec_spread):
            position     = +1 if row_t["signal"] == "LONG" else -1
            entry_price  = exec_spread
            stop_level   = row_t["stop_level"]
            entry_date   = exec_date
            holding_days = 0
            entry_regime = row_t["regime"]

        # ---- MANAGE POSITION ----
        if position != 0:
            holding_days += 1
            exit_price  = None
            exit_reason = None

            # A. Intraday stop-loss
            ca = resolve_contract(row_t1["date"], tenor_a)
            cb = resolve_contract(row_t1["date"], tenor_b)
            ms = get_day_minute_spread(exec_date, ca, cb, minute_lookup)
            if ms is not None:
                if position == +1:
                    if (ms["spread_low_worst"] <= stop_level).any():
                        exit_price  = stop_level
                        exit_reason = "STOP_LOSS"
                elif position == -1:
                    if (ms["spread_hi_worst"] >= stop_level).any():
                        exit_price  = stop_level
                        exit_reason = "STOP_LOSS"

            # B–F. EOD exits (if not already stopped out)
            if exit_reason is None and not pd.isna(exec_spread):
                z_now   = row_t1["z_score"]
                mom_now = row_t1["momentum"]
                ma_s    = row_t1["MA_short"]
                ma_l    = row_t1["MA_long"]
                h20     = row_t1["high20"]
                l20     = row_t1["low20"]

                # B. Failed breakout (trend)
                if entry_regime == "TREND":
                    if position == +1 and not pd.isna(h20) and exec_spread < h20:
                        exit_price  = exec_spread
                        exit_reason = "FAILED_BREAKOUT"
                    elif position == -1 and not pd.isna(l20) and exec_spread > l20:
                        exit_price  = exec_spread
                        exit_reason = "FAILED_BREAKOUT"

                # C. Momentum loss
                if exit_reason is None and not pd.isna(mom_now):
                    prev_mom = rows.iloc[i].get("momentum", np.nan)
                    if not pd.isna(prev_mom):
                        if position == +1 and prev_mom > 0 and mom_now < 0:
                            exit_price  = exec_spread
                            exit_reason = "MOMENTUM_LOSS"
                        elif position == -1 and prev_mom < 0 and mom_now > 0:
                            exit_price  = exec_spread
                            exit_reason = "MOMENTUM_LOSS"

                # D. Trend break (MA cross)
                if exit_reason is None and not (pd.isna(ma_s) or pd.isna(ma_l)):
                    prev_ma_s = rows.iloc[i].get("MA_short", np.nan)
                    prev_ma_l = rows.iloc[i].get("MA_long",  np.nan)
                    if not (pd.isna(prev_ma_s) or pd.isna(prev_ma_l)):
                        if position == +1 and prev_ma_s > prev_ma_l and ma_s < ma_l:
                            exit_price  = exec_spread
                            exit_reason = "TREND_BREAK"
                        elif position == -1 and prev_ma_s < prev_ma_l and ma_s > ma_l:
                            exit_price  = exec_spread
                            exit_reason = "TREND_BREAK"

                # E. MR exit — Z crosses zero
                if exit_reason is None and not pd.isna(z_now):
                    prev_z = rows.iloc[i]["z_score"]
                    if not pd.isna(prev_z):
                        if position == +1 and prev_z < 0 and z_now >= 0:
                            exit_price  = exec_spread
                            exit_reason = "MR_EXIT"
                        elif position == -1 and prev_z > 0 and z_now <= 0:
                            exit_price  = exec_spread
                            exit_reason = "MR_EXIT"

                # F. Time stop
                if exit_reason is None and holding_days >= max_hold:
                    exit_price  = exec_spread
                    exit_reason = "TIME_STOP"

            # Record trade close
            if exit_reason is not None and exit_price is not None:
                pnl = (exit_price - entry_price) * position
                cum_pnl += pnl
                trades.append({
                    "entry_date":   entry_date,
                    "exit_date":    exec_date,
                    "direction":    "LONG" if position == +1 else "SHORT",
                    "entry_price":  round(entry_price, 2),
                    "exit_price":   round(exit_price,  2),
                    "pnl":          round(pnl, 2),
                    "holding_days": holding_days,
                    "regime":       entry_regime,
                    "exit_reason":  exit_reason,
                })
                position = 0
                entry_price = stop_level = entry_date = entry_regime = None
                holding_days = 0

        equity_rows.append({"date": row_t1["date"], "cumulative_pnl": cum_pnl,
                             "position": position})

    equity_df = pd.DataFrame(equity_rows)
    return trades, equity_df


# ---------------------------------------------------------------------------
# SECTION 6 — Metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades: list, equity_df: pd.DataFrame) -> dict:
    if not trades:
        return {}

    df_t = pd.DataFrame(trades)
    n    = len(df_t)
    wins = (df_t["pnl"] > 0).sum()

    gross_profit = df_t[df_t["pnl"] > 0]["pnl"].sum()
    gross_loss   = abs(df_t[df_t["pnl"] < 0]["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Daily PnL for Sharpe
    eq = equity_df.set_index("date")["cumulative_pnl"]
    daily_ret = eq.diff().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * math.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

    max_dd = (equity_df["cumulative_pnl"] - equity_df["cumulative_pnl"].cummax()).min()

    # By regime
    by_regime = {}
    for reg in ["MR", "TREND", "NEUTRAL"]:
        sub = df_t[df_t["regime"] == reg]
        if len(sub) == 0:
            continue
        by_regime[reg] = {
            "n":        len(sub),
            "total":    round(sub["pnl"].sum(), 2),
            "win_rate": round((sub["pnl"] > 0).mean() * 100, 1),
        }

    by_exit = df_t["exit_reason"].value_counts().to_dict()

    # Monthly PnL
    df_t["year"]  = pd.to_datetime(df_t["exit_date"]).dt.year
    df_t["month"] = pd.to_datetime(df_t["exit_date"]).dt.month
    monthly = df_t.groupby(["year", "month"])["pnl"].sum().unstack(fill_value=0)
    monthly.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in monthly.columns]

    return {
        "n_trades":      n,
        "n_long":        (df_t["direction"] == "LONG").sum(),
        "n_short":       (df_t["direction"] == "SHORT").sum(),
        "win_rate":      round(wins / n * 100, 1),
        "avg_pnl":       round(df_t["pnl"].mean(), 2),
        "total_pnl":     round(df_t["pnl"].sum(), 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe":        round(sharpe, 2),
        "max_drawdown":  round(max_dd, 2),
        "avg_hold":      round(df_t["holding_days"].mean(), 1),
        "by_regime":     by_regime,
        "by_exit":       by_exit,
        "monthly_pnl":   monthly,
    }


# ---------------------------------------------------------------------------
# SECTION 7 — Chart Builders
# ---------------------------------------------------------------------------
def _regime_shapes(signals_df: pd.DataFrame, yref: str = "paper",
                   y0: float = 0, y1: float = 1) -> list:
    """Return layout shapes for regime color bands."""
    shapes = []
    dates  = signals_df["date"].values
    regs   = signals_df["regime"].values
    colors = {"MR": "rgba(30,120,220,0.08)", "TREND": "rgba(220,50,50,0.08)"}
    i = 0
    while i < len(regs):
        reg = regs[i]
        if reg not in colors:
            i += 1
            continue
        j = i
        while j < len(regs) and regs[j] == reg:
            j += 1
        shapes.append(dict(
            type="rect",
            xref="x", yref=yref,
            x0=str(dates[i])[:10], x1=str(dates[j - 1])[:10],
            y0=y0, y1=y1,
            fillcolor=colors[reg], line_width=0, layer="below",
        ))
        i = j
    return shapes


def build_equity_chart(equity_df: pd.DataFrame, trades: list,
                       signals_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=equity_df["cumulative_pnl"],
        mode="lines", name="Equity",
        line=dict(color="#00e676", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
    ))

    # Entry/exit markers
    for t in trades:
        color  = "#2196f3" if t["direction"] == "LONG" else "#f44336"
        symbol_e = "triangle-up"   if t["direction"] == "LONG" else "triangle-down"
        symbol_x = "triangle-down" if t["direction"] == "LONG" else "triangle-up"
        ep = equity_df[equity_df["date"].astype(str).str[:10] == str(t["entry_date"])]["cumulative_pnl"]
        xp = equity_df[equity_df["date"].astype(str).str[:10] == str(t["exit_date"])]["cumulative_pnl"]
        if not ep.empty:
            fig.add_trace(go.Scatter(
                x=[t["entry_date"]], y=[ep.iloc[0]], mode="markers",
                marker=dict(symbol=symbol_e, size=9, color=color),
                showlegend=False, hovertext=f"Entry {t['direction']} @ {t['entry_price']}",
            ))
        if not xp.empty:
            fig.add_trace(go.Scatter(
                x=[t["exit_date"]], y=[xp.iloc[0]], mode="markers",
                marker=dict(symbol=symbol_x, size=9, color="white",
                            line=dict(color=color, width=1.5)),
                showlegend=False, hovertext=f"Exit {t['exit_reason']} @ {t['exit_price']}",
            ))

    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
        font=dict(color=DARK_TEXT, size=12),
        margin=dict(l=60, r=30, t=30, b=40),
        xaxis=dict(gridcolor=DARK_GRID, showgrid=True),
        yaxis=dict(gridcolor=DARK_GRID, showgrid=True, title="Cumulative PnL (MYR pts)"),
        hovermode="x unified",
        shapes=_regime_shapes(signals_df),
        showlegend=False,
    )
    return fig


def build_signals_chart(signals_df: pd.DataFrame, params: dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.03,
    )

    # Row 1 — Spread + MAs
    fig.add_trace(go.Scatter(
        x=signals_df["date"], y=signals_df["spread"],
        mode="lines", name="Spread", line=dict(color=DARK_TEXT, width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=signals_df["date"], y=signals_df["MA_short"],
        mode="lines", name=f"MA{params['ma_short']}",
        line=dict(color="#ff9800", width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=signals_df["date"], y=signals_df["MA_long"],
        mode="lines", name=f"MA{params['ma_long']}",
        line=dict(color="#2196f3", width=1, dash="dot"),
    ), row=1, col=1)

    # Row 2 — Z-score bars
    z = signals_df["z_score"]
    colors = ["#f44336" if v >= 0 else "#2196f3" for v in z.fillna(0)]
    fig.add_trace(go.Bar(
        x=signals_df["date"], y=z, name="Z-score",
        marker_color=colors, opacity=0.7,
    ), row=2, col=1)

    thr = params["z_threshold"]
    for lvl, dash in [(thr, "dash"), (-thr, "dash"), (0, "dot")]:
        fig.add_hline(y=lvl, line=dict(color="white", width=0.8, dash=dash), row=2, col=1)

    # Regime shapes
    shapes = _regime_shapes(signals_df, yref="y", y0=signals_df["spread"].min(),
                             y1=signals_df["spread"].max())
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
        font=dict(color=DARK_TEXT, size=12),
        margin=dict(l=60, r=30, t=30, b=40),
        xaxis2=dict(gridcolor=DARK_GRID),
        yaxis=dict(gridcolor=DARK_GRID, title="Spread"),
        yaxis2=dict(gridcolor=DARK_GRID, title="Z-score"),
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.02),
        shapes=shapes,
    )
    return fig


# ---------------------------------------------------------------------------
# SECTION 8 — Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="FCPO Spread Backtest", layout="wide")
    st.title("FCPO Calendar Spread Backtest")

    # --- Load data (cached) ---
    df_daily    = load_daily_data()
    minute_data = load_all_minute_data()

    if "minute_lookup" not in st.session_state:
        st.session_state["minute_lookup"] = build_minute_lookup(minute_data)
    minute_lookup = st.session_state["minute_lookup"]

    # --- Sidebar parameters ---
    with st.sidebar:
        st.header("Parameters")
        spread_pair = st.selectbox("Spread Pair", SPREAD_OPTIONS)
        date_min = df_daily["date"].min().date()
        date_max = df_daily["date"].max().date()
        start_date = st.date_input("Start Date", value=pd.Timestamp("2024-01-01").date(),
                                   min_value=date_min, max_value=date_max)
        end_date   = st.date_input("End Date",   value=pd.Timestamp("2026-03-17").date(),
                                   min_value=date_min, max_value=date_max)
        z_lookback  = st.selectbox("Z-score Lookback", [20, 60, 120], index=1)
        z_threshold = st.slider("Entry Z Threshold",    1.0, 3.5, 2.0, 0.1)
        stop_mult   = st.slider("Stop Multiplier × std", 0.5, 3.0, 1.5, 0.1)
        max_hold    = st.slider("Max Holding Days",     3, 40, 15)
        ma_short    = st.slider("Short MA",             5, 60, 20)
        ma_long     = st.slider("Long MA",             20, 200, 60)
        run_btn     = st.button("Run Backtest", type="primary", use_container_width=True)

    params = dict(
        z_lookback=z_lookback, z_threshold=z_threshold, stop_mult=stop_mult,
        max_hold=max_hold, ma_short=ma_short, ma_long=ma_long,
    )

    # --- Run backtest on button click ---
    if run_btn:
        col_a, col_b = spread_pair.split("/")
        col_a = col_a.strip()
        col_b = col_b.strip()

        if col_a not in df_daily.columns or col_b not in df_daily.columns:
            st.error(f"Columns {col_a} or {col_b} not found in daily data.")
            st.stop()

        with st.spinner("Computing signals…"):
            signals_full = compute_signals(df_daily, col_a, col_b, params)

        # Slice to date range AFTER computing on full history (avoids warm-up NaN)
        mask = (signals_full["date"].dt.date >= start_date) & \
               (signals_full["date"].dt.date <= end_date)
        signals_slice = signals_full[mask].reset_index(drop=True)

        with st.spinner("Running backtest…"):
            trades, equity_df = run_backtest(
                signals_slice, minute_lookup, params, col_a, col_b
            )

        metrics = compute_metrics(trades, equity_df)

        st.session_state["backtest_results"] = {
            "trades":     trades,
            "equity_df":  equity_df,
            "metrics":    metrics,
            "signals_df": signals_slice,
            "params":     params,
            "spread_pair": spread_pair,
            "col_a": col_a,
            "col_b": col_b,
        }

    # --- Render tabs ---
    results = st.session_state.get("backtest_results")

    tab_results, tab_metrics, tab_signals = st.tabs(["Results", "Metrics", "Signals"])

    with tab_results:
        if results is None:
            st.info("Configure parameters and click **Run Backtest** to begin.")
        elif not results["trades"]:
            st.info("No trades generated with current parameters. Try adjusting the Z threshold or date range.")
        else:
            st.subheader(f"Equity Curve — {results['spread_pair']}")
            fig_eq = build_equity_chart(results["equity_df"], results["trades"], results["signals_df"])
            st.plotly_chart(fig_eq, use_container_width=True)

            st.subheader("Trade Log")
            df_trades = pd.DataFrame(results["trades"])
            st.dataframe(df_trades, use_container_width=True, hide_index=True)

            csv = df_trades.to_csv(index=False).encode("utf-8")
            st.download_button("Download Trade Log CSV", csv, "trades.csv", "text/csv")

    with tab_metrics:
        if results is None:
            st.info("Run the backtest first.")
        elif not results["trades"]:
            st.info("No trades to compute metrics.")
        else:
            m = results["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades",    m["n_trades"])
            c2.metric("Win Rate",        f"{m['win_rate']}%")
            c3.metric("Total PnL",       f"{m['total_pnl']:,.0f}")
            c4.metric("Avg PnL/Trade",   f"{m['avg_pnl']:,.1f}")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Profit Factor",   m["profit_factor"])
            c6.metric("Sharpe",          m["sharpe"])
            c7.metric("Max Drawdown",    f"{m['max_drawdown']:,.0f}")
            c8.metric("Avg Hold (days)", m["avg_hold"])

            st.subheader("Monthly PnL")
            monthly = m["monthly_pnl"]
            st.dataframe(
                monthly.style.background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
            )

            col_r, col_e = st.columns(2)
            with col_r:
                st.subheader("By Regime")
                rows_r = []
                for reg, stats in m["by_regime"].items():
                    rows_r.append({"Regime": reg, **stats})
                if rows_r:
                    st.dataframe(pd.DataFrame(rows_r), hide_index=True)

            with col_e:
                st.subheader("By Exit Reason")
                df_ex = pd.DataFrame(
                    [{"Exit Reason": k, "Count": v} for k, v in m["by_exit"].items()]
                )
                st.dataframe(df_ex, hide_index=True)

    with tab_signals:
        if results is None:
            st.info("Run the backtest first.")
        else:
            st.subheader(f"Signal Chart — {results['spread_pair']}")
            fig_sig = build_signals_chart(results["signals_df"], results["params"])
            st.plotly_chart(fig_sig, use_container_width=True)

            with st.expander("Raw Signal Data"):
                st.dataframe(results["signals_df"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
