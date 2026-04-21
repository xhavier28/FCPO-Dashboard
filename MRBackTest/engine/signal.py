"""
engine/signal.py — Core signal engine shared by WF1 (daily) and WF2 (15-min).

Runs Kalman → rolling z-score → bar-by-bar trade simulation.
"""

import numpy as np
import pandas as pd

from shared.kalman import run_kalman


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _annualised_sharpe(returns: pd.Series, bars_per_year: float) -> float:
    """Annualised Sharpe from a bar-return series."""
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (absolute, same units as equity)."""
    roll_max = equity.cummax()
    dd = equity - roll_max
    return float(dd.min())


def _max_drawdown_pct(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction."""
    roll_max = equity.cummax()
    dd_pct   = (equity - roll_max) / roll_max.replace(0, np.nan)
    val = dd_pct.min()
    return float(val) if not np.isnan(val) else float("nan")


def _calmar(net_pnl_total: float, max_dd: float, years: float) -> float:
    if max_dd == 0 or years == 0:
        return float("nan")
    ann_return = net_pnl_total / years
    return float(ann_return / abs(max_dd))


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_signal_engine(
    log_y: pd.Series,
    log_x: pd.Series,
    kalman_delta: float,
    Ve: float,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    lookback: int,
    lot_size: float,
    roundtrip_cost: float,
    bars_per_year: float = 252.0,
    max_hold_multiplier: float = 2.0,
    half_life_bars: float = None,
) -> dict:
    """
    Run the full Kalman + rolling-z signal simulation.

    Parameters
    ----------
    log_y, log_x      : Log-price series with a shared DatetimeIndex
    kalman_delta      : Kalman process noise (1e-5 to 1e-3)
    Ve                : Kalman observation noise variance
    entry_z           : Z-score entry threshold (e.g. 2.0)
    exit_z            : Z-score exit threshold (e.g. 0.5)
    stop_z            : Z-score emergency stop threshold (e.g. 4.0)
    lookback          : Rolling z-score window (bars)
    lot_size          : Lots per trade (MYR value = lot_size × contract_value)
    roundtrip_cost    : Round-trip cost per trade in traded currency
    bars_per_year     : 252 for daily; 252×26 for 15-min
    max_hold_multiplier: Force exit after this multiple of half_life_bars
    half_life_bars    : OU half-life in bars (None → no max-hold)

    Returns
    -------
    dict with:
        trades        : list of trade dicts
        equity_curve  : pd.Series (cumulative net PnL)
        metrics       : dict (Sharpe, MDD, win_rate, Calmar)
        spread        : np.array (Kalman spread_reconstructed)
        z_score       : np.array (rolling z)
        beta_t        : np.array (time-varying hedge ratio)
    """
    assert len(log_y) == len(log_x), "log_y and log_x must have equal length"
    n = len(log_y)

    # Step 1: Kalman filter (inputs are already log prices)
    y_arr = np.asarray(log_y, dtype=float)
    x_arr = np.asarray(log_x, dtype=float)
    k     = run_kalman(y_arr, x_arr, delta=kalman_delta, Ve=Ve, space="raw")
    # space="raw" because log-transform already applied by caller

    spread = k["spread_reconstructed"]
    beta_t = k["beta_t"]

    # Step 2: Rolling z-score
    s      = pd.Series(spread)
    roll_m = s.rolling(lookback, min_periods=lookback // 2).mean()
    roll_s = s.rolling(lookback, min_periods=lookback // 2).std()
    z_arr  = ((s - roll_m) / roll_s.replace(0, np.nan)).fillna(0).values

    # Step 3: Bar-by-bar simulation
    max_hold = int(max_hold_multiplier * half_life_bars) if half_life_bars else None

    trades       = []
    equity_curve = np.zeros(n)

    position     = 0       # +1 = LONG spread, -1 = SHORT spread, 0 = flat
    entry_bar    = None
    entry_z_val  = None
    entry_spread = None
    bars_held    = 0
    cum_pnl      = 0.0

    dates = log_y.index if hasattr(log_y, "index") else pd.RangeIndex(n)

    for t in range(n):
        z = float(z_arr[t])
        current_spread = float(spread[t])

        if position != 0:
            bars_held += 1

        # Exit logic (check before entry)
        if position != 0:
            exit_reason = None

            # Exit: z mean-reverts past the exit threshold
            if position == 1 and z >= -exit_z:
                exit_reason = "exit"
            elif position == -1 and z <= exit_z:
                exit_reason = "exit"

            # Stop: z moves further against us
            if position == 1 and z < -stop_z:
                exit_reason = "stop"
            elif position == -1 and z > stop_z:
                exit_reason = "stop"

            # Max hold
            if max_hold and bars_held >= max_hold:
                exit_reason = "max_hold"

            if exit_reason:
                gross_pnl = position * lot_size * (current_spread - entry_spread)
                net_pnl   = gross_pnl - roundtrip_cost
                cum_pnl  += net_pnl

                trades.append({
                    "entry_dt":    str(dates[entry_bar]) if hasattr(dates[entry_bar], "__str__") else entry_bar,
                    "exit_dt":     str(dates[t]) if hasattr(dates[t], "__str__") else t,
                    "entry_bar":   entry_bar,
                    "exit_bar":    t,
                    "bars_held":   bars_held,
                    "direction":   "LONG" if position == 1 else "SHORT",
                    "entry_z":     round(entry_z_val, 4),
                    "exit_z":      round(z, 4),
                    "exit_reason": exit_reason,
                    "gross_pnl":   round(gross_pnl, 2),
                    "net_pnl":     round(net_pnl, 2),
                })

                position     = 0
                entry_bar    = None
                entry_z_val  = None
                entry_spread = None
                bars_held    = 0

        # Entry logic
        if position == 0:
            if z < -entry_z:
                position     = 1   # LONG spread
                entry_bar    = t
                entry_z_val  = z
                entry_spread = current_spread
                bars_held    = 0
            elif z > entry_z:
                position     = -1  # SHORT spread
                entry_bar    = t
                entry_z_val  = z
                entry_spread = current_spread
                bars_held    = 0

        equity_curve[t] = cum_pnl

    equity_s = pd.Series(equity_curve, index=dates)
    bar_returns = equity_s.diff().fillna(0)

    # Metrics
    n_trades  = len(trades)
    n_wins    = sum(1 for tr in trades if tr["net_pnl"] > 0)
    win_rate  = n_wins / n_trades if n_trades > 0 else float("nan")
    sharpe    = _annualised_sharpe(bar_returns, bars_per_year)
    mdd       = _max_drawdown(equity_s)
    years     = n / bars_per_year
    calmar    = _calmar(cum_pnl, mdd, years)

    avg_bars_held = (
        float(np.mean([tr["bars_held"] for tr in trades]))
        if trades else float("nan")
    )

    metrics = {
        "n_trades":       n_trades,
        "n_wins":         n_wins,
        "win_rate":       round(win_rate, 4) if not np.isnan(win_rate) else None,
        "total_net_pnl":  round(cum_pnl, 2),
        "sharpe":         round(sharpe, 4) if not np.isnan(sharpe) else None,
        "max_drawdown":   round(mdd, 2),
        "calmar":         round(calmar, 4) if not np.isnan(calmar) else None,
        "avg_bars_held":  round(avg_bars_held, 1) if not np.isnan(avg_bars_held) else None,
        "bars_per_year":  bars_per_year,
        "n_bars":         n,
        "years":          round(years, 2),
    }

    return {
        "trades":       trades,
        "equity_curve": equity_s,
        "metrics":      metrics,
        "spread":       spread,
        "z_score":      z_arr,
        "beta_t":       beta_t,
    }
