"""
engine/wf2.py — WF2 15-min intraday pipeline.

WF2 inherits FX rates from WF1. No break exclusion (deliberate — stress test).
70/30 train/test split. Grid search of 12 combinations.
"""

import io
import warnings

import numpy as np
import pandas as pd

from shared.kalman import run_kalman
from shared.ou import fit_ou
from shared.fx_converter import apply_fx_conversion, rate_table_from_editor
from shared.structural_break import check_fx_applied
from engine.signal import run_signal_engine

BARS_PER_DAY  = 26.0       # 15-min bars per trading day (6.5h × 4)
BARS_PER_YEAR = 252.0 * BARS_PER_DAY

FIXED_KALMAN_DELTA = 1e-4
FIXED_VE           = 0.1
FIXED_STOP_Z       = 4.0

# 12-row grid: entry_z × exit_z × lookback (tighter range for intraday)
ENTRY_Z_CANDIDATES  = [1.5, 2.0, 2.5]
EXIT_Z_CANDIDATES   = [0.25, 0.5]
LOOKBACK_CANDIDATES = [60, 120]   # bars (~2.3h and ~4.6h)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_csv(source) -> pd.DataFrame:
    if isinstance(source, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(source))
    return pd.read_csv(source)


def _parse_intraday(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Parse a 15-min bar CSV.
    Expects a datetime column (Date, Time, Datetime) + Close column.
    """
    df.columns = [c.strip() for c in df.columns]

    date_col  = next((c for c in df.columns if c.lower() in ("date", "time", "datetime")), None)
    close_col = next((c for c in df.columns if c.lower() == "close"), None)

    if date_col is None:
        raise ValueError(f"[{label}] No date column. Columns: {list(df.columns)}")
    if close_col is None:
        raise ValueError(f"[{label}] No 'Close' column. Columns: {list(df.columns)}")

    s = df[[date_col, close_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col]).set_index(date_col)[close_col].sort_index()
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = label
    return s


def load_and_prep_wf2(
    bytes_y,
    bytes_x,
    rate_table_df: pd.DataFrame,
    mult_y: float = 1.0,
    mult_x: float = 1.0,
) -> dict:
    """
    Load WF2 15-min CSVs, apply multipliers, FX-convert Asset B.

    Returns dict with: y, x, date_range, n_bars, fx_log, warnings
    """
    df_y = _load_csv(bytes_y)
    df_x = _load_csv(bytes_x)

    raw_y = _parse_intraday(df_y, "Asset A")
    raw_x = _parse_intraday(df_x, "Asset B")

    raw_y = raw_y * mult_y
    raw_x = raw_x * mult_x

    # Align
    common = raw_y.index.intersection(raw_x.index)
    y = raw_y.loc[common]
    x = raw_x.loc[common]

    rate_table = rate_table_from_editor(rate_table_df)
    x_conv, fx_log = apply_fx_conversion(x, rate_table, label_x="Asset B")

    fx_guard = check_fx_applied(y.values, x_conv.values, fx_was_applied=True)

    warns = []
    if fx_guard == "blocked":
        warns.append("15-min: Price ratio > 20x after FX conversion.")
    elif fx_guard == "warning":
        warns.append("15-min: Price ratio 5-20x — verify rates.")

    return {
        "y":          y,
        "x":          x_conv,
        "x_raw":      x,
        "rate_table": rate_table,
        "date_range": (str(y.index[0]), str(y.index[-1])),
        "n_bars":     len(y),
        "fx_log":     fx_log,
        "fx_guard":   fx_guard,
        "warnings":   warns,
    }


# ---------------------------------------------------------------------------
# Rate inheritance from WF1
# ---------------------------------------------------------------------------

def inherit_rates(wf2_dates: pd.DatetimeIndex, wf1_rate_table: pd.DataFrame) -> tuple:
    """
    Verify that all months in WF2 dates are covered by WF1 rate table.

    Returns (rate_table, list_of_warning_strings).
    """
    wf2_months   = pd.DatetimeIndex(wf2_dates).to_period("M").unique()
    wf1_months   = set(wf1_rate_table.index)
    missing      = [str(m) for m in sorted(wf2_months) if m not in wf1_months]

    warns = []
    if missing:
        warns.append(
            f"WF2 covers months not in WF1 rate table: {', '.join(missing)}. "
            "Fallback (mean rate) will be used for those months."
        )

    return wf1_rate_table, warns


# ---------------------------------------------------------------------------
# Train / Test split (70 / 30)
# ---------------------------------------------------------------------------

def split_wf2(y: pd.Series, x: pd.Series) -> dict:
    """70/30 chronological split. Returns dict with train/test keys."""
    n  = len(y)
    i1 = int(n * 0.70)
    return {
        "train_y":     y.iloc[:i1],
        "train_x":     x.iloc[:i1],
        "test_y":      y.iloc[i1:],
        "test_x":      x.iloc[i1:],
        "train_dates": y.index[:i1],
        "test_dates":  y.index[i1:],
    }


# ---------------------------------------------------------------------------
# WF2 grid search (12 rows)
# ---------------------------------------------------------------------------

def run_wf2_grid(
    y_tv: pd.Series,
    x_tv: pd.Series,
    wf1_entry_z: float,
    wf1_exit_z: float,
    wf1_stop_z: float,
    half_life_bars: float,
    lot_size: float,
    roundtrip_cost: float,
) -> pd.DataFrame:
    """
    12-row grid search on WF2 train set.
    Also includes WF1 locked params as an extra reference row.
    """
    log_y_s = pd.Series(np.log(y_tv.values), index=y_tv.index)
    log_x_s = pd.Series(np.log(x_tv.values), index=x_tv.index)

    rows = []
    for entry_z in ENTRY_Z_CANDIDATES:
        for exit_z in EXIT_Z_CANDIDATES:
            for lookback in LOOKBACK_CANDIDATES:
                try:
                    result = run_signal_engine(
                        log_y=log_y_s,
                        log_x=log_x_s,
                        kalman_delta=FIXED_KALMAN_DELTA,
                        Ve=FIXED_VE,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        stop_z=FIXED_STOP_Z,
                        lookback=lookback,
                        lot_size=lot_size,
                        roundtrip_cost=roundtrip_cost,
                        bars_per_year=BARS_PER_YEAR,
                        half_life_bars=half_life_bars,
                    )
                    m = result["metrics"]
                    rows.append({
                        "entry_z":    entry_z,
                        "exit_z":     exit_z,
                        "lookback":   lookback,
                        "n_trades":   m["n_trades"],
                        "win_rate":   m["win_rate"],
                        "sharpe":     m["sharpe"],
                        "calmar":     m["calmar"],
                        "net_pnl":    m["total_net_pnl"],
                        "note":       "",
                    })
                except Exception as e:
                    rows.append({
                        "entry_z": entry_z, "exit_z": exit_z, "lookback": lookback,
                        "n_trades": 0, "win_rate": None, "sharpe": None,
                        "calmar": None, "net_pnl": None, "note": str(e),
                    })

    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# WF2 test run
# ---------------------------------------------------------------------------

def run_wf2_test(
    y_test: pd.Series,
    x_test: pd.Series,
    locked_params: dict,
    lot_size: float,
    roundtrip_cost: float,
) -> dict:
    """
    Run signal engine on WF2 test slice with locked intraday params.

    locked_params: entry_z, exit_z, lookback, half_life_bars
    """
    log_y_s = pd.Series(np.log(y_test.values), index=y_test.index)
    log_x_s = pd.Series(np.log(x_test.values), index=x_test.index)

    result = run_signal_engine(
        log_y=log_y_s,
        log_x=log_x_s,
        kalman_delta=FIXED_KALMAN_DELTA,
        Ve=FIXED_VE,
        entry_z=locked_params["entry_z"],
        exit_z=locked_params["exit_z"],
        stop_z=FIXED_STOP_Z,
        lookback=locked_params["lookback"],
        lot_size=lot_size,
        roundtrip_cost=roundtrip_cost,
        bars_per_year=BARS_PER_YEAR,
        half_life_bars=locked_params.get("half_life_bars"),
    )

    m = result["metrics"]
    if m["total_net_pnl"] > 0 and m["sharpe"] is not None and m["sharpe"] > 0:
        gate = "GO"
    elif m["sharpe"] is not None and m["sharpe"] > -0.5:
        gate = "MARGINAL"
    else:
        gate = "NO-GO"

    result["gate"] = gate
    return result


# ---------------------------------------------------------------------------
# TT Auto Spreader settings
# ---------------------------------------------------------------------------

def compute_tt_settings(
    kalman_result: dict,
    ou_result: dict,
    wf1_locked_params: dict,
    wf2_locked_params: dict,
    last_fcpo_price: float,
    last_soy_price: float,
    lot_size: float,
) -> dict:
    """
    Compute TT Auto Spreader configuration values.

    Lot ratio: round(HR × (25 / 27.22)) displayed as "X A : 1 B"
    HR = median beta_t from the Kalman filter.
    """
    beta_arr = kalman_result.get("beta_t", np.array([1.0]))
    hr       = float(np.nanmedian(beta_arr))

    raw_ratio  = hr * (25.0 / 27.22)
    lot_ratio  = max(1, round(raw_ratio))

    half_life_bars   = ou_result.get("half_life_bars", None)
    half_life_hours  = ou_result.get("half_life_hours", None)
    ou_std           = ou_result.get("ou_std", None)

    return {
        "hedge_ratio":       round(hr, 4),
        "lot_ratio_raw":     round(raw_ratio, 3),
        "lot_ratio":         lot_ratio,
        "lot_ratio_display": f"{lot_ratio} A : 1 B",
        "half_life_bars":    half_life_bars,
        "half_life_hours":   half_life_hours,
        "ou_std_log":        ou_std,
        "entry_z_wf1":       wf1_locked_params.get("entry_z"),
        "exit_z_wf1":        wf1_locked_params.get("exit_z"),
        "lookback_wf1":      wf1_locked_params.get("lookback"),
        "entry_z_wf2":       wf2_locked_params.get("entry_z"),
        "exit_z_wf2":        wf2_locked_params.get("exit_z"),
        "lookback_wf2":      wf2_locked_params.get("lookback"),
        "stop_z":            FIXED_STOP_Z,
        "lot_size":          lot_size,
        "last_price_a":      round(last_fcpo_price, 2),
        "last_price_b":      round(last_soy_price, 2),
    }
