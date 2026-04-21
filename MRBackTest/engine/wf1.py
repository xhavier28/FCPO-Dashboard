"""
engine/wf1.py — WF1 daily pipeline orchestration.

Steps:
  1. Load + prep daily CSVs, apply multipliers + FX conversion
  2. Structural break detection (user-togglable)
  3. Exclusion application
  4. Chronological 60/20/20 train/val/test split
  5. Gating tests (6 tests on train set)
  6. Rolling cointegration + Hurst (time stability)
  7. Grid search on val set (36 combinations)
  8. Lock best params → run test slice
  9. Regime breakdown + kill-switch detection
"""

import io
import warnings

import numpy as np
import pandas as pd

from shared.kalman import run_kalman
from shared.ou import fit_ou
from shared.structural_break import detect_breaks, breaks_to_dicts, apply_exclusions, check_fx_applied
from shared.fx_converter import (
    build_empty_rate_table, apply_fx_conversion, rate_table_from_editor,
)
from tests.adf_kpss import test_stationarity
from tests.coint_eg import test_cointegration_eg
from tests.hurst import hurst_exponent
from tests.johansen import test_cointegration_johansen
from engine.signal import run_signal_engine


BARS_PER_YEAR = 252.0

# Grid search candidates
ENTRY_Z_CANDIDATES = [1.5, 2.0, 2.5]
EXIT_Z_CANDIDATES  = [0.25, 0.5, 0.75]
LOOKBACK_CANDIDATES = [30, 60, 90, 120]

FIXED_STOP_Z   = 4.0
FIXED_KALMAN_DELTA = 1e-4
FIXED_VE       = 0.1


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _load_csv(source) -> pd.DataFrame:
    """Load CSV bytes or path → DataFrame."""
    if isinstance(source, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(source))
    else:
        df = pd.read_csv(source)
    return df


def _parse_price_series(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Extract a daily Close price Series from an uploaded DataFrame.
    Expects a 'Date' column (various formats) and a 'Close' column.
    Returns a pd.Series with DatetimeIndex, sorted ascending, NaN dropped.
    """
    df.columns = [c.strip() for c in df.columns]

    # Find date column
    date_col  = next((c for c in df.columns if c.lower() in ("date", "time", "datetime")), None)
    close_col = next((c for c in df.columns if c.lower() == "close"), None)

    if date_col is None:
        raise ValueError(f"[{label}] No 'Date' column found. Columns: {list(df.columns)}")
    if close_col is None:
        raise ValueError(f"[{label}] No 'Close' column found. Columns: {list(df.columns)}")

    s = df[[date_col, close_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    s = s.set_index(date_col)[close_col].sort_index()
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = label
    return s


def load_and_prep_wf1(
    bytes_y,
    bytes_x,
    rate_table_df: pd.DataFrame,
    mult_y: float = 1.0,
    mult_x: float = 1.0,
) -> dict:
    """
    Load WF1 daily CSVs, apply multipliers, FX-convert Asset B to base currency.

    Parameters
    ----------
    bytes_y       : CSV bytes for Asset A
    bytes_x       : CSV bytes for Asset B
    rate_table_df : st.data_editor output with columns [Month, USDMYR_Rate]
    mult_y        : price multiplier for Asset A (default 1.0)
    mult_x        : price multiplier for Asset B (default 1.0)

    Returns
    -------
    dict with: y, x, rate_table, rate_table_df, date_range, n_bars, fx_log, warnings
    """
    df_y = _load_csv(bytes_y)
    df_x = _load_csv(bytes_x)

    raw_y = _parse_price_series(df_y, "Asset A")
    raw_x = _parse_price_series(df_x, "Asset B")

    # Apply multipliers
    raw_y = raw_y * mult_y
    raw_x = raw_x * mult_x

    # Align on common dates
    common = raw_y.index.intersection(raw_x.index)
    y = raw_y.loc[common]
    x = raw_x.loc[common]

    # Build rate table from editor
    rate_table = rate_table_from_editor(rate_table_df)

    # FX conversion (generic bar-by-bar multiplication by monthly rate)
    x_conv, fx_log = apply_fx_conversion(x, rate_table, label_x="Asset B")

    # FX guard
    fx_guard = check_fx_applied(y.values, x_conv.values, fx_was_applied=True)

    warns = []
    if fx_guard == "blocked":
        warns.append("Price ratio > 20x after FX conversion. Check rates.")
    elif fx_guard == "warning":
        warns.append("Price ratio 5-20x — verify FX rates.")

    return {
        "y":            y,
        "x":            x_conv,
        "x_raw":        x,
        "rate_table":   rate_table,
        "rate_table_df": rate_table_df,
        "date_range":   (str(y.index[0].date()), str(y.index[-1].date())),
        "n_bars":       len(y),
        "fx_log":       fx_log,
        "fx_guard":     fx_guard,
        "warnings":     warns,
    }


# ---------------------------------------------------------------------------
# Structural breaks
# ---------------------------------------------------------------------------

def run_structural_break_step(y: pd.Series, x: pd.Series) -> list:
    """
    Detect structural breaks in (y, x) using all three methods.

    Returns list of break dicts (with 'enabled' key = True by default).
    """
    raw_breaks = detect_breaks(
        y.values, x.values,
        window=30, threshold=3.5, extend_bars=30, min_gap=20,
        reversion_window=10, reversion_threshold=2.0,
    )
    return breaks_to_dicts(raw_breaks, y.index)


def apply_user_exclusions(y: pd.Series, x: pd.Series, periods: list) -> tuple:
    """Apply user-toggled exclusion periods. Returns (y_clean, x_clean, summary)."""
    return apply_exclusions(y, x, periods)


# ---------------------------------------------------------------------------
# Train / Val / Test split (60 / 20 / 20 chronological)
# ---------------------------------------------------------------------------

def split_wf1(y: pd.Series, x: pd.Series) -> dict:
    """
    Chronological 60/20/20 split.

    Returns dict with keys: train_y, train_x, val_y, val_x, test_y, test_x,
                            train_dates, val_dates, test_dates
    """
    n      = len(y)
    i1     = int(n * 0.60)
    i2     = int(n * 0.80)

    splits = {}
    for name, sl in [("train", slice(0, i1)), ("val", slice(i1, i2)), ("test", slice(i2, n))]:
        splits[f"{name}_y"]     = y.iloc[sl]
        splits[f"{name}_x"]     = x.iloc[sl]
        splits[f"{name}_dates"] = y.index[sl]
    return splits


# ---------------------------------------------------------------------------
# Gating tests (run on training set)
# ---------------------------------------------------------------------------

def run_gating_tests(y_train: pd.Series, x_train: pd.Series) -> dict:
    """
    Run all 6 gating tests on the training set.

    Tests:
      1. ADF/KPSS — Y is I(1)
      2. ADF/KPSS — X is I(1)
      3. Engle-Granger cointegration
      4. Johansen cointegration
      5. Hurst exponent on EG static spread
      6. OU fit on Kalman spread_reconstructed

    Returns dict with test results and overall pass/fail.
    """
    results = {}

    # 1+2: Unit root tests
    results["adf_y"] = test_stationarity(y_train, "Asset A")
    results["adf_x"] = test_stationarity(x_train, "Asset B")

    # 3: Engle-Granger
    results["eg"] = test_cointegration_eg(y_train, x_train)

    # 4: Johansen
    results["johansen"] = test_cointegration_johansen(y_train, x_train, freq="daily")

    # 5: Hurst (on EG static spread)
    beta_s  = results["eg"]["beta_static"]
    alpha_s = results["eg"]["alpha_static"]
    spread_static = y_train.values - beta_s * x_train.values - alpha_s
    results["hurst"] = hurst_exponent(spread_static)

    # 6: OU (on Kalman spread)
    k = run_kalman(
        np.log(y_train.values), np.log(x_train.values),
        delta=FIXED_KALMAN_DELTA, Ve=FIXED_VE, space="raw"
    )
    ou = fit_ou(k["spread_reconstructed"], freq="daily", bars_per_day=1.0, space="log")
    results["ou"] = ou

    # Pass/Fail logic
    pass_adf_y   = results["adf_y"]["is_I1"]
    pass_adf_x   = results["adf_x"]["is_I1"]
    pass_eg      = results["eg"]["eg_pvalue"] < 0.10   # borderline passes
    pass_johansen = results["johansen"]["verdict"] in ("cointegrated", "borderline")
    pass_hurst   = results["hurst"].get("tradeable", False)
    pass_ou      = results["ou"].get("verdict") in ("tradeable", "borderline")

    passes = {
        "adf_y":    pass_adf_y,
        "adf_x":    pass_adf_x,
        "eg":       pass_eg,
        "johansen": pass_johansen,
        "hurst":    pass_hurst,
        "ou":       pass_ou,
    }

    n_pass      = sum(passes.values())
    all_pass    = all(passes.values())
    gate_result = "PASS" if all_pass else ("MARGINAL" if n_pass >= 4 else "BLOCKED")

    results["passes"]       = passes
    results["n_pass"]       = n_pass
    results["gate_result"]  = gate_result
    results["ou_std_log"]   = ou.get("ou_std")
    results["half_life_bars"] = ou.get("half_life_bars")

    return results


# ---------------------------------------------------------------------------
# Rolling tests (time stability checks)
# ---------------------------------------------------------------------------

def run_rolling_coint(y: pd.Series, x: pd.Series, window: int = 120) -> pd.Series:
    """
    Rolling cointegration p-values (EG) over windows of `window` bars.
    Returns a Series of p-values indexed by the end-date of each window.
    """
    from statsmodels.tsa.stattools import coint

    n = len(y)
    pvals = {}
    for i in range(window, n + 1):
        y_w = y.iloc[i - window: i]
        x_w = x.iloc[i - window: i]
        try:
            _, p, _ = coint(y_w, x_w, trend="c", maxlag=5)
        except Exception:
            p = float("nan")
        pvals[y.index[i - 1]] = p

    return pd.Series(pvals, name="coint_pval")


def run_rolling_hurst(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling Hurst exponent over windows of `window` bars.
    Returns a Series of Hurst values indexed by end-date.
    """
    n = len(spread)
    hursts = {}
    for i in range(window, n + 1):
        chunk = spread.iloc[i - window: i]
        h = hurst_exponent(chunk)
        hursts[spread.index[i - 1]] = h.get("hurst", float("nan"))

    return pd.Series(hursts, name="hurst_rolling")


def check_rolling_test_fail(pvals: pd.Series, hursts: pd.Series,
                             coint_threshold: float = 0.20,
                             hurst_threshold: float = 0.20) -> dict:
    """
    FAIL if > 20% of rolling windows violate the threshold.
    coint: p > 0.20 → bad window
    hurst: H > 0.50 → bad window
    """
    n_coint     = len(pvals.dropna())
    n_hurst     = len(hursts.dropna())
    bad_coint   = (pvals.dropna() > 0.20).sum()
    bad_hurst   = (hursts.dropna() > 0.50).sum()
    fail_coint  = (bad_coint / n_coint > coint_threshold) if n_coint > 0 else True
    fail_hurst  = (bad_hurst / n_hurst > hurst_threshold) if n_hurst > 0 else True

    return {
        "coint_bad_pct":  round(bad_coint / n_coint, 3) if n_coint > 0 else None,
        "hurst_bad_pct":  round(bad_hurst / n_hurst, 3) if n_hurst > 0 else None,
        "fail_coint":     fail_coint,
        "fail_hurst":     fail_hurst,
        "overall_pass":   not fail_coint and not fail_hurst,
    }


# ---------------------------------------------------------------------------
# Cost threshold check
# ---------------------------------------------------------------------------

def cost_threshold_ok(
    entry_z: float,
    exit_z: float,
    roundtrip_cost: float,
    ou_std_log: float,
    last_fcpo_price: float,
) -> bool:
    """
    Returns True if entry_z is large enough to overcome costs.
    ou_std_price = ou_std_log × last_fcpo_price (approximate MYR scale)
    Threshold: entry_z > exit_z + 3 × (roundtrip_cost / ou_std_price)
    """
    if ou_std_log is None or ou_std_log == 0 or last_fcpo_price == 0:
        return True  # can't check, allow
    ou_std_price = ou_std_log * last_fcpo_price
    min_z = exit_z + 3.0 * (roundtrip_cost / ou_std_price)
    return entry_z > min_z


# ---------------------------------------------------------------------------
# Grid search on val set
# ---------------------------------------------------------------------------

def run_validate_grid(
    y_val: pd.Series,
    x_val: pd.Series,
    ou_std_log: float,
    half_life_bars: float,
    lot_size: float,
    roundtrip_cost: float,
) -> pd.DataFrame:
    """
    Grid search over 36 (entry_z × exit_z × lookback) combinations on val set.

    Returns DataFrame with columns:
        entry_z, exit_z, lookback, n_trades, win_rate, sharpe, calmar, net_pnl,
        cost_ok, rank
    Ranked by Sharpe descending.
    """
    last_price = float(y_val.iloc[-1])
    log_y = np.log(y_val.values)
    log_x = np.log(x_val.values)
    log_y_s = pd.Series(log_y, index=y_val.index)
    log_x_s = pd.Series(log_x, index=x_val.index)

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
                        "cost_ok":    cost_threshold_ok(entry_z, exit_z, roundtrip_cost,
                                                        ou_std_log, last_price),
                    })
                except Exception as e:
                    rows.append({
                        "entry_z": entry_z, "exit_z": exit_z, "lookback": lookback,
                        "n_trades": 0, "win_rate": None, "sharpe": None,
                        "calmar": None, "net_pnl": None, "cost_ok": False,
                        "error": str(e),
                    })

    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Test slice
# ---------------------------------------------------------------------------

def run_test_slice(
    y_test: pd.Series,
    x_test: pd.Series,
    locked_params: dict,
    lot_size: float,
    roundtrip_cost: float,
) -> dict:
    """
    Run signal engine on test set with locked params.

    locked_params must have: entry_z, exit_z, lookback, half_life_bars
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
# Regime breakdown
# ---------------------------------------------------------------------------

def compute_regime_breakdown(trades: list, test_dates) -> dict:
    """
    Split trades into year-bands: 2021 / 2022 / 2023 / 2024-25 / other.
    Returns dict {band_label: {n_trades, net_pnl, win_rate}}.
    """
    def _band(dt_str):
        try:
            yr = pd.Timestamp(dt_str).year
        except Exception:
            return "other"
        if yr <= 2021:
            return "≤2021"
        elif yr == 2022:
            return "2022"
        elif yr == 2023:
            return "2023"
        else:
            return "2024-25"

    bands = {}
    for tr in trades:
        b = _band(tr.get("entry_dt", ""))
        if b not in bands:
            bands[b] = {"n_trades": 0, "net_pnl": 0.0, "n_wins": 0}
        bands[b]["n_trades"] += 1
        bands[b]["net_pnl"]  += tr["net_pnl"]
        if tr["net_pnl"] > 0:
            bands[b]["n_wins"] += 1

    for b, v in bands.items():
        v["win_rate"] = round(v["n_wins"] / v["n_trades"], 3) if v["n_trades"] > 0 else None
        v["net_pnl"]  = round(v["net_pnl"], 2)

    return bands


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------

def detect_kill_switches(test_results: dict, train_beta_range: tuple) -> list:
    """
    Check for kill-switch conditions. Returns list of warning strings.

    Conditions:
      - Beta (hedge ratio) drifted outside [train_min × 0.5, train_max × 1.5]
      - Max drawdown > 3 × avg trade net pnl
      - Win rate < 35%
      - 0 trades in test set
    """
    flags = []
    m     = test_results["metrics"]

    if m["n_trades"] == 0:
        flags.append("KILL: Zero trades executed in test set — signal may be dead.")
        return flags

    if m["win_rate"] is not None and m["win_rate"] < 0.35:
        flags.append(f"KILL: Win rate {m['win_rate']:.1%} < 35% threshold.")

    if m["n_trades"] > 0 and m["max_drawdown"] != 0:
        avg_pnl = m["total_net_pnl"] / m["n_trades"]
        if avg_pnl != 0 and abs(m["max_drawdown"]) > 3 * abs(avg_pnl) * m["n_trades"]:
            flags.append(
                f"KILL: Max drawdown {m['max_drawdown']:.0f} > 3× avg trade PnL."
            )

    # Beta drift check
    trades = test_results.get("trades", [])
    if train_beta_range and len(trades) > 0:
        beta_arr = test_results.get("beta_t", np.array([]))
        if len(beta_arr) > 0:
            test_beta_min = float(np.nanmin(beta_arr))
            test_beta_max = float(np.nanmax(beta_arr))
            lo = train_beta_range[0] * 0.5
            hi = train_beta_range[1] * 1.5
            if test_beta_min < lo or test_beta_max > hi:
                flags.append(
                    f"KILL: Hedge ratio drifted to [{test_beta_min:.2f}, {test_beta_max:.2f}] "
                    f"vs train range [{train_beta_range[0]:.2f}, {train_beta_range[1]:.2f}]."
                )

    return flags
