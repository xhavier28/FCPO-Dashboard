"""
shared/structural_break.py — Structural break detection and exclusion.

Extracted from mr_screener/data/quality.py.
Only: detect_breaks(), apply_exclusions(), _check_fx_applied().
run_quality_check() is dropped — WF1 pipeline owns that logic.
"""

import numpy as np
import pandas as pd
import warnings


# ---------------------------------------------------------------------------
# FX guard
# ---------------------------------------------------------------------------

def check_fx_applied(y, x, fx_was_applied: bool = False) -> str:
    """
    Check median(Y)/median(X) ratio to detect scale incompatibility.

    Returns: "blocked", "warning", or "ok"
    """
    med_y = np.nanmedian(np.asarray(y, dtype=float))
    med_x = np.nanmedian(np.asarray(x, dtype=float))

    if med_x == 0:
        return "blocked"

    ratio = med_y / med_x

    if ratio > 20 or ratio < (1 / 20):
        label = "blocked"
    elif ratio > 5 or ratio < 0.2:
        label = "warning"
    else:
        label = "ok"

    note = " (FX applied)" if fx_was_applied else ""
    print(f"  FX guard{note}: median(Y)={med_y:.4f}, median(X)={med_x:.4f}, "
          f"ratio={ratio:.4f} -> {label.upper()}")
    return label


# ---------------------------------------------------------------------------
# Break detection
# ---------------------------------------------------------------------------

def detect_breaks(
    y, x,
    window: int = 30,
    threshold: float = 3.5,
    extend_bars: int = 30,
    min_gap: int = 20,
    reversion_window: int = 10,
    reversion_threshold: float = 2.0,
) -> list:
    """
    Detect persistent structural breaks via three methods:
      1. Log-ratio z-score
      2. Spread-level z-score (OLS residuals rolling)
      3. Rolling correlation drop

    Breaks that revert within `reversion_window` bars are NOT excluded.

    Returns list of [bar_start, bar_end, peak_z, trigger_method].
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)

    # Method 1: log-ratio z-score
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where((y > 0) & (x > 0), np.log(y) - np.log(x), np.nan)

    s1  = pd.Series(log_ratio)
    rm1 = s1.rolling(window, min_periods=window // 2).mean()
    rs1 = s1.rolling(window, min_periods=window // 2).std().ffill().replace(0, np.nan)
    z1  = ((s1 - rm1) / rs1).fillna(0).abs()

    # Method 2: spread-level z-score (OLS residuals)
    valid  = np.isfinite(y) & np.isfinite(x) & (x != 0)
    z2_arr = np.zeros(n)
    if valid.sum() > window:
        y_v, x_v = y[valid], x[valid]
        beta      = np.polyfit(x_v, y_v, 1)[0]
        residuals = pd.Series(np.where(valid, y - beta * x, np.nan))
        rm2       = residuals.rolling(window, min_periods=window // 2).mean()
        rs2       = residuals.rolling(window, min_periods=window // 2).std().ffill().replace(0, np.nan)
        z2_arr    = ((residuals - rm2) / rs2).fillna(0).abs().values
    z2 = pd.Series(z2_arr)

    # Method 3: rolling correlation drop
    dy     = pd.Series(y).pct_change()
    dx     = pd.Series(x).pct_change()
    roll_c = dy.rolling(window, min_periods=window // 2).corr(dx)
    avg_c  = roll_c.mean()
    std_c  = roll_c.std()
    z3_arr = np.zeros(n)
    if std_c and std_c > 0:
        z3_arr = (-(roll_c - avg_c) / std_c).clip(lower=0).fillna(0).values
    z3 = pd.Series(z3_arr)

    z_combined = pd.Series(np.maximum(z1.values, np.maximum(z2.values, z3.values)))
    candidates = np.where(z_combined.values > threshold)[0]

    if len(candidates) == 0:
        return []

    _method_names = ["log_ratio", "spread_level", "rolling_corr"]

    def _trigger(i):
        return _method_names[int(np.argmax([float(z1.iloc[i]),
                                            float(z2.iloc[i]),
                                            float(z3.iloc[i])]))]

    raw_periods = []
    grp_start   = candidates[0]
    grp_end     = candidates[0]
    peak_z      = float(z_combined.iloc[candidates[0]])
    trigger     = _trigger(candidates[0])

    for idx in candidates[1:]:
        if idx - grp_end <= min_gap:
            grp_end = idx
            new_z   = float(z_combined.iloc[idx])
            if new_z > peak_z:
                peak_z  = new_z
                trigger = _trigger(idx)
        else:
            raw_periods.append([grp_start, grp_end + extend_bars, round(peak_z, 3), trigger])
            grp_start = idx
            grp_end   = idx
            peak_z    = float(z_combined.iloc[idx])
            trigger   = _trigger(idx)
    raw_periods.append([grp_start, grp_end + extend_bars, round(peak_z, 3), trigger])

    for p in raw_periods:
        p[1] = min(p[1], n - 1)

    merged = [list(raw_periods[0])]
    for p in raw_periods[1:]:
        if p[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], p[1])
            if p[2] > merged[-1][2]:
                merged[-1][2] = p[2]
                merged[-1][3] = p[3]
        else:
            merged.append(list(p))

    # Reversion check
    persistent = []
    for p in merged:
        bar_end   = p[1]
        check_end = min(bar_end + reversion_window, n)
        if check_end > bar_end + 1:
            z_after = z_combined.iloc[bar_end + 1: check_end]
            if len(z_after) > 0 and z_after.min() < reversion_threshold:
                print(f"  [reversion] break bar {p[0]}-{p[1]} reverted -> NOT excluded")
                continue
        persistent.append(p)

    return persistent


def breaks_to_dicts(periods: list, index) -> list:
    """Convert raw [bar_start, bar_end, peak_z, trigger_method] to labelled dicts."""
    has_dates = np.issubdtype(np.asarray(index).dtype, np.datetime64)

    _trigger_labels = {
        "log_ratio":    "Log-ratio z-score",
        "spread_level": "Spread-level z-score",
        "rolling_corr": "Rolling correlation drop",
    }

    result = []
    for p in periods:
        bar_start, bar_end, peak_z = p[0], p[1], p[2]
        trigger_method = p[3] if len(p) > 3 else "log_ratio"
        d = {
            "bar_start":      int(bar_start),
            "bar_end":        int(bar_end),
            "n_bars":         int(bar_end - bar_start + 1),
            "peak_z":         float(peak_z),
            "source":         "auto",
            "trigger_method": trigger_method,
            "trigger_label":  _trigger_labels.get(trigger_method, trigger_method),
            "enabled":        True,
        }
        if has_dates and bar_start < len(index):
            d["date_start"] = str(pd.Timestamp(index[bar_start]).date())
        if has_dates and bar_end < len(index):
            d["date_end"] = str(pd.Timestamp(index[min(bar_end, len(index) - 1)]).date())
        result.append(d)
    return result


def apply_exclusions(y: pd.Series, x: pd.Series, periods: list) -> tuple:
    """
    Remove bars belonging to enabled break periods from y and x.

    Parameters
    ----------
    y, x    : aligned price Series with DatetimeIndex
    periods : list of dicts (from breaks_to_dicts), with 'enabled' key

    Returns
    -------
    y_clean, x_clean : filtered Series (index preserved)
    summary          : dict with n_before, n_excluded, n_after
    """
    n       = len(y)
    exclude = np.zeros(n, dtype=bool)

    for p in periods:
        if not p.get("enabled", True):
            continue
        lo = max(0, int(p["bar_start"]))
        hi = min(n - 1, int(p["bar_end"]))
        exclude[lo: hi + 1] = True

    keep  = ~exclude
    y_out = y[keep]
    x_out = x[keep]

    summary = {
        "n_before":   n,
        "n_excluded": int(exclude.sum()),
        "n_after":    int(keep.sum()),
        "n_periods":  sum(1 for p in periods if p.get("enabled", True)),
    }
    return y_out, x_out, summary
