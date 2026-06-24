"""
data/quality.py — Structural break detection + data sufficiency checks.

Entry point: run_quality_check(...)
"""

import warnings
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum bars required before each test can run at all
TEST_MIN = {
    "adf":                    30,
    "kpss":                   30,
    "engle_granger":          50,
    "johansen":               50,
    "hurst_static":          100,
    "kalman":                 50,
    "ou":                     50,
    "rolling_cointegration":  60,
    "rolling_hurst":          40,
}

# Recommended bars for reliable results
TEST_REC = {
    "adf":                   200,
    "kpss":                  200,
    "engle_granger":         200,
    "johansen":              200,
    "hurst_static":          300,
    "kalman":                252,
    "ou":                    200,
    "rolling_cointegration": 120,
    "rolling_hurst":          60,
}

# Which resolution each test needs
TEST_RESOLUTION = {
    "adf":                   "daily",
    "kpss":                  "daily",
    "engle_granger":         "daily",
    "johansen":              "daily",
    "hurst_static":          "daily",
    "kalman":                "intraday",
    "ou":                    "intraday",
    "rolling_cointegration": "daily",
    "rolling_hurst":         "daily",
}

BREAK_DEFAULTS = {
    "window":              30,
    "threshold":            3.5,
    "extend_bars":          30,
    "min_gap":              20,
    "mode":                "auto",
    "manual_periods":      [],
    "reversion_window":    10,
    "reversion_threshold":  2.0,
}


# ---------------------------------------------------------------------------
# FX guard
# ---------------------------------------------------------------------------

def _check_fx_applied(y, x, fx_was_applied):
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

    fx_note = " (FX applied)" if fx_was_applied else ""
    print(f"  FX guard{fx_note}: median(Y)={med_y:.4f}, median(X)={med_x:.4f}, "
          f"ratio={ratio:.4f} -> {label.upper()}")

    return label


# ---------------------------------------------------------------------------
# Break detection
# ---------------------------------------------------------------------------

def detect_breaks(y, x, window=30, threshold=3.5, extend_bars=30, min_gap=20,
                  reversion_window=10, reversion_threshold=2.0):
    """
    Detect persistent structural breaks using three methods:
      1. Log-ratio z-score
      2. Spread-level z-score (OLS residuals rolling)
      3. Rolling correlation drop

    Breaks that revert within `reversion_window` bars are NOT excluded.

    Returns list of [bar_start, bar_end, peak_z, trigger_method].
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)

    # -- Method 1: log-ratio z-score --
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where((y > 0) & (x > 0), np.log(y) - np.log(x), np.nan)

    s1   = pd.Series(log_ratio)
    rm1  = s1.rolling(window, min_periods=window // 2).mean()
    rs1  = s1.rolling(window, min_periods=window // 2).std().ffill().replace(0, np.nan)
    z1   = ((s1 - rm1) / rs1).fillna(0).abs()

    # -- Method 2: spread-level z-score (OLS residuals) --
    valid   = np.isfinite(y) & np.isfinite(x) & (x != 0)
    z2_arr  = np.zeros(n)
    if valid.sum() > window:
        y_v, x_v = y[valid], x[valid]
        beta      = np.polyfit(x_v, y_v, 1)[0]
        residuals = pd.Series(np.where(valid, y - beta * x, np.nan))
        rm2       = residuals.rolling(window, min_periods=window // 2).mean()
        rs2       = residuals.rolling(window, min_periods=window // 2).std().ffill().replace(0, np.nan)
        z2_arr    = ((residuals - rm2) / rs2).fillna(0).abs().values
    z2 = pd.Series(z2_arr)

    # -- Method 3: rolling correlation drop --
    dy       = pd.Series(y).pct_change()
    dx       = pd.Series(x).pct_change()
    roll_c   = dy.rolling(window, min_periods=window // 2).corr(dx)
    avg_c    = roll_c.mean()
    std_c    = roll_c.std()
    z3_arr   = np.zeros(n)
    if std_c and std_c > 0:
        z3_arr = (-(roll_c - avg_c) / std_c).clip(lower=0).fillna(0).values
    z3 = pd.Series(z3_arr)

    # -- Combined z --
    z_combined = pd.Series(np.maximum(z1.values, np.maximum(z2.values, z3.values)))

    candidates = np.where(z_combined.values > threshold)[0]
    if len(candidates) == 0:
        return []

    _method_names = ["log_ratio", "spread_level", "rolling_corr"]

    def _trigger(i):
        return _method_names[int(np.argmax([float(z1.iloc[i]),
                                            float(z2.iloc[i]),
                                            float(z3.iloc[i])]))]

    # Group candidates into raw periods
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

    # Clamp bar_end
    for p in raw_periods:
        p[1] = min(p[1], n - 1)

    # Merge overlapping
    merged = [list(raw_periods[0])]
    for p in raw_periods[1:]:
        if p[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], p[1])
            if p[2] > merged[-1][2]:
                merged[-1][2] = p[2]
                merged[-1][3] = p[3]
        else:
            merged.append(list(p))

    # -- Reversion check: skip breaks that recovered --
    persistent = []
    for p in merged:
        bar_end   = p[1]
        check_end = min(bar_end + reversion_window, n)
        if check_end > bar_end + 1:
            z_after = z_combined.iloc[bar_end + 1: check_end]
            if len(z_after) > 0 and z_after.min() < reversion_threshold:
                print(f"  [reversion] break at bar {p[0]}-{p[1]} "
                      f"reverted (min z={z_after.min():.2f}) -> NOT excluded")
                continue
        persistent.append(p)

    return persistent


# ---------------------------------------------------------------------------
# Break dict helpers
# ---------------------------------------------------------------------------

def _breaks_to_dicts(periods, index):
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
        }
        if has_dates and bar_start < len(index):
            d["date_start"] = str(pd.Timestamp(index[bar_start]).date())
        if has_dates and bar_end < len(index):
            d["date_end"] = str(pd.Timestamp(index[min(bar_end, len(index) - 1)]).date())
        result.append(d)
    return result


def _parse_manual_periods(manual_list, index):
    """Convert list of ("YYYY-MM-DD", "YYYY-MM-DD") tuples to bar-index dicts."""
    if not manual_list:
        return []

    has_dates = np.issubdtype(np.asarray(index).dtype, np.datetime64)
    result    = []

    for entry in manual_list:
        try:
            d_start = pd.Timestamp(entry[0])
            d_end   = pd.Timestamp(entry[1])
        except Exception:
            warnings.warn(f"[quality] Invalid manual period entry: {entry!r} -> skipped")
            continue

        if not has_dates:
            warnings.warn("[quality] Manual periods require datetime index -> skipped")
            continue

        idx_dt        = pd.DatetimeIndex(index)
        bars_in_range = np.where((idx_dt >= d_start) & (idx_dt <= d_end))[0]
        if len(bars_in_range) == 0:
            warnings.warn(f"[quality] Manual period {entry} has no matching bars -> skipped")
            continue

        result.append({
            "bar_start":  int(bars_in_range[0]),
            "bar_end":    int(bars_in_range[-1]),
            "n_bars":     int(len(bars_in_range)),
            "peak_z":     None,
            "source":     "manual",
            "date_start": str(d_start.date()),
            "date_end":   str(d_end.date()),
        })
    return result


def _merge_periods(auto_periods, manual_periods):
    """Merge auto + manual break dicts, sorting by bar_start and merging nearby."""
    all_p = sorted(auto_periods + manual_periods, key=lambda p: p["bar_start"])
    if not all_p:
        return []

    merged = [dict(all_p[0])]
    for p in all_p[1:]:
        prev = merged[-1]
        if p["bar_start"] <= prev["bar_end"] + 5:
            prev["bar_end"] = max(prev["bar_end"], p["bar_end"])
            prev["n_bars"]  = prev["bar_end"] - prev["bar_start"] + 1
            if p["peak_z"] is not None:
                prev["peak_z"] = (
                    max(prev["peak_z"], p["peak_z"])
                    if prev["peak_z"] is not None
                    else p["peak_z"]
                )
            if p["source"] != prev["source"]:
                prev["source"] = "auto+manual"
        else:
            merged.append(dict(p))
    return merged


# ---------------------------------------------------------------------------
# Exclusion
# ---------------------------------------------------------------------------

def _apply_exclusions(y, x, periods_to_exclude):
    """
    Remove bars belonging to break periods from y and x.

    Returns (y_out, x_out, summary, keep_mask).
    keep_mask is a boolean array usable to filter same_day_mask.
    """
    n       = len(y)
    exclude = np.zeros(n, dtype=bool)
    for p in periods_to_exclude:
        lo = max(0, int(p["bar_start"]))
        hi = min(n - 1, int(p["bar_end"]))
        exclude[lo: hi + 1] = True

    keep  = ~exclude
    y_out = y[keep].reset_index(drop=True)
    x_out = x[keep].reset_index(drop=True)

    summary = {
        "n_before":   n,
        "n_excluded": int(exclude.sum()),
        "n_after":    int(keep.sum()),
        "n_periods":  len(periods_to_exclude),
    }
    return y_out, x_out, summary, keep


# ---------------------------------------------------------------------------
# Sufficiency
# ---------------------------------------------------------------------------

def _assess_test(test_name, n):
    """Return readiness dict for a single test."""
    mn  = TEST_MIN.get(test_name, 50)
    rec = TEST_REC.get(test_name, 200)

    if n >= rec:
        status = "good"
    elif n >= mn:
        status = "marginal"
    else:
        status = "blocked"

    return {
        "can_run":   n >= mn,
        "status":    status,
        "n":         n,
        "min":       mn,
        "rec":       rec,
        "shortfall": max(0, mn - n),
    }


def _compute_adaptive_windows(n_daily):
    """Derive rolling-test window sizes from available daily bars."""
    coint_w = max(50, min(120, n_daily // 3))
    hurst_w = max(30, min(60,  n_daily // 4))
    n_coint = max(0, n_daily - coint_w)
    n_hurst = max(0, n_daily - hurst_w)
    return {
        "cointegration":          coint_w,
        "hurst":                  hurst_w,
        "n_rolling_coint_points": n_coint,
        "n_rolling_hurst_points": n_hurst,
    }


def _check_sufficiency(n_daily, n_intraday):
    """Build full sufficiency dict covering all tests."""
    suff = {
        "n_daily":    n_daily,
        "n_intraday": n_intraday,
        "tests":      {},
    }

    for test, resolution in TEST_RESOLUTION.items():
        n_avail           = n_daily if resolution == "daily" else n_intraday
        suff["tests"][test] = _assess_test(test, n_avail)

    # Adjust rolling tests based on adaptive windows
    windows = _compute_adaptive_windows(n_daily)
    for rolling_test, pts_key in (
        ("rolling_cointegration", "n_rolling_coint_points"),
        ("rolling_hurst",         "n_rolling_hurst_points"),
    ):
        if windows[pts_key] < 5:
            suff["tests"][rolling_test]["can_run"] = False
            suff["tests"][rolling_test]["status"]  = "blocked"

    daily_tests  = [t for t, r in TEST_RESOLUTION.items() if r == "daily"]
    blocked_core = any(suff["tests"][t]["status"] == "blocked" for t in daily_tests)
    statuses     = [v["status"] for v in suff["tests"].values()]
    suff["overall"] = "insufficient" if blocked_core else (
        "good" if all(s == "good" for s in statuses) else "marginal"
    )

    return suff


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_quality_check(
    y_daily,
    x_daily,
    y_intraday,
    x_intraday,
    label_y,
    label_x,
    bars_per_day=5.5,
    break_config=None,
    manual_periods=None,
    same_day_mask=None,
    fx_was_applied=False,
):
    """
    Full quality pipeline: FX guard -> break detection -> exclusions -> sufficiency.

    Returns a dict consumed by pipeline.run_pair().
    """
    cfg            = dict(BREAK_DEFAULTS)
    if break_config:
        cfg.update(break_config)
    manual_periods = manual_periods or cfg.get("manual_periods", [])

    print(f"\n{'='*60}")
    print(f"  QUALITY CHECK: {label_y} vs {label_x}")
    print(f"{'='*60}")

    y_d = pd.Series(y_daily).reset_index(drop=True)
    x_d = pd.Series(x_daily).reset_index(drop=True)
    y_i = pd.Series(y_intraday).reset_index(drop=True)
    x_i = pd.Series(x_intraday).reset_index(drop=True)

    # 1. FX guard
    print(f"\n[0/3] FX compatibility guard")
    fx_guard = _check_fx_applied(y_d.values, x_d.values, fx_was_applied)

    if fx_guard == "blocked":
        print("  [BLOCKED] Price ratio > 20x — FX conversion likely needed. Halting.")
        return {
            "y_daily_clean":       y_d,
            "x_daily_clean":       x_d,
            "y_intra_clean":       y_i,
            "x_intra_clean":       x_i,
            "same_day_mask_clean": same_day_mask,
            "break_report": {
                "total":         0,
                "auto_detected": 0,
                "manual":        0,
                "periods":       [],
                "mode":          cfg["mode"],
            },
            "sufficiency": {
                "n_daily":    len(y_d),
                "n_intraday": len(y_i),
                "tests":      {t: _assess_test(t, 0) for t in TEST_RESOLUTION},
                "overall":    "insufficient",
            },
            "windows":  _compute_adaptive_windows(0),
            "can_run":  {t: False for t in TEST_RESOLUTION},
            "overall":  "insufficient",
            "proceed":  False,
            "fx_guard": fx_guard,
        }

    if fx_guard == "warning":
        print("  [WARNING] Price ratio 5-20x — results may be unreliable.")

    # 2. Detect breaks on daily series
    print(f"\n[1/3] Structural break detection  (mode={cfg['mode']})")

    daily_index = (
        y_daily.index if hasattr(y_daily, "index") else pd.RangeIndex(len(y_d))
    )

    raw_breaks   = detect_breaks(
        y_d.values, x_d.values,
        window             = cfg["window"],
        threshold          = cfg["threshold"],
        extend_bars        = cfg["extend_bars"],
        min_gap            = cfg["min_gap"],
        reversion_window   = cfg.get("reversion_window",   BREAK_DEFAULTS["reversion_window"]),
        reversion_threshold= cfg.get("reversion_threshold",BREAK_DEFAULTS["reversion_threshold"]),
    )
    auto_periods = _breaks_to_dicts(raw_breaks, daily_index)

    # 3. Manual periods
    manual_dicts = _parse_manual_periods(manual_periods, daily_index)

    # 4. Merge
    all_periods = _merge_periods(auto_periods, manual_dicts)

    # 5. Apply mode filter
    mode = cfg["mode"]
    if mode == "none":
        periods_to_exclude = []
    elif mode == "manual_only":
        periods_to_exclude = manual_dicts
    else:  # "auto"
        periods_to_exclude = all_periods

    n_auto   = len(auto_periods)
    n_manual = len(manual_dicts)
    n_excl   = len(periods_to_exclude)

    if n_excl == 0:
        print("  No structural breaks detected")
    else:
        for p in periods_to_exclude:
            ds     = p.get("date_start", f"bar {p['bar_start']}")
            de     = p.get("date_end",   f"bar {p['bar_end']}")
            pz     = f"|z|={p['peak_z']:.2f}" if p["peak_z"] is not None else "manual"
            method = p.get("trigger_label", "")
            mstr   = f"  [{method}]" if method else ""
            print(f"  Break: {ds} -> {de}  ({p['n_bars']} bars, {pz}){mstr}")

    # 6. Apply exclusions to daily series
    print(f"\n[2/3] Applying exclusions")
    y_d_clean, x_d_clean, daily_summary, keep_daily = _apply_exclusions(
        y_d, x_d, periods_to_exclude
    )
    print(f"  Daily:    {daily_summary['n_before']} -> {daily_summary['n_after']} bars "
          f"({daily_summary['n_excluded']} excluded)")

    # 7. Apply exclusions to intraday series (scale bar indices by bars_per_day)
    intra_periods = [
        {**p,
         "bar_start": int(p["bar_start"] * bars_per_day),
         "bar_end":   int(p["bar_end"]   * bars_per_day)}
        for p in periods_to_exclude
    ]
    y_i_clean, x_i_clean, intra_summary, keep_intra = _apply_exclusions(
        y_i, x_i, intra_periods
    )
    print(f"  Intraday: {intra_summary['n_before']} -> {intra_summary['n_after']} bars "
          f"({intra_summary['n_excluded']} excluded)")

    # Apply same mask to same_day_mask
    sdm_clean = None
    if same_day_mask is not None:
        sdm_arr = np.asarray(same_day_mask)
        if len(sdm_arr) == len(keep_intra):
            sdm_clean = sdm_arr[keep_intra]
        else:
            sdm_clean = same_day_mask  # length mismatch — pass through unchanged

    # 8. Sufficiency
    print(f"\n[3/3] Data sufficiency")
    suff    = _check_sufficiency(daily_summary["n_after"], intra_summary["n_after"])
    windows = _compute_adaptive_windows(daily_summary["n_after"])

    _icon = {"good": "OK ", "marginal": "~  ", "blocked": "XX "}
    for test, s in suff["tests"].items():
        icon = _icon.get(s["status"], "?  ")
        print(f"  [{icon}] {test:<25} n={s['n']:>4}  min={s['min']:>3}  rec={s['rec']:>4}  {s['status']}")

    overall = suff["overall"]
    print(f"\n  Overall quality: {overall.upper()}")
    print(f"{'='*60}\n")

    # 9. Build break report
    break_report = {
        "total":         n_excl,
        "auto_detected": n_auto,
        "manual":        n_manual,
        "periods":       periods_to_exclude,
        "mode":          mode,
        "daily_summary": daily_summary,
        "intra_summary": intra_summary,
    }

    can_run = {t: s["can_run"] for t, s in suff["tests"].items()}
    proceed = overall != "insufficient"

    return {
        "y_daily_clean":       y_d_clean,
        "x_daily_clean":       x_d_clean,
        "y_intra_clean":       y_i_clean,
        "x_intra_clean":       x_i_clean,
        "same_day_mask_clean": sdm_clean,
        "break_report":        break_report,
        "sufficiency":         suff,
        "windows":             windows,
        "can_run":             can_run,
        "overall":             overall,
        "proceed":             proceed,
        "fx_guard":            fx_guard,
    }
