"""
shared/ou.py — Ornstein-Uhlenbeck process fitting (raw and log-space).

Merged from mr_screener/ou/raw_ou.py + log_ou.py.
No imports from parent project. same_day_mask removed from public API
(not needed for daily WF1; WF2 15-min bars are synchronized).
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox

# Half-life thresholds (in days / hours)
_HL_SWEET_MIN_DAYS = 2
_HL_SWEET_MAX_DAYS = 30
_HL_MAX_DAYS       = 60
_HL_SWEET_MIN_H    = 2
_HL_SWEET_MAX_H    = 24
_HL_MAX_H          = 72


def fit_ou(
    spread,
    dt: float = 1.0,
    freq: str = "daily",
    bars_per_day: float = 1.0,
    space: str = "raw",
) -> dict:
    """
    Fit Ornstein-Uhlenbeck process to a spread via AR(1) regression.

    Parameters
    ----------
    spread       : spread series (spread_reconstructed from Kalman)
    dt           : bar duration (always 1.0 bar)
    freq         : "daily" or "intraday"
    bars_per_day : used for half-life conversions
                   daily → 1.0, 15-min → 26.0 (or actual bars/day)
    space        : "raw" or "log" (recorded in output only)

    Returns
    -------
    dict with OU parameters, half-life, diagnostics, verdict
    """
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]

    if len(s) < 20:
        return {
            "verdict": "reject",
            "reason":  "insufficient observations",
            "space":   space,
        }

    # Autocorrelation diagnostic
    autocorr_lag1 = float(pd.Series(s).autocorr(lag=1))
    if abs(autocorr_lag1) < 0.10:
        print(f"  [OU WARN] spread autocorr lag1={autocorr_lag1:.4f} — near zero, "
              "check that spread_reconstructed (not innovations) was passed.")
    else:
        print(f"  [OU OK] spread autocorr lag1={autocorr_lag1:.4f}")

    S_t  = s[:-1]
    S_t1 = s[1:]

    slope, intercept, r_value, p_value, std_err = linregress(S_t, S_t1)
    beta_ar1  = slope
    residuals = S_t1 - (slope * S_t + intercept)

    if not (0.0 < beta_ar1 < 1.0):
        return {
            "is_valid":        False,
            "beta_ar1":        round(float(beta_ar1), 6),
            "kappa":           None,
            "mu":              None,
            "ou_std":          None,
            "half_life_bars":  None,
            "half_life_hours": None,
            "half_life_days":  None,
            "verdict":         "reject",
            "reject_reason": (
                f"beta_ar1={beta_ar1:.4f} outside (0,1). "
                + ("Spread has unit root — not mean reverting."
                   if beta_ar1 >= 1 else
                   "Spread oscillating or negatively autocorrelated.")
            ),
            "space": space,
        }

    kappa     = -np.log(beta_ar1) / dt
    mu        = intercept / (1.0 - beta_ar1)
    sigma_eps = float(np.std(residuals, ddof=1))
    ou_std    = sigma_eps / np.sqrt(2.0 * kappa * dt) if kappa > 0 else float("nan")

    half_life_bars = float(np.log(2.0) / kappa)

    if freq == "intraday":
        half_life_hours = half_life_bars / bars_per_day * 24 if bars_per_day > 0 else float("nan")
        # For 15-min: bars_per_day=26 → half_life_hours = half_life_bars * 0.25
        half_life_hours = half_life_bars * (24.0 / bars_per_day) if bars_per_day > 0 else float("nan")
        half_life_days  = half_life_hours / 24.0
        hl_ref  = half_life_hours
        hl_unit = "h"
        hl_min_sweet, hl_max_sweet, hl_max = _HL_SWEET_MIN_H, _HL_SWEET_MAX_H, _HL_MAX_H
    else:
        half_life_hours = half_life_bars * (24.0 / bars_per_day) if bars_per_day > 0 else float("nan")
        half_life_days  = half_life_bars
        hl_ref  = half_life_days
        hl_unit = "d"
        hl_min_sweet, hl_max_sweet, hl_max = _HL_SWEET_MIN_DAYS, _HL_SWEET_MAX_DAYS, _HL_MAX_DAYS

    # Diagnostics
    lb_result = acorr_ljungbox(residuals, lags=[5, 10], return_df=True)
    lb_pvals  = lb_result["lb_pvalue"].tolist()
    jb_stat, jb_p = jarque_bera(residuals)
    lb_ok = all(p > 0.05 for p in lb_pvals)

    # Verdict
    hl_reason = None
    if hl_ref < 1.0:
        verdict   = "reject"
        hl_reason = f"Half-life {hl_ref:.2f}{hl_unit} < 1 — too fast or wrong spread."
    elif hl_ref < hl_min_sweet:
        verdict   = "borderline"
        hl_reason = f"Half-life {hl_ref:.1f}{hl_unit} very short."
    elif hl_ref <= hl_max_sweet and lb_ok:
        verdict = "tradeable"
    elif hl_ref <= hl_max:
        verdict = "borderline"
    else:
        verdict   = "borderline"
        hl_reason = f"Half-life {hl_ref:.1f}{hl_unit} > {hl_max}{hl_unit} — capital locked too long."

    return {
        "is_valid":        True,
        "beta_ar1":        round(float(beta_ar1), 6),
        "r_squared":       round(float(r_value ** 2), 4),
        "kappa":           round(float(kappa), 6),
        "mu":              round(float(mu), 6),
        "sigma_eps":       round(float(sigma_eps), 6),
        "ou_std":          round(float(ou_std), 6) if not np.isnan(ou_std) else None,
        "half_life_bars":  round(half_life_bars, 2),
        "half_life_hours": round(half_life_hours, 1) if not np.isnan(half_life_hours) else None,
        "half_life_days":  round(half_life_days, 2) if not np.isnan(half_life_days) else None,
        "in_sweet_spot":   verdict == "tradeable",
        "freq":            freq,
        "bars_per_day":    bars_per_day,
        "lb_pval_5":       round(lb_pvals[0], 4),
        "lb_pval_10":      round(lb_pvals[1], 4),
        "jb_stat":         round(float(jb_stat), 4),
        "jb_pvalue":       round(float(jb_p), 4),
        "verdict":         verdict,
        "hl_reason":       hl_reason,
        "space":           space,
    }
