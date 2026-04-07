import numpy as np
import pandas as pd
from scipy.stats import linregress, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox


def fit_ou(spread, dt: float = 1.0,
           freq: str = "hourly", bars_per_day: float = 5.5) -> dict:
    """
    Fit Ornstein-Uhlenbeck process to a spread via AR(1) regression.

    dt = 1.0 per bar (units follow freq).
    freq: "hourly" or "daily"
    bars_per_day: used for half-life conversion (e.g. 5.5 for FCPO)

    half_life_bars  — in bar units (raw OU math output)
    half_life_hours — in hours
    half_life_days  — in trading days
    """
    from mr_screener.config import THRESHOLDS as T

    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]

    if len(s) < 20:
        return {
            "verdict": "reject",
            "reason":  "insufficient observations",
            "space":   "raw",
        }

    S_t  = s[:-1]
    S_t1 = s[1:]

    slope, intercept, r_value, p_value, std_err = linregress(S_t, S_t1)
    beta_ar1  = slope
    residuals = S_t1 - (slope * S_t + intercept)

    # Validate AR(1) beta
    if not (0.0 < beta_ar1 < 1.0):
        return {
            "is_valid":   False,
            "beta_ar1":   round(float(beta_ar1), 6),
            "kappa":      None,
            "mu":         None,
            "ou_std":     None,
            "half_life":       None,
            "half_life_bars":  None,
            "half_life_hours": None,
            "half_life_days":  None,
            "verdict":    "reject",
            "reject_reason": (
                f"beta_ar1={beta_ar1:.4f} outside (0,1). "
                + ("Spread has unit root — not mean reverting."
                   if beta_ar1 >= 1 else
                   "Spread is oscillating or negatively autocorrelated.")
            ),
            "space": "raw",
        }

    # OU parameters (kappa in 1/bar units)
    kappa     = -np.log(beta_ar1) / dt
    mu        = intercept / (1.0 - beta_ar1)
    sigma_eps = float(np.std(residuals, ddof=1))
    ou_std    = sigma_eps / np.sqrt(2.0 * kappa * dt) if kappa > 0 else float("nan")

    # Half-life in bar, hour, day units
    half_life_bars = float(np.log(2.0) / kappa)
    if freq == "hourly":
        half_life_hours = half_life_bars
        half_life_days  = half_life_bars / bars_per_day
    else:  # daily
        half_life_hours = half_life_bars * bars_per_day
        half_life_days  = half_life_bars

    # Diagnostics
    lb_result = acorr_ljungbox(residuals, lags=[5, 10], return_df=True)
    lb_pvals  = lb_result["lb_pvalue"].tolist()
    jb_stat, jb_p = jarque_bera(residuals)
    lb_ok = all(p > 0.05 for p in lb_pvals)

    # Verdict — tiered, using hours for intraday / days for daily
    hl_reason = None
    if freq == "hourly":
        hl_min_sweet = T["half_life_sweet_min_h"]
        hl_max_sweet = T["half_life_sweet_max_h"]
        hl_max       = T["half_life_max_hours"]
        hl_ref       = half_life_hours
        hl_unit      = "h"
    else:
        hl_min_sweet = T["half_life_sweet_min"]
        hl_max_sweet = T["half_life_sweet_max"]
        hl_max       = T["half_life_max"]
        hl_ref       = half_life_days
        hl_unit      = "d"

    if freq == "hourly" and half_life_hours < 1.0:
        verdict   = "reject"
        hl_reason = (
            f"Half-life {half_life_hours:.2f}h < 1 hour — spread reverts too fast to trade, "
            "or OU fitted wrong spread (check spread_reconstructed)."
        )
    elif freq == "hourly" and half_life_hours < hl_min_sweet:
        verdict   = "borderline"
        hl_reason = f"Half-life {half_life_hours:.1f}h very short — intraday scalp only."
    elif freq == "daily" and half_life_days < 1.0:
        verdict   = "reject"
        hl_reason = (
            f"Half-life {half_life_days:.2f}d < 1 day — too fast or wrong spread."
        )
    elif freq == "daily" and half_life_days < 2.0:
        verdict   = "borderline"
        hl_reason = f"Half-life {half_life_days:.2f}d very short — check data frequency."
    elif hl_ref <= hl_max_sweet and lb_ok:
        verdict = "tradeable"
    elif hl_ref <= hl_max:
        verdict = "borderline"
    else:
        verdict   = "borderline"
        hl_reason = (
            f"Half-life {hl_ref:.1f}{hl_unit} > {hl_max}{hl_unit} — "
            "capital locked up too long."
        )

    return {
        "is_valid":        True,
        "beta_ar1":        round(float(beta_ar1), 6),
        "r_squared":       round(float(r_value ** 2), 4),
        "kappa":           round(float(kappa), 6),
        "mu":              round(float(mu), 6),
        "sigma_eps":       round(float(sigma_eps), 6),
        "ou_std":          round(float(ou_std), 6) if not np.isnan(ou_std) else None,
        "half_life_bars":  round(half_life_bars,  2),
        "half_life_hours": round(half_life_hours, 1),
        "half_life_days":  round(half_life_days,  2),
        "half_life":       round(half_life_days,  2),  # backward-compat alias
        "in_sweet_spot":   verdict == "tradeable",
        "freq":            freq,
        "bars_per_day":    bars_per_day,
        "lb_pval_5":       round(lb_pvals[0], 4),
        "lb_pval_10":      round(lb_pvals[1], 4),
        "jb_stat":         round(float(jb_stat), 4),
        "jb_pvalue":       round(float(jb_p), 4),
        "verdict":         verdict,
        "hl_reason":       hl_reason,
        "space":           "raw",
    }
