import numpy as np
import pandas as pd
from scipy.stats import linregress, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox


def fit_ou(spread, dt: float = 1.0) -> dict:
    """
    Fit Ornstein-Uhlenbeck process to a spread via AR(1) regression.
    dt = 1.0 day (daily bars after bucketing).

    Returns kappa (mean-reversion speed), mu (long-run mean),
    sigma_eps (noise std), ou_std (equilibrium std), half_life (days).
    """
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
    beta_ar1 = slope
    residuals = S_t1 - (slope * S_t + intercept)

    # Validate AR(1) beta
    if not (0.0 < beta_ar1 < 1.0):
        return {
            "beta_ar1": round(float(beta_ar1), 6),
            "verdict":  "reject",
            "reason":   f"beta_ar1={beta_ar1:.4f} outside (0,1) — not mean-reverting",
            "space":    "raw",
        }

    # OU parameters
    kappa     = -np.log(beta_ar1) / dt          # mean-reversion speed (1/hour)
    mu        = intercept / (1.0 - beta_ar1)     # long-run mean
    sigma_eps = float(np.std(residuals, ddof=1))
    ou_std    = sigma_eps / np.sqrt(2.0 * kappa * dt) if kappa > 0 else float("nan")
    half_life = np.log(2.0) / kappa              # days

    # Diagnostics
    lb_result = acorr_ljungbox(residuals, lags=[5, 10], return_df=True)
    lb_pvals  = lb_result["lb_pvalue"].tolist()

    jb_stat, jb_p = jarque_bera(residuals)

    # Verdict
    from mr_screener.config import THRESHOLDS as T
    hl_ok   = T["half_life_min"] <= half_life <= T["half_life_max"]
    hl_sweet = T["half_life_sweet_min"] <= half_life <= T["half_life_sweet_max"]
    lb_ok   = all(p > 0.05 for p in lb_pvals)

    if hl_sweet and lb_ok:
        verdict = "tradeable"
    elif hl_ok:
        verdict = "borderline"
    else:
        verdict = "reject"

    return {
        "beta_ar1":   round(float(beta_ar1), 6),
        "r_squared":  round(float(r_value ** 2), 4),
        "kappa":      round(float(kappa), 6),
        "mu":         round(float(mu), 6),
        "sigma_eps":  round(float(sigma_eps), 6),
        "ou_std":     round(float(ou_std), 6) if not np.isnan(ou_std) else None,
        "half_life":  round(float(half_life), 2),
        "lb_pval_5":  round(lb_pvals[0], 4),
        "lb_pval_10": round(lb_pvals[1], 4),
        "jb_stat":    round(float(jb_stat), 4),
        "jb_pvalue":  round(float(jb_p), 4),
        "verdict":    verdict,
        "space":      "raw",
    }
