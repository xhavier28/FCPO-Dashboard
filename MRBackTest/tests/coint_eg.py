import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def test_cointegration_eg(y: pd.Series, x: pd.Series) -> dict:
    """Engle-Granger cointegration test with static OLS spread as cross-check."""
    eg_stat, eg_pvalue, eg_crit = coint(y, x, trend="c", maxlag=5)

    X = add_constant(x.values)
    model        = OLS(y.values, X).fit()
    alpha_static = model.params[0]
    beta_static  = model.params[1]

    static_spread    = y.values - beta_static * x.values - alpha_static
    _, spread_adf_p, *_ = adfuller(static_spread, autolag="AIC")

    is_cointegrated = eg_pvalue < 0.05

    if is_cointegrated:
        verdict = "cointegrated"
    elif eg_pvalue < 0.10:
        verdict = "borderline"
    else:
        verdict = "not_cointegrated"

    return {
        "eg_stat":           round(eg_stat, 4),
        "eg_pvalue":         round(eg_pvalue, 4),
        "eg_crit_1pct":      round(eg_crit[0], 4),
        "eg_crit_5pct":      round(eg_crit[1], 4),
        "eg_crit_10pct":     round(eg_crit[2], 4),
        "is_cointegrated":   is_cointegrated,
        "beta_static":       round(float(beta_static), 6),
        "alpha_static":      round(float(alpha_static), 6),
        "spread_adf_pvalue": round(spread_adf_p, 4),
        "verdict":           verdict,
        "space":             "raw",
    }
