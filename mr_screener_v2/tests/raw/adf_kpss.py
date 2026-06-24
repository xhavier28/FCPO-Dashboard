import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def test_stationarity(series: pd.Series, name: str) -> dict:
    """
    Test whether a series is I(1).
    ADF: want p > 0.10 (cannot reject unit root → non-stationary).
    KPSS: want p < 0.05 (reject stationarity hypothesis).
    """
    adf_stat, adf_p, adf_lags, *_ = adfuller(series.dropna(), autolag="AIC")

    # KPSS returns (stat, p, lags, crit_vals); p is capped at 0.01/0.10
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series.dropna(), regression="c", nlags="auto")

    is_I1 = (adf_p > 0.10) and (kpss_p < 0.05)

    if is_I1:
        verdict = "I(1)"
    elif adf_p <= 0.05:
        verdict = "stationary"
    else:
        verdict = "ambiguous"

    return {
        "name":       name,
        "adf_stat":   round(adf_stat, 4),
        "adf_pvalue": round(adf_p, 4),
        "adf_lags":   adf_lags,
        "kpss_stat":  round(kpss_stat, 4),
        "kpss_pvalue": round(kpss_p, 4),
        "kpss_lags":  kpss_lags,
        "is_I1":      is_I1,
        "verdict":    verdict,
        "space":      "raw",
    }
