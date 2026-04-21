import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def test_cointegration_johansen(y: pd.Series, x: pd.Series,
                                freq: str = "daily") -> dict:
    """
    Johansen cointegration test on the [y, x] system.

    k_ar_diff=1 for daily (default — appropriate for daily bars).
    k_ar_diff=5 for intraday (absorbs intraday autocorrelation).
    """
    df = pd.concat([y, x], axis=1)
    df.columns = ["y", "x"]
    k_ar_diff = 1 if freq == "daily" else 5

    try:
        result = coint_johansen(df.values, det_order=0, k_ar_diff=k_ar_diff)
    except np.linalg.LinAlgError as e:
        return {
            "trace_stat":           None,
            "trace_crit_95":        None,
            "trace_crit_90":        None,
            "trace_significant":    False,
            "trace_significant_90": False,
            "maxeig_stat":          None,
            "maxeig_crit_95":       None,
            "maxeig_significant":   False,
            "cointegrating_vector": None,
            "verdict":              "not_cointegrated",
            "error":                f"Singular matrix: {e}",
            "space":                "raw",
        }

    trace_stat    = result.lr1[0]
    trace_crit_95 = result.cvt[0, 1]
    trace_crit_90 = result.cvt[0, 0]

    maxeig_stat    = result.lr2[0]
    maxeig_crit_95 = result.cvm[0, 1]

    trace_significant    = bool(trace_stat > trace_crit_95)
    trace_significant_90 = bool(trace_stat > trace_crit_90)
    maxeig_significant   = bool(maxeig_stat > maxeig_crit_95)

    coint_vec = result.evec[:, 0].tolist()

    if trace_significant and maxeig_significant:
        verdict = "cointegrated"
    elif trace_significant or maxeig_significant:
        verdict = "borderline"
    else:
        verdict = "not_cointegrated"

    return {
        "trace_stat":           round(float(trace_stat), 4),
        "trace_crit_95":        round(float(trace_crit_95), 4),
        "trace_crit_90":        round(float(trace_crit_90), 4),
        "trace_significant":    trace_significant,
        "trace_significant_90": trace_significant_90,
        "maxeig_stat":          round(float(maxeig_stat), 4),
        "maxeig_crit_95":       round(float(maxeig_crit_95), 4),
        "maxeig_significant":   maxeig_significant,
        "cointegrating_vector": [round(v, 6) for v in coint_vec],
        "verdict":              verdict,
        "space":                "raw",
    }
