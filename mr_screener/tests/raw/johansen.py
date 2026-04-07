import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def test_cointegration_johansen(y: pd.Series, x: pd.Series) -> dict:
    """Johansen cointegration test on the [y, x] system."""
    df = pd.concat([y, x], axis=1)
    df.columns = ["y", "x"]

    result = coint_johansen(df.values, det_order=0, k_ar_diff=1)

    # Trace test (index 0 = r=0, index 1 = r<=1)
    trace_stat    = result.lr1[0]
    trace_crit_95 = result.cvt[0, 1]   # 95% critical value

    # Max-eigenvalue test
    maxeig_stat    = result.lr2[0]
    maxeig_crit_95 = result.cvm[0, 1]

    trace_significant  = bool(trace_stat  > trace_crit_95)
    maxeig_significant = bool(maxeig_stat > maxeig_crit_95)

    # Cointegrating vector (eigenvector for largest eigenvalue)
    coint_vec = result.evec[:, 0].tolist()

    if trace_significant and maxeig_significant:
        verdict = "cointegrated"
    elif trace_significant or maxeig_significant:
        verdict = "borderline"
    else:
        verdict = "not_cointegrated"

    return {
        "trace_stat":          round(float(trace_stat), 4),
        "trace_crit_95":       round(float(trace_crit_95), 4),
        "trace_significant":   trace_significant,
        "maxeig_stat":         round(float(maxeig_stat), 4),
        "maxeig_crit_95":      round(float(maxeig_crit_95), 4),
        "maxeig_significant":  maxeig_significant,
        "cointegrating_vector": [round(v, 6) for v in coint_vec],
        "verdict":             verdict,
        "space":               "raw",
    }
