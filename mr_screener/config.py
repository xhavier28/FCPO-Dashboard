THRESHOLDS = {
    "adf_pvalue":          0.10,
    "coint_pvalue":        0.05,
    "hurst_max":           0.50,
    "hurst_strong":        0.45,
    "beta_ar1_min":        0.0,
    "beta_ar1_max":        1.0,
    "half_life_min":       1,    # hours
    "half_life_max":       168,  # 1 week
    "half_life_sweet_min": 4,
    "half_life_sweet_max": 48,
}

KALMAN = {"delta": 1e-4, "Ve": 0.001}
