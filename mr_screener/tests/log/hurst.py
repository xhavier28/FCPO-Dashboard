from mr_screener.tests.raw.hurst import hurst_exponent as _test_raw


def hurst_exponent(series, max_lag: int = 100) -> dict:
    """Same as raw version but marks space as 'log'."""
    result = _test_raw(series, max_lag)
    result["space"] = "log"
    return result
