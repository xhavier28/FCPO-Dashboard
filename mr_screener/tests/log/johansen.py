import pandas as pd
from mr_screener.tests.raw.johansen import test_cointegration_johansen as _test_raw


def test_cointegration_johansen(y: pd.Series, x: pd.Series,
                                freq: str = "hourly") -> dict:
    """Same as raw version but marks space as 'log'."""
    result = _test_raw(y, x, freq=freq)
    result["space"] = "log"
    return result
