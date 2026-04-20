import pandas as pd
from mr_screener.tests.raw.coint_eg import test_cointegration_eg as _test_raw


def test_cointegration_eg(y: pd.Series, x: pd.Series) -> dict:
    """Same as raw version but marks space as 'log'."""
    result = _test_raw(y, x)
    result["space"] = "log"
    return result
