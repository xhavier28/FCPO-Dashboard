import pandas as pd
from mr_screener.tests.raw.adf_kpss import test_stationarity as _test_raw


def test_stationarity(series: pd.Series, name: str) -> dict:
    """Same as raw version but marks space as 'log'."""
    result = _test_raw(series, name)
    result["space"] = "log"
    return result
