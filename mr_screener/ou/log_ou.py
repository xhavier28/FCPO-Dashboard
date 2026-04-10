from mr_screener.ou.raw_ou import fit_ou as _fit_raw


def fit_ou(spread, dt: float = 1.0,
           freq: str = "hourly", bars_per_day: float = 5.5,
           same_day_mask=None) -> dict:
    """Identical to raw_ou but marks space as 'log'."""
    result = _fit_raw(spread, dt=dt, freq=freq, bars_per_day=bars_per_day,
                      same_day_mask=same_day_mask)
    result["space"] = "log"
    return result
