from mr_screener.ou.raw_ou import fit_ou as _fit_raw


def fit_ou(spread, dt: float = 1.0) -> dict:
    """Identical to raw_ou but marks space as 'log'."""
    result = _fit_raw(spread, dt)
    result["space"] = "log"
    return result
