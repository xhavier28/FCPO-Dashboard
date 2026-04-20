from mr_screener.kalman.raw_kalman import run_kalman as _run_raw


def run_kalman(log_y, log_x, delta: float, Ve: float) -> dict:
    """Identical to raw_kalman but operates on log-price series."""
    result = _run_raw(log_y, log_x, delta, Ve)
    result["space"] = "log"
    return result
