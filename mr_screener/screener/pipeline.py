from mr_screener.config import KALMAN, THRESHOLDS
from mr_screener.tests.raw import adf_kpss as raw_adf_mod
from mr_screener.tests.raw import coint_eg as raw_eg_mod
from mr_screener.tests.raw import johansen as raw_joh_mod
from mr_screener.tests.raw import hurst as raw_hurst_mod
from mr_screener.tests.log import adf_kpss as log_adf_mod
from mr_screener.tests.log import coint_eg as log_eg_mod
from mr_screener.tests.log import johansen as log_joh_mod
from mr_screener.tests.log import hurst as log_hurst_mod
from mr_screener.kalman import raw_kalman, log_kalman
from mr_screener.ou import raw_ou, log_ou


def run_pair(data: dict, delta: float = None, Ve: float = None) -> dict:
    """
    Full mean-reversion screening pipeline.

    Gate: (eg_coint_raw OR eg_coint_log) AND (hurst_tradeable_raw OR hurst_tradeable_log)
    Only runs Kalman + OU if gate passes.
    """
    delta = delta if delta is not None else KALMAN["delta"]
    Ve    = Ve    if Ve    is not None else KALMAN["Ve"]

    raw_y, raw_x = data["raw_y"], data["raw_x"]
    log_y, log_x = data["log_y"], data["log_x"]

    # ── Raw space tests ───────────────────────────────────────────────────────
    raw = {}
    raw["adf_y"]    = raw_adf_mod.test_stationarity(raw_y, data["label_y"])
    raw["adf_x"]    = raw_adf_mod.test_stationarity(raw_x, data["label_x"])
    raw["coint_eg"] = raw_eg_mod.test_cointegration_eg(raw_y, raw_x)
    raw["johansen"] = raw_joh_mod.test_cointegration_johansen(raw_y, raw_x)

    # Hurst on EG static spread
    raw_spread_static = (
        raw_y.values
        - raw["coint_eg"]["beta_static"] * raw_x.values
        - raw["coint_eg"]["alpha_static"]
    )
    raw["hurst"] = raw_hurst_mod.hurst_exponent(raw_spread_static)

    # ── Log space tests ───────────────────────────────────────────────────────
    log = {}
    log["adf_y"]    = log_adf_mod.test_stationarity(log_y, data["label_y"])
    log["adf_x"]    = log_adf_mod.test_stationarity(log_x, data["label_x"])
    log["coint_eg"] = log_eg_mod.test_cointegration_eg(log_y, log_x)
    log["johansen"] = log_joh_mod.test_cointegration_johansen(log_y, log_x)

    log_spread_static = (
        log_y.values
        - log["coint_eg"]["beta_static"] * log_x.values
        - log["coint_eg"]["alpha_static"]
    )
    log["hurst"] = log_hurst_mod.hurst_exponent(log_spread_static)

    # ── Gate check ────────────────────────────────────────────────────────────
    eg_pass   = raw["coint_eg"]["is_cointegrated"] or log["coint_eg"]["is_cointegrated"]
    hurst_pass = raw["hurst"]["tradeable"] or log["hurst"]["tradeable"]
    gate_passed = eg_pass and hurst_pass

    # ── Kalman + OU (only if gate passes) ────────────────────────────────────
    if gate_passed:
        raw["kalman"] = raw_kalman.run_kalman(raw_y.values, raw_x.values, delta, Ve)
        raw["ou"]     = raw_ou.fit_ou(raw["kalman"]["spread"])

        log["kalman"] = log_kalman.run_kalman(log_y.values, log_x.values, delta, Ve)
        log["ou"]     = log_ou.fit_ou(log["kalman"]["spread"])
    else:
        raw["kalman"] = None
        raw["ou"]     = None
        log["kalman"] = None
        log["ou"]     = None

    return {
        "raw":        raw,
        "log":        log,
        "gate_passed": gate_passed,
        "label_y":    data["label_y"],
        "label_x":    data["label_x"],
        "n_obs":      data["n_obs"],
        "dates":      data["dates"],
    }
