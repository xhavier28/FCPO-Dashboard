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


def evaluate_gate(raw: dict, log: dict, alignment_info: dict | None) -> dict:
    """Classify pair as STRONG / MARGINAL / REJECT and collect warnings."""
    T = THRESHOLDS

    # --- Cointegration ---
    eg_strict  = (raw["coint_eg"]["eg_pvalue"] < 0.05
                  or log["coint_eg"]["eg_pvalue"] < 0.05)
    eg_loose   = (raw["coint_eg"]["eg_pvalue"] < T["coint_pvalue_loose"]
                  or log["coint_eg"]["eg_pvalue"] < T["coint_pvalue_loose"])

    joh_strict = (raw["johansen"].get("trace_significant", False)
                  or log["johansen"].get("trace_significant", False))
    joh_loose  = (raw["johansen"].get("trace_significant_90", False)
                  or log["johansen"].get("trace_significant_90", False))

    coint_strict = eg_strict or joh_strict
    coint_loose  = eg_loose  or joh_loose

    # --- Hurst ---
    def _hurst(r):
        return r["hurst"].get("hurst", float("nan"))

    hurst_strict = (_hurst(raw) < 0.50 or _hurst(log) < 0.50)
    hurst_loose  = (_hurst(raw) < 0.60 or _hurst(log) < 0.60)

    # --- Tier ---
    if coint_strict and hurst_strict:
        tier = "STRONG"
    elif coint_loose:
        tier = "MARGINAL"
    else:
        tier = "REJECT"

    # --- Warnings ---
    warnings = []
    raw_adf_ok = (raw["adf_y"]["verdict"] == "I(1)" and raw["adf_x"]["verdict"] == "I(1)")
    log_adf_ok = (log["adf_y"]["verdict"] == "I(1)" and log["adf_x"]["verdict"] == "I(1)")
    if not raw_adf_ok and not log_adf_ok:
        warnings.append(
            "One or both series may already be stationary — check for roll "
            "adjustment artifacts or structural breaks."
        )
    if not coint_strict and coint_loose:
        eg_best = min(raw["coint_eg"]["eg_pvalue"], log["coint_eg"]["eg_pvalue"])
        warnings.append(
            f"Cointegration marginal — EG p={eg_best:.3f} "
            f"(passes relaxed 0.20 threshold, not strict 0.05)."
        )
    if not hurst_strict and hurst_loose:
        h_best = min(_hurst(raw), _hurst(log))
        warnings.append(
            f"Hurst borderline — {h_best:.3f} "
            f"(above 0.50 but below 0.60, mild trending tendency)."
        )
    if not hurst_loose and tier != "REJECT":
        h_best = min(_hurst(raw), _hurst(log))
        warnings.append(
            f"Hurst elevated — {h_best:.3f} (trending). "
            f"Cointegration still found; proceed with caution."
        )
    n_after = alignment_info.get("n_after", 0) if alignment_info else 0
    if n_after < 200:
        warnings.append(
            f"Only {n_after} bars — results less reliable. "
            f"Upload more history for stronger confidence."
        )

    return {
        "tier":    tier,
        "warnings": warnings,
        "proceed": tier in ("STRONG", "MARGINAL"),
    }


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
    gate_info   = evaluate_gate(raw, log, data.get("alignment"))
    gate_passed = gate_info["proceed"]

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
        "raw":          raw,
        "log":          log,
        "gate_passed":  gate_passed,
        "gate_tier":    gate_info["tier"],
        "gate_warnings": gate_info["warnings"],
        "label_y":      data["label_y"],
        "label_x":      data["label_x"],
        "n_obs":        data["n_obs"],
        "dates":        data["dates"],
    }
