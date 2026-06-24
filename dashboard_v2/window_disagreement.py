# window_disagreement.py
# Shared module for comparing today's metric value across multiple
# historical reference windows.  Called by Event Log and S Calculator
# with their own window definitions — no tab-specific logic here.

import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats

MIN_SAMPLE_SIZE = 15


def compute_window_disagreement(
    today_value,
    windows,
    pct_disagreement_threshold=20.0,
    z_disagreement_threshold=1.0,
):
    """Compare today_value against multiple historical windows.

    Parameters
    ----------
    today_value : float
        The current observation to evaluate.
    windows : dict[str, pd.Series]
        Mapping of window name to a Series of historical values for that
        metric.  Each Series should already be filtered/built by the
        caller (rolling, seasonal, etc.).
    pct_disagreement_threshold : float
        Minimum percentile-rank gap (in points, 0-100 scale) between two
        windows before they are considered "disagreeing".
    z_disagreement_threshold : float
        Minimum Z-score gap between two windows before they are
        considered "disagreeing".

    Returns
    -------
    dict with keys:
        per_window              – per-window pct_rank, z_score, n, low_sample_warning
        pairwise_comparisons    – list of pairwise gap dicts
        overall_disagreement    – True if any eligible pair disagrees
        flagged_windows         – window names that look most outlier-ish
    """
    if np.isnan(today_value):
        return _empty_result(list(windows.keys()))

    # ── Per-window stats ─────────────────────────────────────────────
    per_window = {}
    for name, series in windows.items():
        vals = series.dropna()
        n = len(vals)
        if n == 0:
            per_window[name] = {
                "pct_rank": None, "z_score": None,
                "n": 0, "low_sample_warning": True,
            }
            continue

        pct_rank = stats.percentileofscore(vals, today_value, kind="rank")
        mean = vals.mean()
        std = vals.std()
        z_score = (today_value - mean) / std if std > 0 else 0.0

        per_window[name] = {
            "pct_rank": round(pct_rank, 1),
            "z_score": round(z_score, 2),
            "n": n,
            "low_sample_warning": n < MIN_SAMPLE_SIZE,
        }

    # ── Pairwise comparisons (only between eligible windows) ─────────
    eligible = [
        name for name, info in per_window.items()
        if not info["low_sample_warning"] and info["pct_rank"] is not None
    ]

    pairwise = []
    any_disagree = False

    for a, b in combinations(eligible, 2):
        pct_a = per_window[a]["pct_rank"]
        pct_b = per_window[b]["pct_rank"]
        z_a = per_window[a]["z_score"]
        z_b = per_window[b]["z_score"]

        pct_gap = abs(pct_a - pct_b)
        z_gap = abs(z_a - z_b)
        disagree_pct = pct_gap > pct_disagreement_threshold
        disagree_z = z_gap > z_disagreement_threshold

        # Direction: which window sees today_value as more extreme?
        if abs(z_a) >= abs(z_b):
            direction = f"more extreme on {a} than {b}"
        else:
            direction = f"more extreme on {b} than {a}"

        pair_disagree = disagree_pct or disagree_z
        if pair_disagree:
            any_disagree = True

        pairwise.append({
            "window_a": a,
            "window_b": b,
            "percentile_gap": round(pct_gap, 1),
            "z_gap": round(z_gap, 2),
            "disagree_on_percentile": disagree_pct,
            "disagree_on_zscore": disagree_z,
            "direction": direction,
        })

    # ── Flagged windows: most-outlier relative to peers ──────────────
    flagged = set()
    for comp in pairwise:
        if comp["disagree_on_percentile"] or comp["disagree_on_zscore"]:
            z_a = abs(per_window[comp["window_a"]]["z_score"])
            z_b = abs(per_window[comp["window_b"]]["z_score"])
            if z_a >= z_b:
                flagged.add(comp["window_a"])
            else:
                flagged.add(comp["window_b"])

    return {
        "per_window": per_window,
        "pairwise_comparisons": pairwise,
        "overall_disagreement": any_disagree,
        "flagged_windows": sorted(flagged),
    }


def _empty_result(window_names):
    """Return a safe empty-ish result when today_value is NaN."""
    return {
        "per_window": {
            name: {"pct_rank": None, "z_score": None, "n": 0,
                   "low_sample_warning": True}
            for name in window_names
        },
        "pairwise_comparisons": [],
        "overall_disagreement": False,
        "flagged_windows": [],
    }
