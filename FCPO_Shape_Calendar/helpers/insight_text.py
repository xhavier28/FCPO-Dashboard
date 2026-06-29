"""Generate narrative insight sentences for a given year's shape data."""

import calendar
import pandas as pd


SHAPE_NAMES = {
    "0.0": "Contango",
    "0.1": "Mild Contango",
    "0.2": "Steep Backwardation",
    "1": "Backwardation",
    "2": "Flat",
}

QUARTER_LABELS = {1: "Q1 (Jan-Mar)", 2: "Q2 (Apr-Jun)", 3: "Q3 (Jul-Sep)", 4: "Q4 (Oct-Dec)"}


def _longest_streak(series: pd.Series, shape: str) -> int:
    """Longest consecutive run of `shape` in the series."""
    mask = series == shape
    if not mask.any():
        return 0
    groups = (mask != mask.shift()).cumsum()
    return mask.groupby(groups).sum().max()


def _peak_month(series: pd.Series, dates: pd.DatetimeIndex, shape: str) -> str:
    """Calendar month where shape appears most often."""
    mask = series == shape
    if not mask.any():
        return "N/A"
    months = dates[mask].month
    if len(months) == 0:
        return "N/A"
    peak = months.value_counts().idxmax()
    return calendar.month_abbr[peak]


def compute_year_stats(shapes: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return summary stats DataFrame: % days, longest streak, peak month per shape."""
    total = len(shapes)
    rows = []
    for shape in sorted(shapes.unique()):
        count = (shapes == shape).sum()
        pct = round(100 * count / total, 1) if total > 0 else 0
        streak = _longest_streak(shapes, shape)
        peak = _peak_month(shapes, dates, shape)
        name = SHAPE_NAMES.get(shape, shape)
        rows.append({
            "Shape": f"{shape} ({name})",
            "% Days": pct,
            "Longest Streak": streak,
            "Peak Month": peak,
        })
    return pd.DataFrame(rows)


def generate_year_narrative(shapes: pd.Series, dates: pd.DatetimeIndex, year: int) -> list[str]:
    """Generate 2-4 descriptive sentences about the year's shape distribution."""
    total = len(shapes)
    if total == 0:
        return [f"{year}: No classified trading days available."]

    counts = shapes.value_counts(normalize=True)
    sentences = []

    # Dominant shape
    dominant = counts.index[0]
    dom_pct = counts.iloc[0] * 100
    dom_name = SHAPE_NAMES.get(dominant, dominant)
    if dom_pct > 50:
        # Find which quarter it was most concentrated
        quarters = dates[shapes == dominant].quarter
        q_counts = quarters.value_counts()
        top_q = q_counts.idxmax()
        q_label = QUARTER_LABELS.get(top_q, f"Q{top_q}")
        sentences.append(
            f"Shape {dominant} ({dom_name}) dominated {year} at {dom_pct:.0f}% of trading days, "
            f"most concentrated in {q_label}."
        )
    else:
        top2 = counts.head(2)
        names = [f"{s} ({SHAPE_NAMES.get(s, s)})" for s in top2.index]
        pcts = [f"{p*100:.0f}%" for p in top2.values]
        sentences.append(
            f"No single shape dominated {year}; the year was split between "
            f"Shape {names[0]} ({pcts[0]}) and Shape {names[1]} ({pcts[1]})."
        )

    # Rarest shape
    if len(counts) > 1:
        rarest = counts.index[-1]
        rare_pct = counts.iloc[-1] * 100
        rare_name = SHAPE_NAMES.get(rarest, rarest)
        if rare_pct < 10:
            rare_months = dates[shapes == rarest].month
            if len(rare_months) > 0:
                peak_m = calendar.month_name[rare_months.value_counts().idxmax()]
                sentences.append(
                    f"Shape {rarest} ({rare_name}) appeared only {rare_pct:.0f}% of the time, "
                    f"mostly in {peak_m}."
                )

    # Longest streak
    best_streak = 0
    best_shape = ""
    for shape in counts.index:
        s = _longest_streak(shapes, shape)
        if s > best_streak:
            best_streak = s
            best_shape = shape
    if best_streak >= 10:
        bs_name = SHAPE_NAMES.get(best_shape, best_shape)
        sentences.append(
            f"The longest uninterrupted streak was {best_streak} consecutive days "
            f"of Shape {best_shape} ({bs_name})."
        )

    return sentences[:4]
