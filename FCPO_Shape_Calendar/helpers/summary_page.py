"""Draw combined multi-year summary pages for the front of the PDF."""

import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .insight_text import SHAPE_NAMES

SHAPE_COLOR_MAP = {
    "0.0": "#1D9E75",
    "0.1": "#EF9F27",
    "0.2": "#D85A30",
    "1":   "#7F77DD",
    "2":   "#888780",
}
FIGSIZE = (13, 8.5)


def _draw_table(ax, df, title, title_y=1.05):
    """Draw a DataFrame as a colored table on the axes."""
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", color="#fafafa", pad=10, loc="left")

    n_rows, n_cols = df.shape
    col_width = 1.0 / n_cols
    row_height = 1.0 / (n_rows + 1)

    # Header
    for j, col in enumerate(df.columns):
        ax.text(j * col_width + col_width / 2, 1.0 - row_height / 2, str(col),
                ha="center", va="center", fontsize=6, fontweight="bold", color="#fafafa",
                transform=ax.transAxes)

    # Rows
    for i in range(n_rows):
        for j in range(n_cols):
            val = df.iloc[i, j]
            x = j * col_width + col_width / 2
            y = 1.0 - (i + 1.5) * row_height

            # Color the cell background for shape columns
            if j > 0 and isinstance(val, (int, float)):
                # Intensity based on value
                intensity = min(val / 60, 1.0)  # cap at 60%
                col_name = df.columns[j]
                shape_key = col_name.split(" ")[0] if " " in col_name else col_name
                base_color = SHAPE_COLOR_MAP.get(shape_key, "#444")
                rect = plt.Rectangle(
                    (j * col_width, y - row_height / 2), col_width, row_height,
                    facecolor=base_color, alpha=intensity * 0.6 + 0.05,
                    transform=ax.transAxes, zorder=0
                )
                ax.add_patch(rect)

            text = f"{val:.1f}%" if isinstance(val, float) else str(val)
            ax.text(x, y, text, ha="center", va="center", fontsize=5.5,
                    color="#fafafa", transform=ax.transAxes)


def draw_summary_pages(pdf: PdfPages, shape_df: pd.DataFrame):
    """Draw 1-2 summary pages at the front of the PDF."""
    shape_df = shape_df.copy()
    shape_df["year"] = shape_df.index.year
    shape_df["month"] = shape_df.index.month

    all_shapes = sorted(shape_df["shape"].dropna().unique())

    # === Page 1: Cross-year table + Month aggregate ===
    fig = plt.figure(figsize=FIGSIZE)
    fig.set_facecolor("#0e1117")
    fig.text(0.5, 0.97, "FCPO Shape Calendar — Cross-Year Summary",
             ha="center", fontsize=14, fontweight="bold", color="#fafafa")

    gs = fig.add_gridspec(3, 1, height_ratios=[40, 40, 20], hspace=0.35,
                          top=0.93, bottom=0.05, left=0.05, right=0.95)

    # --- Cross-year table ---
    years = sorted(shape_df["year"].unique())
    rows = []
    for yr in years:
        yr_data = shape_df[shape_df["year"] == yr]["shape"].dropna()
        total = len(yr_data)
        row = {"Year": yr}
        for s in all_shapes:
            name = SHAPE_NAMES.get(s, s)
            count = (yr_data == s).sum()
            pct = 100 * count / total if total > 0 else 0
            row[f"{s} ({name})"] = round(pct, 1)
        row["Total Days"] = total
        rows.append(row)
    cross_year_df = pd.DataFrame(rows)

    ax1 = fig.add_subplot(gs[0])
    _draw_table(ax1, cross_year_df, "Shape Distribution by Year (% of Trading Days)")

    # --- Month aggregate table ---
    rows_m = []
    for m in range(1, 13):
        m_data = shape_df[shape_df["month"] == m]["shape"].dropna()
        total = len(m_data)
        row = {"Month": calendar.month_abbr[m]}
        for s in all_shapes:
            name = SHAPE_NAMES.get(s, s)
            count = (m_data == s).sum()
            pct = 100 * count / total if total > 0 else 0
            row[f"{s} ({name})"] = round(pct, 1)
        row["Total Days"] = total
        rows_m.append(row)
    month_agg_df = pd.DataFrame(rows_m)

    ax2 = fig.add_subplot(gs[1])
    _draw_table(ax2, month_agg_df, "Shape Distribution by Month (Pooled Across All Years)")

    # --- Narrative synthesis ---
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    narrative = _generate_summary_narrative(cross_year_df, month_agg_df, all_shapes)
    for i, sent in enumerate(narrative):
        ax3.text(0.0, 0.9 - i * 0.18, sent, fontsize=7, va="top", color="#ddd",
                 style="italic", transform=ax3.transAxes, wrap=True)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Return tables for console output
    return cross_year_df, month_agg_df


def _generate_summary_narrative(cross_year_df, month_agg_df, all_shapes):
    """Generate 3-6 cross-year synthesis sentences."""
    sentences = []

    # Find most common shape overall
    shape_cols = [c for c in cross_year_df.columns if c not in ("Year", "Total Days")]
    avg_pcts = {col: cross_year_df[col].mean() for col in shape_cols}
    dominant_col = max(avg_pcts, key=avg_pcts.get)
    sentences.append(
        f"Across all years, {dominant_col} was the most common curve shape, averaging "
        f"{avg_pcts[dominant_col]:.1f}% of trading days per year."
    )

    # Monthly seasonality check
    month_cols = [c for c in month_agg_df.columns if c not in ("Month", "Total Days")]
    for col in month_cols:
        vals = month_agg_df[col].values
        if vals.max() > 45:  # Strong monthly concentration
            peak_idx = vals.argmax()
            peak_month = month_agg_df.iloc[peak_idx]["Month"]
            sentences.append(
                f"{col} peaks in {peak_month} ({vals.max():.1f}% of that month's "
                f"days pooled across all years)."
            )

    # Check for consistent seasonal pattern
    shape_std = {col: month_agg_df[col].std() for col in month_cols}
    most_variable = max(shape_std, key=shape_std.get)
    least_variable = min(shape_std, key=shape_std.get)

    if shape_std[most_variable] > 10:
        sentences.append(
            f"{most_variable} shows the strongest seasonal variation "
            f"(monthly std = {shape_std[most_variable]:.1f}pp), suggesting possible "
            f"calendar-driven clustering."
        )
    else:
        sentences.append(
            "No shape shows strong seasonal variation — monthly distributions are "
            "relatively flat across the calendar year."
        )

    if shape_std[least_variable] < 5:
        sentences.append(
            f"{least_variable} is the most evenly distributed across months "
            f"(monthly std = {shape_std[least_variable]:.1f}pp)."
        )

    # Year-to-year stability
    for col in shape_cols[:2]:  # top 2 shapes
        yr_std = cross_year_df[col].std()
        if yr_std > 15:
            sentences.append(
                f"{col} varies substantially across years (yearly std = {yr_std:.1f}pp), "
                f"indicating year-specific drivers beyond pure seasonality."
            )

    return sentences[:6]
