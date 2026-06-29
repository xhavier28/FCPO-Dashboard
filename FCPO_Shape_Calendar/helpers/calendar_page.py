"""Draw one year's landscape calendar page with shape-colored grid + price chart."""

import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from .insight_text import compute_year_stats, generate_year_narrative, SHAPE_NAMES

SHAPE_COLOR_MAP = {
    "0.0": "#1D9E75",
    "0.1": "#EF9F27",
    "0.2": "#D85A30",
    "1":   "#7F77DD",
    "2":   "#888780",
}
NO_DATA_COLOR = "#E5E5E5"

FIGSIZE = (13, 8.5)  # US Letter landscape


def _draw_month_block(ax, year, month, shape_data, spot_data):
    """Draw a single month's calendar grid on the given axes."""
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Month title
    ax.text(3.5, 6.6, calendar.month_abbr[month], ha="center", va="center",
            fontsize=8, fontweight="bold")

    # Day-of-week headers
    for i, d in enumerate(["M", "T", "W", "T", "F", "S", "S"]):
        ax.text(i + 0.5, 6.1, d, ha="center", va="center", fontsize=5, color="#888")

    cal = calendar.Calendar(firstweekday=0)
    weeks = cal.monthdayscalendar(year, month)

    for wi, week in enumerate(weeks):
        for di, day in enumerate(week):
            if day == 0:
                continue
            x = di
            y = 5.5 - wi
            date = pd.Timestamp(year, month, day)

            # Determine color
            if date in shape_data.index and pd.notna(shape_data.loc[date]):
                shape = str(shape_data.loc[date])
                color = SHAPE_COLOR_MAP.get(shape, NO_DATA_COLOR)
            else:
                color = NO_DATA_COLOR

            rect = plt.Rectangle((x, y), 0.95, 0.85, facecolor=color,
                                  edgecolor="white", linewidth=0.3)
            ax.add_patch(rect)

            # Day number
            ax.text(x + 0.12, y + 0.65, str(day), ha="left", va="center",
                    fontsize=4, color="white" if color != NO_DATA_COLOR else "#666")

            # Spot price (small)
            if date in spot_data.index and pd.notna(spot_data.loc[date]):
                price = spot_data.loc[date]
                ax.text(x + 0.48, y + 0.3, f"{price:.0f}", ha="center", va="center",
                        fontsize=3, color="white" if color != NO_DATA_COLOR else "#999",
                        alpha=0.8)


def draw_year_page(pdf: PdfPages, year: int, shape_series: pd.Series,
                   spot_series: pd.Series):
    """Draw a full landscape page for one year and save to pdf."""
    fig = plt.figure(figsize=FIGSIZE)

    # === Top half: Calendar grid (4 cols x 3 rows) ===
    # Use gridspec for layout: top 55% calendar, bottom 35% chart, 10% insights
    gs = fig.add_gridspec(3, 1, height_ratios=[55, 30, 15], hspace=0.3,
                          top=0.95, bottom=0.02, left=0.03, right=0.97)

    # Calendar grid area
    gs_cal = gs[0].subgridspec(3, 4, hspace=0.15, wspace=0.08)

    for month in range(1, 13):
        row = (month - 1) // 4
        col = (month - 1) % 4
        ax = fig.add_subplot(gs_cal[row, col])
        _draw_month_block(ax, year, month, shape_series, spot_series)

    # Year title + legend
    fig.text(0.5, 0.97, f"FCPO Shape Calendar — {year}", ha="center", va="center",
             fontsize=14, fontweight="bold")

    # Legend
    legend_x = 0.03
    for i, (shape, color) in enumerate(SHAPE_COLOR_MAP.items()):
        name = SHAPE_NAMES.get(shape, shape)
        fig.patches.append(mpatches.FancyBboxPatch(
            (legend_x + i * 0.14, 0.955), 0.012, 0.012,
            boxstyle="round,pad=0.002", facecolor=color,
            transform=fig.transFigure, figure=fig
        ))
        fig.text(legend_x + i * 0.14 + 0.018, 0.961, f"{shape} ({name})",
                 fontsize=5, va="center")
    # No data legend
    fig.patches.append(mpatches.FancyBboxPatch(
        (legend_x + 5 * 0.14, 0.955), 0.012, 0.012,
        boxstyle="round,pad=0.002", facecolor=NO_DATA_COLOR,
        transform=fig.transFigure, figure=fig
    ))
    fig.text(legend_x + 5 * 0.14 + 0.018, 0.961, "No data",
             fontsize=5, va="center")

    # === Bottom half: Price line chart with shape shading ===
    ax_price = fig.add_subplot(gs[1])

    year_spot = spot_series[spot_series.index.year == year].sort_index()
    year_shapes = shape_series[shape_series.index.year == year].sort_index()

    if len(year_spot) > 0:
        ax_price.plot(year_spot.index, year_spot.values, color="white",
                      linewidth=0.8, zorder=3)

        # Shape background shading
        all_dates = year_shapes.index
        for i in range(len(all_dates)):
            date = all_dates[i]
            shape = str(year_shapes.iloc[i])
            if shape in SHAPE_COLOR_MAP:
                next_date = all_dates[i + 1] if i + 1 < len(all_dates) else date + pd.Timedelta(days=1)
                ax_price.axvspan(date, next_date, facecolor=SHAPE_COLOR_MAP[shape],
                                alpha=0.3, zorder=1)

        ax_price.set_facecolor("#1a1a2e")
        ax_price.tick_params(labelsize=6, colors="#ccc")
        ax_price.set_ylabel("M1 Price (MYR)", fontsize=7, color="#ccc")
        ax_price.spines["top"].set_visible(False)
        ax_price.spines["right"].set_visible(False)
        for spine in ax_price.spines.values():
            spine.set_color("#444")
        ax_price.grid(axis="y", alpha=0.2, color="#666")
    else:
        ax_price.text(0.5, 0.5, "No spot price data available", ha="center",
                      va="center", fontsize=10, color="#888", transform=ax_price.transAxes)
        ax_price.set_facecolor("#1a1a2e")
        ax_price.axis("off")

    # === Insight block ===
    ax_insight = fig.add_subplot(gs[2])
    ax_insight.axis("off")

    valid_shapes = year_shapes.dropna()
    if len(valid_shapes) > 0:
        stats = compute_year_stats(valid_shapes, valid_shapes.index)
        narrative = generate_year_narrative(valid_shapes, valid_shapes.index, year)

        # Stats table as text
        stats_text = "  |  ".join(
            f"{row['Shape']}: {row['% Days']}%, streak {row['Longest Streak']}d, peak {row['Peak Month']}"
            for _, row in stats.iterrows()
        )
        ax_insight.text(0.0, 0.85, stats_text, fontsize=5, va="top",
                        family="monospace", color="#ccc",
                        transform=ax_insight.transAxes)

        # Narrative
        for i, sent in enumerate(narrative):
            ax_insight.text(0.0, 0.55 - i * 0.22, sent, fontsize=5.5, va="top",
                            color="#ddd", style="italic", transform=ax_insight.transAxes,
                            wrap=True)

    fig.set_facecolor("#0e1117")
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)
