"""Build FCPO Shape Calendar PDF — standalone script, no Streamlit dependency."""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages

# Ensure helpers are importable
sys.path.insert(0, os.path.dirname(__file__))

from helpers.data_loader import load_shape_log
from helpers.calendar_page import draw_year_page
from helpers.summary_page import draw_summary_pages

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "FCPO_Shape_Calendar_2017_to_present.pdf")


def main():
    print("Loading shape log...")
    shape_df = load_shape_log(start_year=2017)
    print(f"  Shape log: {len(shape_df)} rows, {shape_df.index.min().date()} to {shape_df.index.max().date()}")
    print(f"  Unique shapes: {sorted(shape_df['shape'].dropna().unique())}")

    # Use M1 from shape_log directly (already has spot price)
    spot_series = shape_df["spot"].dropna()
    shape_series = shape_df["shape"]
    print(f"  Spot prices from shape_log: {len(spot_series)} rows")

    years = sorted(shape_df.index.year.unique())
    print(f"  Years: {years}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with PdfPages(OUTPUT_PATH) as pdf:
        # Summary pages first
        print("\nBuilding summary pages...")
        cross_year_df, month_agg_df = draw_summary_pages(pdf, shape_df)
        print("  Summary pages built.")

        # Per-year pages
        for year in years:
            year_shapes = shape_series[shape_series.index.year == year]
            year_spots = spot_series[spot_series.index.year == year]
            classified = year_shapes.dropna()
            total_days = len(year_shapes)
            missing = total_days - len(classified)

            draw_year_page(pdf, year, shape_series, spot_series)
            print(f"  {year} page built: {len(classified)} trading days classified, {missing} missing")

    print(f"\nPDF saved to: {OUTPUT_PATH}")

    # Print tables for console review
    print("\n" + "=" * 80)
    print("CROSS-YEAR SUMMARY TABLE (% of trading days per shape)")
    print("=" * 80)
    print(cross_year_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("MONTHLY AGGREGATE TABLE (pooled across all years)")
    print("=" * 80)
    print(month_agg_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
