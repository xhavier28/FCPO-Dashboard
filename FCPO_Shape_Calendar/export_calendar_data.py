"""Export clean CSV data for Shape Calendar PDF generation. No matplotlib/PDF here."""

import os
import sys
import random
import calendar
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from helpers.data_loader import load_shape_log
from helpers.insight_text import SHAPE_NAMES

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SHAPE_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Raw Data", "Research", "daily_shape_log.csv"
)
ALL_SHAPES = ["0.0", "0.1", "0.2", "1", "2"]


def _longest_streak(series, shape):
    mask = series == shape
    if not mask.any():
        return 0, None
    groups = (mask != mask.shift()).cumsum()
    streak_lens = mask.groupby(groups).sum()
    max_streak = streak_lens.max()
    # Find end date of longest streak
    max_group = streak_lens.idxmax()
    end_idx = mask.index[groups == max_group][-1]
    return int(max_streak), end_idx


def _peak_month(series, dates, shape):
    mask = series == shape
    if not mask.any():
        return "N/A"
    months = dates[mask].month
    if len(months) == 0:
        return "N/A"
    return calendar.month_abbr[months.value_counts().idxmax()]


def export_daily(shape_df):
    """Export 1: daily_shape_calendar.csv — every calendar day 2017-01-01 to latest."""
    start = pd.Timestamp("2017-01-01")
    end = shape_df.index.max()
    all_days = pd.date_range(start, end, freq="D")

    out = pd.DataFrame({"date": all_days})
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["weekday"] = out["date"].dt.weekday  # 0=Mon, 6=Sun

    out = out.set_index("date")
    out["shape"] = shape_df["shape"]
    out["spot_price"] = shape_df["spot"]
    out["has_shape"] = out["shape"].notna()
    out["has_price"] = out["spot_price"].notna()
    out = out.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    path = os.path.join(OUTPUT_DIR, "daily_shape_calendar.csv")
    out.to_csv(path, index=False)
    return out


def export_yearly_summary(shape_df):
    """Export 2: yearly_summary_stats.csv."""
    rows = []
    for year in sorted(shape_df.index.year.unique()):
        yr_data = shape_df[shape_df.index.year == year]
        shapes = yr_data["shape"].dropna()
        total = len(shapes)
        row = {"year": year, "total_trading_days": total}

        pcts = {}
        for s in ALL_SHAPES:
            count = (shapes == s).sum()
            pct = round(100 * count / total, 1) if total > 0 else 0
            streak, _ = _longest_streak(shapes, s)
            peak = _peak_month(shapes, shapes.index, s)
            row[f"pct_{s}"] = pct
            row[f"longest_streak_{s}"] = streak
            row[f"peak_month_{s}"] = peak
            pcts[s] = pct

        top_pct = max(pcts.values())
        if top_pct >= 40:
            row["dominant_shape"] = max(pcts, key=pcts.get)
        else:
            row["dominant_shape"] = "split"

        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "yearly_summary_stats.csv")
    df.to_csv(path, index=False)
    return df


def export_cross_year(shape_df):
    """Export 3: shape_calendar_cross_year.csv (re-verify)."""
    rows = []
    for year in sorted(shape_df.index.year.unique()):
        yr_data = shape_df[shape_df.index.year == year]["shape"].dropna()
        total = len(yr_data)
        row = {"Year": year}
        for s in ALL_SHAPES:
            name = SHAPE_NAMES.get(s, s)
            count = (yr_data == s).sum()
            row[f"{s} ({name})"] = round(100 * count / total, 1) if total > 0 else 0
        row["Total Days"] = total
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "shape_calendar_cross_year.csv")
    df.to_csv(path, index=False)
    return df


def export_monthly_aggregate(shape_df):
    """Export 4: shape_calendar_monthly_aggregate.csv (re-verify)."""
    rows = []
    for m in range(1, 13):
        m_data = shape_df[shape_df.index.month == m]["shape"].dropna()
        total = len(m_data)
        row = {"Month": calendar.month_abbr[m]}
        for s in ALL_SHAPES:
            name = SHAPE_NAMES.get(s, s)
            count = (m_data == s).sum()
            row[f"{s} ({name})"] = round(100 * count / total, 1) if total > 0 else 0
        row["Total Days"] = total
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "shape_calendar_monthly_aggregate.csv")
    df.to_csv(path, index=False)
    return df


def export_narrative_facts(shape_df):
    """Export 5: yearly_narrative_facts.csv — raw facts for Claude to phrase."""
    rows = []
    for year in sorted(shape_df.index.year.unique()):
        yr_data = shape_df[shape_df.index.year == year]["shape"].dropna()
        total = len(yr_data)
        if total == 0:
            continue

        counts = yr_data.value_counts(normalize=True)
        top = counts.index[0]
        second = counts.index[1] if len(counts) > 1 else ""
        rarest = counts.index[-1] if len(counts) > 1 else top

        # Longest streak across all shapes
        best_streak = 0
        best_shape = ""
        best_end = None
        for s in counts.index:
            streak, end_date = _longest_streak(yr_data, s)
            if streak > best_streak:
                best_streak = streak
                best_shape = s
                best_end = end_date

        row = {
            "year": year,
            "top_shape": top,
            "top_shape_pct": round(counts.iloc[0] * 100, 1),
            "second_shape": second,
            "second_shape_pct": round(counts.iloc[1] * 100, 1) if len(counts) > 1 else 0,
            "rarest_shape": rarest,
            "rarest_shape_pct": round(counts.iloc[-1] * 100, 1) if len(counts) > 1 else round(counts.iloc[0] * 100, 1),
            "rarest_shape_peak_month": _peak_month(yr_data, yr_data.index, rarest),
            "longest_streak_days": best_streak,
            "longest_streak_shape": best_shape,
            "longest_streak_end_month": calendar.month_abbr[best_end.month] if best_end is not None else "N/A",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "yearly_narrative_facts.csv")
    df.to_csv(path, index=False)
    return df


def integrity_checks(daily_df, shape_df):
    """Run and print integrity checks."""
    print("\n" + "=" * 70)
    print("INTEGRITY CHECKS")
    print("=" * 70)

    # 1. Row count and date range
    print(f"\n1. daily_shape_calendar.csv: {len(daily_df)} rows")
    print(f"   Date range: {daily_df['date'].iloc[0]} to {daily_df['date'].iloc[-1]}")

    # 2. Days with has_shape=False by year
    print("\n2. Days with has_shape=False by year:")
    no_shape = daily_df[~daily_df["has_shape"]]
    for year in sorted(daily_df["year"].unique()):
        count = len(no_shape[no_shape["year"] == year])
        total = len(daily_df[daily_df["year"] == year])
        print(f"   {year}: {count} / {total} days without shape")

    # 3. Confirm dtype
    raw = pd.read_csv(SHAPE_LOG_PATH, dtype={"shape": str}, nrows=3)
    print(f"\n3. shape dtype in raw read: {raw['shape'].dtype} (confirmed str)")

    # 4. Spot-check 3 random dates
    print("\n4. Spot-check 3 random dates:")
    raw_full = pd.read_csv(SHAPE_LOG_PATH, dtype={"shape": str}, parse_dates=["date"])
    raw_full = raw_full[raw_full["date"].dt.year >= 2017].set_index("date")
    trading_dates = daily_df[daily_df["has_shape"]].sample(3, random_state=42)
    for _, row in trading_dates.iterrows():
        dt = pd.Timestamp(row["date"])
        if dt in raw_full.index:
            orig_shape = str(raw_full.loc[dt, "shape"])
            orig_m1 = raw_full.loc[dt, "M1"]
            match_s = "OK" if orig_shape == row["shape"] else "MISMATCH"
            match_p = "OK" if abs(orig_m1 - row["spot_price"]) < 0.01 else "MISMATCH"
            print(f"   {row['date']}: shape={row['shape']} vs orig={orig_shape} [{match_s}], "
                  f"price={row['spot_price']} vs orig={orig_m1} [{match_p}]")
        else:
            print(f"   {row['date']}: NOT FOUND in raw shape_log")


def main():
    print("Loading shape log (dtype={'shape': str})...")
    shape_df = load_shape_log(start_year=2017)
    print(f"  Loaded: {len(shape_df)} rows, {shape_df.index.min().date()} to {shape_df.index.max().date()}")
    print(f"  Unique shapes: {sorted(shape_df['shape'].dropna().unique())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nExporting CSVs...")

    daily_df = export_daily(shape_df)
    print(f"  1. daily_shape_calendar.csv — {len(daily_df)} rows")

    yearly_df = export_yearly_summary(shape_df)
    print(f"  2. yearly_summary_stats.csv — {len(yearly_df)} rows")

    cross_year_df = export_cross_year(shape_df)
    print(f"  3. shape_calendar_cross_year.csv — {len(cross_year_df)} rows")

    monthly_df = export_monthly_aggregate(shape_df)
    print(f"  4. shape_calendar_monthly_aggregate.csv — {len(monthly_df)} rows")

    narrative_df = export_narrative_facts(shape_df)
    print(f"  5. yearly_narrative_facts.csv — {len(narrative_df)} rows")

    integrity_checks(daily_df, shape_df)

    print("\n" + "=" * 70)
    print("ALL EXPORTS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
