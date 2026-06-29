"""Load shape_log.csv and M1 spot price from term structure CSVs."""

import os
import re
import calendar
import pandas as pd

SHAPE_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Raw Data", "Research", "daily_shape_log.csv"
)
TERM_STRUCTURE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "Raw Data", "Term Structure"
)

MONTH_ABBR = {v: k for k, v in enumerate(calendar.month_abbr) if k}


def load_shape_log(start_year: int = 2017) -> pd.DataFrame:
    """Load shape log, filter to start_year+, return with date index."""
    df = pd.read_csv(SHAPE_LOG_PATH, dtype={"shape": str}, parse_dates=["date"])
    df = df[df["date"].dt.year >= start_year].copy()
    df = df[["date", "shape", "M1"]].rename(columns={"M1": "spot"})
    df["shape"] = df["shape"].astype(str).str.strip()
    df = df.set_index("date").sort_index()
    return df


def _front_month(date):
    """Roll to next month on the 16th."""
    if date.day <= 15:
        return date.year, date.month
    else:
        if date.month == 12:
            return date.year + 1, 1
        return date.year, date.month + 1


def load_spot_prices(start_year: int = 2017) -> pd.Series:
    """Load M1 settlement price from term structure CSVs for each trading day."""
    records = []
    for year_folder in sorted(os.listdir(TERM_STRUCTURE_DIR)):
        year_path = os.path.join(TERM_STRUCTURE_DIR, year_folder)
        if not os.path.isdir(year_path):
            continue
        try:
            yr = int(year_folder)
        except ValueError:
            continue
        if yr < start_year - 1:
            continue

        for fname in os.listdir(year_path):
            m = re.match(r"FCPO (\w{3})(\d{2})_Daily\.csv", fname)
            if not m:
                continue
            mon_abbr, yy = m.group(1), m.group(2)
            contract_month = MONTH_ABBR.get(mon_abbr)
            contract_year = 2000 + int(yy)
            if contract_month is None:
                continue

            fpath = os.path.join(year_path, fname)
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue

            if "Timestamp (UTC)" not in df.columns or "Close" not in df.columns:
                continue

            df["date"] = pd.to_datetime(df["Timestamp (UTC)"], errors="coerce")
            df = df.dropna(subset=["date", "Close"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Close"])

            for _, row in df.iterrows():
                fm_year, fm_month = _front_month(row["date"])
                if fm_year == contract_year and fm_month == contract_month:
                    records.append({"date": row["date"], "spot": row["Close"]})

    if not records:
        return pd.Series(dtype=float)

    spot = pd.DataFrame(records)
    spot = spot.drop_duplicates(subset="date", keep="last")
    spot = spot.set_index("date").sort_index()
    spot = spot[spot.index.year >= start_year]
    return spot["spot"]
