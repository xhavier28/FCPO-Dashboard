"""
shared/fx_converter.py — FX conversion for Asset B (SOY → MYR).

SOY conversion formula:
    SOY_MYR = close × 0.01 × 2204.62 × USDMYR_rate

Adapted from mr_screener/data/fx_convert.py.
Added: build_empty_rate_table()
"""

import io
import os

import numpy as np
import pandas as pd


# ── Rate table helpers ────────────────────────────────────────────────────────

def build_empty_rate_table(date_series: pd.Series) -> pd.DataFrame:
    """
    Auto-detect calendar months from a date series and return an editable
    DataFrame with columns [Month, USDMYR_Rate] pre-filled with month labels
    and NaN rates.

    Parameters
    ----------
    date_series : Series with datetime-like values

    Returns
    -------
    DataFrame with columns ['Month', 'USDMYR_Rate']
    """
    dates = pd.to_datetime(date_series)
    periods = dates.dt.to_period("M").unique()
    periods = sorted(periods)

    df = pd.DataFrame({
        "Month":       [str(p) for p in periods],
        "USDMYR_Rate": [np.nan] * len(periods),
    })
    return df


def load_rate_table(source) -> pd.DataFrame:
    """
    Load a monthly FX rate CSV.

    Expected format (header row required):
        month,rate
        2022-01,4.19
        2022-02,4.21
        ...

    `source` may be:
      - str / os.PathLike  → read from disk
      - bytes              → wrap in io.BytesIO
      - file-like object   → use directly
      - DataFrame with columns ['Month', 'USDMYR_Rate']  → parsed directly
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        # Normalise column names: "Month"→"month", "USDMYR_Rate"→"rate"
        col_map = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc == "month":
                col_map[c] = "month"
            elif "rate" in lc:
                col_map[c] = "rate"
        df = df.rename(columns=col_map)
        df["month"] = pd.to_datetime(df["month"].astype(str), format="%Y-%m").dt.to_period("M")
        df["rate"]  = pd.to_numeric(df["rate"], errors="coerce")
        df = df.dropna(subset=["rate"])
        return df.set_index("month")[["rate"]].sort_index()

    def _try_read(f, sep=","):
        return pd.read_csv(f, sep=sep)

    if isinstance(source, (str, os.PathLike)):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Rate table not found: {source}")
        try:
            df = _try_read(source)
        except Exception:
            df = _try_read(source, sep=";")
    else:
        raw = source if isinstance(source, bytes) else source.read()
        try:
            df = _try_read(io.BytesIO(raw))
        except Exception:
            df = _try_read(io.BytesIO(raw), sep=";")

    df.columns = [c.strip().lower() for c in df.columns]
    if "month" not in df.columns or "rate" not in df.columns:
        raise ValueError(
            f"Rate table must have 'month' and 'rate' columns. Got: {list(df.columns)}"
        )

    df["month"] = pd.to_datetime(df["month"], format="%Y-%m").dt.to_period("M")
    df["rate"]  = pd.to_numeric(df["rate"], errors="raise")
    df = df.set_index("month")[["rate"]].sort_index()

    print(f"  [FX TABLE] Loaded {len(df)} monthly rates: "
          f"{df.index[0]} -> {df.index[-1]}  "
          f"range [{df['rate'].min():.4f}, {df['rate'].max():.4f}]")

    return df


def apply_fx_conversion(
    x: pd.Series,
    rate_table: pd.DataFrame,
    label_x: str = "Asset B",
) -> tuple:
    """
    Multiply each bar of `x` by the rate for its calendar month.

    Returns (x_converted, conversion_log).
    """
    x_periods = x.index.to_period("M")
    rates = x_periods.map(lambda p: rate_table["rate"].get(p, None))
    rates = pd.Series(rates, index=x.index, dtype=float)

    table_months   = set(rate_table.index)
    data_months    = set(x_periods)
    missing_months = sorted(data_months - table_months)
    fallback_rate  = float(rate_table["rate"].mean())

    if missing_months:
        n_missing = len(missing_months)
        print(f"  [FX WARN] {n_missing} month(s) in {label_x} not in rate table -> "
              f"fallback {fallback_rate:.4f}")
        for m in missing_months:
            print(f"           {m}")
        rates = rates.fillna(fallback_rate)
    else:
        n_missing = 0

    x_converted = x * rates

    conversion_log = {
        "n_table_months":   len(rate_table),
        "n_missing_months": n_missing,
        "missing_months":   [str(m) for m in missing_months],
        "fallback_rate":    fallback_rate,
        "rates_min":        float(rate_table["rate"].min()),
        "rates_max":        float(rate_table["rate"].max()),
        "rates_mean":       float(rate_table["rate"].mean()),
    }

    print(f"  [FX CONVERT] {label_x}: {len(x)} bars  "
          f"rate range [{conversion_log['rates_min']:.4f}, {conversion_log['rates_max']:.4f}]")

    return x_converted, conversion_log


def apply_soy_conversion(
    x: pd.Series,
    rate_table: pd.DataFrame,
    label_x: str = "SOY",
) -> tuple:
    """
    Convert SOY (CBOT cents/bushel) to MYR/MT:
        SOY_MYR = close × 0.01 × 2204.62 × USDMYR_rate

    Returns (x_converted, conversion_log).
    """
    # First scale cents/bushel → USD/MT (0.01 × 2204.62)
    x_usd_per_mt = x * 0.01 * 2204.62
    return apply_fx_conversion(x_usd_per_mt, rate_table, label_x=label_x)


def rate_table_from_editor(editor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert st.data_editor output (columns: Month, USDMYR_Rate) to indexed rate table.
    Drops rows with NaN rates.
    """
    df = editor_df.copy()
    df.columns = ["month", "rate"]
    df["month"] = pd.to_datetime(df["month"].astype(str), format="%Y-%m").dt.to_period("M")
    df["rate"]  = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["rate"])
    return df.set_index("month")[["rate"]].sort_index()
