"""
fx_convert.py — Optional FX conversion for Asset B.

Asset A (y) is never modified.
Asset B (x) is multiplied bar-by-bar by the monthly rate.
"""

import io
import os

import pandas as pd


# ── Rate table loader ─────────────────────────────────────────────────────────

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
    """
    def _try_read(file_or_path, sep=","):
        return pd.read_csv(file_or_path, sep=sep)

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

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    if "month" not in df.columns or "rate" not in df.columns:
        raise ValueError(
            f"Rate table must have 'month' and 'rate' columns. Got: {list(df.columns)}"
        )

    # Parse month as period
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m").dt.to_period("M")
    df["rate"]  = pd.to_numeric(df["rate"], errors="raise")
    df = df.set_index("month")[["rate"]].sort_index()

    print(f"  [FX TABLE] Loaded {len(df)} monthly rates: "
          f"{df.index[0]} -> {df.index[-1]}  "
          f"range [{df['rate'].min():.4f}, {df['rate'].max():.4f}]")

    return df


# ── Bar-by-bar conversion ─────────────────────────────────────────────────────

def apply_fx_conversion(
    x: pd.Series,
    rate_table: pd.DataFrame,
    label_x: str = "Asset B",
) -> tuple:
    """
    Multiply each bar of `x` by the rate for its calendar month.

    Returns
    -------
    x_converted : pd.Series
    conversion_log : dict
    """
    x_periods = x.index.to_period("M")

    # Build a rate series aligned to x's index
    rates = x_periods.map(lambda p: rate_table["rate"].get(p, None))
    rates = pd.Series(rates, index=x.index, dtype=float)

    table_months = set(rate_table.index)
    data_months  = set(x_periods)
    missing_months = sorted(data_months - table_months)

    fallback_rate = float(rate_table["rate"].mean())

    if missing_months:
        n_missing = len(missing_months)
        print(f"  [FX WARN] {n_missing} month(s) in {label_x} not in rate table -> "
              f"fallback {fallback_rate:.4f} used:")
        for m in missing_months:
            print(f"           {m}")
        rates = rates.fillna(fallback_rate)
    else:
        n_missing = 0

    x_converted = x * rates

    conversion_log = {
        "n_table_months":  len(rate_table),
        "n_missing_months": n_missing,
        "missing_months":  [str(m) for m in missing_months],
        "fallback_rate":   fallback_rate,
        "rates_min":       float(rate_table["rate"].min()),
        "rates_max":       float(rate_table["rate"].max()),
        "rates_mean":      float(rate_table["rate"].mean()),
    }

    print(f"  [FX CONVERT] {label_x}: {len(x)} bars converted  "
          f"rate range [{conversion_log['rates_min']:.4f}, {conversion_log['rates_max']:.4f}]  "
          f"mean {conversion_log['rates_mean']:.4f}")

    return x_converted, conversion_log


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_fx_pipeline(
    y: pd.Series,
    x: pd.Series,
    label_y: str = "Asset A",
    label_x: str = "Asset B",
    apply_fx: bool = False,
    rate_table_source=None,
) -> dict:
    """
    Optionally convert Asset B via FX rates.

    Parameters
    ----------
    y, x              : raw price series (post-multiplier, pre-alignment)
    apply_fx          : user checkbox value
    rate_table_source : path string, bytes, or file-like; may be None

    Returns
    -------
    dict with keys:
        y             : (unchanged) Asset A series
        x             : Asset B series, converted if apply_fx=True and table present
        fx_applied    : bool
        conversion_log: dict (populated only when fx_applied=True)
    """
    if not apply_fx:
        print(f"  [FX] apply_fx=False — no conversion")
        return {"y": y, "x": x, "fx_applied": False, "conversion_log": {}}

    if rate_table_source is None:
        print(f"  [FX] apply_fx=True but no rate table — proceeding without conversion")
        return {"y": y, "x": x, "fx_applied": False, "conversion_log": {}}

    rate_table = load_rate_table(rate_table_source)
    x_converted, log = apply_fx_conversion(x, rate_table, label_x=label_x)

    return {
        "y":             y,
        "x":             x_converted,
        "fx_applied":    True,
        "conversion_log": log,
    }
