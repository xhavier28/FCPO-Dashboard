import io
import numpy as np
import pandas as pd
from mr_screener.config import ALIGNMENT, DATA


def detect_freq(series: pd.Series) -> str:
    """Infer bar frequency from a time-indexed series."""
    if len(series) < 2:
        return "unknown"
    median_gap = series.index.to_series().diff().dt.total_seconds().median() / 3600
    if median_gap >= 20:
        return "1D"
    if median_gap >= 3:
        return "4H"
    if median_gap >= 0.75:
        return "1H"
    return "tick/1m"


def align_any(
    series_y: pd.Series,
    series_x: pd.Series,
    label_y: str,
    label_x: str,
    min_overlap: int = None,
) -> tuple:
    """
    Intra-day positional alignment.

    For each calendar date that appears in BOTH series:
      - Take min(n_bars_y_that_day, n_bars_x_that_day) bars
      - Match positionally within the day (bar 1 with bar 1, etc.)
      - Assumption: both series represent the same intraday session
        in the order they appear in the CSV

    Result: hourly bars, more observations than daily bucketing.
    """
    if min_overlap is None:
        min_overlap = ALIGNMENT["min_overlap"]

    def get_date(series):
        if hasattr(series.index, "date"):
            dates = pd.Series(series.index.date, index=series.index)
        else:
            dates = pd.Series(pd.to_datetime(series.index).date, index=series.index)
        return dates

    dates_y = get_date(series_y)
    dates_x = get_date(series_x)

    unique_y = set(dates_y.values)
    unique_x = set(dates_x.values)
    common_dates = sorted(unique_y.intersection(unique_x))

    if len(common_dates) == 0:
        raise ValueError(
            f"No common dates between {label_y} and {label_x}. "
            "Check that your CSVs cover overlapping periods."
        )

    y_chunks = []
    x_chunks = []
    y_times  = []
    bars_per_day = []

    for date in common_dates:
        y_day = series_y[dates_y.values == date]
        x_day = series_x[dates_x.values == date]
        n_bars = min(len(y_day), len(x_day))
        if n_bars == 0:
            continue
        y_chunks.append(y_day.iloc[:n_bars].values)
        x_chunks.append(x_day.iloc[:n_bars].values)
        y_times.extend(y_day.index[:n_bars])
        bars_per_day.append(n_bars)

    # Build Series with datetime index preserved from y
    y_vals = np.concatenate(y_chunks)
    x_vals = np.concatenate(x_chunks)
    ts_idx  = pd.DatetimeIndex(y_times)

    y_out = pd.Series(y_vals, index=ts_idx)
    x_out = pd.Series(x_vals, index=ts_idx)

    # Drop NaN and zero
    mask  = y_out.notna() & x_out.notna() & (y_out > 0) & (x_out > 0)
    y_out = y_out[mask]
    x_out = x_out[mask]

    n = len(y_out)

    avg_bars = float(np.mean(bars_per_day)) if bars_per_day else 1.0
    min_bars = int(np.min(bars_per_day))    if bars_per_day else 0
    max_bars = int(np.max(bars_per_day))    if bars_per_day else 0

    print(f"  {label_y}: {len(series_y)} input bars, {len(unique_y)} unique dates")
    print(f"  {label_x}: {len(series_x)} input bars, {len(unique_x)} unique dates")
    print(f"  Common dates : {len(common_dates)} days")
    print(f"  Bars per day : avg={avg_bars:.1f}, min={min_bars}, max={max_bars}")
    print(f"  Total matched: {n} hourly bars (~{n/avg_bars:.0f} effective trading days)")

    if n < min_overlap:
        raise ValueError(
            f"Only {n} matched bars. Need {min_overlap}. "
            f"Upload more data — need at least "
            f"{int(min_overlap / avg_bars) + 1} common trading days."
        )
    if n < ALIGNMENT["warn_below"]:
        print(f"  [WARN] {n} bars — usable but {ALIGNMENT['warn_below']}+ gives "
              "more stable OU estimates.")

    return y_out, x_out, {
        "mode":          "intraday_positional",
        "n_input_y":     len(series_y),
        "n_input_x":     len(series_x),
        "n_common_days": len(common_dates),
        "n_after":       n,
        "avg_bars_day":  round(avg_bars, 1),
        "min_bars_day":  min_bars,
        "max_bars_day":  max_bars,
        "date_start":    str(y_out.index[0].date()) if len(y_out) else "?",
        "date_end":      str(y_out.index[-1].date()) if len(y_out) else "?",
    }


def load_pair(
    bytes_y: bytes,
    bytes_x: bytes,
    label_y: str,
    label_x: str,
    mult_y: float = 1.0,
    mult_x: float = 1.0,
) -> dict:
    """Load two CSV files, align intraday positionally, return data dict."""

    def _read(raw_bytes: bytes) -> pd.Series:
        df = pd.read_csv(io.BytesIO(raw_bytes))
        sample = str(df["time"].iloc[0])
        if sample.replace(".", "").isdigit():
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        else:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        close_col = next(c for c in df.columns if c.lower() == "close")
        return df[close_col].astype(float)

    raw_y = _read(bytes_y) * mult_y
    raw_x = _read(bytes_x) * mult_x

    # Detect bar frequency from raw series (before alignment)
    _freq_raw = detect_freq(raw_y)
    freq = "hourly" if _freq_raw in ("1H", "4H", "tick/1m") else "daily"

    aligned_y, aligned_x, info = align_any(raw_y, raw_x, label_y, label_x)

    # Empirical bars per day from alignment; fall back to config
    bars_per_day = info.get("avg_bars_day") or DATA["bars_per_day"]

    log_y = np.log(aligned_y)
    log_x = np.log(aligned_x)

    return {
        "raw_y":       aligned_y,
        "raw_x":       aligned_x,
        "log_y":       log_y,
        "log_x":       log_x,
        "dates":       aligned_y.index,
        "label_y":     label_y,
        "label_x":     label_x,
        "n_obs":       info["n_after"],
        "freq":        freq,
        "bars_per_day": bars_per_day,
        "alignment":   info,
    }
