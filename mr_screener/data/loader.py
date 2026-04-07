import io
import numpy as np
import pandas as pd


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


def bucket_to_date(series: pd.Series, freq: str) -> pd.Series:
    """Collapse a time series to one value per calendar date (last bar of day)."""
    if freq == "1D":
        out = series.copy()
    else:
        out = series.resample("1D").last().dropna()
    out.index = out.index.normalize().tz_localize(None)
    return out.dropna()


def align_any(
    series_y: pd.Series,
    series_x: pd.Series,
    label_y: str,
    label_x: str,
    min_overlap: int = 30,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Bucket both series to daily, inner-join on date, return aligned pair + info.
    Raises ValueError if fewer than min_overlap common dates.
    """
    freq_y = detect_freq(series_y)
    freq_x = detect_freq(series_x)

    n_y_before = len(series_y)
    n_x_before = len(series_x)

    daily_y = bucket_to_date(series_y, freq_y)
    daily_x = bucket_to_date(series_x, freq_x)

    combined = pd.concat([daily_y, daily_x], axis=1, join="inner").dropna()
    combined = combined[(combined.iloc[:, 0] > 0) & (combined.iloc[:, 1] > 0)]

    n_after = len(combined)
    if n_after < min_overlap:
        y_range = f"{daily_y.index.min().date()} – {daily_y.index.max().date()}" if len(daily_y) else "no data"
        x_range = f"{daily_x.index.min().date()} – {daily_x.index.max().date()}" if len(daily_x) else "no data"
        raise ValueError(
            f"Only {n_after} overlapping dates (need {min_overlap}). "
            f"{label_y}: {y_range}  |  {label_x}: {x_range}"
        )

    aligned_y = combined.iloc[:, 0]
    aligned_x = combined.iloc[:, 1]

    info = {
        "n_y_before":  n_y_before,
        "n_x_before":  n_x_before,
        "freq_y":      freq_y,
        "freq_x":      freq_x,
        "n_after":     n_after,
        "date_start":  str(combined.index.min().date()),
        "date_end":    str(combined.index.max().date()),
        "resampled":   True,
    }
    return aligned_y, aligned_x, info


def load_pair(
    bytes_y: bytes,
    bytes_x: bytes,
    label_y: str,
    label_x: str,
    mult_y: float = 1.0,
    mult_x: float = 1.0,
) -> dict:
    """Load two CSV files, bucket to daily bars, align, return data dict."""

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

    aligned_y, aligned_x, info = align_any(raw_y, raw_x, label_y, label_x)

    log_y = np.log(aligned_y)
    log_x = np.log(aligned_x)

    return {
        "raw_y":     aligned_y,
        "raw_x":     aligned_x,
        "log_y":     log_y,
        "log_x":     log_x,
        "dates":     aligned_y.index,
        "label_y":   label_y,
        "label_x":   label_x,
        "n_obs":     info["n_after"],
        "freq":      "1D",
        "alignment": info,
    }
