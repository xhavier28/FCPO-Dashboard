import io
import numpy as np
import pandas as pd


def load_pair(bytes_y, bytes_x, label_y: str, label_x: str) -> dict:
    """Load two CSV files, resample to 1-hour bars, align, return data dict."""

    def _read(raw_bytes: bytes) -> pd.Series:
        df = pd.read_csv(io.BytesIO(raw_bytes))
        # Parse time column — try Unix timestamp first, then datetime string
        sample = str(df["time"].iloc[0])
        if sample.replace(".", "").isdigit():
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        else:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        # Prefer 'close' col; fall back to case-insensitive match
        close_col = next(c for c in df.columns if c.lower() == "close")
        series = df[close_col].astype(float)
        # Resample to 1-hour bars
        series = series.resample("1h").last().dropna()
        return series

    raw_y = _read(bytes_y)
    raw_x = _read(bytes_x)

    # Inner join on common timestamps
    combined = pd.concat([raw_y, raw_x], axis=1, join="inner").dropna()
    combined.columns = ["y", "x"]

    aligned_y = combined["y"]
    aligned_x = combined["x"]

    log_y = np.log(aligned_y)
    log_x = np.log(aligned_x)

    return {
        "raw_y":   aligned_y,
        "raw_x":   aligned_x,
        "log_y":   log_y,
        "log_x":   log_x,
        "dates":   combined.index,
        "label_y": label_y,
        "label_x": label_x,
        "n_obs":   len(combined),
        "freq":    "1H",
    }
