import os
import numpy as np
import pandas as pd

INPUT_PATH  = "Raw Data/Daily Term/fcpo_daily_term_structure.xlsx"
OUTPUT_PATH = "Raw Data/Daily Term/fcpo_daily_term_delta.xlsx"

SPREAD_PAIRS = [
    ("Spread3M4M",   "+3M",  "+4M"),
    ("Spread4M5M",   "+4M",  "+5M"),
    ("Spread5M6M",   "+5M",  "+6M"),
    ("Spread6M7M",   "+6M",  "+7M"),
    ("Spread7M8M",   "+7M",  "+8M"),
    ("Spread8M9M",   "+8M",  "+9M"),
    ("Spread9M10M",  "+9M",  "+10M"),
    ("Spread10M11M", "+10M", "+11M"),
]


def build_delta_table(df):
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Compute spread values (near minus far)
    for name, near, far in SPREAD_PAIRS:
        df[name] = df[near] - df[far]

    # Detect roll days (first trading day after day-of-month crosses 15)
    df["_day"]      = df["date"].dt.day
    df["_prev_day"] = df["_day"].shift(1)
    df["_is_roll"]  = (df["_day"] > 15) & (df["_prev_day"] <= 15)

    spd_cols   = [p[0] for p in SPREAD_PAIRS]          # Spread3M4M … Spread10M11M
    delta_cols = [f"%DeltaS{p[0][6:]}" for p in SPREAD_PAIRS]  # strip "Spread"
    spd_shift  = spd_cols[1:] + [None]                 # next spread for roll-day correction

    for spd, next_spd, delta in zip(spd_cols, spd_shift, delta_cols):
        normal_dod = df[spd] - df[spd].shift(1)
        if next_spd is not None:
            roll_dod = df[spd] - df[next_spd].shift(1)
        else:
            roll_dod = pd.Series(np.nan, index=df.index)
        df[delta] = np.where(df["_is_roll"], roll_dod, normal_dod)

    df.drop(columns=["_day", "_prev_day", "_is_roll"], inplace=True)
    return df


if __name__ == "__main__":
    raw = pd.read_excel(INPUT_PATH)
    raw["date"] = pd.to_datetime(raw["Date"], format="%d %b %Y")
    df = build_delta_table(raw)
    df = df.drop(columns=["date"])   # keep original "Date" string column
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")
    spd_cols = [p[0] for p in SPREAD_PAIRS]
    print(f"Saved {len(df)} rows -> {OUTPUT_PATH}")
    print(f"  Spread columns: {spd_cols[0]} ... {spd_cols[-1]}")
    print(f"  Non-null spreads in Spread3M4M: {df['Spread3M4M'].notna().sum()}")
