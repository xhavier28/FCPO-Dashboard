"""
One-time conversion script: converts 2008-2017 term structure files
from old TradingView format to the standard engine format.

Old format:  MYX_DLY_FCPO{letter}{YYYY}, 1D_{hash}.csv
             Columns: time,open,high,low,close (unix timestamps)

New format:  FCPO {Mmm}{yy}_Daily.csv
             Columns: Timestamp (UTC),Open,High,Low,Close,FCPO {Mmm}{yy}: Volume
"""

import os
import re
import pandas as pd

TERM_DIR = os.path.join(os.path.dirname(__file__), "..", "Raw Data", "Term Structure")

MONTH_CODE_MAP = {
    "F": (1, "Jan"), "G": (2, "Feb"), "H": (3, "Mar"), "J": (4, "Apr"),
    "K": (5, "May"), "M": (6, "Jun"), "N": (7, "Jul"), "Q": (8, "Aug"),
    "U": (9, "Sep"), "V": (10, "Oct"), "X": (11, "Nov"), "Z": (12, "Dec"),
}

PATTERN = re.compile(r"MYX_DLY_FCPO([A-Z])(\d{4}),\s*1D_[a-f0-9]+\.csv")


def convert_file(filepath, year_dir):
    fname = os.path.basename(filepath)
    m = PATTERN.match(fname)
    if not m:
        return None

    letter, year_str = m.group(1), m.group(2)
    if letter not in MONTH_CODE_MAP:
        print(f"  Skipping unknown month code '{letter}': {fname}")
        return None

    month_num, month_abbr = MONTH_CODE_MAP[letter]
    yy = year_str[2:]
    new_name = f"FCPO {month_abbr}{yy}_Daily.csv"
    new_path = os.path.join(year_dir, new_name)

    df = pd.read_csv(filepath)
    # Convert unix timestamp to date string (Malaysia timezone)
    df["Timestamp (UTC)"] = (
        pd.to_datetime(df["time"], unit="s", utc=True)
        .dt.tz_convert("Asia/Kuala_Lumpur")
        .dt.strftime("%Y-%m-%d")
    )
    vol_col = f"FCPO {month_abbr}{yy}: Volume"
    out = pd.DataFrame({
        "Timestamp (UTC)": df["Timestamp (UTC)"],
        "Open": df["open"],
        "High": df["high"],
        "Low": df["low"],
        "Close": df["close"],
        vol_col: 0,
    })
    out.to_csv(new_path, index=False)
    os.remove(filepath)
    return new_name


def main():
    converted = 0
    for year in range(2008, 2018):
        year_dir = os.path.join(TERM_DIR, str(year))
        if not os.path.isdir(year_dir):
            print(f"Directory not found: {year_dir}")
            continue

        files = [f for f in os.listdir(year_dir) if PATTERN.match(f)]
        if not files:
            print(f"{year}: no old-format files found")
            continue

        print(f"{year}: converting {len(files)} files...")
        for fname in sorted(files):
            result = convert_file(os.path.join(year_dir, fname), year_dir)
            if result:
                converted += 1
                print(f"  {fname} -> {result}")

    print(f"\nDone. Converted {converted} files.")


if __name__ == "__main__":
    main()
