import os
import pandas as pd

TERM_DIR    = "Raw Data/Term Structure"
MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
OUTPUT_PATH = "Raw Data/Daily Term/fcpo_daily_term_structure.xlsx"


def load_contracts():
    contracts = {}
    if not os.path.exists(TERM_DIR):
        return contracts
    for year_dir in sorted(os.listdir(TERM_DIR)):
        if not year_dir.isdigit():
            continue
        year = int(year_dir)
        for m, abbr in enumerate(MONTH_ABBRS, 1):
            yy = str(year)[2:]
            path = f"{TERM_DIR}/{year_dir}/FCPO {abbr}{yy}_Daily.csv"
            if not os.path.exists(path):
                continue
            df_c = pd.read_csv(path)
            df_c["date"] = pd.to_datetime(
                df_c["Timestamp (UTC)"], infer_datetime_format=True
            ).dt.normalize()
            series = df_c.set_index("date")["Close"]
            contracts[(year, m)] = series[~series.index.duplicated(keep="last")]
    return contracts


def front_month(date):
    if date.day <= 15:
        return (date.year, date.month)
    else:
        m = date.month + 1
        y = date.year + (1 if m > 12 else 0)
        return (y, m % 12 or 12)


def add_months(ym, n):
    y, m = ym
    m += n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return (y, m)


def build_daily_table(contracts):
    all_dates = sorted(set(d for s in contracts.values() for d in s.index))
    today = pd.Timestamp.today().normalize()
    all_dates = [d for d in all_dates if pd.Timestamp("2020-01-01") <= d <= today]

    col_names = ["Current"] + [f"+{i}M" for i in range(1, 12)]
    records = []
    for date in all_dates:
        fm = front_month(date)
        row = {"Date": date}
        for i, col in enumerate(col_names):
            ym = add_months(fm, i)
            row[col] = contracts[ym][date] if ym in contracts and date in contracts[ym].index else None
        records.append(row)

    df = pd.DataFrame(records)

    # Fill missing Current with +1M on roll days (contract expiry gap ~13th–15th)
    roll_mask = df["Current"].isnull() & df["+1M"].notnull()
    df.loc[roll_mask, "Current"] = df.loc[roll_mask, "+1M"]

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d %b %Y")
    return df, roll_mask.sum()


if __name__ == "__main__":
    contracts = load_contracts()
    df, filled = build_daily_table(contracts)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")
    print(f"Saved {len(df)} rows -> {OUTPUT_PATH}")
    print(f"  Current filled from +1M (roll days): {filled} rows")
