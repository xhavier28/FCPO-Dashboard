"""Verify that adding 2020-2023 files doesn't change 2024+ butterfly data."""
import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
import re
from pathlib import Path
from MRBackTest.shared.tenor_mapping import tenor_to_contract_month

RAW_DIR = Path(r'C:/ClaudeCode/Raw Data/Minutes Data')
OUTPUT_DIR = Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild')
MONTH_MAP = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
TENOR_LABELS = ['Current'] + [f'+{i}M' for i in range(1, 10)]


def parse_bf(fname):
    m = re.match(r'FCPO (\w{3})(\d{2}) 1mo Butterfly_60min\.csv', fname)
    if not m:
        return None
    return (2000 + int(m.group(2)), MONTH_MAP[m.group(1)])


def load_minute_close(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['Timestamp (UTC)'])
    return dict(zip(df['datetime'], df['Close']))


# Build full inventory (all years)
all_inv = {}
for year_dir in sorted((RAW_DIR / 'Butterfly').iterdir()):
    if not year_dir.is_dir():
        continue
    for f in sorted(year_dir.iterdir()):
        if f.suffix != '.csv':
            continue
        ym = parse_bf(f.name)
        if ym:
            all_inv[ym] = f

print(f'Full butterfly inventory: {len(all_inv)} contracts')

# Collect all datetimes from ALL files
all_datetimes = set()
contract_lookup = {}
for ym, fpath in sorted(all_inv.items()):
    data = load_minute_close(fpath)
    contract_lookup[ym] = data
    all_datetimes.update(data.keys())

all_datetimes = sorted(all_datetimes)
dt_2024 = [dt for dt in all_datetimes if dt >= pd.Timestamp('2024-01-01')]
print(f'Total 2024+ datetimes in new build: {len(dt_2024)}')

# Build new 2024+ close values
new_rows = []
for dt in dt_2024:
    d = dt.date()
    row = {'datetime': dt}
    for offset in range(10):
        tenor = 'Current' if offset == 0 else f'+{offset}M'
        target_ym = tenor_to_contract_month(d, offset)
        if target_ym in contract_lookup and dt in contract_lookup[target_ym]:
            row[tenor] = contract_lookup[target_ym][dt]
        else:
            row[tenor] = np.nan
    new_rows.append(row)

new_24 = pd.DataFrame(new_rows).set_index('datetime')

# Load old CSV
old_bfly = pd.read_csv(
    OUTPUT_DIR / 'butterfly_tenor_close_wide.csv',
    parse_dates=['datetime']
).set_index('datetime')
old_24 = old_bfly[old_bfly.index >= '2024-01-01']

# Compare
shared_idx = old_24.index.intersection(new_24.index)
print(f'Shared indices: {len(shared_idx)}')
print(f'Old-only indices: {len(old_24.index.difference(new_24.index))}')
print(f'New-only indices: {len(new_24.index.difference(old_24.index))}')

all_match = True
for col in TENOR_LABELS:
    old_s = old_24.loc[shared_idx, col]
    new_s = new_24.loc[shared_idx, col]
    both_nan = old_s.isna() & new_s.isna()
    both_equal = old_s == new_s
    ok = both_nan | both_equal
    n_diff = (~ok).sum()
    if n_diff > 0:
        all_match = False
        first_diff_idx = (~ok).idxmax()
        print(f'  REAL MISMATCH: {col}: {n_diff} bars differ')
        print(f'    First: {first_diff_idx} old={old_s[first_diff_idx]} new={new_s[first_diff_idx]}')

if all_match:
    print('ALL 2024+ BUTTERFLY DATA MATCHES EXACTLY (NaN-aware comparison).')
else:
    print('*** MISMATCHES FOUND — investigate before saving ***')
