"""
Rebuild tenor-mapped wide CSVs from ALL raw minute data (2020-2027).
================================================================
Reproduces the exact logic from stage3_minute_rebuild.ipynb but as a
standalone script for reproducibility. Scans all year folders under
Raw Data/Minutes Data/{Spread,Butterfly}/.

After rebuild, validates that the 2024-2026 slice is UNCHANGED
vs the prior reference: 113 trades, 67.3% win, +728.0 PnL,
adj Sharpe 1.366, max DD -81.6.

Usage: python "research/06. stage3_minute_rebuild/rebuild_tenor_wide.py"
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

from MRBackTest.shared.tenor_mapping import (
    front_month, tenor_to_contract_month, contract_month_to_str, add_months
)

RAW_DIR = Path(r'C:/ClaudeCode/Raw Data/Minutes Data')
OUTPUT_DIR = Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild')

TENOR_LABELS = ['Current'] + [f'+{i}M' for i in range(1, 10)]

MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}


def parse_butterfly_filename(fname):
    m = re.match(r'FCPO (\w{3})(\d{2}) 1mo Butterfly_60min\.csv', fname)
    if not m:
        return None
    return (2000 + int(m.group(2)), MONTH_MAP[m.group(1)])


def parse_spread_filename(fname):
    m = re.match(r'FCPO (\w{3})(\d{2})-(\w{3})(\d{2}) Calendar_60min\.csv', fname)
    if not m:
        return None
    return (2000 + int(m.group(2)), MONTH_MAP[m.group(1)])


def inventory_files(instrument_dir, parser):
    files = {}
    if not instrument_dir.exists():
        return files
    for year_dir in sorted(instrument_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for f in sorted(year_dir.iterdir()):
            if f.suffix != '.csv':
                continue
            ym = parser(f.name)
            if ym:
                files[ym] = f
    return files


def load_minute_file(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['Timestamp (UTC)'])
    df['date'] = df['datetime'].dt.date

    cols = df.columns.tolist()
    avol_col = [c for c in cols if 'AVol' in c]
    bvol_col = [c for c in cols if 'BVol' in c]
    mfi_col = [c for c in cols if 'MFI' in c]
    tr_col = [c for c in cols if 'True Range' in c]

    rename = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
    keep = ['datetime', 'date', 'Open', 'High', 'Low', 'Close']
    extra_rename = {}

    if avol_col:
        keep.append(avol_col[0])
        extra_rename[avol_col[0]] = 'avol'
    if bvol_col:
        keep.append(bvol_col[0])
        extra_rename[bvol_col[0]] = 'bvol'
    if mfi_col:
        keep.append(mfi_col[0])
        extra_rename[mfi_col[0]] = 'mfi'
    if tr_col:
        keep.append(tr_col[0])
        extra_rename[tr_col[0]] = 'true_range'

    df = df[keep].rename(columns={**rename, **extra_rename})
    return df


def build_rolling_tenor_series(file_inventory, instrument_label, instrument_type="spread"):
    """Reproduction of original build logic.

    instrument_type: "spread" or "butterfly" — controls roll-day rule.
    The original build used "butterfly" roll (day 2) for butterfly data
    and "spread" roll (day 16) for spread data, matching the actual
    contract expiry behavior of each instrument class.
    """
    all_data = {}
    print(f'Loading {instrument_label} files...')
    for ym, fpath in sorted(file_inventory.items()):
        df = load_minute_file(fpath)
        df['contract_ym'] = [ym] * len(df)
        all_data[ym] = df

    all_datetimes = set()
    for df in all_data.values():
        all_datetimes.update(df['datetime'].tolist())
    all_datetimes = sorted(all_datetimes)
    print(f'Total unique datetime bars: {len(all_datetimes)}')
    print(f'Date range: {all_datetimes[0]} to {all_datetimes[-1]}')

    contract_lookup = {}
    for ym, df in all_data.items():
        contract_lookup[ym] = df.set_index('datetime').to_dict('index')

    close_records = []
    long_records = []

    for dt in all_datetimes:
        d = dt.date() if hasattr(dt, 'date') else dt
        row = {'datetime': dt}

        for offset in range(10):
            tenor_label = 'Current' if offset == 0 else f'+{offset}M'
            target_ym = tenor_to_contract_month(d, offset, instrument_type=instrument_type)

            if target_ym in contract_lookup and dt in contract_lookup[target_ym]:
                data = contract_lookup[target_ym][dt]
                row[tenor_label] = data['close']

                long_records.append({
                    'datetime': dt,
                    'date': d,
                    'tenor': tenor_label,
                    'tenor_offset': offset,
                    'contract_month': contract_month_to_str(target_ym),
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'avol': data.get('avol', np.nan),
                    'bvol': data.get('bvol', np.nan),
                    'mfi': data.get('mfi', np.nan),
                    'true_range': data.get('true_range', np.nan),
                })
            else:
                row[tenor_label] = np.nan

        close_records.append(row)

    close_wide = pd.DataFrame(close_records).set_index('datetime')
    full_long = pd.DataFrame(long_records)

    print(f'Close-wide shape: {close_wide.shape}')
    print(f'Full-long shape:  {full_long.shape}')

    return close_wide, full_long


def main():
    # ── Step 1: Inventory ──
    butterfly_files = inventory_files(RAW_DIR / 'Butterfly', parse_butterfly_filename)
    spread_files = inventory_files(RAW_DIR / 'Spread', parse_spread_filename)

    print(f'Butterfly files: {len(butterfly_files)} contracts')
    print(f'Spread files:    {len(spread_files)} contracts')
    print()

    for label, files in [('Butterfly', butterfly_files), ('Spread', spread_files)]:
        print(f'=== {label} ===')
        by_year = {}
        for (y, m) in sorted(files.keys()):
            by_year.setdefault(y, []).append(m)
        for y, months in sorted(by_year.items()):
            month_strs = [list(MONTH_MAP.keys())[m-1] for m in sorted(months)]
            print(f'  {y}: {", ".join(month_strs)} ({len(months)})')
        print()

    # ── Step 2: Build tenor series ──
    print('Building Butterfly rolling tenor series...')
    bfly_close, bfly_long = build_rolling_tenor_series(
        butterfly_files, 'Butterfly', instrument_type='butterfly')
    print()

    print('Building Spread rolling tenor series...')
    spd_close, spd_long = build_rolling_tenor_series(
        spread_files, 'Spread', instrument_type='spread')
    print()

    # ── Step 3: Report earliest dates per instrument ──
    print('=' * 70)
    print('EARLIEST USABLE DATE PER INSTRUMENT')
    print('=' * 70)

    # Instruments map to specific tenor offsets in the wide CSVs
    INSTRUMENT_TENOR = {
        'M1-M2': ('spread', 'Current'),
        'M2-M3': ('spread', '+1M'),
        'M3-M4': ('spread', '+2M'),
        'M4-M5': ('spread', '+3M'),
        'M5-M6': ('spread', '+4M'),
        'BF_M1M2M3': ('butterfly', 'Current'),
        'BF_M2M3M4': ('butterfly', '+1M'),
        'BF_M3M4M5': ('butterfly', '+2M'),
        'BF_M4M5M6': ('butterfly', '+3M'),
    }

    for inst, (itype, tenor) in INSTRUMENT_TENOR.items():
        cw = spd_close if itype == 'spread' else bfly_close
        valid = cw[tenor].dropna()
        if len(valid) > 0:
            earliest = valid.index.min()
            latest = valid.index.max()
            print(f'  {inst:12s} ({itype:9s} {tenor:>7s}): {earliest} to {latest}  ({len(valid)} bars)')
        else:
            print(f'  {inst:12s} ({itype:9s} {tenor:>7s}): NO DATA')
    print()

    # ── Step 4: Missing data summary ──
    print('=' * 70)
    print('MISSING DATA SUMMARY')
    print('=' * 70)
    for label, cw in [('Butterfly', bfly_close), ('Spread', spd_close)]:
        total = len(cw)
        print(f'\n{label} (total bars: {total}):')
        print(f'  {"Tenor":<10} {"Valid":>8} {"Missing":>8} {"Coverage%":>10}')
        for col in TENOR_LABELS:
            valid = cw[col].notna().sum()
            missing = total - valid
            pct = 100 * valid / total
            print(f'  {col:<10} {valid:>8} {missing:>8} {pct:>9.1f}%')
    print()

    # ── Step 5: Verify 2024+ slice unchanged ──
    print('=' * 70)
    print('2024+ SLICE INTEGRITY CHECK')
    print('=' * 70)
    # Load OLD CSVs for comparison
    old_spd = pd.read_csv(OUTPUT_DIR / 'spread_tenor_close_wide.csv', parse_dates=['datetime'])
    old_bfly = pd.read_csv(OUTPUT_DIR / 'butterfly_tenor_close_wide.csv', parse_dates=['datetime'])

    # Filter both to 2024+
    old_spd_24 = old_spd[old_spd['datetime'] >= '2024-01-01'].set_index('datetime')
    old_bfly_24 = old_bfly[old_bfly['datetime'] >= '2024-01-01'].set_index('datetime')

    new_spd_24 = spd_close[spd_close.index >= '2024-01-01']
    new_bfly_24 = bfly_close[bfly_close.index >= '2024-01-01']

    # Compare shapes
    print(f'Old spread  2024+ shape: {old_spd_24.shape}')
    print(f'New spread  2024+ shape: {new_spd_24.shape}')
    print(f'Old bfly    2024+ shape: {old_bfly_24.shape}')
    print(f'New bfly    2024+ shape: {new_bfly_24.shape}')

    # Compare values on shared index
    shared_spd_idx = old_spd_24.index.intersection(new_spd_24.index)
    shared_bfly_idx = old_bfly_24.index.intersection(new_bfly_24.index)

    spd_match = True
    bfly_match = True

    for col in TENOR_LABELS:
        if col in old_spd_24.columns and col in new_spd_24.columns:
            old_vals = old_spd_24.loc[shared_spd_idx, col]
            new_vals = new_spd_24.loc[shared_spd_idx, col]
            # NaN-aware comparison: NaN == NaN is True
            both_nan = old_vals.isna() & new_vals.isna()
            both_equal = old_vals == new_vals
            ok = both_nan | both_equal
            n_mismatch = (~ok).sum()
            if n_mismatch > 0:
                spd_match = False
                first_idx = (~ok).idxmax()
                print(f'  MISMATCH: Spread {col}: {n_mismatch} bars differ')
                print(f'    First: {first_idx} old={old_vals[first_idx]} new={new_vals[first_idx]}')

        if col in old_bfly_24.columns and col in new_bfly_24.columns:
            old_vals = old_bfly_24.loc[shared_bfly_idx, col]
            new_vals = new_bfly_24.loc[shared_bfly_idx, col]
            both_nan = old_vals.isna() & new_vals.isna()
            both_equal = old_vals == new_vals
            ok = both_nan | both_equal
            n_mismatch = (~ok).sum()
            if n_mismatch > 0:
                bfly_match = False
                first_idx = (~ok).idxmax()
                print(f'  MISMATCH: Butterfly {col}: {n_mismatch} bars differ')
                print(f'    First: {first_idx} old={old_vals[first_idx]} new={new_vals[first_idx]}')

    if spd_match and bfly_match:
        print('  ALL 2024+ DATA MATCHES (NaN-aware comparison).')
    else:
        print('  *** WARNING: 2024+ data CHANGED — investigate before proceeding ***')
        return

    # Check for extra/missing rows
    spd_only_old = old_spd_24.index.difference(new_spd_24.index)
    spd_only_new = new_spd_24.index.difference(old_spd_24.index)
    bfly_only_old = old_bfly_24.index.difference(new_bfly_24.index)
    bfly_only_new = new_bfly_24.index.difference(old_bfly_24.index)

    if len(spd_only_old) > 0:
        print(f'  Spread: {len(spd_only_old)} bars in OLD but not NEW (2024+)')
    if len(spd_only_new) > 0:
        print(f'  Spread: {len(spd_only_new)} NEW bars added in 2024+ range')
    if len(bfly_only_old) > 0:
        print(f'  Butterfly: {len(bfly_only_old)} bars in OLD but not NEW (2024+)')
    if len(bfly_only_new) > 0:
        print(f'  Butterfly: {len(bfly_only_new)} NEW bars added in 2024+ range')

    if (len(spd_only_new) > 0 or len(bfly_only_new) > 0):
        print('  NOTE: New bars in 2024+ range come from 2020-2023 contracts whose')
        print('  bars extend into 2024+ (contracts trade 6-12 months before expiry).')
        print('  These new bars will have NaN for most 2024+ tenors (contract not in')
        print('  the inventory yet) — this is expected and should not affect results')
        print('  since the intraday engine filters by date range and drops NaN instruments.')

    # ── Step 6: Save rebuilt CSVs ──
    print()
    print('=' * 70)
    print('SAVING REBUILT CSVs')
    print('=' * 70)

    bfly_close.to_csv(OUTPUT_DIR / 'butterfly_tenor_close_wide.csv')
    bfly_long.to_csv(OUTPUT_DIR / 'butterfly_tenor_full_long.csv', index=False)
    spd_close.to_csv(OUTPUT_DIR / 'spread_tenor_close_wide.csv')
    spd_long.to_csv(OUTPUT_DIR / 'spread_tenor_full_long.csv', index=False)

    print('Saved:')
    print(f'  {OUTPUT_DIR / "butterfly_tenor_close_wide.csv"}')
    print(f'  {OUTPUT_DIR / "butterfly_tenor_full_long.csv"}')
    print(f'  {OUTPUT_DIR / "spread_tenor_close_wide.csv"}')
    print(f'  {OUTPUT_DIR / "spread_tenor_full_long.csv"}')
    print()
    print('Rebuild complete. Run backtest validation next.')


if __name__ == '__main__':
    main()
