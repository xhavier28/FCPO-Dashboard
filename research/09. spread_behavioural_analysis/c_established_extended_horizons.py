"""
C-Established Extended Horizon Check
=====================================
Extends the established-duration C anti-reversion check to 25/30/40d
horizons for 5 candidate instruments, to determine whether low
reversion rate at 5-20d is genuine anti-reversion or slow reversion.

Same signal/episode/z-score methodology as spread_behaviour_sweep.py.

Output: c_established_extended.csv (new file, same schema as sweep_results.csv)
Log: appends to spread_behaviour_log.txt
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:/ClaudeCode')

import os
import pandas as pd
import numpy as np
from datetime import datetime
from models.feature_prep import load_daily_shape_log, load_enriched_shape_log

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

Z_SIGNAL = 1.5
Z_REVERSION = 0.5
EXTENDED_HORIZONS = [25, 30, 40]
LOW_N_THRESH = 20
WINDOW_CAP = 60
MIN_OBS = 10

CANDIDATE_INSTRUMENTS = ['M3-M5', 'M4-M6', 'M6-M7', 'M5-M7', 'M6-M8']
SHAPE_VAL = '1'  # C shape
DUR_LO = 21
DUR_HI = 9999

TENOR_COLS = {
    1: "Current", 2: "+1M", 3: "+2M", 4: "+3M", 5: "+4M", 6: "+5M",
    7: "+6M", 8: "+7M", 9: "+8M", 10: "+9M", 11: "+10M", 12: "+11M",
}

ALL_INSTRUMENTS = {
    'M3-M5': (3, 5), 'M4-M6': (4, 6),
    'M6-M7': (6, 7), 'M5-M7': (5, 7), 'M6-M8': (6, 8),
}

INSTRUMENT_GROUPS = {
    'M3-M5': '2-month-gap-6', 'M4-M6': '2-month-gap-6',
    'M6-M7': 'back-month-5', 'M5-M7': '2-month-gap-6',
    'M6-M8': '2-month-gap-6',
}

OUTPUT_DIR = r'C:/ClaudeCode/research/09. spread_behavioural_analysis'
LOG_FILE = os.path.join(OUTPUT_DIR, 'spread_behaviour_log.txt')
EXT_CSV = os.path.join(OUTPUT_DIR, 'c_established_extended.csv')
TERM_PATH = r'C:/ClaudeCode/Raw Data/Daily Term/fcpo_daily_term_structure.xlsx'

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)


# ══════════════════════════════════════════════════════════════
# DATA LOADING (same as main sweep)
# ══════════════════════════════════════════════════════════════

log('=' * 70)
log(f'C-ESTABLISHED EXTENDED HORIZON CHECK — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
log('=' * 70)
log('')

full_log = load_daily_shape_log().sort_values('date').reset_index(drop=True)
enriched = load_enriched_shape_log().sort_values('date').reset_index(drop=True)

pre_2017 = full_log[full_log['date'] < '2017-01-01'].copy().sort_values('date').reset_index(drop=True)
days_list, episode_list = [], []
ep_id, prev_shape, day_count = 0, None, 0
for i, row in pre_2017.iterrows():
    if row['shape'] != prev_shape:
        prev_shape = row['shape']
        day_count = 1
        ep_id += 1
    else:
        day_count += 1
    days_list.append(day_count)
    episode_list.append(ep_id)
pre_2017['days_in_shape'] = days_list
pre_2017['episode_id'] = episode_list

if 'episode_id' not in enriched.columns:
    enriched['episode_id'] = (enriched['shape'] != enriched['shape'].shift(1)).cumsum() + ep_id

shared_cols = ['date', 'shape', 'days_in_shape', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'episode_id']
df = pd.concat([pre_2017[shared_cols], enriched[shared_cols]],
               ignore_index=True).sort_values('date').reset_index(drop=True)
df = df.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

# Load term structure for M7-M9
term_df = pd.read_excel(TERM_PATH)
term_df['Date'] = pd.to_datetime(term_df['Date'], dayfirst=True)
term_df = term_df.sort_values('Date').reset_index(drop=True)

for tenor_num, col_name in TENOR_COLS.items():
    if tenor_num <= 6:
        continue
    merge_col = f'M{tenor_num}'
    vals = pd.to_numeric(term_df[col_name], errors='coerce')
    merge_df = pd.DataFrame({'date': term_df['Date'], merge_col: vals.values})
    df = df.merge(merge_df, on='date', how='left')

log(f'Panel: {len(df)} rows, {df["date"].min().date()} to {df["date"].max().date()}')

# Compute spreads
for inst, legs in ALL_INSTRUMENTS.items():
    near, far = legs
    df[inst] = df[f'M{near}'] - df[f'M{far}']

# Z-scores
def compute_regime_zscore(df, instrument_col):
    zscores = pd.Series(np.nan, index=df.index)
    means = pd.Series(np.nan, index=df.index)
    for ep_id_val in df['episode_id'].unique():
        mask = df['episode_id'] == ep_id_val
        ep_vals = df.loc[mask, instrument_col]
        if len(ep_vals) < MIN_OBS:
            continue
        for i, (idx, val) in enumerate(ep_vals.items()):
            if i < (MIN_OBS - 1):
                continue
            window_start = max(0, i - (WINDOW_CAP - 1))
            window = ep_vals.iloc[window_start:i + 1]
            mean, std = window.mean(), window.std(ddof=1)
            if std > 0 and not np.isnan(val):
                zscores[idx] = (val - mean) / std
                means[idx] = mean
    return zscores, means

log('Computing z-scores...')
for inst in CANDIDATE_INSTRUMENTS:
    z, m = compute_regime_zscore(df, inst)
    df[f'{inst}_z'] = z
    df[f'{inst}_mean'] = m
    log(f'  {inst}: {z.notna().sum()} valid z-score days')


# ══════════════════════════════════════════════════════════════
# EXTENDED HORIZON SWEEP
# ══════════════════════════════════════════════════════════════

log('')
log('Running extended horizons [25, 30, 40] for C-established signals...')
log('')

sweep_rows = []

for inst in CANDIDATE_INSTRUMENTS:
    z_col = f'{inst}_z'
    mean_col = f'{inst}_mean'
    group = INSTRUMENT_GROUPS[inst]

    shape_mask = df['shape'].astype(str) == SHAPE_VAL
    dur_mask = (df['days_in_shape'] >= DUR_LO) & (df['days_in_shape'] <= DUR_HI)
    z_mask = df[z_col].abs() > Z_SIGNAL
    signal_mask = shape_mask & dur_mask & z_mask
    signal_indices = df.index[signal_mask]

    for h in EXTENDED_HORIZONS:
        n_signals = 0
        n_reverted = 0
        n_continued = 0
        continuation_depths = []
        time_to_turn_vals = []
        half_life_vals = []

        for idx in signal_indices:
            if idx + h >= len(df):
                continue

            signal_z = df.loc[idx, z_col]
            signal_spread = df.loc[idx, inst]
            signal_mean = df.loc[idx, mean_col]

            if pd.isna(signal_z) or pd.isna(signal_spread) or pd.isna(signal_mean):
                continue

            abs_signal_z = abs(signal_z)
            fwd_slice = df.iloc[idx+1:idx+h+1]
            if len(fwd_slice) == 0:
                continue

            n_signals += 1
            fwd_z_vals = fwd_slice[z_col].values
            fwd_z_abs = np.abs(fwd_z_vals)
            valid_fwd_z = ~np.isnan(fwd_z_vals)

            # Reversion: z crosses below 0.5
            reverted = False
            if valid_fwd_z.any():
                reverted = np.any(fwd_z_abs[valid_fwd_z] < Z_REVERSION)
            if reverted:
                n_reverted += 1

            # Continuation: |z| exceeds signal-day |z|
            continued = False
            max_z_reached = abs_signal_z
            peak_day = 0
            if valid_fwd_z.any():
                for d_offset in range(len(fwd_z_abs)):
                    if not np.isnan(fwd_z_vals[d_offset]):
                        fwd_abs = abs(fwd_z_vals[d_offset])
                        if fwd_abs > abs_signal_z:
                            continued = True
                        if fwd_abs > max_z_reached:
                            max_z_reached = fwd_abs
                            peak_day = d_offset + 1

            if continued:
                n_continued += 1
                continuation_depths.append(max_z_reached)
                time_to_turn_vals.append(peak_day)

            # Half-life
            if reverted:
                half_z = abs_signal_z * 0.5
                hl = np.nan
                for d_offset in range(len(fwd_z_abs)):
                    if not np.isnan(fwd_z_vals[d_offset]):
                        if abs(fwd_z_vals[d_offset]) <= half_z:
                            hl = d_offset + 1
                            break
                if not np.isnan(hl):
                    half_life_vals.append(hl)

        reversion_rate = (n_reverted / n_signals * 100) if n_signals > 0 else np.nan
        continuation_rate = (n_continued / n_signals * 100) if n_signals > 0 else np.nan
        anti_rev_depth = np.mean(continuation_depths) if continuation_depths else np.nan
        time_to_turn = np.median(time_to_turn_vals) if time_to_turn_vals else np.nan
        half_life = np.median(half_life_vals) if half_life_vals else np.nan
        net_bias = (reversion_rate - 50.0) if n_signals > 0 else np.nan

        low_n_flag = 'LOW-N' if (0 < n_signals < LOW_N_THRESH) else ''

        sweep_rows.append({
            'instrument': inst,
            'group': group,
            'shape': 'C',
            'duration_bucket': 'established',
            'horizon': h,
            'n_signals': n_signals,
            'reversion_rate_pct': round(reversion_rate, 2) if not np.isnan(reversion_rate) else np.nan,
            'continuation_rate_pct': round(continuation_rate, 2) if not np.isnan(continuation_rate) else np.nan,
            'anti_rev_depth_z': round(anti_rev_depth, 3) if not np.isnan(anti_rev_depth) else np.nan,
            'time_to_turn_days': round(time_to_turn, 1) if not np.isnan(time_to_turn) else np.nan,
            'half_life_days': round(half_life, 1) if not np.isnan(half_life) else np.nan,
            'net_directional_bias_pct': round(net_bias, 2) if not np.isnan(net_bias) else np.nan,
            'low_n_flag': low_n_flag,
        })

ext_df = pd.DataFrame(sweep_rows)
ext_df.to_csv(EXT_CSV, index=False)
log(f'Output: {EXT_CSV} ({len(ext_df)} rows)')

# ══════════════════════════════════════════════════════════════
# FULL TABLE OUTPUT
# ══════════════════════════════════════════════════════════════

log('')
log('FULL TABLE — C-established extended horizons (15 rows):')
log('-' * 130)

cols = ['instrument', 'horizon', 'n_signals', 'reversion_rate_pct',
        'continuation_rate_pct', 'anti_rev_depth_z', 'time_to_turn_days',
        'half_life_days', 'net_directional_bias_pct', 'low_n_flag']

header = f'{"instrument":<12} {"hor":>4} {"N":>5} {"rev%":>7} {"cont%":>7} {"depth_z":>8} {"t2turn":>7} {"hl_d":>6} {"net_bias":>9} {"flag":>6}'
log(header)
log('-' * 130)

for _, row in ext_df.iterrows():
    def fmt(v, w=7, dec=1):
        if isinstance(v, float) and np.isnan(v):
            return f'{"N/A":>{w}}'
        if isinstance(v, float):
            return f'{v:>{w}.{dec}f}'
        return f'{str(v):>{w}}'

    line = (f'{row["instrument"]:<12} {row["horizon"]:>4} {row["n_signals"]:>5} '
            f'{fmt(row["reversion_rate_pct"])} {fmt(row["continuation_rate_pct"])} '
            f'{fmt(row["anti_rev_depth_z"], 8, 2)} {fmt(row["time_to_turn_days"])} '
            f'{fmt(row["half_life_days"], 6)} {fmt(row["net_directional_bias_pct"], 9)} '
            f'{row["low_n_flag"]:>6}')
    log(line)

log('')
log('=' * 70)
log('RUN COMPLETE')
log('=' * 70)

# Append to log
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write('\n\n')
    f.write('\n'.join(log_lines))
    f.write('\n')

print(f'\nLog appended to: {LOG_FILE}')
print(f'Extended CSV: {EXT_CSV}')
print('Done.')
