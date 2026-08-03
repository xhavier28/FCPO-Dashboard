"""
Sweep Comparison: Old Panel vs New Panel — Established-C Cluster
================================================================
Re-runs the behavioral sweep metrics on the new daily_panel_backtest.csv
for the 5 established-C candidate instruments (M3-M5, M4-M6, M6-M7,
M5-M7, M6-M8) at 5/10/20d horizons, and compares side-by-side against
the existing sweep_results.csv values.

Same z-score / signal / metric methodology as spread_behaviour_sweep.py.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:/ClaudeCode')

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

Z_SIGNAL = 1.5
Z_REVERSION = 0.5
HORIZONS = [5, 10, 20]
LOW_N_THRESH = 20
WINDOW_CAP = 60
MIN_OBS = 10

CANDIDATES = ['M3-M5', 'M4-M6', 'M6-M7', 'M5-M7', 'M6-M8']
SHAPE_VAL = 1.0  # C
DUR_LO = 21
DUR_HI = 9999

OUTPUT_DIR = r'C:/ClaudeCode/research/09. spread_behavioural_analysis'
NEW_PANEL_PATH = os.path.join(OUTPUT_DIR, 'daily_panel_backtest.csv')
OLD_SWEEP_PATH = os.path.join(OUTPUT_DIR, 'sweep_results.csv')
LOG_FILE = os.path.join(OUTPUT_DIR, 'spread_behaviour_log.txt')

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)


# ══════════════════════════════════════════════════════════════
# LOAD NEW PANEL
# ══════════════════════════════════════════════════════════════

log('=' * 70)
log(f'SWEEP COMPARISON: OLD vs NEW PANEL -- {datetime.now().strftime("%Y-%m-%d %H:%M")}')
log('=' * 70)
log('')

df = pd.read_csv(NEW_PANEL_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
log(f'New panel: {len(df)} rows, {df["date"].min().date()} to {df["date"].max().date()}')
log(f'Shape-matched rows: {df["shape"].notna().sum()}')


# ══════════════════════════════════════════════════════════════
# Z-SCORE COMPUTATION (same as main sweep)
# ══════════════════════════════════════════════════════════════

def compute_regime_zscore(df, col):
    zscores = pd.Series(np.nan, index=df.index)
    means = pd.Series(np.nan, index=df.index)
    for ep_id_val in df['episode_id'].dropna().unique():
        mask = df['episode_id'] == ep_id_val
        ep_vals = df.loc[mask, col]
        if len(ep_vals) < MIN_OBS:
            continue
        for i, (idx, val) in enumerate(ep_vals.items()):
            if i < (MIN_OBS - 1):
                continue
            ws = max(0, i - (WINDOW_CAP - 1))
            w = ep_vals.iloc[ws:i + 1]
            m, s = w.mean(), w.std(ddof=1)
            if s > 0 and not np.isnan(val):
                zscores[idx] = (val - m) / s
                means[idx] = m
    return zscores, means

log('')
log('Computing z-scores on new panel...')
for inst in CANDIDATES:
    z, m = compute_regime_zscore(df, inst)
    df[f'{inst}_z'] = z
    df[f'{inst}_mean'] = m
    valid = z.notna().sum()
    log(f'  {inst}: {valid} valid z-score days')


# ══════════════════════════════════════════════════════════════
# RUN SWEEP (established-C only, 5 instruments, 3 horizons)
# ══════════════════════════════════════════════════════════════

log('')
log('Running established-C sweep on new panel...')

new_rows = []

for inst in CANDIDATES:
    z_col = f'{inst}_z'

    shape_mask = pd.to_numeric(df['shape'], errors='coerce') == SHAPE_VAL
    dur_mask = (df['days_in_shape'] >= DUR_LO) & (df['days_in_shape'] <= DUR_HI)
    z_mask = df[z_col].abs() > Z_SIGNAL
    signal_mask = shape_mask & dur_mask & z_mask
    signal_indices = df.index[signal_mask]

    for h in HORIZONS:
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
            if pd.isna(signal_z):
                continue

            abs_signal_z = abs(signal_z)
            fwd_slice = df.iloc[idx+1:idx+h+1]
            if len(fwd_slice) == 0:
                continue

            n_signals += 1
            fwd_z_vals = fwd_slice[z_col].values
            fwd_z_abs = np.abs(fwd_z_vals)
            valid_fwd_z = ~np.isnan(fwd_z_vals)

            # Reversion
            reverted = False
            if valid_fwd_z.any():
                reverted = np.any(fwd_z_abs[valid_fwd_z] < Z_REVERSION)
            if reverted:
                n_reverted += 1

            # Continuation
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
        half_life = np.median(half_life_vals) if half_life_vals else np.nan
        net_bias = (reversion_rate - 50.0) if n_signals > 0 else np.nan
        low_n_flag = 'LOW-N' if (0 < n_signals < LOW_N_THRESH) else ''

        new_rows.append({
            'instrument': inst, 'horizon': h, 'n_signals': n_signals,
            'reversion_rate_pct': round(reversion_rate, 2) if not np.isnan(reversion_rate) else np.nan,
            'continuation_rate_pct': round(continuation_rate, 2) if not np.isnan(continuation_rate) else np.nan,
            'anti_rev_depth_z': round(anti_rev_depth, 3) if not np.isnan(anti_rev_depth) else np.nan,
            'half_life_days': round(half_life, 1) if not np.isnan(half_life) else np.nan,
            'net_directional_bias_pct': round(net_bias, 2) if not np.isnan(net_bias) else np.nan,
            'low_n_flag': low_n_flag,
        })

new_df = pd.DataFrame(new_rows)


# ══════════════════════════════════════════════════════════════
# LOAD OLD SWEEP RESULTS FOR COMPARISON
# ══════════════════════════════════════════════════════════════

old_df = pd.read_csv(OLD_SWEEP_PATH)
old_estab_c = old_df[
    (old_df['shape'] == 'C') &
    (old_df['duration_bucket'] == 'established') &
    (old_df['instrument'].isin(CANDIDATES))
][['instrument', 'horizon', 'n_signals', 'reversion_rate_pct',
   'continuation_rate_pct', 'anti_rev_depth_z', 'half_life_days',
   'net_directional_bias_pct', 'low_n_flag']].copy()


# ══════════════════════════════════════════════════════════════
# SIDE-BY-SIDE COMPARISON
# ══════════════════════════════════════════════════════════════

log('')
log('=' * 110)
log('SIDE-BY-SIDE COMPARISON: OLD PANEL vs NEW PANEL')
log('Established-C (21+ days), |z|>1.5 signals, 5 candidate instruments')
log('=' * 110)
log('')

def fmt(v, w=7, dec=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f'{"N/A":>{w}}'
    return f'{v:>{w}.{dec}f}'

header = (f'{"Inst":<8} {"Hor":>4} '
          f'{"N_old":>6} {"N_new":>6} '
          f'{"Rev%_old":>9} {"Rev%_new":>9} {"dRev":>6} '
          f'{"Cont%_old":>10} {"Cont%_new":>10} {"dCont":>6} '
          f'{"Depth_old":>10} {"Depth_new":>10} '
          f'{"HL_old":>7} {"HL_new":>7} '
          f'{"Bias_old":>9} {"Bias_new":>9}')
log(header)
log('-' * 140)

for inst in CANDIDATES:
    for h in HORIZONS:
        old_row = old_estab_c[(old_estab_c['instrument'] == inst) & (old_estab_c['horizon'] == h)]
        new_row = new_df[(new_df['instrument'] == inst) & (new_df['horizon'] == h)]

        if len(old_row) == 0 or len(new_row) == 0:
            continue

        o = old_row.iloc[0]
        n = new_row.iloc[0]

        n_old = int(o['n_signals'])
        n_new = int(n['n_signals'])
        rev_old = o['reversion_rate_pct']
        rev_new = n['reversion_rate_pct']
        cont_old = o['continuation_rate_pct']
        cont_new = n['continuation_rate_pct']
        depth_old = o['anti_rev_depth_z']
        depth_new = n['anti_rev_depth_z']
        hl_old = o['half_life_days']
        hl_new = n['half_life_days']
        bias_old = o['net_directional_bias_pct']
        bias_new = n['net_directional_bias_pct']

        d_rev = (rev_new - rev_old) if not (np.isnan(rev_new) or np.isnan(rev_old)) else np.nan
        d_cont = (cont_new - cont_old) if not (np.isnan(cont_new) or np.isnan(cont_old)) else np.nan

        flag = ''
        if n_old < LOW_N_THRESH or n_new < LOW_N_THRESH:
            flag = ' *'

        line = (f'{inst:<8} {h:>4} '
                f'{n_old:>6} {n_new:>6} '
                f'{fmt(rev_old, 9)} {fmt(rev_new, 9)} {fmt(d_rev, 6)} '
                f'{fmt(cont_old, 10)} {fmt(cont_new, 10)} {fmt(d_cont, 6)} '
                f'{fmt(depth_old, 10, 2)} {fmt(depth_new, 10, 2)} '
                f'{fmt(hl_old)} {fmt(hl_new)} '
                f'{fmt(bias_old, 9)} {fmt(bias_new, 9)}{flag}')
        log(line)
    log('')  # separator between instruments

log('* = at least one panel has LOW-N (<20 signals)')
log('')

# Summary statistics
log('SUMMARY:')
matched = new_df.merge(old_estab_c, on=['instrument', 'horizon'], suffixes=('_new', '_old'))
if len(matched) > 0:
    rev_diffs = matched['reversion_rate_pct_new'] - matched['reversion_rate_pct_old']
    cont_diffs = matched['continuation_rate_pct_new'] - matched['continuation_rate_pct_old']
    log(f'  Reversion rate delta (new - old): mean={rev_diffs.mean():+.1f}pp, '
        f'median={rev_diffs.median():+.1f}pp, range=[{rev_diffs.min():+.1f}, {rev_diffs.max():+.1f}]')
    log(f'  Continuation rate delta (new - old): mean={cont_diffs.mean():+.1f}pp, '
        f'median={cont_diffs.median():+.1f}pp, range=[{cont_diffs.min():+.1f}, {cont_diffs.max():+.1f}]')

    # Check directional agreement
    old_neg = (matched['net_directional_bias_pct_old'] < 0).sum()
    new_neg = (matched['net_directional_bias_pct_new'] < 0).sum()
    both_neg = ((matched['net_directional_bias_pct_old'] < 0) & (matched['net_directional_bias_pct_new'] < 0)).sum()
    log(f'  Cells with negative bias: old={old_neg}/15, new={new_neg}/15, both={both_neg}/15')


log('')
log('=' * 70)
log('COMPARISON COMPLETE -- CHECKPOINT 5')
log('=' * 70)

with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write('\n\n')
    f.write('\n'.join(log_lines))
    f.write('\n')

print(f'\nLog appended to: {LOG_FILE}')
print('Done.')
