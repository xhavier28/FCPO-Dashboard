"""
Full Sweep v2 — Era-Aware, on daily_panel_backtest.csv
======================================================
Replaces the raw-spliced panel as default source.
Same 20 instruments × 5 shapes × 3 durations × 3 horizons sweep,
with an 'era' column per confirmed coverage spec.

Era assignments:
  "full"        — 13 instruments with consistent coverage
  "2008-2017" + "2018+" — M6-M7, M5-M7, M6-M8 (two rows per cell)
  "2018+" only  — M7-M8, M8-M9, BF_M6M7M8, BF_M7M8M9

Z-scores recomputed independently per era for split instruments.
Output: sweep_results_v2.csv
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

SHAPES = {'SB': 0.0, 'MB': 0.1, 'TB': 0.2, 'C': 1.0, 'F': 2.0}
SHAPE_NAMES = {v: k for k, v in SHAPES.items()}
DURATION_BUCKETS = {
    'early':       (1, 7),
    'mid':         (8, 20),
    'established': (21, 9999),
}

OUTPUT_DIR = r'C:/ClaudeCode/research/09. spread_behavioural_analysis'
PANEL_CSV = os.path.join(OUTPUT_DIR, 'daily_panel_backtest.csv')
SWEEP_V2_CSV = os.path.join(OUTPUT_DIR, 'sweep_results_v2.csv')
LOG_FILE = os.path.join(OUTPUT_DIR, 'spread_behaviour_log.txt')

# ── Instrument definitions ──────────────────────────────────

INSTRUMENT_GROUPS = {
    'M1-M2': 'live-9', 'M2-M3': 'live-9', 'M3-M4': 'live-9',
    'M4-M5': 'live-9', 'M5-M6': 'live-9',
    'BF_M1M2M3': 'live-9', 'BF_M2M3M4': 'live-9',
    'BF_M3M4M5': 'live-9', 'BF_M4M5M6': 'live-9',
    'M6-M7': 'back-month-5', 'M7-M8': 'back-month-5',
    'M8-M9': 'back-month-5',
    'BF_M6M7M8': 'back-month-5', 'BF_M7M8M9': 'back-month-5',
    'M1-M3': '2-month-gap-6', 'M2-M4': '2-month-gap-6',
    'M3-M5': '2-month-gap-6', 'M4-M6': '2-month-gap-6',
    'M5-M7': '2-month-gap-6', 'M6-M8': '2-month-gap-6',
}

ALL_20 = list(INSTRUMENT_GROUPS.keys())

# Era assignments (confirmed spec)
ERA_FULL = [
    'M1-M2', 'M2-M3', 'M3-M4', 'M4-M5', 'M5-M6',
    'BF_M1M2M3', 'BF_M2M3M4', 'BF_M3M4M5', 'BF_M4M5M6',
    'M1-M3', 'M2-M4', 'M3-M5', 'M4-M6',
]
ERA_SPLIT = ['M6-M7', 'M5-M7', 'M6-M8']  # two rows: "2008-2017" + "2018+"
ERA_2018_ONLY = ['M7-M8', 'M8-M9', 'BF_M6M7M8', 'BF_M7M8M9']

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)


# ══════════════════════════════════════════════════════════════
# Z-SCORE COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_regime_zscore(df, instrument_col):
    """Per-episode rolling 60-day z-score, min 10 obs."""
    zscores = pd.Series(np.nan, index=df.index)
    means = pd.Series(np.nan, index=df.index)
    for ep_id_val in df['episode_id'].dropna().unique():
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


def run_sweep_on_slice(df, instruments, era_label):
    """Run the full shape × duration × horizon sweep on a dataframe slice.
    Returns list of result dicts with era column."""
    rows = []
    for inst in instruments:
        z_col = f'{inst}_z'
        mean_col = f'{inst}_mean'
        group = INSTRUMENT_GROUPS[inst]

        for shape_name, shape_val in SHAPES.items():
            for dur_bucket, (dur_lo, dur_hi) in DURATION_BUCKETS.items():
                shape_mask = pd.to_numeric(df['shape'], errors='coerce') == shape_val
                dur_mask = (df['days_in_shape'] >= dur_lo) & (df['days_in_shape'] <= dur_hi)
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

                    rows.append({
                        'instrument': inst,
                        'group': group,
                        'era': era_label,
                        'shape': shape_name,
                        'duration_bucket': dur_bucket,
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
    return rows


# ══════════════════════════════════════════════════════════════
# LOAD PANEL
# ══════════════════════════════════════════════════════════════

log('=' * 70)
log(f'FULL SWEEP v2 -- ERA-AWARE -- {datetime.now().strftime("%Y-%m-%d %H:%M")}')
log('=' * 70)
log('')

panel = pd.read_csv(PANEL_CSV, parse_dates=['date'])
panel = panel.sort_values('date').reset_index(drop=True)
log(f'Panel: {len(panel)} rows, {panel["date"].min().date()} to {panel["date"].max().date()}')
log(f'Shape-matched: {panel["shape"].notna().sum()}')
log('')

# Coverage check
log('Coverage check (non-null days by era):')
era_0817_mask = panel['date'] < '2018-01-01'
era_18_mask = panel['date'] >= '2018-01-01'
n_0817 = era_0817_mask.sum()
n_18 = era_18_mask.sum()
log(f'  Panel rows: 2008-2017={n_0817}, 2018+={n_18}')

for inst in ALL_20:
    nn_0817 = panel.loc[era_0817_mask, inst].notna().sum()
    nn_18 = panel.loc[era_18_mask, inst].notna().sum()
    pct_0817 = nn_0817 / n_0817 * 100 if n_0817 > 0 else 0
    pct_18 = nn_18 / n_18 * 100 if n_18 > 0 else 0
    era_tag = 'full' if inst in ERA_FULL else ('split' if inst in ERA_SPLIT else '2018+')
    log(f'  {inst:<14} {pct_0817:>5.1f}% / {pct_18:>5.1f}%  -> {era_tag}')

log('')

# ══════════════════════════════════════════════════════════════
# COMPUTE Z-SCORES AND RUN SWEEP
# ══════════════════════════════════════════════════════════════

all_sweep_rows = []

# ── Part A: "full" instruments — z-scores on full panel ──────

log('Part A: "full" instruments (13) — z-scores on full panel')
for inst in ERA_FULL:
    z, m = compute_regime_zscore(panel, inst)
    panel[f'{inst}_z'] = z
    panel[f'{inst}_mean'] = m
    log(f'  {inst}: {z.notna().sum()} valid z-score days')

log('  Running sweep...')
full_rows = run_sweep_on_slice(panel, ERA_FULL, 'full')
all_sweep_rows.extend(full_rows)
log(f'  -> {len(full_rows)} rows')
log('')

# ── Part B: "2018+" only instruments — filter + z-scores ─────

log('Part B: "2018+" only instruments (4) — filter to 2018+ then z-scores')
panel_18 = panel[panel['date'] >= '2018-01-01'].copy().reset_index(drop=True)
log(f'  2018+ slice: {len(panel_18)} rows')

for inst in ERA_2018_ONLY:
    z, m = compute_regime_zscore(panel_18, inst)
    panel_18[f'{inst}_z'] = z
    panel_18[f'{inst}_mean'] = m
    log(f'  {inst}: {z.notna().sum()} valid z-score days')

log('  Running sweep...')
only18_rows = run_sweep_on_slice(panel_18, ERA_2018_ONLY, '2018+')
all_sweep_rows.extend(only18_rows)
log(f'  -> {len(only18_rows)} rows')
log('')

# ── Part C: era-split instruments — two separate runs ────────

log('Part C: era-split instruments (3) — independent z-scores per era')

# 2008-2017 slice
panel_0817 = panel[panel['date'] < '2018-01-01'].copy().reset_index(drop=True)
log(f'  2008-2017 slice: {len(panel_0817)} rows')

for inst in ERA_SPLIT:
    z, m = compute_regime_zscore(panel_0817, inst)
    panel_0817[f'{inst}_z'] = z
    panel_0817[f'{inst}_mean'] = m
    log(f'  {inst} (2008-2017): {z.notna().sum()} valid z-score days')

log('  Running sweep on 2008-2017...')
split_0817_rows = run_sweep_on_slice(panel_0817, ERA_SPLIT, '2008-2017')
all_sweep_rows.extend(split_0817_rows)
log(f'  -> {len(split_0817_rows)} rows')

# 2018+ slice (reuse panel_18, but need z-scores for split instruments)
for inst in ERA_SPLIT:
    z, m = compute_regime_zscore(panel_18, inst)
    panel_18[f'{inst}_z'] = z
    panel_18[f'{inst}_mean'] = m
    log(f'  {inst} (2018+): {z.notna().sum()} valid z-score days')

log('  Running sweep on 2018+...')
split_18_rows = run_sweep_on_slice(panel_18, ERA_SPLIT, '2018+')
all_sweep_rows.extend(split_18_rows)
log(f'  -> {len(split_18_rows)} rows')
log('')

# ══════════════════════════════════════════════════════════════
# COMBINE AND SAVE
# ══════════════════════════════════════════════════════════════

sweep_df = pd.DataFrame(all_sweep_rows)

# Sort for readability
era_order = {'full': 0, '2008-2017': 1, '2018+': 2}
sweep_df['_era_sort'] = sweep_df['era'].map(era_order)
inst_order = {inst: i for i, inst in enumerate(ALL_20)}
sweep_df['_inst_sort'] = sweep_df['instrument'].map(inst_order)
sweep_df = sweep_df.sort_values(['_inst_sort', '_era_sort', 'shape', 'duration_bucket', 'horizon'])
sweep_df = sweep_df.drop(columns=['_era_sort', '_inst_sort']).reset_index(drop=True)

sweep_df.to_csv(SWEEP_V2_CSV, index=False)

log('=' * 70)
log('SWEEP v2 RESULTS')
log('=' * 70)
log('')
log(f'Total rows: {len(sweep_df)}')
log(f'  Expected: 13*45 + 4*45 + 3*2*45 = 585 + 180 + 270 = 1035')
log(f'  Actual:   {len(sweep_df)}')
if len(sweep_df) != 1035:
    log(f'  *** ROW COUNT MISMATCH — INVESTIGATE ***')
else:
    log(f'  Row count matches expected.')
log('')

# Breakdown by era
for era in ['full', '2008-2017', '2018+']:
    n = (sweep_df['era'] == era).sum()
    n_signals = (sweep_df[sweep_df['era'] == era]['n_signals'] > 0).sum()
    n_zero = (sweep_df[sweep_df['era'] == era]['n_signals'] == 0).sum()
    n_low = (sweep_df[sweep_df['era'] == era]['low_n_flag'] == 'LOW-N').sum()
    log(f'  era="{era}": {n} rows, {n_signals} with signals, {n_zero} zero-signal, {n_low} LOW-N')
log('')

total_signals = (sweep_df['n_signals'] > 0).sum()
total_zero = (sweep_df['n_signals'] == 0).sum()
total_low = (sweep_df['low_n_flag'] == 'LOW-N').sum()
log(f'Overall: {total_signals} cells with signals, {total_zero} empty, {total_low} LOW-N')
log('')

# ══════════════════════════════════════════════════════════════
# FLICKER CHECK
# ══════════════════════════════════════════════════════════════

log('=' * 70)
log('FLICKER RATES CHECK')
log('=' * 70)
log('')
log('Flicker rates are computed from episode structure (shape changes,')
log('episode durations) which comes from daily_shape_log.csv — NOT from')
log('instrument price data. The shape classifier uses M1-M6 curve shape')
log('and is identical regardless of which price panel is used for spread/')
log('butterfly calculation.')
log('')
log('VERDICT: Flicker rates are UNAFFECTED by the panel swap. The existing')
log('flicker_rates.csv remains valid and does not need rebuilding.')
log('')

# ══════════════════════════════════════════════════════════════
# ESTABLISHED-C CLUSTER RE-CHECK
# ══════════════════════════════════════════════════════════════

log('=' * 70)
log('ESTABLISHED-C CLUSTER CONSISTENCY CHECK')
log('=' * 70)
log('')
log('Comparing v2 sweep established-C results against checkpoint 5 and')
log('era-comparison findings for validation.')
log('')

check_insts = ['M3-M5', 'M4-M6', 'M6-M7', 'M5-M7', 'M6-M8']

def fmt(v, w=7, dec=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f'{"N/A":>{w}}'
    return f'{v:>{w}.{dec}f}'

log(f'{"Inst":<8} {"Era":<12} {"Hor":>4} {"N":>5} {"Rev%":>7} {"Cont%":>7} {"Depth":>7} {"HL":>5} {"Bias":>7} {"Flag":>6}')
log('-' * 80)

for inst in check_insts:
    eras = sweep_df[sweep_df['instrument'] == inst]['era'].unique()
    for era in sorted(eras):
        rows = sweep_df[
            (sweep_df['instrument'] == inst) &
            (sweep_df['era'] == era) &
            (sweep_df['shape'] == 'C') &
            (sweep_df['duration_bucket'] == 'established')
        ]
        for _, r in rows.iterrows():
            line = (f'{inst:<8} {era:<12} {r["horizon"]:>4} '
                    f'{r["n_signals"]:>5} '
                    f'{fmt(r["reversion_rate_pct"])} '
                    f'{fmt(r["continuation_rate_pct"])} '
                    f'{fmt(r["anti_rev_depth_z"], 7, 2)} '
                    f'{fmt(r["half_life_days"], 5)} '
                    f'{fmt(r["net_directional_bias_pct"])} '
                    f'{r["low_n_flag"]:>6}')
            log(line)
    log('')

log('')

# ══════════════════════════════════════════════════════════════
# SAMPLE ROWS
# ══════════════════════════════════════════════════════════════

log('=' * 70)
log('SAMPLE ROWS (20 rows)')
log('=' * 70)
log('')

cols = ['instrument', 'group', 'era', 'shape', 'duration_bucket', 'horizon',
        'n_signals', 'reversion_rate_pct', 'continuation_rate_pct',
        'net_directional_bias_pct', 'low_n_flag']

# Pick samples: M1-M2 (full), M6-M7 (split), M7-M8 (2018+ only)
sample_parts = []

# M1-M2, C, established — 3 horizons (full)
s1 = sweep_df[(sweep_df['instrument'] == 'M1-M2') & (sweep_df['shape'] == 'C') &
              (sweep_df['duration_bucket'] == 'established')]
sample_parts.append(s1)

# M1-M2, SB, early — 3 horizons (full)
s2 = sweep_df[(sweep_df['instrument'] == 'M1-M2') & (sweep_df['shape'] == 'SB') &
              (sweep_df['duration_bucket'] == 'early')]
sample_parts.append(s2)

# M6-M7, C, established — both eras × 3 horizons (split)
s3 = sweep_df[(sweep_df['instrument'] == 'M6-M7') & (sweep_df['shape'] == 'C') &
              (sweep_df['duration_bucket'] == 'established')]
sample_parts.append(s3)

# M7-M8, C, established — 3 horizons (2018+ only)
s4 = sweep_df[(sweep_df['instrument'] == 'M7-M8') & (sweep_df['shape'] == 'C') &
              (sweep_df['duration_bucket'] == 'established')]
sample_parts.append(s4)

# M6-M8, SB, mid — both eras (split)
s5 = sweep_df[(sweep_df['instrument'] == 'M6-M8') & (sweep_df['shape'] == 'SB') &
              (sweep_df['duration_bucket'] == 'mid')]
sample_parts.append(s5)

sample = pd.concat(sample_parts).head(20)

header = '  '.join(f'{c:>12}' if c not in ('instrument', 'group', 'era') else f'{c:<12}' for c in cols)
log(header)
log('-' * 140)
for _, row in sample.iterrows():
    vals = []
    for c in cols:
        v = row[c]
        if c in ('instrument', 'group', 'era'):
            vals.append(f'{str(v):<12}')
        elif isinstance(v, float) and np.isnan(v):
            vals.append(f'{"N/A":>12}')
        elif isinstance(v, float):
            vals.append(f'{v:>12.1f}')
        elif isinstance(v, (int, np.integer)):
            vals.append(f'{int(v):>12}')
        else:
            vals.append(f'{str(v):>12}')
    log('  '.join(vals))

log('')

# ══════════════════════════════════════════════════════════════
# LOG
# ══════════════════════════════════════════════════════════════

log('=' * 70)
log('SWEEP v2 COMPLETE')
log('=' * 70)
log('')
log(f'Output: {SWEEP_V2_CSV}')
log(f'This sweep supersedes sweep_results.csv (old panel) as the default')
log(f'source for future analysis. The old file is retained for reference.')
log('')
log('Flicker rates: existing flicker_rates.csv remains valid (episode-')
log('structure only, no price dependency).')

with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write('\n\n')
    f.write('\n'.join(log_lines))
    f.write('\n')

print(f'\nLog appended to: {LOG_FILE}')
print(f'Sweep v2: {SWEEP_V2_CSV}')
print('Done.')
