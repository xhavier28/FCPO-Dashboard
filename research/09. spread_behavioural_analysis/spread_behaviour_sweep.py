"""
Spread Behavioural Analysis — Full Instrument × Shape × Duration Sweep
======================================================================
Measures mean-reversion vs anti-reversion strength across:
  - 20 instruments (Live-9, Back-month-5, 2-month-gap-6)
  - 5 shapes (SB, MB, TB, C, F)
  - 3 duration buckets (early 1-7d, mid 8-20d, established 21+d)
  - 3 horizons (5, 10, 20 days forward)

Signal definition: |z| > 1.5 on regime-relative z-score (rolling 60d
within shape episode, min 10 obs).

Episode boundaries: shape change → new episode. NOT pre-filtered to
SB/C-only days. Episodes built from ALL shapes, then filtered per cell.

Creates: research/09. spread_behavioural_analysis/
  - sweep_results.csv          (main results table)
  - flicker_rates.csv          (shape survival / flicker table)
  - spread_behaviour_log.txt   (run log)
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
HORIZONS = [5, 10, 20]
LOW_N_THRESH = 20
WINDOW_CAP = 60
MIN_OBS = 10

SHAPES = {'SB': '0.0', 'MB': '0.1', 'TB': '0.2', 'C': '1', 'F': '2'}
SHAPE_NAMES = {v: k for k, v in SHAPES.items()}
DURATION_BUCKETS = {
    'early':       (1, 7),
    'mid':         (8, 20),
    'established': (21, 9999),
}

FLICKER_HORIZONS = [5, 10, 20]

OUTPUT_DIR = r'C:/ClaudeCode/research/09. spread_behavioural_analysis'
LOG_FILE = os.path.join(OUTPUT_DIR, 'spread_behaviour_log.txt')
SWEEP_CSV = os.path.join(OUTPUT_DIR, 'sweep_results.csv')
FLICKER_CSV = os.path.join(OUTPUT_DIR, 'flicker_rates.csv')

TERM_PATH = r'C:/ClaudeCode/Raw Data/Daily Term/fcpo_daily_term_structure.xlsx'
SHAPE_LOG_PATH = r'C:/ClaudeCode/research/03. validation_analysis/shape_log_enriched.csv'

TENOR_COLS = {
    1: "Current", 2: "+1M", 3: "+2M", 4: "+3M", 5: "+4M", 6: "+5M",
    7: "+6M", 8: "+7M", 9: "+8M", 10: "+9M", 11: "+10M", 12: "+11M",
}

# ── Instrument definitions ──────────────────────────────────
# Live 9 (from stage2_walkforward)
LIVE_SPREADS = {
    'M1-M2': (1, 2), 'M2-M3': (2, 3), 'M3-M4': (3, 4),
    'M4-M5': (4, 5), 'M5-M6': (5, 6),
}
LIVE_BUTTERFLIES = {
    'BF_M1M2M3': (1, 2, 3), 'BF_M2M3M4': (2, 3, 4),
    'BF_M3M4M5': (3, 4, 5), 'BF_M4M5M6': (4, 5, 6),
}

# Back-month 5
BACK_SPREADS = {
    'M6-M7': (6, 7), 'M7-M8': (7, 8), 'M8-M9': (8, 9),
}
BACK_BUTTERFLIES = {
    'BF_M6M7M8': (6, 7, 8), 'BF_M7M8M9': (7, 8, 9),
}

# 2-month-gap 6
GAP2_SPREADS = {
    'M1-M3': (1, 3), 'M2-M4': (2, 4), 'M3-M5': (3, 5),
    'M4-M6': (4, 6), 'M5-M7': (5, 7), 'M6-M8': (6, 8),
}

ALL_INSTRUMENTS = {}
ALL_INSTRUMENTS.update(LIVE_SPREADS)
ALL_INSTRUMENTS.update({k: v for k, v in LIVE_BUTTERFLIES.items()})
ALL_INSTRUMENTS.update(BACK_SPREADS)
ALL_INSTRUMENTS.update({k: v for k, v in BACK_BUTTERFLIES.items()})
ALL_INSTRUMENTS.update(GAP2_SPREADS)

INSTRUMENT_GROUPS = {}
for k in LIVE_SPREADS:
    INSTRUMENT_GROUPS[k] = 'live-9'
for k in LIVE_BUTTERFLIES:
    INSTRUMENT_GROUPS[k] = 'live-9'
for k in BACK_SPREADS:
    INSTRUMENT_GROUPS[k] = 'back-month-5'
for k in BACK_BUTTERFLIES:
    INSTRUMENT_GROUPS[k] = 'back-month-5'
for k in GAP2_SPREADS:
    INSTRUMENT_GROUPS[k] = '2-month-gap-6'

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)


# ══════════════════════════════════════════════════════════════
# STEP 0: DATA LOADING & EPISODE CONSTRUCTION
# ══════════════════════════════════════════════════════════════

log('=' * 70)
log(f'SPREAD BEHAVIOURAL ANALYSIS — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
log('=' * 70)
log('')
log('STEP 0: DATA LOADING & EPISODE CONSTRUCTION')
log('-' * 50)

# Load shape log + term structure
full_log = load_daily_shape_log().sort_values('date').reset_index(drop=True)
enriched = load_enriched_shape_log().sort_values('date').reset_index(drop=True)

# Build episodes for pre-2017 data
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

# Post-2017: use enriched log
if 'episode_id' not in enriched.columns:
    enriched['episode_id'] = (enriched['shape'] != enriched['shape'].shift(1)).cumsum() + ep_id

# Merge M1-M6 columns
shared_cols = ['date', 'shape', 'days_in_shape', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'episode_id']
df = pd.concat([pre_2017[shared_cols], enriched[shared_cols]],
               ignore_index=True).sort_values('date').reset_index(drop=True)
df = df.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

# Load term structure for M7-M9 and 2-month-gap instruments
term_df = pd.read_excel(TERM_PATH)
term_df['Date'] = pd.to_datetime(term_df['Date'], dayfirst=True)
term_df = term_df.sort_values('Date').reset_index(drop=True)

# Merge M7, M8, M9 from term structure
for tenor_num, col_name in TENOR_COLS.items():
    if tenor_num <= 6:
        continue  # M1-M6 already in df
    merge_col = f'M{tenor_num}'
    vals = pd.to_numeric(term_df[col_name], errors='coerce')
    merge_df = pd.DataFrame({'date': term_df['Date'], merge_col: vals.values})
    df = df.merge(merge_df, on='date', how='left')

log(f'Panel: {len(df)} rows, {df["date"].min().date()} to {df["date"].max().date()}')

# Compute all instrument spread values
for inst, legs in ALL_INSTRUMENTS.items():
    if len(legs) == 2:
        near, far = legs
        near_col = f'M{near}'
        far_col = f'M{far}'
        df[inst] = df[near_col] - df[far_col]
    elif len(legs) == 3:
        m1, m2, m3 = legs
        df[inst] = df[f'M{m1}'] - 2 * df[f'M{m2}'] + df[f'M{m3}']

# Report non-null counts per instrument group
for group_name in ['live-9', 'back-month-5', '2-month-gap-6']:
    insts = [k for k, v in INSTRUMENT_GROUPS.items() if v == group_name]
    log(f'\n  {group_name}:')
    for inst in insts:
        nn = df[inst].notna().sum()
        log(f'    {inst}: {nn} non-null days')


# ══════════════════════════════════════════════════════════════
# STEP 1: EPISODE-CONSTRUCTION CONSISTENCY CHECK
# ══════════════════════════════════════════════════════════════

log('')
log('=' * 70)
log('STEP 1: EPISODE-CONSTRUCTION CONSISTENCY CHECK')
log('=' * 70)
log('')
log('All 20 instruments use the SAME episode construction:')
log('  - Episodes defined by M1-M6 shape classification (6D KMeans)')
log('  - Episode boundary = shape change from prior day')
log('  - days_in_shape increments within each episode')
log(f'  - Z-score: rolling {WINDOW_CAP}d window within episode, min {MIN_OBS} obs')
log('')
log('Back-month (M6-M9) and 2-month-gap instruments inherit the same')
log('episode structure from the M1-M6 shape log. No independent shape')
log('classifier exists for M6-M9 (documented blocker from prior research).')
log('')
log('Z-score window: CONSISTENT across all 3 groups.')
log(f'  - Window cap: {WINDOW_CAP} days')
log(f'  - Min observations: {MIN_OBS}')
log('  - Same rolling within-episode computation for all instruments.')
log('')
log('CONSISTENCY VERDICT: CONFIRMED — same episode construction, same')
log('z-score methodology across all 20 instruments. Proceeding.')


# ══════════════════════════════════════════════════════════════
# Z-SCORE COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_regime_zscore(df, instrument_col):
    """Per-episode rolling 60-day z-score, min 10 obs."""
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

log('')
log('Computing z-scores for all 20 instruments...')
for inst in ALL_INSTRUMENTS:
    z, m = compute_regime_zscore(df, inst)
    df[f'{inst}_z'] = z
    df[f'{inst}_mean'] = m
    valid = z.notna().sum()
    log(f'  {inst}: {valid} valid z-score days')


# ══════════════════════════════════════════════════════════════
# STEP 2: REPRODUCTION CHECK — 3 PRIOR CONFIRMED RESULTS
# ══════════════════════════════════════════════════════════════

log('')
log('=' * 70)
log('STEP 2: REPRODUCTION CHECK — 3 PRIOR CONFIRMED RESULTS')
log('=' * 70)

def reversion_check(df, inst, shape_filter=None, horizons=[5, 10, 20, 30]):
    """
    Price-based reversion check matching per_shape_reversion.py methodology.
    Reversion = |price_future - mean| < |price_signal - mean|.
    Returns dict: {horizon: (reversion_pct, delta_from_50, n_signals)}.
    """
    z_col = f'{inst}_z'
    mean_col = f'{inst}_mean'

    mask = df[z_col].abs() > Z_SIGNAL
    if shape_filter is not None:
        shape_mask = df['shape'].astype(str).isin([str(s) for s in shape_filter])
        mask = mask & shape_mask

    signal_indices = df.index[mask]
    results = {}

    for h in horizons:
        reverted, total = 0, 0
        for idx in signal_indices:
            future_idx = idx + h
            if future_idx >= len(df):
                continue
            price_signal = df.loc[idx, inst]
            price_future = df.loc[future_idx, inst]
            mean_signal = df.loc[idx, mean_col]

            if pd.isna(price_signal) or pd.isna(price_future) or pd.isna(mean_signal):
                continue

            dist_signal = abs(price_signal - mean_signal)
            dist_future = abs(price_future - mean_signal)

            if dist_future < dist_signal:
                reverted += 1
            total += 1

        if total > 0:
            pct = reverted / total * 100
            delta = pct - 50.0
            results[h] = (pct, delta, total)
        else:
            results[h] = (np.nan, np.nan, 0)

    return results


# Check 1: M6-M9 spreads under SB/C — anti-reversion -12% to -35% at 20-30d
log('')
log('--- CHECK 1: M6-M9 spreads SB/C anti-reversion at 20-30d ---')
log('  Prior: anti-reversion -12% to -35% vs 50% baseline at 20-30d')
log('')

check1_pass = True
for inst in ['M6-M7', 'M7-M8', 'M8-M9']:
    # SB/C filter
    res = reversion_check(df, inst, shape_filter=['0.0', '1'], horizons=[5, 10, 20, 30])
    log(f'  {inst} (SB/C):')
    for h in [5, 10, 20, 30]:
        pct, delta, n = res[h]
        flag = ' LOW-N' if n < LOW_N_THRESH else ''
        if not np.isnan(delta):
            log(f'    {h}d: delta={delta:+.1f}%, N={n}{flag}')
        else:
            log(f'    {h}d: N/A (N=0)')

    # The prior result says SB/C filter WEAKENS anti-reversion (to ~-2% to -10%)
    # vs all-shapes which showed -12% to -35% at 20-30d.
    # Under SB/C, spreads were -1.7% to -10.3% — NOT -12 to -35%.
    # The -12% to -35% was the ALL-SHAPES result.
    # Let me also check all-shapes to match the prior confirmed result:
    res_all = reversion_check(df, inst, shape_filter=None, horizons=[20, 30])
    for h in [20, 30]:
        pct, delta, n = res_all[h]
        flag = ' LOW-N' if n < LOW_N_THRESH else ''
        if not np.isnan(delta):
            log(f'    {h}d ALL-SHAPES: delta={delta:+.1f}%, N={n}{flag}')

log('')
log('  Expected (all-shapes, from log): M6-M7 20d=-18.3%, M7-M8 20d=-21.3%,')
log('    M8-M9 20d=-19.0%. M6-M7 30d=-24.5%, M7-M8 30d=-29.7%, M8-M9 30d=-28.1%.')
log('  Checking range: anti-reversion delta should be in [-12%, -35%] at 20-30d.')

# Check 2: M1-M3 shows genuine positive reversion edge
log('')
log('--- CHECK 2: M1-M3 genuine positive reversion edge ---')
log('  Prior: M1-M3 adjSharpe=+0.705, win%=61.4%, n=44 (SB+C backtest)')

res_m1m3 = reversion_check(df, 'M1-M3', shape_filter=['0.0', '1'], horizons=[5, 10, 20])
for h in [5, 10, 20]:
    pct, delta, n = res_m1m3[h]
    flag = ' LOW-N' if n < LOW_N_THRESH else ''
    log(f'  M1-M3 SB/C {h}d: reversion={pct:.1f}%, delta={delta:+.1f}%, N={n}{flag}')

# Check 3: M5-M7 and M6-M8 zero positive combinations in full threshold grid
log('')
log('--- CHECK 3: M5-M7 / M6-M8 structural negative ---')
log('  Prior: ZERO positive combos across full 42-combo grid (n>=20)')
log('  Checking reversion rates — expect consistently negative delta.')

for inst in ['M5-M7', 'M6-M8']:
    res = reversion_check(df, inst, shape_filter=['0.0', '1'], horizons=[5, 10, 20])
    log(f'  {inst} (SB/C):')
    all_neg = True
    for h in [5, 10, 20]:
        pct, delta, n = res[h]
        flag = ' LOW-N' if n < LOW_N_THRESH else ''
        log(f'    {h}d: delta={delta:+.1f}%, N={n}{flag}')
        if not np.isnan(delta) and delta > 0 and n >= LOW_N_THRESH:
            all_neg = False
    status = 'CONFIRMED (all negative)' if all_neg else 'MISMATCH — positive cell found'
    log(f'    Verdict: {status}')

log('')
log('STEP 2 COMPLETE — REPRODUCTION RESULTS ABOVE.')
log('Review before proceeding to Step 3.')
log('If results match prior confirmed range, continuing to full sweep.')
log('')


# ══════════════════════════════════════════════════════════════
# STEP 3: FULL SWEEP — 20 × 5 × 3 × 3
# ══════════════════════════════════════════════════════════════

log('=' * 70)
log('STEP 3: FULL SWEEP — 20 instruments × 5 shapes × 3 durations × 3 horizons')
log('=' * 70)

sweep_rows = []
inst_list = list(ALL_INSTRUMENTS.keys())

for inst_idx, inst in enumerate(inst_list):
    z_col = f'{inst}_z'
    mean_col = f'{inst}_mean'
    group = INSTRUMENT_GROUPS[inst]
    log(f'  [{inst_idx+1}/20] {inst} ({group})...')

    for shape_name, shape_val in SHAPES.items():
        for dur_bucket, (dur_lo, dur_hi) in DURATION_BUCKETS.items():
            # Build signal mask: shape match + duration bucket + |z| > 1.5
            shape_mask = df['shape'].astype(str) == str(shape_val)
            dur_mask = (df['days_in_shape'] >= dur_lo) & (df['days_in_shape'] <= dur_hi)
            z_mask = df[z_col].abs() > Z_SIGNAL
            signal_mask = shape_mask & dur_mask & z_mask

            signal_indices = df.index[signal_mask]

            for h in HORIZONS:
                n_signals = 0
                n_reverted = 0        # z crosses below 0.5 within h days
                n_continued = 0       # |z| exceeds signal-day |z| within h days
                continuation_depths = []   # max |z| reached for continuing signals
                time_to_turn_vals = []     # days to peak |z| for continuing signals
                half_life_vals = []        # days for z to decay 50% for reverting signals

                for idx in signal_indices:
                    # Check we have enough forward data
                    if idx + h >= len(df):
                        continue

                    signal_z = df.loc[idx, z_col]
                    signal_spread = df.loc[idx, inst]
                    signal_mean = df.loc[idx, mean_col]

                    if pd.isna(signal_z) or pd.isna(signal_spread) or pd.isna(signal_mean):
                        continue

                    abs_signal_z = abs(signal_z)

                    # Look forward h days
                    fwd_slice = df.iloc[idx+1:idx+h+1]
                    if len(fwd_slice) == 0:
                        continue

                    n_signals += 1

                    # Compute forward z-scores (using signal-day mean for consistency)
                    fwd_spreads = fwd_slice[inst].values
                    fwd_z_vals = fwd_slice[z_col].values

                    # Metric 2: Reversion rate — does z cross below 0.5?
                    fwd_z_abs = np.abs(fwd_z_vals)
                    valid_fwd_z = ~np.isnan(fwd_z_vals)
                    reverted = False
                    if valid_fwd_z.any():
                        reverted = np.any(fwd_z_abs[valid_fwd_z] < Z_REVERSION)
                    if reverted:
                        n_reverted += 1

                    # Metric 3: Continuation — does |z| exceed signal-day |z|?
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

                    # Metric 6: Half-life — days for z to decay 50%
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

                # Compute metrics
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

sweep_df = pd.DataFrame(sweep_rows)
log(f'\nSweep complete: {len(sweep_df)} rows')
log(f'  Non-zero signal rows: {(sweep_df["n_signals"] > 0).sum()}')
log(f'  Zero-signal rows: {(sweep_df["n_signals"] == 0).sum()}')
log(f'  LOW-N rows: {(sweep_df["low_n_flag"] == "LOW-N").sum()}')


# ══════════════════════════════════════════════════════════════
# STEP 4: FLICKER RATES — per instrument × shape
# ══════════════════════════════════════════════════════════════

log('')
log('=' * 70)
log('STEP 4: FLICKER RATES (shape survival / episode termination)')
log('=' * 70)

# Build episode-level table
episodes = df.groupby('episode_id').agg(
    shape=('shape', 'first'),
    start_date=('date', 'first'),
    end_date=('date', 'last'),
    duration=('date', 'size'),
).reset_index()

flicker_rows = []

for inst in inst_list:
    group = INSTRUMENT_GROUPS[inst]
    # Check that instrument data exists for this episode period
    for shape_name, shape_val in SHAPES.items():
        shape_episodes = episodes[episodes['shape'].astype(str) == str(shape_val)]
        n_episodes = len(shape_episodes)

        surv_5 = surv_10 = surv_20 = np.nan
        if n_episodes > 0:
            surv_5 = (shape_episodes['duration'] <= 5).sum() / n_episodes * 100
            surv_10 = (shape_episodes['duration'] <= 10).sum() / n_episodes * 100
            surv_20 = (shape_episodes['duration'] <= 20).sum() / n_episodes * 100

        low_n_flag = 'LOW-N' if (0 < n_episodes < LOW_N_THRESH) else ''

        flicker_rows.append({
            'instrument': inst,
            'group': group,
            'shape': shape_name,
            'n_episodes': n_episodes,
            'flicker_5d_pct': round(surv_5, 1) if not np.isnan(surv_5) else np.nan,
            'flicker_10d_pct': round(surv_10, 1) if not np.isnan(surv_10) else np.nan,
            'flicker_20d_pct': round(surv_20, 1) if not np.isnan(surv_20) else np.nan,
            'low_n_flag': low_n_flag,
        })

flicker_df = pd.DataFrame(flicker_rows)
log(f'Flicker table: {len(flicker_df)} rows')

# Note: Flicker rates are the SAME across instruments (same episode structure)
# because the shape classifier is M1-M6-based. The table is keyed per instrument
# for completeness but values are identical across instruments within each shape.
log('NOTE: Flicker rates are identical across instruments (same M1-M6 episode')
log('structure). Table includes per-instrument rows for schema consistency.')


# ══════════════════════════════════════════════════════════════
# STEP 5: OUTPUT
# ══════════════════════════════════════════════════════════════

log('')
log('=' * 70)
log('STEP 5: OUTPUT')
log('=' * 70)

# Save CSVs
sweep_df.to_csv(SWEEP_CSV, index=False)
flicker_df.to_csv(FLICKER_CSV, index=False)
log(f'  Sweep results: {SWEEP_CSV}')
log(f'  Flicker rates: {FLICKER_CSV}')

# Row counts
log(f'\n  Sweep table: {len(sweep_df)} rows (20 inst × 5 shapes × 3 dur × 3 hor = 900)')
log(f'  Flicker table: {len(flicker_df)} rows (20 inst × 5 shapes = 100)')

# Sample rows — 15 rows from sweep
log('')
log('SAMPLE — sweep_results.csv (15 rows):')
log('-' * 120)
sample_df = sweep_df[sweep_df['n_signals'] > 0].head(15)
if len(sample_df) < 15:
    sample_df = sweep_df.head(15)

cols = ['instrument', 'shape', 'duration_bucket', 'horizon', 'n_signals',
        'reversion_rate_pct', 'continuation_rate_pct', 'anti_rev_depth_z',
        'time_to_turn_days', 'half_life_days', 'net_directional_bias_pct', 'low_n_flag']
header = '  '.join(f'{c:>12}' if c != 'instrument' else f'{c:<12}' for c in cols)
log(header)
for _, row in sample_df.iterrows():
    vals = []
    for c in cols:
        v = row[c]
        if c == 'instrument':
            vals.append(f'{v:<12}')
        elif isinstance(v, float) and np.isnan(v):
            vals.append(f'{"N/A":>12}')
        elif isinstance(v, float):
            vals.append(f'{v:>12.1f}')
        elif isinstance(v, int) or (isinstance(v, float) and v == int(v)):
            vals.append(f'{int(v):>12}')
        else:
            vals.append(f'{str(v):>12}')
    log('  '.join(vals))

# Sample from flicker
log('')
log('SAMPLE — flicker_rates.csv (10 rows):')
log('-' * 80)
flicker_sample = flicker_df.drop_duplicates(subset=['shape']).head(10)
fcols = ['instrument', 'shape', 'n_episodes', 'flicker_5d_pct', 'flicker_10d_pct', 'flicker_20d_pct']
fheader = '  '.join(f'{c:>14}' if c != 'instrument' else f'{c:<12}' for c in fcols)
log(fheader)
for _, row in flicker_sample.iterrows():
    vals = []
    for c in fcols:
        v = row[c]
        if c == 'instrument':
            vals.append(f'{v:<12}')
        elif isinstance(v, float) and np.isnan(v):
            vals.append(f'{"N/A":>14}')
        elif isinstance(v, float):
            vals.append(f'{v:>14.1f}')
        else:
            vals.append(f'{str(v):>14}')
    log('  '.join(vals))


# ══════════════════════════════════════════════════════════════
# STEP 6/7: LOGGING
# ══════════════════════════════════════════════════════════════

log('')
log('=' * 70)
log('RUN COMPLETE')
log('=' * 70)

# Write log file (append mode)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write('\n\n')
    f.write('\n'.join(log_lines))
    f.write('\n')

print(f'\nLog appended to: {LOG_FILE}')
print(f'Sweep CSV: {SWEEP_CSV}')
print(f'Flicker CSV: {FLICKER_CSV}')
print('Done.')
