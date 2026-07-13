"""
MR Duration Sensitivity Sweep + Holding Period Measurement
==========================================================
Supplements Stage 0 MR results (does NOT overwrite them).

Part 1: Duration threshold sensitivity at 5+/10+/15+/20+ days
Part 2: Holding period distribution for reverted trades
Part 3: Append results to backtest_analysis.txt
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
from datetime import datetime
from models.pm_engine import predict as pm_predict
from models.tm_engine import predict as tm_predict
from models.feature_prep import load_daily_shape_log, load_enriched_shape_log, TM_FEATURES

LOG_FILE = r'C:/ClaudeCode/research/04. backtest_analysis/backtest_analysis.txt'
OOS_START = '2022-01-01'
PM_CONFIDENCE_THRESHOLD = 0.70

# ── INSTRUMENTS ──────────────────────────────────────────────
CALENDAR_PAIRS = [
    ('M1-M2', 'M1', 'M2'),
    ('M2-M3', 'M2', 'M3'),
    ('M3-M4', 'M3', 'M4'),
    ('M4-M5', 'M4', 'M5'),
    ('M5-M6', 'M5', 'M6'),
]
BUTTERFLIES = [
    ('BF_M1M2M3', 'M1', 'M2', 'M3'),
    ('BF_M2M3M4', 'M2', 'M3', 'M4'),
    ('BF_M3M4M5', 'M3', 'M4', 'M5'),
    ('BF_M4M5M6', 'M4', 'M5', 'M6'),
]
ALL_INSTRUMENTS = [p[0] for p in CALENDAR_PAIRS] + [b[0] for b in BUTTERFLIES]

DURATION_THRESHOLDS = [5, 10, 15, 20]

# ══════════════════════════════════════════════════════════════
# DATA SETUP (replicates cells 1-4 from stage0_signal_validity)
# ══════════════════════════════════════════════════════════════

print('Loading data...')
full_log = load_daily_shape_log()
full_log = full_log.sort_values('date').reset_index(drop=True)

enriched = load_enriched_shape_log()
enriched = enriched.sort_values('date').reset_index(drop=True)

# Pre-2017: compute episode tracking
pre_2017 = full_log[full_log['date'] < '2017-01-01'].copy()
pre_2017 = pre_2017.sort_values('date').reset_index(drop=True)

days_list = []
episode_list = []
ep_id = 0
prev_shape = None
day_count = 0
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

# Enriched: add episode_id if missing
if 'episode_id' not in enriched.columns:
    enriched['episode_id'] = (enriched['shape'] != enriched['shape'].shift(1)).cumsum() + ep_id

shared_cols = ['date', 'shape', 'days_in_shape', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'episode_id']
df = pd.concat([
    pre_2017[shared_cols],
    enriched[shared_cols]
], ignore_index=True).sort_values('date').reset_index(drop=True)
df = df.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

print(f'Combined panel: {len(df)} rows, {df["date"].min().date()} to {df["date"].max().date()}')

# ── Compute spreads & butterflies ────────────────────────────
for name, near, far in CALENDAR_PAIRS:
    df[name] = df[near] - df[far]
for name, m1, m2, m3 in BUTTERFLIES:
    df[name] = df[m1] - 2 * df[m2] + df[m3]

# ── Regime-relative z-scores ────────────────────────────────
def compute_regime_zscore(df, instrument_col):
    zscores = pd.Series(np.nan, index=df.index)
    for ep_id in df['episode_id'].unique():
        mask = df['episode_id'] == ep_id
        ep_vals = df.loc[mask, instrument_col]
        if len(ep_vals) < 10:
            continue
        for i, (idx, val) in enumerate(ep_vals.items()):
            if i < 9:
                continue
            window_start = max(0, i - 59)
            window = ep_vals.iloc[window_start:i+1]
            mean = window.mean()
            std = window.std()
            if std > 0 and not np.isnan(val):
                zscores[idx] = (val - mean) / std
    return zscores

print('Computing z-scores...')
for inst in ALL_INSTRUMENTS:
    df[f'{inst}_z'] = compute_regime_zscore(df, inst)
    print(f'  {inst}: {df[f"{inst}_z"].notna().sum()} valid days')

# ── Forward z-scores (for reversion measurement) ────────────
for inst in ALL_INSTRUMENTS:
    z_col = f'{inst}_z'
    for h in range(1, 21):
        df[f'{inst}_z_t{h}'] = df[z_col].shift(-h)

# ── PM predictions (2017+ only) ─────────────────────────────
print('Running PM predictions...')
model_start = pd.Timestamp('2017-01-01')
model_dates_idx = df[df['date'] >= model_start].index

pm_level_col = pd.Series(np.nan, index=df.index)
for i in model_dates_idx:
    dt = df.loc[i, 'date']
    obs_shape = str(df.loc[i, 'shape'])
    try:
        pm = pm_predict(dt)
        pred = pm.get('predicted_shape')
        conf = pm.get('confidence')
        probs = pm.get('shape_probs', {})
        if pd.isna(pred) or pd.isna(conf):
            continue
        pred = str(pred)
        if pred == obs_shape and conf >= PM_CONFIDENCE_THRESHOLD:
            pm_level_col[i] = 0
        else:
            if isinstance(probs, dict) and probs:
                sorted_shapes = sorted(probs.items(), key=lambda x: -x[1])
                top2 = [s for s, _ in sorted_shapes[:2]]
                if obs_shape in top2:
                    pm_level_col[i] = 1
                else:
                    pm_level_col[i] = 2
            elif pred == obs_shape:
                pm_level_col[i] = 1
            else:
                pm_level_col[i] = 2
    except Exception:
        pass

df['pm_level'] = pm_level_col
print(f'PM levels computed. Distribution:\n{df["pm_level"].value_counts().sort_index()}')

# ══════════════════════════════════════════════════════════════
# PART 1 — DURATION SENSITIVITY SWEEP
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('PART 1: Duration Sensitivity Sweep')
print('='*60)

Z_THRESHOLD = 2.0
oos_mask = df['date'] >= OOS_START
resting_mask = df['shape'].isin(['0.0', '1'])

results_p1 = []  # rows for combined table

for inst in ALL_INSTRUMENTS:
    z_col = f'{inst}_z'
    for dur_thresh in DURATION_THRESHOLDS:
        # Signal ON: resting shape + days_in_shape >= dur_thresh + |z| > 2.0 + PM Level 0
        sig_on = (
            oos_mask &
            resting_mask &
            (df['days_in_shape'] >= dur_thresh) &
            (df[z_col].abs() > Z_THRESHOLD) &
            df[z_col].notna() &
            (df['pm_level'] == 0)
        )
        on_data = df[sig_on]
        n_on = len(on_data)

        # Episode count
        if n_on > 0:
            ep_changes = (on_data['episode_id'] != on_data['episode_id'].shift(1))
            # Also count breaks in consecutive Signal ON days
            idx_diff = on_data.index.to_series().diff()
            gap_breaks = idx_diff > 1
            n_episodes = (ep_changes | gap_breaks).sum()
        else:
            n_episodes = 0

        # Reversion rates at 5d, 10d, 20d
        rev_rates = {}
        for h in [5, 10, 20]:
            fwd_col = f'{inst}_z_t{h}'
            valid = on_data[[z_col, fwd_col]].dropna()
            if len(valid) > 0:
                partial = (valid[fwd_col].abs() < valid[z_col].abs()).mean() * 100
                rev_rates[f'rev_{h}d'] = round(partial, 1)
            else:
                rev_rates[f'rev_{h}d'] = np.nan

        results_p1.append({
            'instrument': inst,
            'dur_thresh': dur_thresh,
            'n_on': n_on,
            'n_episodes': n_episodes,
            'rev_5d': rev_rates['rev_5d'],
            'rev_10d': rev_rates['rev_10d'],
            'rev_20d': rev_rates['rev_20d'],
        })

        print(f'  {inst} dur>={dur_thresh}d: n_ON={n_on} ({n_episodes} ep) '
              f'rev@10d={rev_rates["rev_10d"]}%')

# Build combined table
p1_df = pd.DataFrame(results_p1)

print('\n--- COMBINED TABLE: n ON / rev@10d by instrument x duration ---')
for inst in ALL_INSTRUMENTS:
    row_data = p1_df[p1_df['instrument'] == inst]
    cells = []
    for _, r in row_data.iterrows():
        cells.append(f'{int(r["n_on"])}/{r["rev_10d"]}%')
    print(f'  {inst:12s}  ' + '  '.join(f'{c:>12s}' for c in cells))


# ══════════════════════════════════════════════════════════════
# PART 2 — HOLDING PERIOD MEASUREMENT
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('PART 2: Holding Period Measurement')
print('='*60)

holding_results = []  # per instrument, per duration threshold

for dur_thresh in DURATION_THRESHOLDS:
    for inst in ALL_INSTRUMENTS:
        z_col = f'{inst}_z'

        # Signal ON days (OOS)
        sig_on = (
            oos_mask &
            resting_mask &
            (df['days_in_shape'] >= dur_thresh) &
            (df[z_col].abs() > Z_THRESHOLD) &
            df[z_col].notna() &
            (df['pm_level'] == 0)
        )
        on_indices = df[sig_on].index.tolist()

        holding_days_list = []
        non_reverted = 0
        total_valid = 0

        for idx in on_indices:
            entry_z = abs(df.loc[idx, z_col])
            # Look forward up to 20 days for z crossing below 0.5
            reverted = False
            for h in range(1, 21):
                fwd_col = f'{inst}_z_t{h}'
                if fwd_col not in df.columns:
                    break
                fwd_z = df.loc[idx, fwd_col]
                if pd.isna(fwd_z):
                    continue
                # Reversion exit: |z| < 0.5
                if abs(fwd_z) < 0.5:
                    holding_days_list.append(h)
                    reverted = True
                    break
            if not reverted:
                non_reverted += 1
            total_valid += 1

        if holding_days_list:
            hp = np.array(holding_days_list)
            holding_results.append({
                'dur_thresh': dur_thresh,
                'instrument': inst,
                'n_signal_on': total_valid,
                'n_reverted': len(hp),
                'n_non_reverted': non_reverted,
                'mean_hp': round(np.mean(hp), 1),
                'median_hp': round(np.median(hp), 1),
                'min_hp': int(np.min(hp)),
                'max_hp': int(np.max(hp)),
                'pct_within_5d': round((hp <= 5).mean() * 100, 1),
                'pct_within_10d': round((hp <= 10).mean() * 100, 1),
                'pct_within_15d': round((hp <= 15).mean() * 100, 1),
                'pct_non_revert_20d': round(non_reverted / total_valid * 100, 1) if total_valid > 0 else np.nan,
            })
        elif total_valid > 0:
            holding_results.append({
                'dur_thresh': dur_thresh,
                'instrument': inst,
                'n_signal_on': total_valid,
                'n_reverted': 0,
                'n_non_reverted': non_reverted,
                'mean_hp': np.nan,
                'median_hp': np.nan,
                'min_hp': np.nan,
                'max_hp': np.nan,
                'pct_within_5d': 0.0,
                'pct_within_10d': 0.0,
                'pct_within_15d': 0.0,
                'pct_non_revert_20d': round(non_reverted / total_valid * 100, 1),
            })

hp_df = pd.DataFrame(holding_results)

# Print holding period results per duration threshold
for dur_thresh in DURATION_THRESHOLDS:
    sub = hp_df[hp_df['dur_thresh'] == dur_thresh]
    if len(sub) == 0:
        print(f'\n  dur>={dur_thresh}d: no Signal ON days')
        continue
    print(f'\n--- Holding Period: dur>={dur_thresh}d ---')
    for _, r in sub.iterrows():
        print(f'  {r["instrument"]:12s}: n={r["n_signal_on"]}, reverted={r["n_reverted"]}, '
              f'mean={r["mean_hp"]}d, median={r["median_hp"]}d, '
              f'min={r["min_hp"]}, max={r["max_hp"]}, '
              f'<=5d={r["pct_within_5d"]}%, <=10d={r["pct_within_10d"]}%, '
              f'<=15d={r["pct_within_15d"]}%, no_rev={r["pct_non_revert_20d"]}%')

    # Pooled stats for this duration threshold
    all_hp = []
    total_on = 0
    total_non_rev = 0
    for _, r in sub.iterrows():
        total_on += r['n_signal_on']
        total_non_rev += r['n_non_reverted']
    # Re-compute from raw data
    for inst in ALL_INSTRUMENTS:
        z_col = f'{inst}_z'
        sig_on = (
            oos_mask & resting_mask &
            (df['days_in_shape'] >= dur_thresh) &
            (df[z_col].abs() > Z_THRESHOLD) &
            df[z_col].notna() &
            (df['pm_level'] == 0)
        )
        for idx in df[sig_on].index:
            for h in range(1, 21):
                fwd_z = df.loc[idx, f'{inst}_z_t{h}']
                if pd.isna(fwd_z):
                    continue
                if abs(fwd_z) < 0.5:
                    all_hp.append(h)
                    break

    if all_hp:
        hp_arr = np.array(all_hp)
        print(f'  POOLED (dur>={dur_thresh}d): n_on={total_on}, reverted={len(hp_arr)}, '
              f'non_rev={total_non_rev}')
        print(f'    mean={np.mean(hp_arr):.1f}d, median={np.median(hp_arr):.1f}d, '
              f'min={np.min(hp_arr)}, max={np.max(hp_arr)}')
        print(f'    <=5d={((hp_arr<=5).mean()*100):.1f}%, '
              f'<=10d={((hp_arr<=10).mean()*100):.1f}%, '
              f'<=15d={((hp_arr<=15).mean()*100):.1f}%, '
              f'non_rev_20d={total_non_rev/total_on*100:.1f}%')


# ══════════════════════════════════════════════════════════════
# PART 3 — LOGGING
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('PART 3: Appending to backtest_analysis.txt')
print('='*60)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
lines = []
lines.append('')
lines.append('')
lines.append('='*70)
lines.append(f'MR — DURATION SENSITIVITY & HOLDING PERIOD — {timestamp}')
lines.append('='*70)
lines.append('')
lines.append('--- PART 1: DURATION SENSITIVITY SWEEP ---')
lines.append(f'z-threshold: {Z_THRESHOLD}, PM Level 0 filter, OOS only (>= {OOS_START})')
lines.append(f'Instruments: {", ".join(ALL_INSTRUMENTS)}')
lines.append('')

# Part 1.3: Combined table
header_dur = '  '.join(f'{"dur>=" + str(d) + "d":>14s}' for d in DURATION_THRESHOLDS)
lines.append(f'  {"Instrument":14s}  {header_dur}')
lines.append(f'  {"-"*14}  ' + '  '.join(['-'*14]*len(DURATION_THRESHOLDS)))

for inst in ALL_INSTRUMENTS:
    cells = []
    for dur in DURATION_THRESHOLDS:
        row = p1_df[(p1_df['instrument'] == inst) & (p1_df['dur_thresh'] == dur)]
        if len(row) > 0:
            r = row.iloc[0]
            n = int(r['n_on'])
            ep = int(r['n_episodes'])
            rev = r['rev_10d']
            if pd.isna(rev):
                cells.append(f'{n}({ep}ep)/nan')
            else:
                cells.append(f'{n}({ep}ep)/{rev}%')
        else:
            cells.append('—')
    lines.append(f'  {inst:14s}  ' + '  '.join(f'{c:>14s}' for c in cells))

lines.append('')
lines.append('  Format: n_ON(episodes)/rev@10d')
lines.append('')

# Part 1.4: Tradeoff summary
lines.append('--- PART 1.4: TRADEOFF SUMMARY ---')
lines.append('')
for dur in DURATION_THRESHOLDS:
    sub = p1_df[p1_df['dur_thresh'] == dur]
    total_n = sub['n_on'].sum()
    total_ep = sub['n_episodes'].sum()
    valid_rev = sub['rev_10d'].dropna()
    avg_rev = round(valid_rev.mean(), 1) if len(valid_rev) > 0 else np.nan
    lines.append(f'  dur>={dur:2d}d: total n_ON={int(total_n):3d}, '
                 f'total episodes={int(total_ep):3d}, '
                 f'avg rev@10d={avg_rev}%')
lines.append('')

# Compute gain vs baseline (20+)
baseline = p1_df[p1_df['dur_thresh'] == 20]
for dur in [5, 10, 15]:
    sub = p1_df[p1_df['dur_thresh'] == dur]
    n_gain = sub['n_on'].sum() - baseline['n_on'].sum()
    ep_gain = sub['n_episodes'].sum() - baseline['n_episodes'].sum()
    valid_sub = sub['rev_10d'].dropna()
    valid_base = baseline['rev_10d'].dropna()
    avg_sub = valid_sub.mean() if len(valid_sub) > 0 else np.nan
    avg_base = valid_base.mean() if len(valid_base) > 0 else np.nan
    rev_delta = round(avg_sub - avg_base, 1) if not (pd.isna(avg_sub) or pd.isna(avg_base)) else np.nan
    lines.append(f'  dur>={dur:2d}d vs 20+: +{int(n_gain)} Signal ON days, '
                 f'+{int(ep_gain)} episodes, '
                 f'rev@10d delta={rev_delta}pp')

lines.append('')

# Part 2: Holding period
lines.append('--- PART 2: HOLDING PERIOD DISTRIBUTION ---')
lines.append('  Exit condition: |z| < 0.5 within 20-day window')
lines.append('')

for dur_thresh in DURATION_THRESHOLDS:
    sub = hp_df[hp_df['dur_thresh'] == dur_thresh]
    if len(sub) == 0:
        lines.append(f'  [dur>={dur_thresh}d]: no Signal ON days')
        lines.append('')
        continue

    lines.append(f'  [dur>={dur_thresh}d]')

    for _, r in sub.iterrows():
        lines.append(f'    {r["instrument"]:12s}: n={int(r["n_signal_on"])}, '
                     f'reverted={int(r["n_reverted"])}, '
                     f'mean={r["mean_hp"]}d, median={r["median_hp"]}d, '
                     f'min={r["min_hp"]}, max={r["max_hp"]}, '
                     f'<=5d={r["pct_within_5d"]}%, <=10d={r["pct_within_10d"]}%, '
                     f'<=15d={r["pct_within_15d"]}%, '
                     f'non_rev_20d={r["pct_non_revert_20d"]}%')

    # Pooled
    all_hp = []
    total_on = 0
    total_non_rev = 0
    for _, r in sub.iterrows():
        total_on += int(r['n_signal_on'])
        total_non_rev += int(r['n_non_reverted'])
    for inst in ALL_INSTRUMENTS:
        z_col = f'{inst}_z'
        sig_on = (
            oos_mask & resting_mask &
            (df['days_in_shape'] >= dur_thresh) &
            (df[z_col].abs() > Z_THRESHOLD) &
            df[z_col].notna() &
            (df['pm_level'] == 0)
        )
        for idx in df[sig_on].index:
            for h in range(1, 21):
                fwd_z = df.loc[idx, f'{inst}_z_t{h}']
                if pd.isna(fwd_z):
                    continue
                if abs(fwd_z) < 0.5:
                    all_hp.append(h)
                    break

    if all_hp:
        hp_arr = np.array(all_hp)
        lines.append(f'    {"POOLED":12s}: n={total_on}, reverted={len(hp_arr)}, '
                     f'non_rev={total_non_rev}')
        lines.append(f'      mean={np.mean(hp_arr):.1f}d, median={np.median(hp_arr):.1f}d, '
                     f'min={np.min(hp_arr)}, max={np.max(hp_arr)}')
        lines.append(f'      <=5d={((hp_arr<=5).mean()*100):.1f}%, '
                     f'<=10d={((hp_arr<=10).mean()*100):.1f}%, '
                     f'<=15d={((hp_arr<=15).mean()*100):.1f}%, '
                     f'non_rev_20d={total_non_rev/total_on*100:.1f}%')
    lines.append('')

# Write to log
log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('\nDone.')
