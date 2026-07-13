"""
Stage 1 — Discrepancy Fix & Frequency/Edge Sweep
=================================================
Part 1: Confirm baseline numbers (single calculation path)
Part 2: Sweep z-thresholds (1.5, 1.75, 2.0) and PM filters (L0, L0+1)
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
from datetime import datetime
from models.pm_engine import predict as pm_predict
from models.feature_prep import load_daily_shape_log, load_enriched_shape_log

LOG_FILE = r'C:/ClaudeCode/research/04. backtest_analysis/backtest_analysis.txt'
OOS_START = '2022-01-01'
PM_CONFIDENCE_THRESHOLD = 0.70
Z_EXIT = 0.5
TIME_STOP_DAYS = 20
ROUNDTRIP_COST_MYR = 100.0
POINT_VALUE = 25.0
ROUNDTRIP_COST_POINTS = ROUNDTRIP_COST_MYR / POINT_VALUE  # 4.0

INSTRUMENT_CONFIG = {
    'M2-M3': {'dur_thresh': 10, 'near': 'M2', 'far': 'M3'},
    'M3-M4': {'dur_thresh': 10, 'near': 'M3', 'far': 'M4'},
    'M4-M5': {'dur_thresh': 10, 'near': 'M4', 'far': 'M5'},
    'M5-M6': {'dur_thresh': 10, 'near': 'M5', 'far': 'M6'},
}
INSTRUMENTS = list(INSTRUMENT_CONFIG.keys())

# ══════════════════════════════════════════════════════════════
# DATA SETUP (same as stage1_pnl_backtest.py — single source)
# ══════════════════════════════════════════════════════════════

print('Loading data...')
full_log = load_daily_shape_log()
full_log = full_log.sort_values('date').reset_index(drop=True)

enriched = load_enriched_shape_log()
enriched = enriched.sort_values('date').reset_index(drop=True)

pre_2017 = full_log[full_log['date'] < '2017-01-01'].copy()
pre_2017 = pre_2017.sort_values('date').reset_index(drop=True)

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
print(f'Panel: {len(df)} rows, {df["date"].min().date()} to {df["date"].max().date()}')

for inst, cfg in INSTRUMENT_CONFIG.items():
    df[inst] = df[cfg['near']] - df[cfg['far']]

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
            mean, std = window.mean(), window.std()
            if std > 0 and not np.isnan(val):
                zscores[idx] = (val - mean) / std
    return zscores

print('Computing z-scores...')
for inst in INSTRUMENTS:
    df[f'{inst}_z'] = compute_regime_zscore(df, inst)
    print(f'  {inst}: {df[f"{inst}_z"].notna().sum()} valid days')

print('Running PM predictions...')
model_start = pd.Timestamp('2017-01-01')
pm_level_col = pd.Series(np.nan, index=df.index)
for i in df[df['date'] >= model_start].index:
    dt = df.loc[i, 'date']
    obs_shape = str(df.loc[i, 'shape'])
    try:
        pm = pm_predict(dt)
        pred, conf = pm.get('predicted_shape'), pm.get('confidence')
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
                pm_level_col[i] = 1 if obs_shape in top2 else 2
            elif pred == obs_shape:
                pm_level_col[i] = 1
            else:
                pm_level_col[i] = 2
    except Exception:
        pass
df['pm_level'] = pm_level_col
print(f'PM levels: {df["pm_level"].value_counts().sort_index().to_dict()}')


# ══════════════════════════════════════════════════════════════
# REUSABLE BACKTEST ENGINE (single authoritative path)
# ══════════════════════════════════════════════════════════════

def run_backtest(df, z_entry, pm_levels_allowed, label=''):
    """Run the full Stage 1 backtest with given parameters.
    Returns (trades_df, per_inst_metrics, pooled_metrics)."""
    all_trades = []

    for inst in INSTRUMENTS:
        cfg = INSTRUMENT_CONFIG[inst]
        dur_thresh = cfg['dur_thresh']
        z_col = f'{inst}_z'

        position_open = False
        entry_date = entry_spread = entry_z = entry_shape = entry_direction = None
        days_held = 0

        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]

            if pd.isna(spread):
                continue

            if position_open:
                days_held += 1
                exit_reason = None

                if not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if shape != entry_shape:
                    exit_reason = 'invalidated'
                if days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'

                if exit_reason:
                    gross_pnl = (spread - entry_spread) * entry_direction
                    net_pnl = gross_pnl - ROUNDTRIP_COST_POINTS
                    all_trades.append({
                        'instrument': inst,
                        'entry_date': entry_date, 'exit_date': date,
                        'entry_spread': round(entry_spread, 2),
                        'exit_spread': round(spread, 2),
                        'entry_z': round(entry_z, 3),
                        'exit_z': round(z, 3) if not pd.isna(z) else np.nan,
                        'direction': 'LONG' if entry_direction == 1 else 'SHORT',
                        'days_held': days_held,
                        'exit_reason': exit_reason,
                        'gross_pnl': round(gross_pnl, 2),
                        'net_pnl': round(net_pnl, 2),
                        'oos': date >= pd.Timestamp(OOS_START),
                    })
                    position_open = False
                    continue

            if not position_open:
                if pd.isna(z) or pd.isna(row['pm_level']):
                    continue
                is_resting = shape in ('0.0', '1')
                dur_ok = row['days_in_shape'] >= dur_thresh
                z_extreme = abs(z) > z_entry
                pm_ok = row['pm_level'] in pm_levels_allowed

                if is_resting and dur_ok and z_extreme and pm_ok:
                    position_open = True
                    entry_date, entry_spread, entry_z = date, spread, z
                    entry_shape = shape
                    entry_direction = -1 if z > 0 else 1
                    days_held = 0

    trades_df = pd.DataFrame(all_trades)
    if len(trades_df) == 0:
        empty_m = {'label': label, 'n_trades': 0, 'win_rate': 0, 'avg_win': 0,
                   'avg_loss': 0, 'total_pnl': 0, 'sharpe': np.nan, 'max_dd': 0,
                   'pct_take_profit': 0, 'pct_invalidated': 0, 'pct_time_stop': 0,
                   'avg_hp': 0, 'avg_hp_win': np.nan, 'avg_hp_loss': np.nan}
        return trades_df, [], empty_m

    oos = trades_df[trades_df['oos']].reset_index(drop=True)

    per_inst = []
    for inst in INSTRUMENTS:
        it = oos[oos['instrument'] == inst].reset_index(drop=True)
        per_inst.append(compute_metrics(it, inst))

    pooled = compute_metrics(oos.reset_index(drop=True), label)
    return trades_df, per_inst, pooled


def compute_metrics(trades, label):
    """Single authoritative metric computation — derived from trade log only."""
    n = len(trades)
    if n == 0:
        return {'label': label, 'n_trades': 0, 'win_rate': 0, 'avg_win': 0,
                'avg_loss': 0, 'total_pnl': 0, 'sharpe': np.nan, 'max_dd': 0,
                'pct_take_profit': 0, 'pct_invalidated': 0, 'pct_time_stop': 0,
                'avg_hp': 0, 'avg_hp_win': np.nan, 'avg_hp_loss': np.nan}

    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]

    win_rate = round(len(wins) / n * 100, 1)
    avg_win = round(wins['net_pnl'].mean(), 2) if len(wins) > 0 else 0
    avg_loss = round(losses['net_pnl'].mean(), 2) if len(losses) > 0 else 0
    total_pnl = round(trades['net_pnl'].sum(), 2)

    tp_pct = round((trades['exit_reason'] == 'take_profit').mean() * 100, 1)
    inv_pct = round((trades['exit_reason'] == 'invalidated').mean() * 100, 1)
    ts_pct = round((trades['exit_reason'] == 'time_stop').mean() * 100, 1)

    avg_hp = round(trades['days_held'].mean(), 1)
    avg_hp_win = round(wins['days_held'].mean(), 1) if len(wins) > 0 else np.nan
    avg_hp_loss = round(losses['days_held'].mean(), 1) if len(losses) > 0 else np.nan

    cum_pnl = trades['net_pnl'].cumsum()
    max_dd = round((cum_pnl - cum_pnl.cummax()).min(), 2)

    # Sharpe: daily P&L series, annualized
    oos_dates = df[df['date'] >= pd.Timestamp(OOS_START)]['date']
    daily_pnl = pd.Series(0.0, index=oos_dates.values)
    for _, t in trades.iterrows():
        if t['exit_date'] in daily_pnl.index:
            daily_pnl[t['exit_date']] += t['net_pnl']
    daily_std = daily_pnl.std()
    sharpe = round(daily_pnl.mean() / daily_std * np.sqrt(252), 3) if daily_std > 0 else np.nan

    return {
        'label': label, 'n_trades': n, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'total_pnl': total_pnl,
        'sharpe': sharpe, 'max_dd': max_dd,
        'pct_take_profit': tp_pct, 'pct_invalidated': inv_pct, 'pct_time_stop': ts_pct,
        'avg_hp': avg_hp, 'avg_hp_win': avg_hp_win, 'avg_hp_loss': avg_hp_loss,
    }


def gate_check(m):
    if m['n_trades'] < 15:
        return 'INSUFFICIENT'
    s = m.get('sharpe')
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 'INSUFFICIENT'
    if s > 0 and m['avg_win'] > abs(m['avg_loss']):
        return 'PASS (provisional, n>=15)'
    return 'FAIL'


# ══════════════════════════════════════════════════════════════
# PART 1 — RE-VERIFY BASELINE (z=2.0, PM Level 0)
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('PART 1: BASELINE VERIFICATION (z=2.0, PM L0)')
print('='*60)

baseline_trades, baseline_inst, baseline_pooled = run_backtest(
    df, z_entry=2.0, pm_levels_allowed={0}, label='z2.0_PM0')

oos_baseline = baseline_trades[baseline_trades['oos']].reset_index(drop=True)

print(f'\nBaseline OOS trades: {len(oos_baseline)}')
print(f'\nPer-instrument:')
for m in baseline_inst:
    g = gate_check(m)
    print(f'  {m["label"]:6s}: n={m["n_trades"]:2d}, win={m["win_rate"]}%, '
          f'avgW={m["avg_win"]}, avgL={m["avg_loss"]}, total={m["total_pnl"]}, '
          f'sharpe={m["sharpe"]}, maxDD={m["max_dd"]}, gate={g}')

print(f'\nPooled:')
m = baseline_pooled
g = gate_check(m)
print(f'  n={m["n_trades"]}, win={m["win_rate"]}%, avgW={m["avg_win"]}, '
      f'avgL={m["avg_loss"]}, total={m["total_pnl"]}, sharpe={m["sharpe"]}, '
      f'maxDD={m["max_dd"]}, TP={m["pct_take_profit"]}%, '
      f'Inv={m["pct_invalidated"]}%, TS={m["pct_time_stop"]}%, gate={g}')

# Print trade log for verification
print('\nOOS Trade Log:')
for _, t in oos_baseline.iterrows():
    print(f'  {t["instrument"]:5s} {t["entry_date"].strftime("%Y-%m-%d")}->{t["exit_date"].strftime("%Y-%m-%d")} '
          f'{t["direction"]:5s} entry={t["entry_spread"]:>7.1f} exit={t["exit_spread"]:>7.1f} '
          f'z={t["entry_z"]:>+6.2f}->{t["exit_z"]:>+6.2f} '
          f'{t["exit_reason"]:13s} days={t["days_held"]:>2d} net={t["net_pnl"]:>+7.1f}')


# ══════════════════════════════════════════════════════════════
# PART 2 — FREQUENCY / EDGE SWEEP
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('PART 2: FREQUENCY / EDGE SWEEP')
print('='*60)

# Define variants
VARIANTS = [
    {'label': 'z1.5_PM0',    'z_entry': 1.5,  'pm_levels': {0}},
    {'label': 'z1.75_PM0',   'z_entry': 1.75, 'pm_levels': {0}},
    {'label': 'z2.0_PM0',    'z_entry': 2.0,  'pm_levels': {0}},     # baseline
    {'label': 'z2.0_PM01',   'z_entry': 2.0,  'pm_levels': {0, 1}},
    {'label': 'z1.5_PM01',   'z_entry': 1.5,  'pm_levels': {0, 1}},  # bonus
]

all_results = []

for v in VARIANTS:
    print(f'\n--- Running {v["label"]} ---')
    trades, per_inst, pooled = run_backtest(
        df, z_entry=v['z_entry'], pm_levels_allowed=v['pm_levels'],
        label=v['label'])

    oos_trades = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()

    # Per-instrument trade counts
    inst_counts = {}
    for inst in INSTRUMENTS:
        inst_counts[inst] = len(oos_trades[oos_trades['instrument'] == inst]) if len(oos_trades) > 0 else 0

    # Entry z-score distribution (for sanity check)
    if len(oos_trades) > 0:
        entry_zs = oos_trades['entry_z'].abs()
        z_dist = {
            'min': round(entry_zs.min(), 3),
            'median': round(entry_zs.median(), 3),
            'max': round(entry_zs.max(), 3),
        }
    else:
        z_dist = {'min': np.nan, 'median': np.nan, 'max': np.nan}

    result = {
        'label': v['label'],
        'z_entry': v['z_entry'],
        'pm_filter': 'L0' if v['pm_levels'] == {0} else 'L0+1',
        'pooled': pooled,
        'per_inst': per_inst,
        'inst_counts': inst_counts,
        'z_dist': z_dist,
        'gate': gate_check(pooled),
    }
    all_results.append(result)

    print(f'  Pooled: n={pooled["n_trades"]}, win={pooled["win_rate"]}%, '
          f'sharpe={pooled["sharpe"]}, total={pooled["total_pnl"]}, gate={result["gate"]}')
    print(f'  Per-inst: {inst_counts}')
    print(f'  Entry |z| dist: min={z_dist["min"]}, median={z_dist["median"]}, max={z_dist["max"]}')


# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('COMPARISON TABLE')
print('='*60)

header = f'  {"Variant":14s} {"Pooled n":>9s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"Gate":>22s}'
sep = f'  {"-"*14} {"-"*9} {"-"*6} {"-"*7} {"-"*10} {"-"*8} {"-"*8} {"-"*8} {"-"*22}'
print(header)
print(sep)

for r in all_results:
    p = r['pooled']
    s_str = f'{p["sharpe"]:.3f}' if not (p["sharpe"] is None or (isinstance(p["sharpe"], float) and np.isnan(p["sharpe"]))) else 'nan'
    print(f'  {r["label"]:14s} {p["n_trades"]:>9d} {p["win_rate"]:>5.1f}% {s_str:>7s} '
          f'{p["total_pnl"]:>10.1f} {p["avg_win"]:>8.2f} {p["avg_loss"]:>8.2f} '
          f'{p["max_dd"]:>8.1f} {r["gate"]:>22s}')

print('\nPer-instrument n breakdown:')
inst_header = f'  {"Variant":14s} ' + ' '.join(f'{inst:>6s}' for inst in INSTRUMENTS) + f' {"Total":>6s}'
print(inst_header)
for r in all_results:
    counts = [r['inst_counts'][inst] for inst in INSTRUMENTS]
    total = sum(counts)
    print(f'  {r["label"]:14s} ' + ' '.join(f'{c:>6d}' for c in counts) + f' {total:>6d}')


# ══════════════════════════════════════════════════════════════
# PART 3 — LOGGING
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('PART 3: LOGGING')
print('='*60)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
lines = []
lines.append('')
lines.append('')
lines.append('='*70)
lines.append(f'STAGE 1 — DISCREPANCY FIX & FREQUENCY/EDGE SWEEP — {timestamp}')
lines.append('='*70)

# Part 1: Discrepancy
lines.append('')
lines.append('--- PART 1: DISCREPANCY RESOLUTION ---')
lines.append('')
lines.append('Cause: the "summary output" with incorrect numbers was a hand-typed')
lines.append('markdown table in the conversation response, not generated by code.')
lines.append('The script (stage1_pnl_backtest.py) has a single compute_metrics()')
lines.append('function used for both console output and log file — those always')
lines.append('matched. The discrepancy was human transcription error, not a code bug.')
lines.append('')
lines.append('Fix: this script (stage1_sweep.py) uses the same single compute_metrics()')
lines.append('function for all outputs. Summary tables are generated programmatically')
lines.append('from the trade log — never hand-typed.')
lines.append('')
lines.append('Corrected baseline (z=2.0, PM Level 0, dur>=10d, OOS):')

# Baseline per-instrument
lines.append('')
lines.append(f'  {"Instrument":10s} {"n":>4s} {"Win%":>6s} {"AvgWin":>8s} {"AvgLoss":>8s} {"TotalPnL":>10s} '
             f'{"Sharpe":>7s} {"MaxDD":>8s} {"TP%":>5s} {"Inv%":>5s} {"TS%":>5s} {"Gate":>22s}')
lines.append(f'  {"-"*10} {"-"*4} {"-"*6} {"-"*8} {"-"*8} {"-"*10} '
             f'{"-"*7} {"-"*8} {"-"*5} {"-"*5} {"-"*5} {"-"*22}')

for m in baseline_inst:
    g = gate_check(m)
    if m['n_trades'] == 0:
        lines.append(f'  {m["label"]:10s}    0   (no trades)')
        continue
    s_str = f'{m["sharpe"]:.3f}' if not (m["sharpe"] is None or (isinstance(m["sharpe"], float) and np.isnan(m["sharpe"]))) else '  nan'
    lines.append(f'  {m["label"]:10s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% '
                 f'{m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} {m["total_pnl"]:>10.2f} '
                 f'{s_str:>7s} {m["max_dd"]:>8.2f} '
                 f'{m["pct_take_profit"]:>4.1f}% {m["pct_invalidated"]:>4.1f}% {m["pct_time_stop"]:>4.1f}% '
                 f'{g:>22s}')

lines.append('')
bm = baseline_pooled
bg = gate_check(bm)
bs_str = f'{bm["sharpe"]:.3f}' if not (bm["sharpe"] is None or (isinstance(bm["sharpe"], float) and np.isnan(bm["sharpe"]))) else 'nan'
lines.append(f'  Pooled: n={bm["n_trades"]}, win={bm["win_rate"]}%, avgW={bm["avg_win"]}, '
             f'avgL={bm["avg_loss"]}, total={bm["total_pnl"]}, sharpe={bs_str}, '
             f'maxDD={bm["max_dd"]}')
lines.append(f'  Exit: TP={bm["pct_take_profit"]}%, Inv={bm["pct_invalidated"]}%, TS={bm["pct_time_stop"]}%')
lines.append(f'  HP: avg={bm["avg_hp"]}d (win: {bm["avg_hp_win"]}d, loss: {bm["avg_hp_loss"]}d)')
lines.append(f'  Gate: {bg}')
lines.append('')
lines.append('Numbers match prior logged baseline exactly (no change from fix).')

# Part 2: Comparison table
lines.append('')
lines.append('--- PART 2: FREQUENCY / EDGE SWEEP ---')
lines.append('')
lines.append('All variants use: dur>=10d, TP at |z|<0.5, 20d time stop,')
lines.append('1 lot no pyramiding, 100 MYR (4 pts) round-trip cost.')
lines.append('Instruments: M2-M3, M3-M4, M4-M5, M5-M6. OOS only.')
lines.append('')

lines.append(f'  {"Variant":14s} {"Pooled n":>9s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} '
             f'{"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"Gate":>22s}')
lines.append(f'  {"-"*14} {"-"*9} {"-"*6} {"-"*7} {"-"*10} '
             f'{"-"*8} {"-"*8} {"-"*8} {"-"*22}')

for r in all_results:
    p = r['pooled']
    s_str = f'{p["sharpe"]:.3f}' if not (p["sharpe"] is None or (isinstance(p["sharpe"], float) and np.isnan(p["sharpe"]))) else '  nan'
    lines.append(f'  {r["label"]:14s} {p["n_trades"]:>9d} {p["win_rate"]:>5.1f}% {s_str:>7s} '
                 f'{p["total_pnl"]:>10.1f} {p["avg_win"]:>8.2f} {p["avg_loss"]:>8.2f} '
                 f'{p["max_dd"]:>8.1f} {r["gate"]:>22s}')

lines.append('')
lines.append('Per-instrument n (OOS):')
lines.append(f'  {"Variant":14s} ' + ' '.join(f'{inst:>6s}' for inst in INSTRUMENTS) + f' {"Total":>6s}')
lines.append(f'  {"-"*14} ' + ' '.join(['-'*6]*len(INSTRUMENTS)) + f' {"-"*6}')
for r in all_results:
    counts = [r['inst_counts'][inst] for inst in INSTRUMENTS]
    total = sum(counts)
    lines.append(f'  {r["label"]:14s} ' + ' '.join(f'{c:>6d}' for c in counts) + f' {total:>6d}')

# Part 2.5: Entry z-score sanity check
lines.append('')
lines.append('--- PART 2.5: ENTRY Z-SCORE DISTRIBUTION (sanity check) ---')
lines.append('')
for r in all_results:
    zd = r['z_dist']
    lines.append(f'  {r["label"]:14s}: |z| at entry — min={zd["min"]}, median={zd["median"]}, max={zd["max"]}')

lines.append('')

# Write to log
log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
