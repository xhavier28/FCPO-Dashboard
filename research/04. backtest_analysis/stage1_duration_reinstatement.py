"""
Stage 1 — Duration Floor & Instrument Reinstatement
====================================================
Part 1: Duration floor sweep (10d/5d/3d/1d) at z>1.5, 4 instruments
Part 2: Reinstate M1-M2 + 4 butterflies at z>1.5/dur>=10d
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
Z_ENTRY = 1.5
Z_EXIT = 0.5
TIME_STOP_DAYS = 20
ROUNDTRIP_COST_MYR = 100.0
POINT_VALUE = 25.0
ROUNDTRIP_COST_POINTS = ROUNDTRIP_COST_MYR / POINT_VALUE  # 4.0

# All 9 instruments
ALL_INSTRUMENT_CONFIG = {
    'M1-M2':     {'near': 'M1', 'far': 'M2'},
    'M2-M3':     {'near': 'M2', 'far': 'M3'},
    'M3-M4':     {'near': 'M3', 'far': 'M4'},
    'M4-M5':     {'near': 'M4', 'far': 'M5'},
    'M5-M6':     {'near': 'M5', 'far': 'M6'},
}
BUTTERFLY_CONFIG = {
    'BF_M1M2M3': {'legs': ('M1', 'M2', 'M3')},
    'BF_M2M3M4': {'legs': ('M2', 'M3', 'M4')},
    'BF_M3M4M5': {'legs': ('M3', 'M4', 'M5')},
    'BF_M4M5M6': {'legs': ('M4', 'M5', 'M6')},
}

CORE_4 = ['M2-M3', 'M3-M4', 'M4-M5', 'M5-M6']
REINSTATED_5 = ['M1-M2', 'BF_M1M2M3', 'BF_M2M3M4', 'BF_M3M4M5', 'BF_M4M5M6']
ALL_9 = CORE_4 + REINSTATED_5

# ══════════════════════════════════════════════════════════════
# DATA SETUP
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

# Compute calendar spreads
for name, cfg in ALL_INSTRUMENT_CONFIG.items():
    df[name] = df[cfg['near']] - df[cfg['far']]

# Compute butterflies
for name, cfg in BUTTERFLY_CONFIG.items():
    m1, m2, m3 = cfg['legs']
    df[name] = df[m1] - 2 * df[m2] + df[m3]

# Regime-relative z-scores
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
for inst in ALL_9:
    df[f'{inst}_z'] = compute_regime_zscore(df, inst)
    print(f'  {inst}: {df[f"{inst}_z"].notna().sum()} valid days')

# PM predictions
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
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

def run_backtest(df, instruments, dur_thresh, z_entry=Z_ENTRY, label=''):
    """Run Stage 1 backtest. Returns (trades_df, per_inst_metrics, pooled_metrics)."""
    all_trades = []

    for inst in instruments:
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
                    # Shape survival: was shape unchanged throughout?
                    shape_survived = (exit_reason != 'invalidated')
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
                        'shape_survived': shape_survived,
                    })
                    position_open = False
                    continue

            if not position_open:
                if pd.isna(z) or pd.isna(row['pm_level']):
                    continue
                is_resting = shape in ('0.0', '1')
                dur_ok = row['days_in_shape'] >= dur_thresh
                z_extreme = abs(z) > z_entry
                pm_ok = row['pm_level'] == 0

                if is_resting and dur_ok and z_extreme and pm_ok:
                    position_open = True
                    entry_date, entry_spread, entry_z = date, spread, z
                    entry_shape = shape
                    entry_direction = -1 if z > 0 else 1
                    days_held = 0

    trades_df = pd.DataFrame(all_trades)
    if len(trades_df) == 0:
        empty = {'label': label, 'n_trades': 0, 'win_rate': 0, 'avg_win': 0,
                 'avg_loss': 0, 'total_pnl': 0, 'sharpe': np.nan, 'max_dd': 0,
                 'pct_take_profit': 0, 'pct_invalidated': 0, 'pct_time_stop': 0,
                 'avg_hp': 0, 'avg_hp_win': np.nan, 'avg_hp_loss': np.nan,
                 'shape_survival': 0}
        return trades_df, [], empty

    oos = trades_df[trades_df['oos']].reset_index(drop=True)
    per_inst = [compute_metrics(oos[oos['instrument'] == inst].reset_index(drop=True), inst) for inst in instruments]
    pooled = compute_metrics(oos.reset_index(drop=True), label)
    return trades_df, per_inst, pooled


def compute_metrics(trades, label):
    n = len(trades)
    if n == 0:
        return {'label': label, 'n_trades': 0, 'win_rate': 0, 'avg_win': 0,
                'avg_loss': 0, 'total_pnl': 0, 'sharpe': np.nan, 'max_dd': 0,
                'pct_take_profit': 0, 'pct_invalidated': 0, 'pct_time_stop': 0,
                'avg_hp': 0, 'avg_hp_win': np.nan, 'avg_hp_loss': np.nan,
                'shape_survival': 0}

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

    # Shape survival
    shape_survival = round(trades['shape_survived'].mean() * 100, 1) if 'shape_survived' in trades.columns else np.nan

    # Sharpe
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
        'shape_survival': shape_survival,
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


def fmt_sharpe(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return '  nan'
    return f'{s:.3f}'


def get_z_dist(trades):
    if len(trades) == 0:
        return {'min': np.nan, 'median': np.nan, 'max': np.nan}
    zs = trades['entry_z'].abs()
    return {'min': round(zs.min(), 3), 'median': round(zs.median(), 3), 'max': round(zs.max(), 3)}


# ══════════════════════════════════════════════════════════════
# PART 1 — DURATION FLOOR (4 instruments, z>1.5)
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('PART 1: DURATION FLOOR (z>1.5, 4 instruments)')
print('='*60)

DUR_THRESHOLDS = [10, 5, 3]  # will add 1 if 3 holds up

dur_results = []
for dur in DUR_THRESHOLDS:
    label = f'dur>={dur}d'
    print(f'\n--- {label} ---')
    trades, per_inst, pooled = run_backtest(df, CORE_4, dur_thresh=dur, label=label)
    oos = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()

    inst_counts = {inst: len(oos[oos['instrument'] == inst]) if len(oos) > 0 else 0 for inst in CORE_4}
    z_dist = get_z_dist(oos)

    dur_results.append({
        'dur': dur, 'label': label, 'pooled': pooled, 'per_inst': per_inst,
        'inst_counts': inst_counts, 'z_dist': z_dist, 'gate': gate_check(pooled),
    })

    print(f'  Pooled: n={pooled["n_trades"]}, win={pooled["win_rate"]}%, '
          f'sharpe={fmt_sharpe(pooled["sharpe"])}, total={pooled["total_pnl"]}, '
          f'shapeSurv={pooled["shape_survival"]}%, gate={gate_check(pooled)}')
    print(f'  Per-inst: {inst_counts}')
    print(f'  Entry |z|: min={z_dist["min"]}, med={z_dist["median"]}, max={z_dist["max"]}')

# Check if dur>=3 holds up — if so, test dur>=1
dur3_pooled = dur_results[-1]['pooled']
dur10_pooled = dur_results[0]['pooled']
sharpe_3 = dur3_pooled.get('sharpe', 0) or 0
sharpe_10 = dur10_pooled.get('sharpe', 0) or 0

if sharpe_3 > 0 and sharpe_3 >= sharpe_10 * 0.5:  # no catastrophic degradation
    print(f'\ndur>=3d Sharpe ({sharpe_3:.3f}) holds up vs dur>=10d ({sharpe_10:.3f}) — testing dur>=1d')
    dur = 1
    label = f'dur>={dur}d'
    trades, per_inst, pooled = run_backtest(df, CORE_4, dur_thresh=dur, label=label)
    oos = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()
    inst_counts = {inst: len(oos[oos['instrument'] == inst]) if len(oos) > 0 else 0 for inst in CORE_4}
    z_dist = get_z_dist(oos)
    dur_results.append({
        'dur': dur, 'label': label, 'pooled': pooled, 'per_inst': per_inst,
        'inst_counts': inst_counts, 'z_dist': z_dist, 'gate': gate_check(pooled),
    })
    print(f'  Pooled: n={pooled["n_trades"]}, win={pooled["win_rate"]}%, '
          f'sharpe={fmt_sharpe(pooled["sharpe"])}, total={pooled["total_pnl"]}, '
          f'shapeSurv={pooled["shape_survival"]}%, gate={gate_check(pooled)}')
    print(f'  Per-inst: {inst_counts}')
else:
    print(f'\ndur>=3d degraded significantly — skipping dur>=1d test')

# Comparison table
print('\n--- DURATION FLOOR COMPARISON TABLE ---')
header = f'  {"Variant":10s} {"n":>5s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"ShpSurv%":>9s} {"Gate":>22s}'
print(header)
for r in dur_results:
    p = r['pooled']
    print(f'  {r["label"]:10s} {p["n_trades"]:>5d} {p["win_rate"]:>5.1f}% {fmt_sharpe(p["sharpe"]):>7s} '
          f'{p["total_pnl"]:>10.1f} {p["avg_win"]:>8.2f} {p["avg_loss"]:>8.2f} '
          f'{p["max_dd"]:>8.1f} {p["shape_survival"]:>8.1f}% {r["gate"]:>22s}')


# ══════════════════════════════════════════════════════════════
# PART 2 — REINSTATE EXCLUDED INSTRUMENTS (z>1.5, dur>=10d)
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('PART 2: REINSTATE EXCLUDED INSTRUMENTS (z>1.5, dur>=10d)')
print('='*60)

# Run each reinstated instrument individually
reinstated_results = []
for inst in REINSTATED_5:
    print(f'\n--- {inst} ---')
    trades, per_inst, pooled = run_backtest(df, [inst], dur_thresh=10, label=inst)
    oos = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()
    z_dist = get_z_dist(oos)

    reinstated_results.append({
        'instrument': inst, 'pooled': pooled, 'z_dist': z_dist,
        'gate': gate_check(pooled),
    })

    p = pooled
    print(f'  n={p["n_trades"]}, win={p["win_rate"]}%, sharpe={fmt_sharpe(p["sharpe"])}, '
          f'total={p["total_pnl"]}, avgW={p["avg_win"]}, avgL={p["avg_loss"]}, '
          f'maxDD={p["max_dd"]}, shapeSurv={p["shape_survival"]}%')
    print(f'  Exit: TP={p["pct_take_profit"]}%, Inv={p["pct_invalidated"]}%, TS={p["pct_time_stop"]}%')
    print(f'  Entry |z|: min={z_dist["min"]}, med={z_dist["median"]}, max={z_dist["max"]}')

# 4-instrument baseline (already computed as dur>=10d in Part 1)
baseline_4 = dur_results[0]  # dur>=10d

# 9-instrument pooled
print('\n--- 9-INSTRUMENT POOLED (z>1.5, dur>=10d) ---')
trades_9, per_inst_9, pooled_9 = run_backtest(df, ALL_9, dur_thresh=10, label='ALL_9')
oos_9 = trades_9[trades_9['oos']].reset_index(drop=True) if len(trades_9) > 0 else pd.DataFrame()
inst_counts_9 = {inst: len(oos_9[oos_9['instrument'] == inst]) if len(oos_9) > 0 else 0 for inst in ALL_9}
z_dist_9 = get_z_dist(oos_9)

p9 = pooled_9
print(f'  n={p9["n_trades"]}, win={p9["win_rate"]}%, sharpe={fmt_sharpe(p9["sharpe"])}, '
      f'total={p9["total_pnl"]}, avgW={p9["avg_win"]}, avgL={p9["avg_loss"]}, '
      f'maxDD={p9["max_dd"]}, shapeSurv={p9["shape_survival"]}%')
print(f'  Per-inst: {inst_counts_9}')

# Comparison: 4 vs 9 instruments
print('\n--- 4 vs 9 INSTRUMENT COMPARISON ---')
b4 = baseline_4['pooled']
print(f'  4-inst: n={b4["n_trades"]}, win={b4["win_rate"]}%, sharpe={fmt_sharpe(b4["sharpe"])}, '
      f'total={b4["total_pnl"]}, maxDD={b4["max_dd"]}')
print(f'  9-inst: n={p9["n_trades"]}, win={p9["win_rate"]}%, sharpe={fmt_sharpe(p9["sharpe"])}, '
      f'total={p9["total_pnl"]}, maxDD={p9["max_dd"]}')

# Flag net-negative instruments
print('\n--- NET-NEGATIVE CHECK ---')
# Compute 4-inst + each reinstated instrument to see marginal impact
for rr in reinstated_results:
    inst = rr['instrument']
    test_set = CORE_4 + [inst]
    _, _, pooled_5 = run_backtest(df, test_set, dur_thresh=10, label=f'4+{inst}')
    delta_sharpe = (pooled_5['sharpe'] or 0) - (b4['sharpe'] or 0)
    delta_pnl = pooled_5['total_pnl'] - b4['total_pnl']
    flag = ''
    if delta_sharpe < -0.05 or delta_pnl < 0:
        flag = ' *** NET-NEGATIVE ***'
    elif delta_sharpe < 0:
        flag = ' (marginal drag on Sharpe)'
    print(f'  +{inst:12s}: Sharpe {fmt_sharpe(b4["sharpe"])} -> {fmt_sharpe(pooled_5["sharpe"])} '
          f'(delta={delta_sharpe:+.3f}), PnL {b4["total_pnl"]} -> {pooled_5["total_pnl"]} '
          f'(delta={delta_pnl:+.1f}){flag}')


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
lines.append(f'STAGE 1 — DURATION FLOOR & INSTRUMENT REINSTATEMENT — {timestamp}')
lines.append('='*70)

# ── Part 1: Duration floor ──────────────────────────────────
lines.append('')
lines.append('--- PART 1: DURATION FLOOR (z>1.5, PM L0, 4 instruments) ---')
lines.append(f'Instruments: {", ".join(CORE_4)}')
lines.append(f'z entry: >{Z_ENTRY}, z exit: <{Z_EXIT}, time stop: {TIME_STOP_DAYS}d')
lines.append(f'Cost: {ROUNDTRIP_COST_POINTS} pts/RT. OOS >= {OOS_START}')
lines.append('')

lines.append(f'  {"Variant":10s} {"n":>5s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} '
             f'{"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"ShpSurv%":>9s} {"Gate":>22s}')
lines.append(f'  {"-"*10} {"-"*5} {"-"*6} {"-"*7} {"-"*10} '
             f'{"-"*8} {"-"*8} {"-"*8} {"-"*9} {"-"*22}')

for r in dur_results:
    p = r['pooled']
    lines.append(f'  {r["label"]:10s} {p["n_trades"]:>5d} {p["win_rate"]:>5.1f}% {fmt_sharpe(p["sharpe"]):>7s} '
                 f'{p["total_pnl"]:>10.1f} {p["avg_win"]:>8.2f} {p["avg_loss"]:>8.2f} '
                 f'{p["max_dd"]:>8.1f} {p["shape_survival"]:>8.1f}% {r["gate"]:>22s}')

lines.append('')
lines.append('Per-instrument n (OOS):')
lines.append(f'  {"Variant":10s} ' + ' '.join(f'{inst:>6s}' for inst in CORE_4) + f' {"Total":>6s}')
lines.append(f'  {"-"*10} ' + ' '.join(['-'*6]*len(CORE_4)) + f' {"-"*6}')
for r in dur_results:
    counts = [r['inst_counts'][inst] for inst in CORE_4]
    total = sum(counts)
    lines.append(f'  {r["label"]:10s} ' + ' '.join(f'{c:>6d}' for c in counts) + f' {total:>6d}')

lines.append('')
lines.append('Entry |z| distribution (OOS):')
for r in dur_results:
    zd = r['z_dist']
    lines.append(f'  {r["label"]:10s}: min={zd["min"]}, median={zd["median"]}, max={zd["max"]}')

# ── Part 2: Instrument reinstatement ────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 2: INSTRUMENT REINSTATEMENT (z>1.5, PM L0, dur>=10d) ---')
lines.append('')

lines.append(f'  {"Instrument":12s} {"n":>4s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} '
             f'{"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"ShpSurv%":>9s} '
             f'{"TP%":>5s} {"Inv%":>5s} {"TS%":>5s}')
lines.append(f'  {"-"*12} {"-"*4} {"-"*6} {"-"*7} {"-"*10} '
             f'{"-"*8} {"-"*8} {"-"*8} {"-"*9} '
             f'{"-"*5} {"-"*5} {"-"*5}')

for rr in reinstated_results:
    p = rr['pooled']
    if p['n_trades'] == 0:
        lines.append(f'  {rr["instrument"]:12s}    0   (no trades)')
        continue
    lines.append(f'  {rr["instrument"]:12s} {p["n_trades"]:>4d} {p["win_rate"]:>5.1f}% {fmt_sharpe(p["sharpe"]):>7s} '
                 f'{p["total_pnl"]:>10.1f} {p["avg_win"]:>8.2f} {p["avg_loss"]:>8.2f} '
                 f'{p["max_dd"]:>8.1f} {p["shape_survival"]:>8.1f}% '
                 f'{p["pct_take_profit"]:>4.1f}% {p["pct_invalidated"]:>4.1f}% {p["pct_time_stop"]:>4.1f}%')

lines.append('')
lines.append('Entry |z| distribution (OOS):')
for rr in reinstated_results:
    zd = rr['z_dist']
    lines.append(f'  {rr["instrument"]:12s}: min={zd["min"]}, median={zd["median"]}, max={zd["max"]}')

# Part 2.3: 4 vs 9 comparison
lines.append('')
lines.append('--- PART 2.3: 4 vs 9 INSTRUMENT COMPARISON (z>1.5, dur>=10d) ---')
lines.append('')
lines.append(f'  {"Set":14s} {"n":>5s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} '
             f'{"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"ShpSurv%":>9s} {"Gate":>22s}')
lines.append(f'  {"-"*14} {"-"*5} {"-"*6} {"-"*7} {"-"*10} '
             f'{"-"*8} {"-"*8} {"-"*8} {"-"*9} {"-"*22}')

for lbl, p in [('4-inst (core)', b4), ('9-inst (all)', p9)]:
    g = gate_check(p)
    lines.append(f'  {lbl:14s} {p["n_trades"]:>5d} {p["win_rate"]:>5.1f}% {fmt_sharpe(p["sharpe"]):>7s} '
                 f'{p["total_pnl"]:>10.1f} {p["avg_win"]:>8.2f} {p["avg_loss"]:>8.2f} '
                 f'{p["max_dd"]:>8.1f} {p["shape_survival"]:>8.1f}% {g:>22s}')

lines.append('')
lines.append('9-inst per-instrument n (OOS):')
lines.append(f'  ' + ' '.join(f'{inst:>12s}' for inst in ALL_9))
lines.append(f'  ' + ' '.join(f'{inst_counts_9[inst]:>12d}' for inst in ALL_9))

# Part 2.4: Net-negative flags
lines.append('')
lines.append('--- PART 2.4: NET-NEGATIVE CHECK (marginal impact of adding each instrument) ---')
lines.append('')
for rr in reinstated_results:
    inst = rr['instrument']
    test_set = CORE_4 + [inst]
    _, _, pooled_5 = run_backtest(df, test_set, dur_thresh=10, label=f'4+{inst}')
    delta_sharpe = (pooled_5['sharpe'] or 0) - (b4['sharpe'] or 0)
    delta_pnl = pooled_5['total_pnl'] - b4['total_pnl']
    flag = ''
    if delta_sharpe < -0.05 or delta_pnl < 0:
        flag = '  *** NET-NEGATIVE ***'
    elif delta_sharpe < 0:
        flag = '  (marginal Sharpe drag)'
    lines.append(f'  +{inst:12s}: Sharpe {fmt_sharpe(b4["sharpe"])}->{fmt_sharpe(pooled_5["sharpe"])} '
                 f'({delta_sharpe:+.3f}), PnL {b4["total_pnl"]}->{pooled_5["total_pnl"]} '
                 f'({delta_pnl:+.1f}){flag}')

lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
