"""
MR — Win%/PnL Backfill & Cross-Window Loss-Type Check
======================================================
Part 1: Backfill duration robustness table with win%/PnL per window
Part 2: Backfill M5-M6 exclusion table with win% per window
Part 3: Loss-type breakdown per window (4 buckets: A/B/C/D)
Part 4: Logging

Reuses PM/TM/z-score infrastructure from Stage 2.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
from datetime import datetime
from models.pm_engine import predict as pm_predict
from models.tm_engine import predict as tm_predict
from models.feature_prep import load_daily_shape_log, load_enriched_shape_log

LOG_FILE = r'C:/ClaudeCode/research/05. stage2_walkforward/stage2_walkforward_log.txt'
PM_CONFIDENCE_THRESHOLD = 0.70
Z_ENTRY = 1.5
Z_EXIT = 0.5
TIME_STOP_DAYS = 20
ROUNDTRIP_COST_MYR = 100.0
POINT_VALUE = 25.0
ROUNDTRIP_COST_POINTS = ROUNDTRIP_COST_MYR / POINT_VALUE  # 4.0
TM_REGIME_RISK_THRESH = 0.70

ALL_INSTRUMENT_CONFIG = {
    'M1-M2': {'near': 'M1', 'far': 'M2'},
    'M2-M3': {'near': 'M2', 'far': 'M3'},
    'M3-M4': {'near': 'M3', 'far': 'M4'},
    'M4-M5': {'near': 'M4', 'far': 'M5'},
    'M5-M6': {'near': 'M5', 'far': 'M6'},
}
BUTTERFLY_CONFIG = {
    'BF_M1M2M3': {'legs': ('M1', 'M2', 'M3')},
    'BF_M2M3M4': {'legs': ('M2', 'M3', 'M4')},
    'BF_M3M4M5': {'legs': ('M3', 'M4', 'M5')},
    'BF_M4M5M6': {'legs': ('M4', 'M5', 'M6')},
}

ALL_9 = ['M2-M3', 'M3-M4', 'M4-M5', 'M5-M6', 'M1-M2',
         'BF_M1M2M3', 'BF_M2M3M4', 'BF_M3M4M5', 'BF_M4M5M6']
ALL_8 = [i for i in ALL_9 if i != 'M5-M6']

WINDOWS = [
    {'name': 'W1 (2019-2020)', 'test_start': '2019-01-01', 'test_end': '2020-12-31'},
    {'name': 'W2 (2021-2022)', 'test_start': '2021-01-01', 'test_end': '2022-12-31'},
    {'name': 'W3 (2023-2024)', 'test_start': '2023-01-01', 'test_end': '2024-12-31'},
    {'name': 'W4 (2025-2026)', 'test_start': '2025-01-01', 'test_end': '2026-12-31'},
]

# ══════════════════════════════════════════════════════════════
# DATA SETUP (same as Stage 2)
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

for name, cfg in ALL_INSTRUMENT_CONFIG.items():
    df[name] = df[cfg['near']] - df[cfg['far']]
for name, cfg in BUTTERFLY_CONFIG.items():
    m1, m2, m3 = cfg['legs']
    df[name] = df[m1] - 2 * df[m2] + df[m3]

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

# TM cache
print('Pre-computing TM persistence probabilities...')
tm_cache = {}
for idx in df[df['date'] >= model_start].index:
    dt = df.loc[idx, 'date']
    dt_ts = pd.Timestamp(dt)
    try:
        result = tm_predict(dt_ts, '1w')
        if 'error' not in result:
            current_shape = str(result['current_shape'])
            all_probs = result.get('all_probs', {})
            tm_cache[dt_ts] = {
                'current_shape': current_shape,
                'persistence_prob': all_probs.get(current_shape, np.nan),
                'all_probs': all_probs,
            }
    except Exception:
        pass
print(f'  Cached {len(tm_cache)} TM predictions')


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE (extended: tracks max_abs_z during hold)
# ══════════════════════════════════════════════════════════════

def run_backtest(df, instruments, dur_thresh, z_entry, test_start, test_end):
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    all_trades = []
    for inst in instruments:
        z_col = f'{inst}_z'
        position_open = False
        entry_date = entry_spread = entry_z = entry_shape = entry_direction = None
        days_held = 0
        max_abs_z = 0.0
        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
            if pd.isna(spread):
                continue
            if position_open:
                days_held += 1
                # Track max abs z during hold
                if not pd.isna(z):
                    max_abs_z = max(max_abs_z, abs(z))
                exit_reason = None
                if shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < TM_REGIME_RISK_THRESH:
                            exit_reason = 'regime_risk'
                if not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if shape != entry_shape:
                    exit_reason = 'invalidated'
                if days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'
                if exit_reason:
                    gross_pnl = (spread - entry_spread) * entry_direction
                    net_pnl = gross_pnl - ROUNDTRIP_COST_POINTS
                    if ts <= date <= te:
                        all_trades.append({
                            'instrument': inst,
                            'entry_date': entry_date, 'exit_date': date,
                            'entry_spread': round(entry_spread, 2),
                            'exit_spread': round(spread, 2),
                            'entry_z': round(entry_z, 3),
                            'direction': 'LONG' if entry_direction == 1 else 'SHORT',
                            'days_held': days_held,
                            'exit_reason': exit_reason,
                            'gross_pnl': round(gross_pnl, 2),
                            'net_pnl': round(net_pnl, 2),
                            'shape_survived': exit_reason != 'invalidated',
                            'max_abs_z': round(max_abs_z, 3),
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
                    max_abs_z = abs(z)
    return pd.DataFrame(all_trades)


def compute_metrics(trades, label, test_start, test_end):
    n = len(trades)
    if n == 0:
        return {'label': label, 'n_trades': 0, 'win_rate': 0, 'avg_win': 0,
                'avg_loss': 0, 'total_pnl': 0, 'sharpe': np.nan, 'max_dd': 0,
                'avg_hp': 0, 'shape_survival': 0}
    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]
    win_rate = round(len(wins) / n * 100, 1)
    avg_win = round(wins['net_pnl'].mean(), 2) if len(wins) > 0 else 0
    avg_loss = round(losses['net_pnl'].mean(), 2) if len(losses) > 0 else 0
    total_pnl = round(trades['net_pnl'].sum(), 2)
    cum_pnl = trades['net_pnl'].cumsum()
    max_dd = round((cum_pnl - cum_pnl.cummax()).min(), 2)
    shape_survival = round(trades['shape_survived'].mean() * 100, 1)
    ts_dt, te_dt = pd.Timestamp(test_start), pd.Timestamp(test_end)
    window_dates = df[(df['date'] >= ts_dt) & (df['date'] <= te_dt)]['date']
    daily_pnl = pd.Series(0.0, index=window_dates.values)
    for _, t in trades.iterrows():
        if t['exit_date'] in daily_pnl.index:
            daily_pnl[t['exit_date']] += t['net_pnl']
    daily_std = daily_pnl.std()
    sharpe = round(daily_pnl.mean() / daily_std * np.sqrt(252), 3) if daily_std > 0 else np.nan
    return {
        'label': label, 'n_trades': n, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'total_pnl': total_pnl,
        'sharpe': sharpe, 'max_dd': max_dd, 'avg_hp': round(trades['days_held'].mean(), 1),
        'shape_survival': shape_survival,
    }


def fmt(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return '  nan'
    return f'{s:.3f}'


# ══════════════════════════════════════════════════════════════
# PART 1 — DURATION ROBUSTNESS TABLE BACKFILL
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 1: DURATION ROBUSTNESS TABLE BACKFILL (win%/PnL)')
print('='*70)

DUR_THRESHOLDS = [3, 5, 10]
dur_data = {}  # dur -> list of {win_rate, total_pnl, n} per window

for dur in DUR_THRESHOLDS:
    dur_data[dur] = []
    for w in WINDOWS:
        trades = run_backtest(df, ALL_9, dur_thresh=dur, z_entry=Z_ENTRY,
                              test_start=w['test_start'], test_end=w['test_end'])
        m = compute_metrics(trades, f'dur>={dur}d', w['test_start'], w['test_end'])
        # Count trades per year
        if len(trades) > 0:
            trades['exit_year'] = trades['exit_date'].dt.year
            yearly_counts = trades.groupby('exit_year').size().to_dict()
        else:
            yearly_counts = {}
        dur_data[dur].append({
            'win_rate': m['win_rate'], 'total_pnl': m['total_pnl'],
            'n': m['n_trades'], 'yearly_counts': yearly_counts,
        })
        print(f'  dur>={dur}d, {w["name"]}: n={m["n_trades"]}, win={m["win_rate"]}%, PnL={m["total_pnl"]}')

# Print updated table
print('\n--- UPDATED DURATION ROBUSTNESS TABLE ---')
header = (f'  {"Duration":>8s}  {"W1 Win%":>7s}  {"W1 PnL":>8s}  {"W2 Win%":>7s}  {"W2 PnL":>8s}  '
          f'{"W3 Win%":>7s}  {"W3 PnL":>8s}  {"W4 Win%":>7s}  {"W4 PnL":>8s}  '
          f'{"Avg Win%":>8s}  {"Total PnL":>9s}')
print(header)
for dur in DUR_THRESHOLDS:
    d = dur_data[dur]
    avg_win = np.mean([x['win_rate'] for x in d])
    total_pnl = sum(x['total_pnl'] for x in d)
    print(f'  dur>={dur:>2d}d  {d[0]["win_rate"]:>6.1f}%  {d[0]["total_pnl"]:>8.1f}  '
          f'{d[1]["win_rate"]:>6.1f}%  {d[1]["total_pnl"]:>8.1f}  '
          f'{d[2]["win_rate"]:>6.1f}%  {d[2]["total_pnl"]:>8.1f}  '
          f'{d[3]["win_rate"]:>6.1f}%  {d[3]["total_pnl"]:>8.1f}  '
          f'{avg_win:>7.1f}%  {total_pnl:>9.1f}')

# Trades per year
print('\n--- TRADES PER YEAR (dur>=3d, locked config) ---')
d3 = dur_data[3]
all_years = sorted(set().union(*[x['yearly_counts'].keys() for x in d3]))
year_header = f'  {"Year":>6s}'
for y in all_years:
    year_header += f'  {y:>4d}'
print(year_header)
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"][:2]:>6s}'
    for y in all_years:
        row += f'  {d3[i]["yearly_counts"].get(y, 0):>4d}'
    print(row)


# ══════════════════════════════════════════════════════════════
# PART 2 — M5-M6 EXCLUSION TABLE BACKFILL
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 2: M5-M6 EXCLUSION TABLE BACKFILL (win%)')
print('='*70)

excl_data = {}  # '9-inst' or '8-inst' -> list of {win_rate, total_pnl, n} per window

for inst_set_name, inst_set in [('9-inst', ALL_9), ('8-inst', ALL_8)]:
    excl_data[inst_set_name] = []
    for w in WINDOWS:
        trades = run_backtest(df, inst_set, dur_thresh=3, z_entry=Z_ENTRY,
                              test_start=w['test_start'], test_end=w['test_end'])
        m = compute_metrics(trades, inst_set_name, w['test_start'], w['test_end'])
        excl_data[inst_set_name].append({
            'win_rate': m['win_rate'], 'total_pnl': m['total_pnl'],
            'n': m['n_trades'],
        })
        print(f'  {inst_set_name}, {w["name"]}: n={m["n_trades"]}, win={m["win_rate"]}%, PnL={m["total_pnl"]}')

# Print updated table
print('\n--- UPDATED M5-M6 EXCLUSION TABLE ---')
header = (f'  {"Window":16s}  {"9i Win%":>7s}  {"8i Win%":>7s}  '
          f'{"9i PnL":>8s}  {"8i PnL":>8s}  {"9i n":>5s}  {"8i n":>5s}')
print(header)
for i, w in enumerate(WINDOWS):
    r9 = excl_data['9-inst'][i]
    r8 = excl_data['8-inst'][i]
    print(f'  {w["name"]:16s}  {r9["win_rate"]:>6.1f}%  {r8["win_rate"]:>6.1f}%  '
          f'{r9["total_pnl"]:>8.1f}  {r8["total_pnl"]:>8.1f}  '
          f'{r9["n"]:>5d}  {r8["n"]:>5d}')


# ══════════════════════════════════════════════════════════════
# PART 3 — LOSS-TYPE BREAKDOWN PER WINDOW
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 3: LOSS-TYPE BREAKDOWN PER WINDOW')
print('='*70)
print('Final locked config: z>1.5, dur>=3d, 9 instruments, PM L0, TM RR <70%')
print('Bucket definitions:')
print('  A (Regime break): exit_reason == invalidated')
print('  B (Adverse extension): exit_reason == time_stop AND max|z| > entry|z|')
print('  C (Stalled): exit_reason == time_stop AND max|z| <= entry|z|')
print('  D (Regime-risk exit): exit_reason == regime_risk (reported separately)')

window_loss_data = []  # list of dicts per window

for w in WINDOWS:
    print(f'\n{"="*60}')
    print(f'WINDOW: {w["name"]}')
    print(f'{"="*60}')

    trades = run_backtest(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                          test_start=w['test_start'], test_end=w['test_end'])

    if len(trades) == 0:
        print('  No trades.')
        window_loss_data.append({'window': w['name'], 'n_total': 0,
                                  'bucketA': [], 'bucketB': [], 'bucketC': [], 'bucketD': []})
        continue

    n_total = len(trades)
    losses = trades[trades['net_pnl'] <= 0].copy()
    n_losses = len(losses)

    # Bucket D: regime_risk exits (can be wins or losses)
    regime_risk_trades = trades[trades['exit_reason'] == 'regime_risk'].copy()
    n_regime_risk = len(regime_risk_trades)
    rr_net_pnl = regime_risk_trades['net_pnl'].sum() if n_regime_risk > 0 else 0.0
    rr_wins = (regime_risk_trades['net_pnl'] > 0).sum() if n_regime_risk > 0 else 0
    rr_losses_n = (regime_risk_trades['net_pnl'] <= 0).sum() if n_regime_risk > 0 else 0

    # For loss classification, only consider losing trades NOT from regime_risk
    # But also include regime_risk losses in the D bucket
    losses_excl_rr = losses[losses['exit_reason'] != 'regime_risk'].copy()
    n_losses_excl_rr = len(losses_excl_rr)

    # Bucket A: invalidated (regime break)
    bucket_a = losses_excl_rr[losses_excl_rr['exit_reason'] == 'invalidated']
    # Bucket B: time_stop AND max|z| > entry|z| (adverse extension)
    time_stop_losses = losses_excl_rr[losses_excl_rr['exit_reason'] == 'time_stop']
    bucket_b = time_stop_losses[time_stop_losses['max_abs_z'] > time_stop_losses['entry_z'].abs()]
    # Bucket C: time_stop AND max|z| <= entry|z| (stalled)
    bucket_c = time_stop_losses[time_stop_losses['max_abs_z'] <= time_stop_losses['entry_z'].abs()]
    # Take-profit losses (net_pnl <= 0 due to costs, but exit_reason = take_profit)
    tp_losses = losses_excl_rr[losses_excl_rr['exit_reason'] == 'take_profit']

    print(f'\n  Total trades: {n_total}, Total losses (excl regime-risk): {n_losses_excl_rr}')
    if len(tp_losses) > 0:
        print(f'  Note: {len(tp_losses)} take-profit exits are net losses (cost-induced)')

    # Classify take-profit losses: they reverted to mean but cost ate the profit
    # Include them in a note but they don't fit A/B/C cleanly
    # For percentage calculation, base = bucket_a + bucket_b + bucket_c + tp_losses
    n_classifiable = len(bucket_a) + len(bucket_b) + len(bucket_c) + len(tp_losses)

    def bucket_stats(bucket, name, n_base):
        n = len(bucket)
        pct = round(n / n_base * 100, 1) if n_base > 0 else 0.0
        total_pnl = round(bucket['net_pnl'].sum(), 2) if n > 0 else 0.0
        avg_loss = round(bucket['net_pnl'].mean(), 2) if n > 0 else 0.0
        if n > 0:
            top_inst = bucket['instrument'].value_counts().head(3)
            top_str = ', '.join([f'{inst}({cnt})' for inst, cnt in top_inst.items()])
        else:
            top_str = '-'
        return {'name': name, 'n': n, 'pct': pct, 'total_pnl': total_pnl,
                'avg_loss': avg_loss, 'top_instruments': top_str}

    stats_a = bucket_stats(bucket_a, 'A (Regime break)', n_classifiable)
    stats_b = bucket_stats(bucket_b, 'B (Adverse ext)', n_classifiable)
    stats_c = bucket_stats(bucket_c, 'C (Stalled)', n_classifiable)
    stats_tp = bucket_stats(tp_losses, 'TP-loss (cost)', n_classifiable)

    print(f'\n  {"Bucket":20s}  {"n":>4s}  {"% losses":>8s}  {"Total PnL":>10s}  {"Avg Loss":>9s}  {"Top instruments":s}')
    print(f'  {"-"*20}  {"-"*4}  {"-"*8}  {"-"*10}  {"-"*9}  {"-"*25}')
    for s in [stats_a, stats_b, stats_c, stats_tp]:
        print(f'  {s["name"]:20s}  {s["n"]:>4d}  {s["pct"]:>7.1f}%  {s["total_pnl"]:>10.1f}  {s["avg_loss"]:>9.2f}  {s["top_instruments"]}')

    print(f'\n  Regime-risk exits (Bucket D — reported separately):')
    print(f'    n={n_regime_risk}, net PnL={rr_net_pnl:.1f}, wins={rr_wins}, losses={rr_losses_n}')
    if n_regime_risk > 0:
        avg_rr = regime_risk_trades['net_pnl'].mean()
        print(f'    avg PnL per trade={avg_rr:.2f}, net {"GAIN" if rr_net_pnl > 0 else "LOSS"}')

    # Trades per year
    trades['exit_year'] = trades['exit_date'].dt.year
    yearly = trades.groupby('exit_year').size()
    print(f'\n  Trades per year: {dict(yearly)}')

    window_loss_data.append({
        'window': w['name'],
        'n_total': n_total,
        'n_losses': n_losses_excl_rr,
        'n_classifiable': n_classifiable,
        'stats_a': stats_a,
        'stats_b': stats_b,
        'stats_c': stats_c,
        'stats_tp': stats_tp,
        'n_regime_risk': n_regime_risk,
        'rr_net_pnl': rr_net_pnl,
        'rr_wins': rr_wins,
        'rr_losses_n': rr_losses_n,
        'yearly': dict(yearly),
    })


# ── 3.3 Cross-window comparison table ──
print('\n' + '='*60)
print('3.3 — CROSS-WINDOW LOSS-CAUSE COMPARISON')
print('='*60)

header = (f'  {"Window":16s}  {"%BucketA":>8s}  {"%BucketB":>8s}  {"%BucketC":>8s}  '
          f'{"n(RR)":>6s}  {"RR netPnL":>10s}')
print(header)
print(f'  {"-"*16}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*6}  {"-"*10}')

for wld in window_loss_data:
    if wld['n_total'] == 0:
        continue
    print(f'  {wld["window"]:16s}  {wld["stats_a"]["pct"]:>7.1f}%  {wld["stats_b"]["pct"]:>7.1f}%  '
          f'{wld["stats_c"]["pct"]:>7.1f}%  {wld["n_regime_risk"]:>6d}  {wld["rr_net_pnl"]:>10.1f}')

# ── 3.4 Explicit statement on Bucket A dominance ──
print('\n' + '='*60)
print('3.4 — BUCKET A DOMINANCE CHECK')
print('='*60)

a_pcts = [wld['stats_a']['pct'] for wld in window_loss_data if wld['n_total'] > 0]
a_dominant = all(p >= 50 for p in a_pcts)
a_min = min(a_pcts)
a_max = max(a_pcts)
a_range = a_max - a_min

print(f'Bucket A (regime break) share across windows: {a_pcts}')
print(f'Range: {a_min:.1f}% to {a_max:.1f}% (spread: {a_range:.1f}pp)')
if a_dominant:
    print(f'Bucket A is the DOMINANT loss cause (>50%) in ALL 4 windows.')
    if a_range <= 15:
        print('The dominance is CONSISTENT (spread <= 15pp).')
    else:
        print(f'The dominance is present but the share VARIES meaningfully ({a_range:.1f}pp spread).')
else:
    print('Bucket A is NOT dominant in all windows — loss-cause mix shifts.')
print()
if a_dominant and a_range <= 15:
    print('Implication: The earlier conclusion that stop-loss/scale-in tools')
    print('(targeting z-extension, Bucket B) will not help much is REINFORCED —')
    print('regime breaks remain the consistent dominant loss cause across all windows.')
elif a_dominant:
    print('Implication: Regime breaks are always the largest loss cause, but the')
    print('share varies, suggesting the conclusion about stop-loss/scale-in holds')
    print('on average but may be weaker in some market regimes.')
else:
    print('Implication: The earlier conclusion about regime-break dominance does')
    print('NOT hold consistently. Stop-loss/scale-in tools may have more value')
    print('in windows where Bucket B/C share is higher.')


# ══════════════════════════════════════════════════════════════
# PART 4 — LOGGING
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 4: LOGGING')
print('='*70)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
lines = []
lines.append('')
lines.append('')
lines.append('=' * 70)
lines.append(f'MR — WIN%/PNL BACKFILL & CROSS-WINDOW LOSS-TYPE CHECK — {timestamp}')
lines.append('=' * 70)
lines.append('')
lines.append('Config: z>1.5, dur>=3d, 9 instruments, PM L0, TM regime-risk <70%,')
lines.append('100 MYR (4.0 pts) round-trip cost.')

# ── Part 1: Duration robustness backfill ──
lines.append('')
lines.append('')
lines.append('--- PART 1: DURATION ROBUSTNESS TABLE (backfilled with Win%/PnL) ---')
lines.append('')
lines.append(f'  {"Duration":>8s}  {"W1 Win%":>7s}  {"W1 PnL":>8s}  {"W2 Win%":>7s}  {"W2 PnL":>8s}  '
             f'{"W3 Win%":>7s}  {"W3 PnL":>8s}  {"W4 Win%":>7s}  {"W4 PnL":>8s}  '
             f'{"Avg Win%":>8s}  {"Total PnL":>9s}')
lines.append(f'  {"-"*8}  {"-"*7}  {"-"*8}  {"-"*7}  {"-"*8}  '
             f'{"-"*7}  {"-"*8}  {"-"*7}  {"-"*8}  '
             f'{"-"*8}  {"-"*9}')
for dur in DUR_THRESHOLDS:
    d = dur_data[dur]
    avg_win = np.mean([x['win_rate'] for x in d])
    total_pnl = sum(x['total_pnl'] for x in d)
    lines.append(f'  dur>={dur:>2d}d  {d[0]["win_rate"]:>6.1f}%  {d[0]["total_pnl"]:>8.1f}  '
                 f'{d[1]["win_rate"]:>6.1f}%  {d[1]["total_pnl"]:>8.1f}  '
                 f'{d[2]["win_rate"]:>6.1f}%  {d[2]["total_pnl"]:>8.1f}  '
                 f'{d[3]["win_rate"]:>6.1f}%  {d[3]["total_pnl"]:>8.1f}  '
                 f'{avg_win:>7.1f}%  {total_pnl:>9.1f}')

lines.append('')
lines.append('Trades per year at dur>=3d (locked config):')
d3 = dur_data[3]
all_years = sorted(set().union(*[x['yearly_counts'].keys() for x in d3]))
year_header = f'  {"Window":16s}'
for y in all_years:
    year_header += f'  {y:>4d}'
lines.append(year_header)
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for y in all_years:
        row += f'  {d3[i]["yearly_counts"].get(y, 0):>4d}'
    lines.append(row)

# ── Part 2: M5-M6 exclusion backfill ──
lines.append('')
lines.append('')
lines.append('--- PART 2: M5-M6 EXCLUSION TABLE (backfilled with Win%) ---')
lines.append('')
lines.append(f'  {"Window":16s}  {"9i Win%":>7s}  {"8i Win%":>7s}  '
             f'{"9i PnL":>8s}  {"8i PnL":>8s}  {"9i n":>5s}  {"8i n":>5s}')
lines.append(f'  {"-"*16}  {"-"*7}  {"-"*7}  '
             f'{"-"*8}  {"-"*8}  {"-"*5}  {"-"*5}')
for i, w in enumerate(WINDOWS):
    r9 = excl_data['9-inst'][i]
    r8 = excl_data['8-inst'][i]
    lines.append(f'  {w["name"]:16s}  {r9["win_rate"]:>6.1f}%  {r8["win_rate"]:>6.1f}%  '
                 f'{r9["total_pnl"]:>8.1f}  {r8["total_pnl"]:>8.1f}  '
                 f'{r9["n"]:>5d}  {r8["n"]:>5d}')

# ── Part 3: Loss-type breakdown per window ──
lines.append('')
lines.append('')
lines.append('--- PART 3: LOSS-TYPE BREAKDOWN PER WINDOW ---')
lines.append('')
lines.append('Bucket definitions:')
lines.append('  A (Regime break): exit_reason == invalidated')
lines.append('  B (Adverse extension): exit_reason == time_stop AND max|z| > entry|z|')
lines.append('  C (Stalled): exit_reason == time_stop AND max|z| <= entry|z|')
lines.append('  D (Regime-risk exit): exit_reason == regime_risk (reported separately)')
lines.append('  TP-loss: exit_reason == take_profit but net_pnl <= 0 (cost-induced)')

for wld in window_loss_data:
    if wld['n_total'] == 0:
        continue
    lines.append('')
    lines.append(f'  {wld["window"]}:')
    lines.append(f'  Total trades: {wld["n_total"]}, Losses (excl regime-risk): {wld["n_losses"]}')
    lines.append('')
    lines.append(f'    {"Bucket":20s}  {"n":>4s}  {"% losses":>8s}  {"Total PnL":>10s}  {"Avg Loss":>9s}  Top instruments')
    lines.append(f'    {"-"*20}  {"-"*4}  {"-"*8}  {"-"*10}  {"-"*9}  {"-"*25}')
    for s in [wld['stats_a'], wld['stats_b'], wld['stats_c'], wld['stats_tp']]:
        lines.append(f'    {s["name"]:20s}  {s["n"]:>4d}  {s["pct"]:>7.1f}%  {s["total_pnl"]:>10.1f}  {s["avg_loss"]:>9.2f}  {s["top_instruments"]}')
    lines.append('')
    lines.append(f'    Regime-risk exits (Bucket D): n={wld["n_regime_risk"]}, '
                 f'net PnL={wld["rr_net_pnl"]:.1f}, wins={wld["rr_wins"]}, losses={wld["rr_losses_n"]}')
    yearly_str = ', '.join(f'{y}: {int(n)}' for y, n in sorted(wld['yearly'].items()))
    lines.append(f'    Trades per year: {{{yearly_str}}}')

# ── 3.3 Cross-window comparison ──
lines.append('')
lines.append('')
lines.append('--- 3.3: CROSS-WINDOW LOSS-CAUSE COMPARISON ---')
lines.append('')
lines.append(f'  {"Window":16s}  {"%BucketA":>8s}  {"%BucketB":>8s}  {"%BucketC":>8s}  '
             f'{"n(RR)":>6s}  {"RR netPnL":>10s}')
lines.append(f'  {"-"*16}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*6}  {"-"*10}')
for wld in window_loss_data:
    if wld['n_total'] == 0:
        continue
    lines.append(f'  {wld["window"]:16s}  {wld["stats_a"]["pct"]:>7.1f}%  {wld["stats_b"]["pct"]:>7.1f}%  '
                 f'{wld["stats_c"]["pct"]:>7.1f}%  {wld["n_regime_risk"]:>6d}  {wld["rr_net_pnl"]:>10.1f}')

# ── 3.4 Explicit statement ──
lines.append('')
lines.append('')
lines.append('--- 3.4: BUCKET A DOMINANCE CHECK ---')
lines.append('')
lines.append(f'Bucket A (regime break) share across windows: {a_pcts}')
lines.append(f'Range: {a_min:.1f}% to {a_max:.1f}% (spread: {a_range:.1f}pp)')
if a_dominant:
    lines.append(f'Bucket A is the DOMINANT loss cause (>50%) in ALL 4 windows.')
    if a_range <= 15:
        lines.append('The dominance is CONSISTENT (spread <= 15pp).')
    else:
        lines.append(f'The dominance is present but the share VARIES meaningfully ({a_range:.1f}pp spread).')
else:
    lines.append('Bucket A is NOT dominant in all windows — loss-cause mix shifts.')

lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
