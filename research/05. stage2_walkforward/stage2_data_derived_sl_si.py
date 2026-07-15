"""
MR — Data-Derived Stop-Loss/Scale-In Threshold Discovery
=========================================================
Part 1: Empirical "point of no return" from max|z| bucket analysis
Part 2: Scale-in threshold from same empirical approach
Part 3: Validate data-derived thresholds across 4 windows
Part 4: Logging

Base config: z>1.5, dur>=3d, 9 instruments, PM L0, TM RR<50%, 100 MYR RT cost.
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
TM_THRESH = 0.50

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

WINDOWS = [
    {'name': 'W1 (2019-2020)', 'test_start': '2019-01-01', 'test_end': '2020-12-31'},
    {'name': 'W2 (2021-2022)', 'test_start': '2021-01-01', 'test_end': '2022-12-31'},
    {'name': 'W3 (2023-2024)', 'test_start': '2023-01-01', 'test_end': '2024-12-31'},
    {'name': 'W4 (2025-2026)', 'test_start': '2025-01-01', 'test_end': '2026-12-31'},
]

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
# BACKTEST ENGINE (baseline, no SL — tracks max_abs_z)
# ══════════════════════════════════════════════════════════════

def run_backtest_baseline(df, instruments, dur_thresh, z_entry, test_start, test_end):
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
                if not pd.isna(z):
                    max_abs_z = max(max_abs_z, abs(z))
                exit_reason = None
                if shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < TM_THRESH:
                            exit_reason = 'regime_risk'
                if exit_reason is None and not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if exit_reason is None and shape != entry_shape:
                    exit_reason = 'invalidated'
                if exit_reason is None and days_held >= TIME_STOP_DAYS:
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


def run_backtest_sl(df, instruments, dur_thresh, z_entry, test_start, test_end,
                    stop_loss_z=None):
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
                if not pd.isna(z):
                    max_abs_z = max(max_abs_z, abs(z))
                exit_reason = None
                if stop_loss_z is not None and not pd.isna(z) and abs(z) >= stop_loss_z:
                    exit_reason = 'stop_loss'
                if exit_reason is None and shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < TM_THRESH:
                            exit_reason = 'regime_risk'
                if exit_reason is None and not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if exit_reason is None and shape != entry_shape:
                    exit_reason = 'invalidated'
                if exit_reason is None and days_held >= TIME_STOP_DAYS:
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


def run_backtest_scalein(df, instruments, dur_thresh, z_entry, test_start, test_end,
                         scalein_z=None):
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    all_trades = []
    for inst in instruments:
        z_col = f'{inst}_z'
        position_open = False
        entry_date = entry_spread = entry_z = entry_shape = entry_direction = None
        days_held = 0
        lots = 0
        scalein_date = scalein_spread = scalein_z_val = None
        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
            if pd.isna(spread):
                continue
            if position_open:
                days_held += 1
                if scalein_z is not None and lots == 1 and not pd.isna(z) and shape == entry_shape:
                    if entry_direction == -1 and z >= scalein_z:
                        lots = 2
                        scalein_date = date
                        scalein_spread = spread
                        scalein_z_val = z
                    elif entry_direction == 1 and z <= -scalein_z:
                        lots = 2
                        scalein_date = date
                        scalein_spread = spread
                        scalein_z_val = z
                exit_reason = None
                if shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < TM_THRESH:
                            exit_reason = 'regime_risk'
                if exit_reason is None and not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if exit_reason is None and shape != entry_shape:
                    exit_reason = 'invalidated'
                if exit_reason is None and days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'
                if exit_reason:
                    gross_pnl_1 = (spread - entry_spread) * entry_direction
                    gross_pnl_2 = 0.0
                    if lots == 2:
                        gross_pnl_2 = (spread - scalein_spread) * entry_direction
                    total_gross = gross_pnl_1 + gross_pnl_2
                    total_cost = ROUNDTRIP_COST_POINTS * lots
                    total_net = total_gross - total_cost
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
                            'gross_pnl_lot1': round(gross_pnl_1, 2),
                            'gross_pnl_lot2': round(gross_pnl_2, 2),
                            'net_pnl': round(total_net, 2),
                            'n_lots': lots,
                            'scalein_date': scalein_date if lots == 2 else None,
                            'scalein_spread': round(scalein_spread, 2) if lots == 2 else None,
                        })
                    position_open = False
                    lots = 0
                    scalein_date = scalein_spread = scalein_z_val = None
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
                    lots = 1
                    scalein_date = scalein_spread = scalein_z_val = None
    return pd.DataFrame(all_trades)


def run_backtest_combined(df, instruments, dur_thresh, z_entry, test_start, test_end,
                          stop_loss_z=None, scalein_z=None):
    """Combined SL + SI backtest."""
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    all_trades = []
    for inst in instruments:
        z_col = f'{inst}_z'
        position_open = False
        entry_date = entry_spread = entry_z = entry_shape = entry_direction = None
        days_held = 0
        lots = 0
        scalein_date = scalein_spread = scalein_z_val = None
        max_abs_z = 0.0
        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
            if pd.isna(spread):
                continue
            if position_open:
                days_held += 1
                if not pd.isna(z):
                    max_abs_z = max(max_abs_z, abs(z))
                # Scale-in check before exits
                if scalein_z is not None and lots == 1 and not pd.isna(z) and shape == entry_shape:
                    if entry_direction == -1 and z >= scalein_z:
                        lots = 2; scalein_date = date; scalein_spread = spread; scalein_z_val = z
                    elif entry_direction == 1 and z <= -scalein_z:
                        lots = 2; scalein_date = date; scalein_spread = spread; scalein_z_val = z
                exit_reason = None
                if stop_loss_z is not None and not pd.isna(z) and abs(z) >= stop_loss_z:
                    exit_reason = 'stop_loss'
                if exit_reason is None and shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < TM_THRESH:
                            exit_reason = 'regime_risk'
                if exit_reason is None and not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if exit_reason is None and shape != entry_shape:
                    exit_reason = 'invalidated'
                if exit_reason is None and days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'
                if exit_reason:
                    gross_pnl_1 = (spread - entry_spread) * entry_direction
                    gross_pnl_2 = (spread - scalein_spread) * entry_direction if lots == 2 else 0.0
                    total_net = gross_pnl_1 + gross_pnl_2 - ROUNDTRIP_COST_POINTS * lots
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
                            'net_pnl': round(total_net, 2),
                            'n_lots': lots,
                            'scalein_date': scalein_date if lots == 2 else None,
                            'scalein_spread': round(scalein_spread, 2) if lots == 2 else None,
                        })
                    position_open = False
                    lots = 0; scalein_date = scalein_spread = scalein_z_val = None
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
                    days_held = 0; lots = 1; max_abs_z = abs(z)
                    scalein_date = scalein_spread = scalein_z_val = None
    return pd.DataFrame(all_trades)


def compute_daily_portfolio_sharpe(trades, test_start, test_end, is_scalein=False):
    if len(trades) == 0:
        return np.nan
    ts_dt, te_dt = pd.Timestamp(test_start), pd.Timestamp(test_end)
    window_dates = df[(df['date'] >= ts_dt) & (df['date'] <= te_dt)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=window_dates)
    for _, t in trades.iterrows():
        entry_dt, exit_dt = t['entry_date'], t['exit_date']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']
        trade_days = df[(df['date'] > entry_dt) & (df['date'] <= exit_dt)]
        prev_spread_1 = t['entry_spread']
        n_lots = int(t.get('n_lots', 1)) if is_scalein else 1
        has_lot2 = is_scalein and n_lots == 2 and t.get('scalein_date') is not None
        scalein_dt = t.get('scalein_date') if has_lot2 else None
        prev_spread_2 = t.get('scalein_spread', np.nan) if has_lot2 else np.nan
        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            cs = day_row[inst]
            if pd.isna(cs):
                continue
            if dt in daily_pnl.index:
                daily_pnl[dt] += (cs - prev_spread_1) * direction
                prev_spread_1 = cs
                if has_lot2:
                    if dt > scalein_dt:
                        daily_pnl[dt] += (cs - prev_spread_2) * direction
                        prev_spread_2 = cs
                    elif dt == scalein_dt:
                        prev_spread_2 = cs
            else:
                prev_spread_1 = cs
                if has_lot2 and dt >= scalein_dt:
                    prev_spread_2 = cs
        if exit_dt in daily_pnl.index:
            daily_pnl[exit_dt] -= ROUNDTRIP_COST_POINTS * (n_lots if is_scalein else 1)
    daily_std = daily_pnl.std()
    if daily_std > 0:
        return round(daily_pnl.mean() / daily_std * np.sqrt(252), 3)
    return np.nan


def fmt(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return '  nan'
    return f'{s:.3f}'


# ══════════════════════════════════════════════════════════════
# PART 1 — EMPIRICAL "POINT OF NO RETURN" DISCOVERY
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 1: EMPIRICAL MAX|Z| BUCKET ANALYSIS')
print('='*70)

# Collect ALL trades across all 4 windows (baseline, no SL)
all_trades_list = []
for w in WINDOWS:
    trades = run_backtest_baseline(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                    test_start=w['test_start'], test_end=w['test_end'])
    if len(trades) > 0:
        trades['window'] = w['name']
        all_trades_list.append(trades)

all_trades = pd.concat(all_trades_list, ignore_index=True)
print(f'Total trades across all 4 windows: {len(all_trades)}')

# 1.2 — Bucket by max|z| reached during hold
bins = [(1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, float('inf'))]
bin_labels = ['[1.5-2.0)', '[2.0-2.5)', '[2.5-3.0)', '[3.0-3.5)', '[3.5-4.0)', '[4.0+)']

print(f'\n--- 1.2 MAX|Z| BUCKET TABLE ---')
print(f'  {"Bin":12s}  {"n":>5s}  {"Win%":>6s}  {"AvgPnL":>8s}  {"%TP":>6s}  {"%Inv":>6s}  {"%TS":>6s}  {"%RR":>6s}')
print(f'  {"-"*12}  {"-"*5}  {"-"*6}  {"-"*8}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}')

bucket_stats = []
for (lo, hi), label in zip(bins, bin_labels):
    mask = (all_trades['max_abs_z'] >= lo) & (all_trades['max_abs_z'] < hi)
    bucket = all_trades[mask]
    n = len(bucket)
    if n == 0:
        bucket_stats.append({'label': label, 'n': 0, 'win_rate': 0, 'avg_pnl': 0,
                              'pct_tp': 0, 'pct_inv': 0, 'pct_ts': 0, 'pct_rr': 0})
        print(f'  {label:12s}  {0:>5d}  {"—":>6s}  {"—":>8s}  {"—":>6s}  {"—":>6s}  {"—":>6s}  {"—":>6s}')
        continue
    win_rate = round((bucket['net_pnl'] > 0).sum() / n * 100, 1)
    avg_pnl = round(bucket['net_pnl'].mean(), 2)
    pct_tp = round((bucket['exit_reason'] == 'take_profit').sum() / n * 100, 1)
    pct_inv = round((bucket['exit_reason'] == 'invalidated').sum() / n * 100, 1)
    pct_ts = round((bucket['exit_reason'] == 'time_stop').sum() / n * 100, 1)
    pct_rr = round((bucket['exit_reason'] == 'regime_risk').sum() / n * 100, 1)
    bucket_stats.append({'label': label, 'n': n, 'win_rate': win_rate, 'avg_pnl': avg_pnl,
                          'pct_tp': pct_tp, 'pct_inv': pct_inv, 'pct_ts': pct_ts, 'pct_rr': pct_rr})
    print(f'  {label:12s}  {n:>5d}  {win_rate:>5.1f}%  {avg_pnl:>8.2f}  {pct_tp:>5.1f}%  '
          f'{pct_inv:>5.1f}%  {pct_ts:>5.1f}%  {pct_rr:>5.1f}%')

# Also do a cumulative "reached at least X" view
print(f'\n--- CUMULATIVE: trades that reached max|z| >= threshold ---')
print(f'  {"Threshold":>10s}  {"n":>5s}  {"Win%":>6s}  {"AvgPnL":>8s}  {"%TP":>6s}  {"%Inv":>6s}')
print(f'  {"-"*10}  {"-"*5}  {"-"*6}  {"-"*8}  {"-"*6}  {"-"*6}')

cum_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
candidate_sl = None
cum_stats = []
for thresh in cum_thresholds:
    mask = all_trades['max_abs_z'] >= thresh
    bucket = all_trades[mask]
    n = len(bucket)
    if n == 0:
        cum_stats.append({'thresh': thresh, 'n': 0, 'win_rate': 0, 'avg_pnl': 0})
        continue
    win_rate = round((bucket['net_pnl'] > 0).sum() / n * 100, 1)
    avg_pnl = round(bucket['net_pnl'].mean(), 2)
    pct_tp = round((bucket['exit_reason'] == 'take_profit').sum() / n * 100, 1)
    pct_inv = round((bucket['exit_reason'] == 'invalidated').sum() / n * 100, 1)
    cum_stats.append({'thresh': thresh, 'n': n, 'win_rate': win_rate, 'avg_pnl': avg_pnl,
                       'pct_tp': pct_tp, 'pct_inv': pct_inv})
    marker = ''
    if candidate_sl is None and win_rate < 50:
        candidate_sl = thresh
        marker = ' <-- FIRST <50%'
    print(f'  {thresh:>10.1f}  {n:>5d}  {win_rate:>5.1f}%  {avg_pnl:>8.2f}  {pct_tp:>5.1f}%  {pct_inv:>5.1f}%{marker}')

# 1.3/1.4 — Identify candidate
print(f'\n--- 1.3/1.4 CANDIDATE STOP-LOSS THRESHOLD ---')
if candidate_sl is not None:
    cs = [s for s in cum_stats if s['thresh'] == candidate_sl][0]
    print(f'Trades that reach max|z| >= {candidate_sl} have a win rate of {cs["win_rate"]}%.')
    print(f'This is the empirical point where further stretch stops being a')
    print(f'buying opportunity and starts being a warning sign.')
    print(f'Candidate stop-loss: |z| >= {candidate_sl}')
else:
    print('No cumulative threshold drops below 50% win rate.')
    print('Win rate remains positive at all max|z| levels tested.')
    # Use the threshold with lowest win rate as a "least bad" candidate for testing
    valid_cum = [s for s in cum_stats if s['n'] >= 5]
    if valid_cum:
        worst = min(valid_cum, key=lambda s: s['win_rate'])
        candidate_sl = worst['thresh']
        print(f'Lowest win rate found: {worst["win_rate"]}% at max|z| >= {worst["thresh"]} (n={worst["n"]})')
        print(f'Using |z| >= {candidate_sl} as candidate for validation, but note')
        print(f'win rate never drops below 50% — no empirical "point of no return" exists.')
    else:
        candidate_sl = 4.0  # fallback
        print(f'Insufficient data at high z levels. Using |z| >= 4.0 as fallback.')

# 1.5 — False-positive / true-positive analysis
print(f'\n--- 1.5 FALSE-POSITIVE / TRUE-POSITIVE ANALYSIS (at |z| >= {candidate_sl}) ---')
would_trigger = all_trades[all_trades['max_abs_z'] >= candidate_sl]
n_trigger = len(would_trigger)
n_trigger_loss = (would_trigger['net_pnl'] <= 0).sum()
n_trigger_win = (would_trigger['net_pnl'] > 0).sum()
tp_rate = round(n_trigger_loss / n_trigger * 100, 1) if n_trigger > 0 else 0
fp_rate = round(n_trigger_win / n_trigger * 100, 1) if n_trigger > 0 else 0
print(f'  Trades reaching |z| >= {candidate_sl}: {n_trigger}')
print(f'  Of those, eventually LOST (true positives for SL): {n_trigger_loss} ({tp_rate}%)')
print(f'  Of those, eventually WON (false positives for SL): {n_trigger_win} ({fp_rate}%)')
print(f'  True-positive rate: {tp_rate}%, False-positive rate: {fp_rate}%')
if fp_rate > 40:
    print(f'  WARNING: False-positive rate >{40}% — SL at this level would cut')
    print(f'  too many winners that temporarily stretch before recovering.')


# ══════════════════════════════════════════════════════════════
# PART 2 — SCALE-IN THRESHOLD, SAME EMPIRICAL APPROACH
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 2: SCALE-IN EMPIRICAL ANALYSIS')
print('='*70)

# Among trades that eventually WON, what was their max|z|?
winners = all_trades[all_trades['net_pnl'] > 0].copy()
losers = all_trades[all_trades['net_pnl'] <= 0].copy()

print(f'\n--- 2.1/2.2 AMONG WINNERS: max|z| reached and eventual P&L ---')
print(f'  {"Bin":12s}  {"n_win":>6s}  {"AvgPnL":>8s}  {"MedianPnL":>10s}  {"%ofWins":>8s}')
print(f'  {"-"*12}  {"-"*6}  {"-"*8}  {"-"*10}  {"-"*8}')

n_total_wins = len(winners)
si_bucket_stats = []
for (lo, hi), label in zip(bins, bin_labels):
    mask = (winners['max_abs_z'] >= lo) & (winners['max_abs_z'] < hi)
    bucket = winners[mask]
    n = len(bucket)
    if n == 0:
        si_bucket_stats.append({'label': label, 'n': 0, 'avg_pnl': 0, 'median_pnl': 0})
        continue
    avg_pnl = round(bucket['net_pnl'].mean(), 2)
    median_pnl = round(bucket['net_pnl'].median(), 2)
    pct_of_wins = round(n / n_total_wins * 100, 1)
    si_bucket_stats.append({'label': label, 'n': n, 'avg_pnl': avg_pnl, 'median_pnl': median_pnl})
    print(f'  {label:12s}  {n:>6d}  {avg_pnl:>8.2f}  {median_pnl:>10.2f}  {pct_of_wins:>7.1f}%')

# Cumulative: among winners that reached at least X, what's their avg PnL?
print(f'\n--- CUMULATIVE: among WINNERS that reached max|z| >= threshold ---')
print(f'  {"Threshold":>10s}  {"n_win":>6s}  {"AvgPnL":>8s}  {"n_loss":>6s}  {"WinRate":>8s}')
print(f'  {"-"*10}  {"-"*6}  {"-"*8}  {"-"*6}  {"-"*8}')

candidate_si = None
for thresh in cum_thresholds:
    w_mask = winners['max_abs_z'] >= thresh
    l_mask = losers['max_abs_z'] >= thresh
    n_w = w_mask.sum()
    n_l = l_mask.sum()
    total = n_w + n_l
    if n_w == 0:
        continue
    avg_pnl = round(winners[w_mask]['net_pnl'].mean(), 2)
    win_rate = round(n_w / total * 100, 1) if total > 0 else 0
    marker = ''
    # Best scale-in: highest avg PnL among winners with decent win rate
    print(f'  {thresh:>10.1f}  {n_w:>6d}  {avg_pnl:>8.2f}  {n_l:>6d}  {win_rate:>7.1f}%{marker}')


# ══════════════════════════════════════════════════════════════
# PART 3 — VALIDATE DATA-DERIVED THRESHOLDS ACROSS 4 WINDOWS
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print(f'PART 3: VALIDATE DATA-DERIVED THRESHOLDS')
print('='*70)
print(f'Candidate SL: |z| >= {candidate_sl}')

# 3.1 — Test SL individually
print(f'\n--- 3.1 STOP-LOSS AT |z| >= {candidate_sl} (individual) ---')
sl_individual = {}
for label, sl_z in [('No-SL', None), (f'SL>={candidate_sl}', candidate_sl)]:
    sl_individual[label] = []
    for w in WINDOWS:
        trades = run_backtest_sl(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                 test_start=w['test_start'], test_end=w['test_end'],
                                 stop_loss_z=sl_z)
        adj = compute_daily_portfolio_sharpe(trades, w['test_start'], w['test_end'])
        n = len(trades)
        total_pnl = round(trades['net_pnl'].sum(), 2) if n > 0 else 0
        win_rate = round((trades['net_pnl'] > 0).sum() / n * 100, 1) if n > 0 else 0
        n_sl = len(trades[trades['exit_reason'] == 'stop_loss']) if sl_z and n > 0 else 0
        sl_individual[label].append({'adj': adj, 'n': n, 'total_pnl': total_pnl,
                                      'win_rate': win_rate, 'n_sl': n_sl})
        print(f'  {label:12s}, {w["name"]}: adj={fmt(adj)}, n={n}, win={win_rate}%, '
              f'PnL={total_pnl}' + (f', SL_exits={n_sl}' if sl_z else ''))

# 3.1b — Test SI individually (use same thresholds as prior: 1.75 and 2.0)
# From Part 2, we'll pick whichever showed best empirical justification
# For now, test both
print(f'\n--- 3.1b SCALE-IN (individual, for comparison) ---')
si_individual = {}
for label, si_z in [('No-SI', None), ('SI>=1.75', 1.75), ('SI>=2.0', 2.0)]:
    si_individual[label] = []
    for w in WINDOWS:
        trades = run_backtest_scalein(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                      test_start=w['test_start'], test_end=w['test_end'],
                                      scalein_z=si_z)
        adj = compute_daily_portfolio_sharpe(trades, w['test_start'], w['test_end'],
                                             is_scalein=(si_z is not None))
        n = len(trades)
        total_pnl = round(trades['net_pnl'].sum(), 2) if n > 0 else 0
        win_rate = round((trades['net_pnl'] > 0).sum() / n * 100, 1) if n > 0 else 0
        si_individual[label].append({'adj': adj, 'n': n, 'total_pnl': total_pnl,
                                      'win_rate': win_rate})
        print(f'  {label:8s}, {w["name"]}: adj={fmt(adj)}, n={n}, win={win_rate}%, PnL={total_pnl}')

# 3.2 — Combined SL + SI
print(f'\n--- 3.2 COMBINED SL + SI ---')
combined_results = {}
combos = [
    ('No-SL/No-SI', None, None),
    (f'SL>={candidate_sl} only', candidate_sl, None),
    ('SI>=2.0 only', None, 2.0),
    (f'SL>={candidate_sl}+SI>=2.0', candidate_sl, 2.0),
]

for label, sl_z, si_z in combos:
    combined_results[label] = []
    for w in WINDOWS:
        trades = run_backtest_combined(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                        test_start=w['test_start'], test_end=w['test_end'],
                                        stop_loss_z=sl_z, scalein_z=si_z)
        is_si = si_z is not None
        adj = compute_daily_portfolio_sharpe(trades, w['test_start'], w['test_end'],
                                             is_scalein=is_si)
        n = len(trades)
        total_pnl = round(trades['net_pnl'].sum(), 2) if n > 0 else 0
        win_rate = round((trades['net_pnl'] > 0).sum() / n * 100, 1) if n > 0 else 0
        combined_results[label].append({'adj': adj, 'n': n, 'total_pnl': total_pnl,
                                         'win_rate': win_rate})
        print(f'  {label:22s}, {w["name"]}: adj={fmt(adj)}, n={n}, win={win_rate}%, PnL={total_pnl}')

# 3.3 — Comparison table
print(f'\n--- 3.3 FINAL COMPARISON TABLE ---')
print(f'\nAdjusted Sharpe:')
header = f'  {"Window":16s}'
for label, _, _ in combos:
    header += f'  {label:>22s}'
print(header)
print(f'  {"-"*16}' + f'  {"-"*22}' * len(combos))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for label, _, _ in combos:
        row += f'  {fmt(combined_results[label][i]["adj"]):>22s}'
    print(row)

# Worst-case and average
print(f'\n  {"Metric":16s}', end='')
for label, _, _ in combos:
    print(f'  {label:>22s}', end='')
print()
for metric_name, metric_fn in [('Worst-case', min), ('Average', lambda x: np.mean(x))]:
    print(f'  {metric_name:16s}', end='')
    for label, _, _ in combos:
        vals = [combined_results[label][i]['adj'] for i in range(4)]
        val = metric_fn(vals)
        print(f'  {fmt(val):>22s}', end='')
    print()

# Total PnL
print(f'\nTotal PnL:')
header = f'  {"Window":16s}'
for label, _, _ in combos:
    header += f'  {label:>22s}'
print(header)
print(f'  {"-"*16}' + f'  {"-"*22}' * len(combos))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for label, _, _ in combos:
        row += f'  {combined_results[label][i]["total_pnl"]:>22.1f}'
    print(row)

# 3.4 — Worst-case-first selection
print(f'\n--- 3.4 WORST-CASE-FIRST SELECTION ---')
baseline_worst = min(combined_results['No-SL/No-SI'][i]['adj'] for i in range(4))
baseline_avg = np.mean([combined_results['No-SL/No-SI'][i]['adj'] for i in range(4)])
print(f'Baseline (No-SL/No-SI): worst={fmt(baseline_worst)}, avg={fmt(baseline_avg)}')

for label, _, _ in combos[1:]:
    worst = min(combined_results[label][i]['adj'] for i in range(4))
    avg = np.mean([combined_results[label][i]['adj'] for i in range(4)])
    improves = worst > baseline_worst
    print(f'{label:22s}: worst={fmt(worst)}, avg={fmt(avg)}, '
          f'improves worst-case: {"YES" if improves else "NO"}')

any_improves = any(
    min(combined_results[label][i]['adj'] for i in range(4)) > baseline_worst
    for label, _, _ in combos[1:]
)

if not any_improves:
    print(f'\nNo data-derived threshold (SL or SI, individually or combined) improves')
    print(f'worst-case adjusted Sharpe over the current locked config ({fmt(baseline_worst)}).')
    print(f'Current locked configuration (no SL, no SI) should stand.')
else:
    best_label = max(
        [(label, min(combined_results[label][i]['adj'] for i in range(4))) for label, _, _ in combos[1:]],
        key=lambda x: x[1]
    )
    print(f'\nBest option: {best_label[0]} (worst-case={fmt(best_label[1])})')


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
lines.append(f'MR — DATA-DERIVED STOP-LOSS/SCALE-IN THRESHOLD DISCOVERY — {timestamp}')
lines.append('=' * 70)
lines.append('')
lines.append('Base config: z>1.5, dur>=3d, 9 instruments, PM L0, TM RR<50%,')
lines.append('100 MYR (4.0 pts) round-trip cost.')
lines.append(f'Total trades across all 4 windows: {len(all_trades)}')

# Part 1
lines.append('')
lines.append('')
lines.append('--- PART 1: EMPIRICAL MAX|Z| BUCKET ANALYSIS ---')
lines.append('')
lines.append('Bucket table (by max|z| reached during hold):')
lines.append(f'  {"Bin":12s}  {"n":>5s}  {"Win%":>6s}  {"AvgPnL":>8s}  {"%TP":>6s}  {"%Inv":>6s}  {"%TS":>6s}  {"%RR":>6s}')
lines.append(f'  {"-"*12}  {"-"*5}  {"-"*6}  {"-"*8}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}')
for s in bucket_stats:
    if s['n'] == 0:
        lines.append(f'  {s["label"]:12s}  {0:>5d}  {"—":>6s}  {"—":>8s}  {"—":>6s}  {"—":>6s}  {"—":>6s}  {"—":>6s}')
    else:
        lines.append(f'  {s["label"]:12s}  {s["n"]:>5d}  {s["win_rate"]:>5.1f}%  {s["avg_pnl"]:>8.2f}  '
                     f'{s["pct_tp"]:>5.1f}%  {s["pct_inv"]:>5.1f}%  {s["pct_ts"]:>5.1f}%  {s["pct_rr"]:>5.1f}%')

lines.append('')
lines.append('Cumulative view (trades reaching max|z| >= threshold):')
lines.append(f'  {"Threshold":>10s}  {"n":>5s}  {"Win%":>6s}  {"AvgPnL":>8s}')
lines.append(f'  {"-"*10}  {"-"*5}  {"-"*6}  {"-"*8}')
for s in cum_stats:
    if s['n'] == 0:
        continue
    marker = ' <-- FIRST <50%' if s['thresh'] == candidate_sl and s['win_rate'] < 50 else ''
    lines.append(f'  {s["thresh"]:>10.1f}  {s["n"]:>5d}  {s["win_rate"]:>5.1f}%  {s["avg_pnl"]:>8.2f}{marker}')

lines.append('')
if candidate_sl and any(s['win_rate'] < 50 for s in cum_stats if s['thresh'] == candidate_sl and s['n'] > 0):
    cs = [s for s in cum_stats if s['thresh'] == candidate_sl][0]
    lines.append(f'Candidate SL threshold: |z| >= {candidate_sl} (win rate = {cs["win_rate"]}%)')
else:
    lines.append(f'No threshold drops below 50% win rate. Candidate SL: |z| >= {candidate_sl}')
    lines.append(f'(used lowest-win-rate threshold with n >= 5 for validation)')

lines.append('')
lines.append(f'False-positive / true-positive at |z| >= {candidate_sl}:')
lines.append(f'  Trades reaching threshold: {n_trigger}')
lines.append(f'  True positives (eventually lost): {n_trigger_loss} ({tp_rate}%)')
lines.append(f'  False positives (eventually won): {n_trigger_win} ({fp_rate}%)')

# Part 2
lines.append('')
lines.append('')
lines.append('--- PART 2: SCALE-IN EMPIRICAL ANALYSIS ---')
lines.append('')
lines.append('Among WINNERS, max|z| reached and eventual P&L:')
lines.append(f'  {"Bin":12s}  {"n_win":>6s}  {"AvgPnL":>8s}  {"MedianPnL":>10s}')
lines.append(f'  {"-"*12}  {"-"*6}  {"-"*8}  {"-"*10}')
for s in si_bucket_stats:
    if s['n'] == 0:
        continue
    lines.append(f'  {s["label"]:12s}  {s["n"]:>6d}  {s["avg_pnl"]:>8.2f}  {s["median_pnl"]:>10.2f}')

# Part 3
lines.append('')
lines.append('')
lines.append('--- PART 3: VALIDATION ACROSS 4 WINDOWS ---')
lines.append('')
lines.append('Adjusted Sharpe:')
header = f'  {"Window":16s}'
for label, _, _ in combos:
    header += f'  {label:>22s}'
lines.append(header)
lines.append(f'  {"-"*16}' + f'  {"-"*22}' * len(combos))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for label, _, _ in combos:
        row += f'  {fmt(combined_results[label][i]["adj"]):>22s}'
    lines.append(row)

lines.append('')
lines.append(f'  {"Metric":16s}' + ''.join(f'  {label:>22s}' for label, _, _ in combos))
for metric_name, metric_fn in [('Worst-case', min), ('Average', lambda x: np.mean(x))]:
    vals_str = ''
    for label, _, _ in combos:
        vals = [combined_results[label][i]['adj'] for i in range(4)]
        vals_str += f'  {fmt(metric_fn(vals)):>22s}'
    lines.append(f'  {metric_name:16s}{vals_str}')

lines.append('')
lines.append('Total PnL:')
header = f'  {"Window":16s}'
for label, _, _ in combos:
    header += f'  {label:>22s}'
lines.append(header)
lines.append(f'  {"-"*16}' + f'  {"-"*22}' * len(combos))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for label, _, _ in combos:
        row += f'  {combined_results[label][i]["total_pnl"]:>22.1f}'
    lines.append(row)

lines.append('')
lines.append('Worst-case-first selection:')
lines.append(f'  Baseline (No-SL/No-SI): worst={fmt(baseline_worst)}, avg={fmt(baseline_avg)}')
for label, _, _ in combos[1:]:
    worst = min(combined_results[label][i]['adj'] for i in range(4))
    avg = np.mean([combined_results[label][i]['adj'] for i in range(4)])
    improves = worst > baseline_worst
    lines.append(f'  {label:22s}: worst={fmt(worst)}, avg={fmt(avg)}, '
                 f'improves: {"YES" if improves else "NO"}')

if not any_improves:
    lines.append('')
    lines.append('No data-derived threshold improves worst-case adjusted Sharpe.')
    lines.append('Current locked configuration (no SL, no SI) stands.')

lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
