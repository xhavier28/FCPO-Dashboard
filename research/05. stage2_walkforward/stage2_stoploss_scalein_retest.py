"""
MR — Tighter Stop-Loss/Scale-In Re-Test Across All Windows
===========================================================
Part 1: Stop-loss at |z|>=2.25 and 2.5, all 4 windows
Part 2: Scale-in at |z|>=1.75 and 2.0, all 4 windows (isolated)
Part 3: Combined test if either shows worst-case improvement
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
TM_THRESH = 0.50  # Current locked config: RR<50%

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

SL_THRESHOLDS = [None, 2.25, 2.5]
SI_THRESHOLDS = [None, 1.75, 2.0]

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
# BACKTEST ENGINE — STOP-LOSS VARIANT
# ══════════════════════════════════════════════════════════════

def run_backtest_sl(df, instruments, dur_thresh, z_entry, test_start, test_end,
                    stop_loss_z=None):
    """Backtest with optional stop-loss. TM RR<50% always active."""
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
                # Stop-loss checked FIRST
                if stop_loss_z is not None and not pd.isna(z) and abs(z) >= stop_loss_z:
                    exit_reason = 'stop_loss'
                # TM regime-risk exit
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


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE — SCALE-IN VARIANT
# ══════════════════════════════════════════════════════════════

def run_backtest_scalein(df, instruments, dur_thresh, z_entry, test_start, test_end,
                         scalein_z=None):
    """Backtest with optional scale-in (max 2 lots). TM RR<50% active. No stop-loss."""
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
                # Scale-in check BEFORE exit checks
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
                # TM regime-risk exit
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
                            'scalein_z': round(scalein_z_val, 3) if lots == 2 else None,
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


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(trades, test_start, test_end):
    n = len(trades)
    if n == 0:
        return {'n': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0,
                'total_pnl': 0, 'max_dd': 0}
    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]
    return {
        'n': n,
        'win_rate': round(len(wins) / n * 100, 1),
        'avg_win': round(wins['net_pnl'].mean(), 2) if len(wins) > 0 else 0,
        'avg_loss': round(losses['net_pnl'].mean(), 2) if len(losses) > 0 else 0,
        'total_pnl': round(trades['net_pnl'].sum(), 2),
        'max_dd': round((trades['net_pnl'].cumsum() - trades['net_pnl'].cumsum().cummax()).min(), 2),
    }


def compute_daily_portfolio_sharpe(trades, test_start, test_end, is_scalein=False):
    """Adjusted Sharpe using mark-to-market daily PnL."""
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
                # Lot 1 MTM
                daily_pnl[dt] += (cs - prev_spread_1) * direction
                prev_spread_1 = cs
                # Lot 2 MTM (only after scale-in date)
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
# PART 1 — TIGHTER STOP-LOSS SWEEP
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 1: TIGHTER STOP-LOSS SWEEP (ALL 4 WINDOWS)')
print('='*70)
print('Base: z>1.5, dur>=3d, 9 instruments, PM L0, TM RR<50%')
print(f'Stop-loss thresholds: {SL_THRESHOLDS}')

sl_results = {}  # sl_thresh -> list of per-window dicts

for sl in SL_THRESHOLDS:
    sl_label = f'SL|z|>={sl}' if sl is not None else 'No-SL'
    sl_results[sl] = []
    print(f'\n--- {sl_label} ---')
    for w in WINDOWS:
        trades = run_backtest_sl(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                 test_start=w['test_start'], test_end=w['test_end'],
                                 stop_loss_z=sl)
        adj = compute_daily_portfolio_sharpe(trades, w['test_start'], w['test_end'])
        m = compute_metrics(trades, w['test_start'], w['test_end'])

        # Bucket B analysis: adverse-extension losses
        n_bucket_b = 0
        bucket_b_caught = 0
        if len(trades) > 0 and 'max_abs_z' in trades.columns:
            losses = trades[trades['net_pnl'] <= 0]
            ts_losses = losses[losses['exit_reason'] == 'time_stop']
            bucket_b = ts_losses[ts_losses['max_abs_z'] > ts_losses['entry_z'].abs()]
            n_bucket_b = len(bucket_b)
            if sl is not None:
                # How many stop-loss exits occurred
                sl_exits = len(trades[trades['exit_reason'] == 'stop_loss'])
            else:
                sl_exits = 0

        # Count stop-loss exits
        n_sl_exits = 0
        sl_exit_pnl = 0.0
        if sl is not None and len(trades) > 0:
            sl_trades = trades[trades['exit_reason'] == 'stop_loss']
            n_sl_exits = len(sl_trades)
            sl_exit_pnl = round(sl_trades['net_pnl'].sum(), 2) if n_sl_exits > 0 else 0.0

        sl_results[sl].append({
            'adj': adj, **m,
            'n_bucket_b': n_bucket_b,
            'n_sl_exits': n_sl_exits,
            'sl_exit_pnl': sl_exit_pnl,
        })
        print(f'  {w["name"]}: adj={fmt(adj)}, n={m["n"]}, win={m["win_rate"]}%, '
              f'PnL={m["total_pnl"]}, DD={m["max_dd"]}, '
              f'SL_exits={n_sl_exits}, SL_PnL={sl_exit_pnl}, BucketB={n_bucket_b}')

# ── Bucket B catch-rate via counterfactual ──
# For each window, compare baseline (no-SL) Bucket B trades against
# what happens with SL active
print('\n--- BUCKET B CATCH-RATE ANALYSIS ---')
for wi, w in enumerate(WINDOWS):
    baseline_trades = run_backtest_sl(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                      test_start=w['test_start'], test_end=w['test_end'],
                                      stop_loss_z=None)
    if len(baseline_trades) == 0:
        continue
    losses_base = baseline_trades[baseline_trades['net_pnl'] <= 0]
    ts_losses = losses_base[losses_base['exit_reason'] == 'time_stop']
    bucket_b = ts_losses[ts_losses['max_abs_z'] > ts_losses['entry_z'].abs()]
    n_bb = len(bucket_b)
    print(f'  {w["name"]}: {n_bb} Bucket B losses in baseline')

    for sl in [2.25, 2.5]:
        # Check how many Bucket B trades would have been caught by this SL
        caught = 0
        if n_bb > 0:
            for _, bt in bucket_b.iterrows():
                if bt['max_abs_z'] >= sl:
                    caught += 1
        # Store in results
        sl_results[sl][wi]['bucket_b_caught'] = caught
        sl_results[sl][wi]['bucket_b_total'] = n_bb
        print(f'    SL>={sl}: caught {caught}/{n_bb} Bucket B trades')

# Comparison table
print('\n--- STOP-LOSS COMPARISON TABLE ---')
print(f'\n  {"Window":16s}  {"No-SL":>10s}  {"SL>=2.25":>10s}  {"SL>=2.5":>10s}  '
      f'{"BB(2.25)":>10s}  {"BB(2.5)":>10s}')
print(f'  {"-"*16}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')
for i, w in enumerate(WINDOWS):
    bb_225 = sl_results[2.25][i].get('bucket_b_caught', 0)
    bb_225_tot = sl_results[2.25][i].get('bucket_b_total', 0)
    bb_25 = sl_results[2.5][i].get('bucket_b_caught', 0)
    bb_25_tot = sl_results[2.5][i].get('bucket_b_total', 0)
    print(f'  {w["name"]:16s}  {fmt(sl_results[None][i]["adj"]):>10s}  '
          f'{fmt(sl_results[2.25][i]["adj"]):>10s}  '
          f'{fmt(sl_results[2.5][i]["adj"]):>10s}  '
          f'{bb_225}/{bb_225_tot:>9}  {bb_25}/{bb_25_tot:>9}')

# W1/W3 vs W2/W4 split
print('\n--- 1.4 W1/W3 (adverse-ext present) vs W2/W4 (pure regime-break) ---')
for sl in [None, 2.25, 2.5]:
    sl_label = f'SL>={sl}' if sl is not None else 'No-SL'
    w13_adjs = [sl_results[sl][0]['adj'], sl_results[sl][2]['adj']]
    w24_adjs = [sl_results[sl][1]['adj'], sl_results[sl][3]['adj']]
    w13_pnl = sl_results[sl][0]['total_pnl'] + sl_results[sl][2]['total_pnl']
    w24_pnl = sl_results[sl][1]['total_pnl'] + sl_results[sl][3]['total_pnl']
    print(f'  {sl_label:10s}: W1/W3 adj=[{fmt(w13_adjs[0])}, {fmt(w13_adjs[1])}] PnL={w13_pnl:.0f}  |  '
          f'W2/W4 adj=[{fmt(w24_adjs[0])}, {fmt(w24_adjs[1])}] PnL={w24_pnl:.0f}')

# Delta from baseline
print('\n  Adj Sharpe delta from No-SL baseline:')
for sl in [2.25, 2.5]:
    sl_label = f'SL>={sl}'
    for i, w in enumerate(WINDOWS):
        delta = sl_results[sl][i]['adj'] - sl_results[None][i]['adj']
        print(f'    {sl_label}, {w["name"]}: {delta:+.3f}')


# ══════════════════════════════════════════════════════════════
# PART 2 — TIGHTER SCALE-IN SWEEP
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 2: TIGHTER SCALE-IN SWEEP (ALL 4 WINDOWS)')
print('='*70)
print('Base: z>1.5, dur>=3d, 9 instruments, PM L0, TM RR<50%')
print(f'Scale-in thresholds: {SI_THRESHOLDS}')

si_results = {}  # si_thresh -> list of per-window dicts

for si in SI_THRESHOLDS:
    si_label = f'SI|z|>={si}' if si is not None else 'No-SI'
    si_results[si] = []
    print(f'\n--- {si_label} ---')
    for w in WINDOWS:
        trades = run_backtest_scalein(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                      test_start=w['test_start'], test_end=w['test_end'],
                                      scalein_z=si)
        adj = compute_daily_portfolio_sharpe(trades, w['test_start'], w['test_end'],
                                             is_scalein=(si is not None))
        m = compute_metrics(trades, w['test_start'], w['test_end'])

        # Scale-in analysis
        n_si_triggered = 0
        si_helped = 0
        si_hurt = 0
        si_marginal_pnl = 0.0
        if si is not None and len(trades) > 0 and 'n_lots' in trades.columns:
            si_trades = trades[trades['n_lots'] == 2]
            n_si_triggered = len(si_trades)
            for _, t in si_trades.iterrows():
                pnl_1lot = t['gross_pnl_lot1'] - ROUNDTRIP_COST_POINTS
                pnl_2lot = t['net_pnl']
                diff = pnl_2lot - pnl_1lot
                if diff > 0:
                    si_helped += 1
                else:
                    si_hurt += 1
                si_marginal_pnl += diff

        si_results[si].append({
            'adj': adj, **m,
            'n_si_triggered': n_si_triggered,
            'si_helped': si_helped,
            'si_hurt': si_hurt,
            'si_marginal_pnl': round(si_marginal_pnl, 2),
        })
        print(f'  {w["name"]}: adj={fmt(adj)}, n={m["n"]}, win={m["win_rate"]}%, '
              f'PnL={m["total_pnl"]}, DD={m["max_dd"]}, '
              f'SI_triggered={n_si_triggered}, helped={si_helped}, hurt={si_hurt}, '
              f'marginal={si_marginal_pnl:.1f}')

# Comparison table
print('\n--- SCALE-IN COMPARISON TABLE ---')
print(f'\n  {"Window":16s}  {"No-SI":>10s}  {"SI>=1.75":>10s}  {"SI>=2.0":>10s}  '
      f'{"Trig(1.75)":>10s}  {"H/H(1.75)":>10s}  '
      f'{"Trig(2.0)":>10s}  {"H/H(2.0)":>10s}')
print(f'  {"-"*16}' + f'  {"-"*10}' * 7)
for i, w in enumerate(WINDOWS):
    r_none = si_results[None][i]
    r_175 = si_results[1.75][i]
    r_20 = si_results[2.0][i]
    print(f'  {w["name"]:16s}  {fmt(r_none["adj"]):>10s}  {fmt(r_175["adj"]):>10s}  {fmt(r_20["adj"]):>10s}  '
          f'{r_175["n_si_triggered"]:>10d}  {r_175["si_helped"]}H/{r_175["si_hurt"]}Hu{" ":>3s}  '
          f'{r_20["n_si_triggered"]:>10d}  {r_20["si_helped"]}H/{r_20["si_hurt"]}Hu')

# W1/W3 vs W2/W4 split
print('\n--- 2.4 W1/W3 (adverse-ext present) vs W2/W4 (pure regime-break) ---')
for si in SI_THRESHOLDS:
    si_label = f'SI>={si}' if si is not None else 'No-SI'
    w13_adjs = [si_results[si][0]['adj'], si_results[si][2]['adj']]
    w24_adjs = [si_results[si][1]['adj'], si_results[si][3]['adj']]
    w13_pnl = si_results[si][0]['total_pnl'] + si_results[si][2]['total_pnl']
    w24_pnl = si_results[si][1]['total_pnl'] + si_results[si][3]['total_pnl']
    print(f'  {si_label:10s}: W1/W3 adj=[{fmt(w13_adjs[0])}, {fmt(w13_adjs[1])}] PnL={w13_pnl:.0f}  |  '
          f'W2/W4 adj=[{fmt(w24_adjs[0])}, {fmt(w24_adjs[1])}] PnL={w24_pnl:.0f}')

# Delta from baseline
print('\n  Adj Sharpe delta from No-SI baseline:')
for si in [1.75, 2.0]:
    si_label = f'SI>={si}'
    for i, w in enumerate(WINDOWS):
        delta = si_results[si][i]['adj'] - si_results[None][i]['adj']
        print(f'    {si_label}, {w["name"]}: {delta:+.3f}')


# ══════════════════════════════════════════════════════════════
# PART 3 — COMBINED TEST (IF EITHER SHOWS PROMISE)
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 3: COMBINED TEST')
print('='*70)

# Check if any SL threshold improves worst-case adj Sharpe
baseline_worst = min(sl_results[None][i]['adj'] for i in range(4))
sl_improves = {}
for sl in [2.25, 2.5]:
    worst = min(sl_results[sl][i]['adj'] for i in range(4))
    sl_improves[sl] = worst > baseline_worst
    print(f'SL>={sl}: worst-case={fmt(worst)}, baseline worst-case={fmt(baseline_worst)}, '
          f'improves: {"YES" if sl_improves[sl] else "NO"}')

# Check if any SI threshold improves worst-case adj Sharpe
si_baseline_worst = min(si_results[None][i]['adj'] for i in range(4))
si_improves = {}
for si in [1.75, 2.0]:
    worst = min(si_results[si][i]['adj'] for i in range(4))
    si_improves[si] = worst > si_baseline_worst
    print(f'SI>={si}: worst-case={fmt(worst)}, baseline worst-case={fmt(si_baseline_worst)}, '
          f'improves: {"YES" if si_improves[si] else "NO"}')

any_sl_helps = any(sl_improves.values())
any_si_helps = any(si_improves.values())

if not any_sl_helps and not any_si_helps:
    print('\nNEITHER stop-loss nor scale-in at any tested threshold improves')
    print('worst-case adjusted Sharpe across the 4 windows.')
    print('Skipping combined test — current locked config is not improved.')
else:
    # Run combined test with whichever helped
    if any_sl_helps:
        best_sl = max([sl for sl in [2.25, 2.5] if sl_improves[sl]],
                      key=lambda sl: min(sl_results[sl][i]['adj'] for i in range(4)))
        print(f'\nBest SL threshold: SL>={best_sl}')
    if any_si_helps:
        best_si = max([si for si in [1.75, 2.0] if si_improves[si]],
                      key=lambda si: min(si_results[si][i]['adj'] for i in range(4)))
        print(f'Best SI threshold: SI>={best_si}')
    print('(Combined test would run here with the best threshold(s))')


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
lines.append(f'MR — TIGHTER STOP-LOSS/SCALE-IN RE-TEST (ALL WINDOWS) — {timestamp}')
lines.append('=' * 70)
lines.append('')
lines.append('Base config: z>1.5, dur>=3d, 9 instruments, PM L0, TM RR<50%,')
lines.append('100 MYR (4.0 pts) round-trip cost.')

# ── Part 1: Stop-loss ──
lines.append('')
lines.append('')
lines.append('--- PART 1: STOP-LOSS SWEEP ---')
lines.append('')
lines.append('Adjusted Sharpe:')
lines.append(f'  {"Window":16s}  {"No-SL":>10s}  {"SL>=2.25":>10s}  {"SL>=2.5":>10s}')
lines.append(f'  {"-"*16}  {"-"*10}  {"-"*10}  {"-"*10}')
for i, w in enumerate(WINDOWS):
    lines.append(f'  {w["name"]:16s}  {fmt(sl_results[None][i]["adj"]):>10s}  '
                 f'{fmt(sl_results[2.25][i]["adj"]):>10s}  '
                 f'{fmt(sl_results[2.5][i]["adj"]):>10s}')

lines.append('')
lines.append('Total PnL:')
lines.append(f'  {"Window":16s}  {"No-SL":>10s}  {"SL>=2.25":>10s}  {"SL>=2.5":>10s}')
lines.append(f'  {"-"*16}  {"-"*10}  {"-"*10}  {"-"*10}')
for i, w in enumerate(WINDOWS):
    lines.append(f'  {w["name"]:16s}  {sl_results[None][i]["total_pnl"]:>10.1f}  '
                 f'{sl_results[2.25][i]["total_pnl"]:>10.1f}  '
                 f'{sl_results[2.5][i]["total_pnl"]:>10.1f}')

lines.append('')
lines.append('Win Rate:')
lines.append(f'  {"Window":16s}  {"No-SL":>10s}  {"SL>=2.25":>10s}  {"SL>=2.5":>10s}')
lines.append(f'  {"-"*16}  {"-"*10}  {"-"*10}  {"-"*10}')
for i, w in enumerate(WINDOWS):
    lines.append(f'  {w["name"]:16s}  {sl_results[None][i]["win_rate"]:>9.1f}%  '
                 f'{sl_results[2.25][i]["win_rate"]:>9.1f}%  '
                 f'{sl_results[2.5][i]["win_rate"]:>9.1f}%')

lines.append('')
lines.append('Stop-Loss Exit Detail:')
for sl in [2.25, 2.5]:
    lines.append(f'  SL>={sl}:')
    lines.append(f'    {"Window":16s}  {"SL exits":>8s}  {"SL PnL":>8s}  {"BB caught":>10s}')
    lines.append(f'    {"-"*16}  {"-"*8}  {"-"*8}  {"-"*10}')
    for i, w in enumerate(WINDOWS):
        r = sl_results[sl][i]
        bb_c = r.get('bucket_b_caught', '-')
        bb_t = r.get('bucket_b_total', '-')
        lines.append(f'    {w["name"]:16s}  {r["n_sl_exits"]:>8d}  {r["sl_exit_pnl"]:>8.1f}  '
                     f'{bb_c}/{bb_t}')
    lines.append('')

lines.append('Adj Sharpe delta from No-SL baseline:')
for sl in [2.25, 2.5]:
    for i, w in enumerate(WINDOWS):
        delta = sl_results[sl][i]['adj'] - sl_results[None][i]['adj']
        lines.append(f'  SL>={sl}, {w["name"]}: {delta:+.3f}')
    lines.append('')

lines.append('W1/W3 (adverse-ext present) vs W2/W4 (pure regime-break):')
for sl in [None, 2.25, 2.5]:
    sl_label = f'SL>={sl}' if sl is not None else 'No-SL'
    w13_adjs = [sl_results[sl][0]['adj'], sl_results[sl][2]['adj']]
    w24_adjs = [sl_results[sl][1]['adj'], sl_results[sl][3]['adj']]
    w13_pnl = sl_results[sl][0]['total_pnl'] + sl_results[sl][2]['total_pnl']
    w24_pnl = sl_results[sl][1]['total_pnl'] + sl_results[sl][3]['total_pnl']
    lines.append(f'  {sl_label:10s}: W1/W3 adj=[{fmt(w13_adjs[0])}, {fmt(w13_adjs[1])}] PnL={w13_pnl:.0f}  |  '
                 f'W2/W4 adj=[{fmt(w24_adjs[0])}, {fmt(w24_adjs[1])}] PnL={w24_pnl:.0f}')

# ── Part 2: Scale-in ──
lines.append('')
lines.append('')
lines.append('--- PART 2: SCALE-IN SWEEP ---')
lines.append('')
lines.append('Adjusted Sharpe:')
lines.append(f'  {"Window":16s}  {"No-SI":>10s}  {"SI>=1.75":>10s}  {"SI>=2.0":>10s}')
lines.append(f'  {"-"*16}  {"-"*10}  {"-"*10}  {"-"*10}')
for i, w in enumerate(WINDOWS):
    lines.append(f'  {w["name"]:16s}  {fmt(si_results[None][i]["adj"]):>10s}  '
                 f'{fmt(si_results[1.75][i]["adj"]):>10s}  '
                 f'{fmt(si_results[2.0][i]["adj"]):>10s}')

lines.append('')
lines.append('Total PnL:')
lines.append(f'  {"Window":16s}  {"No-SI":>10s}  {"SI>=1.75":>10s}  {"SI>=2.0":>10s}')
lines.append(f'  {"-"*16}  {"-"*10}  {"-"*10}  {"-"*10}')
for i, w in enumerate(WINDOWS):
    lines.append(f'  {w["name"]:16s}  {si_results[None][i]["total_pnl"]:>10.1f}  '
                 f'{si_results[1.75][i]["total_pnl"]:>10.1f}  '
                 f'{si_results[2.0][i]["total_pnl"]:>10.1f}')

lines.append('')
lines.append('Win Rate:')
lines.append(f'  {"Window":16s}  {"No-SI":>10s}  {"SI>=1.75":>10s}  {"SI>=2.0":>10s}')
lines.append(f'  {"-"*16}  {"-"*10}  {"-"*10}  {"-"*10}')
for i, w in enumerate(WINDOWS):
    lines.append(f'  {w["name"]:16s}  {si_results[None][i]["win_rate"]:>9.1f}%  '
                 f'{si_results[1.75][i]["win_rate"]:>9.1f}%  '
                 f'{si_results[2.0][i]["win_rate"]:>9.1f}%')

lines.append('')
lines.append('Scale-In Detail:')
for si in [1.75, 2.0]:
    lines.append(f'  SI>={si}:')
    lines.append(f'    {"Window":16s}  {"Triggered":>9s}  {"Helped":>6s}  {"Hurt":>6s}  {"Marginal PnL":>12s}')
    lines.append(f'    {"-"*16}  {"-"*9}  {"-"*6}  {"-"*6}  {"-"*12}')
    for i, w in enumerate(WINDOWS):
        r = si_results[si][i]
        lines.append(f'    {w["name"]:16s}  {r["n_si_triggered"]:>9d}  {r["si_helped"]:>6d}  '
                     f'{r["si_hurt"]:>6d}  {r["si_marginal_pnl"]:>12.1f}')
    lines.append('')

lines.append('Adj Sharpe delta from No-SI baseline:')
for si in [1.75, 2.0]:
    for i, w in enumerate(WINDOWS):
        delta = si_results[si][i]['adj'] - si_results[None][i]['adj']
        lines.append(f'  SI>={si}, {w["name"]}: {delta:+.3f}')
    lines.append('')

lines.append('W1/W3 (adverse-ext present) vs W2/W4 (pure regime-break):')
for si in SI_THRESHOLDS:
    si_label = f'SI>={si}' if si is not None else 'No-SI'
    w13_adjs = [si_results[si][0]['adj'], si_results[si][2]['adj']]
    w24_adjs = [si_results[si][1]['adj'], si_results[si][3]['adj']]
    w13_pnl = si_results[si][0]['total_pnl'] + si_results[si][2]['total_pnl']
    w24_pnl = si_results[si][1]['total_pnl'] + si_results[si][3]['total_pnl']
    lines.append(f'  {si_label:10s}: W1/W3 adj=[{fmt(w13_adjs[0])}, {fmt(w13_adjs[1])}] PnL={w13_pnl:.0f}  |  '
                 f'W2/W4 adj=[{fmt(w24_adjs[0])}, {fmt(w24_adjs[1])}] PnL={w24_pnl:.0f}')

# ── Part 3: Combined test ──
lines.append('')
lines.append('')
lines.append('--- PART 3: COMBINED TEST ---')
lines.append('')
lines.append(f'Baseline worst-case adj Sharpe (No-SL): {fmt(baseline_worst)}')
for sl in [2.25, 2.5]:
    worst = min(sl_results[sl][i]['adj'] for i in range(4))
    lines.append(f'SL>={sl} worst-case: {fmt(worst)}, improves: {"YES" if sl_improves[sl] else "NO"}')
lines.append(f'Baseline worst-case adj Sharpe (No-SI): {fmt(si_baseline_worst)}')
for si in [1.75, 2.0]:
    worst = min(si_results[si][i]['adj'] for i in range(4))
    lines.append(f'SI>={si} worst-case: {fmt(worst)}, improves: {"YES" if si_improves[si] else "NO"}')

if not any_sl_helps and not any_si_helps:
    lines.append('')
    lines.append('Neither stop-loss nor scale-in at any tested threshold improves')
    lines.append('worst-case adjusted Sharpe. Current locked config unchanged.')
else:
    lines.append('')
    if any_sl_helps:
        lines.append(f'Best SL threshold: SL>={best_sl}')
    if any_si_helps:
        lines.append(f'Best SI threshold: SI>={best_si}')

lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
