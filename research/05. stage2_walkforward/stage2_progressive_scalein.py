"""
MR — Progressive Scale-In (2-Lot Base + Tranches)
===================================================
Part 1: 2-lot base at entry, +1 at |z|>=1.75, +1 at |z|>=2.0, +1 at |z|>=2.25
         (max 5 lots). All lots exit together. Cost = 4.0 pts per lot.
Part 2: Run across all 4 windows with tranche distribution + P&L attribution.
Part 3: Comparison vs. current locked config (1-lot, no pyramid).
         If later tranches net-negative, test 3-lot cap variant.
Part 4: Logging.

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

# Tranche thresholds: entry is 2 lots at |z|>1.5, then add 1 lot at each level
TRANCHE_LEVELS = [1.75, 2.0, 2.25]  # |z| thresholds for tranches 2, 3, 4

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
# BACKTEST ENGINE — BASELINE (1 lot, no pyramid)
# ══════════════════════════════════════════════════════════════

def run_backtest_baseline(df, instruments, dur_thresh, z_entry, test_start, test_end):
    """Baseline backtest: 1 lot, no pyramiding. TM RR<50%."""
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
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
                            'n_lots': 1,
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
    return pd.DataFrame(all_trades)


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE — PROGRESSIVE SCALE-IN
# ══════════════════════════════════════════════════════════════

def run_backtest_progressive(df, instruments, dur_thresh, z_entry, test_start, test_end,
                              tranche_levels, max_lots=5):
    """
    Progressive scale-in backtest.
    - Entry: 2 lots at |z|>1.5
    - Tranche 2: +1 lot at |z|>=1.75 (3 total)
    - Tranche 3: +1 lot at |z|>=2.0 (4 total)
    - Tranche 4: +1 lot at |z|>=2.25 (5 total)
    Each tranche fires once per trade. All lots exit together.
    Cost = ROUNDTRIP_COST_POINTS per lot.
    """
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    all_trades = []
    for inst in instruments:
        z_col = f'{inst}_z'
        position_open = False
        entry_date = entry_spread = entry_z = entry_shape = entry_direction = None
        days_held = 0
        lots = 0
        # Track each tranche's entry spread for P&L attribution
        # tranche_spreads[i] = spread at which tranche i was added
        # tranche_spreads[0] = entry spread (2 lots), [1] = tranche 2 spread, etc.
        tranche_spreads = []
        tranche_lots = []  # lots added at each tranche
        tranche_fired = []  # which tranche levels have fired

        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
            if pd.isna(spread):
                continue
            if position_open:
                days_held += 1

                # Scale-in checks BEFORE exit checks (each tranche fires once)
                if not pd.isna(z) and shape == entry_shape:
                    for ti, t_level in enumerate(tranche_levels):
                        if tranche_fired[ti]:
                            continue
                        if lots >= max_lots:
                            break
                        # Direction-aware: z must extend further in the same direction
                        if entry_direction == -1 and z >= t_level:
                            tranche_fired[ti] = True
                            tranche_spreads.append(spread)
                            tranche_lots.append(1)
                            lots += 1
                        elif entry_direction == 1 and z <= -t_level:
                            tranche_fired[ti] = True
                            tranche_spreads.append(spread)
                            tranche_lots.append(1)
                            lots += 1

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
                    # Compute per-tranche PnL
                    # Tranche 0 = base (2 lots)
                    pnl_per_tranche = []
                    for ti in range(len(tranche_spreads)):
                        t_lots = tranche_lots[ti]
                        gross = (spread - tranche_spreads[ti]) * entry_direction * t_lots
                        cost = ROUNDTRIP_COST_POINTS * t_lots
                        net = gross - cost
                        pnl_per_tranche.append({
                            'lots': t_lots,
                            'entry_spread': tranche_spreads[ti],
                            'gross': round(gross, 2),
                            'cost': round(cost, 2),
                            'net': round(net, 2),
                        })
                    total_gross = sum(p['gross'] for p in pnl_per_tranche)
                    total_cost = ROUNDTRIP_COST_POINTS * lots
                    total_net = total_gross - total_cost

                    if ts <= date <= te:
                        trade_rec = {
                            'instrument': inst,
                            'entry_date': entry_date, 'exit_date': date,
                            'entry_spread': round(entry_spread, 2),
                            'exit_spread': round(spread, 2),
                            'entry_z': round(entry_z, 3),
                            'direction': 'LONG' if entry_direction == 1 else 'SHORT',
                            'days_held': days_held,
                            'exit_reason': exit_reason,
                            'total_gross': round(total_gross, 2),
                            'net_pnl': round(total_net, 2),
                            'n_lots': lots,
                            'n_tranches': len(tranche_spreads),
                        }
                        # Per-tranche PnL for attribution
                        # t0 = base (2 lots), t1 = tranche at 1.75, t2 at 2.0, t3 at 2.25
                        for ti, tp in enumerate(pnl_per_tranche):
                            trade_rec[f't{ti}_lots'] = tp['lots']
                            trade_rec[f't{ti}_spread'] = tp['entry_spread']
                            trade_rec[f't{ti}_net'] = tp['net']
                        all_trades.append(trade_rec)

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
                    lots = 2  # Base position = 2 lots
                    tranche_spreads = [spread]  # t0 = entry
                    tranche_lots = [2]  # t0 = 2 lots
                    tranche_fired = [False] * len(tranche_levels)

    return pd.DataFrame(all_trades)


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE — CAPPED PROGRESSIVE (3-lot max)
# ══════════════════════════════════════════════════════════════

def run_backtest_progressive_capped(df, instruments, dur_thresh, z_entry, test_start, test_end,
                                     cap_lots=3):
    """2-lot base + only 1 tranche at |z|>=1.75, capped at 3 lots total."""
    return run_backtest_progressive(df, instruments, dur_thresh, z_entry, test_start, test_end,
                                    tranche_levels=[1.75], max_lots=cap_lots)


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


def compute_daily_portfolio_sharpe(trades, test_start, test_end, is_progressive=False):
    """Adjusted Sharpe using mark-to-market daily PnL.
    For progressive sizing, handles multiple tranches with different entry dates/spreads."""
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

        if is_progressive:
            # Reconstruct tranche schedule from trade record
            # We need to replay the trade to know when each tranche was added
            # Since we don't store tranche dates in the trade record,
            # we need to re-derive them from the spread data
            n_tranches = int(t.get('n_tranches', 1))
            tranche_spreads_list = []
            tranche_lots_list = []
            for ti in range(n_tranches):
                ts_key = f't{ti}_spread'
                tl_key = f't{ti}_lots'
                if ts_key in t and not pd.isna(t[ts_key]):
                    tranche_spreads_list.append(t[ts_key])
                    tranche_lots_list.append(int(t[tl_key]))

            # For MTM: we know entry (t0) spread, and subsequent tranche spreads
            # We need to figure out on which day each tranche was added
            # by matching the spread price to the day
            # Tranche 0 is always at entry_dt
            tranche_dates = [entry_dt]
            tranche_prev = [t['entry_spread']]  # prev spread for MTM

            # For subsequent tranches, find the first day their spread matches
            for ti in range(1, len(tranche_spreads_list)):
                t_spread = tranche_spreads_list[ti]
                found = False
                for _, day_row in trade_days.iterrows():
                    cs = day_row[inst]
                    if pd.isna(cs):
                        continue
                    if abs(cs - t_spread) < 1e-6:
                        tranche_dates.append(day_row['date'])
                        tranche_prev.append(t_spread)
                        found = True
                        break
                if not found:
                    # Fallback: use entry_dt (shouldn't happen)
                    tranche_dates.append(entry_dt)
                    tranche_prev.append(t_spread)

            # Now do MTM day by day
            for _, day_row in trade_days.iterrows():
                dt = day_row['date']
                cs = day_row[inst]
                if pd.isna(cs):
                    continue
                if dt not in daily_pnl.index:
                    # Update prev spreads
                    for ti in range(len(tranche_dates)):
                        if dt >= tranche_dates[ti]:
                            tranche_prev[ti] = cs
                    continue

                # For each active tranche, compute MTM
                for ti in range(len(tranche_dates)):
                    t_lots = tranche_lots_list[ti]
                    if dt > tranche_dates[ti]:
                        # Active: MTM this tranche
                        daily_pnl[dt] += (cs - tranche_prev[ti]) * direction * t_lots
                        tranche_prev[ti] = cs
                    elif dt == tranche_dates[ti] and ti > 0:
                        # Tranche added today, set prev
                        tranche_prev[ti] = cs

            # Deduct cost on exit day
            if exit_dt in daily_pnl.index:
                total_lots = int(t['n_lots'])
                daily_pnl[exit_dt] -= ROUNDTRIP_COST_POINTS * total_lots
        else:
            # Simple 1-lot MTM
            prev_spread = t['entry_spread']
            for _, day_row in trade_days.iterrows():
                dt = day_row['date']
                cs = day_row[inst]
                if pd.isna(cs):
                    continue
                if dt in daily_pnl.index:
                    daily_pnl[dt] += (cs - prev_spread) * direction
                    prev_spread = cs
                else:
                    prev_spread = cs
            if exit_dt in daily_pnl.index:
                daily_pnl[exit_dt] -= ROUNDTRIP_COST_POINTS

    daily_std = daily_pnl.std()
    if daily_std > 0:
        return round(daily_pnl.mean() / daily_std * np.sqrt(252), 3)
    return np.nan


def fmt(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return '  nan'
    return f'{s:.3f}'


# ══════════════════════════════════════════════════════════════
# PART 1 & 2 — RUN PROGRESSIVE SCALE-IN ACROSS ALL 4 WINDOWS
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 1-2: PROGRESSIVE SCALE-IN (2-LOT BASE + TRANCHES)')
print('='*70)
print('Entry: 2 lots at |z|>1.5')
print('Tranche 2: +1 lot at |z|>=1.75 (3 total)')
print('Tranche 3: +1 lot at |z|>=2.0 (4 total)')
print('Tranche 4: +1 lot at |z|>=2.25 (5 total, max)')
print('Cost: 4.0 pts per lot round-trip')
print()

# Also run baseline for comparison
baseline_results = []
progressive_results = []

for w in WINDOWS:
    print(f'\n--- {w["name"]} ---')

    # Baseline (1 lot)
    bl_trades = run_backtest_baseline(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                       test_start=w['test_start'], test_end=w['test_end'])
    bl_adj = compute_daily_portfolio_sharpe(bl_trades, w['test_start'], w['test_end'],
                                             is_progressive=False)
    bl_m = compute_metrics(bl_trades, w['test_start'], w['test_end'])
    baseline_results.append({'adj': bl_adj, **bl_m, 'trades': bl_trades})
    print(f'  Baseline (1-lot): adj={fmt(bl_adj)}, n={bl_m["n"]}, win={bl_m["win_rate"]}%, '
          f'PnL={bl_m["total_pnl"]}, DD={bl_m["max_dd"]}')

    # Progressive (2-lot base + tranches)
    pg_trades = run_backtest_progressive(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                          test_start=w['test_start'], test_end=w['test_end'],
                                          tranche_levels=TRANCHE_LEVELS, max_lots=5)
    pg_adj = compute_daily_portfolio_sharpe(pg_trades, w['test_start'], w['test_end'],
                                             is_progressive=True)
    pg_m = compute_metrics(pg_trades, w['test_start'], w['test_end'])
    progressive_results.append({'adj': pg_adj, **pg_m, 'trades': pg_trades})
    print(f'  Progressive:      adj={fmt(pg_adj)}, n={pg_m["n"]}, win={pg_m["win_rate"]}%, '
          f'PnL={pg_m["total_pnl"]}, DD={pg_m["max_dd"]}')

    # Tranche distribution
    if len(pg_trades) > 0:
        n_total = len(pg_trades)
        n_2lots = len(pg_trades[pg_trades['n_lots'] == 2])
        n_3lots = len(pg_trades[pg_trades['n_lots'] == 3])
        n_4lots = len(pg_trades[pg_trades['n_lots'] == 4])
        n_5lots = len(pg_trades[pg_trades['n_lots'] == 5])
        avg_lots = pg_trades['n_lots'].mean()
        print(f'  Tranche distribution:')
        print(f'    2 lots only: {n_2lots}/{n_total} ({n_2lots/n_total*100:.1f}%)')
        print(f'    3 lots:      {n_3lots}/{n_total} ({n_3lots/n_total*100:.1f}%)')
        print(f'    4 lots:      {n_4lots}/{n_total} ({n_4lots/n_total*100:.1f}%)')
        print(f'    5 lots:      {n_5lots}/{n_total} ({n_5lots/n_total*100:.1f}%)')
        print(f'    Avg lots:    {avg_lots:.2f}')


# ══════════════════════════════════════════════════════════════
# PART 2.3 — PER-TRANCHE P&L ATTRIBUTION
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 2.3: PER-TRANCHE P&L ATTRIBUTION')
print('='*70)

# For trades that extended beyond 2 lots, compute marginal contribution of each added tranche
for wi, w in enumerate(WINDOWS):
    pg_trades = progressive_results[wi]['trades']
    print(f'\n--- {w["name"]} ---')
    if len(pg_trades) == 0:
        print('  No trades.')
        continue

    # Base (t0) = 2 lots: always present
    t0_net = pg_trades['t0_net'].sum() if 't0_net' in pg_trades.columns else 0
    print(f'  Base (2 lots, t0): total net = {t0_net:.1f}')

    # Tranche 2 (t1) = +1 lot at |z|>=1.75
    extended = pg_trades[pg_trades['n_lots'] >= 3]
    if len(extended) > 0 and 't1_net' in pg_trades.columns:
        t1_vals = extended['t1_net']
        t1_total = t1_vals.sum()
        t1_wins = (t1_vals > 0).sum()
        t1_losses = (t1_vals <= 0).sum()
        print(f'  Tranche 2 (+1 at 1.75, t1): n={len(extended)}, net={t1_total:.1f}, '
              f'wins={t1_wins}, losses={t1_losses}')
    else:
        print(f'  Tranche 2 (+1 at 1.75, t1): n=0')

    # Tranche 3 (t2) = +1 lot at |z|>=2.0
    extended_4 = pg_trades[pg_trades['n_lots'] >= 4]
    if len(extended_4) > 0 and 't2_net' in pg_trades.columns:
        t2_vals = extended_4['t2_net']
        t2_total = t2_vals.sum()
        t2_wins = (t2_vals > 0).sum()
        t2_losses = (t2_vals <= 0).sum()
        print(f'  Tranche 3 (+1 at 2.0,  t2): n={len(extended_4)}, net={t2_total:.1f}, '
              f'wins={t2_wins}, losses={t2_losses}')
    else:
        print(f'  Tranche 3 (+1 at 2.0,  t2): n=0')

    # Tranche 4 (t3) = +1 lot at |z|>=2.25
    extended_5 = pg_trades[pg_trades['n_lots'] >= 5]
    if len(extended_5) > 0 and 't3_net' in pg_trades.columns:
        t3_vals = extended_5['t3_net']
        t3_total = t3_vals.sum()
        t3_wins = (t3_vals > 0).sum()
        t3_losses = (t3_vals <= 0).sum()
        print(f'  Tranche 4 (+1 at 2.25, t3): n={len(extended_5)}, net={t3_total:.1f}, '
              f'wins={t3_wins}, losses={t3_losses}')
    else:
        print(f'  Tranche 4 (+1 at 2.25, t3): n=0')


# ══════════════════════════════════════════════════════════════
# PART 3 — COMPARISON TABLE
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 3: COMPARISON — CURRENT LOCKED vs PROGRESSIVE')
print('='*70)

print(f'\n  {"Window":16s}  {"1-Lot AdjSh":>11s}  {"Prog AdjSh":>11s}  {"Delta":>7s}  '
      f'{"1-Lot PnL":>10s}  {"Prog PnL":>10s}')
print(f'  {"-"*16}  {"-"*11}  {"-"*11}  {"-"*7}  {"-"*10}  {"-"*10}')
for i, w in enumerate(WINDOWS):
    bl_adj = baseline_results[i]['adj']
    pg_adj = progressive_results[i]['adj']
    delta = pg_adj - bl_adj
    bl_pnl = baseline_results[i]['total_pnl']
    pg_pnl = progressive_results[i]['total_pnl']
    print(f'  {w["name"]:16s}  {fmt(bl_adj):>11s}  {fmt(pg_adj):>11s}  {delta:>+7.3f}  '
          f'{bl_pnl:>10.1f}  {pg_pnl:>10.1f}')

baseline_worst = min(baseline_results[i]['adj'] for i in range(4))
prog_worst = min(progressive_results[i]['adj'] for i in range(4))
baseline_avg = np.mean([baseline_results[i]['adj'] for i in range(4)])
prog_avg = np.mean([progressive_results[i]['adj'] for i in range(4)])

print(f'\n  Worst-case adj Sharpe: Baseline={fmt(baseline_worst)}, Progressive={fmt(prog_worst)}')
print(f'  Average adj Sharpe:   Baseline={fmt(baseline_avg)}, Progressive={fmt(prog_avg)}')

if prog_worst > baseline_worst:
    print(f'  → Progressive IMPROVES worst-case by {prog_worst - baseline_worst:+.3f}')
elif prog_worst < baseline_worst:
    print(f'  → Progressive DEGRADES worst-case by {prog_worst - baseline_worst:+.3f}')
else:
    print(f'  → Progressive MATCHES worst-case')


# ══════════════════════════════════════════════════════════════
# PART 3.3 — CAPPED-AT-3-LOTS VARIANT (if later tranches net-negative)
# ══════════════════════════════════════════════════════════════

# Check if later tranches (t2, t3 = |z|>=2.0, 2.25) are net-negative overall
later_tranche_net = 0.0
for wi in range(4):
    pg_trades = progressive_results[wi]['trades']
    if len(pg_trades) == 0:
        continue
    ext4 = pg_trades[pg_trades['n_lots'] >= 4]
    if len(ext4) > 0 and 't2_net' in pg_trades.columns:
        later_tranche_net += ext4['t2_net'].sum()
    ext5 = pg_trades[pg_trades['n_lots'] >= 5]
    if len(ext5) > 0 and 't3_net' in pg_trades.columns:
        later_tranche_net += ext5['t3_net'].sum()

print(f'\n  Later tranches (t2+t3, |z|>=2.0/2.25) combined net: {later_tranche_net:.1f}')

run_capped = later_tranche_net <= 0
if run_capped:
    print('  → Later tranches are net-negative. Testing 3-lot cap variant...')
else:
    print('  → Later tranches are net-positive. Testing 3-lot cap variant anyway for completeness...')

# Always run capped variant for completeness
print('\n--- CAPPED VARIANT: 2-lot base + 1 tranche at |z|>=1.75 (max 3 lots) ---')
capped_results = []
for w in WINDOWS:
    cp_trades = run_backtest_progressive_capped(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                                                  test_start=w['test_start'], test_end=w['test_end'],
                                                  cap_lots=3)
    cp_adj = compute_daily_portfolio_sharpe(cp_trades, w['test_start'], w['test_end'],
                                              is_progressive=True)
    cp_m = compute_metrics(cp_trades, w['test_start'], w['test_end'])
    capped_results.append({'adj': cp_adj, **cp_m, 'trades': cp_trades})
    print(f'  {w["name"]}: adj={fmt(cp_adj)}, n={cp_m["n"]}, win={cp_m["win_rate"]}%, '
          f'PnL={cp_m["total_pnl"]}, DD={cp_m["max_dd"]}')

    # Tranche distribution
    if len(cp_trades) > 0:
        n_t = len(cp_trades)
        n_2 = len(cp_trades[cp_trades['n_lots'] == 2])
        n_3 = len(cp_trades[cp_trades['n_lots'] == 3])
        print(f'    2 lots only: {n_2}/{n_t} ({n_2/n_t*100:.1f}%), 3 lots: {n_3}/{n_t} ({n_3/n_t*100:.1f}%), '
              f'avg lots: {cp_trades["n_lots"].mean():.2f}')

capped_worst = min(capped_results[i]['adj'] for i in range(4))
capped_avg = np.mean([capped_results[i]['adj'] for i in range(4)])

print(f'\n--- FINAL 3-WAY COMPARISON ---')
print(f'\n  {"Window":16s}  {"1-Lot":>8s}  {"Prog(5)":>8s}  {"Cap(3)":>8s}')
print(f'  {"-"*16}  {"-"*8}  {"-"*8}  {"-"*8}')
for i, w in enumerate(WINDOWS):
    print(f'  {w["name"]:16s}  {fmt(baseline_results[i]["adj"]):>8s}  '
          f'{fmt(progressive_results[i]["adj"]):>8s}  {fmt(capped_results[i]["adj"]):>8s}')
print(f'  {"Worst-case":16s}  {fmt(baseline_worst):>8s}  {fmt(prog_worst):>8s}  {fmt(capped_worst):>8s}')
print(f'  {"Average":16s}  {fmt(baseline_avg):>8s}  {fmt(prog_avg):>8s}  {fmt(capped_avg):>8s}')

print(f'\n  Worst-case verdict:')
print(f'    Baseline (1-lot):        {fmt(baseline_worst)}')
print(f'    Progressive (5-lot max): {fmt(prog_worst)} ({prog_worst - baseline_worst:+.3f})')
print(f'    Capped (3-lot max):      {fmt(capped_worst)} ({capped_worst - baseline_worst:+.3f})')


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
lines.append(f'MR — PROGRESSIVE SCALE-IN (2-LOT BASE + TRANCHES) — {timestamp}')
lines.append('=' * 70)
lines.append('')
lines.append('Base config: z>1.5, dur>=3d, 9 instruments, PM L0, TM RR<50%,')
lines.append('100 MYR (4.0 pts) round-trip cost per lot.')
lines.append('')
lines.append('Sizing structure:')
lines.append('  Entry:     2 lots at |z| > 1.5')
lines.append('  Tranche 2: +1 lot at |z| >= 1.75 (3 total)')
lines.append('  Tranche 3: +1 lot at |z| >= 2.0  (4 total)')
lines.append('  Tranche 4: +1 lot at |z| >= 2.25 (5 total, max)')
lines.append('  All lots exit together on first exit condition.')
lines.append('')

# Part 2: Full results per window
lines.append('')
lines.append('--- PART 2: FULL RESULTS PER WINDOW ---')
lines.append('')
for wi, w in enumerate(WINDOWS):
    pg = progressive_results[wi]
    pg_trades = pg['trades']
    lines.append(f'{w["name"]}:')
    lines.append(f'  Adj Sharpe: {fmt(pg["adj"])}, n={pg["n"]}, win={pg["win_rate"]}%, '
                 f'avg_win={pg["avg_win"]}, avg_loss={pg["avg_loss"]}')
    lines.append(f'  Total PnL: {pg["total_pnl"]:.1f}, Max DD: {pg["max_dd"]:.1f}')

    if len(pg_trades) > 0:
        n_t = len(pg_trades)
        n_2 = len(pg_trades[pg_trades['n_lots'] == 2])
        n_3 = len(pg_trades[pg_trades['n_lots'] == 3])
        n_4 = len(pg_trades[pg_trades['n_lots'] == 4])
        n_5 = len(pg_trades[pg_trades['n_lots'] == 5])
        avg_l = pg_trades['n_lots'].mean()
        lines.append(f'  Tranche distribution:')
        lines.append(f'    2 lots only: {n_2}/{n_t} ({n_2/n_t*100:.1f}%)')
        lines.append(f'    3 lots:      {n_3}/{n_t} ({n_3/n_t*100:.1f}%)')
        lines.append(f'    4 lots:      {n_4}/{n_t} ({n_4/n_t*100:.1f}%)')
        lines.append(f'    5 lots:      {n_5}/{n_t} ({n_5/n_t*100:.1f}%)')
        lines.append(f'    Avg lots:    {avg_l:.2f}')
    lines.append('')

# Part 2.3: Per-tranche P&L attribution
lines.append('')
lines.append('--- PER-TRANCHE P&L ATTRIBUTION (trades that extended beyond base) ---')
lines.append('')
for wi, w in enumerate(WINDOWS):
    pg_trades = progressive_results[wi]['trades']
    lines.append(f'{w["name"]}:')
    if len(pg_trades) == 0:
        lines.append('  No trades.')
        lines.append('')
        continue

    t0_net = pg_trades['t0_net'].sum() if 't0_net' in pg_trades.columns else 0
    lines.append(f'  Base (2 lots, t0): total net = {t0_net:.1f}')

    ext3 = pg_trades[pg_trades['n_lots'] >= 3]
    if len(ext3) > 0 and 't1_net' in pg_trades.columns:
        t1_vals = ext3['t1_net']
        lines.append(f'  Tranche 2 (+1 at 1.75, t1): n={len(ext3)}, net={t1_vals.sum():.1f}, '
                     f'wins={(t1_vals > 0).sum()}, losses={(t1_vals <= 0).sum()}')
    else:
        lines.append(f'  Tranche 2 (+1 at 1.75, t1): n=0')

    ext4 = pg_trades[pg_trades['n_lots'] >= 4]
    if len(ext4) > 0 and 't2_net' in pg_trades.columns:
        t2_vals = ext4['t2_net']
        lines.append(f'  Tranche 3 (+1 at 2.0,  t2): n={len(ext4)}, net={t2_vals.sum():.1f}, '
                     f'wins={(t2_vals > 0).sum()}, losses={(t2_vals <= 0).sum()}')
    else:
        lines.append(f'  Tranche 3 (+1 at 2.0,  t2): n=0')

    ext5 = pg_trades[pg_trades['n_lots'] >= 5]
    if len(ext5) > 0 and 't3_net' in pg_trades.columns:
        t3_vals = ext5['t3_net']
        lines.append(f'  Tranche 4 (+1 at 2.25, t3): n={len(ext5)}, net={t3_vals.sum():.1f}, '
                     f'wins={(t3_vals > 0).sum()}, losses={(t3_vals <= 0).sum()}')
    else:
        lines.append(f'  Tranche 4 (+1 at 2.25, t3): n=0')
    lines.append('')

lines.append(f'  Later tranches (t2+t3, |z|>=2.0/2.25) combined net: {later_tranche_net:.1f}')
lines.append('')

# Part 3: Comparison table
lines.append('')
lines.append('--- COMPARISON TABLE ---')
lines.append('')
lines.append(f'  {"Window":16s}  {"1-Lot AdjSh":>11s}  {"Prog AdjSh":>11s}  {"Delta":>7s}  '
             f'{"1-Lot PnL":>10s}  {"Prog PnL":>10s}')
lines.append(f'  {"-"*16}  {"-"*11}  {"-"*11}  {"-"*7}  {"-"*10}  {"-"*10}')
for i, w in enumerate(WINDOWS):
    bl_adj = baseline_results[i]['adj']
    pg_adj = progressive_results[i]['adj']
    delta = pg_adj - bl_adj
    bl_pnl = baseline_results[i]['total_pnl']
    pg_pnl = progressive_results[i]['total_pnl']
    lines.append(f'  {w["name"]:16s}  {fmt(bl_adj):>11s}  {fmt(pg_adj):>11s}  {delta:>+7.3f}  '
                 f'{bl_pnl:>10.1f}  {pg_pnl:>10.1f}')

lines.append('')
lines.append(f'  Worst-case: Baseline={fmt(baseline_worst)}, Progressive={fmt(prog_worst)} '
             f'({prog_worst - baseline_worst:+.3f})')
lines.append(f'  Average:    Baseline={fmt(baseline_avg)}, Progressive={fmt(prog_avg)} '
             f'({prog_avg - baseline_avg:+.3f})')

# Capped variant
lines.append('')
lines.append('')
lines.append('--- CAPPED VARIANT: 2-lot base + 1 tranche at |z|>=1.75 (max 3 lots) ---')
lines.append('')
for wi, w in enumerate(WINDOWS):
    cp = capped_results[wi]
    cp_trades = cp['trades']
    lines.append(f'{w["name"]}:')
    lines.append(f'  Adj Sharpe: {fmt(cp["adj"])}, n={cp["n"]}, win={cp["win_rate"]}%, '
                 f'PnL={cp["total_pnl"]:.1f}, DD={cp["max_dd"]:.1f}')
    if len(cp_trades) > 0:
        n_t = len(cp_trades)
        n_2 = len(cp_trades[cp_trades['n_lots'] == 2])
        n_3 = len(cp_trades[cp_trades['n_lots'] == 3])
        lines.append(f'  2 lots: {n_2}/{n_t} ({n_2/n_t*100:.1f}%), '
                     f'3 lots: {n_3}/{n_t} ({n_3/n_t*100:.1f}%), '
                     f'avg: {cp_trades["n_lots"].mean():.2f}')
    lines.append('')

lines.append('')
lines.append('--- FINAL 3-WAY COMPARISON ---')
lines.append('')
lines.append(f'  {"Window":16s}  {"1-Lot":>8s}  {"Prog(5)":>8s}  {"Cap(3)":>8s}')
lines.append(f'  {"-"*16}  {"-"*8}  {"-"*8}  {"-"*8}')
for i, w in enumerate(WINDOWS):
    lines.append(f'  {w["name"]:16s}  {fmt(baseline_results[i]["adj"]):>8s}  '
                 f'{fmt(progressive_results[i]["adj"]):>8s}  {fmt(capped_results[i]["adj"]):>8s}')
lines.append(f'  {"Worst-case":16s}  {fmt(baseline_worst):>8s}  {fmt(prog_worst):>8s}  {fmt(capped_worst):>8s}')
lines.append(f'  {"Average":16s}  {fmt(baseline_avg):>8s}  {fmt(prog_avg):>8s}  {fmt(capped_avg):>8s}')

lines.append('')
lines.append('Worst-case verdict:')
lines.append(f'  Baseline (1-lot):        {fmt(baseline_worst)}')
lines.append(f'  Progressive (5-lot max): {fmt(prog_worst)} ({prog_worst - baseline_worst:+.3f})')
lines.append(f'  Capped (3-lot max):      {fmt(capped_worst)} ({capped_worst - baseline_worst:+.3f})')
lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
