"""
Stage 1 — Scale-In Trigger Sweep & TM Regime-Risk Exit Overlay
================================================================
Part 1: Scale-in trigger sweep (|z|>=1.75 / 2.00 / 2.25), isolated
Part 2: TM 1-week persistence_prob exit overlay (<70%/<60%/<50%)
Part 3: Log everything to backtest_analysis.txt
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

LOG_FILE = r'C:/ClaudeCode/research/04. backtest_analysis/backtest_analysis.txt'
OOS_START = '2022-01-01'
PM_CONFIDENCE_THRESHOLD = 0.70
Z_ENTRY = 1.5
Z_EXIT = 0.5
TIME_STOP_DAYS = 20
ROUNDTRIP_COST_MYR = 100.0
POINT_VALUE = 25.0
ROUNDTRIP_COST_POINTS = ROUNDTRIP_COST_MYR / POINT_VALUE  # 4.0

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
# DATA SETUP (same as prior Stage 1 scripts)
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
# PRE-COMPUTE TM PERSISTENCE PROBS (cache for speed)
# ══════════════════════════════════════════════════════════════

print('Pre-computing TM 1-week persistence probabilities (OOS only)...')
# Only need OOS dates where positions could be open
oos_dates = df[df['date'] >= pd.Timestamp(OOS_START)]['date'].values
tm_cache = {}  # date -> {current_shape, all_probs}

for dt in oos_dates:
    dt_ts = pd.Timestamp(dt)
    try:
        result = tm_predict(dt_ts, '1w')
        if 'error' not in result:
            current_shape = str(result['current_shape'])
            all_probs = result.get('all_probs', {})
            # persistence_prob = probability that current shape persists
            persistence_prob = all_probs.get(current_shape, np.nan)
            tm_cache[dt_ts] = {
                'current_shape': current_shape,
                'persistence_prob': persistence_prob,
                'all_probs': all_probs,
            }
    except Exception:
        pass

print(f'  Cached {len(tm_cache)} TM predictions')


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINES
# ══════════════════════════════════════════════════════════════

def run_baseline(df, instruments, dur_thresh, z_entry=Z_ENTRY, label=''):
    """Run baseline backtest (no stop-loss, no scale-in). Returns trades_df."""
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
                        'shape_survived': exit_reason != 'invalidated',
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


def run_scalein(df, instruments, dur_thresh, scalein_z, z_entry=Z_ENTRY, label=''):
    """Run backtest with scale-in (max 2 lots, no stop-loss). Returns trades_df."""
    all_trades = []
    for inst in instruments:
        z_col = f'{inst}_z'
        position_open = False
        lots = 0
        entry_date = entry_spread = entry_z = entry_shape = entry_direction = None
        scalein_date = scalein_spread = scalein_z_val = None
        days_held = 0

        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
            if pd.isna(spread):
                continue

            if position_open:
                days_held += 1

                # Check scale-in before exits
                if lots == 1 and not pd.isna(z) and shape == entry_shape:
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
                if not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if shape != entry_shape:
                    exit_reason = 'invalidated'
                if days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'

                if exit_reason:
                    gross_pnl_1 = (spread - entry_spread) * entry_direction
                    gross_pnl_2 = 0.0
                    if lots == 2:
                        gross_pnl_2 = (spread - scalein_spread) * entry_direction
                    total_gross = gross_pnl_1 + gross_pnl_2
                    total_cost = ROUNDTRIP_COST_POINTS * lots
                    total_net = total_gross - total_cost

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
                        'lots': lots,
                        'gross_pnl': round(total_gross, 2),
                        'net_pnl': round(total_net, 2),
                        'gross_pnl_lot1': round(gross_pnl_1, 2),
                        'gross_pnl_lot2': round(gross_pnl_2, 2) if lots == 2 else np.nan,
                        'scalein_date': scalein_date,
                        'scalein_spread': round(scalein_spread, 2) if scalein_spread is not None else np.nan,
                        'scalein_z': round(scalein_z_val, 3) if scalein_z_val is not None else np.nan,
                        'oos': date >= pd.Timestamp(OOS_START),
                        'shape_survived': exit_reason != 'invalidated',
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
                    lots = 1
                    entry_date, entry_spread, entry_z = date, spread, z
                    entry_shape = shape
                    entry_direction = -1 if z > 0 else 1
                    days_held = 0

    return pd.DataFrame(all_trades)


def run_regime_risk(df, instruments, dur_thresh, persist_thresh,
                    z_entry=Z_ENTRY, label=''):
    """Run backtest with regime-risk exit overlay (no stop-loss/scale-in).
    persist_thresh: exit if persistence_prob < this value (e.g. 0.50)."""
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

                # Check all exit conditions — whichever fires first wins
                # Regime-risk: persistence_prob for ENTRY shape drops below threshold
                # Only check if shape hasn't already changed (shape == entry_shape)
                if shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date))
                    if tm_data is not None:
                        # persistence_prob = P(entry_shape persists 1 week from now)
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < persist_thresh:
                            exit_reason = 'regime_risk'

                # Standard exits (can override regime_risk if they also fire)
                if not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if shape != entry_shape:
                    exit_reason = 'invalidated'
                if days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'

                if exit_reason:
                    gross_pnl = (spread - entry_spread) * entry_direction
                    net_pnl = gross_pnl - ROUNDTRIP_COST_POINTS

                    # Record persistence_prob at exit for analysis
                    exit_pp = np.nan
                    tm_data_exit = tm_cache.get(pd.Timestamp(date))
                    if tm_data_exit is not None:
                        exit_pp = tm_data_exit['all_probs'].get(str(entry_shape), np.nan)

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
                        'shape_survived': exit_reason not in ('invalidated',),
                        'entry_shape': entry_shape,
                        'exit_persist_prob': round(exit_pp, 4) if not np.isnan(exit_pp) else np.nan,
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


def compute_metrics(trades, label, df_ref):
    """Compute full metrics including all exit types."""
    n = len(trades)
    if n == 0:
        return {'label': label, 'n_trades': 0, 'win_rate': 0, 'avg_win': 0,
                'avg_loss': 0, 'total_pnl': 0, 'sharpe': np.nan, 'max_dd': 0,
                'pct_take_profit': 0, 'pct_invalidated': 0, 'pct_time_stop': 0,
                'pct_stop_loss': 0, 'pct_regime_risk': 0, 'avg_hp': 0,
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
    sl_pct = round((trades['exit_reason'] == 'stop_loss').mean() * 100, 1)
    rr_pct = round((trades['exit_reason'] == 'regime_risk').mean() * 100, 1)

    avg_hp = round(trades['days_held'].mean(), 1)
    cum_pnl = trades['net_pnl'].cumsum()
    max_dd = round((cum_pnl - cum_pnl.cummax()).min(), 2)
    shape_survival = round(trades['shape_survived'].mean() * 100, 1)

    # Naive Sharpe
    oos_dates = df_ref[df_ref['date'] >= pd.Timestamp(OOS_START)]['date']
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
        'pct_take_profit': tp_pct, 'pct_invalidated': inv_pct,
        'pct_time_stop': ts_pct, 'pct_stop_loss': sl_pct,
        'pct_regime_risk': rr_pct, 'avg_hp': avg_hp,
        'shape_survival': shape_survival,
    }


def compute_daily_portfolio_sharpe(trades, df_ref):
    """Daily portfolio Sharpe using mark-to-market."""
    if len(trades) == 0:
        return np.nan
    oos_dates = df_ref[df_ref['date'] >= pd.Timestamp(OOS_START)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=oos_dates)

    for _, t in trades.iterrows():
        entry_dt = t['entry_date']
        exit_dt = t['exit_date']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']
        n_lots = t.get('lots', 1)

        trade_days = df_ref[(df_ref['date'] > entry_dt) & (df_ref['date'] <= exit_dt)].copy()
        if len(trade_days) == 0:
            continue

        prev_spread_1 = t['entry_spread']
        has_lot2 = n_lots == 2 and not pd.isna(t.get('scalein_date', np.nan))
        scalein_dt = t.get('scalein_date', None) if has_lot2 else None
        prev_spread_2 = t.get('scalein_spread', np.nan) if has_lot2 else np.nan

        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            current_spread = day_row[inst]
            if pd.isna(current_spread):
                continue
            day_mtm = (current_spread - prev_spread_1) * direction
            if dt in daily_pnl.index:
                daily_pnl[dt] += day_mtm
            prev_spread_1 = current_spread

            if has_lot2 and scalein_dt is not None and dt > scalein_dt:
                day_mtm_2 = (current_spread - prev_spread_2) * direction
                if dt in daily_pnl.index:
                    daily_pnl[dt] += day_mtm_2
                prev_spread_2 = current_spread
            elif has_lot2 and scalein_dt is not None and dt == scalein_dt:
                prev_spread_2 = current_spread

        cost = ROUNDTRIP_COST_POINTS * n_lots
        if exit_dt in daily_pnl.index:
            daily_pnl[exit_dt] -= cost

    daily_std = daily_pnl.std()
    if daily_std > 0:
        return round(daily_pnl.mean() / daily_std * np.sqrt(252), 3)
    return np.nan


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


# ══════════════════════════════════════════════════════════════
# PART 1 — SCALE-IN TRIGGER SWEEP
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 1: SCALE-IN TRIGGER SWEEP')
print('='*70)

# Run baseline (no scale-in)
print('Running baseline (no scale-in)...')
baseline_trades = run_baseline(df, ALL_9, dur_thresh=10, label='No SI')
oos_baseline = baseline_trades[baseline_trades['oos']].reset_index(drop=True)
baseline_metrics = compute_metrics(oos_baseline, 'No SI', df)
baseline_adj = compute_daily_portfolio_sharpe(oos_baseline, df)

SI_THRESHOLDS = [1.75, 2.00, 2.25]
si_results = []

for si_z in SI_THRESHOLDS:
    label = f'SI |z|>={si_z}'
    print(f'Running {label}...')
    trades = run_scalein(df, ALL_9, dur_thresh=10, scalein_z=si_z, label=label)
    oos = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()
    metrics = compute_metrics(oos, label, df)
    adj = compute_daily_portfolio_sharpe(oos, df)

    # Scale-in triggered analysis
    si_triggered = oos[oos['lots'] == 2].reset_index(drop=True) if len(oos) > 0 and 'lots' in oos.columns else pd.DataFrame()
    n_triggered = len(si_triggered)

    helped = hurt = 0
    worst_diff = 0.0
    worst_trade = None
    overshoot_vals = []

    if n_triggered > 0:
        for _, t in si_triggered.iterrows():
            pnl_1lot = t['gross_pnl_lot1'] - ROUNDTRIP_COST_POINTS
            pnl_2lot = t['net_pnl']
            diff = pnl_2lot - pnl_1lot
            if diff > 0:
                helped += 1
            else:
                hurt += 1
            if diff < worst_diff:
                worst_diff = diff
                worst_trade = t

            # Overshoot: actual |SIz| - threshold
            if not pd.isna(t['scalein_z']):
                overshoot_vals.append(abs(t['scalein_z']) - si_z)

    avg_overshoot = np.mean(overshoot_vals) if overshoot_vals else np.nan

    si_results.append({
        'si_z': si_z, 'label': label, 'metrics': metrics, 'adj': adj,
        'n_triggered': n_triggered, 'helped': helped, 'hurt': hurt,
        'worst_diff': worst_diff,
        'worst_inst': worst_trade['instrument'] if worst_trade is not None else '-',
        'worst_date': str(worst_trade['entry_date'].date()) if worst_trade is not None else '-',
        'avg_overshoot': avg_overshoot,
        'oos': oos,
    })

    m = metrics
    print(f'  n={m["n_trades"]}, win={m["win_rate"]}%, naive={fmt_sharpe(m["sharpe"])}, '
          f'adj={fmt_sharpe(adj)}, PnL={m["total_pnl"]}, DD={m["max_dd"]}, '
          f'SI_trig={n_triggered}, helped={helped}/{n_triggered}')

# Comparison table
print('\n--- SCALE-IN COMPARISON TABLE ---')
header = (f'  {"Variant":14s} {"n":>4s} {"Win%":>6s} {"AdjShrp":>8s} '
          f'{"TotalPnL":>10s} {"MaxDD":>8s} {"SI_n":>5s} {"Helped":>7s} {"Hurt":>5s} '
          f'{"AvgOS":>6s}')
print(header)

# Baseline row
print(f'  {"No SI":14s} {baseline_metrics["n_trades"]:>4d} {baseline_metrics["win_rate"]:>5.1f}% '
      f'{fmt_sharpe(baseline_adj):>8s} '
      f'{baseline_metrics["total_pnl"]:>10.1f} {baseline_metrics["max_dd"]:>8.1f} '
      f'{"  -":>5s} {"    -":>7s} {"  -":>5s} {"  -":>6s}')

for r in si_results:
    m = r['metrics']
    os_str = f'{r["avg_overshoot"]:.3f}' if not np.isnan(r['avg_overshoot']) else '  n/a'
    print(f'  {r["label"]:14s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% '
          f'{fmt_sharpe(r["adj"]):>8s} '
          f'{m["total_pnl"]:>10.1f} {m["max_dd"]:>8.1f} '
          f'{r["n_triggered"]:>5d} {r["helped"]:>3d}/{r["n_triggered"]:>3d} {r["hurt"]:>5d} '
          f'{os_str:>6s}')

# Worst-case contributor per threshold
print('\n  Worst-case scale-in contributor per threshold:')
for r in si_results:
    print(f'    {r["label"]}: {r["worst_inst"]} ({r["worst_date"]}), '
          f'marginal loss = {r["worst_diff"]:+.2f}')


# ══════════════════════════════════════════════════════════════
# PART 2 — TM REGIME-RISK EXIT OVERLAY
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 2: TM REGIME-RISK EXIT OVERLAY')
print('='*70)

PERSIST_THRESHOLDS = [0.70, 0.60, 0.50]
rr_results = []

# Baseline (no regime-risk exit) — reuse already-computed baseline
rr_results.append({
    'thresh': None, 'label': 'No RR exit',
    'metrics': baseline_metrics, 'adj': baseline_adj,
    'oos': oos_baseline,
})

for pt in PERSIST_THRESHOLDS:
    label = f'RR <{int(pt*100)}%'
    print(f'Running {label}...')
    trades = run_regime_risk(df, ALL_9, dur_thresh=10, persist_thresh=pt, label=label)
    oos = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()
    metrics = compute_metrics(oos, label, df)
    adj = compute_daily_portfolio_sharpe(oos, df)

    rr_results.append({
        'thresh': pt, 'label': label, 'metrics': metrics, 'adj': adj,
        'oos': oos,
    })

    m = metrics
    print(f'  n={m["n_trades"]}, win={m["win_rate"]}%, naive={fmt_sharpe(m["sharpe"])}, '
          f'adj={fmt_sharpe(adj)}, PnL={m["total_pnl"]}, DD={m["max_dd"]}')
    print(f'  TP={m["pct_take_profit"]}%, Inv={m["pct_invalidated"]}%, '
          f'TS={m["pct_time_stop"]}%, RR={m["pct_regime_risk"]}%')

# Comparison table
print('\n--- REGIME-RISK EXIT COMPARISON TABLE ---')
header = (f'  {"Variant":12s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"AdjShrp":>8s} '
          f'{"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} '
          f'{"TP%":>5s} {"Inv%":>5s} {"TS%":>5s} {"RR%":>5s} {"Gate":>22s}')
print(header)
for r in rr_results:
    m = r['metrics']
    g = gate_check(m)
    print(f'  {r["label"]:12s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
          f'{fmt_sharpe(r["adj"]):>8s} '
          f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
          f'{m["max_dd"]:>8.1f} '
          f'{m["pct_take_profit"]:>4.1f}% {m["pct_invalidated"]:>4.1f}% '
          f'{m["pct_time_stop"]:>4.1f}% {m["pct_regime_risk"]:>4.1f}% '
          f'{g:>22s}')

# ── 2.5 — Early-warning lead-time analysis ───────────────────
print('\n--- REGIME-RISK EARLY-WARNING ANALYSIS ---')

# For each regime-risk threshold, find trades where regime_risk fired,
# and compare against what would have happened under baseline rules
for r in rr_results[1:]:  # skip baseline
    pt = r['thresh']
    label = r['label']
    oos_rr = r['oos']

    if len(oos_rr) == 0:
        continue

    rr_exits = oos_rr[oos_rr['exit_reason'] == 'regime_risk'].reset_index(drop=True)
    print(f'\n  {label}: {len(rr_exits)} regime_risk exits')

    if len(rr_exits) == 0:
        print(f'    No regime_risk exits at this threshold.')
        continue

    # For each regime_risk exit, find when the shape ACTUALLY flipped
    # (or if it never flipped within 20d, note that)
    lead_times = []
    pnl_diffs = []
    trade_details = []

    for _, t in rr_exits.iterrows():
        rr_exit_date = t['exit_date']
        entry_date = t['entry_date']
        entry_shape = t.get('entry_shape', None)
        entry_spread = t['entry_spread']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']
        rr_pnl = t['net_pnl']

        # If entry_shape not stored, infer from baseline
        if entry_shape is None:
            entry_row = df[df['date'] == entry_date]
            if len(entry_row) > 0:
                entry_shape = entry_row.iloc[0]['shape']

        # Find when shape actually changes after entry
        future_days = df[(df['date'] > entry_date)].copy()
        actual_flip_date = None
        actual_flip_spread = None
        days_to_flip = None

        for _, fd in future_days.iterrows():
            if fd['shape'] != entry_shape:
                actual_flip_date = fd['date']
                actual_flip_spread = fd[inst]
                break

        # Also find baseline exit (what would have happened without regime_risk)
        # Simulate: continue from rr_exit_date under original rules
        baseline_exit_date = None
        baseline_exit_spread = None
        baseline_exit_reason = None

        # Re-simulate from entry under baseline rules
        sim_days_held = 0
        for _, fd in df[(df['date'] > entry_date)].iterrows():
            sim_days_held += 1
            sim_z = fd[f'{inst}_z']
            sim_spread = fd[inst]
            sim_shape = fd['shape']
            if pd.isna(sim_spread):
                continue

            exit_r = None
            if not pd.isna(sim_z) and abs(sim_z) < Z_EXIT:
                exit_r = 'take_profit'
            if sim_shape != entry_shape:
                exit_r = 'invalidated'
            if sim_days_held >= TIME_STOP_DAYS:
                exit_r = 'time_stop'

            if exit_r:
                baseline_exit_date = fd['date']
                baseline_exit_spread = sim_spread
                baseline_exit_reason = exit_r
                break

        # Compute lead time: days between regime_risk exit and actual flip
        if actual_flip_date is not None:
            # Count trading days between rr_exit_date and actual_flip_date
            between = df[(df['date'] > rr_exit_date) & (df['date'] <= actual_flip_date)]
            lead_time = len(between)
        else:
            lead_time = -1  # shape never flipped (would have been TP or time_stop)

        # P&L comparison: regime_risk exit P&L vs baseline exit P&L
        baseline_pnl = np.nan
        if baseline_exit_spread is not None and not pd.isna(baseline_exit_spread):
            baseline_gross = (baseline_exit_spread - entry_spread) * direction
            baseline_pnl = baseline_gross - ROUNDTRIP_COST_POINTS

        pnl_diff = rr_pnl - baseline_pnl if not np.isnan(baseline_pnl) else np.nan

        lead_times.append(lead_time)
        pnl_diffs.append(pnl_diff)
        trade_details.append({
            'entry_date': entry_date,
            'instrument': inst,
            'direction': t['direction'],
            'rr_exit_date': rr_exit_date,
            'rr_pnl': rr_pnl,
            'baseline_exit_date': baseline_exit_date,
            'baseline_exit_reason': baseline_exit_reason,
            'baseline_pnl': baseline_pnl,
            'pnl_diff': pnl_diff,
            'lead_time': lead_time,
            'actual_flip_date': actual_flip_date,
        })

    # Summary
    lt_arr = np.array(lead_times)
    genuine_early = (lt_arr > 0).sum()
    same_day = (lt_arr == 0).sum()
    no_flip = (lt_arr < 0).sum()

    print(f'    Genuine early exit (>0 days before flip): {genuine_early}')
    print(f'    Same-day as flip (0 days lead):           {same_day}')
    print(f'    Shape never flipped (TP/TS would have):   {no_flip}')
    if genuine_early > 0:
        early_lts = lt_arr[lt_arr > 0]
        print(f'    Avg lead time (genuine early only):       {early_lts.mean():.1f} days')

    # P&L impact
    valid_diffs = [d for d in pnl_diffs if not np.isnan(d)]
    if valid_diffs:
        saved = sum(1 for d in valid_diffs if d > 0)
        hurt = sum(1 for d in valid_diffs if d < 0)
        neutral = sum(1 for d in valid_diffs if d == 0)
        total_diff = sum(valid_diffs)
        print(f'    P&L vs baseline: saved={saved}, hurt={hurt}, neutral={neutral}')
        print(f'    Total P&L difference: {total_diff:+.2f}')

    # Trade-level detail
    print(f'\n    {"Entry":>10s} {"Inst":>12s} {"Dir":>5s} {"RR_Exit":>10s} {"RR_PnL":>8s} '
          f'{"BL_Exit":>10s} {"BL_Reason":>12s} {"BL_PnL":>8s} {"Diff":>8s} {"Lead":>5s}')
    for td in trade_details:
        bl_date = str(td['baseline_exit_date'].date()) if td['baseline_exit_date'] is not None else '       n/a'
        bl_reason = td['baseline_exit_reason'] if td['baseline_exit_reason'] else 'n/a'
        bl_pnl = f'{td["baseline_pnl"]:>+8.2f}' if not np.isnan(td['baseline_pnl']) else '     n/a'
        diff = f'{td["pnl_diff"]:>+8.2f}' if not np.isnan(td['pnl_diff']) else '     n/a'
        lt = f'{td["lead_time"]:>5d}' if td['lead_time'] >= 0 else ' none'
        print(f'    {str(td["entry_date"].date()):>10s} {td["instrument"]:>12s} {td["direction"]:>5s} '
              f'{str(td["rr_exit_date"].date()):>10s} {td["rr_pnl"]:>+8.2f} '
              f'{bl_date:>10s} {bl_reason:>12s} {bl_pnl} {diff} {lt}')

# ── 2.7 — Exit-type reclassification breakdown ───────────────
print('\n--- EXIT-TYPE RECLASSIFICATION ---')
print('How many baseline "invalidated" exits got reclassified as "regime_risk"?')

baseline_inv_count = (oos_baseline['exit_reason'] == 'invalidated').sum()
print(f'\nBaseline invalidated exits: {baseline_inv_count}')

for r in rr_results[1:]:
    pt = r['thresh']
    label = r['label']
    oos_rr = r['oos']
    if len(oos_rr) == 0:
        continue

    new_inv = (oos_rr['exit_reason'] == 'invalidated').sum()
    new_rr = (oos_rr['exit_reason'] == 'regime_risk').sum()
    reclassified = baseline_inv_count - new_inv  # how many shifted from inv to rr
    still_inv = new_inv

    print(f'  {label}: regime_risk={new_rr}, still_invalidated={still_inv}, '
          f'reclassified from inv={reclassified}')
    if baseline_inv_count > 0:
        print(f'    {reclassified}/{baseline_inv_count} ({reclassified/baseline_inv_count*100:.1f}%) '
              f'of former invalidated exits now exit as regime_risk instead')


# ══════════════════════════════════════════════════════════════
# PART 3 — LOGGING
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 3: LOGGING')
print('='*70)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
lines = []
lines.append('')
lines.append('')
lines.append('=' * 70)
lines.append(f'STAGE 1 — SCALE-IN SWEEP & TM REGIME-RISK EXIT OVERLAY — {timestamp}')
lines.append('=' * 70)
lines.append('')
lines.append('Config: z>1.5, PM L0, dur>=10d, 9 instruments, OOS >= 2022-01-01')
lines.append('Cost: 100 MYR (4.0 pts) round-trip')
lines.append('Reference baseline: naive Sharpe 1.812, adjusted 1.546')

# ── Part 1: Scale-In Sweep ───────────────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 1: SCALE-IN TRIGGER SWEEP (no stop-loss) ---')
lines.append('')
lines.append('Scale-in rule: add 2nd lot when |z| crosses threshold in same direction.')
lines.append('Max 2 lots. Both exit together on first exit condition.')
lines.append('')

lines.append(f'  {"Variant":14s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"AdjShrp":>8s} '
             f'{"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} '
             f'{"SI_n":>5s} {"Helped":>7s} {"Hurt":>5s} {"AvgOS":>6s} {"Gate":>22s}')
lines.append(f'  {"-"*14} {"-"*4} {"-"*6} {"-"*7} {"-"*8} '
             f'{"-"*10} {"-"*8} {"-"*8} {"-"*8} '
             f'{"-"*5} {"-"*7} {"-"*5} {"-"*6} {"-"*22}')

# Baseline row
bm = baseline_metrics
bg = gate_check(bm)
lines.append(f'  {"No SI":14s} {bm["n_trades"]:>4d} {bm["win_rate"]:>5.1f}% {fmt_sharpe(bm["sharpe"]):>7s} '
             f'{fmt_sharpe(baseline_adj):>8s} '
             f'{bm["total_pnl"]:>10.1f} {bm["avg_win"]:>8.2f} {bm["avg_loss"]:>8.2f} '
             f'{bm["max_dd"]:>8.1f} '
             f'{"  -":>5s} {"    -":>7s} {"  -":>5s} {"  -":>6s} {bg:>22s}')

for r in si_results:
    m = r['metrics']
    g = gate_check(m)
    os_str = f'{r["avg_overshoot"]:.3f}' if not np.isnan(r['avg_overshoot']) else '  n/a'
    lines.append(f'  {r["label"]:14s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
                 f'{fmt_sharpe(r["adj"]):>8s} '
                 f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
                 f'{m["max_dd"]:>8.1f} '
                 f'{r["n_triggered"]:>5d} {r["helped"]:>3d}/{r["n_triggered"]:>3d} {r["hurt"]:>5d} '
                 f'{os_str:>6s} {g:>22s}')

lines.append('')
lines.append('Worst-case scale-in contributor per threshold:')
for r in si_results:
    lines.append(f'  {r["label"]}: {r["worst_inst"]} ({r["worst_date"]}), '
                 f'marginal loss = {r["worst_diff"]:+.2f}')

lines.append('')
lines.append('Overshoot = actual |SIz| minus threshold (daily-bar granularity).')
lines.append('Lower overshoot = trigger price closer to intended threshold.')

# ── Part 2: TM Regime-Risk Exit ──────────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 2: TM REGIME-RISK EXIT OVERLAY ---')
lines.append('')
lines.append('Exit condition: TM 1-week persistence_prob for entry shape drops')
lines.append('below threshold. Checked alongside TP/Inv/TS — first to fire wins.')
lines.append('No stop-loss or scale-in combined (isolated test).')
lines.append('')

lines.append(f'  {"Variant":12s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"AdjShrp":>8s} '
             f'{"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} '
             f'{"TP%":>5s} {"Inv%":>5s} {"TS%":>5s} {"RR%":>5s} {"Gate":>22s}')
lines.append(f'  {"-"*12} {"-"*4} {"-"*6} {"-"*7} {"-"*8} '
             f'{"-"*10} {"-"*8} {"-"*8} {"-"*8} '
             f'{"-"*5} {"-"*5} {"-"*5} {"-"*5} {"-"*22}')

for r in rr_results:
    m = r['metrics']
    g = gate_check(m)
    lines.append(f'  {r["label"]:12s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
                 f'{fmt_sharpe(r["adj"]):>8s} '
                 f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
                 f'{m["max_dd"]:>8.1f} '
                 f'{m["pct_take_profit"]:>4.1f}% {m["pct_invalidated"]:>4.1f}% '
                 f'{m["pct_time_stop"]:>4.1f}% {m["pct_regime_risk"]:>4.1f}% '
                 f'{g:>22s}')

# Early-warning analysis
lines.append('')
lines.append('')
lines.append('--- REGIME-RISK EARLY-WARNING ANALYSIS (Part 2.5) ---')
lines.append('')
lines.append('For each regime_risk exit: how many days before the shape actually')
lines.append('flipped, and what was the P&L difference vs. holding to baseline exit?')

for r in rr_results[1:]:
    pt = r['thresh']
    label = r['label']
    oos_rr = r['oos']
    if len(oos_rr) == 0:
        continue

    rr_exits = oos_rr[oos_rr['exit_reason'] == 'regime_risk'].reset_index(drop=True)
    lines.append(f'\n  {label}: {len(rr_exits)} regime_risk exits')

    if len(rr_exits) == 0:
        lines.append(f'    No regime_risk exits at this threshold.')
        continue

    lead_times = []
    pnl_diffs = []
    trade_details = []

    for _, t in rr_exits.iterrows():
        rr_exit_date = t['exit_date']
        entry_date = t['entry_date']
        entry_shape = t.get('entry_shape', None)
        entry_spread = t['entry_spread']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']
        rr_pnl = t['net_pnl']

        if entry_shape is None:
            entry_row = df[df['date'] == entry_date]
            if len(entry_row) > 0:
                entry_shape = entry_row.iloc[0]['shape']

        future_days = df[(df['date'] > entry_date)].copy()
        actual_flip_date = None
        for _, fd in future_days.iterrows():
            if fd['shape'] != entry_shape:
                actual_flip_date = fd['date']
                break

        # Baseline simulation
        sim_days_held = 0
        baseline_exit_date = None
        baseline_exit_spread = None
        baseline_exit_reason = None
        for _, fd in df[(df['date'] > entry_date)].iterrows():
            sim_days_held += 1
            sim_z = fd[f'{inst}_z']
            sim_spread = fd[inst]
            sim_shape = fd['shape']
            if pd.isna(sim_spread):
                continue
            exit_r = None
            if not pd.isna(sim_z) and abs(sim_z) < Z_EXIT:
                exit_r = 'take_profit'
            if sim_shape != entry_shape:
                exit_r = 'invalidated'
            if sim_days_held >= TIME_STOP_DAYS:
                exit_r = 'time_stop'
            if exit_r:
                baseline_exit_date = fd['date']
                baseline_exit_spread = sim_spread
                baseline_exit_reason = exit_r
                break

        if actual_flip_date is not None:
            between = df[(df['date'] > rr_exit_date) & (df['date'] <= actual_flip_date)]
            lead_time = len(between)
        else:
            lead_time = -1

        baseline_pnl = np.nan
        if baseline_exit_spread is not None and not pd.isna(baseline_exit_spread):
            baseline_gross = (baseline_exit_spread - entry_spread) * direction
            baseline_pnl = baseline_gross - ROUNDTRIP_COST_POINTS

        pnl_diff = rr_pnl - baseline_pnl if not np.isnan(baseline_pnl) else np.nan
        lead_times.append(lead_time)
        pnl_diffs.append(pnl_diff)
        trade_details.append({
            'entry_date': entry_date, 'instrument': inst,
            'direction': t['direction'],
            'rr_exit_date': rr_exit_date, 'rr_pnl': rr_pnl,
            'baseline_exit_date': baseline_exit_date,
            'baseline_exit_reason': baseline_exit_reason,
            'baseline_pnl': baseline_pnl, 'pnl_diff': pnl_diff,
            'lead_time': lead_time,
        })

    lt_arr = np.array(lead_times)
    genuine_early = (lt_arr > 0).sum()
    same_day = (lt_arr == 0).sum()
    no_flip = (lt_arr < 0).sum()

    lines.append(f'    Genuine early exit (>0 days before flip): {genuine_early}')
    lines.append(f'    Same-day as flip (0 days lead):           {same_day}')
    lines.append(f'    Shape never flipped (TP/TS would have):   {no_flip}')
    if genuine_early > 0:
        early_lts = lt_arr[lt_arr > 0]
        lines.append(f'    Avg lead time (genuine early only):       {early_lts.mean():.1f} days')

    valid_diffs = [d for d in pnl_diffs if not np.isnan(d)]
    if valid_diffs:
        saved = sum(1 for d in valid_diffs if d > 0)
        hurt_count = sum(1 for d in valid_diffs if d < 0)
        neutral = sum(1 for d in valid_diffs if d == 0)
        total_diff = sum(valid_diffs)
        lines.append(f'    P&L vs baseline: saved={saved}, hurt={hurt_count}, neutral={neutral}')
        lines.append(f'    Total P&L difference: {total_diff:+.2f}')

    lines.append('')
    lines.append(f'    {"Entry":>10s} {"Inst":>12s} {"Dir":>5s} {"RR_Exit":>10s} {"RR_PnL":>8s} '
                 f'{"BL_Exit":>10s} {"BL_Reason":>12s} {"BL_PnL":>8s} {"Diff":>8s} {"Lead":>5s}')
    lines.append(f'    {"-"*10} {"-"*12} {"-"*5} {"-"*10} {"-"*8} '
                 f'{"-"*10} {"-"*12} {"-"*8} {"-"*8} {"-"*5}')
    for td in trade_details:
        bl_date = str(td['baseline_exit_date'].date()) if td['baseline_exit_date'] is not None else '       n/a'
        bl_reason = td['baseline_exit_reason'] if td['baseline_exit_reason'] else 'n/a'
        bl_pnl = f'{td["baseline_pnl"]:>+8.2f}' if not np.isnan(td['baseline_pnl']) else '     n/a'
        diff = f'{td["pnl_diff"]:>+8.2f}' if not np.isnan(td['pnl_diff']) else '     n/a'
        lt = f'{td["lead_time"]:>5d}' if td['lead_time'] >= 0 else ' none'
        lines.append(f'    {str(td["entry_date"].date()):>10s} {td["instrument"]:>12s} {td["direction"]:>5s} '
                     f'{str(td["rr_exit_date"].date()):>10s} {td["rr_pnl"]:>+8.2f} '
                     f'{bl_date:>10s} {bl_reason:>12s} {bl_pnl} {diff} {lt}')

# Exit-type reclassification
lines.append('')
lines.append('')
lines.append('--- EXIT-TYPE RECLASSIFICATION (Part 2.7) ---')
lines.append('')
lines.append(f'Baseline invalidated exits: {baseline_inv_count}')
lines.append('')

for r in rr_results[1:]:
    pt = r['thresh']
    label = r['label']
    oos_rr = r['oos']
    if len(oos_rr) == 0:
        continue

    new_inv = (oos_rr['exit_reason'] == 'invalidated').sum()
    new_rr = (oos_rr['exit_reason'] == 'regime_risk').sum()
    reclassified = baseline_inv_count - new_inv
    lines.append(f'  {label}: regime_risk={new_rr}, still_invalidated={new_inv}, '
                 f'reclassified={reclassified}/{baseline_inv_count} '
                 f'({reclassified/baseline_inv_count*100:.1f}%)')

lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
