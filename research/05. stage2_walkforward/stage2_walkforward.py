"""
Stage 2 — Walk-Forward Parameter Stability
============================================
Tests whether Stage 1 locked parameters (z>1.5, dur>=3d, 9 instruments,
PM L0, TM regime-risk <70%) are stable across 4 independent historical
windows, or tuned to one specific OOS period.

Windows:
  W1: Train 2008-2018 -> Test 2019-2020
  W2: Train 2008-2020 -> Test 2021-2022
  W3: Train 2008-2022 -> Test 2023-2024
  W4: Train 2008-2024 -> Test 2025-2026

Creates: research/05. stage2_walkforward/stage2_walkforward_log.txt
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
Z_EXIT = 0.5
TIME_STOP_DAYS = 20
ROUNDTRIP_COST_MYR = 100.0
POINT_VALUE = 25.0
ROUNDTRIP_COST_POINTS = ROUNDTRIP_COST_MYR / POINT_VALUE  # 4.0
TM_REGIME_RISK_THRESH = 0.70  # exit if persistence_prob < 70%

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

CORE_4 = ['M2-M3', 'M3-M4', 'M4-M5', 'M5-M6']
REINSTATED_5 = ['M1-M2', 'BF_M1M2M3', 'BF_M2M3M4', 'BF_M3M4M5', 'BF_M4M5M6']
ALL_9 = CORE_4 + REINSTATED_5

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

# Compute calendar spreads
for name, cfg in ALL_INSTRUMENT_CONFIG.items():
    df[name] = df[cfg['near']] - df[cfg['far']]
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

# ══════════════════════════════════════════════════════════════
# PART 1 — PM/TM AVAILABILITY
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 1: PM/TM AVAILABILITY')
print('='*70)

# PM predictions across full 2017+ range
print('Running PM predictions across full panel (2017+)...')
model_start = pd.Timestamp('2017-01-01')
pm_level_col = pd.Series(np.nan, index=df.index)
pm_ok_count = 0
pm_fail_count = 0
pm_first_date = None
pm_last_date = None

for i in df[df['date'] >= model_start].index:
    dt = df.loc[i, 'date']
    obs_shape = str(df.loc[i, 'shape'])
    try:
        pm = pm_predict(dt)
        pred, conf = pm.get('predicted_shape'), pm.get('confidence')
        probs = pm.get('shape_probs', {})
        if pd.isna(pred) or pd.isna(conf):
            pm_fail_count += 1
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
        pm_ok_count += 1
        if pm_first_date is None:
            pm_first_date = dt
        pm_last_date = dt
    except Exception:
        pm_fail_count += 1

df['pm_level'] = pm_level_col
print(f'PM: {pm_ok_count} predictions OK, {pm_fail_count} failed')
print(f'PM available: {pm_first_date.date() if pm_first_date else "N/A"} to {pm_last_date.date() if pm_last_date else "N/A"}')
print(f'PM levels: {df["pm_level"].value_counts().sort_index().to_dict()}')

# Per-window PM availability
for w in WINDOWS:
    mask = (df['date'] >= pd.Timestamp(w['test_start'])) & (df['date'] <= pd.Timestamp(w['test_end']))
    pm_avail = df.loc[mask, 'pm_level'].notna().sum()
    pm_total = mask.sum()
    print(f'  {w["name"]}: {pm_avail}/{pm_total} days with PM prediction')

# TM predictions across full 2017+ range
print('\nPre-computing TM persistence probabilities (2017+)...')
tm_cache = {}
tm_ok_count = 0
tm_fail_count = 0
tm_first_date = None
tm_last_date = None

for idx in df[df['date'] >= model_start].index:
    dt = df.loc[idx, 'date']
    dt_ts = pd.Timestamp(dt)
    try:
        result = tm_predict(dt_ts, '1w')
        if 'error' not in result:
            current_shape = str(result['current_shape'])
            all_probs = result.get('all_probs', {})
            persistence_prob = all_probs.get(current_shape, np.nan)
            tm_cache[dt_ts] = {
                'current_shape': current_shape,
                'persistence_prob': persistence_prob,
                'all_probs': all_probs,
            }
            tm_ok_count += 1
            if tm_first_date is None:
                tm_first_date = dt_ts
            tm_last_date = dt_ts
        else:
            tm_fail_count += 1
    except Exception:
        tm_fail_count += 1

print(f'TM: {tm_ok_count} predictions OK, {tm_fail_count} failed')
print(f'TM available: {tm_first_date.date() if tm_first_date else "N/A"} to {tm_last_date.date() if tm_last_date else "N/A"}')

# Per-window TM availability
for w in WINDOWS:
    ws = pd.Timestamp(w['test_start'])
    we = pd.Timestamp(w['test_end'])
    tm_avail = sum(1 for dt in tm_cache if ws <= dt <= we)
    mask = (df['date'] >= ws) & (df['date'] <= we)
    total = mask.sum()
    print(f'  {w["name"]}: {tm_avail}/{total} days with TM prediction')

# Check first valid TM date (all features non-NaN)
first_valid_dates = sorted(tm_cache.keys())
print(f'First valid TM date: {first_valid_dates[0].date() if first_valid_dates else "N/A"}')


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE (with TM regime-risk exit)
# ══════════════════════════════════════════════════════════════

def run_backtest(df, instruments, dur_thresh, z_entry, test_start, test_end,
                 use_tm_exit=True, label=''):
    """Run backtest over a specific test window.
    Trades are only counted if exit_date falls within [test_start, test_end]."""
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

                # TM regime-risk exit (only if shape hasn't changed yet)
                if use_tm_exit and shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < TM_REGIME_RISK_THRESH:
                            exit_reason = 'regime_risk'

                # Standard exits
                if not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if shape != entry_shape:
                    exit_reason = 'invalidated'
                if days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'

                if exit_reason:
                    gross_pnl = (spread - entry_spread) * entry_direction
                    net_pnl = gross_pnl - ROUNDTRIP_COST_POINTS

                    # Only count trades that exit within the test window
                    if ts <= date <= te:
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


def compute_metrics(trades, label, test_start, test_end, df_ref):
    """Compute metrics for trades within a specific test window."""
    n = len(trades)
    if n == 0:
        return {'label': label, 'n_trades': 0, 'win_rate': 0, 'avg_win': 0,
                'avg_loss': 0, 'total_pnl': 0, 'sharpe': np.nan, 'max_dd': 0,
                'pct_take_profit': 0, 'pct_invalidated': 0, 'pct_time_stop': 0,
                'pct_regime_risk': 0, 'avg_hp': 0, 'shape_survival': 0}

    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]
    win_rate = round(len(wins) / n * 100, 1)
    avg_win = round(wins['net_pnl'].mean(), 2) if len(wins) > 0 else 0
    avg_loss = round(losses['net_pnl'].mean(), 2) if len(losses) > 0 else 0
    total_pnl = round(trades['net_pnl'].sum(), 2)

    tp_pct = round((trades['exit_reason'] == 'take_profit').mean() * 100, 1)
    inv_pct = round((trades['exit_reason'] == 'invalidated').mean() * 100, 1)
    ts_pct = round((trades['exit_reason'] == 'time_stop').mean() * 100, 1)
    rr_pct = round((trades['exit_reason'] == 'regime_risk').mean() * 100, 1)

    avg_hp = round(trades['days_held'].mean(), 1)
    cum_pnl = trades['net_pnl'].cumsum()
    max_dd = round((cum_pnl - cum_pnl.cummax()).min(), 2) if n > 0 else 0

    shape_survival = round(trades['shape_survived'].mean() * 100, 1) if 'shape_survived' in trades.columns else np.nan

    # Naive Sharpe over the test window
    ts_dt = pd.Timestamp(test_start)
    te_dt = pd.Timestamp(test_end)
    window_dates = df_ref[(df_ref['date'] >= ts_dt) & (df_ref['date'] <= te_dt)]['date']
    daily_pnl = pd.Series(0.0, index=window_dates.values)
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
        'pct_time_stop': ts_pct, 'pct_regime_risk': rr_pct,
        'avg_hp': avg_hp, 'shape_survival': shape_survival,
    }


def compute_daily_portfolio_sharpe(trades, test_start, test_end, df_ref):
    """Daily portfolio Sharpe using mark-to-market within a test window."""
    if len(trades) == 0:
        return np.nan
    ts_dt = pd.Timestamp(test_start)
    te_dt = pd.Timestamp(test_end)
    window_dates = df_ref[(df_ref['date'] >= ts_dt) & (df_ref['date'] <= te_dt)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=window_dates)

    for _, t in trades.iterrows():
        entry_dt = t['entry_date']
        exit_dt = t['exit_date']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']

        trade_days = df_ref[(df_ref['date'] > entry_dt) & (df_ref['date'] <= exit_dt)].copy()
        if len(trade_days) == 0:
            continue

        prev_spread = t['entry_spread']
        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            current_spread = day_row[inst]
            if pd.isna(current_spread):
                continue
            day_mtm = (current_spread - prev_spread) * direction
            if dt in daily_pnl.index:
                daily_pnl[dt] += day_mtm
            prev_spread = current_spread

        if exit_dt in daily_pnl.index:
            daily_pnl[exit_dt] -= ROUNDTRIP_COST_POINTS

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
# PART 2/3 — PER-WINDOW PARAMETER SWEEP
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 2/3: PER-WINDOW PARAMETER SWEEP')
print('='*70)

Z_THRESHOLDS = [1.5, 1.75, 2.0]
DUR_THRESHOLDS = [10, 5, 3]

# Store all results
window_results = {}

for w in WINDOWS:
    wname = w['name']
    ts, te = w['test_start'], w['test_end']
    print(f'\n{"="*60}')
    print(f'WINDOW: {wname} (test {ts} to {te})')
    print(f'{"="*60}')

    window_results[wname] = {
        'z_sweep': [],
        'dur_sweep': [],
        'locked_config': None,
        'inst_breakdown': {},
    }

    # ── Z-score sweep (dur fixed at 3d) ──
    print(f'\n  Z-score sweep (dur>=3d fixed):')
    for z_entry in Z_THRESHOLDS:
        label = f'z>{z_entry}'
        trades = run_backtest(df, ALL_9, dur_thresh=3, z_entry=z_entry,
                              test_start=ts, test_end=te, label=label)
        metrics = compute_metrics(trades, label, ts, te, df)
        adj = compute_daily_portfolio_sharpe(trades, ts, te, df)
        g = gate_check(metrics)

        window_results[wname]['z_sweep'].append({
            'z': z_entry, 'label': label, 'metrics': metrics, 'adj': adj, 'gate': g,
        })

        print(f'    {label:8s}: n={metrics["n_trades"]:>3d}, win={metrics["win_rate"]:>5.1f}%, '
              f'naive={fmt_sharpe(metrics["sharpe"])}, adj={fmt_sharpe(adj)}, '
              f'PnL={metrics["total_pnl"]:>8.1f}, DD={metrics["max_dd"]:>8.1f}, gate={g}')

    # ── Duration sweep (z fixed at 1.5) ──
    print(f'\n  Duration sweep (z>1.5 fixed):')
    for dur in DUR_THRESHOLDS:
        label = f'dur>={dur}d'
        trades = run_backtest(df, ALL_9, dur_thresh=dur, z_entry=1.5,
                              test_start=ts, test_end=te, label=label)
        metrics = compute_metrics(trades, label, ts, te, df)
        adj = compute_daily_portfolio_sharpe(trades, ts, te, df)
        g = gate_check(metrics)

        window_results[wname]['dur_sweep'].append({
            'dur': dur, 'label': label, 'metrics': metrics, 'adj': adj, 'gate': g,
        })

        print(f'    {label:10s}: n={metrics["n_trades"]:>3d}, win={metrics["win_rate"]:>5.1f}%, '
              f'naive={fmt_sharpe(metrics["sharpe"])}, adj={fmt_sharpe(adj)}, '
              f'PnL={metrics["total_pnl"]:>8.1f}, DD={metrics["max_dd"]:>8.1f}, gate={g}')

    # ── Locked config (z>1.5, dur>=3d) — for Part 4.3 ──
    # Already computed in z_sweep (z>1.5) which uses dur=3d
    locked = window_results[wname]['z_sweep'][0]  # z>1.5, dur>=3d
    window_results[wname]['locked_config'] = locked

    # ── Per-instrument breakdown at locked config ──
    print(f'\n  Per-instrument breakdown (z>1.5, dur>=3d):')
    trades_locked = run_backtest(df, ALL_9, dur_thresh=3, z_entry=1.5,
                                 test_start=ts, test_end=te, label='locked')
    for inst in ALL_9:
        inst_trades = trades_locked[trades_locked['instrument'] == inst] if len(trades_locked) > 0 else pd.DataFrame()
        inst_metrics = compute_metrics(inst_trades, inst, ts, te, df)
        inst_adj = compute_daily_portfolio_sharpe(inst_trades, ts, te, df)
        window_results[wname]['inst_breakdown'][inst] = {
            'metrics': inst_metrics, 'adj': inst_adj,
        }
        n = inst_metrics['n_trades']
        if n > 0:
            print(f'    {inst:12s}: n={n:>3d}, win={inst_metrics["win_rate"]:>5.1f}%, '
                  f'naive={fmt_sharpe(inst_metrics["sharpe"])}, PnL={inst_metrics["total_pnl"]:>8.1f}')
        else:
            print(f'    {inst:12s}: n=  0 (no trades)')


# ══════════════════════════════════════════════════════════════
# PART 4 — STABILITY VERDICT
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 4: STABILITY VERDICT')
print('='*70)

# 4.1 — Summary table
print('\n--- 4.1 SUMMARY TABLE ---')

# Find locally-optimal z and dur per window
summary_rows = []
for w in WINDOWS:
    wname = w['name']
    wr = window_results[wname]

    # Best z by adj Sharpe
    best_z_result = max(wr['z_sweep'], key=lambda r: r['adj'] if r['adj'] is not None and not np.isnan(r['adj']) else -999)
    best_z = best_z_result['z']
    best_z_adj = best_z_result['adj']

    # Best dur by adj Sharpe
    best_dur_result = max(wr['dur_sweep'], key=lambda r: r['adj'] if r['adj'] is not None and not np.isnan(r['adj']) else -999)
    best_dur = best_dur_result['dur']
    best_dur_adj = best_dur_result['adj']

    # Locked config
    locked = wr['locked_config']
    locked_adj = locked['adj']
    locked_n = locked['metrics']['n_trades']

    summary_rows.append({
        'window': wname,
        'best_z': best_z, 'best_z_adj': best_z_adj,
        'best_dur': best_dur, 'best_dur_adj': best_dur_adj,
        'locked_adj': locked_adj, 'locked_n': locked_n,
    })

header = (f'  {"Window":16s} {"BestZ":>6s} {"BestDur":>8s} '
          f'{"LockedAdj":>10s} {"BestZAdj":>10s} {"BestDurAdj":>11s} {"n(locked)":>10s}')
print(header)
print(f'  {"-"*16} {"-"*6} {"-"*8} {"-"*10} {"-"*10} {"-"*11} {"-"*10}')
for sr in summary_rows:
    print(f'  {sr["window"]:16s} {sr["best_z"]:>6.2f} {sr["best_dur"]:>6d}d '
          f'{fmt_sharpe(sr["locked_adj"]):>10s} {fmt_sharpe(sr["best_z_adj"]):>10s} '
          f'{fmt_sharpe(sr["best_dur_adj"]):>11s} {sr["locked_n"]:>10d}')

# 4.2 — Stability rule
print('\n--- 4.2 STABILITY VERDICT ---')
best_zs = [sr['best_z'] for sr in summary_rows]
best_durs = [sr['best_dur'] for sr in summary_rows]

z_stable = all(1.25 <= z <= 1.75 for z in best_zs)
dur_stable = all(1 <= d <= 5 for d in best_durs)

print(f'Locally-optimal z per window: {best_zs}')
print(f'  Rule: stable if all within 1.5 +/- 0.25 (i.e. [1.25, 1.75])')
print(f'  Result: {"STABLE" if z_stable else "UNSTABLE"}')

print(f'\nLocally-optimal dur per window: {best_durs}')
print(f'  Rule: stable if all within 3d +/- 2d (i.e. [1, 5])')
print(f'  Result: {"STABLE" if dur_stable else "UNSTABLE"}')

overall_stable = z_stable and dur_stable
print(f'\nOverall verdict: {"STABLE" if overall_stable else "UNSTABLE"}')

# 4.3 — Fixed config across all windows
print('\n--- 4.3 FIXED CONFIG (z>1.5, dur>=3d) ACROSS ALL WINDOWS ---')
all_positive = True
for sr in summary_rows:
    adj = sr['locked_adj']
    if adj is None or np.isnan(adj) or adj <= 0:
        all_positive = False
    status = 'POSITIVE' if (adj is not None and not np.isnan(adj) and adj > 0) else 'NON-POSITIVE'
    print(f'  {sr["window"]:16s}: adj Sharpe = {fmt_sharpe(adj)}, n = {sr["locked_n"]}, {status}')

print(f'\nFixed config positive in ALL windows: {"YES" if all_positive else "NO"}')

# 4.4 — Instrument ranking consistency
print('\n--- 4.4 INSTRUMENT RANKING CONSISTENCY ---')

# Build ranking per window (by total PnL at locked config)
rankings = {}
for w in WINDOWS:
    wname = w['name']
    wr = window_results[wname]
    inst_pnl = []
    for inst in ALL_9:
        ib = wr['inst_breakdown'][inst]
        pnl = ib['metrics']['total_pnl']
        inst_pnl.append((inst, pnl))
    # Sort by PnL descending
    inst_pnl.sort(key=lambda x: -x[1])
    rankings[wname] = inst_pnl

# Print rankings
print(f'\n  {"Rank":>4s}', end='')
for w in WINDOWS:
    print(f'  {w["name"]:>16s}', end='')
print()
print(f'  {"-"*4}', end='')
for _ in WINDOWS:
    print(f'  {"-"*16}', end='')
print()

for rank in range(len(ALL_9)):
    print(f'  {rank+1:>4d}', end='')
    for w in WINDOWS:
        inst, pnl = rankings[w['name']][rank]
        print(f'  {inst:>10s}({pnl:>+5.0f})', end='')
    print()

# Check BF_M1M2M3 and M5-M6 consistency
print('\nBF_M1M2M3 rank per window:', end='')
for w in WINDOWS:
    wname = w['name']
    for rank, (inst, _) in enumerate(rankings[wname]):
        if inst == 'BF_M1M2M3':
            print(f'  {wname}: #{rank+1}', end='')
            break
print()

print('M5-M6 rank per window:', end='')
for w in WINDOWS:
    wname = w['name']
    for rank, (inst, _) in enumerate(rankings[wname]):
        if inst == 'M5-M6':
            print(f'  {wname}: #{rank+1}', end='')
            break
print()


# ══════════════════════════════════════════════════════════════
# PART 5 — LOGGING
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 5: LOGGING')
print('='*70)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
lines = []
lines.append('=' * 70)
lines.append(f'STAGE 2 — WALK-FORWARD PARAMETER STABILITY — {timestamp}')
lines.append('=' * 70)
lines.append('')
lines.append('Stage 1 locked config: z>1.5, dur>=3d, 9 instruments, PM L0,')
lines.append('TM regime-risk exit <70%, 100 MYR (4.0 pts) round-trip cost.')
lines.append('')
lines.append('Windows:')
lines.append('  W1: Train 2008-2018 -> Test 2019-2020')
lines.append('  W2: Train 2008-2020 -> Test 2021-2022')
lines.append('  W3: Train 2008-2022 -> Test 2023-2024')
lines.append('  W4: Train 2008-2024 -> Test 2025-2026')

# ── Part 1: PM/TM availability ──────────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 1: PM/TM AVAILABILITY ---')
lines.append('')
lines.append('PM/TM outputs generated for MR walk-forward testing — NOT a')
lines.append('re-validation of PM/TM\'s own accuracy, which remains test-only')
lines.append('validated on 2025-2026.')
lines.append('')
lines.append(f'PM predictions: {pm_ok_count} OK, {pm_fail_count} failed')
lines.append(f'PM available range: {pm_first_date.date() if pm_first_date else "N/A"} to {pm_last_date.date() if pm_last_date else "N/A"}')
lines.append(f'TM predictions: {tm_ok_count} OK, {tm_fail_count} failed')
lines.append(f'TM available range: {tm_first_date.date() if tm_first_date else "N/A"} to {tm_last_date.date() if tm_last_date else "N/A"}')
lines.append(f'First valid TM date: {first_valid_dates[0].date() if first_valid_dates else "N/A"}')
lines.append('')
lines.append('Per-window availability:')
for w in WINDOWS:
    mask = (df['date'] >= pd.Timestamp(w['test_start'])) & (df['date'] <= pd.Timestamp(w['test_end']))
    pm_avail = df.loc[mask, 'pm_level'].notna().sum()
    pm_total = mask.sum()
    ws = pd.Timestamp(w['test_start'])
    we = pd.Timestamp(w['test_end'])
    tm_avail = sum(1 for dt in tm_cache if ws <= dt <= we)
    lines.append(f'  {w["name"]}: PM {pm_avail}/{pm_total}, TM {tm_avail}/{pm_total}')
lines.append('')
lines.append('NOTE: PM and TM models were trained on 2017-2024 data. Predictions')
lines.append('for W1-W3 test windows (2019-2024) are technically in-sample for')
lines.append('the PM/TM models themselves, though out-of-sample for the MR')
lines.append('backtest z-score computation. This is acceptable because PM/TM')
lines.append('are used only as filters/overlays, not as the primary signal.')

# ── Part 3: Per-window sweep results ─────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 3: PER-WINDOW PARAMETER SWEEP ---')

for w in WINDOWS:
    wname = w['name']
    wr = window_results[wname]
    lines.append('')
    lines.append(f'  {"="*50}')
    lines.append(f'  WINDOW: {wname} (test {w["test_start"]} to {w["test_end"]})')
    lines.append(f'  {"="*50}')

    # Z-score sweep
    lines.append('')
    lines.append(f'  Z-score sweep (dur>=3d fixed):')
    lines.append(f'    {"Variant":8s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"AdjShrp":>8s} '
                 f'{"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"Gate":>22s}')
    lines.append(f'    {"-"*8} {"-"*4} {"-"*6} {"-"*7} {"-"*8} '
                 f'{"-"*10} {"-"*8} {"-"*8} {"-"*8} {"-"*22}')
    for r in wr['z_sweep']:
        m = r['metrics']
        lines.append(f'    {r["label"]:8s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
                     f'{fmt_sharpe(r["adj"]):>8s} '
                     f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
                     f'{m["max_dd"]:>8.1f} {r["gate"]:>22s}')

    # Duration sweep
    lines.append('')
    lines.append(f'  Duration sweep (z>1.5 fixed):')
    lines.append(f'    {"Variant":10s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"AdjShrp":>8s} '
                 f'{"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"Gate":>22s}')
    lines.append(f'    {"-"*10} {"-"*4} {"-"*6} {"-"*7} {"-"*8} '
                 f'{"-"*10} {"-"*8} {"-"*8} {"-"*8} {"-"*22}')
    for r in wr['dur_sweep']:
        m = r['metrics']
        lines.append(f'    {r["label"]:10s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
                     f'{fmt_sharpe(r["adj"]):>8s} '
                     f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
                     f'{m["max_dd"]:>8.1f} {r["gate"]:>22s}')

    # Per-instrument breakdown
    lines.append('')
    lines.append(f'  Per-instrument breakdown (z>1.5, dur>=3d):')
    lines.append(f'    {"Instrument":12s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"TotalPnL":>10s}')
    lines.append(f'    {"-"*12} {"-"*4} {"-"*6} {"-"*7} {"-"*10}')
    for inst in ALL_9:
        ib = wr['inst_breakdown'][inst]
        m = ib['metrics']
        if m['n_trades'] > 0:
            lines.append(f'    {inst:12s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% '
                         f'{fmt_sharpe(m["sharpe"]):>7s} {m["total_pnl"]:>10.1f}')
        else:
            lines.append(f'    {inst:12s}    0   (no trades)')

# ── Part 4: Stability verdict ────────────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 4: STABILITY VERDICT ---')

# 4.1 Summary table
lines.append('')
lines.append('4.1 — Summary table:')
lines.append('')
lines.append(f'  {"Window":16s} {"BestZ":>6s} {"BestDur":>8s} '
             f'{"LockedAdj":>10s} {"BestZAdj":>10s} {"BestDurAdj":>11s} {"n(locked)":>10s}')
lines.append(f'  {"-"*16} {"-"*6} {"-"*8} {"-"*10} {"-"*10} {"-"*11} {"-"*10}')
for sr in summary_rows:
    lines.append(f'  {sr["window"]:16s} {sr["best_z"]:>6.2f} {sr["best_dur"]:>6d}d '
                 f'{fmt_sharpe(sr["locked_adj"]):>10s} {fmt_sharpe(sr["best_z_adj"]):>10s} '
                 f'{fmt_sharpe(sr["best_dur_adj"]):>11s} {sr["locked_n"]:>10d}')

# 4.2 Stability verdict
lines.append('')
lines.append('4.2 — Stability rule:')
lines.append(f'  Locally-optimal z per window: {best_zs}')
lines.append(f'  Rule: stable if all within [1.25, 1.75] (i.e. 1.5 +/- 0.25)')
lines.append(f'  Z-score: {"STABLE" if z_stable else "UNSTABLE"}')
lines.append('')
lines.append(f'  Locally-optimal dur per window: {best_durs}')
lines.append(f'  Rule: stable if all within [1, 5] (i.e. 3d +/- 2d)')
lines.append(f'  Duration: {"STABLE" if dur_stable else "UNSTABLE"}')
lines.append('')
lines.append(f'  Overall verdict: {"STABLE" if overall_stable else "UNSTABLE"}')

# 4.3 Fixed config across all windows
lines.append('')
lines.append('4.3 — Fixed config (z>1.5, dur>=3d) across all windows:')
for sr in summary_rows:
    adj = sr['locked_adj']
    status = 'POSITIVE' if (adj is not None and not np.isnan(adj) and adj > 0) else 'NON-POSITIVE'
    lines.append(f'  {sr["window"]:16s}: adj Sharpe = {fmt_sharpe(adj)}, n = {sr["locked_n"]}, {status}')
lines.append(f'  Fixed config positive in ALL windows: {"YES" if all_positive else "NO"}')

# 4.4 Instrument ranking consistency
lines.append('')
lines.append('4.4 — Instrument ranking by total PnL (z>1.5, dur>=3d):')
lines.append('')
rank_header = f'  {"Rank":>4s}'
for w in WINDOWS:
    rank_header += f'  {w["name"]:>16s}'
lines.append(rank_header)
rank_sep = f'  {"-"*4}'
for _ in WINDOWS:
    rank_sep += f'  {"-"*16}'
lines.append(rank_sep)
for rank in range(len(ALL_9)):
    line = f'  {rank+1:>4d}'
    for w in WINDOWS:
        inst, pnl = rankings[w['name']][rank]
        line += f'  {inst:>10s}({pnl:>+5.0f})'
    lines.append(line)

lines.append('')
bf_ranks = []
m5m6_ranks = []
for w in WINDOWS:
    wname = w['name']
    for rank, (inst, _) in enumerate(rankings[wname]):
        if inst == 'BF_M1M2M3':
            bf_ranks.append(rank+1)
        if inst == 'M5-M6':
            m5m6_ranks.append(rank+1)

lines.append(f'BF_M1M2M3 rank per window: {bf_ranks}')
lines.append(f'M5-M6 rank per window: {m5m6_ranks}')

bf_consistent = all(r <= 3 for r in bf_ranks)
m5m6_bottom = all(r >= 7 for r in m5m6_ranks)
lines.append(f'BF_M1M2M3 in top 3 every window: {"YES" if bf_consistent else "NO"}')
lines.append(f'M5-M6 in bottom 3 every window: {"YES" if m5m6_bottom else "NO"}')

lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults written to {LOG_FILE}')
print('Done.')
