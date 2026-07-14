"""
Stage 1 — Loss Classification, Stop-Loss, and Scale-In Test
=============================================================
Part 1: Classify every losing trade into Bucket A/B/C
Part 2: Test z-score stop-loss at |z|>=3.0/3.5/4.0
Part 3: Scale-in (pyramiding) on top of best stop-loss
Part 4: Log everything to backtest_analysis.txt
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
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
# BACKTEST ENGINE — Extended with stop-loss, z-path tracking
# ══════════════════════════════════════════════════════════════

def run_backtest_extended(df, instruments, dur_thresh, z_entry=Z_ENTRY,
                          stop_loss_z=None, label=''):
    """Run backtest with optional z-score stop-loss.
    Tracks z-score path during each trade for loss classification.
    Returns trades_df with extra columns: max_adverse_z, z_path."""
    all_trades = []

    for inst in instruments:
        z_col = f'{inst}_z'

        position_open = False
        entry_date = entry_spread = entry_z = entry_shape = entry_direction = None
        days_held = 0
        z_path = []  # track z-scores during holding period

        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]

            if pd.isna(spread):
                continue

            if position_open:
                days_held += 1
                if not pd.isna(z):
                    z_path.append(z)
                exit_reason = None

                # Stop-loss checked FIRST (before other exits)
                if stop_loss_z is not None and not pd.isna(z) and abs(z) >= stop_loss_z:
                    exit_reason = 'stop_loss'

                # Then the original three exits
                if exit_reason is None and not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if exit_reason is None and shape != entry_shape:
                    exit_reason = 'invalidated'
                if exit_reason is None and days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'

                if exit_reason:
                    gross_pnl = (spread - entry_spread) * entry_direction
                    net_pnl = gross_pnl - ROUNDTRIP_COST_POINTS
                    shape_survived = (exit_reason not in ('invalidated',))

                    # Compute max adverse z extension
                    # "adverse" = z moved FURTHER from zero than entry
                    max_adverse_z = np.nan
                    if z_path:
                        abs_zs = [abs(zv) for zv in z_path if not np.isnan(zv)]
                        if abs_zs:
                            max_adverse_z = max(abs_zs)

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
                        'max_adverse_z': round(max_adverse_z, 3) if not np.isnan(max_adverse_z) else np.nan,
                        'z_path': list(z_path),  # full path for analysis
                    })
                    position_open = False
                    z_path = []
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
                    z_path = []

    trades_df = pd.DataFrame(all_trades)
    return trades_df


def run_backtest_scalein(df, instruments, dur_thresh, z_entry=Z_ENTRY,
                         stop_loss_z=None, scalein_z=2.0, label=''):
    """Run backtest with stop-loss AND scale-in (max 2 lots).
    Scale-in: if already in 1-lot position and |z| >= scalein_z in same
    direction (still same shape), add a 2nd lot. Both exit together."""
    all_trades = []

    for inst in instruments:
        z_col = f'{inst}_z'

        position_open = False
        lots = 0  # 0, 1, or 2
        entry_date = entry_spread = entry_z = entry_shape = entry_direction = None
        scalein_date = scalein_spread = scalein_z_val = None
        days_held = 0
        z_path = []

        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]

            if pd.isna(spread):
                continue

            if position_open:
                days_held += 1
                if not pd.isna(z):
                    z_path.append(z)
                exit_reason = None

                # Check scale-in BEFORE exits (can add 2nd lot same day)
                if lots == 1 and not pd.isna(z) and shape == entry_shape:
                    # z must extend further in SAME direction as entry
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

                # Stop-loss checked FIRST
                if stop_loss_z is not None and not pd.isna(z) and abs(z) >= stop_loss_z:
                    exit_reason = 'stop_loss'

                if exit_reason is None and not pd.isna(z) and abs(z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if exit_reason is None and shape != entry_shape:
                    exit_reason = 'invalidated'
                if exit_reason is None and days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'

                if exit_reason:
                    # Lot 1 P&L
                    gross_pnl_1 = (spread - entry_spread) * entry_direction
                    # Lot 2 P&L (if scaled in)
                    gross_pnl_2 = 0.0
                    if lots == 2:
                        gross_pnl_2 = (spread - scalein_spread) * entry_direction

                    total_gross = gross_pnl_1 + gross_pnl_2
                    total_cost = ROUNDTRIP_COST_POINTS * lots
                    total_net = total_gross - total_cost

                    shape_survived = (exit_reason not in ('invalidated',))
                    max_adverse_z = np.nan
                    if z_path:
                        abs_zs = [abs(zv) for zv in z_path if not np.isnan(zv)]
                        if abs_zs:
                            max_adverse_z = max(abs_zs)

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
                        'shape_survived': shape_survived,
                        'max_adverse_z': round(max_adverse_z, 3) if not np.isnan(max_adverse_z) else np.nan,
                        'z_path': list(z_path),
                    })
                    position_open = False
                    lots = 0
                    scalein_date = scalein_spread = scalein_z_val = None
                    z_path = []
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
                    z_path = []

    trades_df = pd.DataFrame(all_trades)
    return trades_df


def compute_metrics_ext(trades, label, df_ref):
    """Compute metrics with 4 exit types (including stop_loss)."""
    n = len(trades)
    if n == 0:
        return {'label': label, 'n_trades': 0, 'win_rate': 0, 'avg_win': 0,
                'avg_loss': 0, 'total_pnl': 0, 'sharpe': np.nan, 'max_dd': 0,
                'pct_take_profit': 0, 'pct_invalidated': 0, 'pct_time_stop': 0,
                'pct_stop_loss': 0, 'avg_hp': 0, 'shape_survival': 0}

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
        'avg_hp': avg_hp, 'shape_survival': shape_survival,
    }


def compute_daily_portfolio_sharpe(trades, df_ref):
    """Daily portfolio Sharpe using mark-to-market across all positions."""
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

        # For scale-in trades, we need to handle 2 lots with different entry spreads
        # Lot 1 always active from entry_date
        prev_spread_1 = t['entry_spread']
        # Lot 2 active from scalein_date (if applicable)
        has_lot2 = n_lots == 2 and not pd.isna(t.get('scalein_date', np.nan))
        scalein_dt = t.get('scalein_date', None) if has_lot2 else None
        prev_spread_2 = t.get('scalein_spread', np.nan) if has_lot2 else np.nan

        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            current_spread = day_row[inst]
            if pd.isna(current_spread):
                continue

            # Lot 1 MTM
            day_mtm = (current_spread - prev_spread_1) * direction
            if dt in daily_pnl.index:
                daily_pnl[dt] += day_mtm
            prev_spread_1 = current_spread

            # Lot 2 MTM (only after scale-in date)
            if has_lot2 and scalein_dt is not None and dt > scalein_dt:
                day_mtm_2 = (current_spread - prev_spread_2) * direction
                if dt in daily_pnl.index:
                    daily_pnl[dt] += day_mtm_2
                prev_spread_2 = current_spread
            elif has_lot2 and scalein_dt is not None and dt == scalein_dt:
                # Scale-in day: lot 2 enters at scalein_spread, no MTM yet
                prev_spread_2 = current_spread

        # Subtract cost on exit day
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
# PART 1 — LOSS CLASSIFICATION
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 1: LOSS CLASSIFICATION')
print('='*70)

# Run baseline (no stop-loss) with z-path tracking
print('Running baseline backtest (no stop-loss)...')
baseline_trades = run_backtest_extended(df, ALL_9, dur_thresh=10, label='baseline')
oos_baseline = baseline_trades[baseline_trades['oos']].reset_index(drop=True)
print(f'Total OOS trades: {len(oos_baseline)}')

# Losing trades
losing = oos_baseline[oos_baseline['net_pnl'] < 0].reset_index(drop=True)
print(f'Losing trades: {len(losing)}')

# Classify each losing trade
buckets = []
for i, t in losing.iterrows():
    entry_z_abs = abs(t['entry_z'])

    if t['exit_reason'] == 'invalidated':
        bucket = 'A'
    elif t['exit_reason'] == 'time_stop':
        # Check if z ever extended beyond entry |z| during holding
        max_z = t['max_adverse_z']
        if not np.isnan(max_z) and max_z > entry_z_abs:
            bucket = 'B'
        else:
            bucket = 'C'
    else:
        # take_profit that still lost (possible with cost)
        # or any other exit — classify based on z extension
        max_z = t['max_adverse_z']
        if not np.isnan(max_z) and max_z > entry_z_abs:
            bucket = 'B'
        else:
            bucket = 'C'

    buckets.append(bucket)

losing['bucket'] = buckets

# Report
print('\n--- LOSS CLASSIFICATION TABLE ---')
print(f'  {"Bucket":8s} {"n":>4s} {"% Losses":>9s} {"Total PnL":>10s} {"Avg Loss":>9s} {"Top Instruments"}')
print(f'  {"-"*8} {"-"*4} {"-"*9} {"-"*10} {"-"*9} {"-"*30}')

bucket_summary = []
for b_name, b_desc in [('A', 'Regime break'), ('B', 'Adverse extension'), ('C', 'Stalled')]:
    b_trades = losing[losing['bucket'] == b_name]
    n = len(b_trades)
    pct = n / len(losing) * 100 if len(losing) > 0 else 0
    total = b_trades['net_pnl'].sum()
    avg = b_trades['net_pnl'].mean() if n > 0 else 0

    # Top instruments
    if n > 0:
        inst_counts = b_trades['instrument'].value_counts().head(3)
        top_inst = ', '.join(f'{inst}({cnt})' for inst, cnt in inst_counts.items())
    else:
        top_inst = '-'

    bucket_summary.append({
        'bucket': b_name, 'desc': b_desc, 'n': n, 'pct': round(pct, 1),
        'total_pnl': round(total, 2), 'avg_loss': round(avg, 2), 'top_instruments': top_inst,
    })

    print(f'  {b_name} ({b_desc:18s}) {n:>4d} {pct:>8.1f}% {total:>10.2f} {avg:>9.2f}  {top_inst}')

# Detailed losing trade log
print('\n--- FULL LOSING TRADE LOG ---')
print(f'  {"Entry":>10s} {"Exit":>10s} {"Inst":>12s} {"Dir":>5s} {"EntZ":>7s} '
      f'{"MaxZ":>7s} {"ExitType":>12s} {"Days":>4s} {"Net":>8s} {"Bkt":>3s}')
for _, t in losing.iterrows():
    mz = f'{t["max_adverse_z"]:.3f}' if not pd.isna(t['max_adverse_z']) else '    nan'
    print(f'  {str(t["entry_date"].date()):>10s} {str(t["exit_date"].date()):>10s} '
          f'{t["instrument"]:>12s} {t["direction"]:>5s} {abs(t["entry_z"]):>7.3f} '
          f'{mz:>7s} {t["exit_reason"]:>12s} {t["days_held"]:>4d} '
          f'{t["net_pnl"]:>+8.2f} {t["bucket"]:>3s}')


# ══════════════════════════════════════════════════════════════
# PART 2 — Z-SCORE STOP-LOSS TEST
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 2: Z-SCORE STOP-LOSS TEST')
print('='*70)

STOP_LOSS_THRESHOLDS = [None, 3.0, 3.5, 4.0]
sl_results = []

for sl_z in STOP_LOSS_THRESHOLDS:
    sl_label = f'No stop' if sl_z is None else f'SL |z|>={sl_z}'
    print(f'\nRunning: {sl_label}...')

    if sl_z is None:
        # Use existing baseline trades
        trades = baseline_trades.copy()
    else:
        trades = run_backtest_extended(df, ALL_9, dur_thresh=10, stop_loss_z=sl_z,
                                       label=sl_label)

    oos = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()
    metrics = compute_metrics_ext(oos, sl_label, df)
    adj_sharpe = compute_daily_portfolio_sharpe(oos, df)
    gate = gate_check(metrics)

    sl_results.append({
        'sl_z': sl_z, 'label': sl_label, 'trades': trades, 'oos': oos,
        'metrics': metrics, 'adj_sharpe': adj_sharpe, 'gate': gate,
    })

    m = metrics
    print(f'  n={m["n_trades"]}, win={m["win_rate"]}%, naive_sharpe={fmt_sharpe(m["sharpe"])}, '
          f'adj_sharpe={fmt_sharpe(adj_sharpe)}, total={m["total_pnl"]}, maxDD={m["max_dd"]}')
    print(f'  TP={m["pct_take_profit"]}%, Inv={m["pct_invalidated"]}%, '
          f'TS={m["pct_time_stop"]}%, SL={m["pct_stop_loss"]}%, gate={gate}')

# Comparison table
print('\n--- STOP-LOSS COMPARISON TABLE ---')
header = (f'  {"Variant":14s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"AdjShrp":>8s} '
          f'{"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} '
          f'{"TP%":>5s} {"Inv%":>5s} {"TS%":>5s} {"SL%":>5s} {"Gate":>22s}')
print(header)
for r in sl_results:
    m = r['metrics']
    print(f'  {r["label"]:14s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
          f'{fmt_sharpe(r["adj_sharpe"]):>8s} '
          f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
          f'{m["max_dd"]:>8.1f} '
          f'{m["pct_take_profit"]:>4.1f}% {m["pct_invalidated"]:>4.1f}% '
          f'{m["pct_time_stop"]:>4.1f}% {m["pct_stop_loss"]:>4.1f}% '
          f'{r["gate"]:>22s}')

# Bucket B catch-rate analysis
print('\n--- BUCKET B CATCH-RATE ANALYSIS ---')
bucket_b = losing[losing['bucket'] == 'B'].copy()
print(f'Total Bucket B trades: {len(bucket_b)}')

if len(bucket_b) > 0:
    print(f'\n  {"SL Threshold":>14s} {"Caught":>7s} {"Missed":>7s} '
          f'{"B_PnL_Base":>11s} {"B_PnL_SL":>11s} {"PnL_Diff":>9s} {"Helped?":>8s}')

    for sl_z in [3.0, 3.5, 4.0]:
        # For each Bucket B trade, check if stop-loss would have fired
        # and what the P&L would have been at the stop-loss point
        caught = 0
        missed = 0
        base_pnl_sum = 0.0
        sl_pnl_sum = 0.0

        for _, bt in bucket_b.iterrows():
            entry_z_abs = abs(bt['entry_z'])
            z_path = bt['z_path']
            entry_spread = bt['entry_spread']
            direction = 1 if bt['direction'] == 'LONG' else -1
            inst = bt['instrument']
            entry_date = bt['entry_date']

            # Find if/when z exceeds sl_z threshold
            trade_days = df[(df['date'] > entry_date) & (df['date'] <= bt['exit_date'])].copy()
            sl_fired = False
            sl_pnl = np.nan

            for _, day_row in trade_days.iterrows():
                z_val = day_row[f'{inst}_z']
                spread_val = day_row[inst]
                if not pd.isna(z_val) and abs(z_val) >= sl_z:
                    # Would have exited here
                    sl_fired = True
                    gross = (spread_val - entry_spread) * direction
                    sl_pnl = gross - ROUNDTRIP_COST_POINTS
                    break

            if sl_fired:
                caught += 1
                base_pnl_sum += bt['net_pnl']
                sl_pnl_sum += sl_pnl
            else:
                missed += 1

        diff = sl_pnl_sum - base_pnl_sum if caught > 0 else 0
        helped = 'YES' if diff > 0 else 'NO'
        print(f'  SL |z|>={sl_z:3.1f}    {caught:>7d} {missed:>7d} '
              f'{base_pnl_sum:>11.2f} {sl_pnl_sum:>11.2f} {diff:>+9.2f} {helped:>8s}')

    # Trade-by-trade detail for Bucket B
    print('\n  Bucket B trade-level detail (what would stop-loss do?):')
    print(f'    {"Entry":>10s} {"Inst":>12s} {"EntZ":>6s} {"MaxZ":>6s} '
          f'{"Base_Net":>9s} {"SL3.0":>8s} {"SL3.5":>8s} {"SL4.0":>8s}')
    for _, bt in bucket_b.iterrows():
        entry_spread = bt['entry_spread']
        direction = 1 if bt['direction'] == 'LONG' else -1
        inst = bt['instrument']
        entry_date = bt['entry_date']
        trade_days = df[(df['date'] > entry_date) & (df['date'] <= bt['exit_date'])].copy()

        sl_pnls = {}
        for sl_z in [3.0, 3.5, 4.0]:
            sl_pnls[sl_z] = np.nan
            for _, day_row in trade_days.iterrows():
                z_val = day_row[f'{inst}_z']
                spread_val = day_row[inst]
                if not pd.isna(z_val) and abs(z_val) >= sl_z:
                    gross = (spread_val - entry_spread) * direction
                    sl_pnls[sl_z] = gross - ROUNDTRIP_COST_POINTS
                    break

        sl30 = f'{sl_pnls[3.0]:>+8.2f}' if not np.isnan(sl_pnls[3.0]) else '     n/a'
        sl35 = f'{sl_pnls[3.5]:>+8.2f}' if not np.isnan(sl_pnls[3.5]) else '     n/a'
        sl40 = f'{sl_pnls[4.0]:>+8.2f}' if not np.isnan(sl_pnls[4.0]) else '     n/a'

        print(f'    {str(bt["entry_date"].date()):>10s} {bt["instrument"]:>12s} '
              f'{abs(bt["entry_z"]):>6.3f} {bt["max_adverse_z"]:>6.3f} '
              f'{bt["net_pnl"]:>+9.2f} {sl30} {sl35} {sl40}')


# ══════════════════════════════════════════════════════════════
# PART 3 — SCALE-IN (PYRAMIDING) TEST
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 3: SCALE-IN (PYRAMIDING) TEST')
print('='*70)

# Pick best stop-loss from Part 2 (best adjusted Sharpe)
best_sl = None
best_adj_sharpe = sl_results[0]['adj_sharpe']  # baseline (no stop)
for r in sl_results[1:]:  # skip baseline
    if r['adj_sharpe'] is not None and not np.isnan(r['adj_sharpe']):
        if best_adj_sharpe is None or np.isnan(best_adj_sharpe) or r['adj_sharpe'] > best_adj_sharpe:
            best_sl = r['sl_z']
            best_adj_sharpe = r['adj_sharpe']

if best_sl is None:
    # None improved on baseline — use 3.0 as default per instructions
    best_sl = 3.0
    print(f'No stop-loss improved adjusted Sharpe. Using default |z|>=3.0 for scale-in test.')
else:
    print(f'Best stop-loss threshold: |z|>={best_sl} (adj Sharpe={best_adj_sharpe:.3f})')

# Run scale-in + stop-loss
print(f'\nRunning scale-in (max 2 lots, scale-in at |z|>=2.0) + SL |z|>={best_sl}...')
scalein_trades = run_backtest_scalein(df, ALL_9, dur_thresh=10,
                                      stop_loss_z=best_sl, scalein_z=2.0,
                                      label=f'SL{best_sl}+ScaleIn')
oos_scalein = scalein_trades[scalein_trades['oos']].reset_index(drop=True) if len(scalein_trades) > 0 else pd.DataFrame()

# Metrics for scale-in
scalein_metrics = compute_metrics_ext(oos_scalein, f'SL{best_sl}+ScaleIn', df)
scalein_adj_sharpe = compute_daily_portfolio_sharpe(oos_scalein, df)
scalein_gate = gate_check(scalein_metrics)

print(f'  n={scalein_metrics["n_trades"]}, win={scalein_metrics["win_rate"]}%, '
      f'naive={fmt_sharpe(scalein_metrics["sharpe"])}, adj={fmt_sharpe(scalein_adj_sharpe)}, '
      f'total={scalein_metrics["total_pnl"]}, maxDD={scalein_metrics["max_dd"]}')

# Find the SL-only result for comparison
sl_only_result = [r for r in sl_results if r['sl_z'] == best_sl]
if sl_only_result:
    sl_only = sl_only_result[0]
else:
    sl_only = sl_results[0]  # fallback to baseline

# 3-way comparison table
print('\n--- 3-WAY COMPARISON: BASELINE vs SL-ONLY vs SL+SCALE-IN ---')
compare_configs = [
    ('No stop/no SI', sl_results[0]['metrics'], sl_results[0]['adj_sharpe'], sl_results[0]['gate']),
    (f'SL|z|>={best_sl} only', sl_only['metrics'], sl_only['adj_sharpe'], sl_only['gate']),
    (f'SL{best_sl}+ScaleIn', scalein_metrics, scalein_adj_sharpe, scalein_gate),
]

header = (f'  {"Variant":18s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"AdjShrp":>8s} '
          f'{"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"Gate":>22s}')
print(header)
for lbl, m, adj, g in compare_configs:
    print(f'  {lbl:18s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
          f'{fmt_sharpe(adj):>8s} '
          f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
          f'{m["max_dd"]:>8.1f} {g:>22s}')

# Scale-in triggered trades analysis
print('\n--- SCALE-IN TRIGGERED TRADES ANALYSIS ---')
if len(oos_scalein) > 0 and 'lots' in oos_scalein.columns:
    si_triggered = oos_scalein[oos_scalein['lots'] == 2].reset_index(drop=True)
    si_single = oos_scalein[oos_scalein['lots'] == 1].reset_index(drop=True)

    print(f'Total OOS trades: {len(oos_scalein)}')
    print(f'Scale-in triggered: {len(si_triggered)} ({len(si_triggered)/len(oos_scalein)*100:.1f}%)')
    print(f'Single-lot only: {len(si_single)}')

    if len(si_triggered) > 0:
        # For each scale-in trade, compare 2-lot P&L vs what 1-lot would have been
        print(f'\n  {"Entry":>10s} {"Inst":>12s} {"Dir":>5s} {"EntZ":>6s} {"SIz":>6s} '
              f'{"1Lot_Net":>9s} {"2Lot_Net":>9s} {"Diff":>8s} {"Exit":>12s}')

        si_total_1lot = 0.0
        si_total_2lot = 0.0
        si_wins = 0
        si_losses = 0

        for _, t in si_triggered.iterrows():
            # 1-lot P&L = gross_pnl_lot1 - 1 * cost
            pnl_1lot = t['gross_pnl_lot1'] - ROUNDTRIP_COST_POINTS
            pnl_2lot = t['net_pnl']  # already includes 2-lot cost
            diff = pnl_2lot - pnl_1lot

            si_total_1lot += pnl_1lot
            si_total_2lot += pnl_2lot
            if diff > 0:
                si_wins += 1
            else:
                si_losses += 1

            si_z = f'{abs(t["scalein_z"]):.3f}' if not pd.isna(t['scalein_z']) else '   n/a'
            print(f'  {str(t["entry_date"].date()):>10s} {t["instrument"]:>12s} '
                  f'{t["direction"]:>5s} {abs(t["entry_z"]):>6.3f} {si_z:>6s} '
                  f'{pnl_1lot:>+9.2f} {pnl_2lot:>+9.2f} {diff:>+8.2f} {t["exit_reason"]:>12s}')

        print(f'\n  Summary: scale-in helped on {si_wins}/{len(si_triggered)} trades, '
              f'hurt on {si_losses}/{len(si_triggered)}')
        print(f'  Total P&L (1-lot): {si_total_1lot:+.2f}')
        print(f'  Total P&L (2-lot): {si_total_2lot:+.2f}')
        print(f'  Marginal P&L from scale-in: {si_total_2lot - si_total_1lot:+.2f}')
    else:
        print('  No scale-in trades triggered.')
else:
    print('  No trades to analyze.')


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
lines.append(f'STAGE 1 — LOSS CLASSIFICATION & STOP-LOSS/SCALE-IN TEST — {timestamp}')
lines.append('=' * 70)
lines.append('')
lines.append('Config: z>1.5, PM L0, dur>=10d, 9 instruments, OOS >= 2022-01-01')
lines.append('Cost: 100 MYR (4.0 pts) round-trip')
lines.append(f'Reference baseline: naive Sharpe 1.812, adjusted 1.546')

# ── Part 1: Loss Classification ──────────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 1: LOSS CLASSIFICATION ---')
lines.append('')
lines.append(f'Total OOS trades: {len(oos_baseline)}, Losing trades: {len(losing)}')
lines.append('')
lines.append('Bucket definitions:')
lines.append('  A (Regime break):      exit_type == invalidated')
lines.append('  B (Adverse extension): exit_type == time_stop AND max|z| during hold > entry|z|')
lines.append('  C (Stalled):           exit_type == time_stop AND max|z| during hold <= entry|z|')
lines.append('')

lines.append(f'  {"Bucket":30s} {"n":>4s} {"% Losses":>9s} {"Total PnL":>10s} {"Avg Loss":>9s}  Top Instruments')
lines.append(f'  {"-"*30} {"-"*4} {"-"*9} {"-"*10} {"-"*9}  {"-"*30}')
for bs in bucket_summary:
    lines.append(f'  {bs["bucket"]} ({bs["desc"]:26s}) {bs["n"]:>4d} {bs["pct"]:>8.1f}% '
                 f'{bs["total_pnl"]:>10.2f} {bs["avg_loss"]:>9.2f}  {bs["top_instruments"]}')

lines.append('')
lines.append('Full losing trade log:')
lines.append(f'  {"Entry":>10s} {"Exit":>10s} {"Instrument":>12s} {"Dir":>5s} {"EntZ":>7s} '
             f'{"MaxZ":>7s} {"ExitType":>12s} {"Days":>4s} {"Net PnL":>8s} {"Bkt":>3s}')
lines.append(f'  {"-"*10} {"-"*10} {"-"*12} {"-"*5} {"-"*7} '
             f'{"-"*7} {"-"*12} {"-"*4} {"-"*8} {"-"*3}')
for _, t in losing.iterrows():
    mz = f'{t["max_adverse_z"]:.3f}' if not pd.isna(t['max_adverse_z']) else '    nan'
    lines.append(f'  {str(t["entry_date"].date()):>10s} {str(t["exit_date"].date()):>10s} '
                 f'{t["instrument"]:>12s} {t["direction"]:>5s} {abs(t["entry_z"]):>7.3f} '
                 f'{mz:>7s} {t["exit_reason"]:>12s} {t["days_held"]:>4d} '
                 f'{t["net_pnl"]:>+8.2f} {t["bucket"]:>3s}')

# ── Part 2: Stop-Loss Comparison ─────────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 2: STOP-LOSS COMPARISON ---')
lines.append('')

lines.append(f'  {"Variant":14s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"AdjShrp":>8s} '
             f'{"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} '
             f'{"TP%":>5s} {"Inv%":>5s} {"TS%":>5s} {"SL%":>5s} {"Gate":>22s}')
lines.append(f'  {"-"*14} {"-"*4} {"-"*6} {"-"*7} {"-"*8} '
             f'{"-"*10} {"-"*8} {"-"*8} {"-"*8} '
             f'{"-"*5} {"-"*5} {"-"*5} {"-"*5} {"-"*22}')
for r in sl_results:
    m = r['metrics']
    lines.append(f'  {r["label"]:14s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
                 f'{fmt_sharpe(r["adj_sharpe"]):>8s} '
                 f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
                 f'{m["max_dd"]:>8.1f} '
                 f'{m["pct_take_profit"]:>4.1f}% {m["pct_invalidated"]:>4.1f}% '
                 f'{m["pct_time_stop"]:>4.1f}% {m["pct_stop_loss"]:>4.1f}% '
                 f'{r["gate"]:>22s}')

# Bucket B catch-rate
lines.append('')
lines.append('Bucket B catch-rate analysis:')
lines.append(f'Total Bucket B trades: {len(bucket_b)}')
lines.append('')

if len(bucket_b) > 0:
    lines.append(f'  {"SL Threshold":>14s} {"Caught":>7s} {"Missed":>7s} '
                 f'{"B_PnL_Base":>11s} {"B_PnL_SL":>11s} {"PnL_Diff":>9s} {"Helped?":>8s}')
    lines.append(f'  {"-"*14} {"-"*7} {"-"*7} {"-"*11} {"-"*11} {"-"*9} {"-"*8}')

    for sl_z in [3.0, 3.5, 4.0]:
        caught = 0
        missed = 0
        base_pnl_sum = 0.0
        sl_pnl_sum = 0.0

        for _, bt in bucket_b.iterrows():
            entry_spread = bt['entry_spread']
            direction = 1 if bt['direction'] == 'LONG' else -1
            inst = bt['instrument']
            entry_date = bt['entry_date']
            trade_days = df[(df['date'] > entry_date) & (df['date'] <= bt['exit_date'])].copy()

            sl_fired = False
            for _, day_row in trade_days.iterrows():
                z_val = day_row[f'{inst}_z']
                spread_val = day_row[inst]
                if not pd.isna(z_val) and abs(z_val) >= sl_z:
                    sl_fired = True
                    gross = (spread_val - entry_spread) * direction
                    sl_pnl = gross - ROUNDTRIP_COST_POINTS
                    break

            if sl_fired:
                caught += 1
                base_pnl_sum += bt['net_pnl']
                sl_pnl_sum += sl_pnl
            else:
                missed += 1

        diff = sl_pnl_sum - base_pnl_sum if caught > 0 else 0
        helped = 'YES' if diff > 0 else 'NO'
        lines.append(f'  SL |z|>={sl_z:3.1f}    {caught:>7d} {missed:>7d} '
                     f'{base_pnl_sum:>11.2f} {sl_pnl_sum:>11.2f} {diff:>+9.2f} {helped:>8s}')

    # Trade-level Bucket B detail
    lines.append('')
    lines.append('  Bucket B trade-level detail:')
    lines.append(f'    {"Entry":>10s} {"Inst":>12s} {"EntZ":>6s} {"MaxZ":>6s} '
                 f'{"Base_Net":>9s} {"SL3.0":>8s} {"SL3.5":>8s} {"SL4.0":>8s}')
    lines.append(f'    {"-"*10} {"-"*12} {"-"*6} {"-"*6} '
                 f'{"-"*9} {"-"*8} {"-"*8} {"-"*8}')

    for _, bt in bucket_b.iterrows():
        entry_spread = bt['entry_spread']
        direction = 1 if bt['direction'] == 'LONG' else -1
        inst = bt['instrument']
        entry_date = bt['entry_date']
        trade_days = df[(df['date'] > entry_date) & (df['date'] <= bt['exit_date'])].copy()

        sl_pnls = {}
        for sl_z in [3.0, 3.5, 4.0]:
            sl_pnls[sl_z] = np.nan
            for _, day_row in trade_days.iterrows():
                z_val = day_row[f'{inst}_z']
                spread_val = day_row[inst]
                if not pd.isna(z_val) and abs(z_val) >= sl_z:
                    gross = (spread_val - entry_spread) * direction
                    sl_pnls[sl_z] = gross - ROUNDTRIP_COST_POINTS
                    break

        sl30 = f'{sl_pnls[3.0]:>+8.2f}' if not np.isnan(sl_pnls[3.0]) else '     n/a'
        sl35 = f'{sl_pnls[3.5]:>+8.2f}' if not np.isnan(sl_pnls[3.5]) else '     n/a'
        sl40 = f'{sl_pnls[4.0]:>+8.2f}' if not np.isnan(sl_pnls[4.0]) else '     n/a'

        lines.append(f'    {str(bt["entry_date"].date()):>10s} {bt["instrument"]:>12s} '
                     f'{abs(bt["entry_z"]):>6.3f} {bt["max_adverse_z"]:>6.3f} '
                     f'{bt["net_pnl"]:>+9.2f} {sl30} {sl35} {sl40}')

# ── Part 3: Scale-In Comparison ──────────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 3: STOP-LOSS + SCALE-IN COMPARISON ---')
lines.append('')
lines.append(f'Best stop-loss from Part 2: |z|>={best_sl}')
lines.append(f'Scale-in rule: add 2nd lot when |z|>=2.0 in same direction (max 2 lots)')
lines.append(f'All lots exit together on first exit condition.')
lines.append('')

lines.append(f'  {"Variant":18s} {"n":>4s} {"Win%":>6s} {"NvShrp":>7s} {"AdjShrp":>8s} '
             f'{"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"Gate":>22s}')
lines.append(f'  {"-"*18} {"-"*4} {"-"*6} {"-"*7} {"-"*8} '
             f'{"-"*10} {"-"*8} {"-"*8} {"-"*8} {"-"*22}')
for lbl, m, adj, g in compare_configs:
    lines.append(f'  {lbl:18s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
                 f'{fmt_sharpe(adj):>8s} '
                 f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
                 f'{m["max_dd"]:>8.1f} {g:>22s}')

# Scale-in triggered analysis
lines.append('')
lines.append('Scale-in triggered trades analysis:')
if len(oos_scalein) > 0 and 'lots' in oos_scalein.columns:
    si_triggered = oos_scalein[oos_scalein['lots'] == 2].reset_index(drop=True)
    lines.append(f'Total OOS trades: {len(oos_scalein)}, Scale-in triggered: {len(si_triggered)} '
                 f'({len(si_triggered)/len(oos_scalein)*100:.1f}%)')

    if len(si_triggered) > 0:
        lines.append('')
        lines.append(f'  {"Entry":>10s} {"Inst":>12s} {"Dir":>5s} {"EntZ":>6s} {"SIz":>6s} '
                     f'{"1Lot_Net":>9s} {"2Lot_Net":>9s} {"Diff":>8s} {"Exit":>12s}')
        lines.append(f'  {"-"*10} {"-"*12} {"-"*5} {"-"*6} {"-"*6} '
                     f'{"-"*9} {"-"*9} {"-"*8} {"-"*12}')

        si_total_1lot = 0.0
        si_total_2lot = 0.0
        si_wins = 0
        si_losses = 0

        for _, t in si_triggered.iterrows():
            pnl_1lot = t['gross_pnl_lot1'] - ROUNDTRIP_COST_POINTS
            pnl_2lot = t['net_pnl']
            diff = pnl_2lot - pnl_1lot
            si_total_1lot += pnl_1lot
            si_total_2lot += pnl_2lot
            if diff > 0:
                si_wins += 1
            else:
                si_losses += 1

            si_z = f'{abs(t["scalein_z"]):.3f}' if not pd.isna(t['scalein_z']) else '   n/a'
            lines.append(f'  {str(t["entry_date"].date()):>10s} {t["instrument"]:>12s} '
                         f'{t["direction"]:>5s} {abs(t["entry_z"]):>6.3f} {si_z:>6s} '
                         f'{pnl_1lot:>+9.2f} {pnl_2lot:>+9.2f} {diff:>+8.2f} {t["exit_reason"]:>12s}')

        lines.append('')
        lines.append(f'  Scale-in helped: {si_wins}/{len(si_triggered)} trades, '
                     f'hurt: {si_losses}/{len(si_triggered)}')
        lines.append(f'  Total P&L (1-lot only): {si_total_1lot:+.2f}')
        lines.append(f'  Total P&L (2-lot):      {si_total_2lot:+.2f}')
        lines.append(f'  Marginal P&L from SI:   {si_total_2lot - si_total_1lot:+.2f}')
    else:
        lines.append('  No scale-in trades triggered.')
else:
    lines.append('  No trades to analyze.')

lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
