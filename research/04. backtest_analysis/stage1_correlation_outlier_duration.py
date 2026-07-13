"""
Stage 1 — Correlation Check, M1-M2 Outlier, Duration x 9-Instrument
=====================================================================
Part 1: 9x9 Signal ON correlation matrix + correlation-adjusted Sharpe
Part 2: M1-M2 average loss outlier investigation
Part 3: 9-instrument pooled at dur>=10d/5d/3d comparison
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
# DATA SETUP (same as stage1_duration_reinstatement.py)
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
# BACKTEST ENGINE (reused from stage1_duration_reinstatement.py)
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


def compute_daily_portfolio_sharpe(trades):
    """Compute Sharpe using daily portfolio P&L (sum of all open positions' daily mark-to-market).
    This naturally accounts for correlation since simultaneous positions are combined."""
    if len(trades) == 0:
        return np.nan

    oos_dates = df[df['date'] >= pd.Timestamp(OOS_START)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=oos_dates)

    for _, t in trades.iterrows():
        entry_dt = t['entry_date']
        exit_dt = t['exit_date']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']

        # Get daily spread values for this trade's holding period
        trade_days = df[(df['date'] > entry_dt) & (df['date'] <= exit_dt)].copy()
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

        # Subtract cost on exit day
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
# PART 1 — 9x9 SIGNAL ON CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 1: 9x9 SIGNAL ON CORRELATION MATRIX (z>1.5, PM L0, dur>=10d)')
print('='*70)

# Build Signal ON indicator per instrument per day (OOS only)
oos_df = df[df['date'] >= pd.Timestamp(OOS_START)].copy().reset_index(drop=True)

signal_on = pd.DataFrame(index=oos_df.index)
for inst in ALL_9:
    z_col = f'{inst}_z'
    on_mask = (
        (oos_df['shape'].isin(['0.0', '1'])) &
        (oos_df['days_in_shape'] >= 10) &
        (oos_df[z_col].abs() > Z_ENTRY) &
        (oos_df['pm_level'] == 0) &
        (oos_df[z_col].notna())
    )
    signal_on[inst] = on_mask.astype(int)

# Correlation matrix
corr_matrix = signal_on.corr()
print('\n9x9 Signal ON day correlation matrix (OOS):')
print(corr_matrix.round(2).to_string())

# High-correlation pairs (>0.3)
print('\nPairs with correlation > 0.3:')
high_corr_pairs = []
for i, inst_a in enumerate(ALL_9):
    for j, inst_b in enumerate(ALL_9):
        if j > i:
            c = corr_matrix.loc[inst_a, inst_b]
            if c > 0.3:
                high_corr_pairs.append((inst_a, inst_b, round(c, 3)))
                print(f'  {inst_a} / {inst_b}: {c:.3f}')
if not high_corr_pairs:
    print('  (none)')

# Day overlap stat
days_on_per_day = signal_on.sum(axis=1)
total_signal_on_days = (days_on_per_day > 0).sum()
days_with_2plus = (days_on_per_day >= 2).sum()
print(f'\nDay overlap: {days_with_2plus} days with 2+ instruments ON simultaneously '
      f'out of {total_signal_on_days} total Signal ON days')
max_simultaneous = days_on_per_day.max()
print(f'Max simultaneous instruments ON: {max_simultaneous}')

# Distribution of simultaneous instruments
print('Distribution of simultaneous instruments ON:')
for n_on in range(1, int(max_simultaneous) + 1):
    count = (days_on_per_day == n_on).sum()
    if count > 0:
        print(f'  {n_on} instrument(s): {count} days')

# 1.3 — Correlation-adjusted Sharpe using daily portfolio P&L (method b)
print('\n--- CORRELATION-ADJUSTED SHARPE (daily portfolio P&L method) ---')

# Run 9-instrument backtest at dur>=10d to get trades
trades_9_10d, per_inst_9_10d, pooled_9_10d = run_backtest(df, ALL_9, dur_thresh=10, label='9-inst dur>=10d')
oos_trades_9_10d = trades_9_10d[trades_9_10d['oos']].reset_index(drop=True) if len(trades_9_10d) > 0 else pd.DataFrame()

naive_sharpe = pooled_9_10d['sharpe']
adjusted_sharpe = compute_daily_portfolio_sharpe(oos_trades_9_10d)

print(f'Naive pooled Sharpe (exit-day attribution):     {fmt_sharpe(naive_sharpe)}')
print(f'Daily portfolio Sharpe (daily MTM, all positions): {fmt_sharpe(adjusted_sharpe)}')
if not np.isnan(naive_sharpe) and not np.isnan(adjusted_sharpe):
    gap = naive_sharpe - adjusted_sharpe
    print(f'Gap: {gap:+.3f} ({gap/naive_sharpe*100:+.1f}%)')


# ══════════════════════════════════════════════════════════════
# PART 2 — M1-M2 AVERAGE LOSS OUTLIER
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 2: M1-M2 AVERAGE LOSS OUTLIER')
print('='*70)

# Get M1-M2 trades from the 9-instrument dur>=10d run
m1m2_trades = oos_trades_9_10d[oos_trades_9_10d['instrument'] == 'M1-M2'].reset_index(drop=True)
m1m2_losses = m1m2_trades[m1m2_trades['net_pnl'] <= 0].reset_index(drop=True)

print(f'\nM1-M2 total trades (OOS): {len(m1m2_trades)}')
print(f'M1-M2 losing trades: {len(m1m2_losses)}')

if len(m1m2_losses) > 0:
    print(f'M1-M2 average loss: {m1m2_losses["net_pnl"].mean():.2f}')
    print(f'\nFull losing trade log:')
    for _, t in m1m2_losses.iterrows():
        print(f'  {t["entry_date"].date()} -> {t["exit_date"].date()} '
              f'{t["direction"]:5s} entry={t["entry_spread"]:>8.1f} exit={t["exit_spread"]:>8.1f} '
              f'z={t["entry_z"]:+.3f}->{t["exit_z"]:+.3f} '
              f'{t["exit_reason"]:12s} days={t["days_held"]:>2d} '
              f'gross={t["gross_pnl"]:>+8.1f} net={t["net_pnl"]:>+8.1f}')

# Compare spread scale across instruments
print('\n--- SPREAD SCALE COMPARISON (OOS period) ---')
oos_mask = df['date'] >= pd.Timestamp(OOS_START)
for inst in ALL_9:
    vals = df.loc[oos_mask, inst].dropna()
    print(f'  {inst:12s}: mean={vals.mean():>8.1f} std={vals.std():>8.1f} '
          f'min={vals.min():>8.1f} max={vals.max():>8.1f} range={vals.max()-vals.min():>8.1f}')

# Compare average loss across all instruments at dur>=10d
print('\n--- AVERAGE LOSS COMPARISON ACROSS ALL 9 INSTRUMENTS (OOS, dur>=10d) ---')
for inst in ALL_9:
    inst_trades = oos_trades_9_10d[oos_trades_9_10d['instrument'] == inst]
    inst_losses = inst_trades[inst_trades['net_pnl'] <= 0]
    if len(inst_losses) > 0:
        avg_l = inst_losses['net_pnl'].mean()
        max_l = inst_losses['net_pnl'].min()
        n_l = len(inst_losses)
        print(f'  {inst:12s}: n_losses={n_l:>2d}, avg_loss={avg_l:>8.2f}, worst_loss={max_l:>8.2f}')
    else:
        print(f'  {inst:12s}: no losses')

# Outlier classification
if len(m1m2_losses) > 0:
    worst_loss = m1m2_losses['net_pnl'].min()
    second_worst = m1m2_losses.nlargest(2, 'net_pnl', keep='all')  # actually want smallest
    sorted_losses = m1m2_losses.sort_values('net_pnl')

    # Check if driven by 1-2 extreme trades
    if len(sorted_losses) >= 2:
        worst_2 = sorted_losses.head(2)['net_pnl'].sum()
        total_loss = sorted_losses['net_pnl'].sum()
        pct_from_worst_2 = worst_2 / total_loss * 100 if total_loss != 0 else 0
        print(f'\nOutlier analysis:')
        print(f'  Worst 2 losses account for {pct_from_worst_2:.1f}% of total loss')

    # Check spread scale issue
    m1m2_std = df.loc[oos_mask, 'M1-M2'].dropna().std()
    other_stds = [df.loc[oos_mask, inst].dropna().std() for inst in CORE_4]
    avg_other_std = np.mean(other_stds)
    ratio = m1m2_std / avg_other_std
    print(f'  M1-M2 spread std: {m1m2_std:.1f} vs avg other calendar pair std: {avg_other_std:.1f} (ratio: {ratio:.2f}x)')

    # Check M1-M2 range vs others
    m1m2_range = df.loc[oos_mask, 'M1-M2'].dropna().max() - df.loc[oos_mask, 'M1-M2'].dropna().min()
    other_ranges = [df.loc[oos_mask, inst].dropna().max() - df.loc[oos_mask, inst].dropna().min() for inst in CORE_4]
    avg_other_range = np.mean(other_ranges)
    range_ratio = m1m2_range / avg_other_range
    print(f'  M1-M2 spread range: {m1m2_range:.0f} vs avg other calendar pair range: {avg_other_range:.0f} (ratio: {range_ratio:.2f}x)')

    # Classify
    if ratio > 2.0:
        classification = 'c'
        classification_text = ('(c) SCALE ISSUE: M1-M2 spread has significantly larger standard deviation '
                               f'({m1m2_std:.1f} vs {avg_other_std:.1f}, {ratio:.1f}x) and range '
                               f'({m1m2_range:.0f} vs {avg_other_range:.0f}, {range_ratio:.1f}x). '
                               'Losses are larger in absolute points because M1-M2 moves in larger increments. '
                               'This is NOT a calculation bug but a genuine scale difference — M1-M2 '
                               'carries more point risk per trade than deferred spreads.')
    elif len(sorted_losses) >= 2 and pct_from_worst_2 > 80:
        classification = 'a'
        classification_text = (f'(a) FEW EXTREME TRADES: worst 2 losses account for {pct_from_worst_2:.1f}% '
                               'of total loss. The average is skewed by outlier trades.')
    else:
        classification = 'b'
        classification_text = ('(b) SYSTEMATIC PATTERN: M1-M2 losses are consistently larger across '
                               'most losing trades, not driven by one or two outliers.')

    print(f'\nClassification: {classification_text}')


# ══════════════════════════════════════════════════════════════
# PART 3 — 9-INSTRUMENT POOLED AT dur>=5d AND dur>=3d
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 3: 9-INSTRUMENT DURATION COMPARISON')
print('='*70)

duration_configs = [
    (10, '9-inst dur>=10d', trades_9_10d, per_inst_9_10d, pooled_9_10d),
]

for dur in [5, 3]:
    label = f'9-inst dur>={dur}d'
    print(f'\nRunning {label}...')
    trades, per_inst, pooled = run_backtest(df, ALL_9, dur_thresh=dur, label=label)
    duration_configs.append((dur, label, trades, per_inst, pooled))

# Summary comparison table
print('\n--- 9-INSTRUMENT POOLED COMPARISON ---')
header = f'  {"Variant":20s} {"n":>5s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"ShpSurv%":>9s} {"Gate":>22s}'
print(header)
for dur, label, trades, per_inst, pooled in duration_configs:
    p = pooled
    g = gate_check(p)
    print(f'  {label:20s} {p["n_trades"]:>5d} {p["win_rate"]:>5.1f}% {fmt_sharpe(p["sharpe"]):>7s} '
          f'{p["total_pnl"]:>10.1f} {p["avg_win"]:>8.2f} {p["avg_loss"]:>8.2f} '
          f'{p["max_dd"]:>8.1f} {p["shape_survival"]:>8.1f}% {g:>22s}')

# Per-instrument breakdown for dur>=5d and dur>=3d
for dur, label, trades, per_inst, pooled in duration_configs:
    if dur == 10:
        continue  # already reported in prior analysis
    oos = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()
    print(f'\n--- PER-INSTRUMENT BREAKDOWN: {label} ---')
    header2 = f'  {"Instrument":12s} {"n":>4s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} {"AvgWin":>8s} {"AvgLoss":>8s} {"ShpSurv%":>9s}'
    print(header2)
    for m in per_inst:
        if m['n_trades'] == 0:
            print(f'  {m["label"]:12s}    0   (no trades)')
            continue
        print(f'  {m["label"]:12s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
              f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
              f'{m["shape_survival"]:>8.1f}%')

# Also compute daily portfolio Sharpe for dur>=5d and dur>=3d
print('\n--- DAILY PORTFOLIO SHARPE (all duration variants) ---')
for dur, label, trades, per_inst, pooled in duration_configs:
    oos = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()
    adj_sharpe = compute_daily_portfolio_sharpe(oos)
    print(f'  {label:20s}: naive={fmt_sharpe(pooled["sharpe"])}, daily_portfolio={fmt_sharpe(adj_sharpe)}')


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
lines.append('='*70)
lines.append(f'STAGE 1 — CORRELATION CHECK, M1-M2 OUTLIER, DURATION x 9-INSTRUMENT — {timestamp}')
lines.append('='*70)

# ── Part 1: Correlation matrix ────────────────────────────────
lines.append('')
lines.append('--- PART 1: 9x9 SIGNAL ON CORRELATION MATRIX (z>1.5, PM L0, dur>=10d, OOS) ---')
lines.append('')
lines.append('Signal ON = resting shape (SB/C) + days_in_shape >= 10 + |z| > 1.5 + PM Level 0')
lines.append('')

# Format correlation matrix
corr_str = corr_matrix.round(2).to_string()
for line in corr_str.split('\n'):
    lines.append(f'  {line}')

lines.append('')
lines.append('High-correlation pairs (>0.3):')
if high_corr_pairs:
    for a, b, c in high_corr_pairs:
        lines.append(f'  {a} / {b}: {c:.3f}')
else:
    lines.append('  (none)')

lines.append('')
lines.append(f'Day overlap: {days_with_2plus} days with 2+ instruments ON simultaneously '
             f'out of {total_signal_on_days} total Signal ON days')
lines.append(f'Max simultaneous instruments ON: {max_simultaneous}')
lines.append('')
lines.append('Distribution of simultaneous instruments ON:')
for n_on in range(1, int(max_simultaneous) + 1):
    count = (days_on_per_day == n_on).sum()
    if count > 0:
        lines.append(f'  {n_on} instrument(s): {count} days')

lines.append('')
lines.append('--- CORRELATION-ADJUSTED SHARPE ---')
lines.append('')
lines.append('Method: daily portfolio P&L (option b)')
lines.append('Sum of all open positions\' daily mark-to-market across all 9 instruments')
lines.append('on each calendar day. Simultaneous positions combine naturally, accounting')
lines.append('for correlation without explicit adjustment.')
lines.append('')
lines.append(f'  Naive pooled Sharpe (exit-day attribution):        {fmt_sharpe(naive_sharpe)}')
lines.append(f'  Daily portfolio Sharpe (daily MTM, all positions): {fmt_sharpe(adjusted_sharpe)}')
if not np.isnan(naive_sharpe) and not np.isnan(adjusted_sharpe):
    gap = naive_sharpe - adjusted_sharpe
    lines.append(f'  Gap: {gap:+.3f} ({gap/naive_sharpe*100:+.1f}%)')

# ── Part 2: M1-M2 outlier ────────────────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 2: M1-M2 AVERAGE LOSS OUTLIER ---')
lines.append('')
lines.append(f'M1-M2 total OOS trades: {len(m1m2_trades)}, losing: {len(m1m2_losses)}')
if len(m1m2_losses) > 0:
    lines.append(f'M1-M2 avg loss: {m1m2_losses["net_pnl"].mean():.2f} points')
lines.append('')
lines.append('Full losing trade log (M1-M2):')
if len(m1m2_losses) > 0:
    lines.append(f'  {"Entry":>10s} {"Exit":>10s} {"Dir":>5s} {"EntSpd":>8s} {"ExSpd":>8s} '
                 f'{"EntZ":>7s} {"ExZ":>7s} {"ExitType":>12s} {"Days":>4s} '
                 f'{"Gross":>8s} {"Net":>8s}')
    lines.append(f'  {"-"*10} {"-"*10} {"-"*5} {"-"*8} {"-"*8} '
                 f'{"-"*7} {"-"*7} {"-"*12} {"-"*4} '
                 f'{"-"*8} {"-"*8}')
    for _, t in m1m2_losses.iterrows():
        ez = f'{t["exit_z"]:+.3f}' if not pd.isna(t['exit_z']) else '    nan'
        lines.append(f'  {str(t["entry_date"].date()):>10s} {str(t["exit_date"].date()):>10s} '
                     f'{t["direction"]:>5s} {t["entry_spread"]:>8.1f} {t["exit_spread"]:>8.1f} '
                     f'{t["entry_z"]:>+7.3f} {ez:>7s} {t["exit_reason"]:>12s} {t["days_held"]:>4d} '
                     f'{t["gross_pnl"]:>+8.1f} {t["net_pnl"]:>+8.1f}')
else:
    lines.append('  (no losses)')

lines.append('')
lines.append('Spread scale comparison (OOS):')
lines.append(f'  {"Instrument":12s} {"Mean":>8s} {"Std":>8s} {"Min":>8s} {"Max":>8s} {"Range":>8s}')
lines.append(f'  {"-"*12} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
for inst in ALL_9:
    vals = df.loc[oos_mask, inst].dropna()
    lines.append(f'  {inst:12s} {vals.mean():>8.1f} {vals.std():>8.1f} '
                 f'{vals.min():>8.1f} {vals.max():>8.1f} {vals.max()-vals.min():>8.1f}')

lines.append('')
lines.append('Average loss comparison (all 9, OOS, dur>=10d):')
lines.append(f'  {"Instrument":12s} {"nLoss":>5s} {"AvgLoss":>8s} {"WorstLoss":>10s}')
lines.append(f'  {"-"*12} {"-"*5} {"-"*8} {"-"*10}')
for inst in ALL_9:
    inst_trades = oos_trades_9_10d[oos_trades_9_10d['instrument'] == inst]
    inst_losses = inst_trades[inst_trades['net_pnl'] <= 0]
    if len(inst_losses) > 0:
        lines.append(f'  {inst:12s} {len(inst_losses):>5d} {inst_losses["net_pnl"].mean():>8.2f} {inst_losses["net_pnl"].min():>10.2f}')
    else:
        lines.append(f'  {inst:12s}     0      n/a        n/a')

lines.append('')
lines.append(f'Classification: {classification_text}')

# ── Part 3: Duration x 9-instrument ──────────────────────────
lines.append('')
lines.append('')
lines.append('--- PART 3: 9-INSTRUMENT POOLED DURATION COMPARISON ---')
lines.append('')
lines.append('Config: z>1.5, PM L0, 100 MYR (4 pts) RT cost, OOS >= 2022-01-01')
lines.append('')
lines.append(f'  {"Variant":20s} {"n":>5s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} '
             f'{"AvgWin":>8s} {"AvgLoss":>8s} {"MaxDD":>8s} {"ShpSurv%":>9s} {"Gate":>22s}')
lines.append(f'  {"-"*20} {"-"*5} {"-"*6} {"-"*7} {"-"*10} '
             f'{"-"*8} {"-"*8} {"-"*8} {"-"*9} {"-"*22}')
for dur, label, trades, per_inst, pooled in duration_configs:
    p = pooled
    g = gate_check(p)
    lines.append(f'  {label:20s} {p["n_trades"]:>5d} {p["win_rate"]:>5.1f}% {fmt_sharpe(p["sharpe"]):>7s} '
                 f'{p["total_pnl"]:>10.1f} {p["avg_win"]:>8.2f} {p["avg_loss"]:>8.2f} '
                 f'{p["max_dd"]:>8.1f} {p["shape_survival"]:>8.1f}% {g:>22s}')

# Daily portfolio Sharpe for all variants
lines.append('')
lines.append('Daily portfolio Sharpe (correlation-adjusted):')
for dur, label, trades, per_inst, pooled in duration_configs:
    oos = trades[trades['oos']].reset_index(drop=True) if len(trades) > 0 else pd.DataFrame()
    adj_sharpe = compute_daily_portfolio_sharpe(oos)
    lines.append(f'  {label:20s}: naive={fmt_sharpe(pooled["sharpe"])}, daily_portfolio={fmt_sharpe(adj_sharpe)}')

# Per-instrument breakdown for dur>=5d and dur>=3d
for dur, label, trades, per_inst, pooled in duration_configs:
    if dur == 10:
        continue
    lines.append('')
    lines.append(f'--- PER-INSTRUMENT: {label} ---')
    lines.append(f'  {"Instrument":12s} {"n":>4s} {"Win%":>6s} {"Sharpe":>7s} {"TotalPnL":>10s} '
                 f'{"AvgWin":>8s} {"AvgLoss":>8s} {"ShpSurv%":>9s}')
    lines.append(f'  {"-"*12} {"-"*4} {"-"*6} {"-"*7} {"-"*10} '
                 f'{"-"*8} {"-"*8} {"-"*9}')
    for m in per_inst:
        if m['n_trades'] == 0:
            lines.append(f'  {m["label"]:12s}    0   (no trades)')
            continue
        lines.append(f'  {m["label"]:12s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {fmt_sharpe(m["sharpe"]):>7s} '
                     f'{m["total_pnl"]:>10.1f} {m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} '
                     f'{m["shape_survival"]:>8.1f}%')

lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
