"""
MR — TM Regime-Risk Threshold Re-Test (60%/50%) Across Windows
===============================================================
Part 1: Test <60% and <50% thresholds across all 4 windows
Part 2: Selection (worst-case-first + majority-positive check)
Part 3: Logging

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

# Thresholds to test: None = no overlay, then 70%, 60%, 50%
TM_THRESHOLDS = [None, 0.70, 0.60, 0.50]
THRESHOLD_LABELS = {None: 'No-Overlay', 0.70: 'RR<70%', 0.60: 'RR<60%', 0.50: 'RR<50%'}

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
# BACKTEST ENGINE (configurable TM threshold)
# ══════════════════════════════════════════════════════════════

def run_backtest(df, instruments, dur_thresh, z_entry, test_start, test_end,
                 tm_thresh=None):
    """Run backtest. tm_thresh=None disables regime-risk exit overlay."""
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
                # TM regime-risk exit (only if threshold is set and shape unchanged)
                if tm_thresh is not None and shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < tm_thresh:
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


def compute_daily_portfolio_sharpe(trades, test_start, test_end):
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
        prev_spread = t['entry_spread']
        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            cs = day_row[inst]
            if pd.isna(cs):
                continue
            if dt in daily_pnl.index:
                daily_pnl[dt] += (cs - prev_spread) * direction
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
# PART 1 — RE-TEST AT ALL THRESHOLDS ACROSS ALL 4 WINDOWS
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 1: TM REGIME-RISK THRESHOLD RE-TEST')
print('='*70)
print('Base config: z>1.5, dur>=3d, 9 instruments, PM L0, 100 MYR RT cost')

# results[threshold][window_idx] = {adj_sharpe, n_trades, total_pnl, n_rr, rr_pnl, rr_wins, rr_losses, win_rate}
results = {}

for thresh in TM_THRESHOLDS:
    label = THRESHOLD_LABELS[thresh]
    results[thresh] = []
    print(f'\n--- {label} ---')
    for w in WINDOWS:
        trades = run_backtest(df, ALL_9, dur_thresh=3, z_entry=Z_ENTRY,
                              test_start=w['test_start'], test_end=w['test_end'],
                              tm_thresh=thresh)
        adj = compute_daily_portfolio_sharpe(trades, w['test_start'], w['test_end'])
        n = len(trades)
        total_pnl = round(trades['net_pnl'].sum(), 2) if n > 0 else 0.0
        win_rate = round((trades['net_pnl'] > 0).sum() / n * 100, 1) if n > 0 else 0.0

        # Regime-risk exit stats
        if n > 0:
            rr_trades = trades[trades['exit_reason'] == 'regime_risk']
            n_rr = len(rr_trades)
            rr_pnl = round(rr_trades['net_pnl'].sum(), 2) if n_rr > 0 else 0.0
            rr_wins = int((rr_trades['net_pnl'] > 0).sum()) if n_rr > 0 else 0
            rr_losses = int((rr_trades['net_pnl'] <= 0).sum()) if n_rr > 0 else 0
        else:
            n_rr = 0
            rr_pnl = 0.0
            rr_wins = 0
            rr_losses = 0

        results[thresh].append({
            'adj': adj, 'n': n, 'total_pnl': total_pnl, 'win_rate': win_rate,
            'n_rr': n_rr, 'rr_pnl': rr_pnl, 'rr_wins': rr_wins, 'rr_losses': rr_losses,
        })
        print(f'  {w["name"]}: adj={fmt(adj)}, n={n}, win={win_rate}%, PnL={total_pnl}, '
              f'RR_exits={n_rr}, RR_PnL={rr_pnl}, RR_W/L={rr_wins}/{rr_losses}')

# ── Comparison table ──
print('\n' + '='*70)
print('COMPARISON TABLE')
print('='*70)

# Table 1: Adjusted Sharpe
print('\n--- Adjusted Sharpe ---')
header = f'  {"Window":16s}'
for thresh in TM_THRESHOLDS:
    header += f'  {THRESHOLD_LABELS[thresh]:>12s}'
print(header)
print(f'  {"-"*16}' + f'  {"-"*12}' * len(TM_THRESHOLDS))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for thresh in TM_THRESHOLDS:
        row += f'  {fmt(results[thresh][i]["adj"]):>12s}'
    print(row)

# Table 2: Regime-risk exit net PnL
print('\n--- Regime-Risk Exit Net PnL ---')
header = f'  {"Window":16s}'
for thresh in TM_THRESHOLDS:
    if thresh is None:
        header += f'  {"N/A":>12s}'
    else:
        header += f'  {THRESHOLD_LABELS[thresh]:>12s}'
print(header)
print(f'  {"-"*16}' + f'  {"-"*12}' * len(TM_THRESHOLDS))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for thresh in TM_THRESHOLDS:
        if thresh is None:
            row += f'  {"—":>12s}'
        else:
            row += f'  {results[thresh][i]["rr_pnl"]:>12.1f}'
    print(row)

# Table 3: Regime-risk exit count + W/L
print('\n--- Regime-Risk Exit Count (wins/losses) ---')
header = f'  {"Window":16s}'
for thresh in TM_THRESHOLDS:
    if thresh is None:
        header += f'  {"N/A":>12s}'
    else:
        header += f'  {THRESHOLD_LABELS[thresh]:>12s}'
print(header)
print(f'  {"-"*16}' + f'  {"-"*12}' * len(TM_THRESHOLDS))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for thresh in TM_THRESHOLDS:
        if thresh is None:
            row += f'  {"—":>12s}'
        else:
            r = results[thresh][i]
            row += f'  {r["n_rr"]:>3d} ({r["rr_wins"]}W/{r["rr_losses"]}L)'
    print(row)

# Table 4: Total PnL
print('\n--- Total Portfolio PnL ---')
header = f'  {"Window":16s}'
for thresh in TM_THRESHOLDS:
    header += f'  {THRESHOLD_LABELS[thresh]:>12s}'
print(header)
print(f'  {"-"*16}' + f'  {"-"*12}' * len(TM_THRESHOLDS))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for thresh in TM_THRESHOLDS:
        row += f'  {results[thresh][i]["total_pnl"]:>12.1f}'
    print(row)


# ══════════════════════════════════════════════════════════════
# PART 2 — SELECTION
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 2: SELECTION')
print('='*70)

# 2.1 — Worst-case-first robustness
print('\n--- 2.1 Worst-Case-First Selection ---')
thresh_stats = {}
for thresh in TM_THRESHOLDS:
    adjs = [results[thresh][i]['adj'] for i in range(4)]
    worst = min(adjs)
    avg = np.mean(adjs)
    thresh_stats[thresh] = {'worst': worst, 'avg': avg, 'adjs': adjs}
    label = THRESHOLD_LABELS[thresh]
    print(f'  {label:12s}: worst={fmt(worst)}, avg={fmt(avg)}, per-window={[fmt(a) for a in adjs]}')

best_thresh = max(TM_THRESHOLDS, key=lambda t: thresh_stats[t]['worst'])
# Tiebreaker: if worst-cases within 0.05, use avg
candidates = [t for t in TM_THRESHOLDS
              if thresh_stats[t]['worst'] >= thresh_stats[best_thresh]['worst'] - 0.05]
if len(candidates) > 1:
    best_thresh = max(candidates, key=lambda t: thresh_stats[t]['avg'])

print(f'\n  Winner (worst-case-first): {THRESHOLD_LABELS[best_thresh]}')
print(f'    worst-case adj Sharpe = {fmt(thresh_stats[best_thresh]["worst"])}')
print(f'    avg adj Sharpe = {fmt(thresh_stats[best_thresh]["avg"])}')

# 2.2 — Majority-positive exit PnL check
print('\n--- 2.2 Majority-Positive Exit PnL Check ---')
for thresh in [0.70, 0.60, 0.50]:
    label = THRESHOLD_LABELS[thresh]
    positive_windows = sum(1 for i in range(4) if results[thresh][i]['rr_pnl'] > 0)
    neg_windows = 4 - positive_windows
    total_rr_pnl = sum(results[thresh][i]['rr_pnl'] for i in range(4))
    print(f'  {label:8s}: positive in {positive_windows}/4 windows, '
          f'total RR exit PnL = {total_rr_pnl:.1f}')

# 2.3 — Disagreement check
print('\n--- 2.3 Criteria Comparison ---')
# Find majority-positive winner
maj_pos_winner = None
for thresh in [0.70, 0.60, 0.50]:
    positive_windows = sum(1 for i in range(4) if results[thresh][i]['rr_pnl'] > 0)
    if positive_windows >= 3:
        if maj_pos_winner is None:
            maj_pos_winner = thresh

if maj_pos_winner is not None:
    print(f'  Worst-case-first winner: {THRESHOLD_LABELS[best_thresh]}')
    print(f'  Majority-positive winner: {THRESHOLD_LABELS[maj_pos_winner]}')
    if best_thresh == maj_pos_winner:
        print('  Criteria AGREE.')
    elif best_thresh is None and maj_pos_winner is not None:
        print('  Criteria DISAGREE: worst-case prefers no overlay, majority-positive prefers an overlay.')
    else:
        print(f'  Criteria DISAGREE: worst-case prefers {THRESHOLD_LABELS[best_thresh]}, '
              f'majority-positive prefers {THRESHOLD_LABELS[maj_pos_winner]}.')
else:
    print(f'  Worst-case-first winner: {THRESHOLD_LABELS[best_thresh]}')
    print('  Majority-positive winner: NONE (no threshold achieves net-positive in 3+ windows)')
    if best_thresh is None:
        print('  Both criteria point to dropping the overlay.')
    else:
        print(f'  Criteria partially disagree: worst-case keeps {THRESHOLD_LABELS[best_thresh]}, '
              f'but majority-positive test finds no threshold works in 3+ windows.')

# 2.4 — Explicit majority check
print('\n--- 2.4 Does Any Threshold Achieve Net-Positive in 3+ Windows? ---')
any_majority = False
for thresh in [0.70, 0.60, 0.50]:
    label = THRESHOLD_LABELS[thresh]
    positive_windows = sum(1 for i in range(4) if results[thresh][i]['rr_pnl'] > 0)
    status = 'YES' if positive_windows >= 3 else 'NO'
    if positive_windows >= 3:
        any_majority = True
    per_window = [f'{"+" if results[thresh][i]["rr_pnl"] > 0 else "-"}{abs(results[thresh][i]["rr_pnl"]):.0f}'
                  for i in range(4)]
    print(f'  {label:8s}: {positive_windows}/4 positive — {status}  [{", ".join(per_window)}]')

if any_majority:
    print('\n  At least one threshold achieves net-positive in 3+ windows.')
else:
    print('\n  NO threshold achieves net-positive regime-risk exit PnL in 3 or more')
    print('  of the 4 windows. This is a strong signal that the overlay does not')
    print('  generalize — it only worked in the window it was originally tuned on.')


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
lines.append(f'MR — TM REGIME-RISK THRESHOLD RE-TEST (60%/50%) — {timestamp}')
lines.append('=' * 70)
lines.append('')
lines.append('Base config: z>1.5, dur>=3d, 9 instruments, PM L0, 100 MYR RT cost.')
lines.append('Tests whether stricter TM regime-risk thresholds (60%, 50%) perform')
lines.append('more consistently than the original 70%, or whether the overlay')
lines.append('should be dropped entirely.')

# ── Part 1: Full comparison table ──
lines.append('')
lines.append('')
lines.append('--- PART 1: FULL COMPARISON TABLE ---')

# Adjusted Sharpe
lines.append('')
lines.append('Adjusted Sharpe:')
header = f'  {"Window":16s}'
for thresh in TM_THRESHOLDS:
    header += f'  {THRESHOLD_LABELS[thresh]:>12s}'
lines.append(header)
lines.append(f'  {"-"*16}' + f'  {"-"*12}' * len(TM_THRESHOLDS))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for thresh in TM_THRESHOLDS:
        row += f'  {fmt(results[thresh][i]["adj"]):>12s}'
    lines.append(row)

# Total PnL
lines.append('')
lines.append('Total Portfolio PnL:')
header = f'  {"Window":16s}'
for thresh in TM_THRESHOLDS:
    header += f'  {THRESHOLD_LABELS[thresh]:>12s}'
lines.append(header)
lines.append(f'  {"-"*16}' + f'  {"-"*12}' * len(TM_THRESHOLDS))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for thresh in TM_THRESHOLDS:
        row += f'  {results[thresh][i]["total_pnl"]:>12.1f}'
    lines.append(row)

# Win Rate
lines.append('')
lines.append('Win Rate:')
header = f'  {"Window":16s}'
for thresh in TM_THRESHOLDS:
    header += f'  {THRESHOLD_LABELS[thresh]:>12s}'
lines.append(header)
lines.append(f'  {"-"*16}' + f'  {"-"*12}' * len(TM_THRESHOLDS))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for thresh in TM_THRESHOLDS:
        row += f'  {results[thresh][i]["win_rate"]:>11.1f}%'
    lines.append(row)

# Trade count
lines.append('')
lines.append('Trade Count:')
header = f'  {"Window":16s}'
for thresh in TM_THRESHOLDS:
    header += f'  {THRESHOLD_LABELS[thresh]:>12s}'
lines.append(header)
lines.append(f'  {"-"*16}' + f'  {"-"*12}' * len(TM_THRESHOLDS))
for i, w in enumerate(WINDOWS):
    row = f'  {w["name"]:16s}'
    for thresh in TM_THRESHOLDS:
        row += f'  {results[thresh][i]["n"]:>12d}'
    lines.append(row)

# Regime-risk exit details
lines.append('')
lines.append('Regime-Risk Exit Detail:')
lines.append(f'  {"Window":16s}  {"Metric":10s}', )
for thresh in [0.70, 0.60, 0.50]:
    lines.append(f'')
lines.append('')

# Simpler format: one block per threshold
for thresh in [0.70, 0.60, 0.50]:
    label = THRESHOLD_LABELS[thresh]
    lines.append(f'  {label}:')
    lines.append(f'    {"Window":16s}  {"n_exits":>7s}  {"net PnL":>8s}  {"W/L":>7s}')
    lines.append(f'    {"-"*16}  {"-"*7}  {"-"*8}  {"-"*7}')
    for i, w in enumerate(WINDOWS):
        r = results[thresh][i]
        lines.append(f'    {w["name"]:16s}  {r["n_rr"]:>7d}  {r["rr_pnl"]:>8.1f}  '
                     f'{r["rr_wins"]}W/{r["rr_losses"]}L')
    lines.append('')

# ── Part 2: Selection ──
lines.append('')
lines.append('--- PART 2: SELECTION ---')
lines.append('')

# 2.1
lines.append('2.1 — Worst-case-first selection:')
for thresh in TM_THRESHOLDS:
    label = THRESHOLD_LABELS[thresh]
    s = thresh_stats[thresh]
    marker = ' <-- SELECTED' if thresh == best_thresh else ''
    lines.append(f'  {label:12s}: worst={fmt(s["worst"])}, avg={fmt(s["avg"])}{marker}')
lines.append(f'  Winner: {THRESHOLD_LABELS[best_thresh]}')

# 2.2
lines.append('')
lines.append('2.2 — Majority-positive exit PnL check:')
for thresh in [0.70, 0.60, 0.50]:
    label = THRESHOLD_LABELS[thresh]
    positive_windows = sum(1 for i in range(4) if results[thresh][i]['rr_pnl'] > 0)
    total_rr_pnl = sum(results[thresh][i]['rr_pnl'] for i in range(4))
    lines.append(f'  {label:8s}: positive in {positive_windows}/4 windows, '
                 f'total RR exit PnL = {total_rr_pnl:.1f}')

# 2.3
lines.append('')
lines.append('2.3 — Criteria comparison:')
lines.append(f'  Worst-case-first winner: {THRESHOLD_LABELS[best_thresh]}')
if maj_pos_winner is not None:
    lines.append(f'  Majority-positive winner: {THRESHOLD_LABELS[maj_pos_winner]}')
    if best_thresh == maj_pos_winner:
        lines.append('  Criteria AGREE.')
    else:
        lines.append(f'  Criteria DISAGREE.')
else:
    lines.append('  Majority-positive winner: NONE (no threshold net-positive in 3+ windows)')

# 2.4
lines.append('')
lines.append('2.4 — Does any threshold achieve net-positive in 3+ windows?')
for thresh in [0.70, 0.60, 0.50]:
    label = THRESHOLD_LABELS[thresh]
    positive_windows = sum(1 for i in range(4) if results[thresh][i]['rr_pnl'] > 0)
    status = 'YES' if positive_windows >= 3 else 'NO'
    per_window = [f'{"+" if results[thresh][i]["rr_pnl"] > 0 else "-"}{abs(results[thresh][i]["rr_pnl"]):.0f}'
                  for i in range(4)]
    lines.append(f'  {label:8s}: {positive_windows}/4 positive — {status}  [{", ".join(per_window)}]')

if any_majority:
    lines.append('  Result: At least one threshold achieves net-positive in 3+ windows.')
else:
    lines.append('  Result: NO threshold achieves net-positive in 3+ of 4 windows.')

lines.append('')

log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
