"""
Fallback Audit: How many trades used rolling-label pricing
due to missing pinned-contract data?
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
from MRBackTest.shared.tenor_mapping import front_month, tenor_to_contract_month, add_months
import os

# --- Constants (match stage2_walkforward.py exactly) ---
PM_CONFIDENCE_THRESHOLD = 0.70
Z_EXIT = 0.5
TIME_STOP_DAYS = 20
ROUNDTRIP_COST_MYR = 100.0
POINT_VALUE = 25.0
ROUNDTRIP_COST_POINTS = ROUNDTRIP_COST_MYR / POINT_VALUE

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

INSTRUMENT_TENOR_OFFSETS = {
    'M1-M2': {'type': 'spread', 'near_offset': 0, 'far_offset': 1},
    'M2-M3': {'type': 'spread', 'near_offset': 1, 'far_offset': 2},
    'M3-M4': {'type': 'spread', 'near_offset': 2, 'far_offset': 3},
    'M4-M5': {'type': 'spread', 'near_offset': 3, 'far_offset': 4},
    'M5-M6': {'type': 'spread', 'near_offset': 4, 'far_offset': 5},
    'BF_M1M2M3': {'type': 'butterfly', 'offsets': [0, 1, 2]},
    'BF_M2M3M4': {'type': 'butterfly', 'offsets': [1, 2, 3]},
    'BF_M3M4M5': {'type': 'butterfly', 'offsets': [2, 3, 4]},
    'BF_M4M5M6': {'type': 'butterfly', 'offsets': [3, 4, 5]},
}

TERM_DIR = r'C:/ClaudeCode/Raw Data/Term Structure'
MONTH_ABBRS_CONTRACT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --- Data loading (same as walkforward) ---
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

# TM predictions (for reference, not used in no-overlay config)
print('Running TM predictions...')
tm_cache = {}
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
    except Exception:
        pass

# Contract prices
def _load_contract_prices():
    contracts = {}
    if not os.path.exists(TERM_DIR):
        return contracts
    for year_dir in sorted(os.listdir(TERM_DIR)):
        if not year_dir.isdigit():
            continue
        year = int(year_dir)
        for m, abbr in enumerate(MONTH_ABBRS_CONTRACT, 1):
            path = os.path.join(TERM_DIR, year_dir, f"FCPO {abbr}{str(year)[2:]}_Daily.csv")
            if not os.path.exists(path):
                continue
            df_c = pd.read_csv(path)
            df_c["date"] = pd.to_datetime(df_c["Timestamp (UTC)"], format='mixed').dt.normalize()
            series = df_c.set_index("date")["Close"]
            contracts[(year, m)] = series[~series.index.duplicated(keep="last")]
    return contracts

print('Loading per-contract prices...')
_contract_prices = _load_contract_prices()
print(f'  Loaded {len(_contract_prices)} contract series')

def _resolve_contracts(inst, date):
    cfg = INSTRUMENT_TENOR_OFFSETS[inst]
    inst_type = cfg['type']
    if inst_type == 'spread':
        near = tenor_to_contract_month(date, cfg['near_offset'], inst_type)
        far = tenor_to_contract_month(date, cfg['far_offset'], inst_type)
        return (near, far)
    else:
        return tuple(tenor_to_contract_month(date, o, inst_type) for o in cfg['offsets'])

def _get_contract_spread(contracts, resolved, date):
    if len(resolved) == 2:
        near_s = contracts.get(resolved[0], pd.Series(dtype=float))
        far_s = contracts.get(resolved[1], pd.Series(dtype=float))
        near_p = near_s.get(date, np.nan) if len(near_s) > 0 else np.nan
        far_p = far_s.get(date, np.nan) if len(far_s) > 0 else np.nan
        if pd.isna(near_p) or pd.isna(far_p):
            return np.nan
        return near_p - far_p
    else:
        prices = []
        for ym in resolved:
            s = contracts.get(ym, pd.Series(dtype=float))
            p = s.get(date, np.nan) if len(s) > 0 else np.nan
            if pd.isna(p):
                return np.nan
            prices.append(p)
        return prices[0] - 2 * prices[1] + prices[2]

def _build_pinned_episode_history(contracts, resolved, df, episode_id, up_to_date):
    ep_mask = df['episode_id'] == episode_id
    ep_dates = df.loc[ep_mask, 'date']
    ep_dates = ep_dates[ep_dates <= up_to_date]
    prices = []
    for d in ep_dates:
        p = _get_contract_spread(contracts, resolved, d)
        if not pd.isna(p):
            prices.append(p)
    return prices

def _compute_pinned_zscore(episode_prices, current_price):
    if pd.isna(current_price) or len(episode_prices) < 10:
        return np.nan
    window = episode_prices[-60:]
    mean = np.mean(window)
    std = np.std(window, ddof=1)
    if std <= 0:
        return np.nan
    return (current_price - mean) / std

# --- INSTRUMENTED BACKTEST ---
def run_backtest_instrumented(df, instruments, dur_thresh, z_entry, test_start, test_end):
    """Same as run_backtest but tracks entry/hold fallback events."""
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    all_trades = []

    for inst in instruments:
        z_col = f'{inst}_z'
        position_open = False
        entry_date = entry_spread = entry_z = entry_shape = entry_direction = None
        days_held = 0
        resolved_contracts = None
        episode_price_history = []
        # Fallback tracking
        entry_fallback = False
        hold_fallback_days = 0
        hold_total_days = 0

        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
            if pd.isna(spread):
                continue

            if position_open:
                days_held += 1
                hold_total_days += 1

                pinned_spread = _get_contract_spread(_contract_prices, resolved_contracts, date)
                if pd.isna(pinned_spread):
                    pinned_spread = spread  # fallback
                    hold_fallback_days += 1

                episode_price_history.append(pinned_spread)
                pinned_z = _compute_pinned_zscore(episode_price_history, pinned_spread)

                exit_reason = None

                # No TM overlay (corrected best config)

                if not pd.isna(pinned_z) and abs(pinned_z) < Z_EXIT:
                    exit_reason = 'take_profit'
                if shape != entry_shape:
                    exit_reason = 'invalidated'
                if days_held >= TIME_STOP_DAYS:
                    exit_reason = 'time_stop'

                if exit_reason:
                    gross_pnl = (pinned_spread - entry_spread) * entry_direction
                    net_pnl = gross_pnl - ROUNDTRIP_COST_POINTS

                    if ts <= date <= te:
                        all_trades.append({
                            'instrument': inst,
                            'entry_date': entry_date, 'exit_date': date,
                            'entry_spread': round(entry_spread, 2),
                            'exit_spread': round(pinned_spread, 2),
                            'entry_z': round(entry_z, 3),
                            'exit_z': round(pinned_z, 3) if not pd.isna(pinned_z) else np.nan,
                            'direction': 'LONG' if entry_direction == 1 else 'SHORT',
                            'days_held': days_held,
                            'exit_reason': exit_reason,
                            'gross_pnl': round(gross_pnl, 2),
                            'net_pnl': round(net_pnl, 2),
                            'pinned_contracts': str(resolved_contracts),
                            'entry_fallback': entry_fallback,
                            'hold_fallback_days': hold_fallback_days,
                            'hold_total_days': hold_total_days,
                            'any_fallback': entry_fallback or hold_fallback_days > 0,
                        })
                    position_open = False
                    resolved_contracts = None
                    episode_price_history = []
                    continue

            if not position_open:
                if pd.isna(z) or pd.isna(row['pm_level']):
                    continue
                is_resting = shape in ('0.0', '1')
                dur_ok = row['days_in_shape'] >= dur_thresh
                z_extreme = abs(z) > z_entry
                pm_ok = row['pm_level'] == 0
                if is_resting and dur_ok and z_extreme and pm_ok:
                    resolved_contracts = _resolve_contracts(inst, date)
                    pinned_entry = _get_contract_spread(_contract_prices, resolved_contracts, date)
                    entry_fallback = pd.isna(pinned_entry)
                    if entry_fallback:
                        pinned_entry = spread

                    episode_price_history = _build_pinned_episode_history(
                        _contract_prices, resolved_contracts, df, row['episode_id'], date)

                    position_open = True
                    entry_date = date
                    entry_spread = pinned_entry
                    entry_z = z
                    entry_shape = shape
                    entry_direction = -1 if z > 0 else 1
                    days_held = 0
                    hold_fallback_days = 0
                    hold_total_days = 0

    return pd.DataFrame(all_trades)


def compute_metrics(trades, label):
    n = len(trades)
    if n == 0:
        return {'label': label, 'n_trades': 0, 'win_rate': 0, 'avg_win': 0,
                'avg_loss': 0, 'total_pnl': 0, 'avg_hp': 0}
    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]
    return {
        'label': label,
        'n_trades': n,
        'win_rate': round(len(wins) / n * 100, 1),
        'avg_win': round(wins['net_pnl'].mean(), 2) if len(wins) > 0 else 0,
        'avg_loss': round(losses['net_pnl'].mean(), 2) if len(losses) > 0 else 0,
        'total_pnl': round(trades['net_pnl'].sum(), 2),
        'avg_hp': round(trades['days_held'].mean(), 1),
    }


def compute_adj_sharpe(trades, test_start, test_end):
    if len(trades) == 0:
        return np.nan
    ts_dt = pd.Timestamp(test_start)
    te_dt = pd.Timestamp(test_end)
    window_dates = df[(df['date'] >= ts_dt) & (df['date'] <= te_dt)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=window_dates)

    for _, t in trades.iterrows():
        entry_dt = t['entry_date']
        exit_dt = t['exit_date']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']

        pinned = None
        if 'pinned_contracts' in t.index and pd.notna(t.get('pinned_contracts')):
            try:
                pinned = eval(t['pinned_contracts'])
            except Exception:
                pass

        trade_days = df[(df['date'] > entry_dt) & (df['date'] <= exit_dt)].copy()
        if len(trade_days) == 0:
            continue

        prev_spread = t['entry_spread']
        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            if pinned and _contract_prices:
                current_spread = _get_contract_spread(_contract_prices, pinned, dt)
                if pd.isna(current_spread):
                    current_spread = day_row[inst]
            else:
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


# === RUN ===
print('\n' + '='*70)
print('FALLBACK AUDIT - PINNED vs ROLLING-LABEL PRICING')
print('='*70)
print('Config: z>1.5, dur>=3d, 9 instruments, PM L0, No TM overlay')
print()

all_window_trades = []

for w in WINDOWS:
    wname = w['name']
    ts, te = w['test_start'], w['test_end']

    trades = run_backtest_instrumented(df, ALL_9, dur_thresh=3, z_entry=1.5,
                                       test_start=ts, test_end=te)
    trades['window'] = wname
    all_window_trades.append(trades)

    n_total = len(trades)
    n_entry_fb = int(trades['entry_fallback'].sum())
    n_any_fb = int(trades['any_fallback'].sum())
    n_pinned = n_total - n_any_fb

    print(f'{wname}: {n_total} trades total')
    print(f'  Entry fallback (pinned price NaN at entry):  {n_entry_fb}')
    print(f'  Any fallback (entry OR hold):                {n_any_fb}')
    print(f'  Fully pinned (no fallback anywhere):         {n_pinned}')

    if n_any_fb > 0:
        fb_trades = trades[trades['any_fallback']]
        avg_fb_days = fb_trades['hold_fallback_days'].mean()
        avg_fb_pct = (fb_trades['hold_fallback_days'] / fb_trades['hold_total_days'].clip(lower=1) * 100).mean()
        print(f'  Avg fallback days per affected trade:         {avg_fb_days:.1f}')
        print(f'  Avg % of hold days using fallback:            {avg_fb_pct:.1f}%')

    # Show which contracts are missing
    if n_any_fb > 0:
        fb_contracts = fb_trades['pinned_contracts'].value_counts().head(5)
        print(f'  Top missing-data contracts:')
        for c, cnt in fb_contracts.items():
            print(f'    {c}: {cnt} trades')
    print()

# Combine all
all_trades = pd.concat(all_window_trades, ignore_index=True)
n_total = len(all_trades)
n_any_fb = int(all_trades['any_fallback'].sum())
n_pinned = n_total - n_any_fb
n_entry_fb = int(all_trades['entry_fallback'].sum())

print('='*70)
print(f'AGGREGATE: {n_total} trades total')
print(f'  Fully pinned:    {n_pinned} ({n_pinned/n_total*100:.1f}%)')
print(f'  Any fallback:    {n_any_fb} ({n_any_fb/n_total*100:.1f}%)')
print(f'  Entry fallback:  {n_entry_fb} ({n_entry_fb/n_total*100:.1f}%)')
print()

# === SPLIT RESULTS ===
print('='*70)
print('WALK-FORWARD RESULTS SPLIT: PINNED-ONLY vs FALLBACK')
print('='*70)
print()

pinned_trades = all_trades[~all_trades['any_fallback']]
fallback_trades = all_trades[all_trades['any_fallback']]

# Per-window split
print(f'{"Window":<18s} {"Group":<12s} {"n":>4s} {"Win%":>6s} {"AvgWin":>8s} {"AvgLoss":>8s} {"TotalPnL":>10s} {"AvgHP":>6s}')
print(f'{"-"*18} {"-"*12} {"-"*4} {"-"*6} {"-"*8} {"-"*8} {"-"*10} {"-"*6}')

for w in WINDOWS:
    wname = w['name']
    ts, te = w['test_start'], w['test_end']

    w_trades = all_trades[all_trades['window'] == wname]
    w_pinned = w_trades[~w_trades['any_fallback']]
    w_fallback = w_trades[w_trades['any_fallback']]

    for group_name, group_trades in [('ALL', w_trades), ('Pinned', w_pinned), ('Fallback', w_fallback)]:
        m = compute_metrics(group_trades, group_name)
        print(f'{wname:<18s} {group_name:<12s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% '
              f'{m["avg_win"]:>8.2f} {m["avg_loss"]:>8.02f} {m["total_pnl"]:>10.1f} {m["avg_hp"]:>5.1f}')
    print()

# Aggregate split
print(f'{"AGGREGATE":<18s} {"Group":<12s} {"n":>4s} {"Win%":>6s} {"AvgWin":>8s} {"AvgLoss":>8s} {"TotalPnL":>10s} {"AvgHP":>6s}')
print(f'{"-"*18} {"-"*12} {"-"*4} {"-"*6} {"-"*8} {"-"*8} {"-"*10} {"-"*6}')
for group_name, group_trades in [('ALL', all_trades), ('Pinned', pinned_trades), ('Fallback', fallback_trades)]:
    m = compute_metrics(group_trades, group_name)
    print(f'{"ALL WINDOWS":<18s} {group_name:<12s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% '
          f'{m["avg_win"]:>8.02f} {m["avg_loss"]:>8.02f} {m["total_pnl"]:>10.1f} {m["avg_hp"]:>5.1f}')

# Per-window adj Sharpe for each group
print()
print('='*70)
print('ADJUSTED SHARPE BY GROUP')
print('='*70)
print()
print(f'{"Window":<18s} {"All AdjSh":>10s} {"Pinned AdjSh":>13s} {"Fallback AdjSh":>15s}')
print(f'{"-"*18} {"-"*10} {"-"*13} {"-"*15}')

for w in WINDOWS:
    wname = w['name']
    ts, te = w['test_start'], w['test_end']

    w_trades = all_trades[all_trades['window'] == wname]
    w_pinned = w_trades[~w_trades['any_fallback']]
    w_fallback = w_trades[w_trades['any_fallback']]

    adj_all = compute_adj_sharpe(w_trades, ts, te)
    adj_pinned = compute_adj_sharpe(w_pinned, ts, te)
    adj_fb = compute_adj_sharpe(w_fallback, ts, te)

    def fs(v):
        return f'{v:.3f}' if v is not None and not np.isnan(v) else '  nan'

    print(f'{wname:<18s} {fs(adj_all):>10s} {fs(adj_pinned):>13s} {fs(adj_fb):>15s}')

# Detail: which instruments have fallback trades?
print()
print('='*70)
print('FALLBACK DETAIL BY INSTRUMENT')
print('='*70)
print()
print(f'{"Instrument":<12s} {"Total":>6s} {"Pinned":>7s} {"Fallback":>9s} {"FB%":>5s} {"Pinned PnL":>11s} {"FB PnL":>8s}')
print(f'{"-"*12} {"-"*6} {"-"*7} {"-"*9} {"-"*5} {"-"*11} {"-"*8}')

for inst in ALL_9:
    inst_all = all_trades[all_trades['instrument'] == inst]
    inst_pinned = inst_all[~inst_all['any_fallback']]
    inst_fb = inst_all[inst_all['any_fallback']]
    n_all = len(inst_all)
    n_p = len(inst_pinned)
    n_f = len(inst_fb)
    fb_pct = n_f / n_all * 100 if n_all > 0 else 0
    pnl_p = inst_pinned['net_pnl'].sum() if n_p > 0 else 0
    pnl_f = inst_fb['net_pnl'].sum() if n_f > 0 else 0
    print(f'{inst:<12s} {n_all:>6d} {n_p:>7d} {n_f:>9d} {fb_pct:>4.1f}% {pnl_p:>11.1f} {pnl_f:>8.1f}')

# Show sample fallback trades
print()
print('='*70)
print('SAMPLE FALLBACK TRADES (first 15)')
print('='*70)
if len(fallback_trades) > 0:
    fb_sample = fallback_trades.head(15)
    for _, t in fb_sample.iterrows():
        print(f'  {t["instrument"]:12s} {t["entry_date"].date()} -> {t["exit_date"].date()} '
              f'held={t["days_held"]}d entry_fb={t["entry_fallback"]} '
              f'hold_fb_days={t["hold_fallback_days"]}/{t["hold_total_days"]} '
              f'PnL={t["net_pnl"]:+.1f} pinned={t["pinned_contracts"]}')
else:
    print('  No fallback trades found.')

print('\nDone.')
