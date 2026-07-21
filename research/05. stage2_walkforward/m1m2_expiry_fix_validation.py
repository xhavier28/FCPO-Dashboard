"""
M1-M2 Expiry Buffer Fix — Validation + Result Check + Sensitivity Sweep

Part 3: Validate fallback events eliminated
Part 4: M1-M2-only results at locked thresholds
Part 5: Entry/exit z-score sensitivity sweep (M1-M2 only)
"""
import sys, os
sys.path.insert(0, r'C:\ClaudeCode')
sys.path.insert(0, r'C:\ClaudeCode\MRBackTest')
os.chdir(r'C:\ClaudeCode')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy

from MRBackTest.engine.backtest_engine import (
    build_panel, load_contract_prices, resolve_contracts,
    get_contract_spread, compute_pinned_zscore, build_pinned_episode_history,
    m1m2_expiry_exit_due, M1M2_FORCE_CLOSE_DAY,
    compute_metrics, _compute_daily_portfolio_sharpe,
    DEFAULT_CONFIG, ALL_9, WINDOWS, RESTING_SHAPES,
    _panel_cache, _run_window_daily,
    INSTRUMENT_TENOR_OFFSETS, PM_CONFIDENCE_THRESHOLD,
    get_cost_points,
)

print('='*70)
print('M1-M2 EXPIRY BUFFER FIX — VALIDATION + RESULT CHECK + SENSITIVITY')
print(f'Run date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print(f'Buffer rule: force-close M1-M2 on day >= {M1M2_FORCE_CLOSE_DAY} of near-leg delivery month')
print('='*70)

# Build panel
df, tm_cache = build_panel()
contracts = _panel_cache.get('contracts', {})
print()

# ══════════════════════════════════════════════════════════════
# PART 3 — FALLBACK VALIDATION
# ══════════════════════════════════════════════════════════════
print('='*70)
print('PART 3 — FALLBACK VALIDATION')
print('='*70)

# Run M1-M2 only across ALL windows with the fix applied
config_m1m2 = DEFAULT_CONFIG.copy()
config_m1m2['instruments'] = ['M1-M2']

# Collect trades + track fallback events
all_m1m2_trades = []
fallback_events_new = 0
total_hold_days_new = 0

for w in WINDOWS:
    trades = _run_window_daily(df, tm_cache, config_m1m2,
                                w['test_start'], w['test_end'],
                                contracts=contracts)
    if len(trades) > 0:
        trades['window'] = w['name']
        all_m1m2_trades.append(trades)

if all_m1m2_trades:
    all_m1m2_df = pd.concat(all_m1m2_trades, ignore_index=True)
else:
    all_m1m2_df = pd.DataFrame()

print(f'\nM1-M2 trades (all windows, with fix): {len(all_m1m2_df)}')

# Now re-run WITHOUT the fix to count fallback events for comparison.
# We do this by running a manual loop that mirrors _run_window_daily
# but instruments the fallback counting.

def count_fallback_events_m1m2(df, tm_cache, config, contracts):
    """Run M1-M2 backtest manually, counting fallback events.
    Returns (trades_list, fallback_entry_count, fallback_hold_count, fallback_details)."""
    inst = 'M1-M2'
    z_col = f'{inst}_z'
    entry_z = config['entry_z']
    exit_z = config['exit_z']
    dur_thresh = config['duration_threshold']
    pm_filter = config['pm_filter_level']
    tm_thresh = config['tm_regime_risk_threshold']
    time_stop = config['time_stop_days']
    cost_per_lot = get_cost_points(inst, config)

    fb_entry = 0
    fb_hold = 0
    fb_hold_total = 0
    fb_details = []

    position_open = False
    entry_date = entry_spread = entry_z_val = entry_shape = entry_direction = None
    days_held = 0
    resolved_contracts = None
    episode_price_history = []

    for idx in df.index:
        row = df.loc[idx]
        date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
        if pd.isna(spread):
            continue

        if position_open:
            days_held += 1
            fb_hold_total += 1

            pinned_spread = get_contract_spread(contracts, resolved_contracts, date)
            if pd.isna(pinned_spread):
                fb_hold += 1
                pinned_spread = spread
                fb_details.append({
                    'type': 'hold', 'date': date, 'entry_date': entry_date,
                    'resolved': str(resolved_contracts),
                })

            episode_price_history.append(pinned_spread)
            pinned_z = compute_pinned_zscore(episode_price_history, pinned_spread)

            exit_reason = None
            if tm_thresh is not None and shape == entry_shape:
                tm_data = tm_cache.get(pd.Timestamp(date))
                if tm_data is not None:
                    pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                    if not np.isnan(pp) and pp < tm_thresh:
                        exit_reason = 'regime_risk'
            if not pd.isna(pinned_z) and abs(pinned_z) < exit_z:
                exit_reason = 'take_profit'
            if shape != entry_shape:
                exit_reason = 'invalidated'
            if days_held >= time_stop:
                exit_reason = 'time_stop'
            # Apply expiry buffer
            if exit_reason is None and m1m2_expiry_exit_due(inst, resolved_contracts, date):
                exit_reason = 'expiry_buffer'

            if exit_reason:
                position_open = False
                resolved_contracts = None
                episode_price_history = []
                continue

        if not position_open:
            if pd.isna(z) or pd.isna(row['pm_level']):
                continue
            is_resting = shape in RESTING_SHAPES
            dur_ok = row['days_in_shape'] >= dur_thresh
            z_extreme = abs(z) > entry_z
            pm_ok = row['pm_level'] <= pm_filter
            if is_resting and dur_ok and z_extreme and pm_ok:
                resolved_contracts = resolve_contracts(inst, date)
                if m1m2_expiry_exit_due(inst, resolved_contracts, date):
                    resolved_contracts = None
                    continue
                pinned_entry = get_contract_spread(contracts, resolved_contracts, date)
                if pd.isna(pinned_entry):
                    fb_entry += 1
                    pinned_entry = spread
                    fb_details.append({
                        'type': 'entry', 'date': date,
                        'resolved': str(resolved_contracts),
                    })
                episode_price_history = build_pinned_episode_history(
                    contracts, resolved_contracts, df, row['episode_id'], date)
                position_open = True
                entry_date = date
                entry_spread = pinned_entry
                entry_z_val = z
                entry_shape = shape
                entry_direction = -1 if z > 0 else 1
                days_held = 0

    return fb_entry, fb_hold, fb_hold_total, fb_details

fb_entry, fb_hold, fb_hold_total, fb_details = count_fallback_events_m1m2(
    df, tm_cache, config_m1m2, contracts)

print(f'\nFallback events WITH fix applied:')
print(f'  Entry fallbacks:  {fb_entry}')
print(f'  Hold fallbacks:   {fb_hold} / {fb_hold_total} hold-days')
print(f'  Total:            {fb_entry + fb_hold}')
print(f'  (Prior count was 488 M1-M2 fallback events without fix)')

# Spot-check: show details of remaining fallbacks (if any)
if fb_details:
    print(f'\n--- Remaining Fallback Details (first 10) ---')
    for d in fb_details[:10]:
        print(f'  {d["type"]:5s}  {d["date"]}  contracts={d["resolved"]}')
else:
    print(f'\n  ALL M1-M2 fallback events eliminated.')

# Also run WITHOUT the fix to show the comparison
def count_fallback_events_m1m2_nofix(df, tm_cache, config, contracts):
    """Same as above but WITHOUT the expiry buffer exit."""
    inst = 'M1-M2'
    z_col = f'{inst}_z'
    entry_z = config['entry_z']
    exit_z = config['exit_z']
    dur_thresh = config['duration_threshold']
    pm_filter = config['pm_filter_level']
    tm_thresh = config['tm_regime_risk_threshold']
    time_stop = config['time_stop_days']

    fb_entry = 0
    fb_hold = 0

    position_open = False
    entry_date = entry_spread = entry_z_val = entry_shape = entry_direction = None
    days_held = 0
    resolved_contracts = None
    episode_price_history = []

    for idx in df.index:
        row = df.loc[idx]
        date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
        if pd.isna(spread):
            continue

        if position_open:
            days_held += 1
            pinned_spread = get_contract_spread(contracts, resolved_contracts, date)
            if pd.isna(pinned_spread):
                fb_hold += 1
                pinned_spread = spread
            episode_price_history.append(pinned_spread)
            pinned_z = compute_pinned_zscore(episode_price_history, pinned_spread)

            exit_reason = None
            if tm_thresh is not None and shape == entry_shape:
                tm_data = tm_cache.get(pd.Timestamp(date))
                if tm_data is not None:
                    pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                    if not np.isnan(pp) and pp < tm_thresh:
                        exit_reason = 'regime_risk'
            if not pd.isna(pinned_z) and abs(pinned_z) < exit_z:
                exit_reason = 'take_profit'
            if shape != entry_shape:
                exit_reason = 'invalidated'
            if days_held >= time_stop:
                exit_reason = 'time_stop'
            # NO expiry buffer here
            if exit_reason:
                position_open = False
                resolved_contracts = None
                episode_price_history = []
                continue

        if not position_open:
            if pd.isna(z) or pd.isna(row['pm_level']):
                continue
            is_resting = shape in RESTING_SHAPES
            dur_ok = row['days_in_shape'] >= dur_thresh
            z_extreme = abs(z) > entry_z
            pm_ok = row['pm_level'] <= pm_filter
            if is_resting and dur_ok and z_extreme and pm_ok:
                resolved_contracts = resolve_contracts(inst, date)
                # NO entry block here
                pinned_entry = get_contract_spread(contracts, resolved_contracts, date)
                if pd.isna(pinned_entry):
                    fb_entry += 1
                    pinned_entry = spread
                episode_price_history = build_pinned_episode_history(
                    contracts, resolved_contracts, df, row['episode_id'], date)
                position_open = True
                entry_date = date
                entry_spread = pinned_entry
                entry_z_val = z
                entry_shape = shape
                entry_direction = -1 if z > 0 else 1
                days_held = 0

    return fb_entry, fb_hold

fb_entry_old, fb_hold_old = count_fallback_events_m1m2_nofix(df, tm_cache, config_m1m2, contracts)
print(f'\nFallback events WITHOUT fix (baseline):')
print(f'  Entry fallbacks:  {fb_entry_old}')
print(f'  Hold fallbacks:   {fb_hold_old}')
print(f'  Total:            {fb_entry_old + fb_hold_old}')

# Spot-check: show exit reasons for trades that now get expiry_buffer
print(f'\n--- Exit Reason Distribution (M1-M2, with fix) ---')
if len(all_m1m2_df) > 0:
    for reason, count in all_m1m2_df['exit_reason'].value_counts().items():
        print(f'  {reason:<20s} {count:>4d}')
    expiry_trades = all_m1m2_df[all_m1m2_df['exit_reason'] == 'expiry_buffer']
    print(f'\n--- Expiry Buffer Trades (spot-check, first 10) ---')
    if len(expiry_trades) > 0:
        for _, t in expiry_trades.head(10).iterrows():
            print(f'  {t["window"]}  entry={t["entry_date"].strftime("%Y-%m-%d")}  '
                  f'exit={t["exit_date"].strftime("%Y-%m-%d")}  '
                  f'dir={t["direction"]}  pnl={t["net_pnl"]:+.1f}  '
                  f'contracts={t.get("pinned_contracts", "?")}')
    else:
        print(f'  (none)')

print(f'\n{"="*70}')
print(f'PART 3 COMPLETE — review above before Part 4')
print(f'{"="*70}')

# ══════════════════════════════════════════════════════════════
# PART 4 — M1-M2 ONLY RESULT CHECK AT LOCKED THRESHOLDS
# ══════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'PART 4 — M1-M2 ONLY RESULT CHECK (locked thresholds)')
print(f'{"="*70}')

# Per-window M1-M2 metrics
print(f'\n{"Window":<20s} {"n":>4s} {"win%":>6s} {"PnL":>8s} {"adjSh":>8s} {"maxDD":>8s} {"avgHP":>6s}')
print('-'*60)
for w in WINDOWS:
    wt = all_m1m2_df[all_m1m2_df['window'] == w['name']] if len(all_m1m2_df) > 0 else pd.DataFrame()
    m = compute_metrics(wt, df, w['test_start'], w['test_end'])
    sh_str = f'{m["adj_sharpe"]:.3f}' if not pd.isna(m['adj_sharpe']) else 'N/A'
    print(f'{w["label"]:<20s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {m["total_pnl"]:>+8.1f} '
          f'{sh_str:>8s} {m["max_dd"]:>+8.1f} {m["avg_hp"]:>6.1f}')

# Full-period M1-M2
if len(all_m1m2_df) > 0:
    full_m = compute_metrics(all_m1m2_df, df, '2019-01-01', '2026-12-31')
    sh_str = f'{full_m["adj_sharpe"]:.3f}' if not pd.isna(full_m['adj_sharpe']) else 'N/A'
    print(f'{"FULL (2019-2026)":<20s} {full_m["n_trades"]:>4d} {full_m["win_rate"]:>5.1f}% '
          f'{full_m["total_pnl"]:>+8.1f} {sh_str:>8s} {full_m["max_dd"]:>+8.1f} {full_m["avg_hp"]:>6.1f}')

print(f'\n--- Exit Reason Breakdown ---')
if len(all_m1m2_df) > 0:
    for reason, count in all_m1m2_df['exit_reason'].value_counts().items():
        pct = count / len(all_m1m2_df) * 100
        print(f'  {reason:<20s} {count:>4d} ({pct:>5.1f}%)')

print(f'\n{"="*70}')
print(f'PART 4 COMPLETE — review above before Part 5')
print(f'{"="*70}')

# ══════════════════════════════════════════════════════════════
# PART 5 — ENTRY/EXIT Z-SCORE SENSITIVITY SWEEP (M1-M2 ONLY)
# ══════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'PART 5 — ENTRY/EXIT Z-SCORE SENSITIVITY SWEEP (M1-M2 ONLY)')
print(f'{"="*70}')

entry_grid = [1.25, 1.5, 1.75, 2.0]
exit_grid = [0.25, 0.5, 0.75]

print(f'\nEntry grid: {entry_grid}')
print(f'Exit grid:  {exit_grid}')
print(f'Duration:   >= 3 days (held constant)')
print(f'Period:     Full 2019-2026 (all 4 windows combined)')
print()

results = []
for entry_z in entry_grid:
    for exit_z_val in exit_grid:
        cfg = DEFAULT_CONFIG.copy()
        cfg['instruments'] = ['M1-M2']
        cfg['entry_z'] = entry_z
        cfg['exit_z'] = exit_z_val

        all_trades = []
        for w in WINDOWS:
            trades = _run_window_daily(df, tm_cache, cfg,
                                        w['test_start'], w['test_end'],
                                        contracts=contracts)
            if len(trades) > 0:
                all_trades.append(trades)

        if all_trades:
            combined = pd.concat(all_trades, ignore_index=True)
        else:
            combined = pd.DataFrame()

        m = compute_metrics(combined, df, '2019-01-01', '2026-12-31')
        results.append({
            'entry_z': entry_z,
            'exit_z': exit_z_val,
            'n': m['n_trades'],
            'win%': m['win_rate'],
            'pnl': m['total_pnl'],
            'adj_sharpe': m['adj_sharpe'],
            'max_dd': m['max_dd'],
            'avg_hp': m['avg_hp'],
        })

# Print table
print(f'{"entry_z":>8s} {"exit_z":>8s} {"n":>5s} {"win%":>6s} {"PnL":>9s} {"adjSh":>8s} {"maxDD":>8s} {"avgHP":>6s}  {"note":s}')
print('-'*75)

# Find baseline for comparison
baseline_sh = None
for r in results:
    if r['entry_z'] == 1.5 and r['exit_z'] == 0.5:
        baseline_sh = r['adj_sharpe']
        break

for r in results:
    sh_str = f'{r["adj_sharpe"]:.3f}' if not pd.isna(r['adj_sharpe']) else 'N/A'
    note = ''
    if r['entry_z'] == 1.5 and r['exit_z'] == 0.5:
        note = '<-- LOCKED BASELINE'
    elif r['n'] < 20:
        note = 'LOW-N WARNING'
    elif baseline_sh is not None and not pd.isna(r['adj_sharpe']):
        if r['adj_sharpe'] > baseline_sh + 0.1:
            note = f'BETTER (+{r["adj_sharpe"] - baseline_sh:.3f})'
    print(f'{r["entry_z"]:>8.2f} {r["exit_z"]:>8.2f} {r["n"]:>5d} {r["win%"]:>5.1f}% '
          f'{r["pnl"]:>+9.1f} {sh_str:>8s} {r["max_dd"]:>+8.1f} {r["avg_hp"]:>6.1f}  {note}')

# Also run per-window for the best candidate(s)
best = max([r for r in results if r['n'] >= 20], key=lambda x: x['adj_sharpe'] if not pd.isna(x['adj_sharpe']) else -999)
print(f'\n--- Best Candidate (n>=20): entry_z={best["entry_z"]}, exit_z={best["exit_z"]} ---')
print(f'  adj Sharpe = {best["adj_sharpe"]:.3f}, n = {best["n"]}, PnL = {best["pnl"]:+.1f}')

if best['entry_z'] != 1.5 or best['exit_z'] != 0.5:
    # Run per-window breakdown for best candidate
    cfg_best = DEFAULT_CONFIG.copy()
    cfg_best['instruments'] = ['M1-M2']
    cfg_best['entry_z'] = best['entry_z']
    cfg_best['exit_z'] = best['exit_z']

    print(f'\n  Per-window breakdown:')
    print(f'  {"Window":<20s} {"n":>4s} {"win%":>6s} {"PnL":>8s} {"adjSh":>8s}')
    for w in WINDOWS:
        trades = _run_window_daily(df, tm_cache, cfg_best,
                                    w['test_start'], w['test_end'],
                                    contracts=contracts)
        m = compute_metrics(trades, df, w['test_start'], w['test_end'])
        sh_str = f'{m["adj_sharpe"]:.3f}' if not pd.isna(m['adj_sharpe']) else 'N/A'
        print(f'  {w["label"]:<20s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% {m["total_pnl"]:>+8.1f} {sh_str:>8s}')

print(f'\n{"="*70}')
print(f'ALL PARTS COMPLETE')
print(f'{"="*70}')
