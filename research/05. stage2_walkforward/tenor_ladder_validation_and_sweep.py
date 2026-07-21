"""
Tenor-Ladder Integration — Validation + Full-Panel Sensitivity Sweep
MINUTE DATA ONLY, 2024-2026, no daily engine, no W1-W4 splits.

Part 3: Validate generic expiry_exit_due() for all 9 instruments
Part 4: Full-panel entry/exit z-score sensitivity sweep
"""
import sys, os
sys.path.insert(0, r'C:\ClaudeCode')
sys.path.insert(0, r'C:\ClaudeCode\MRBackTest')
os.chdir(r'C:\ClaudeCode')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from datetime import datetime

from MRBackTest.engine.backtest_engine import (
    build_panel, load_intraday_data,
    _run_window_intraday,
    compute_metrics,
    expiry_exit_due, EXPIRY_FORCE_CLOSE_DAY,
    resolve_contracts, get_contract_spread,
    compute_pinned_zscore, build_pinned_episode_history,
    DEFAULT_CONFIG, ALL_9, INTRADAY_START, INTRADAY_END,
    _panel_cache, RESTING_SHAPES,
    get_cost_points, INSTRUMENT_TENOR_OFFSETS,
)
from MRBackTest.shared.tenor_mapping import contract_month_to_str

TEST_START = INTRADAY_START
TEST_END   = INTRADAY_END

print('='*70)
print('TENOR-LADDER INTEGRATION — VALIDATION + FULL-PANEL SWEEP')
print(f'Run date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print(f'Generic expiry rule: force-close when any leg day >= {EXPIRY_FORCE_CLOSE_DAY}')
print(f'  of that leg\'s delivery month (7 days before 15th LTD)')
print(f'Data: 60-min intraday bars, 2024-2026')
print('='*70)

# Build panel
df, tm_cache = build_panel()
contracts = _panel_cache.get('contracts', {})
intraday_df = load_intraday_data()
print()


# ══════════════════════════════════════════════════════════════
# PART 3 — VALIDATION
# ══════════════════════════════════════════════════════════════
print('='*70)
print('PART 3 — VALIDATION (all 9 instruments, minute data)')
print('='*70)

# --- 3.1: Fallback/gap events per instrument ---
# Run the backtest manually to count fallback events per instrument,
# both with the generic expiry_exit_due and without.

def count_fallback_events(df, tm_cache, config, contracts, intraday_df, use_expiry):
    """Run intraday backtest manually for M1-M2 to count fallback events.
    For other instruments, fallback is checked at daily-level exit pricing.
    Returns per-instrument fallback counts."""
    ts = pd.Timestamp(TEST_START)
    te = pd.Timestamp(TEST_END)
    instruments = config['instruments']
    entry_z = config['entry_z']
    exit_z = config['exit_z']
    dur_thresh = config['duration_threshold']
    pm_filter = config['pm_filter_level']
    tm_thresh = config['tm_regime_risk_threshold']
    time_stop = config['time_stop_days']

    daily_mask = (df['date'] >= ts) & (df['date'] <= te)
    daily_rows = df[daily_mask].set_index('date')

    intra = intraday_df[(intraday_df['date'] >= ts) & (intraday_df['date'] <= te)].copy()
    intra_by_date = dict(list(intra.groupby('date')))

    fallback_counts = {}
    for inst in instruments:
        mean_col = f'{inst}_mean'
        std_col = f'{inst}_std'
        fb_count = 0

        position_open = False
        resolved_contracts = None
        entry_date = entry_shape = entry_direction = None
        entry_spread = entry_z_val = None
        days_held = 0

        for date_ts in daily_rows.index:
            day = daily_rows.loc[date_ts]
            if isinstance(day, pd.DataFrame):
                day = day.iloc[-1]

            shape = day['shape']
            pm_level = day.get('pm_level', np.nan)
            daily_mean = day.get(mean_col, np.nan)
            daily_std = day.get(std_col, np.nan)

            if position_open:
                days_held += 1
                exit_reason = None

                if tm_thresh is not None and shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date_ts))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < tm_thresh:
                            exit_reason = 'regime_risk'
                if shape != entry_shape:
                    exit_reason = 'invalidated'
                if days_held >= time_stop:
                    exit_reason = 'time_stop'
                if use_expiry and exit_reason is None and expiry_exit_due(inst, resolved_contracts, date_ts):
                    exit_reason = 'expiry_buffer'

                if exit_reason:
                    # Check if pinned price would be NaN at exit
                    if exit_reason in ('invalidated', 'time_stop', 'regime_risk', 'expiry_buffer'):
                        pinned_price = get_contract_spread(contracts, resolved_contracts, date_ts) if resolved_contracts else np.nan
                        if pd.isna(pinned_price):
                            fb_count += 1
                    position_open = False
                    resolved_contracts = None
                    continue

                # Intraday TP/SL check
                day_bars = intra_by_date.get(date_ts)
                if day_bars is not None and inst in day_bars.columns and not pd.isna(daily_std) and daily_std > 0:
                    for _, bar in day_bars.iterrows():
                        bar_price = bar[inst]
                        if pd.isna(bar_price):
                            continue
                        bar_z = (bar_price - daily_mean) / daily_std
                        if abs(bar_z) < exit_z:
                            position_open = False
                            resolved_contracts = None
                            break

            if not position_open:
                if pd.isna(pm_level) or pd.isna(daily_mean) or pd.isna(daily_std) or daily_std <= 0:
                    continue
                is_resting = shape in RESTING_SHAPES
                dur_ok = day['days_in_shape'] >= dur_thresh if 'days_in_shape' in day.index else False
                pm_ok = pm_level <= pm_filter
                if is_resting and dur_ok and pm_ok:
                    day_bars = intra_by_date.get(date_ts)
                    if day_bars is not None and inst in day_bars.columns:
                        for _, bar in day_bars.iterrows():
                            bar_price = bar[inst]
                            if pd.isna(bar_price):
                                continue
                            bar_z = (bar_price - daily_mean) / daily_std
                            if abs(bar_z) > entry_z:
                                resolved_contracts = resolve_contracts(inst, date_ts)
                                if use_expiry and expiry_exit_due(inst, resolved_contracts, date_ts):
                                    resolved_contracts = None
                                    continue
                                position_open = True
                                entry_date = date_ts
                                entry_spread = bar_price
                                entry_z_val = bar_z
                                entry_shape = shape
                                entry_direction = -1 if bar_z > 0 else 1
                                days_held = 0
                                break

        fallback_counts[inst] = fb_count

    return fallback_counts

config_all9 = DEFAULT_CONFIG.copy()
config_all9['instruments'] = ALL_9

print(f'\n--- 3.1: Fallback Events Per Instrument ---')
print(f'{"Instrument":<12s} {"WITHOUT fix":>12s} {"WITH fix":>10s} {"Delta":>8s}')
print('-'*45)

fb_without = count_fallback_events(df, tm_cache, config_all9, contracts, intraday_df, use_expiry=False)
fb_with = count_fallback_events(df, tm_cache, config_all9, contracts, intraday_df, use_expiry=True)

for inst in ALL_9:
    bef = fb_without[inst]
    aft = fb_with[inst]
    delta = aft - bef
    print(f'{inst:<12s} {bef:>12d} {aft:>10d} {delta:>+8d}')

total_bef = sum(fb_without.values())
total_aft = sum(fb_with.values())
print(f'{"TOTAL":<12s} {total_bef:>12d} {total_aft:>10d} {total_aft - total_bef:>+8d}')

# --- 3.2: Spot-check trades ---
print(f'\n--- 3.2: Spot-Check Trades ---')
trades_with_fix = _run_window_intraday(df, tm_cache, config_all9, intraday_df,
                                         TEST_START, TEST_END, contracts=contracts)

# Spot-check M1-M2 (regression), plus 2 other instruments
spot_check_insts = ['M1-M2', 'BF_M1M2M3', 'M2-M3']
for inst in spot_check_insts:
    inst_trades = trades_with_fix[trades_with_fix['instrument'] == inst] if len(trades_with_fix) > 0 else pd.DataFrame()
    print(f'\n  {inst}: {len(inst_trades)} trades')
    if len(inst_trades) > 0:
        # Show expiry_buffer trades if any, else show last 3
        eb = inst_trades[inst_trades['exit_reason'] == 'expiry_buffer']
        if len(eb) > 0:
            print(f'    Expiry buffer exits: {len(eb)}')
            for _, t in eb.head(3).iterrows():
                print(f'      entry={t["entry_date"].strftime("%Y-%m-%d")}  '
                      f'exit={t["exit_date"].strftime("%Y-%m-%d")}  '
                      f'dir={t["direction"]}  pnl={t["net_pnl"]:+.1f}  hp={t["days_held"]}d')
        else:
            print(f'    No expiry_buffer exits (expected for {inst})')
            for _, t in inst_trades.tail(3).iterrows():
                print(f'      entry={t["entry_date"].strftime("%Y-%m-%d")}  '
                      f'exit={t["exit_date"].strftime("%Y-%m-%d")}  '
                      f'reason={t["exit_reason"]}  pnl={t["net_pnl"]:+.1f}  hp={t["days_held"]}d')

# --- 3.3-3.4: Full-panel result at locked thresholds ---
print(f'\n--- 3.3: Full-Panel Result (locked thresholds, with tenor-ladder) ---')
m_fix = compute_metrics(trades_with_fix, df, TEST_START, TEST_END)

print(f'  Trades:       {m_fix["n_trades"]}')
print(f'  Win rate:     {m_fix["win_rate"]:.1f}%')
print(f'  Total PnL:    {m_fix["total_pnl"]:+.1f} pts')
sh = f'{m_fix["adj_sharpe"]:.3f}' if not pd.isna(m_fix['adj_sharpe']) else 'N/A'
print(f'  Adj Sharpe:   {sh}')
print(f'  Max DD:       {m_fix["max_dd"]:+.1f} pts')
print(f'  Avg Hold:     {m_fix["avg_hp"]:.1f} days')

print(f'\n--- 3.4: Comparison vs Pre-Integration Baseline ---')
print(f'  Pre-integration (M1-M2-only fix): 113 trades, 52.2% win, +408 pts, adj Sharpe 0.693')
print(f'  Post-integration (generic expiry): {m_fix["n_trades"]} trades, {m_fix["win_rate"]:.1f}% win, '
      f'{m_fix["total_pnl"]:+.1f} pts, adj Sharpe {sh}')
if not pd.isna(m_fix['adj_sharpe']):
    delta = m_fix['adj_sharpe'] - 0.693
    print(f'  Delta adj Sharpe: {delta:+.3f}')

# Per-instrument breakdown
print(f'\n--- Per-Instrument (with tenor-ladder) ---')
print(f'{"Instrument":<12s} {"n":>4s} {"win%":>6s} {"PnL":>8s} {"avg":>7s} {"HP":>5s}')
for inst in ALL_9:
    t = trades_with_fix[trades_with_fix['instrument'] == inst] if len(trades_with_fix) > 0 else pd.DataFrame()
    n = len(t)
    if n == 0:
        print(f'{inst:<12s} {0:>4d}')
        continue
    wins = (t['net_pnl'] > 0).sum()
    print(f'{inst:<12s} {n:>4d} {wins/n*100:>5.1f}% {t["net_pnl"].sum():>+8.1f} '
          f'{t["net_pnl"].mean():>+7.1f} {t["days_held"].mean():>5.1f}')

# Exit reason distribution
print(f'\n--- Exit Reasons (all 9, with fix) ---')
if len(trades_with_fix) > 0:
    for reason, count in trades_with_fix['exit_reason'].value_counts().items():
        pct = count / len(trades_with_fix) * 100
        print(f'  {reason:<20s} {count:>4d} ({pct:>5.1f}%)')

print(f'\n{"="*70}')
print(f'PART 3 COMPLETE — review before Part 4')
print(f'{"="*70}')


# ══════════════════════════════════════════════════════════════
# PART 4 — FULL-PANEL ENTRY/EXIT SENSITIVITY SWEEP
# ══════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'PART 4 — FULL-PANEL SENSITIVITY SWEEP (ALL 9, MINUTE DATA)')
print(f'{"="*70}')

entry_grid = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
exit_grid = [0.25, 0.5, 0.75]

print(f'\nEntry grid: {entry_grid}')
print(f'Exit grid:  {exit_grid}')
print(f'Duration:   >= 3 days (held constant)')
print(f'All 9 instruments, minute data, 2024-2026, full period')
print()

# --- 4.1-4.2: Portfolio-level results ---
results = []
per_inst_results = {}  # (entry_z, exit_z) -> per-instrument trades

for entry_z in entry_grid:
    for exit_z_val in exit_grid:
        cfg = DEFAULT_CONFIG.copy()
        cfg['instruments'] = ALL_9
        cfg['entry_z'] = entry_z
        cfg['exit_z'] = exit_z_val

        trades = _run_window_intraday(df, tm_cache, cfg, intraday_df,
                                        TEST_START, TEST_END, contracts=contracts)
        m = compute_metrics(trades, df, TEST_START, TEST_END)

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
        per_inst_results[(entry_z, exit_z_val)] = trades

# Find baseline
baseline_sh = None
for r in results:
    if r['entry_z'] == 1.5 and r['exit_z'] == 0.5:
        baseline_sh = r['adj_sharpe']
        break

print(f'--- 4.2: Portfolio-Level Results ---')
print(f'{"entry_z":>8s} {"exit_z":>8s} {"n":>5s} {"win%":>6s} {"PnL":>9s} {"adjSh":>8s} {"maxDD":>8s} {"avgHP":>6s}  {"note":s}')
print('-'*85)

for r in results:
    sh_str = f'{r["adj_sharpe"]:.3f}' if not pd.isna(r['adj_sharpe']) else 'N/A'
    note = ''
    if r['entry_z'] == 1.5 and r['exit_z'] == 0.5:
        note = '<-- LOCKED BASELINE'
    elif r['n'] < 30:
        note = 'LOW-N WARNING'
    elif baseline_sh is not None and not pd.isna(r['adj_sharpe']):
        if r['adj_sharpe'] > baseline_sh + 0.05:
            note = f'BETTER (+{r["adj_sharpe"] - baseline_sh:.3f})'
        elif r['adj_sharpe'] < baseline_sh - 0.05:
            note = f'WORSE ({r["adj_sharpe"] - baseline_sh:.3f})'
    print(f'{r["entry_z"]:>8.2f} {r["exit_z"]:>8.2f} {r["n"]:>5d} {r["win%"]:>5.1f}% '
          f'{r["pnl"]:>+9.1f} {sh_str:>8s} {r["max_dd"]:>+8.1f} {r["avg_hp"]:>6.1f}  {note}')

# --- 4.3: Per-instrument breakdown for key combos ---
print(f'\n--- 4.3: Per-Instrument Breakdown ---')
key_combos = [(1.25, 0.5), (1.25, 0.75), (1.5, 0.5), (1.75, 0.5), (2.0, 0.5)]
# Only show combos that exist in our grid
key_combos = [(e, x) for e, x in key_combos if (e, x) in per_inst_results]

for entry_z, exit_z_val in key_combos:
    trades = per_inst_results[(entry_z, exit_z_val)]
    combo_label = f'entry={entry_z}/exit={exit_z_val}'
    if entry_z == 1.5 and exit_z_val == 0.5:
        combo_label += ' (BASELINE)'
    print(f'\n  {combo_label}:')
    print(f'  {"Instrument":<12s} {"n":>4s} {"win%":>6s} {"PnL":>8s} {"avg":>7s}')
    for inst in ALL_9:
        t = trades[trades['instrument'] == inst] if len(trades) > 0 else pd.DataFrame()
        n = len(t)
        if n == 0:
            print(f'  {inst:<12s} {0:>4d}')
            continue
        wins = (t['net_pnl'] > 0).sum()
        print(f'  {inst:<12s} {n:>4d} {wins/n*100:>5.1f}% {t["net_pnl"].sum():>+8.1f} {t["net_pnl"].mean():>+7.1f}')

# --- 4.4-4.5: Best candidate and M1-M2 finding check ---
valid_results = [r for r in results if r['n'] >= 30 and not pd.isna(r['adj_sharpe'])]
if not valid_results:
    valid_results = [r for r in results if r['n'] >= 20 and not pd.isna(r['adj_sharpe'])]

if valid_results:
    best = max(valid_results, key=lambda x: x['adj_sharpe'])
    print(f'\n--- 4.5: Best Portfolio Candidate (n>={30 if best["n"] >= 30 else 20}) ---')
    print(f'  entry_z={best["entry_z"]}, exit_z={best["exit_z"]}')
    print(f'  adj Sharpe = {best["adj_sharpe"]:.3f}, n = {best["n"]}, PnL = {best["pnl"]:+.1f}')
    if baseline_sh is not None and not pd.isna(best['adj_sharpe']):
        print(f'  Delta vs locked baseline (1.5/0.5): {best["adj_sharpe"] - baseline_sh:+.3f}')

    # Check if M1-M2 finding holds at portfolio level
    m1m2_best_combo = (1.25, 0.75)
    m1m2_combo_result = next((r for r in results if r['entry_z'] == 1.25 and r['exit_z'] == 0.75), None)
    if m1m2_combo_result and baseline_sh is not None:
        print(f'\n  M1-M2-only finding check (entry 1.25/exit 0.75):')
        print(f'    Portfolio adj Sharpe at 1.25/0.75: '
              f'{m1m2_combo_result["adj_sharpe"]:.3f}' if not pd.isna(m1m2_combo_result['adj_sharpe']) else 'N/A')
        print(f'    Portfolio adj Sharpe at 1.50/0.50: {baseline_sh:.3f}')
        if not pd.isna(m1m2_combo_result['adj_sharpe']):
            holds = m1m2_combo_result['adj_sharpe'] > baseline_sh
            print(f'    M1-M2 finding {"HOLDS" if holds else "DOES NOT HOLD"} at portfolio level '
                  f'({m1m2_combo_result["adj_sharpe"] - baseline_sh:+.3f})')

print(f'\n{"="*70}')
print(f'ALL PARTS COMPLETE')
print(f'{"="*70}')
