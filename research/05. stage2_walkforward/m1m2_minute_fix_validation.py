"""
M1-M2 Expiry Buffer Fix — MINUTE DATA Validation + Result Check + Sensitivity

All results are on 60-min intraday data, 2024-2026 period only.
No daily engine, no W1-W4 window splits.

Part 3: Validate fallback events eliminated on minute data
Part 4: M1-M2-only results at locked thresholds (minute data)
Part 5: Entry/exit z-score sensitivity sweep (M1-M2 only, minute data)
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
    run_backtest_intraday, _run_window_intraday,
    compute_metrics,
    m1m2_expiry_exit_due, M1M2_FORCE_CLOSE_DAY,
    resolve_contracts, get_contract_spread,
    DEFAULT_CONFIG, ALL_9, INTRADAY_START, INTRADAY_END,
    _panel_cache, RESTING_SHAPES,
    get_cost_points,
)

TEST_START = INTRADAY_START  # '2024-01-01'
TEST_END   = INTRADAY_END    # '2026-12-31'

print('='*70)
print('M1-M2 EXPIRY BUFFER FIX — MINUTE DATA (2024-2026)')
print(f'Run date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print(f'Buffer rule: force-close M1-M2 on day >= {M1M2_FORCE_CLOSE_DAY} of near-leg delivery month')
print(f'Data: 60-min intraday bars only, NO daily engine')
print('='*70)

# Build panel (needed for daily z-score baseline, PM, TM, shape)
df, tm_cache = build_panel()
contracts = _panel_cache.get('contracts', {})
intraday_df = load_intraday_data()
print()

# ══════════════════════════════════════════════════════════════
# PART 3 — FALLBACK VALIDATION ON MINUTE DATA
# ══════════════════════════════════════════════════════════════
print('='*70)
print('PART 3 — FALLBACK VALIDATION (MINUTE DATA)')
print('='*70)

# Run M1-M2 only WITH fix
config_m1m2 = DEFAULT_CONFIG.copy()
config_m1m2['instruments'] = ['M1-M2']

trades_with_fix = _run_window_intraday(df, tm_cache, config_m1m2, intraday_df,
                                         TEST_START, TEST_END, contracts=contracts)
print(f'\nM1-M2 trades WITH fix: {len(trades_with_fix)}')

if len(trades_with_fix) > 0:
    print(f'\n--- Exit Reason Distribution ---')
    for reason, count in trades_with_fix['exit_reason'].value_counts().items():
        pct = count / len(trades_with_fix) * 100
        print(f'  {reason:<20s} {count:>4d} ({pct:>5.1f}%)')

    expiry_trades = trades_with_fix[trades_with_fix['exit_reason'] == 'expiry_buffer']
    print(f'\n--- Expiry Buffer Trades (spot-check, first 10) ---')
    if len(expiry_trades) > 0:
        for _, t in expiry_trades.head(10).iterrows():
            entry_dt = t.get('entry_datetime', t['entry_date'])
            print(f'  entry={t["entry_date"].strftime("%Y-%m-%d")}  '
                  f'exit={t["exit_date"].strftime("%Y-%m-%d")}  '
                  f'dir={t["direction"]}  pnl={t["net_pnl"]:+.1f}  hp={t["days_held"]}d')
    else:
        print(f'  (none)')

# To count fallback events, we need to check the intraday engine path.
# The intraday engine uses tenor-labeled bars (not pinned) for entry/exit prices,
# and only uses pinned contracts for daily-level exit P&L.
# "Fallback" in the intraday context means: daily-level exit tried pinned contract
# price, got NaN, fell back to tenor-labeled or last intraday bar.
# We can detect this by examining whether the exit happened on a date where
# the pinned contract price would have been NaN.

print(f'\n--- Fallback Analysis ---')
if len(trades_with_fix) > 0:
    fb_count = 0
    for _, t in trades_with_fix.iterrows():
        exit_date = t['exit_date']
        # Check if pinned contract price exists on exit date
        resolved = resolve_contracts('M1-M2', t['entry_date'])
        pinned_price = get_contract_spread(contracts, resolved, exit_date)
        if pd.isna(pinned_price) and t['exit_reason'] in ('invalidated', 'time_stop', 'regime_risk', 'expiry_buffer'):
            fb_count += 1
    print(f'  Trades where daily-level exit had NaN pinned price: {fb_count}')
    print(f'  (Intraday TP/SL exits use bar price directly, not pinned)')
else:
    print(f'  No trades to check.')

# Run WITHOUT fix for comparison (temporarily bypass the expiry buffer)
# We do this by running with a config that includes M1-M2 but patching the
# m1m2_expiry_exit_due function. Simpler: just note that the "without fix"
# numbers come from the prior run (adj Sharpe 0.675 for all 9 instruments).
# For a clean M1-M2-only count, let's check what the prior all-9 run produced
# for M1-M2 specifically.

# Run all-9 WITH fix to get the comparison baseline
config_all9 = DEFAULT_CONFIG.copy()
config_all9['instruments'] = ALL_9
trades_all9_fix = _run_window_intraday(df, tm_cache, config_all9, intraday_df,
                                         TEST_START, TEST_END, contracts=contracts)
m1m2_in_all9 = trades_all9_fix[trades_all9_fix['instrument'] == 'M1-M2'] if len(trades_all9_fix) > 0 else pd.DataFrame()
print(f'\n  M1-M2 trades within all-9 run (with fix): {len(m1m2_in_all9)}')
print(f'  (Should match M1-M2-only run: {len(trades_with_fix)})')

print(f'\n{"="*70}')
print(f'PART 3 COMPLETE')
print(f'{"="*70}')

# ══════════════════════════════════════════════════════════════
# PART 4 — M1-M2 ONLY RESULT CHECK (MINUTE DATA, LOCKED THRESHOLDS)
# ══════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'PART 4 — M1-M2 ONLY RESULT CHECK (MINUTE DATA, locked thresholds)')
print(f'{"="*70}')

m1m2_metrics = compute_metrics(trades_with_fix, df, TEST_START, TEST_END)

print(f'\nM1-M2 ONLY — Minute Data — 2024-2026 — Locked Config (z>1.5, exit<0.5, dur>=3)')
print(f'  Trades:       {m1m2_metrics["n_trades"]}')
print(f'  Win rate:     {m1m2_metrics["win_rate"]:.1f}%')
print(f'  Total PnL:    {m1m2_metrics["total_pnl"]:+.1f} pts')
sh_str = f'{m1m2_metrics["adj_sharpe"]:.3f}' if not pd.isna(m1m2_metrics['adj_sharpe']) else 'N/A'
print(f'  Adj Sharpe:   {sh_str}')
ns_str = f'{m1m2_metrics["naive_sharpe"]:.3f}' if not pd.isna(m1m2_metrics['naive_sharpe']) else 'N/A'
print(f'  Naive Sharpe: {ns_str}')
print(f'  Max DD:       {m1m2_metrics["max_dd"]:+.1f} pts')
print(f'  Avg Hold:     {m1m2_metrics["avg_hp"]:.1f} days')
print(f'  Avg Win:      {m1m2_metrics["avg_win"]:+.1f} pts')
print(f'  Avg Loss:     {m1m2_metrics["avg_loss"]:+.1f} pts')

# Also report all-9 metrics WITH fix for context
all9_metrics = compute_metrics(trades_all9_fix, df, TEST_START, TEST_END)
print(f'\nAll-9 WITH fix (context) — Minute Data — 2024-2026:')
print(f'  Trades:       {all9_metrics["n_trades"]}')
print(f'  Win rate:     {all9_metrics["win_rate"]:.1f}%')
print(f'  Total PnL:    {all9_metrics["total_pnl"]:+.1f} pts')
a9sh = f'{all9_metrics["adj_sharpe"]:.3f}' if not pd.isna(all9_metrics['adj_sharpe']) else 'N/A'
print(f'  Adj Sharpe:   {a9sh}')
print(f'  Max DD:       {all9_metrics["max_dd"]:+.1f} pts')

print(f'\nComparison vs prior contaminated all-9 minute result:')
print(f'  Prior all-9 adj Sharpe:   0.675 (contaminated)')
print(f'  Current all-9 adj Sharpe: {a9sh} (with M1-M2 expiry fix)')
if not pd.isna(all9_metrics['adj_sharpe']):
    delta = all9_metrics['adj_sharpe'] - 0.675
    print(f'  Delta:                    {delta:+.3f}')

# Per-instrument in all-9 for context
print(f'\n--- Per-Instrument (all-9, with fix) ---')
print(f'{"Instrument":<12s} {"n":>4s} {"win%":>6s} {"PnL":>8s} {"avg":>7s}')
for inst in ALL_9:
    t = trades_all9_fix[trades_all9_fix['instrument'] == inst] if len(trades_all9_fix) > 0 else pd.DataFrame()
    n = len(t)
    if n == 0:
        print(f'{inst:<12s} {0:>4d}')
        continue
    wins = (t['net_pnl'] > 0).sum()
    print(f'{inst:<12s} {n:>4d} {wins/n*100:>5.1f}% {t["net_pnl"].sum():>+8.1f} {t["net_pnl"].mean():>+7.1f}')

print(f'\n{"="*70}')
print(f'PART 4 COMPLETE')
print(f'{"="*70}')

# ══════════════════════════════════════════════════════════════
# PART 5 — ENTRY/EXIT Z-SCORE SENSITIVITY (MINUTE DATA, M1-M2 ONLY)
# ══════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'PART 5 — ENTRY/EXIT Z-SCORE SENSITIVITY (MINUTE DATA, M1-M2 ONLY)')
print(f'{"="*70}')

entry_grid = [1.25, 1.5, 1.75, 2.0]
exit_grid = [0.25, 0.5, 0.75]

print(f'\nEntry grid: {entry_grid}')
print(f'Exit grid:  {exit_grid}')
print(f'Duration:   >= 3 days (held constant)')
print(f'Period:     2024-2026 (full minute-data period, single result per combo)')
print()

results = []
for entry_z in entry_grid:
    for exit_z_val in exit_grid:
        cfg = DEFAULT_CONFIG.copy()
        cfg['instruments'] = ['M1-M2']
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
            'naive_sharpe': m['naive_sharpe'],
            'max_dd': m['max_dd'],
            'avg_hp': m['avg_hp'],
        })

# Find baseline
baseline_sh = None
for r in results:
    if r['entry_z'] == 1.5 and r['exit_z'] == 0.5:
        baseline_sh = r['adj_sharpe']
        break

print(f'{"entry_z":>8s} {"exit_z":>8s} {"n":>5s} {"win%":>6s} {"PnL":>9s} {"adjSh":>8s} {"naiveSh":>8s} {"maxDD":>8s} {"avgHP":>6s}  {"note":s}')
print('-'*90)

for r in results:
    sh_str = f'{r["adj_sharpe"]:.3f}' if not pd.isna(r['adj_sharpe']) else 'N/A'
    ns_str = f'{r["naive_sharpe"]:.3f}' if not pd.isna(r['naive_sharpe']) else 'N/A'
    note = ''
    if r['entry_z'] == 1.5 and r['exit_z'] == 0.5:
        note = '<-- LOCKED BASELINE'
    elif r['n'] < 10:
        note = 'LOW-N WARNING'
    elif baseline_sh is not None and not pd.isna(r['adj_sharpe']):
        if r['adj_sharpe'] > baseline_sh + 0.1:
            note = f'BETTER (+{r["adj_sharpe"] - baseline_sh:.3f})'
    print(f'{r["entry_z"]:>8.2f} {r["exit_z"]:>8.2f} {r["n"]:>5d} {r["win%"]:>5.1f}% '
          f'{r["pnl"]:>+9.1f} {sh_str:>8s} {ns_str:>8s} {r["max_dd"]:>+8.1f} {r["avg_hp"]:>6.1f}  {note}')

# Best candidate
valid_results = [r for r in results if r['n'] >= 10 and not pd.isna(r['adj_sharpe'])]
if valid_results:
    best = max(valid_results, key=lambda x: x['adj_sharpe'])
    print(f'\n--- Best Candidate (n>=10): entry_z={best["entry_z"]}, exit_z={best["exit_z"]} ---')
    print(f'  adj Sharpe = {best["adj_sharpe"]:.3f}, n = {best["n"]}, PnL = {best["pnl"]:+.1f}')
    if best['entry_z'] != 1.5 or best['exit_z'] != 0.5:
        print(f'  Delta vs baseline: {best["adj_sharpe"] - baseline_sh:+.3f}')

print(f'\n{"="*70}')
print(f'ALL PARTS COMPLETE')
print(f'{"="*70}')
