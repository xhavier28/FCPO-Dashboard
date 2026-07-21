"""
Stage 3: Run locked MR-Daily config in 60-min intraday mode (2024-2026)
and compare against daily-data results for the same period.

Uses the existing backtest_engine.py 60-min intraday machinery:
  - Z-score baseline from DAILY settlements (frozen at close)
  - Entry/exit checks at every 60-min bar during the day
  - Corrected pinned-contract pricing (with rolling-label fallback)

Also runs daily mode restricted to 2024-2026 for side-by-side comparison.
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
    run_backtest_intraday, _run_window_daily,
    compute_metrics, _compute_naive_sharpe, _compute_daily_portfolio_sharpe,
    DEFAULT_CONFIG, ALL_9, INTRADAY_START, INTRADAY_END,
    _panel_cache,
)

# ── CONFIG ──────────────────────────────────────────────────────
LOCKED_CONFIG = DEFAULT_CONFIG.copy()
LOCKED_CONFIG.update({
    'entry_z': 1.5,
    'exit_z': 0.5,
    'duration_threshold': 3,
    'first_entry_lots': 1,
    'time_stop_days': 20,
    'stop_loss_z': None,
    'scale_in_tiers': [],
    'instruments': ALL_9,
    'pm_filter_level': 0,
    'tm_regime_risk_threshold': 0.50,
    'cost_spread_myr': 100.0,
    'cost_butterfly_myr': 100.0,
})

TEST_START = INTRADAY_START   # '2024-01-01'
TEST_END   = INTRADAY_END     # '2026-12-31'

print('='*70)
print('STAGE 3 — 60-MIN INTRADAY vs DAILY COMPARISON (2024-2026)')
print(f'Run date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('='*70)

# ── 1. BUILD PANEL ─────────────────────────────────────────────
df, tm_cache = build_panel()
print()

# ── 2. RUN INTRADAY (60-MIN) ───────────────────────────────────
print('Running 60-min intraday backtest...')
intraday_df = load_intraday_data()
intra_results = run_backtest_intraday(LOCKED_CONFIG, df, tm_cache, intraday_df)
intra_trades = intra_results['intraday']
print(f'  Intraday trades: {len(intra_trades)}')

# ── 3. RUN DAILY (same 2024-2026 window) ───────────────────────
print('Running daily backtest for 2024-2026...')
contracts = _panel_cache.get('contracts', {})
daily_trades = _run_window_daily(df, tm_cache, LOCKED_CONFIG, TEST_START, TEST_END,
                                  contracts=contracts)
print(f'  Daily trades: {len(daily_trades)}')

# ── 4. COMPUTE METRICS ─────────────────────────────────────────
print('\nComputing metrics...')
intra_metrics = compute_metrics(intra_trades, df, TEST_START, TEST_END)
daily_metrics = compute_metrics(daily_trades, df, TEST_START, TEST_END)

# ── 5. PER-INSTRUMENT BREAKDOWN ────────────────────────────────
def per_instrument_stats(trades, label):
    """Return per-instrument summary as DataFrame."""
    if len(trades) == 0:
        return pd.DataFrame()
    rows = []
    for inst in ALL_9:
        t = trades[trades['instrument'] == inst]
        n = len(t)
        if n == 0:
            rows.append({'instrument': inst, 'n': 0})
            continue
        wins = (t['net_pnl'] > 0).sum()
        rows.append({
            'instrument': inst,
            'n': n,
            'win%': round(wins / n * 100, 1),
            'pnl': round(t['net_pnl'].sum(), 1),
            'avg_pnl': round(t['net_pnl'].mean(), 1),
            'avg_hp': round(t['days_held'].mean(), 1),
        })
    return pd.DataFrame(rows)

intra_by_inst = per_instrument_stats(intra_trades, 'intraday')
daily_by_inst = per_instrument_stats(daily_trades, 'daily')

# ── 6. EXIT REASON BREAKDOWN ───────────────────────────────────
def exit_breakdown(trades):
    if len(trades) == 0:
        return {}
    vc = trades['exit_reason'].value_counts()
    return {k: int(v) for k, v in vc.items()}

intra_exits = exit_breakdown(intra_trades)
daily_exits = exit_breakdown(daily_trades)

# ── 7. FALLBACK COUNT (intraday doesn't have pinned_contracts col, but check) ──
intra_fallback = 'N/A (tenor-labeled intraday bars; pinned used for daily-level exits)'
daily_fallback = 0
if len(daily_trades) > 0 and 'pinned_contracts' in daily_trades.columns:
    # Count trades where pinned_contracts is missing/nan
    daily_fallback = daily_trades['pinned_contracts'].isna().sum()

# ── 8. PRINT RESULTS ───────────────────────────────────────────
print('\n' + '='*70)
print('RESULTS: 60-MIN INTRADAY vs DAILY (2024-2026)')
print('='*70)

print(f'\n{"Metric":<25s}  {"Intraday (60min)":>18s}  {"Daily":>18s}  {"Delta":>10s}')
print('-'*75)

def fmt(v, fmt_str='.1f'):
    if pd.isna(v) or v is None:
        return 'N/A'
    return f'{v:{fmt_str}}'

metrics_display = [
    ('n_trades',       'Trades',           'd'),
    ('win_rate',       'Win %',            '.1f'),
    ('total_pnl',      'Total PnL (pts)',  '.1f'),
    ('avg_win',        'Avg Win',          '.1f'),
    ('avg_loss',       'Avg Loss',         '.1f'),
    ('avg_hp',         'Avg Hold (days)',   '.1f'),
    ('naive_sharpe',   'Naive Sharpe',     '.3f'),
    ('adj_sharpe',     'Adj Sharpe',       '.3f'),
    ('max_dd',         'Max DD (pts)',     '.1f'),
    ('pct_take_profit','% Take Profit',    '.1f'),
    ('pct_invalidated','% Invalidated',    '.1f'),
    ('pct_time_stop',  '% Time Stop',      '.1f'),
    ('pct_regime_risk','% Regime Risk',     '.1f'),
    ('shape_survival', '% Shape Survived',  '.1f'),
]

for key, label, f in metrics_display:
    iv = intra_metrics.get(key)
    dv = daily_metrics.get(key)
    if pd.notna(iv) and pd.notna(dv):
        delta = iv - dv
        delta_s = f'{delta:+{f}}'
    else:
        delta_s = ''
    print(f'{label:<25s}  {fmt(iv, f):>18s}  {fmt(dv, f):>18s}  {delta_s:>10s}')

print(f'\n--- Per-Instrument Breakdown (Intraday) ---')
if len(intra_by_inst) > 0:
    print(f'{"Instrument":<12s} {"n":>4s} {"win%":>6s} {"PnL":>8s} {"avg":>7s} {"HP":>5s}')
    for _, r in intra_by_inst.iterrows():
        if r['n'] == 0:
            print(f'{r["instrument"]:<12s} {0:>4d}')
            continue
        print(f'{r["instrument"]:<12s} {r["n"]:>4.0f} {r["win%"]:>5.1f}% {r["pnl"]:>+8.1f} {r["avg_pnl"]:>+7.1f} {r["avg_hp"]:>5.1f}')

print(f'\n--- Per-Instrument Breakdown (Daily) ---')
if len(daily_by_inst) > 0:
    print(f'{"Instrument":<12s} {"n":>4s} {"win%":>6s} {"PnL":>8s} {"avg":>7s} {"HP":>5s}')
    for _, r in daily_by_inst.iterrows():
        if r['n'] == 0:
            print(f'{r["instrument"]:<12s} {0:>4d}')
            continue
        print(f'{r["instrument"]:<12s} {r["n"]:>4.0f} {r["win%"]:>5.1f}% {r["pnl"]:>+8.1f} {r["avg_pnl"]:>+7.1f} {r["avg_hp"]:>5.1f}')

print(f'\n--- Exit Reason Comparison ---')
all_reasons = sorted(set(list(intra_exits.keys()) + list(daily_exits.keys())))
print(f'{"Reason":<20s} {"Intraday":>10s} {"Daily":>10s}')
for r in all_reasons:
    print(f'{r:<20s} {intra_exits.get(r, 0):>10d} {daily_exits.get(r, 0):>10d}')

print(f'\n--- Residual Fallback ---')
print(f'  Intraday: {intra_fallback}')
print(f'  Daily:    {daily_fallback} trades with missing pinned_contracts')

# ── 9. LOG TO FILE ──────────────────────────────────────────────
log_path = r'C:\ClaudeCode\research\06. stage3_minute_rebuild\stage3_minute_rebuild_log.txt'
with open(log_path, 'a', encoding='utf-8') as f:
    f.write('\n\n')
    f.write('='*70 + '\n')
    f.write(f'STAGE 3 — 60-MIN INTRADAY vs DAILY COMPARISON — {datetime.now().strftime("%Y-%m-%d")}\n')
    f.write('='*70 + '\n')
    f.write(f'\nConfig: z>1.5, z_exit=0.5, dur>=3d, 1 lot, PM L0, TM RR<50%,\n')
    f.write(f'  all 9 instruments, 100 MYR cost, 20d time stop, no SL, no scale-in\n')
    f.write(f'Period: {TEST_START} to {TEST_END} (minute data available range)\n')
    f.write(f'\nDesign:\n')
    f.write(f'  - Z-score from DAILY settlement (frozen at close, regime-relative)\n')
    f.write(f'  - Intraday mode: entry/exit checked at every 60-min bar\n')
    f.write(f'  - Daily mode: entry/exit checked at daily close only\n')
    f.write(f'  - Both use corrected pinned-contract pricing\n')
    f.write(f'  - Intraday bars are TENOR-LABELED (per-contract 60-min data unavailable)\n')
    f.write(f'  - Daily-level exits (shape change, time stop, TM) use pinned contracts\n')

    f.write(f'\n--- Side-by-Side Comparison ---\n')
    f.write(f'{"Metric":<25s}  {"Intraday (60min)":>18s}  {"Daily":>18s}  {"Delta":>10s}\n')
    f.write('-'*75 + '\n')
    for key, label, fm in metrics_display:
        iv = intra_metrics.get(key)
        dv = daily_metrics.get(key)
        if pd.notna(iv) and pd.notna(dv):
            delta = iv - dv
            delta_s = f'{delta:+{fm}}'
        else:
            delta_s = ''
        f.write(f'{label:<25s}  {fmt(iv, fm):>18s}  {fmt(dv, fm):>18s}  {delta_s:>10s}\n')

    f.write(f'\n--- Per-Instrument (Intraday) ---\n')
    f.write(f'{"Instrument":<12s} {"n":>4s} {"win%":>6s} {"PnL":>8s} {"avg":>7s} {"HP":>5s}\n')
    for _, r in intra_by_inst.iterrows():
        if r['n'] == 0:
            f.write(f'{r["instrument"]:<12s} {0:>4d}\n')
        else:
            f.write(f'{r["instrument"]:<12s} {r["n"]:>4.0f} {r["win%"]:>5.1f}% {r["pnl"]:>+8.1f} {r["avg_pnl"]:>+7.1f} {r["avg_hp"]:>5.1f}\n')

    f.write(f'\n--- Per-Instrument (Daily) ---\n')
    f.write(f'{"Instrument":<12s} {"n":>4s} {"win%":>6s} {"PnL":>8s} {"avg":>7s} {"HP":>5s}\n')
    for _, r in daily_by_inst.iterrows():
        if r['n'] == 0:
            f.write(f'{r["instrument"]:<12s} {0:>4d}\n')
        else:
            f.write(f'{r["instrument"]:<12s} {r["n"]:>4.0f} {r["win%"]:>5.1f}% {r["pnl"]:>+8.1f} {r["avg_pnl"]:>+7.1f} {r["avg_hp"]:>5.1f}\n')

    f.write(f'\n--- Exit Reason Comparison ---\n')
    f.write(f'{"Reason":<20s} {"Intraday":>10s} {"Daily":>10s}\n')
    for r in all_reasons:
        f.write(f'{r:<20s} {intra_exits.get(r, 0):>10d} {daily_exits.get(r, 0):>10d}\n')

    f.write(f'\n--- Residual Fallback ---\n')
    f.write(f'  Intraday: {intra_fallback}\n')
    f.write(f'  Daily:    {daily_fallback} trades with missing pinned_contracts\n')

    # Key interpretation
    f.write(f'\n--- Interpretation ---\n')
    intra_n = intra_metrics['n_trades']
    daily_n = daily_metrics['n_trades']
    intra_sh = intra_metrics['adj_sharpe']
    daily_sh = daily_metrics['adj_sharpe']
    intra_pnl = intra_metrics['total_pnl']
    daily_pnl = daily_metrics['total_pnl']

    if intra_n > daily_n:
        f.write(f'  Trade count: Intraday finds {intra_n - daily_n} MORE trades ({intra_n} vs {daily_n})\n')
        f.write(f'    → 60-min bars detect z-threshold crossings that daily close misses\n')
    elif intra_n < daily_n:
        f.write(f'  Trade count: Intraday finds {daily_n - intra_n} FEWER trades ({intra_n} vs {daily_n})\n')
        f.write(f'    → Some daily signals may not cross threshold at 60-min resolution\n')
    else:
        f.write(f'  Trade count: IDENTICAL ({intra_n})\n')

    if pd.notna(intra_sh) and pd.notna(daily_sh):
        if intra_sh > daily_sh:
            f.write(f'  Adj Sharpe: Intraday BETTER ({intra_sh:.3f} vs {daily_sh:.3f})\n')
            f.write(f'    → Intraday timing improves risk-adjusted returns\n')
        else:
            f.write(f'  Adj Sharpe: Intraday WORSE ({intra_sh:.3f} vs {daily_sh:.3f})\n')
            f.write(f'    → Intraday timing does NOT improve risk-adjusted returns\n')

    f.write(f'  PnL: Intraday {intra_pnl:+.1f} vs Daily {daily_pnl:+.1f} (delta {intra_pnl - daily_pnl:+.1f} pts)\n')

print(f'\nResults logged to: {log_path}')
print('Done.')
