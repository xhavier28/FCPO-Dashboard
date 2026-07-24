"""
Validate Part 3: period-only sensitivity table.
3.1: Full-range 1.5/0.5 = 332 trades, 70.8% win, +5102.4, Sharpe 1.112, DD -287.8
3.2: Full-range 1.0/0.1 = 527 trades, 65.7% win, +6336.0, Sharpe 1.234, DD -392.5
3.3: Best combo per period (P1, P2, P3) — does it shift?
"""
import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np

from MRBackTest.engine.backtest_engine import (
    build_panel, load_intraday_data, run_backtest_intraday,
    compute_metrics, compute_daily_portfolio_sharpe_configurable,
    INTRADAY_START, INTRADAY_END, ALL_9,
)

TIME_PERIODS = {
    'P1 2020-2021': ('2020-01-01', '2021-12-31'),
    'P2 2022-2023': ('2022-01-01', '2023-12-31'),
    'P3 2024-2026': ('2024-01-01', '2026-12-31'),
}

def assign_period(entry_date):
    d = pd.Timestamp(entry_date)
    if d < pd.Timestamp('2022-01-01'): return 'P1 2020-2021'
    elif d < pd.Timestamp('2024-01-01'): return 'P2 2022-2023'
    return 'P3 2024-2026'

def period_metrics(trades_subset, df_panel, period_start, period_end, config):
    n = len(trades_subset)
    if n == 0:
        return {'n': 0, 'win_pct': np.nan, 'adj_sharpe': np.nan, 'pnl': 0.0, 'max_dd': 0.0}
    wins = (trades_subset['net_pnl'] > 0).sum()
    pnl = round(trades_subset['net_pnl'].sum(), 1)
    cum = trades_subset['net_pnl'].cumsum()
    max_dd = round((cum - cum.cummax()).min(), 1)
    adj_sh = compute_daily_portfolio_sharpe_configurable(
        trades_subset, period_start, period_end, df_panel, config)
    return {
        'n': n, 'win_pct': round(wins / n * 100, 1),
        'adj_sharpe': round(adj_sh, 3) if not np.isnan(adj_sh) else np.nan,
        'pnl': pnl, 'max_dd': max_dd,
    }

def make_config(entry_z, exit_z):
    return {
        'entry_z': entry_z, 'exit_z': exit_z,
        'duration_threshold': 3, 'first_entry_lots': 1, 'time_stop_days': 20,
        'stop_loss_z': None,
        'scale_in_tiers': [
            {'z_level': 1.75, 'lots': 1, 'enabled': False},
            {'z_level': 2.0, 'lots': 1, 'enabled': False},
            {'z_level': 2.25, 'lots': 1, 'enabled': False},
        ],
        'instruments': ALL_9, 'pm_filter_level': 0,
        'tm_regime_risk_threshold': 0.50,
        'cost_spread_myr': 22.0, 'cost_butterfly_myr': 44.0,
        'data_resolution': '60min', 'date_range': 'intraday',
    }

print('Loading data...')
df_panel, tm_cache = build_panel()
intraday_df = load_intraday_data()

# ── 3.1 + 3.2: Spot checks ──
for label, ez, xz, ref in [
    ('3.1', 1.5, 0.5, {'n': 332, 'win': 70.8, 'pnl': 5102.4, 'sh': 1.112, 'dd': -287.8}),
    ('3.2', 1.0, 0.1, {'n': 527, 'win': 65.7, 'pnl': 6336.0, 'sh': 1.234, 'dd': -392.5}),
]:
    cfg = make_config(ez, xz)
    res = run_backtest_intraday(cfg, df_panel, tm_cache, intraday_df)
    trades = res['intraday']
    m = compute_metrics(trades, df_panel, INTRADAY_START, INTRADAY_END)
    sh = compute_daily_portfolio_sharpe_configurable(trades, INTRADAY_START, INTRADAY_END, df_panel, cfg)
    ok_n = m['n_trades'] == ref['n']
    ok_w = abs(m['win_rate'] - ref['win']) < 0.1
    ok_p = abs(m['total_pnl'] - ref['pnl']) < 0.5
    ok_s = abs(sh - ref['sh']) < 0.005
    ok_d = abs(m['max_dd'] - ref['dd']) < 0.5
    all_ok = all([ok_n, ok_w, ok_p, ok_s, ok_d])
    status = 'PASS' if all_ok else 'FAIL'
    print(f'\nPart {label} (entry={ez}, exit={xz}): {status}')
    print(f'  n={m["n_trades"]}({ref["n"]}) w={m["win_rate"]:.1f}({ref["win"]}) '
          f'pnl={m["total_pnl"]:.1f}({ref["pnl"]}) sh={sh:.3f}({ref["sh"]}) dd={m["max_dd"]:.1f}({ref["dd"]})')

# ── 3.3: Best combo per period ──
print('\n' + '=' * 70)
print('Part 3.3 — BEST COMBO PER PERIOD')
print('=' * 70)

entry_vals = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
exit_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

all_rows = []
for ez in entry_vals:
    for xz in exit_vals:
        cfg = make_config(ez, xz)
        res = run_backtest_intraday(cfg, df_panel, tm_cache, intraday_df)
        trades = res['intraday']
        if len(trades) == 0:
            continue
        trades_tagged = trades.copy()
        trades_tagged['_period'] = trades_tagged['entry_date'].apply(assign_period)

        row = {'Entry Z': ez, 'Exit Z': xz}

        # Full range
        m_full = compute_metrics(trades, df_panel, INTRADAY_START, INTRADAY_END)
        sh_full = compute_daily_portfolio_sharpe_configurable(trades, INTRADAY_START, INTRADAY_END, df_panel, cfg)
        row['Full n'] = m_full['n_trades']
        row['Full Sharpe'] = round(sh_full, 3) if not np.isnan(sh_full) else np.nan

        for pk, (p_start, p_end) in TIME_PERIODS.items():
            pt = trades_tagged[trades_tagged['_period'] == pk].reset_index(drop=True)
            pm = period_metrics(pt, df_panel, p_start, p_end, cfg)
            row[f'{pk} n'] = pm['n']
            row[f'{pk} Sharpe'] = pm['adj_sharpe']
            row[f'{pk} Win%'] = pm['win_pct']
            row[f'{pk} PnL'] = pm['pnl']

        all_rows.append(row)

df = pd.DataFrame(all_rows)

# Best per period
for pk in list(TIME_PERIODS.keys()) + ['Full']:
    sh_col = f'{pk} Sharpe' if pk != 'Full' else 'Full Sharpe'
    n_col = f'{pk} n' if pk != 'Full' else 'Full n'
    qualified = df[df[n_col] >= 8]
    if len(qualified) == 0:
        print(f'\n{pk}: no qualified rows (all n<8)')
        continue
    best_idx = qualified[sh_col].idxmax()
    best = qualified.loc[best_idx]
    print(f'\n{pk}: BEST = entry={best["Entry Z"]:.2f} / exit={best["Exit Z"]:.1f}')
    print(f'  Sharpe={best[sh_col]:.3f}, n={int(best[n_col])}')
    if pk != 'Full':
        print(f'  Win%={best[f"{pk} Win%"]:.1f}, PnL={best[f"{pk} PnL"]:+.1f}')

# State plainly
print('\n' + '=' * 70)
p1_best = df.loc[df[df['P1 2020-2021 n'] >= 8]['P1 2020-2021 Sharpe'].idxmax()]
p2_best = df.loc[df[df['P2 2022-2023 n'] >= 8]['P2 2022-2023 Sharpe'].idxmax()]
p3_best = df.loc[df[df['P3 2024-2026 n'] >= 8]['P3 2024-2026 Sharpe'].idxmax()]
full_best = df.loc[df[df['Full n'] >= 8]['Full Sharpe'].idxmax()]

combos = [
    ('P1', p1_best['Entry Z'], p1_best['Exit Z']),
    ('P2', p2_best['Entry Z'], p2_best['Exit Z']),
    ('P3', p3_best['Entry Z'], p3_best['Exit Z']),
    ('Full', full_best['Entry Z'], full_best['Exit Z']),
]
same = len(set((e, x) for _, e, x in combos)) == 1
print(f'P1 best: {combos[0][1]:.2f}/{combos[0][2]:.1f}')
print(f'P2 best: {combos[1][1]:.2f}/{combos[1][2]:.1f}')
print(f'P3 best: {combos[2][1]:.2f}/{combos[2][2]:.1f}')
print(f'Full best: {combos[3][1]:.2f}/{combos[3][2]:.1f}')
if same:
    print('CONSISTENT: Same combo wins in all periods and full range.')
else:
    print('SHIFTS: Best combo changes across periods.')
print()
print('VALIDATION COMPLETE.')
