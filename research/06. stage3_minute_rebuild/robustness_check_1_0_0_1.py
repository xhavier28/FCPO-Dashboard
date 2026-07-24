"""
Robustness check for entry=1.0/exit=0.1 vs locked 1.5/0.5.
1. Regime x Period breakdown for both configs
2. Per-instrument breakdown for both configs
"""
import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np

from MRBackTest.engine.backtest_engine import (
    build_panel, load_intraday_data, run_backtest_intraday,
    compute_metrics, compute_daily_portfolio_sharpe_configurable,
    INTRADAY_START, INTRADAY_END, ALL_9, POINT_VALUE,
)

TIME_PERIODS = {
    'P1 2020-2021': ('2020-01-01', '2021-12-31'),
    'P2 2022-2023': ('2022-01-01', '2023-12-31'),
    'P3 2024-2026': ('2024-01-01', '2026-12-31'),
}
SHAPE_REGIMES = ['SB', 'C', 'SB+C']

def classify_regime(shape):
    s = str(shape)
    if s == '0.0': return 'SB'
    elif s == '1': return 'C'
    return 'Transitional'

def assign_period(entry_date):
    d = pd.Timestamp(entry_date)
    if d < pd.Timestamp('2022-01-01'): return 'P1 2020-2021'
    elif d < pd.Timestamp('2024-01-01'): return 'P2 2022-2023'
    return 'P3 2024-2026'

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

def cell_metrics(trades_subset, df_panel, period_start, period_end, config):
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

# Load data
print('Loading data...')
df_panel, tm_cache = build_panel()
intraday_df = load_intraday_data()

configs = {
    'LOCKED 1.5/0.5': make_config(1.5, 0.5),
    'CANDIDATE 1.0/0.1': make_config(1.0, 0.1),
}

for label, config in configs.items():
    print()
    print('=' * 80)
    print(f'  {label}')
    print('=' * 80)

    res = run_backtest_intraday(config, df_panel, tm_cache, intraday_df)
    trades = res['intraday']
    m = compute_metrics(trades, df_panel, INTRADAY_START, INTRADAY_END)
    adj_sh = compute_daily_portfolio_sharpe_configurable(
        trades, INTRADAY_START, INTRADAY_END, df_panel, config)
    print(f'  TOTAL: n={m["n_trades"]}, win%={m["win_rate"]:.1f}, '
          f'PnL={m["total_pnl"]:+.1f}, adjSh={adj_sh:.3f}, maxDD={m["max_dd"]:+.1f}')

    # ── 1. Regime x Period ──
    print()
    print('  --- REGIME x PERIOD ---')
    trades_bd = trades.copy()
    trades_bd['_regime'] = trades_bd['shape'].apply(classify_regime)
    trades_bd['_period'] = trades_bd['entry_date'].apply(assign_period)

    period_keys = list(TIME_PERIODS.keys()) + ['Full Range']

    # Header
    hdr = f'  {"Regime":<7}'
    for pk in period_keys:
        short = pk.replace('2020-2021', '20-21').replace('2022-2023', '22-23').replace('2024-2026', '24-26')
        hdr += f' | {short:^30s}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    for regime in SHAPE_REGIMES:
        line = f'  {regime:<7}'
        for pk in period_keys:
            if pk == 'Full Range':
                p_start, p_end = INTRADAY_START, INTRADAY_END
                period_trades = trades_bd
            else:
                p_start, p_end = TIME_PERIODS[pk]
                period_trades = trades_bd[trades_bd['_period'] == pk]

            if regime == 'SB+C':
                ct = period_trades[period_trades['_regime'].isin(['SB', 'C'])]
            else:
                ct = period_trades[period_trades['_regime'] == regime]

            ct = ct.reset_index(drop=True)
            cm = cell_metrics(ct, df_panel, p_start, p_end, config)
            wp = f'{cm["win_pct"]:.1f}' if not np.isnan(cm["win_pct"]) else ' n/a'
            sh = f'{cm["adj_sharpe"]:.3f}' if not np.isnan(cm["adj_sharpe"]) else ' n/a '
            line += f' | n={cm["n"]:>3d} w={wp:>5s}% sh={sh:>6s}'
        print(line)

    # ── 2. Per-instrument breakdown ──
    print()
    print('  --- PER-INSTRUMENT ---')
    print(f'  {"Instrument":<14} {"n":>4} {"Win%":>6} {"PnL":>9} {"adjSh":>7} {"MaxDD":>8}')
    print('  ' + '-' * 52)

    for inst in ALL_9:
        inst_trades = trades[trades['instrument'] == inst].reset_index(drop=True)
        n_inst = len(inst_trades)
        if n_inst == 0:
            print(f'  {inst:<14} {0:>4} {"n/a":>6} {"0.0":>9} {"n/a":>7} {"0.0":>8}')
            continue
        wins = (inst_trades['net_pnl'] > 0).sum()
        pnl = inst_trades['net_pnl'].sum()
        cum = inst_trades['net_pnl'].cumsum()
        dd = (cum - cum.cummax()).min()

        # Per-instrument Sharpe: use only this instrument's trades
        inst_sh = compute_daily_portfolio_sharpe_configurable(
            inst_trades, INTRADAY_START, INTRADAY_END, df_panel, config)
        sh_str = f'{inst_sh:.3f}' if not np.isnan(inst_sh) else 'n/a'

        print(f'  {inst:<14} {n_inst:>4} {wins/n_inst*100:>5.1f}% {pnl:>+8.1f} {sh_str:>7} {dd:>+7.1f}')

    # ── 3. Exit reason breakdown ──
    print()
    print('  --- EXIT REASONS ---')
    if 'exit_reason' in trades.columns:
        exit_counts = trades['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            pct = count / len(trades) * 100
            subset = trades[trades['exit_reason'] == reason]
            avg_pnl = subset['net_pnl'].mean()
            print(f'  {reason:<16} n={count:>4} ({pct:>5.1f}%)  avg_pnl={avg_pnl:>+7.1f}')

print()
print('ROBUSTNESS CHECK COMPLETE.')
