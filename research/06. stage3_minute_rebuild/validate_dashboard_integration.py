"""
Validate dashboard integration: Parts 3.1, 3.2, 3.3.
Runs the backtest engine directly (no Streamlit) to confirm numbers.
"""
import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
from pathlib import Path

from MRBackTest.engine.backtest_engine import (
    build_panel, load_intraday_data, run_backtest_intraday,
    compute_metrics, compute_daily_portfolio_sharpe_configurable,
    INTRADAY_START, INTRADAY_END, ALL_9, POINT_VALUE,
)

# ── Helpers (mirror app_v2.py) ──
SHAPE_REGIMES = ['SB', 'C', 'SB+C']
TIME_PERIODS = {
    'P1 2020-2021': ('2020-01-01', '2021-12-31'),
    'P2 2022-2023': ('2022-01-01', '2023-12-31'),
    'P3 2024-2026': ('2024-01-01', '2026-12-31'),
}

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
        'n': n,
        'win_pct': round(wins / n * 100, 1),
        'adj_sharpe': round(adj_sh, 3) if not np.isnan(adj_sh) else np.nan,
        'pnl': pnl,
        'max_dd': max_dd,
    }


def make_config(entry_z=1.5, exit_z=0.5):
    return {
        'entry_z': entry_z,
        'exit_z': exit_z,
        'duration_threshold': 3,
        'first_entry_lots': 1,
        'time_stop_days': 20,
        'stop_loss_z': None,
        'scale_in_tiers': [
            {'z_level': 1.75, 'lots': 1, 'enabled': False},
            {'z_level': 2.0, 'lots': 1, 'enabled': False},
            {'z_level': 2.25, 'lots': 1, 'enabled': False},
        ],
        'instruments': ALL_9,
        'pm_filter_level': 0,
        'tm_regime_risk_threshold': 0.50,
        'cost_spread_myr': 22.0,
        'cost_butterfly_myr': 44.0,
        'data_resolution': '60min',
        'date_range': 'intraday',
    }


# ═══════════════════════════════════════════════════
# LOAD DATA (once)
# ═══════════════════════════════════════════════════
print('Loading panel + intraday data...')
df_panel, tm_cache = build_panel()
intraday_df = load_intraday_data()
print(f'  INTRADAY_START = {INTRADAY_START}')
print(f'  Intraday rows: {len(intraday_df)}')
print()

# ═══════════════════════════════════════════════════
# PART 3.1: Baseline validation (entry=1.5, exit=0.5)
# ═══════════════════════════════════════════════════
print('=' * 70)
print('PART 3.1 — BASELINE VALIDATION (entry=1.5, exit=0.5)')
print('=' * 70)

config_baseline = make_config(1.5, 0.5)
res = run_backtest_intraday(config_baseline, df_panel, tm_cache, intraday_df)
trades = res['intraday']
m = compute_metrics(trades, df_panel, INTRADAY_START, INTRADAY_END)
adj_sharpe = compute_daily_portfolio_sharpe_configurable(
    trades, INTRADAY_START, INTRADAY_END, df_panel, config_baseline)

print(f'  Trades:     {m["n_trades"]}  (ref: 332)')
print(f'  Win%:       {m["win_rate"]:.1f}  (ref: 70.8)')
print(f'  PnL:        {m["total_pnl"]:.1f}  (ref: 5102.4)')
print(f'  Adj Sharpe: {adj_sharpe:.3f}  (ref: 1.112)')
print(f'  Max DD:     {m["max_dd"]:.1f}  (ref: -287.8)')

# Assert
checks = []
checks.append(('n_trades', m['n_trades'] == 332))
checks.append(('win_rate', abs(m['win_rate'] - 70.8) < 0.1))
checks.append(('pnl', abs(m['total_pnl'] - 5102.4) < 0.5))
checks.append(('adj_sharpe', abs(adj_sharpe - 1.112) < 0.005))
checks.append(('max_dd', abs(m['max_dd'] - (-287.8)) < 0.5))

all_pass = all(ok for _, ok in checks)
for name, ok in checks:
    status = 'PASS' if ok else 'FAIL'
    print(f'  {status}: {name}')

if all_pass:
    print('  >>> ALL PART 3.1 CHECKS PASSED')
else:
    print('  >>> SOME CHECKS FAILED — investigate')
print()

# ═══════════════════════════════════════════════════
# PART 3.2: Regime x Period validation (1.5/0.5)
# ═══════════════════════════════════════════════════
print('=' * 70)
print('PART 3.2 — REGIME x PERIOD BREAKDOWN (entry=1.5, exit=0.5)')
print('=' * 70)

trades_bd = trades.copy()
trades_bd['_regime'] = trades_bd['shape'].apply(classify_regime)
trades_bd['_period'] = trades_bd['entry_date'].apply(assign_period)

# Print full breakdown
period_keys = list(TIME_PERIODS.keys()) + ['Full Range']
header = f'{"Regime":<8}'
for pk in period_keys:
    header += f'  {pk:>16s}(n/win%/sharpe)'
print(header)

for regime in SHAPE_REGIMES:
    line = f'{regime:<8}'
    for pk in period_keys:
        if pk == 'Full Range':
            p_start, p_end = INTRADAY_START, INTRADAY_END
            period_trades = trades_bd
        else:
            p_start, p_end = TIME_PERIODS[pk]
            period_trades = trades_bd[trades_bd['_period'] == pk]

        if regime == 'SB+C':
            cell_trades = period_trades[period_trades['_regime'].isin(['SB', 'C'])]
        else:
            cell_trades = period_trades[period_trades['_regime'] == regime]

        cell_trades = cell_trades.reset_index(drop=True)
        cm = cell_metrics(cell_trades, df_panel, p_start, p_end, config_baseline)
        wp = f'{cm["win_pct"]:.1f}' if not np.isnan(cm["win_pct"]) else '  nan'
        sh = f'{cm["adj_sharpe"]:.3f}' if not np.isnan(cm["adj_sharpe"]) else '  nan'
        line += f'  n={cm["n"]:>3d} w={wp:>5s} sh={sh:>6s}'

    print(line)

# Full range SB+C = total baseline
sb_c_full = trades_bd[trades_bd['_regime'].isin(['SB', 'C'])].reset_index(drop=True)
sb_c_n = len(sb_c_full)
print(f'\nSB+C Full Range n = {sb_c_n} (should equal total trades = {m["n_trades"]})')
if sb_c_n == m['n_trades']:
    print('  >>> CONFIRMED: SB+C Full Range matches baseline total (zero Transitional trades)')
else:
    t_n = len(trades_bd[trades_bd['_regime'] == 'Transitional'])
    print(f'  Transitional trades: {t_n}')
print()

# ═══════════════════════════════════════════════════
# PART 3.3: Full 42-combo grid
# ═══════════════════════════════════════════════════
print('=' * 70)
print('PART 3.3 — FULL 42-COMBO GRID')
print('=' * 70)

entry_vals = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
exit_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

sweep_rows = []
for ez in entry_vals:
    for xz in exit_vals:
        sc = make_config(ez, xz)
        sr = run_backtest_intraday(sc, df_panel, tm_cache, intraday_df)
        st_trades = sr['intraday']
        sm = compute_metrics(st_trades, df_panel, INTRADAY_START, INTRADAY_END)
        s_sharpe = compute_daily_portfolio_sharpe_configurable(
            st_trades, INTRADAY_START, INTRADAY_END, df_panel, sc)

        sweep_rows.append({
            'Entry Z': ez,
            'Exit Z': xz,
            'Trades': sm['n_trades'],
            'Win%': sm['win_rate'],
            'PnL': sm['total_pnl'],
            'Adj Sharpe': s_sharpe if not np.isnan(s_sharpe) else 0.0,
            'Max DD': sm['max_dd'],
        })
        print(f'  e={ez:.2f} x={xz:.1f}: n={sm["n_trades"]:>4d}  w={sm["win_rate"]:.1f}%  '
              f'pnl={sm["total_pnl"]:>+8.1f}  sharpe={s_sharpe:.3f}  dd={sm["max_dd"]:>+.1f}')

sweep_df = pd.DataFrame(sweep_rows)

# Find best by Adj Sharpe (require >= 30 trades)
qualified = sweep_df[sweep_df['Trades'] >= 30].copy()
best_idx = qualified['Adj Sharpe'].idxmax()
best = qualified.iloc[qualified['Adj Sharpe'].values.argmax()]

print()
print(f'BEST (Adj Sharpe, n>=30): entry={best["Entry Z"]:.2f}, exit={best["Exit Z"]:.1f}')
print(f'  Trades={int(best["Trades"])}, Win%={best["Win%"]:.1f}, '
      f'PnL={best["PnL"]:+.1f}, Sharpe={best["Adj Sharpe"]:.3f}, MaxDD={best["Max DD"]:+.1f}')

# Compare to locked config (1.5/0.5)
locked = sweep_df[(sweep_df['Entry Z'] == 1.5) & (sweep_df['Exit Z'] == 0.5)].iloc[0]
print(f'\nLOCKED (1.5/0.5): Trades={int(locked["Trades"])}, Sharpe={locked["Adj Sharpe"]:.3f}, '
      f'PnL={locked["PnL"]:+.1f}')

if best['Entry Z'] == 1.5 and best['Exit Z'] == 0.5:
    print('>>> LOCKED CONFIG IS STILL THE BEST')
else:
    print(f'>>> NEW BEST FOUND: entry={best["Entry Z"]:.2f}, exit={best["Exit Z"]:.1f}')
    print(f'    Delta Sharpe: {best["Adj Sharpe"] - locked["Adj Sharpe"]:+.3f}')

# Save full grid
sweep_df.to_csv(Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild/full_42combo_grid.csv'), index=False)
print('\nSaved: full_42combo_grid.csv')
print()
print('VALIDATION COMPLETE.')
