"""
Shape Regime × Time Period Cross-Tabulation
============================================
Parts A + B + C from the original extension instruction + amendment.

A. Full 2020-2026 baseline with locked config, shape-regime breakdown
B. Entry/exit threshold sensitivity sweep with shape gating
C. Cross-tabulate shape regime × time period (4×3 grid per threshold)

Usage: python "research/06. stage3_minute_rebuild/shape_regime_time_crosstab.py"
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from MRBackTest.engine.backtest_engine import (
    build_panel, compute_metrics,
    compute_daily_portfolio_sharpe_configurable,
    ALL_9, _panel_cache, _intraday_cache,
    SPREAD_TENOR_MAP, BUTTERFLY_TENOR_MAP,
    get_cost_points,
)
import importlib.util
_sweep_path = str(Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild/invalidated_delay_sweep.py'))
_spec = importlib.util.spec_from_file_location('invalidated_delay_sweep', _sweep_path)
_sweep_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep_mod)
run_intraday_with_delay = _sweep_mod.run_intraday_with_delay

OUTPUT_DIR = Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild')

# ── Period boundaries (confirmed, no overlap) ──
PERIODS = {
    'P1_2020-2021': ('2020-01-01', '2021-12-31'),
    'P2_2022-2023': ('2022-01-01', '2023-12-31'),
    'P3_2024-2026': ('2024-01-01', '2026-12-31'),
}

# ── Shape regime classification ──
RESTING_SHAPES = ('0.0', '1')

def classify_shape_regime(shape):
    """Classify entry shape into regime bucket."""
    s = str(shape)
    if s == '0.0':
        return 'SB'
    elif s == '1':
        return 'C'
    else:
        return 'Transitional'


def tag_time_period(entry_date):
    """Assign trade to a time period based on entry date."""
    d = pd.Timestamp(entry_date)
    if d < pd.Timestamp('2022-01-01'):
        return 'P1_2020-2021'
    elif d < pd.Timestamp('2024-01-01'):
        return 'P2_2022-2023'
    else:
        return 'P3_2024-2026'


def load_extended_intraday():
    """Load intraday data with FULL 2020-2026 range (not filtered to 2024+)."""
    # Clear cache to force reload
    _intraday_cache.clear()

    base = Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild')
    spread_path = base / 'spread_tenor_close_wide.csv'
    bf_path = base / 'butterfly_tenor_close_wide.csv'

    print('Loading EXTENDED 60-min intraday data (2020-2026)...')
    spread_df = pd.read_csv(spread_path, parse_dates=['datetime'])
    bf_df = pd.read_csv(bf_path, parse_dates=['datetime'])

    # Do NOT filter to 2024+ — use full range from 2020-01-01
    spread_df = spread_df[spread_df['datetime'] >= '2020-01-01'].copy()
    bf_df = bf_df[bf_df['datetime'] >= '2020-01-01'].copy()

    spread_renamed = spread_df[['datetime']].copy()
    for tenor, inst in SPREAD_TENOR_MAP.items():
        if tenor in spread_df.columns:
            spread_renamed[inst] = spread_df[tenor]

    bf_renamed = bf_df[['datetime']].copy()
    for tenor, inst in BUTTERFLY_TENOR_MAP.items():
        if tenor in bf_df.columns:
            bf_renamed[inst] = bf_df[tenor]

    intraday = pd.merge(spread_renamed, bf_renamed, on='datetime', how='outer')
    intraday = intraday.sort_values('datetime').reset_index(drop=True)
    intraday['date'] = intraday['datetime'].dt.normalize()

    print(f'  Intraday: {len(intraday)} bars, '
          f'{intraday["datetime"].min()} to {intraday["datetime"].max()}')

    return intraday


def compute_adj_sharpe_for_trades(trades, test_start, test_end, df, config):
    """Compute adj Sharpe for a subset of trades within a date range."""
    if len(trades) == 0:
        return np.nan
    return compute_daily_portfolio_sharpe_configurable(
        trades, test_start, test_end, df, config)


def compute_cell_metrics(trades, df, test_start, test_end, config):
    """Compute metrics for a cross-tab cell."""
    n = len(trades)
    if n == 0:
        return {'n': 0, 'win_pct': np.nan, 'adj_sharpe': np.nan,
                'pnl': 0.0, 'max_dd': 0.0, 'low_n': True}

    wins = (trades['net_pnl'] > 0).sum()
    win_pct = round(wins / n * 100, 1)
    pnl = round(trades['net_pnl'].sum(), 1)
    cum = trades['net_pnl'].cumsum()
    max_dd = round((cum - cum.cummax()).min(), 1)

    adj_sharpe = compute_adj_sharpe_for_trades(trades, test_start, test_end, df, config)

    return {
        'n': n,
        'win_pct': win_pct,
        'adj_sharpe': round(adj_sharpe, 3) if not np.isnan(adj_sharpe) else np.nan,
        'pnl': pnl,
        'max_dd': max_dd,
        'low_n': n < 8,
    }


def run_sweep(df, tm_cache, intraday_df, config_base, contracts):
    """Run threshold sweep across entry/exit combinations.
    Returns dict: {(entry_z, exit_z): DataFrame of trades}"""
    entry_zs = [1.25, 1.50, 1.75, 2.00, 2.25, 2.50]
    exit_zs = [0.25, 0.50, 0.75]

    results = {}
    for ez in entry_zs:
        for xz in exit_zs:
            config = config_base.copy()
            config['entry_z'] = ez
            config['exit_z'] = xz

            trades, _ = run_intraday_with_delay(
                df, tm_cache, config, intraday_df,
                '2020-01-01', '2026-12-31',
                invalidated_delay_days=0,
                contracts=contracts,
            )
            results[(ez, xz)] = trades
            n = len(trades)
            print(f'  entry={ez:.2f} exit={xz:.2f}: {n} trades')

    return results


def build_crosstab(trades_df, df, config):
    """Build 4×3 shape regime × time period cross-tab for a set of trades."""
    shape_regimes = ['SB', 'C', 'Transitional', 'SB+C']
    period_keys = list(PERIODS.keys())

    rows = []
    for regime in shape_regimes:
        for pk in period_keys:
            p_start, p_end = PERIODS[pk]

            # Filter trades by time period
            period_trades = trades_df[
                (trades_df['entry_date'] >= pd.Timestamp(p_start)) &
                (trades_df['entry_date'] <= pd.Timestamp(p_end))
            ]

            # Filter by shape regime
            if regime == 'SB+C':
                regime_trades = period_trades[period_trades['shape_regime'].isin(['SB', 'C'])]
            else:
                regime_trades = period_trades[period_trades['shape_regime'] == regime]

            regime_trades = regime_trades.reset_index(drop=True)
            cell = compute_cell_metrics(regime_trades, df, p_start, p_end, config)
            cell['shape_regime'] = regime
            cell['period'] = pk
            rows.append(cell)

    return pd.DataFrame(rows)


def format_summary_grid(crosstab_df, label):
    """Format a 4×3 cross-tab into a readable text grid."""
    lines = []
    lines.append(f'\n  === {label} ===')
    lines.append(f'  {"Regime":<14s}  {"P1 2020-2021":>20s}  {"P2 2022-2023":>20s}  {"P3 2024-2026":>20s}')
    lines.append('  ' + '-' * 80)

    for regime in ['SB', 'C', 'Transitional', 'SB+C']:
        cells = []
        for pk in PERIODS.keys():
            row = crosstab_df[(crosstab_df['shape_regime'] == regime) &
                              (crosstab_df['period'] == pk)]
            if len(row) == 0 or row.iloc[0]['n'] == 0:
                cells.append('       --       ')
            else:
                r = row.iloc[0]
                low = ' LOW-N' if r['low_n'] else ''
                sh = f'{r["adj_sharpe"]:.3f}' if not np.isnan(r['adj_sharpe']) else '  nan'
                cells.append(f'n={r["n"]:>3d} {r["win_pct"]:>5.1f}% Sh={sh}{low}')
        sep = '  ' if regime != 'SB+C' else '  '
        line = f'  {regime:<14s}  {"  ".join(cells)}'
        lines.append(line)

        # Second line with PnL + MaxDD
        detail_cells = []
        for pk in PERIODS.keys():
            row = crosstab_df[(crosstab_df['shape_regime'] == regime) &
                              (crosstab_df['period'] == pk)]
            if len(row) == 0 or row.iloc[0]['n'] == 0:
                detail_cells.append('                    ')
            else:
                r = row.iloc[0]
                detail_cells.append(f'PnL={r["pnl"]:>+7.1f} DD={r["max_dd"]:>+6.1f}')
        lines.append(f'  {"":14s}  {"  ".join(detail_cells)}')

    return '\n'.join(lines)


def main():
    print('=' * 80)
    print('SHAPE REGIME × TIME PERIOD CROSS-TABULATION')
    print(f'Date: {datetime.now().strftime("%Y-%m-%d")}')
    print('=' * 80)

    # ── Load data ──
    df, tm_cache = build_panel()
    intraday_df = load_extended_intraday()
    contracts = _panel_cache.get('contracts', {})

    config_base = {
        'entry_z': 1.5, 'exit_z': 0.5, 'duration_threshold': 3,
        'first_entry_lots': 1, 'time_stop_days': 20,
        'stop_loss_z': None, 'scale_in_tiers': [],
        'instruments': ALL_9, 'pm_filter_level': 0,
        'tm_regime_risk_threshold': 0.50,
        'cost_spread_myr': 22.0, 'cost_butterfly_myr': 44.0,
    }

    # ══════════════════════════════════════════════════════════════
    # PART A — FULL 2020-2026 BASELINE
    # ══════════════════════════════════════════════════════════════

    print('\n' + '=' * 80)
    print('PART A — FULL 2020-2026 BASELINE (locked config)')
    print('=' * 80)

    # A.5: 2024-2026 subset check (already validated, re-confirm here)
    trades_24, _ = run_intraday_with_delay(
        df, tm_cache, config_base, intraday_df,
        '2024-01-01', '2026-12-31',
        invalidated_delay_days=0, contracts=contracts,
    )
    m24 = compute_metrics(trades_24, df, '2024-01-01', '2026-12-31')
    sh24 = compute_adj_sharpe_for_trades(trades_24, '2024-01-01', '2026-12-31', df, config_base)
    print(f'\nA.5 Subset check (2024-2026):')
    print(f'  Trades={m24["n_trades"]}, Win%={m24["win_rate"]}, PnL={m24["total_pnl"]}, '
          f'adjSh={sh24:.3f}, maxDD={m24["max_dd"]}')
    assert m24['n_trades'] == 113, f'Expected 113 trades, got {m24["n_trades"]}'
    assert abs(m24['total_pnl'] - 728.0) < 0.5, f'Expected PnL ~728.0, got {m24["total_pnl"]}'
    assert abs(sh24 - 1.366) < 0.002, f'Expected adjSh ~1.366, got {sh24}'
    print('  MATCHES reference exactly.')

    # Full 2020-2026 run
    trades_full, _ = run_intraday_with_delay(
        df, tm_cache, config_base, intraday_df,
        '2020-01-01', '2026-12-31',
        invalidated_delay_days=0, contracts=contracts,
    )
    m_full = compute_metrics(trades_full, df, '2020-01-01', '2026-12-31')
    sh_full = compute_adj_sharpe_for_trades(trades_full, '2020-01-01', '2026-12-31', df, config_base)
    print(f'\nFull 2020-2026 baseline:')
    print(f'  Trades={m_full["n_trades"]}, Win%={m_full["win_rate"]}, '
          f'PnL={m_full["total_pnl"]}, adjSh={sh_full:.3f}, maxDD={m_full["max_dd"]}')

    # Tag trades with shape regime and time period
    trades_full['shape_regime'] = trades_full['shape'].apply(classify_shape_regime)
    trades_full['time_period'] = trades_full['entry_date'].apply(tag_time_period)

    # Shape regime breakdown
    print('\nShape regime breakdown (full 2020-2026):')
    for regime in ['SB', 'C', 'Transitional', 'SB+C']:
        if regime == 'SB+C':
            rt = trades_full[trades_full['shape_regime'].isin(['SB', 'C'])]
        else:
            rt = trades_full[trades_full['shape_regime'] == regime]
        n = len(rt)
        if n > 0:
            wins = (rt['net_pnl'] > 0).sum()
            wr = round(wins / n * 100, 1)
            pnl = round(rt['net_pnl'].sum(), 1)
            cum = rt['net_pnl'].cumsum()
            dd = round((cum - cum.cummax()).min(), 1)
            low = ' LOW-N' if n < 8 else ''
            print(f'  {regime:<14s}: n={n:>4d}, win%={wr:>5.1f}, PnL={pnl:>+8.1f}, maxDD={dd:>+7.1f}{low}')
        else:
            print(f'  {regime:<14s}: n=0')

    # Time period breakdown
    print('\nTime period breakdown (full 2020-2026):')
    for pk, (ps, pe) in PERIODS.items():
        pt = trades_full[trades_full['time_period'] == pk]
        n = len(pt)
        if n > 0:
            wins = (pt['net_pnl'] > 0).sum()
            wr = round(wins / n * 100, 1)
            pnl = round(pt['net_pnl'].sum(), 1)
            sh = compute_adj_sharpe_for_trades(pt.reset_index(drop=True), ps, pe, df, config_base)
            cum = pt['net_pnl'].cumsum()
            dd = round((cum - cum.cummax()).min(), 1)
            print(f'  {pk}: n={n:>4d}, win%={wr:>5.1f}, PnL={pnl:>+8.1f}, adjSh={sh:.3f}, maxDD={dd:>+7.1f}')
        else:
            print(f'  {pk}: n=0')

    # Per-instrument
    print('\nPer-instrument (full 2020-2026):')
    print(f'  {"Instrument":>12s}  {"N":>4s}  {"Win%":>6s}  {"PnL":>8s}  {"TP":>3s}  {"Inv":>3s}  {"TS":>3s}  {"EB":>3s}')
    for inst in ALL_9:
        it = trades_full[trades_full['instrument'] == inst]
        n = len(it)
        if n == 0:
            continue
        wins = (it['net_pnl'] > 0).sum()
        wr = round(wins / n * 100, 1)
        pnl = round(it['net_pnl'].sum(), 1)
        tp = (it['exit_reason'] == 'take_profit').sum()
        inv = (it['exit_reason'] == 'invalidated').sum()
        ts_ = (it['exit_reason'] == 'time_stop').sum()
        eb = (it['exit_reason'] == 'expiry_buffer').sum()
        print(f'  {inst:>12s}  {n:>4d}  {wr:>5.1f}%  {pnl:>+8.1f}  {tp:>3d}  {inv:>3d}  {ts_:>3d}  {eb:>3d}')

    # ══════════════════════════════════════════════════════════════
    # PART A — CROSS-TAB: LOCKED CONFIG (entry=1.5, exit=0.5)
    # ══════════════════════════════════════════════════════════════

    print('\n' + '=' * 80)
    print('PART C.3 — SUMMARY VIEW: LOCKED CONFIG (1.5/0.5)')
    print('=' * 80)

    crosstab_locked = build_crosstab(trades_full, df, config_base)
    print(format_summary_grid(crosstab_locked, 'Locked Config: entry=1.5 / exit=0.5'))

    # ══════════════════════════════════════════════════════════════
    # PART B — THRESHOLD SENSITIVITY SWEEP (full 2020-2026)
    # ══════════════════════════════════════════════════════════════

    print('\n' + '=' * 80)
    print('PART B — THRESHOLD SENSITIVITY SWEEP (2020-2026, all 9 instruments)')
    print('=' * 80)

    sweep_results = run_sweep(df, tm_cache, intraday_df, config_base, contracts)

    # Tag all sweep trades
    for key, trades in sweep_results.items():
        if len(trades) > 0:
            trades['shape_regime'] = trades['shape'].apply(classify_shape_regime)
            trades['time_period'] = trades['entry_date'].apply(tag_time_period)

    # Portfolio-level sweep summary
    print('\n--- Portfolio-Level Sweep Results (2020-2026) ---')
    print(f'  {"entry":>5s}  {"exit":>5s}  {"n":>5s}  {"win%":>6s}  {"PnL":>8s}  {"adjSh":>7s}  {"maxDD":>7s}')

    best_key = None
    best_sharpe = -999

    for (ez, xz), trades in sorted(sweep_results.items()):
        n = len(trades)
        if n == 0:
            continue
        wins = (trades['net_pnl'] > 0).sum()
        wr = round(wins / n * 100, 1)
        pnl = round(trades['net_pnl'].sum(), 1)
        sh = compute_adj_sharpe_for_trades(trades, '2020-01-01', '2026-12-31', df, config_base)
        cum = trades['net_pnl'].cumsum()
        dd = round((cum - cum.cummax()).min(), 1)
        note = ''
        if ez == 1.5 and xz == 0.5:
            note = '  LOCKED'
        if n >= 30 and sh > best_sharpe:
            best_sharpe = sh
            best_key = (ez, xz)
            if not note:
                note = '  BEST' if n >= 30 else '  LOW-N'
        if n < 30:
            note += '  LOW-N'
        print(f'  {ez:>5.2f}  {xz:>5.2f}  {n:>5d}  {wr:>5.1f}%  {pnl:>+8.1f}  {sh:>7.3f}  {dd:>+7.1f}{note}')

    # ══════════════════════════════════════════════════════════════
    # PART B — SB+C GATED SWEEP
    # ══════════════════════════════════════════════════════════════

    print('\n--- SB+C Gated Sweep Results (2020-2026) ---')
    print(f'  {"entry":>5s}  {"exit":>5s}  {"n":>5s}  {"win%":>6s}  {"PnL":>8s}  {"adjSh":>7s}  {"maxDD":>7s}')

    best_sbc_key = None
    best_sbc_sharpe = -999

    for (ez, xz), trades in sorted(sweep_results.items()):
        if len(trades) == 0:
            continue
        sbc = trades[trades['shape_regime'].isin(['SB', 'C'])].reset_index(drop=True)
        n = len(sbc)
        if n == 0:
            continue
        wins = (sbc['net_pnl'] > 0).sum()
        wr = round(wins / n * 100, 1)
        pnl = round(sbc['net_pnl'].sum(), 1)
        sh = compute_adj_sharpe_for_trades(sbc, '2020-01-01', '2026-12-31', df, config_base)
        cum = sbc['net_pnl'].cumsum()
        dd = round((cum - cum.cummax()).min(), 1)
        note = ''
        if ez == 1.5 and xz == 0.5:
            note = '  LOCKED'
        if n >= 30 and sh > best_sbc_sharpe:
            best_sbc_sharpe = sh
            best_sbc_key = (ez, xz)
        if n < 30:
            note += '  LOW-N'
        print(f'  {ez:>5.2f}  {xz:>5.2f}  {n:>5d}  {wr:>5.1f}%  {pnl:>+8.1f}  {sh:>7.3f}  {dd:>+7.1f}{note}')

    # ══════════════════════════════════════════════════════════════
    # PART C.3 — SUMMARY: BEST FULL-RANGE CANDIDATE
    # ══════════════════════════════════════════════════════════════

    if best_key:
        print(f'\n{"=" * 80}')
        print(f'PART C.3 — SUMMARY VIEW: BEST CANDIDATE ({best_key[0]}/{best_key[1]})')
        print('=' * 80)

        best_trades = sweep_results[best_key]
        crosstab_best = build_crosstab(best_trades, df, config_base)
        print(format_summary_grid(crosstab_best,
              f'Best Candidate: entry={best_key[0]} / exit={best_key[1]}'))

    # ══════════════════════════════════════════════════════════════
    # PART C.1/C.2 — FULL CROSS-TAB TABLE (all thresholds)
    # ══════════════════════════════════════════════════════════════

    print(f'\n{"=" * 80}')
    print('PART C.1 — FULL CROSS-TAB (all threshold × shape × period combinations)')
    print('=' * 80)

    all_crosstab_rows = []
    for (ez, xz), trades in sorted(sweep_results.items()):
        if len(trades) == 0:
            continue
        ct = build_crosstab(trades, df, config_base)
        ct['entry_z'] = ez
        ct['exit_z'] = xz
        all_crosstab_rows.append(ct)

    full_crosstab = pd.concat(all_crosstab_rows, ignore_index=True)

    # Reorder columns
    col_order = ['entry_z', 'exit_z', 'shape_regime', 'period', 'n', 'win_pct',
                 'adj_sharpe', 'pnl', 'max_dd', 'low_n']
    full_crosstab = full_crosstab[col_order]

    csv_path = OUTPUT_DIR / 'shape_regime_time_crosstab.csv'
    full_crosstab.to_csv(csv_path, index=False)
    print(f'  Full cross-tab saved to: {csv_path}')
    print(f'  Total rows: {len(full_crosstab)} ({len(full_crosstab)//12} threshold combos × 12 cells)')

    # Show count of LOW-N cells
    low_n_count = full_crosstab['low_n'].sum()
    total_cells = len(full_crosstab)
    print(f'  LOW-N cells (n<8): {low_n_count} / {total_cells} ({100*low_n_count/total_cells:.1f}%)')

    # ══════════════════════════════════════════════════════════════
    # PART C.5 — CROSS-PERIOD CONSISTENCY STATEMENT
    # ══════════════════════════════════════════════════════════════

    print(f'\n{"=" * 80}')
    print('PART C.5 — CROSS-PERIOD CONSISTENCY OF SB+C EDGE')
    print('=' * 80)

    # Check locked config SB+C across periods
    locked_ct = crosstab_locked[crosstab_locked['shape_regime'] == 'SB+C']
    print('\nLocked config (1.5/0.5) SB+C by period:')
    for _, row in locked_ct.iterrows():
        low = ' LOW-N' if row['low_n'] else ''
        sh = f'{row["adj_sharpe"]:.3f}' if not np.isnan(row['adj_sharpe']) else '  nan'
        print(f'  {row["period"]}: n={row["n"]:>3d}, win%={row["win_pct"]:>5.1f}%, '
              f'PnL={row["pnl"]:>+8.1f}, adjSh={sh}, maxDD={row["max_dd"]:>+7.1f}{low}')

    # Check if all periods are profitable and have reasonable n
    sbc_periods = locked_ct.to_dict('records')
    all_positive = all(r['pnl'] > 0 for r in sbc_periods if r['n'] > 0)
    any_low_n = any(r['low_n'] for r in sbc_periods)
    sharpes = [r['adj_sharpe'] for r in sbc_periods if r['n'] >= 8 and not np.isnan(r['adj_sharpe'])]

    print(f'\n  All periods PnL > 0: {all_positive}')
    print(f'  Any LOW-N periods:   {any_low_n}')
    if sharpes:
        print(f'  Sharpe range (n>=8): {min(sharpes):.3f} to {max(sharpes):.3f}')
        spread = max(sharpes) - min(sharpes)
        print(f'  Sharpe spread:       {spread:.3f}')

    # Plain statement
    print('\n  ASSESSMENT:')
    if all_positive and not any_low_n and len(sharpes) >= 3:
        if min(sharpes) > 0.5:
            print('  The SB+C edge appears CONSISTENT across all three time periods.')
            print('  No single period dominates — the signal works in backwardated (2020-2021),')
            print('  contango (2022-2023), and mixed (2024-2026) market regimes.')
        else:
            print('  The SB+C edge is INCONSISTENT — at least one period has weak performance.')
            weakest = min(sbc_periods, key=lambda r: r['adj_sharpe'] if not np.isnan(r['adj_sharpe']) else 999)
            print(f'  Weakest: {weakest["period"]} (adjSh={weakest["adj_sharpe"]:.3f})')
    elif any_low_n:
        print('  CANNOT ASSESS — one or more periods have insufficient trade count (LOW-N).')
        print('  The SB+C filter may be too restrictive for the available data in some periods.')
        for r in sbc_periods:
            if r['low_n']:
                print(f'  LOW-N period: {r["period"]} (n={r["n"]})')
    else:
        print('  MIXED — some periods show negative PnL for SB+C.')

    # ══════════════════════════════════════════════════════════════
    # LOG
    # ══════════════════════════════════════════════════════════════

    log_path = OUTPUT_DIR / 'stage3_minute_rebuild_log.txt'
    with open(log_path, 'a') as f:
        f.write('\n\n')
        f.write('=' * 70 + '\n')
        f.write(f'SHAPE REGIME × TIME PERIOD CROSS-TABULATION — {datetime.now().strftime("%Y-%m-%d")}\n')
        f.write('=' * 70 + '\n')
        f.write('\n')
        f.write('Config: z>1.5, z_exit=0.5, dur>=3d, 1 lot, PM L0, TM RR<50%,\n')
        f.write('  all 9 instruments, spread=22 MYR, butterfly=44 MYR,\n')
        f.write('  20d time stop, no SL, no scale-in\n')
        f.write(f'Period: 2020-01-01 to latest available (60-min bars)\n')
        f.write(f'Rebuilt tenor-mapped CSVs: 2020-2027 (89 butterfly + 90 spread contracts)\n')
        f.write('\n')
        f.write(f'Full 2020-2026 baseline: n={m_full["n_trades"]}, win%={m_full["win_rate"]}, '
                f'PnL={m_full["total_pnl"]}, adjSh={sh_full:.3f}, maxDD={m_full["max_dd"]}\n')
        f.write(f'2024-2026 subset check: n=113, win%=67.3, PnL=+728.0, adjSh=1.366 CONFIRMED\n')
        f.write('\n')
        f.write('Time period boundaries:\n')
        f.write('  P1: 2020-01-01 to 2021-12-31\n')
        f.write('  P2: 2022-01-01 to 2023-12-31\n')
        f.write('  P3: 2024-01-01 to latest available\n')
        f.write('\n')
        f.write('Shape regimes: SB (shape 0.0), C (shape 1), Transitional (all others),\n')
        f.write('  SB+C (union of SB and C = resting shapes only)\n')
        f.write('\n')
        f.write(f'Full cross-tab exported to: {csv_path}\n')
        f.write(f'Total cells: {len(full_crosstab)} ({len(full_crosstab)//12} threshold combos × 12)\n')
        f.write(f'LOW-N cells (n<8): {low_n_count} / {total_cells}\n')

    print(f'\nLog appended to: {log_path}')
    print('Done.')


if __name__ == '__main__':
    main()
