"""
Invalidated-Exit Delay Sensitivity Sweep
==========================================
Tests what happens when invalidated exits (shape break mid-trade) are
delayed by N extra days instead of firing immediately.

Rules (confirmed with user):
  - If shape reverts back to original within the delay window, the trade
    UN-INVALIDATES and continues (delay cancelled).
  - Other exits (expiry_buffer, time_stop, take_profit, stop_loss, regime_risk)
    take priority and fire immediately even during a pending delay.
  - Sweep: delay = 0 (baseline), 1, 2, 3 days.
  - Locked baseline: entry=1.5, exit=0.5, duration>=3, time_stop=20.

Usage: python "research/06. stage3_minute_rebuild/invalidated_delay_sweep.py"
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
from MRBackTest.engine.backtest_engine import (
    build_panel, load_intraday_data, get_cost_points, resolve_contracts,
    get_contract_spread, expiry_exit_due, compute_metrics,
    compute_daily_portfolio_sharpe_configurable,
    ALL_9, SPREAD_INSTRUMENTS, BUTTERFLY_INSTRUMENTS,
    INSTRUMENT_TENOR_OFFSETS, RESTING_SHAPES,
    INTRADAY_START, INTRADAY_END, _panel_cache,
)


def run_intraday_with_delay(df, tm_cache, config, intraday_df, test_start, test_end,
                             invalidated_delay_days=0, contracts=None):
    """Run 60-min intraday backtest with configurable invalidated-exit delay.

    invalidated_delay_days=0 → immediate exit (baseline behavior).
    invalidated_delay_days=N → shape break starts a countdown; if shape
    reverts within N days, trade un-invalidates and continues.
    Other exits (TP, SL, time_stop, expiry_buffer, regime_risk) override
    and fire immediately even during a pending delay.
    """
    if contracts is None:
        contracts = _panel_cache.get('contracts', {})

    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    instruments = config['instruments']
    entry_z = config['entry_z']
    exit_z = config['exit_z']
    dur_thresh = config['duration_threshold']
    pm_filter = config['pm_filter_level']
    tm_thresh = config['tm_regime_risk_threshold']
    time_stop = config['time_stop_days']
    stop_loss_z = config.get('stop_loss_z')
    first_lots = config['first_entry_lots']
    scale_tiers = [t for t in config.get('scale_in_tiers', []) if t.get('enabled', True)]

    daily_mask = (df['date'] >= ts) & (df['date'] <= te)
    daily_rows = df[daily_mask].set_index('date')

    intra = intraday_df[(intraday_df['date'] >= ts) & (intraday_df['date'] <= te)].copy()
    if len(intra) == 0:
        return pd.DataFrame(), {'recovered': 0, 'still_invalidated': 0}

    intra_by_date = dict(list(intra.groupby('date')))
    all_trades = []
    recovery_stats = {'recovered': 0, 'still_invalidated': 0}

    for inst in instruments:
        cost_per_lot = get_cost_points(inst, config)
        mean_col = f'{inst}_mean'
        std_col = f'{inst}_std'

        position_open = False
        entry_date = entry_spread = entry_z_val = entry_shape = entry_direction = None
        entry_datetime = None
        days_held = 0
        total_lots = 0
        max_abs_z = 0.0
        tier_entries = []
        tier_fired = set()
        resolved_contracts = None

        # Delay state
        invalidated_pending = False
        invalidated_countdown = 0

        for date_ts in daily_rows.index:
            if date_ts not in daily_rows.index:
                continue
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

                # TM regime-risk
                if tm_thresh is not None and shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date_ts))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < tm_thresh:
                            exit_reason = 'regime_risk'

                # Invalidated logic WITH delay
                if exit_reason is None:
                    if shape != entry_shape:
                        if invalidated_delay_days == 0:
                            exit_reason = 'invalidated'
                        else:
                            if not invalidated_pending:
                                invalidated_pending = True
                                invalidated_countdown = invalidated_delay_days
                            else:
                                invalidated_countdown -= 1
                            if invalidated_countdown <= 0:
                                exit_reason = 'invalidated'
                                recovery_stats['still_invalidated'] += 1
                    else:
                        # Shape matches entry shape — if we had a pending invalidation, cancel it
                        if invalidated_pending:
                            invalidated_pending = False
                            invalidated_countdown = 0
                            recovery_stats['recovered'] += 1

                # Time stop (overrides pending delay)
                if days_held >= time_stop:
                    exit_reason = 'time_stop'
                    if invalidated_pending:
                        invalidated_pending = False
                        invalidated_countdown = 0

                # Expiry buffer (overrides pending delay)
                if exit_reason is None and expiry_exit_due(inst, resolved_contracts, date_ts):
                    exit_reason = 'expiry_buffer'
                    if invalidated_pending:
                        invalidated_pending = False
                        invalidated_countdown = 0

                if exit_reason:
                    daily_spread = get_contract_spread(contracts, resolved_contracts, date_ts) if resolved_contracts else np.nan
                    if pd.isna(daily_spread):
                        daily_spread = day[inst] if inst in day.index else np.nan
                    if pd.isna(daily_spread):
                        day_bars = intra_by_date.get(date_ts)
                        if day_bars is not None and inst in day_bars.columns:
                            valid = day_bars[inst].dropna()
                            daily_spread = valid.iloc[-1] if len(valid) > 0 else np.nan

                    if not pd.isna(daily_spread):
                        base_gross = (daily_spread - entry_spread) * entry_direction * first_lots
                        tier_gross = sum(
                            (daily_spread - te_entry['entry_spread']) * entry_direction * te_entry['lots']
                            for te_entry in tier_entries
                        )
                        gross_pnl = base_gross + tier_gross
                        total_cost = total_lots * cost_per_lot
                        net_pnl = gross_pnl - total_cost

                        all_trades.append({
                            'instrument': inst,
                            'entry_date': entry_date, 'exit_date': date_ts,
                            'entry_datetime': entry_datetime,
                            'entry_spread': round(entry_spread, 2),
                            'exit_spread': round(daily_spread, 2),
                            'entry_z': round(entry_z_val, 3),
                            'exit_z': np.nan,
                            'direction': 'LONG' if entry_direction == 1 else 'SHORT',
                            'days_held': days_held,
                            'exit_reason': exit_reason,
                            'gross_pnl': round(gross_pnl, 2),
                            'net_pnl': round(net_pnl, 2),
                            'total_lots': total_lots,
                            'n_tiers_fired': len(tier_entries),
                            'shape': entry_shape,
                            'shape_survived': exit_reason != 'invalidated',
                            'max_abs_z': round(max_abs_z, 3),
                        })

                    position_open = False
                    tier_entries = []
                    tier_fired = set()
                    max_abs_z = 0.0
                    resolved_contracts = None
                    invalidated_pending = False
                    invalidated_countdown = 0
                    continue

                # Intraday exits: TP, SL, scale-in (these override pending delay)
                day_bars = intra_by_date.get(date_ts)
                if day_bars is not None and inst in day_bars.columns and not pd.isna(daily_std) and daily_std > 0:
                    for _, bar in day_bars.iterrows():
                        bar_price = bar[inst]
                        if pd.isna(bar_price):
                            continue
                        bar_z = (bar_price - daily_mean) / daily_std
                        max_abs_z = max(max_abs_z, abs(bar_z))

                        # Scale-in
                        for ti, tier in enumerate(scale_tiers):
                            if ti not in tier_fired:
                                should_fire = False
                                if entry_direction == -1:
                                    should_fire = bar_z >= tier['z_level']
                                else:
                                    should_fire = bar_z <= -tier['z_level']
                                if should_fire:
                                    tier_fired.add(ti)
                                    tier_entries.append({
                                        'tier': ti + 1, 'z_level': tier['z_level'],
                                        'lots': tier['lots'], 'entry_spread': bar_price,
                                    })
                                    total_lots += tier['lots']

                        # Take profit (overrides pending delay)
                        if abs(bar_z) < exit_z:
                            base_gross = (bar_price - entry_spread) * entry_direction * first_lots
                            tier_gross = sum(
                                (bar_price - te_entry['entry_spread']) * entry_direction * te_entry['lots']
                                for te_entry in tier_entries
                            )
                            gross_pnl = base_gross + tier_gross
                            total_cost = total_lots * cost_per_lot
                            net_pnl = gross_pnl - total_cost

                            all_trades.append({
                                'instrument': inst,
                                'entry_date': entry_date, 'exit_date': date_ts,
                                'entry_datetime': entry_datetime,
                                'entry_spread': round(entry_spread, 2),
                                'exit_spread': round(bar_price, 2),
                                'entry_z': round(entry_z_val, 3),
                                'exit_z': round(bar_z, 3),
                                'direction': 'LONG' if entry_direction == 1 else 'SHORT',
                                'days_held': days_held,
                                'exit_reason': 'take_profit',
                                'gross_pnl': round(gross_pnl, 2),
                                'net_pnl': round(net_pnl, 2),
                                'total_lots': total_lots,
                                'n_tiers_fired': len(tier_entries),
                                'shape': entry_shape,
                                'shape_survived': True,
                                'max_abs_z': round(max_abs_z, 3),
                            })
                            position_open = False
                            tier_entries = []
                            tier_fired = set()
                            max_abs_z = 0.0
                            resolved_contracts = None
                            invalidated_pending = False
                            invalidated_countdown = 0
                            break

                        # Stop loss (overrides pending delay)
                        if stop_loss_z is not None:
                            sl_triggered = False
                            if entry_direction == -1 and bar_z >= stop_loss_z:
                                sl_triggered = True
                            elif entry_direction == 1 and bar_z <= -stop_loss_z:
                                sl_triggered = True
                            if sl_triggered:
                                base_gross = (bar_price - entry_spread) * entry_direction * first_lots
                                tier_gross = sum(
                                    (bar_price - te_entry['entry_spread']) * entry_direction * te_entry['lots']
                                    for te_entry in tier_entries
                                )
                                gross_pnl = base_gross + tier_gross
                                total_cost = total_lots * cost_per_lot
                                net_pnl = gross_pnl - total_cost

                                all_trades.append({
                                    'instrument': inst,
                                    'entry_date': entry_date, 'exit_date': date_ts,
                                    'entry_datetime': entry_datetime,
                                    'entry_spread': round(entry_spread, 2),
                                    'exit_spread': round(bar_price, 2),
                                    'entry_z': round(entry_z_val, 3),
                                    'exit_z': round(bar_z, 3),
                                    'direction': 'LONG' if entry_direction == 1 else 'SHORT',
                                    'days_held': days_held,
                                    'exit_reason': 'stop_loss',
                                    'gross_pnl': round(gross_pnl, 2),
                                    'net_pnl': round(net_pnl, 2),
                                    'total_lots': total_lots,
                                    'n_tiers_fired': len(tier_entries),
                                    'shape': entry_shape,
                                    'shape_survived': True,
                                    'max_abs_z': round(max_abs_z, 3),
                                })
                                position_open = False
                                tier_entries = []
                                tier_fired = set()
                                max_abs_z = 0.0
                                resolved_contracts = None
                                invalidated_pending = False
                                invalidated_countdown = 0
                                break

            # Entry check
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
                                if expiry_exit_due(inst, resolved_contracts, date_ts):
                                    resolved_contracts = None
                                    continue
                                position_open = True
                                entry_date = date_ts
                                entry_datetime = bar['datetime']
                                entry_spread = bar_price
                                entry_z_val = bar_z
                                entry_shape = shape
                                entry_direction = -1 if bar_z > 0 else 1
                                days_held = 0
                                total_lots = first_lots
                                tier_entries = []
                                tier_fired = set()
                                max_abs_z = abs(bar_z)
                                invalidated_pending = False
                                invalidated_countdown = 0
                                break

    return pd.DataFrame(all_trades), recovery_stats


def main():
    print('Loading data...')
    df, tm_cache = build_panel()
    intraday_df = load_intraday_data()
    contracts = _panel_cache.get('contracts', {})

    config = {
        'entry_z': 1.5, 'exit_z': 0.5, 'duration_threshold': 3,
        'first_entry_lots': 1, 'time_stop_days': 20,
        'stop_loss_z': None, 'scale_in_tiers': [],
        'instruments': ALL_9, 'pm_filter_level': 0,
        'tm_regime_risk_threshold': 0.50,
        'cost_spread_myr': 22.0, 'cost_butterfly_myr': 44.0,
    }

    delays = [0, 1, 2, 3]

    print()
    print('=' * 80)
    print('INVALIDATED-EXIT DELAY SENSITIVITY SWEEP')
    print('Locked baseline: entry=1.5, exit=0.5, duration>=3d, time_stop=20d')
    print('Cost: spread=22, butterfly=44 MYR')
    print('Period: 2024-2026 (minute data, 60-min bars)')
    print('=' * 80)

    # Portfolio-level results
    print()
    print('--- PORTFOLIO-LEVEL RESULTS ---')
    print(f'{"Delay":>5s}  {"Trades":>6s}  {"Win%":>6s}  {"PnL":>8s}  {"AdjSh":>7s}  '
          f'{"MaxDD":>7s}  {"AvgHP":>5s}  {"Inv":>4s}  {"Recovered":>9s}  {"StillInv":>8s}')

    all_results = {}
    for delay in delays:
        trades, recovery = run_intraday_with_delay(
            df, tm_cache, config, intraday_df,
            INTRADAY_START, INTRADAY_END,
            invalidated_delay_days=delay,
            contracts=contracts,
        )
        m = compute_metrics(trades, df, INTRADAY_START, INTRADAY_END)
        adj_sharpe = compute_daily_portfolio_sharpe_configurable(
            trades, INTRADAY_START, INTRADAY_END, df, config)

        inv_count = (trades['exit_reason'] == 'invalidated').sum() if len(trades) > 0 else 0

        print(f'{delay:>5d}  {m["n_trades"]:>6d}  {m["win_rate"]:>5.1f}%  {m["total_pnl"]:>+8.1f}  '
              f'{adj_sharpe:>7.3f}  {m["max_dd"]:>+7.1f}  {m["avg_hp"]:>5.1f}  '
              f'{inv_count:>4d}  {recovery["recovered"]:>9d}  {recovery["still_invalidated"]:>8d}')

        all_results[delay] = {
            'trades': trades, 'metrics': m, 'adj_sharpe': adj_sharpe,
            'recovery': recovery, 'inv_count': inv_count,
        }

    # Per-instrument breakdown
    print()
    print('--- PER-INSTRUMENT BREAKDOWN ---')
    for delay in delays:
        trades = all_results[delay]['trades']
        print(f'\n  Delay = {delay}d:')
        print(f'  {"Instrument":>12s}  {"N":>4s}  {"Win%":>6s}  {"PnL":>8s}  {"TP":>3s}  {"Inv":>3s}  {"TS":>3s}  {"EB":>3s}  {"RR":>3s}')
        for inst in ALL_9:
            if len(trades) == 0:
                continue
            it = trades[trades['instrument'] == inst]
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
            rr = (it['exit_reason'] == 'regime_risk').sum()
            print(f'  {inst:>12s}  {n:>4d}  {wr:>5.1f}%  {pnl:>+8.1f}  {tp:>3d}  {inv:>3d}  {ts_:>3d}  {eb:>3d}  {rr:>3d}')

    # Summary
    print()
    print('--- SUMMARY ---')
    baseline_sharpe = all_results[0]['adj_sharpe']
    for delay in delays:
        r = all_results[delay]
        delta = r['adj_sharpe'] - baseline_sharpe
        print(f'  Delay {delay}d: adj Sharpe = {r["adj_sharpe"]:.3f} (delta = {delta:+.3f})')

    best_delay = max(delays, key=lambda d: all_results[d]['adj_sharpe'])
    print(f'\n  Best delay: {best_delay}d (adj Sharpe = {all_results[best_delay]["adj_sharpe"]:.3f})')
    if best_delay > 0:
        print(f'  Improvement over baseline: {all_results[best_delay]["adj_sharpe"] - baseline_sharpe:+.3f}')
    else:
        print('  No delay improves over baseline.')


if __name__ == '__main__':
    main()
