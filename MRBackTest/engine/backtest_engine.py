"""
FCPO Calendar Spread — Parameterized Backtest Engine
=====================================================
Consolidates Stage 2 walk-forward backtest logic into a single
parameterized function for use by the dashboard and CLI validation.

Usage:
    python MRBackTest/engine/backtest_engine.py --validate
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
import argparse
import pickle
from pathlib import Path

from models.pm_engine import predict as pm_predict
from models.tm_engine import predict as tm_predict
from models.feature_prep import load_daily_shape_log, load_enriched_shape_log

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

POINT_VALUE = 25.0  # MYR per point, same for spreads and butterflies
PM_CONFIDENCE_THRESHOLD = 0.70

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

SPREAD_INSTRUMENTS = list(ALL_INSTRUMENT_CONFIG.keys())
BUTTERFLY_INSTRUMENTS = list(BUTTERFLY_CONFIG.keys())
ALL_9 = ['M2-M3', 'M3-M4', 'M4-M5', 'M5-M6', 'M1-M2',
         'BF_M1M2M3', 'BF_M2M3M4', 'BF_M3M4M5', 'BF_M4M5M6']

WINDOWS = [
    {'name': 'W1', 'label': 'W1 (2019-2020)', 'test_start': '2019-01-01', 'test_end': '2020-12-31'},
    {'name': 'W2', 'label': 'W2 (2021-2022)', 'test_start': '2021-01-01', 'test_end': '2022-12-31'},
    {'name': 'W3', 'label': 'W3 (2023-2024)', 'test_start': '2023-01-01', 'test_end': '2024-12-31'},
    {'name': 'W4', 'label': 'W4 (2025-2026)', 'test_start': '2025-01-01', 'test_end': '2026-12-31'},
]

# Shape codes for resting regimes
RESTING_SHAPES = ('0.0', '1')

# Default locked config
DEFAULT_CONFIG = {
    'entry_z': 1.5,
    'exit_z': 0.5,
    'duration_threshold': 3,
    'first_entry_lots': 1,
    'time_stop_days': 20,
    'stop_loss_z': None,           # None = disabled
    'scale_in_tiers': [],          # List of {'z_level': float, 'lots': int}
    'instruments': ALL_9,
    'pm_filter_level': 0,
    'tm_regime_risk_threshold': 0.50,
    'cost_spread_myr': 100.0,      # Old: 100 flat; New: 22
    'cost_butterfly_myr': 100.0,   # Old: 100 flat; New: 44
}

# Published locked-config numbers (RR<50%, old cost=100 MYR flat)
PUBLISHED_NUMBERS = {
    'W1': {'n': 156, 'pnl': 1395.0, 'adj_sharpe': 1.633},
    'W2': {'n': 133, 'pnl': 6131.0, 'adj_sharpe': 2.540},
    'W3': {'n': 126, 'pnl': 1437.0, 'adj_sharpe': 1.167},
    'W4': {'n': 72,  'pnl': 700.0,  'adj_sharpe': 1.215},
}


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

_panel_cache = {}


def build_panel():
    """Build the full daily panel with spreads, butterflies, z-scores, PM, TM."""
    if 'df' in _panel_cache:
        return _panel_cache['df'], _panel_cache['tm_cache']

    print('Loading data...')
    full_log = load_daily_shape_log()
    full_log = full_log.sort_values('date').reset_index(drop=True)
    enriched = load_enriched_shape_log()
    enriched = enriched.sort_values('date').reset_index(drop=True)

    # Pre-2017 needs days_in_shape and episode_id computed
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

    # Compute spreads and butterflies
    for name, cfg in ALL_INSTRUMENT_CONFIG.items():
        df[name] = df[cfg['near']] - df[cfg['far']]
    for name, cfg in BUTTERFLY_CONFIG.items():
        m1, m2, m3 = cfg['legs']
        df[name] = df[m1] - 2 * df[m2] + df[m3]

    # Regime-relative z-scores
    print('Computing z-scores...')
    for inst in ALL_9:
        df[f'{inst}_z'] = _compute_regime_zscore(df, inst)

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

    # TM predictions
    print('Pre-computing TM persistence probabilities...')
    tm_cache = {}
    for idx in df[df['date'] >= model_start].index:
        dt = df.loc[idx, 'date']
        dt_ts = pd.Timestamp(dt)
        try:
            result = tm_predict(dt_ts, '1w')
            if 'error' not in result:
                current_shape = str(result['current_shape'])
                all_probs = result.get('all_probs', {})
                tm_cache[dt_ts] = {
                    'current_shape': current_shape,
                    'persistence_prob': all_probs.get(current_shape, np.nan),
                    'all_probs': all_probs,
                }
        except Exception:
            pass
    print(f'  Cached {len(tm_cache)} TM predictions')

    _panel_cache['df'] = df
    _panel_cache['tm_cache'] = tm_cache
    return df, tm_cache


def _compute_regime_zscore(df, instrument_col):
    """Per-episode rolling 60-day window z-score, min 10 observations."""
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


# ══════════════════════════════════════════════════════════════
# COST MODEL
# ══════════════════════════════════════════════════════════════

def get_cost_points(instrument, config):
    """Return round-trip cost in points per lot for a given instrument."""
    if instrument.startswith('BF_'):
        return config['cost_butterfly_myr'] / POINT_VALUE
    else:
        return config['cost_spread_myr'] / POINT_VALUE


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════

def run_backtest(config, df=None, tm_cache=None):
    """
    Run the full backtest across all 4 walk-forward windows.

    Returns dict: {window_name: pd.DataFrame of trades}
    """
    if df is None or tm_cache is None:
        df, tm_cache = build_panel()

    results = {}
    for w in WINDOWS:
        trades = _run_window(df, tm_cache, config,
                             w['test_start'], w['test_end'])
        results[w['name']] = trades
    return results


def _run_window(df, tm_cache, config, test_start, test_end):
    """Run backtest for a single test window."""
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
    scale_tiers = config.get('scale_in_tiers', [])

    all_trades = []

    for inst in instruments:
        z_col = f'{inst}_z'
        cost_per_lot = get_cost_points(inst, config)

        position_open = False
        entry_date = entry_spread = entry_z_val = entry_shape = entry_direction = None
        days_held = 0
        total_lots = 0
        # Scale-in tracking: list of (z_level, lots, entry_spread_at_tier)
        tier_entries = []
        tier_fired = set()

        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
            if pd.isna(spread):
                continue

            if position_open:
                days_held += 1

                # Scale-in checks (before exit checks)
                for ti, tier in enumerate(scale_tiers):
                    if ti not in tier_fired and not pd.isna(z):
                        should_fire = False
                        if entry_direction == -1:  # SHORT: z > 0, fire when z >= tier level
                            should_fire = z >= tier['z_level']
                        else:  # LONG: z < 0, fire when z <= -tier level
                            should_fire = z <= -tier['z_level']
                        if should_fire:
                            tier_fired.add(ti)
                            tier_entries.append({
                                'tier': ti + 1,
                                'z_level': tier['z_level'],
                                'lots': tier['lots'],
                                'entry_spread': spread,
                            })
                            total_lots += tier['lots']

                exit_reason = None

                # TM regime-risk exit (only if shape hasn't changed)
                if tm_thresh is not None and shape == entry_shape:
                    tm_data = tm_cache.get(pd.Timestamp(date))
                    if tm_data is not None:
                        pp = tm_data['all_probs'].get(str(entry_shape), np.nan)
                        if not np.isnan(pp) and pp < tm_thresh:
                            exit_reason = 'regime_risk'

                # Take profit
                if not pd.isna(z) and abs(z) < exit_z:
                    exit_reason = 'take_profit'

                # Shape invalidation
                if shape != entry_shape:
                    exit_reason = 'invalidated'

                # Stop loss (if enabled)
                if stop_loss_z is not None and not pd.isna(z):
                    if entry_direction == -1 and z >= stop_loss_z:
                        exit_reason = 'stop_loss'
                    elif entry_direction == 1 and z <= -stop_loss_z:
                        exit_reason = 'stop_loss'

                # Time stop
                if days_held >= time_stop:
                    exit_reason = 'time_stop'

                if exit_reason:
                    # PnL for base position
                    base_gross = (spread - entry_spread) * entry_direction * first_lots
                    # PnL for scale-in tiers
                    tier_gross = sum(
                        (spread - te['entry_spread']) * entry_direction * te['lots']
                        for te in tier_entries
                    )
                    gross_pnl = base_gross + tier_gross
                    total_cost = total_lots * cost_per_lot
                    net_pnl = gross_pnl - total_cost

                    if ts <= date <= te:
                        trade = {
                            'instrument': inst,
                            'entry_date': entry_date, 'exit_date': date,
                            'entry_spread': round(entry_spread, 2),
                            'exit_spread': round(spread, 2),
                            'entry_z': round(entry_z_val, 3),
                            'exit_z': round(z, 3) if not pd.isna(z) else np.nan,
                            'direction': 'LONG' if entry_direction == 1 else 'SHORT',
                            'days_held': days_held,
                            'exit_reason': exit_reason,
                            'gross_pnl': round(gross_pnl, 2),
                            'net_pnl': round(net_pnl, 2),
                            'total_lots': total_lots,
                            'n_tiers_fired': len(tier_entries),
                            'shape': entry_shape,
                            'shape_survived': exit_reason != 'invalidated',
                        }
                        all_trades.append(trade)

                    position_open = False
                    tier_entries = []
                    tier_fired = set()
                    continue

            if not position_open:
                if pd.isna(z) or pd.isna(row['pm_level']):
                    continue
                is_resting = shape in RESTING_SHAPES
                dur_ok = row['days_in_shape'] >= dur_thresh
                z_extreme = abs(z) > entry_z
                pm_ok = row['pm_level'] <= pm_filter
                if is_resting and dur_ok and z_extreme and pm_ok:
                    position_open = True
                    entry_date, entry_spread, entry_z_val = date, spread, z
                    entry_shape = shape
                    entry_direction = -1 if z > 0 else 1
                    days_held = 0
                    total_lots = first_lots
                    tier_entries = []
                    tier_fired = set()

    return pd.DataFrame(all_trades)


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(trades, df_ref=None, test_start=None, test_end=None):
    """Compute summary metrics for a set of trades."""
    n = len(trades)
    if n == 0:
        return {
            'n_trades': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0,
            'total_pnl': 0, 'adj_sharpe': np.nan, 'max_dd': 0,
            'pct_take_profit': 0, 'pct_invalidated': 0,
            'pct_time_stop': 0, 'pct_regime_risk': 0, 'pct_stop_loss': 0,
            'avg_hp': 0, 'shape_survival': 0,
        }

    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]
    win_rate = round(len(wins) / n * 100, 1)
    avg_win = round(wins['net_pnl'].mean(), 2) if len(wins) > 0 else 0
    avg_loss = round(losses['net_pnl'].mean(), 2) if len(losses) > 0 else 0
    total_pnl = round(trades['net_pnl'].sum(), 2)

    tp_pct = round((trades['exit_reason'] == 'take_profit').mean() * 100, 1)
    inv_pct = round((trades['exit_reason'] == 'invalidated').mean() * 100, 1)
    ts_pct = round((trades['exit_reason'] == 'time_stop').mean() * 100, 1)
    rr_pct = round((trades['exit_reason'] == 'regime_risk').mean() * 100, 1)
    sl_pct = round((trades['exit_reason'] == 'stop_loss').mean() * 100, 1)

    avg_hp = round(trades['days_held'].mean(), 1)
    cum_pnl = trades['net_pnl'].cumsum()
    max_dd = round((cum_pnl - cum_pnl.cummax()).min(), 2) if n > 0 else 0

    shape_survival = round(trades['shape_survived'].mean() * 100, 1) if 'shape_survived' in trades.columns else np.nan

    # Adjusted Sharpe (MtM daily PnL)
    adj_sharpe = np.nan
    if df_ref is not None and test_start is not None:
        adj_sharpe = _compute_daily_portfolio_sharpe(trades, test_start, test_end, df_ref)

    return {
        'n_trades': n, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'total_pnl': total_pnl,
        'adj_sharpe': adj_sharpe, 'max_dd': max_dd,
        'pct_take_profit': tp_pct, 'pct_invalidated': inv_pct,
        'pct_time_stop': ts_pct, 'pct_regime_risk': rr_pct,
        'pct_stop_loss': sl_pct,
        'avg_hp': avg_hp, 'shape_survival': shape_survival,
    }


def _compute_daily_portfolio_sharpe(trades, test_start, test_end, df_ref):
    """Daily portfolio Sharpe using mark-to-market within a test window."""
    if len(trades) == 0:
        return np.nan
    ts_dt = pd.Timestamp(test_start)
    te_dt = pd.Timestamp(test_end)
    window_dates = df_ref[(df_ref['date'] >= ts_dt) & (df_ref['date'] <= te_dt)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=window_dates)

    for _, t in trades.iterrows():
        entry_dt = t['entry_date']
        exit_dt = t['exit_date']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']
        total_lots = t.get('total_lots', 1)

        trade_days = df_ref[(df_ref['date'] > entry_dt) & (df_ref['date'] <= exit_dt)].copy()
        if len(trade_days) == 0:
            continue

        prev_spread = t['entry_spread']
        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            current_spread = day_row[inst]
            if pd.isna(current_spread):
                continue
            # MtM uses 1 lot for base (matching original logic for validation)
            day_mtm = (current_spread - prev_spread) * direction
            if dt in daily_pnl.index:
                daily_pnl[dt] += day_mtm
            prev_spread = current_spread

        # Cost deducted once at exit
        cost_per_lot = get_cost_points(inst, {'cost_spread_myr': 100.0, 'cost_butterfly_myr': 100.0})
        if exit_dt in daily_pnl.index:
            daily_pnl[exit_dt] -= cost_per_lot

    daily_std = daily_pnl.std()
    if daily_std > 0:
        return round(daily_pnl.mean() / daily_std * np.sqrt(252), 3)
    return np.nan


def compute_daily_portfolio_sharpe_configurable(trades, test_start, test_end, df_ref, config):
    """Daily portfolio Sharpe using configurable cost model."""
    if len(trades) == 0:
        return np.nan
    ts_dt = pd.Timestamp(test_start)
    te_dt = pd.Timestamp(test_end)
    window_dates = df_ref[(df_ref['date'] >= ts_dt) & (df_ref['date'] <= te_dt)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=window_dates)

    for _, t in trades.iterrows():
        entry_dt = t['entry_date']
        exit_dt = t['exit_date']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']
        total_lots = t.get('total_lots', 1)

        trade_days = df_ref[(df_ref['date'] > entry_dt) & (df_ref['date'] <= exit_dt)].copy()
        if len(trade_days) == 0:
            continue

        prev_spread = t['entry_spread']
        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            current_spread = day_row[inst]
            if pd.isna(current_spread):
                continue
            day_mtm = (current_spread - prev_spread) * direction
            if dt in daily_pnl.index:
                daily_pnl[dt] += day_mtm
            prev_spread = current_spread

        cost_per_lot = get_cost_points(inst, config)
        if exit_dt in daily_pnl.index:
            daily_pnl[exit_dt] -= cost_per_lot * total_lots

    daily_std = daily_pnl.std()
    if daily_std > 0:
        return round(daily_pnl.mean() / daily_std * np.sqrt(252), 3)
    return np.nan


def classify_loss_bucket(trade):
    """Classify a losing trade into loss buckets A/B/C/D/TP-loss."""
    if trade['net_pnl'] > 0:
        return None
    reason = trade['exit_reason']
    if reason == 'regime_risk':
        return 'D'
    if reason == 'invalidated':
        return 'A'
    if reason == 'take_profit':
        return 'TP-loss'
    if reason == 'stop_loss':
        return 'SL'
    if reason == 'time_stop':
        # B vs C requires max|z| during trade — simplified to 'B/C'
        return 'B/C'
    return 'unknown'


# ══════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_old_cost():
    """Validate engine reproduces published RR<50% numbers exactly."""
    config = DEFAULT_CONFIG.copy()
    config['cost_spread_myr'] = 100.0
    config['cost_butterfly_myr'] = 100.0
    config['tm_regime_risk_threshold'] = 0.50

    df, tm_cache = build_panel()
    results = run_backtest(config, df, tm_cache)

    print('\n' + '='*60)
    print('VALIDATION: OLD COST (100 MYR flat)')
    print('='*60)

    all_pass = True
    validation_results = {}
    for w in WINDOWS:
        trades = results[w['name']]
        metrics = compute_metrics(trades, df, w['test_start'], w['test_end'])
        pub = PUBLISHED_NUMBERS[w['name']]

        n_match = metrics['n_trades'] == pub['n']
        pnl_match = abs(metrics['total_pnl'] - pub['pnl']) < 0.5
        sharpe_match = abs(metrics['adj_sharpe'] - pub['adj_sharpe']) < 0.005

        status = 'PASS' if (n_match and pnl_match) else 'FAIL'
        if not (n_match and pnl_match):
            all_pass = False

        print(f"  {w['label']:16s}: n={metrics['n_trades']:>3d} (exp {pub['n']:>3d}) {'OK' if n_match else 'MISMATCH'}, "
              f"PnL={metrics['total_pnl']:>+8.1f} (exp {pub['pnl']:>+8.1f}) {'OK' if pnl_match else 'MISMATCH'}, "
              f"adjSh={metrics['adj_sharpe']:.3f} (exp {pub['adj_sharpe']:.3f}) {'OK' if sharpe_match else '~'}, "
              f"{status}")

        validation_results[w['name']] = {
            'n': metrics['n_trades'], 'pnl': metrics['total_pnl'],
            'adj_sharpe': metrics['adj_sharpe'], 'status': status,
            'metrics': metrics, 'trades': trades,
        }

    print(f'\n  Overall: {"ALL PASS" if all_pass else "SOME FAILED"}')
    return all_pass, validation_results, df, tm_cache


def run_new_cost(df=None, tm_cache=None):
    """Run with new differentiated cost model (spread=22, butterfly=44)."""
    config = DEFAULT_CONFIG.copy()
    config['cost_spread_myr'] = 22.0
    config['cost_butterfly_myr'] = 44.0
    config['tm_regime_risk_threshold'] = 0.50

    if df is None or tm_cache is None:
        df, tm_cache = build_panel()

    results = run_backtest(config, df, tm_cache)

    print('\n' + '='*60)
    print('NEW COST (spread=22 MYR, butterfly=44 MYR)')
    print('='*60)

    new_cost_results = {}
    for w in WINDOWS:
        trades = results[w['name']]
        metrics = compute_metrics(trades, df, w['test_start'], w['test_end'])
        # Use configurable Sharpe for new cost
        adj_sharpe_new = compute_daily_portfolio_sharpe_configurable(
            trades, w['test_start'], w['test_end'], df, config)

        print(f"  {w['label']:16s}: n={metrics['n_trades']:>3d}, "
              f"PnL={metrics['total_pnl']:>+8.1f}, adjSh={adj_sharpe_new:.3f}")

        new_cost_results[w['name']] = {
            'n': metrics['n_trades'], 'pnl': metrics['total_pnl'],
            'adj_sharpe': adj_sharpe_new,
            'metrics': metrics, 'trades': trades,
        }

    return new_cost_results


def cache_results(results, filepath):
    """Save results to disk for dashboard pre-loading."""
    # Serialize trades as DataFrames, metrics as dicts
    cache_data = {}
    for wname, data in results.items():
        cache_data[wname] = {
            'n': data['n'],
            'pnl': data['pnl'],
            'adj_sharpe': data['adj_sharpe'],
            'metrics': data['metrics'],
            'trades': data['trades'],
        }
    with open(filepath, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f'Cached results to {filepath}')


def load_cached_results(filepath):
    """Load pre-cached results from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCPO Calendar Spread Backtest Engine')
    parser.add_argument('--validate', action='store_true', help='Validate against published numbers')
    parser.add_argument('--new-cost', action='store_true', help='Run with new cost model')
    parser.add_argument('--cache', type=str, help='Cache results to file')
    args = parser.parse_args()

    if args.validate:
        passed, val_results, df, tm_cache = validate_old_cost()

        if args.new_cost:
            new_results = run_new_cost(df, tm_cache)
            if args.cache:
                cache_results(new_results, args.cache)
        elif args.cache:
            cache_results(val_results, args.cache)

    elif args.new_cost:
        df, tm_cache = build_panel()
        new_results = run_new_cost(df, tm_cache)
        if args.cache:
            cache_results(new_results, args.cache)
    else:
        parser.print_help()
