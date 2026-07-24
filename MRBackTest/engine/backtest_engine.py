"""
FCPO Calendar Spread — Parameterized Backtest Engine
=====================================================
Consolidates Stage 2 walk-forward backtest logic into a single
parameterized function for use by the dashboard and CLI validation.

Supports both daily and 60-min intraday resolution.

Usage:
    python MRBackTest/engine/backtest_engine.py --validate
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
import argparse
import pickle
import os
from pathlib import Path

from models.pm_engine import predict as pm_predict
from models.tm_engine import predict as tm_predict
from models.feature_prep import load_daily_shape_log, load_enriched_shape_log
from MRBackTest.shared.tenor_mapping import front_month, tenor_to_contract_month, add_months

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

# Tenor → instrument mapping for 60-min wide CSVs
SPREAD_TENOR_MAP = {
    'Current': 'M1-M2', '+1M': 'M2-M3', '+2M': 'M3-M4',
    '+3M': 'M4-M5', '+4M': 'M5-M6',
}
BUTTERFLY_TENOR_MAP = {
    'Current': 'BF_M1M2M3', '+1M': 'BF_M2M3M4',
    '+2M': 'BF_M3M4M5', '+3M': 'BF_M4M5M6',
}

WINDOWS = [
    {'name': 'W1', 'label': 'W1 (2019-2020)', 'test_start': '2019-01-01', 'test_end': '2020-12-31'},
    {'name': 'W2', 'label': 'W2 (2021-2022)', 'test_start': '2021-01-01', 'test_end': '2022-12-31'},
    {'name': 'W3', 'label': 'W3 (2023-2024)', 'test_start': '2023-01-01', 'test_end': '2024-12-31'},
    {'name': 'W4', 'label': 'W4 (2025-2026)', 'test_start': '2025-01-01', 'test_end': '2026-12-31'},
]

INTRADAY_START = '2020-01-01'  # extended from 2024 after tenor-mapped CSV rebuild (2026-07-24)
INTRADAY_END = '2026-12-31'  # loads up to most recent available

RESTING_SHAPES = ('0.0', '1')

# ── Expiry buffer (all instruments) ──
# FCPO last trading day = 15th of delivery month (or preceding market day).
# Generic rule: force-close any open position when the current date falls
# within EXPIRY_BUFFER_DAYS calendar days of ANY leg's LTD. This is checked
# per resolved contract leg — if any leg (near, far, or middle for butterflies)
# is in its delivery month and date.day >= FORCE_CLOSE_DAY, the position exits.
#
# In practice, only M1-M2 (near_offset=0, spread roll day 16) triggers this
# regularly, because all other instruments have near legs at least 1 month
# from LTD at entry. The generic check provides a safety net for all 9.
#
# SUPERSEDES the prior M1-M2-only m1m2_expiry_exit_due() (2026-07-21).
EXPIRY_BUFFER_DAYS = 7  # calendar days before 15th LTD
EXPIRY_FORCE_CLOSE_DAY = 15 - EXPIRY_BUFFER_DAYS  # = 8th of delivery month


def expiry_exit_due(inst, resolved_contracts, date):
    """Check if ANY leg of a position is within the expiry buffer zone.
    Returns True if date is in any leg's delivery month and date.day >= 8.
    Works for all instruments (spreads: 2 legs, butterflies: 3 legs)."""
    if resolved_contracts is None:
        return False
    if not hasattr(date, 'year'):
        return False
    for ym in resolved_contracts:
        leg_year, leg_month = ym
        if date.year == leg_year and date.month == leg_month and date.day >= EXPIRY_FORCE_CLOSE_DAY:
            return True
    return False


# Backward compatibility alias
def m1m2_expiry_exit_due(inst, resolved_contracts, date):
    """Deprecated: use expiry_exit_due() instead. Kept for existing scripts."""
    return expiry_exit_due(inst, resolved_contracts, date)


# ── Contract pinning infrastructure ──
TERM_DIR = r'C:/ClaudeCode/Raw Data/Term Structure'
MONTH_ABBRS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

INSTRUMENT_TENOR_OFFSETS = {
    'M1-M2': {'type': 'spread', 'near_offset': 0, 'far_offset': 1},
    'M2-M3': {'type': 'spread', 'near_offset': 1, 'far_offset': 2},
    'M3-M4': {'type': 'spread', 'near_offset': 2, 'far_offset': 3},
    'M4-M5': {'type': 'spread', 'near_offset': 3, 'far_offset': 4},
    'M5-M6': {'type': 'spread', 'near_offset': 4, 'far_offset': 5},
    'BF_M1M2M3': {'type': 'butterfly', 'offsets': [0, 1, 2]},
    'BF_M2M3M4': {'type': 'butterfly', 'offsets': [1, 2, 3]},
    'BF_M3M4M5': {'type': 'butterfly', 'offsets': [2, 3, 4]},
    'BF_M4M5M6': {'type': 'butterfly', 'offsets': [3, 4, 5]},
}


def load_contract_prices():
    """Load all per-contract daily close prices.
    Returns dict: {(year, month): pd.Series indexed by pd.Timestamp date}"""
    contracts = {}
    if not os.path.exists(TERM_DIR):
        return contracts
    for year_dir in sorted(os.listdir(TERM_DIR)):
        if not year_dir.isdigit():
            continue
        year = int(year_dir)
        for m, abbr in enumerate(MONTH_ABBRS, 1):
            path = os.path.join(TERM_DIR, year_dir, f"FCPO {abbr}{str(year)[2:]}_Daily.csv")
            if not os.path.exists(path):
                continue
            df_c = pd.read_csv(path)
            df_c["date"] = pd.to_datetime(df_c["Timestamp (UTC)"], format='mixed').dt.normalize()
            series = df_c.set_index("date")["Close"]
            contracts[(year, m)] = series[~series.index.duplicated(keep="last")]
    return contracts


def resolve_contracts(inst, date):
    """At entry, resolve instrument to specific contract months.
    Returns tuple of (year, month) tuples for each leg."""
    cfg = INSTRUMENT_TENOR_OFFSETS[inst]
    inst_type = cfg['type']
    if inst_type == 'spread':
        near = tenor_to_contract_month(date, cfg['near_offset'], inst_type)
        far = tenor_to_contract_month(date, cfg['far_offset'], inst_type)
        return (near, far)
    else:  # butterfly
        return tuple(tenor_to_contract_month(date, o, inst_type)
                     for o in cfg['offsets'])


def get_contract_spread(contracts, resolved, date):
    """Look up spread/butterfly price from pinned contracts on a given date."""
    if len(resolved) == 2:  # spread
        near_ym, far_ym = resolved
        near_s = contracts.get(near_ym, pd.Series(dtype=float))
        far_s = contracts.get(far_ym, pd.Series(dtype=float))
        near_p = near_s.get(date, np.nan) if len(near_s) > 0 else np.nan
        far_p = far_s.get(date, np.nan) if len(far_s) > 0 else np.nan
        if pd.isna(near_p) or pd.isna(far_p):
            return np.nan
        return near_p - far_p
    else:  # butterfly (3 legs)
        prices = []
        for ym in resolved:
            s = contracts.get(ym, pd.Series(dtype=float))
            p = s.get(date, np.nan) if len(s) > 0 else np.nan
            if pd.isna(p):
                return np.nan
            prices.append(p)
        return prices[0] - 2 * prices[1] + prices[2]


def build_pinned_episode_history(contracts, resolved, df, episode_id, up_to_date):
    """Build price history for pinned contracts within current episode up to a date.
    Returns list of floats (prices). Used to seed z-score window at entry."""
    ep_mask = df['episode_id'] == episode_id
    ep_dates = df.loc[ep_mask, 'date']
    ep_dates = ep_dates[ep_dates <= up_to_date]
    prices = []
    for d in ep_dates:
        p = get_contract_spread(contracts, resolved, d)
        if not pd.isna(p):
            prices.append(p)
    return prices


def compute_pinned_zscore(episode_prices, current_price):
    """Compute z-score for pinned contract pair using episode history.
    episode_prices: list of prices (floats) in chronological order.
    Uses 60-day rolling window, min 10 obs."""
    if pd.isna(current_price):
        return np.nan
    if len(episode_prices) < 10:
        return np.nan
    window = episode_prices[-60:]
    mean = np.mean(window)
    std = np.std(window, ddof=1)
    if std <= 0:
        return np.nan
    return (current_price - mean) / std


# Default locked config
DEFAULT_CONFIG = {
    'entry_z': 1.5,
    'exit_z': 0.5,
    'duration_threshold': 3,
    'first_entry_lots': 1,
    'time_stop_days': 20,
    'stop_loss_z': None,
    'scale_in_tiers': [],          # List of {'z_level': float, 'lots': int, 'enabled': bool}
    'instruments': ALL_9,
    'pm_filter_level': 0,
    'tm_regime_risk_threshold': 0.50,
    'cost_spread_myr': 100.0,
    'cost_butterfly_myr': 100.0,
    'data_resolution': 'daily',    # 'daily' or '60min'
    'date_range': 'all',           # 'all', 'W1', 'W2', 'W3', 'W4', or 'intraday'
}

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
_intraday_cache = {}


def build_panel():
    """Build the full daily panel with spreads, butterflies, z-scores, PM, TM.
    Also stores per-instrument daily mean/std for intraday z-score computation.
    Also loads per-contract prices for contract pinning."""
    if 'df' in _panel_cache:
        return _panel_cache['df'], _panel_cache['tm_cache']

    print('Loading data...')
    full_log = load_daily_shape_log()
    full_log = full_log.sort_values('date').reset_index(drop=True)
    enriched = load_enriched_shape_log()
    enriched = enriched.sort_values('date').reset_index(drop=True)

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

    for name, cfg in ALL_INSTRUMENT_CONFIG.items():
        df[name] = df[cfg['near']] - df[cfg['far']]
    for name, cfg in BUTTERFLY_CONFIG.items():
        m1, m2, m3 = cfg['legs']
        df[name] = df[m1] - 2 * df[m2] + df[m3]

    # Regime-relative z-scores — also store mean/std for intraday use
    print('Computing z-scores...')
    for inst in ALL_9:
        z_vals, mean_vals, std_vals = _compute_regime_zscore_with_params(df, inst)
        df[f'{inst}_z'] = z_vals
        df[f'{inst}_mean'] = mean_vals
        df[f'{inst}_std'] = std_vals

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

    # Load per-contract prices for contract pinning
    print('Loading per-contract prices for pinning...')
    contract_prices = load_contract_prices()
    print(f'  Loaded {len(contract_prices)} contract series')

    _panel_cache['df'] = df
    _panel_cache['tm_cache'] = tm_cache
    _panel_cache['contracts'] = contract_prices
    return df, tm_cache


def _compute_regime_zscore_with_params(df, instrument_col):
    """Per-episode rolling 60-day window z-score, min 10 observations.
    Returns (z_series, mean_series, std_series) for intraday z-score use."""
    zscores = pd.Series(np.nan, index=df.index)
    means = pd.Series(np.nan, index=df.index)
    stds = pd.Series(np.nan, index=df.index)
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
            means[idx] = mean
            stds[idx] = std
            if std > 0 and not np.isnan(val):
                zscores[idx] = (val - mean) / std
    return zscores, means, stds


def load_intraday_data():
    """Load 60-min tenor-mapped wide CSVs for spreads and butterflies.
    Returns a DataFrame with datetime index and instrument-named columns."""
    if 'intraday' in _intraday_cache:
        return _intraday_cache['intraday']

    base = Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild')
    spread_path = base / 'spread_tenor_close_wide.csv'
    bf_path = base / 'butterfly_tenor_close_wide.csv'

    print('Loading 60-min intraday data...')
    spread_df = pd.read_csv(spread_path, parse_dates=['datetime'])
    bf_df = pd.read_csv(bf_path, parse_dates=['datetime'])

    # Filter to INTRADAY_START (2020-01-01 after tenor CSV rebuild)
    spread_df = spread_df[spread_df['datetime'] >= INTRADAY_START].copy()
    bf_df = bf_df[bf_df['datetime'] >= INTRADAY_START].copy()

    # Rename tenor columns to instrument names
    spread_renamed = spread_df[['datetime']].copy()
    for tenor, inst in SPREAD_TENOR_MAP.items():
        if tenor in spread_df.columns:
            spread_renamed[inst] = spread_df[tenor]

    bf_renamed = bf_df[['datetime']].copy()
    for tenor, inst in BUTTERFLY_TENOR_MAP.items():
        if tenor in bf_df.columns:
            bf_renamed[inst] = bf_df[tenor]

    # Merge on datetime
    intraday = pd.merge(spread_renamed, bf_renamed, on='datetime', how='outer')
    intraday = intraday.sort_values('datetime').reset_index(drop=True)
    intraday['date'] = intraday['datetime'].dt.normalize()

    print(f'  Intraday: {len(intraday)} bars, '
          f'{intraday["datetime"].min()} to {intraday["datetime"].max()}')

    _intraday_cache['intraday'] = intraday
    return intraday


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
# BACKTEST ENGINE — DAILY
# ══════════════════════════════════════════════════════════════

def run_backtest(config, df=None, tm_cache=None):
    """Run the daily backtest across all 4 walk-forward windows.
    Returns dict: {window_name: pd.DataFrame of trades}"""
    if df is None or tm_cache is None:
        df, tm_cache = build_panel()

    contracts = _panel_cache.get('contracts', {})
    results = {}
    for w in WINDOWS:
        trades = _run_window_daily(df, tm_cache, config,
                                   w['test_start'], w['test_end'],
                                   contracts=contracts)
        results[w['name']] = trades
    return results


def _run_window_daily(df, tm_cache, config, test_start, test_end, contracts=None):
    """Run daily backtest for a single test window.
    Uses contract pinning: at entry, resolve rolling tenor to specific contracts.
    All exit checks (TP, SL, scale-in) use pinned contract prices and z-scores."""
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

    all_trades = []

    for inst in instruments:
        z_col = f'{inst}_z'
        cost_per_lot = get_cost_points(inst, config)

        position_open = False
        entry_date = entry_spread = entry_z_val = entry_shape = entry_direction = None
        days_held = 0
        total_lots = 0
        max_abs_z = 0.0
        tier_entries = []
        tier_fired = set()
        # Contract pinning state
        resolved_contracts = None
        episode_price_history = []

        for idx in df.index:
            row = df.loc[idx]
            date, shape, z, spread = row['date'], row['shape'], row[z_col], row[inst]
            if pd.isna(spread):
                continue

            if position_open:
                days_held += 1

                # Get pinned contract price for today
                pinned_spread = get_contract_spread(contracts, resolved_contracts, date)
                if pd.isna(pinned_spread):
                    pinned_spread = spread  # fallback to rolling if contract data missing

                # Append to episode history and compute pinned z-score
                episode_price_history.append(pinned_spread)
                pinned_z = compute_pinned_zscore(episode_price_history, pinned_spread)

                # Track max|z| for B/C bucket classification (use pinned z)
                if not pd.isna(pinned_z):
                    max_abs_z = max(max_abs_z, abs(pinned_z))

                # Scale-in checks (before exit checks) — use pinned z and price
                for ti, tier in enumerate(scale_tiers):
                    if ti not in tier_fired and not pd.isna(pinned_z):
                        should_fire = False
                        if entry_direction == -1:
                            should_fire = pinned_z >= tier['z_level']
                        else:
                            should_fire = pinned_z <= -tier['z_level']
                        if should_fire:
                            tier_fired.add(ti)
                            tier_entries.append({
                                'tier': ti + 1,
                                'z_level': tier['z_level'],
                                'lots': tier['lots'],
                                'entry_spread': pinned_spread,
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

                # Take profit — use pinned z-score
                if not pd.isna(pinned_z) and abs(pinned_z) < exit_z:
                    exit_reason = 'take_profit'

                # Shape invalidation
                if shape != entry_shape:
                    exit_reason = 'invalidated'

                # Stop loss — use pinned z-score
                if stop_loss_z is not None and not pd.isna(pinned_z):
                    if entry_direction == -1 and pinned_z >= stop_loss_z:
                        exit_reason = 'stop_loss'
                    elif entry_direction == 1 and pinned_z <= -stop_loss_z:
                        exit_reason = 'stop_loss'

                # Time stop
                if days_held >= time_stop:
                    exit_reason = 'time_stop'

                # M1-M2 expiry buffer: force-close before near-leg stops trading
                if exit_reason is None and expiry_exit_due(inst, resolved_contracts, date):
                    exit_reason = 'expiry_buffer'

                if exit_reason:
                    # P&L uses pinned contract prices
                    base_gross = (pinned_spread - entry_spread) * entry_direction * first_lots
                    tier_gross = sum(
                        (pinned_spread - te_tier['entry_spread']) * entry_direction * te_tier['lots']
                        for te_tier in tier_entries
                    )
                    gross_pnl = base_gross + tier_gross
                    total_cost = total_lots * cost_per_lot
                    net_pnl = gross_pnl - total_cost

                    if ts <= date <= te:
                        trade = {
                            'instrument': inst,
                            'entry_date': entry_date, 'exit_date': date,
                            'entry_spread': round(entry_spread, 2),
                            'exit_spread': round(pinned_spread, 2),
                            'entry_z': round(entry_z_val, 3),
                            'exit_z': round(pinned_z, 3) if not pd.isna(pinned_z) else np.nan,
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
                            'pinned_contracts': str(resolved_contracts),
                        }
                        all_trades.append(trade)

                    position_open = False
                    tier_entries = []
                    tier_fired = set()
                    max_abs_z = 0.0
                    resolved_contracts = None
                    episode_price_history = []
                    continue

            if not position_open:
                # Entry decision uses ROLLING label z-score (current tenor structure)
                if pd.isna(z) or pd.isna(row['pm_level']):
                    continue
                is_resting = shape in RESTING_SHAPES
                dur_ok = row['days_in_shape'] >= dur_thresh
                z_extreme = abs(z) > entry_z
                pm_ok = row['pm_level'] <= pm_filter
                if is_resting and dur_ok and z_extreme and pm_ok:
                    # Resolve rolling tenor to specific contracts at entry
                    resolved_contracts = resolve_contracts(inst, date)

                    # Block entry if already in the expiry buffer zone (any leg)
                    if expiry_exit_due(inst, resolved_contracts, date):
                        resolved_contracts = None
                        continue

                    pinned_entry = get_contract_spread(contracts, resolved_contracts, date)
                    if pd.isna(pinned_entry):
                        pinned_entry = spread  # fallback

                    # Seed episode price history from pinned contracts
                    episode_price_history = build_pinned_episode_history(
                        contracts, resolved_contracts, df, row['episode_id'], date)

                    position_open = True
                    entry_date = date
                    entry_spread = pinned_entry
                    entry_z_val = z  # entry z from rolling label (entry decision)
                    entry_shape = shape
                    entry_direction = -1 if z > 0 else 1
                    days_held = 0
                    total_lots = first_lots
                    tier_entries = []
                    tier_fired = set()
                    max_abs_z = abs(z)

    return pd.DataFrame(all_trades)


# ══════════════════════════════════════════════════════════════
# BACKTEST ENGINE — 60-MIN INTRADAY
# ══════════════════════════════════════════════════════════════

def run_backtest_intraday(config, df=None, tm_cache=None, intraday_df=None):
    """Run backtest using 60-min bars for entry/exit timing.
    Daily panel provides shape, PM, TM, z-score mean/std.
    Intraday bars provide precise price for threshold crossings.
    NOTE: Intraday bars use tenor-labeled prices (per-contract 60-min data
    doesn't exist). Daily settlement P&L uses pinned contracts.

    Returns dict with single key 'intraday': pd.DataFrame of trades."""
    if df is None or tm_cache is None:
        df, tm_cache = build_panel()
    if intraday_df is None:
        intraday_df = load_intraday_data()

    contracts = _panel_cache.get('contracts', {})
    trades = _run_window_intraday(df, tm_cache, config, intraday_df,
                                   INTRADAY_START, INTRADAY_END,
                                   contracts=contracts)
    return {'intraday': trades}


def _run_window_intraday(df, tm_cache, config, intraday_df, test_start, test_end,
                          contracts=None):
    """Run 60-min intraday backtest.
    Z-score baseline is DAILY (mean/std from daily settlements).
    60-min bar prices are checked against daily thresholds for precise timing.
    NOTE: Intraday bars cannot be pinned (per-contract 60-min data doesn't exist).
    Daily-level exits use pinned contract prices where available."""
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

    # Build daily lookup: date -> {shape, days_in_shape, pm_level, per-inst mean/std}
    daily_mask = (df['date'] >= ts) & (df['date'] <= te)
    daily_rows = df[daily_mask].set_index('date')

    # Filter intraday to test range
    intra = intraday_df[(intraday_df['date'] >= ts) & (intraday_df['date'] <= te)].copy()
    if len(intra) == 0:
        return pd.DataFrame()

    # Group intraday bars by date
    intra_by_date = dict(list(intra.groupby('date')))

    all_trades = []

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
        # Contract pinning state for intraday
        resolved_contracts = None

        # Iterate day by day
        for date_ts in daily_rows.index:
            if date_ts not in daily_rows.index:
                continue
            day = daily_rows.loc[date_ts]
            # Handle duplicate dates (take last)
            if isinstance(day, pd.DataFrame):
                day = day.iloc[-1]

            shape = day['shape']
            pm_level = day.get('pm_level', np.nan)
            daily_mean = day.get(mean_col, np.nan)
            daily_std = day.get(std_col, np.nan)

            if position_open:
                days_held += 1

                # Daily-level exits first: shape change, time stop, TM regime-risk
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

                # M1-M2 expiry buffer: force-close before near-leg stops trading
                if exit_reason is None and expiry_exit_due(inst, resolved_contracts, date_ts):
                    exit_reason = 'expiry_buffer'

                # If daily exit triggered, use pinned contract price for P&L
                if exit_reason:
                    # Try pinned contract price first
                    daily_spread = get_contract_spread(contracts, resolved_contracts, date_ts) if resolved_contracts else np.nan
                    if pd.isna(daily_spread):
                        # Fallback to tenor-labeled daily price
                        daily_spread = day[inst] if inst in day.index else np.nan
                    if pd.isna(daily_spread):
                        # Try last intraday bar of the day
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
                    continue

                # Intraday exits: scan 60-min bars for TP, SL, scale-in
                day_bars = intra_by_date.get(date_ts)
                if day_bars is not None and inst in day_bars.columns and not pd.isna(daily_std) and daily_std > 0:
                    for _, bar in day_bars.iterrows():
                        bar_price = bar[inst]
                        if pd.isna(bar_price):
                            continue
                        bar_z = (bar_price - daily_mean) / daily_std

                        max_abs_z = max(max_abs_z, abs(bar_z))

                        # Scale-in checks
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

                        # Take profit (intraday)
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
                            break

                        # Stop loss (intraday)
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
                                break

            # Entry check (daily-level gate, intraday price for execution)
            if not position_open:
                if pd.isna(pm_level) or pd.isna(daily_mean) or pd.isna(daily_std) or daily_std <= 0:
                    continue
                is_resting = shape in RESTING_SHAPES
                dur_ok = day['days_in_shape'] >= dur_thresh if 'days_in_shape' in day.index else False
                pm_ok = pm_level <= pm_filter

                if is_resting and dur_ok and pm_ok:
                    # Scan intraday bars for z-threshold crossing
                    day_bars = intra_by_date.get(date_ts)
                    if day_bars is not None and inst in day_bars.columns:
                        for _, bar in day_bars.iterrows():
                            bar_price = bar[inst]
                            if pd.isna(bar_price):
                                continue
                            bar_z = (bar_price - daily_mean) / daily_std
                            if abs(bar_z) > entry_z:
                                # Resolve contracts at entry for pinning
                                resolved_contracts = resolve_contracts(inst, date_ts)
                                # Block M1-M2 entry if already in expiry buffer zone
                                if expiry_exit_due(inst, resolved_contracts, date_ts):
                                    resolved_contracts = None
                                    continue
                                position_open = True
                                entry_date = date_ts
                                entry_datetime = bar['datetime']
                                entry_spread = bar_price  # intraday bar price (tenor-labeled, known limitation)
                                entry_z_val = bar_z
                                entry_shape = shape
                                entry_direction = -1 if bar_z > 0 else 1
                                days_held = 0
                                total_lots = first_lots
                                tier_entries = []
                                tier_fired = set()
                                max_abs_z = abs(bar_z)
                                break

    return pd.DataFrame(all_trades)


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(trades, df_ref=None, test_start=None, test_end=None, config=None):
    """Compute summary metrics for a set of trades.
    Returns both naive and adjusted Sharpe."""
    n = len(trades)
    if n == 0:
        return {
            'n_trades': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0,
            'total_pnl': 0, 'naive_sharpe': np.nan, 'adj_sharpe': np.nan,
            'max_dd': 0,
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

    # Naive Sharpe (exit-day PnL attribution)
    naive_sharpe = np.nan
    if df_ref is not None and test_start is not None:
        naive_sharpe = _compute_naive_sharpe(trades, test_start, test_end, df_ref)

    # Adjusted Sharpe (MtM daily PnL) — for validation uses 100 MYR flat
    adj_sharpe = np.nan
    if df_ref is not None and test_start is not None:
        adj_sharpe = _compute_daily_portfolio_sharpe(trades, test_start, test_end, df_ref)

    return {
        'n_trades': n, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'total_pnl': total_pnl,
        'naive_sharpe': naive_sharpe, 'adj_sharpe': adj_sharpe, 'max_dd': max_dd,
        'pct_take_profit': tp_pct, 'pct_invalidated': inv_pct,
        'pct_time_stop': ts_pct, 'pct_regime_risk': rr_pct,
        'pct_stop_loss': sl_pct,
        'avg_hp': avg_hp, 'shape_survival': shape_survival,
    }


def _compute_naive_sharpe(trades, test_start, test_end, df_ref):
    """Naive Sharpe: attribute each trade's net PnL to its exit date."""
    if len(trades) == 0:
        return np.nan
    ts_dt, te_dt = pd.Timestamp(test_start), pd.Timestamp(test_end)
    window_dates = df_ref[(df_ref['date'] >= ts_dt) & (df_ref['date'] <= te_dt)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=window_dates)
    for _, t in trades.iterrows():
        exit_dt = t['exit_date']
        if exit_dt in daily_pnl.index:
            daily_pnl[exit_dt] += t['net_pnl']
    daily_std = daily_pnl.std()
    if daily_std > 0:
        return round(daily_pnl.mean() / daily_std * np.sqrt(252), 3)
    return np.nan


def _compute_daily_portfolio_sharpe(trades, test_start, test_end, df_ref):
    """Daily portfolio Sharpe using mark-to-market within a test window.
    Uses 100 MYR flat cost for backward compatibility with published validation.
    Uses pinned contract prices for MtM when available."""
    if len(trades) == 0:
        return np.nan
    ts_dt = pd.Timestamp(test_start)
    te_dt = pd.Timestamp(test_end)
    window_dates = df_ref[(df_ref['date'] >= ts_dt) & (df_ref['date'] <= te_dt)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=window_dates)
    contracts = _panel_cache.get('contracts', {})

    for _, t in trades.iterrows():
        entry_dt = t['entry_date']
        exit_dt = t['exit_date']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']

        # Parse pinned contracts if available
        pinned = None
        if 'pinned_contracts' in t.index and pd.notna(t.get('pinned_contracts')):
            try:
                pinned = eval(t['pinned_contracts'])
            except Exception:
                pass

        trade_days = df_ref[(df_ref['date'] > entry_dt) & (df_ref['date'] <= exit_dt)].copy()
        if len(trade_days) == 0:
            continue

        prev_spread = t['entry_spread']
        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            # Use pinned contract price for MtM
            if pinned and contracts:
                current_spread = get_contract_spread(contracts, pinned, dt)
                if pd.isna(current_spread):
                    current_spread = day_row[inst]  # fallback
            else:
                current_spread = day_row[inst]
            if pd.isna(current_spread):
                continue
            day_mtm = (current_spread - prev_spread) * direction
            if dt in daily_pnl.index:
                daily_pnl[dt] += day_mtm
            prev_spread = current_spread

        cost_per_lot = get_cost_points(inst, {'cost_spread_myr': 100.0, 'cost_butterfly_myr': 100.0})
        if exit_dt in daily_pnl.index:
            daily_pnl[exit_dt] -= cost_per_lot

    daily_std = daily_pnl.std()
    if daily_std > 0:
        return round(daily_pnl.mean() / daily_std * np.sqrt(252), 3)
    return np.nan


def compute_daily_portfolio_sharpe_configurable(trades, test_start, test_end, df_ref, config):
    """Daily portfolio Sharpe using configurable cost model.
    Uses pinned contract prices for MtM when available."""
    if len(trades) == 0:
        return np.nan
    ts_dt = pd.Timestamp(test_start)
    te_dt = pd.Timestamp(test_end)
    window_dates = df_ref[(df_ref['date'] >= ts_dt) & (df_ref['date'] <= te_dt)]['date'].sort_values().values
    daily_pnl = pd.Series(0.0, index=window_dates)
    contracts = _panel_cache.get('contracts', {})

    for _, t in trades.iterrows():
        entry_dt = t['entry_date']
        exit_dt = t['exit_date']
        direction = 1 if t['direction'] == 'LONG' else -1
        inst = t['instrument']
        total_lots = t.get('total_lots', 1)

        # Parse pinned contracts if available
        pinned = None
        if 'pinned_contracts' in t.index and pd.notna(t.get('pinned_contracts')):
            try:
                pinned = eval(t['pinned_contracts'])
            except Exception:
                pass

        trade_days = df_ref[(df_ref['date'] > entry_dt) & (df_ref['date'] <= exit_dt)].copy()
        if len(trade_days) == 0:
            continue

        prev_spread = t['entry_spread']
        for _, day_row in trade_days.iterrows():
            dt = day_row['date']
            if pinned and contracts:
                current_spread = get_contract_spread(contracts, pinned, dt)
                if pd.isna(current_spread):
                    current_spread = day_row[inst]
            else:
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
    """Classify a losing trade into loss buckets A/B/C/D/TP-loss/SL.
    Uses max_abs_z to separate B (adverse extension) from C (stalled)."""
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
        entry_abs_z = abs(trade['entry_z'])
        max_z = trade.get('max_abs_z', np.nan)
        if pd.isna(max_z):
            return 'B/C'
        if max_z > entry_abs_z:
            return 'B'  # Adverse extension: z moved further against
        else:
            return 'C'  # Stalled: z never exceeded entry level
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
