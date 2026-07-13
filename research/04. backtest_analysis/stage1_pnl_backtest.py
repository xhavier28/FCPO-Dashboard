"""
Stage 1 — P&L Backtest for MR Calendar Spread Strategy
=======================================================
Instruments: M2-M3, M3-M4, M4-M5, M5-M6 (dur>=10d for all)
Entry/exit/sizing/cost rules per specification.

Also produces Part 1 duration threshold decision table.
Appends all results to backtest_analysis.txt.
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import pandas as pd
import numpy as np
from datetime import datetime
from models.pm_engine import predict as pm_predict
from models.feature_prep import load_daily_shape_log, load_enriched_shape_log

LOG_FILE = r'C:/ClaudeCode/research/04. backtest_analysis/backtest_analysis.txt'
OOS_START = '2022-01-01'
PM_CONFIDENCE_THRESHOLD = 0.70
Z_ENTRY = 2.0
Z_EXIT = 0.5
TIME_STOP_DAYS = 20

# Transaction cost: 100 MYR per round trip (MRBackTest/app.py default)
# FCPO = 25 tonnes/lot → 1 spread point = 25 MYR per lot
# Cost in points: 100 / 25 = 4 points per round trip
ROUNDTRIP_COST_MYR = 100.0
POINT_VALUE = 25.0  # MYR per point per lot (standard FCPO)
ROUNDTRIP_COST_POINTS = ROUNDTRIP_COST_MYR / POINT_VALUE  # = 4.0

# Per-instrument duration thresholds (from Part 1 decision)
# All instruments classified FLAT or IMPROVES_AT_LOWER → all use dur>=10d
INSTRUMENT_CONFIG = {
    'M2-M3': {'dur_thresh': 10, 'near': 'M2', 'far': 'M3'},
    'M3-M4': {'dur_thresh': 10, 'near': 'M3', 'far': 'M4'},
    'M4-M5': {'dur_thresh': 10, 'near': 'M4', 'far': 'M5'},
    'M5-M6': {'dur_thresh': 10, 'near': 'M5', 'far': 'M6'},
}
INSTRUMENTS = list(INSTRUMENT_CONFIG.keys())

# ══════════════════════════════════════════════════════════════
# DATA SETUP (reuses Stage 0 logic exactly)
# ══════════════════════════════════════════════════════════════

print('Loading data...')
full_log = load_daily_shape_log()
full_log = full_log.sort_values('date').reset_index(drop=True)

enriched = load_enriched_shape_log()
enriched = enriched.sort_values('date').reset_index(drop=True)

# Pre-2017: compute episode tracking
pre_2017 = full_log[full_log['date'] < '2017-01-01'].copy()
pre_2017 = pre_2017.sort_values('date').reset_index(drop=True)

days_list = []
episode_list = []
ep_id = 0
prev_shape = None
day_count = 0
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
df = pd.concat([
    pre_2017[shared_cols],
    enriched[shared_cols]
], ignore_index=True).sort_values('date').reset_index(drop=True)
df = df.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

print(f'Panel: {len(df)} rows, {df["date"].min().date()} to {df["date"].max().date()}')

# Compute spreads
for inst, cfg in INSTRUMENT_CONFIG.items():
    df[inst] = df[cfg['near']] - df[cfg['far']]

# ── Regime-relative z-scores (exact Stage 0 logic) ──────────
def compute_regime_zscore(df, instrument_col):
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
            mean = window.mean()
            std = window.std()
            if std > 0 and not np.isnan(val):
                zscores[idx] = (val - mean) / std
    return zscores

print('Computing z-scores...')
for inst in INSTRUMENTS:
    df[f'{inst}_z'] = compute_regime_zscore(df, inst)
    print(f'  {inst}: {df[f"{inst}_z"].notna().sum()} valid days')

# ── PM predictions (2017+ only) ─────────────────────────────
print('Running PM predictions...')
model_start = pd.Timestamp('2017-01-01')
model_dates_idx = df[df['date'] >= model_start].index

pm_level_col = pd.Series(np.nan, index=df.index)
for i in model_dates_idx:
    dt = df.loc[i, 'date']
    obs_shape = str(df.loc[i, 'shape'])
    try:
        pm = pm_predict(dt)
        pred = pm.get('predicted_shape')
        conf = pm.get('confidence')
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
                if obs_shape in top2:
                    pm_level_col[i] = 1
                else:
                    pm_level_col[i] = 2
            elif pred == obs_shape:
                pm_level_col[i] = 1
            else:
                pm_level_col[i] = 2
    except Exception:
        pass

df['pm_level'] = pm_level_col
print(f'PM levels: {df["pm_level"].value_counts().sort_index().to_dict()}')


# ══════════════════════════════════════════════════════════════
# STAGE 1 P&L BACKTEST
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('STAGE 1: P&L BACKTEST')
print('='*60)

all_trades = []

for inst in INSTRUMENTS:
    cfg = INSTRUMENT_CONFIG[inst]
    dur_thresh = cfg['dur_thresh']
    z_col = f'{inst}_z'
    spread_col = inst

    # Walk through each day chronologically
    position_open = False
    entry_date = None
    entry_spread = None
    entry_z = None
    entry_shape = None
    entry_direction = None  # +1 LONG, -1 SHORT
    days_held = 0
    skipped_signals = 0

    for idx in df.index:
        row = df.loc[idx]
        date = row['date']
        shape = row['shape']
        z = row[z_col]
        spread = row[spread_col]

        if pd.isna(spread):
            continue

        # Check exit first if position is open
        if position_open:
            days_held += 1
            exit_reason = None

            # a) Take profit: |z| < 0.5
            if not pd.isna(z) and abs(z) < Z_EXIT:
                exit_reason = 'take_profit'

            # b) Signal invalidated: shape changed from entry shape
            if shape != entry_shape:
                exit_reason = 'invalidated'

            # c) Time stop: 20 trading days
            if days_held >= TIME_STOP_DAYS:
                exit_reason = 'time_stop'

            if exit_reason:
                gross_pnl = (spread - entry_spread) * entry_direction
                net_pnl = gross_pnl - ROUNDTRIP_COST_POINTS

                trade = {
                    'instrument': inst,
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_spread': round(entry_spread, 2),
                    'exit_spread': round(spread, 2),
                    'entry_z': round(entry_z, 3),
                    'exit_z': round(z, 3) if not pd.isna(z) else np.nan,
                    'direction': 'LONG' if entry_direction == 1 else 'SHORT',
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'gross_pnl': round(gross_pnl, 2),
                    'net_pnl': round(net_pnl, 2),
                    'oos': date >= pd.Timestamp(OOS_START),
                }
                all_trades.append(trade)
                position_open = False
                continue  # don't re-enter on exit day

        # Check entry (only if no position open)
        if not position_open:
            if pd.isna(z) or pd.isna(row['pm_level']):
                continue

            # Signal ON conditions
            is_resting = shape in ('0.0', '1')
            dur_ok = row['days_in_shape'] >= dur_thresh
            z_extreme = abs(z) > Z_ENTRY
            pm_ok = row['pm_level'] == 0

            if is_resting and dur_ok and z_extreme and pm_ok:
                position_open = True
                entry_date = date
                entry_spread = spread
                entry_z = z
                entry_shape = shape
                # Direction: z > 2.0 → SHORT (bet on narrowing); z < -2.0 → LONG
                entry_direction = -1 if z > 0 else 1
                days_held = 0

    # Log skipped signals count
    print(f'  {inst}: position walk complete')

trades_df = pd.DataFrame(all_trades)
print(f'\nTotal trades: {len(trades_df)}')
print(f'  IS: {(~trades_df["oos"]).sum()}, OOS: {trades_df["oos"].sum()}')


# ══════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_metrics(trades, label, df_daily=None):
    """Compute all required metrics for a set of trades."""
    n = len(trades)
    if n == 0:
        return {'label': label, 'n_trades': 0}

    wins = trades[trades['net_pnl'] > 0]
    losses = trades[trades['net_pnl'] <= 0]

    win_rate = len(wins) / n * 100
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
    total_pnl = trades['net_pnl'].sum()

    # Exit breakdown
    tp_pct = (trades['exit_reason'] == 'take_profit').mean() * 100
    inv_pct = (trades['exit_reason'] == 'invalidated').mean() * 100
    ts_pct = (trades['exit_reason'] == 'time_stop').mean() * 100

    # Holding period
    avg_hp = trades['days_held'].mean()
    avg_hp_win = wins['days_held'].mean() if len(wins) > 0 else np.nan
    avg_hp_loss = losses['days_held'].mean() if len(losses) > 0 else np.nan

    # Maximum drawdown on cumulative P&L
    cum_pnl = trades['net_pnl'].cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_dd = drawdown.min()

    # Sharpe ratio: daily P&L series, annualized
    # Build daily P&L series from trades
    if df_daily is not None and n > 0:
        oos_dates = df_daily[df_daily['date'] >= pd.Timestamp(OOS_START)]['date']
        daily_pnl = pd.Series(0.0, index=oos_dates.values)

        for _, t in trades.iterrows():
            exit_dt = t['exit_date']
            if exit_dt in daily_pnl.index:
                daily_pnl[exit_dt] += t['net_pnl']

        daily_mean = daily_pnl.mean()
        daily_std = daily_pnl.std()
        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else np.nan
    else:
        sharpe = np.nan

    return {
        'label': label,
        'n_trades': n,
        'win_rate': round(win_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'total_pnl': round(total_pnl, 2),
        'sharpe': round(sharpe, 3) if not pd.isna(sharpe) else np.nan,
        'max_dd': round(max_dd, 2),
        'pct_take_profit': round(tp_pct, 1),
        'pct_invalidated': round(inv_pct, 1),
        'pct_time_stop': round(ts_pct, 1),
        'avg_hp': round(avg_hp, 1),
        'avg_hp_win': round(avg_hp_win, 1) if not pd.isna(avg_hp_win) else np.nan,
        'avg_hp_loss': round(avg_hp_loss, 1) if not pd.isna(avg_hp_loss) else np.nan,
    }


# Per-instrument metrics (OOS only)
oos_trades = trades_df[trades_df['oos']].copy()
oos_trades = oos_trades.reset_index(drop=True)

print('\n--- PER-INSTRUMENT METRICS (OOS) ---')
per_inst_metrics = []
for inst in INSTRUMENTS:
    inst_trades = oos_trades[oos_trades['instrument'] == inst].reset_index(drop=True)
    m = compute_metrics(inst_trades, inst, df)
    per_inst_metrics.append(m)
    print(f'  {inst}: n={m["n_trades"]}, win={m.get("win_rate","-")}%, '
          f'total_pnl={m.get("total_pnl","-")}, sharpe={m.get("sharpe","-")}')

# Pooled metrics
print('\n--- POOLED METRICS (OOS) ---')
pooled_m = compute_metrics(oos_trades.reset_index(drop=True), 'POOLED', df)
print(f'  POOLED: n={pooled_m["n_trades"]}, win={pooled_m.get("win_rate","-")}%, '
      f'total_pnl={pooled_m.get("total_pnl","-")}, sharpe={pooled_m.get("sharpe","-")}')


# Gate check
def gate_check(m):
    if m['n_trades'] < 15:
        return 'INSUFFICIENT'
    if m.get('sharpe', np.nan) is np.nan or np.isnan(m.get('sharpe', np.nan)):
        return 'INSUFFICIENT'
    if m['sharpe'] > 0 and m['avg_win'] > abs(m['avg_loss']):
        return 'PASS (provisional, n>=15)'
    return 'FAIL'

print('\n--- GATE CHECK (OOS) ---')
for m in per_inst_metrics:
    gate = gate_check(m)
    print(f'  {m["label"]}: {gate}')
pooled_gate = gate_check(pooled_m)
print(f'  POOLED: {pooled_gate}')


# ══════════════════════════════════════════════════════════════
# DETAILED TRADE LOG
# ══════════════════════════════════════════════════════════════

print('\n--- OOS TRADE LOG ---')
for _, t in oos_trades.iterrows():
    print(f'  {t["instrument"]} {t["entry_date"].date()}->{t["exit_date"].date()} '
          f'{t["direction"]} entry={t["entry_spread"]} exit={t["exit_spread"]} '
          f'z={t["entry_z"]}->{t["exit_z"]} {t["exit_reason"]} '
          f'days={t["days_held"]} pnl={t["net_pnl"]}')


# ══════════════════════════════════════════════════════════════
# PART 3 — LOGGING
# ══════════════════════════════════════════════════════════════

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
lines = []

# ── Part 1: Duration decision table ─────────────────────────
lines.append('')
lines.append('')
lines.append('='*70)
lines.append(f'MR — PER-INSTRUMENT DURATION THRESHOLD DECISION — {timestamp}')
lines.append('='*70)
lines.append('')
lines.append('Source: existing duration sensitivity sweep (backtest_analysis.txt)')
lines.append('Classification rule:')
lines.append('  IMPROVES_AT_LOWER: rev@10d(dur>=10) >= rev@10d(dur>=20)')
lines.append('  IMPROVES_AT_HIGHER: rev@10d(dur>=20) > rev@10d(dur>=10) by >5pp')
lines.append('  FLAT: |delta| <= 5pp')
lines.append('')

# Data from the existing sweep
dur_data = {
    'M1-M2':      {'rev10_dur10': 100.0, 'rev10_dur20': 100.0},
    'M2-M3':      {'rev10_dur10': 100.0, 'rev10_dur20': 100.0},
    'M3-M4':      {'rev10_dur10': 95.2,  'rev10_dur20': 100.0},
    'M4-M5':      {'rev10_dur10': 93.3,  'rev10_dur20': 85.7},
    'M5-M6':      {'rev10_dur10': 100.0, 'rev10_dur20': 100.0},
    'BF_M1M2M3':  {'rev10_dur10': 95.7,  'rev10_dur20': 85.7},
    'BF_M2M3M4':  {'rev10_dur10': 100.0, 'rev10_dur20': 100.0},
    'BF_M3M4M5':  {'rev10_dur10': 86.7,  'rev10_dur20': 50.0},
    'BF_M4M5M6':  {'rev10_dur10': 100.0, 'rev10_dur20': 100.0},
}

lines.append(f'  {"Instrument":14s} {"rev@10d(dur>=10)":>18s} {"rev@10d(dur>=20)":>18s} {"Delta":>8s} {"Classification":>22s} {"Recommended":>12s}')
lines.append(f'  {"-"*14} {"-"*18} {"-"*18} {"-"*8} {"-"*22} {"-"*12}')

for inst_name, d in dur_data.items():
    r10 = d['rev10_dur10']
    r20 = d['rev10_dur20']
    delta = r10 - r20
    if delta >= 0:
        cls = 'IMPROVES_AT_LOWER' if delta > 5 else 'FLAT'
    else:
        cls = 'IMPROVES_AT_HIGHER' if abs(delta) > 5 else 'FLAT'
    rec = 'dur>=10d' if cls != 'IMPROVES_AT_HIGHER' else 'dur>=20d'
    lines.append(f'  {inst_name:14s} {r10:>17.1f}% {r20:>17.1f}% {delta:>+7.1f}pp {cls:>22s} {rec:>12s}')

lines.append('')
lines.append('All instruments: FLAT or IMPROVES_AT_LOWER → dur>=10d for all.')
lines.append('Stage 1 backtest instruments (M2-M3, M3-M4, M4-M5, M5-M6): all dur>=10d.')

# ── Part 2: Stage 1 P&L backtest ────────────────────────────
lines.append('')
lines.append('')
lines.append('='*70)
lines.append(f'STAGE 1 — P&L BACKTEST — {timestamp}')
lines.append('='*70)
lines.append('')
lines.append('--- CONFIGURATION ---')
lines.append(f'Instruments: {", ".join(INSTRUMENTS)}')
lines.append(f'Duration thresholds: all dur>=10d (per decision table above)')
lines.append(f'Z-score entry: |z| > {Z_ENTRY}')
lines.append(f'Z-score exit (take profit): |z| < {Z_EXIT}')
lines.append(f'Time stop: {TIME_STOP_DAYS} trading days')
lines.append(f'Position sizing: 1 lot per trade, no pyramiding')
lines.append(f'PM filter: Level 0 (pm_confidence >= {PM_CONFIDENCE_THRESHOLD})')
lines.append(f'Transaction cost: {ROUNDTRIP_COST_MYR} MYR per round trip')
lines.append(f'  Source: MRBackTest/app.py default (line 62)')
lines.append(f'  Point value: {POINT_VALUE} MYR/point (standard FCPO 25 tonnes/lot)')
lines.append(f'  Cost in points: {ROUNDTRIP_COST_POINTS} points per round trip')
lines.append(f'OOS period: >= {OOS_START}')
lines.append(f'NOTE: n>=15 threshold used instead of original n>=30 due to thin OOS')
lines.append(f'  episode counts documented in Stage 0 — results should be treated')
lines.append(f'  as provisional')
lines.append('')

# Per-instrument table
lines.append('--- PER-INSTRUMENT METRICS (OOS) ---')
lines.append(f'  {"Instrument":10s} {"n":>4s} {"Win%":>6s} {"AvgWin":>8s} {"AvgLoss":>8s} {"TotalPnL":>10s} '
             f'{"Sharpe":>7s} {"MaxDD":>8s} {"TP%":>5s} {"Inv%":>5s} {"TS%":>5s} '
             f'{"AvgHP":>6s} {"HP_W":>5s} {"HP_L":>5s} {"Gate":>22s}')
lines.append(f'  {"-"*10} {"-"*4} {"-"*6} {"-"*8} {"-"*8} {"-"*10} '
             f'{"-"*7} {"-"*8} {"-"*5} {"-"*5} {"-"*5} '
             f'{"-"*6} {"-"*5} {"-"*5} {"-"*22}')

for m in per_inst_metrics:
    gate = gate_check(m)
    if m['n_trades'] == 0:
        lines.append(f'  {m["label"]:10s} {0:>4d}   (no trades)')
        continue
    lines.append(f'  {m["label"]:10s} {m["n_trades"]:>4d} {m["win_rate"]:>5.1f}% '
                 f'{m["avg_win"]:>8.2f} {m["avg_loss"]:>8.2f} {m["total_pnl"]:>10.2f} '
                 f'{m["sharpe"]:>7.3f} {m["max_dd"]:>8.2f} '
                 f'{m["pct_take_profit"]:>4.1f}% {m["pct_invalidated"]:>4.1f}% {m["pct_time_stop"]:>4.1f}% '
                 f'{m["avg_hp"]:>6.1f} '
                 f'{"%.1f" % m["avg_hp_win"] if not pd.isna(m["avg_hp_win"]) else "—":>5s} '
                 f'{"%.1f" % m["avg_hp_loss"] if not pd.isna(m["avg_hp_loss"]) else "—":>5s} '
                 f'{gate:>22s}')

lines.append('')

# Pooled table
lines.append('--- POOLED METRICS (all 4 instruments, OOS) ---')
m = pooled_m
gate = gate_check(m)
if m['n_trades'] > 0:
    lines.append(f'  n_trades:       {m["n_trades"]}')
    lines.append(f'  win_rate:       {m["win_rate"]}%')
    lines.append(f'  avg_win:        {m["avg_win"]} points')
    lines.append(f'  avg_loss:       {m["avg_loss"]} points')
    lines.append(f'  total_pnl:      {m["total_pnl"]} points')
    lines.append(f'  sharpe:         {m["sharpe"]}')
    lines.append(f'  max_drawdown:   {m["max_dd"]} points')
    lines.append(f'  exit_breakdown: TP={m["pct_take_profit"]}%, Inv={m["pct_invalidated"]}%, TS={m["pct_time_stop"]}%')
    lines.append(f'  avg_hp:         {m["avg_hp"]} days (win: {m["avg_hp_win"]}, loss: {m["avg_hp_loss"]})')
    lines.append(f'  gate:           {gate}')
else:
    lines.append(f'  No trades')

lines.append('')

# Exit breakdown per instrument
lines.append('--- EXIT TYPE BREAKDOWN (OOS, per instrument) ---')
for inst in INSTRUMENTS:
    inst_trades = oos_trades[oos_trades['instrument'] == inst]
    n = len(inst_trades)
    if n == 0:
        lines.append(f'  {inst}: no trades')
        continue
    tp = (inst_trades['exit_reason'] == 'take_profit').sum()
    inv = (inst_trades['exit_reason'] == 'invalidated').sum()
    ts = (inst_trades['exit_reason'] == 'time_stop').sum()
    lines.append(f'  {inst}: n={n}, take_profit={tp} ({tp/n*100:.1f}%), '
                 f'invalidated={inv} ({inv/n*100:.1f}%), '
                 f'time_stop={ts} ({ts/n*100:.1f}%)')

lines.append('')

# Trade log
lines.append('--- OOS TRADE LOG ---')
for _, t in oos_trades.iterrows():
    lines.append(f'  {t["instrument"]:5s} {t["entry_date"].strftime("%Y-%m-%d")}->{t["exit_date"].strftime("%Y-%m-%d")} '
                 f'{t["direction"]:5s} entry={t["entry_spread"]:>7.1f} exit={t["exit_spread"]:>7.1f} '
                 f'z={t["entry_z"]:>+6.2f}->{t["exit_z"]:>+6.2f} '
                 f'{t["exit_reason"]:13s} days={t["days_held"]:>2d} '
                 f'gross={t["gross_pnl"]:>+7.1f} net={t["net_pnl"]:>+7.1f}')

lines.append('')

# IS summary (for context)
is_trades = trades_df[~trades_df['oos']].copy().reset_index(drop=True)
if len(is_trades) > 0:
    is_m = compute_metrics(is_trades, 'IS', df)
    lines.append('--- IN-SAMPLE SUMMARY (for context only, not part of gate) ---')
    lines.append(f'  n_trades: {is_m["n_trades"]}, win_rate: {is_m["win_rate"]}%, '
                 f'total_pnl: {is_m["total_pnl"]} points, sharpe: {is_m["sharpe"]}')
    lines.append('')

# Write to log
log_text = '\n'.join(lines)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_text)

print(f'\nResults appended to {LOG_FILE}')
print('Done.')
