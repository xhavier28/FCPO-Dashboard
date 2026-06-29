"""
Phase A Follow-Up: Sample Sizes + Persist Output
Re-runs analysis with n columns, episode counts for event flags, RF sample context.
Saves complete output to research/outputs/Phase_A_Variable_Screening.txt
"""
import sys, os, warnings, io
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
import calendar

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
RAW = os.path.join(BASE, 'Raw Data')
OUTPUT_PATH = os.path.join(BASE, 'research', 'outputs', 'Phase_A_Variable_Screening.txt')

# Capture all output to both console and file
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

file_buf = open(OUTPUT_PATH, 'w', encoding='utf-8')
tee = Tee(sys.stdout, file_buf)
_print = print
def print(*args, **kwargs):
    kwargs['file'] = tee
    _print(*args, **kwargs)

ALL_SHAPES = ['0.0', '0.1', '0.2', '1', '2']
SHAPE_NAMES = {'0.0': 'Contango', '0.1': 'Mild Contango', '0.2': 'Steep Backwardation',
               '1': 'Backwardation', '2': 'Flat'}

# ============================================================
print("=" * 80)
print("PHASE A: REGIME PERSISTENCE VARIABLE SCREENING — COMPLETE REPORT")
print("=" * 80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Trusted window: 2017-W01 through most recent complete week")
print(f"Frequency: Weekly (ISO week, Friday/last trading day)")

# ============================================================
# 1. REBUILD WEEKLY PANEL (same logic as original Phase A)
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 1: DATA LOADING & WEEKLY PANEL CONSTRUCTION")
print("=" * 80)

# --- Shape log ---
shape_log = pd.read_csv(os.path.join(RAW, 'Research', 'daily_shape_log.csv'),
                         dtype={'shape': str}, parse_dates=['date'])
print(f"\nShape log: {len(shape_log)} rows, dtype shape={shape_log['shape'].dtype}")
print(f"  Date range: {shape_log['date'].min().date()} to {shape_log['date'].max().date()}")

# --- MPOB ---
stock_df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'FCPO Stock 3Y.xlsx'))
stock_df.columns = ['date', 'stock']
stock_df = stock_df.iloc[1:]
stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
stock_df = stock_df.dropna(subset=['date'])
stock_df['stock'] = pd.to_numeric(stock_df['stock'], errors='coerce')
stock_df = stock_df.dropna(subset=['stock']).sort_values('date').reset_index(drop=True)

export_df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'MPOB Export 3Y.xlsx'))
export_df.columns = ['date', 'export']
export_df = export_df.iloc[1:]
export_df['date'] = pd.to_datetime(export_df['date'], errors='coerce')
export_df = export_df.dropna(subset=['date'])
export_df['export'] = pd.to_numeric(export_df['export'], errors='coerce')
export_df = export_df.dropna(subset=['export']).sort_values('date').reset_index(drop=True)

prod_df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'MPOB Production 3Y.xlsx'))
prod_df.columns = ['date', 'production']
prod_df = prod_df.iloc[1:]
prod_df['date'] = pd.to_datetime(prod_df['date'], errors='coerce')
prod_df = prod_df.dropna(subset=['date'])
prod_df['production'] = pd.to_numeric(prod_df['production'], errors='coerce')
prod_df = prod_df.dropna(subset=['production']).sort_values('date').reset_index(drop=True)

print(f"\nMPOB Stock: {stock_df['date'].min().date()} to {stock_df['date'].max().date()} ({len(stock_df)} months)")
print(f"MPOB Production: {prod_df['date'].min().date()} to {prod_df['date'].max().date()} ({len(prod_df)} months)")
print(f"MPOB Export: {export_df['date'].min().date()} to {export_df['date'].max().date()} ({len(export_df)} months)")
print("NOTE: stock_to_usage_ratio = stock/production (no import data for full usage calc)")

# --- FX ---
fx_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'FX_IDC_USDMYR, 1D_1227e.csv'))
fx_df['date'] = pd.to_datetime(fx_df['time'], unit='s')
fx_df = fx_df[['date', 'close']].rename(columns={'close': 'usd_myr'}).sort_values('date')
print(f"\nUSD/MYR: {fx_df['date'].min().date()} to {fx_df['date'].max().date()}")

# --- Palm-soy spread ---
ps_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'MYX_DLY_FCPO1!_2_CBOT_DL_ZL1!, 1D_61912.csv'))
ps_df['date'] = pd.to_datetime(ps_df['time'], unit='s')
ps_df = ps_df[['date', 'close']].rename(columns={'close': 'palm_soy_spread'}).sort_values('date')
print(f"Palm-soy spread (TV ratio): {ps_df['date'].min().date()} to {ps_df['date'].max().date()}, range {ps_df['palm_soy_spread'].min():.1f}-{ps_df['palm_soy_spread'].max():.1f}")

# --- Crude oil ---
cl_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'NYMEX_DL_CL1!, 1D_84001.csv'))
cl_df['date'] = pd.to_datetime(cl_df['time'], unit='s')
cl_df = cl_df[['date', 'close']].rename(columns={'close': 'crude_oil_price'}).sort_values('date')
print(f"Crude oil (WTI): {cl_df['date'].min().date()} to {cl_df['date'].max().date()}")

# --- Macro Event Calendar ---
macro_df = pd.read_excel(os.path.join(RAW, 'Variable Analysis Extra Data', 'FCPO_Macro_Event_Calendar.xlsx'),
                          sheet_name='Macro Event Calendar', header=4)
flag_cols = {'Indonesia Flag': 'event_indonesia', 'Malaysia Flag': 'event_malaysia',
             'China Flag': 'event_china', 'India Flag': 'event_india', 'Global/Other Flag': 'event_global'}
macro_clean = macro_df[['Year', 'Month']].copy()
for orig, new in flag_cols.items():
    macro_clean[new] = pd.to_numeric(macro_df[orig], errors='coerce').fillna(0).astype(int)
macro_clean['Year'] = pd.to_numeric(macro_clean['Year'], errors='coerce')
macro_clean['Month'] = pd.to_numeric(macro_clean['Month'], errors='coerce')
macro_clean = macro_clean.dropna(subset=['Year', 'Month'])
macro_clean['date'] = pd.to_datetime(
    macro_clean['Year'].astype(int).astype(str) + '-' + macro_clean['Month'].astype(int).astype(str) + '-01')
macro_clean = macro_clean.drop(columns=['Year', 'Month']).set_index('date').sort_index()
print(f"Macro calendar: {macro_clean.index.min().date()} to {macro_clean.index.max().date()}")

# --- ENSO ONI ---
oni_lines = []
with open(os.path.join(RAW, 'ENSO', 'oni.ascii.txt')) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4:
            try:
                seas, yr, total, anom = parts[0], int(parts[1]), float(parts[2]), float(parts[3])
                oni_lines.append({'season': seas, 'year': yr, 'oni': anom})
            except (ValueError, IndexError):
                continue
oni_df = pd.DataFrame(oni_lines)
season_to_month = {'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
oni_df['month'] = oni_df['season'].map(season_to_month)
oni_df['date'] = pd.to_datetime(oni_df['year'].astype(str) + '-' + oni_df['month'].astype(str) + '-01')
oni_df = oni_df[['date', 'oni']].set_index('date').sort_index()
print(f"ENSO ONI: {oni_df.index.min().date()} to {oni_df.index.max().date()}")

# --- Build daily panel ---
daily = shape_log[['date', 'shape', 'M1']].copy().rename(columns={'M1': 'spot'})
daily = daily.set_index('date').sort_index()
daily = daily['2017-01-01':]
daily['iso_year'] = daily.index.isocalendar().year.values
daily['iso_week'] = daily.index.isocalendar().week.values
daily['week_key'] = daily['iso_year'].astype(str) + '-W' + daily['iso_week'].astype(str).str.zfill(2)

# --- Build weekly panel ---
weekly = daily.groupby('week_key').agg(
    shape=('shape', 'last'), spot=('spot', 'last'),
    week_end_date=('spot', lambda x: x.index[-1])
).sort_values('week_end_date')
weekly['shape_prev'] = weekly['shape'].shift(1)

daily['spot_5d_chg'] = daily['spot'].pct_change(5)
weekly['momentum_5d_sign'] = daily.groupby(
    daily['iso_year'].astype(str) + '-W' + daily['iso_week'].astype(str).str.zfill(2)
)['spot_5d_chg'].last().reindex(weekly.index)
weekly['momentum_5d_sign'] = np.sign(weekly['momentum_5d_sign'])

# Merge external daily -> weekly
fx_daily = fx_df.set_index('date')['usd_myr'].sort_index()
daily['usd_myr'] = fx_daily.reindex(daily.index, method='ffill')
weekly['usd_myr'] = daily.groupby('week_key')['usd_myr'].mean().reindex(weekly.index)

ps_daily = ps_df.set_index('date')['palm_soy_spread'].sort_index()
daily['palm_soy'] = ps_daily.reindex(daily.index, method='ffill')
weekly['palm_soy_spread'] = daily.groupby('week_key')['palm_soy'].mean().reindex(weekly.index)

cl_daily = cl_df.set_index('date')['crude_oil_price'].sort_index()
daily['crude'] = cl_daily.reindex(daily.index, method='ffill')
weekly['crude_oil_price'] = daily.groupby('week_key')['crude'].mean().reindex(weekly.index)

weekly['crude_oil_chg_4w'] = weekly['crude_oil_price'].pct_change(4) * 100
weekly['usd_myr_chg_4w'] = weekly['usd_myr'].pct_change(4) * 100

# MPOB monthly -> weekly
mpob = stock_df.set_index('date')[['stock']].join(
    prod_df.set_index('date')[['production']], how='outer'
).join(export_df.set_index('date')[['export']], how='outer').sort_index()
mpob['stock_to_usage_ratio'] = mpob['stock'] / mpob['production']
mpob['export_yoy_pct'] = mpob['export'].pct_change(12) * 100
mpob['production_yoy_pct'] = mpob['production'].pct_change(12) * 100

for col in ['stock_to_usage_ratio', 'export_yoy_pct', 'production_yoy_pct']:
    mpob_series = mpob[col].dropna()
    vals = []
    for _, row in weekly.iterrows():
        dt = row['week_end_date']
        mask = mpob_series.index <= dt
        vals.append(mpob_series[mask].iloc[-1] if mask.any() else np.nan)
    weekly[col] = vals

oni_series = oni_df['oni']
vals = []
for _, row in weekly.iterrows():
    dt = row['week_end_date']
    mask = oni_series.index <= dt
    vals.append(oni_series[mask].iloc[-1] if mask.any() else np.nan)
weekly['enso_oni'] = vals

for col in macro_clean.columns:
    series = macro_clean[col]
    vals = []
    for _, row in weekly.iterrows():
        dt = row['week_end_date']
        mask = series.index <= dt
        vals.append(series[mask].iloc[-1] if mask.any() else np.nan)
    weekly[col] = vals

weekly['event_any_flag_active'] = (weekly[['event_indonesia','event_malaysia','event_china',
                                           'event_india','event_global']].abs().sum(axis=1) > 0).astype(int)
weekly['event_net_sum'] = weekly[['event_indonesia','event_malaysia','event_china',
                                  'event_india','event_global']].sum(axis=1)

# Targets
for N in [4, 12]:
    weekly[f'shape_plus_{N}w'] = weekly['shape'].shift(-N)
    weekly[f'persists_{N}w'] = (weekly['shape'] == weekly[f'shape_plus_{N}w']).astype(int)
    weekly.loc[weekly[f'shape_plus_{N}w'].isna(), f'persists_{N}w'] = np.nan

weekly = weekly[weekly['week_end_date'] >= '2017-01-01']

print(f"\nWeekly panel: {weekly.shape[0]} weeks x {weekly.shape[1]} columns")
print(f"Date range: {weekly['week_end_date'].min().date()} to {weekly['week_end_date'].max().date()}")
print(f"\nPer-column non-null counts:")
candidate_vars = ['stock_to_usage_ratio', 'export_yoy_pct', 'production_yoy_pct', 'enso_oni',
                  'palm_soy_spread', 'crude_oil_price', 'crude_oil_chg_4w', 'usd_myr', 'usd_myr_chg_4w',
                  'momentum_5d_sign', 'event_indonesia', 'event_malaysia', 'event_china', 'event_india',
                  'event_global', 'event_any_flag_active', 'event_net_sum']
for col in candidate_vars:
    n = weekly[col].notna().sum()
    print(f"  {col}: {n}/{len(weekly)}")

# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 2: BASE PERSISTENCE RATES (naive baselines)")
print("=" * 80)

print(f"\n{'Shape':<8} {'Name':<22} {'4w persist':<14} {'4w n':<8} {'12w persist':<14} {'12w n':<8}")
print("-" * 80)
for s in ALL_SHAPES:
    mask = weekly['shape'] == s
    name = SHAPE_NAMES[s]
    results = {}
    for N in [4, 12]:
        sub = weekly.loc[mask, f'persists_{N}w'].dropna()
        results[N] = (sub.mean() * 100 if len(sub) > 0 else 0, len(sub))
    print(f"{s:<8} {name:<22} {results[4][0]:>6.1f}%       {results[4][1]:<8} {results[12][0]:>6.1f}%       {results[12][1]:<8}")

for N in [4, 12]:
    sub = weekly[f'persists_{N}w'].dropna()
    print(f"{'POOLED':<8} {'All shapes':<22} {sub.mean()*100:>6.1f}%       {len(sub):<8}", end="  " if N == 4 else "\n")

# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 3: SURVIVAL EPISODES")
print("=" * 80)

weekly_sorted = weekly.sort_values('week_end_date').copy()
weekly_sorted['regime_start'] = weekly_sorted['shape'] != weekly_sorted['shape_prev']

episodes = []
current_shape = None
start_idx = None
for i, (idx, row) in enumerate(weekly_sorted.iterrows()):
    if row['regime_start'] or current_shape is None:
        if current_shape is not None:
            episodes.append({'shape': current_shape, 'start_week': start_idx,
                             'start_date': start_date, 'duration_weeks': i - start_i, 'censored': False})
        current_shape = row['shape']
        start_idx = idx
        start_date = row['week_end_date']
        start_i = i
if current_shape is not None:
    episodes.append({'shape': current_shape, 'start_week': start_idx,
                     'start_date': start_date, 'duration_weeks': len(weekly_sorted) - start_i, 'censored': True})

ep_df = pd.DataFrame(episodes)

print(f"\nTotal episodes: {len(ep_df)}")
print(f"\n{'Shape':<8} {'Name':<22} {'Episodes':>8} {'Censored':>9} {'Cens%':>7} {'Obs mean':>9} {'KM median':>10}")
print("-" * 80)
for s in ALL_SHAPES:
    sub = ep_df[ep_df['shape'] == s]
    n = len(sub)
    nc = sub['censored'].sum()
    cp = 100 * nc / n if n > 0 else 0
    obs = sub[~sub['censored']]['duration_weeks']
    mn = obs.mean() if len(obs) > 0 else float('nan')
    if n > 0:
        kmf = KaplanMeierFitter()
        kmf.fit(sub['duration_weeks'], event_observed=~sub['censored'])
        km = kmf.median_survival_time_
    else:
        km = float('nan')
    print(f"{s:<8} {SHAPE_NAMES[s]:<22} {n:>8} {nc:>9} {cp:>6.1f}% {mn:>8.1f}w {km:>9}w")

# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 4: UNIVARIATE SCREENING WITH SAMPLE SIZES")
print("=" * 80)

sparse_vars = {'event_indonesia', 'event_malaysia', 'event_china', 'event_india', 'event_global',
               'event_any_flag_active', 'momentum_5d_sign'}

results = []

for var in candidate_vars:
    for N in [4, 12]:
        target_col = f'persists_{N}w'
        sub = weekly[[var, target_col, 'shape']].dropna()
        if len(sub) < 20:
            continue

        if var in sparse_vars:
            med = sub[var].median()
            high_mask = sub[var] > med
            low_mask = sub[var] <= med
            split_type = 'median'
        else:
            q33 = sub[var].quantile(0.333)
            q67 = sub[var].quantile(0.667)
            high_mask = sub[var] >= q67
            low_mask = sub[var] <= q33
            split_type = 'tercile'

        if high_mask.sum() < 3 or low_mask.sum() < 3:
            continue

        h_persist = sub.loc[high_mask, target_col]
        l_persist = sub.loc[low_mask, target_col]
        high_rate = h_persist.mean() * 100
        low_rate = l_persist.mean() * 100

        results.append({
            'variable': var, 'horizon': f'{N}w', 'shape': 'POOLED',
            'high_pct': round(high_rate, 1), 'low_pct': round(low_rate, 1),
            'diff_pp': round(high_rate - low_rate, 1), 'abs_diff': round(abs(high_rate - low_rate), 1),
            'n_high': int(high_mask.sum()), 'n_low': int(low_mask.sum()),
            'n_high_persist': int(h_persist.sum()), 'n_low_persist': int(l_persist.sum()),
            'split': split_type
        })

        for s in ALL_SHAPES:
            s_sub = sub[sub['shape'] == s]
            if len(s_sub) < 10:
                continue
            if var in sparse_vars:
                h = s_sub[var] > med
                l = s_sub[var] <= med
            else:
                h = s_sub[var] >= q67
                l = s_sub[var] <= q33
            if h.sum() < 3 or l.sum() < 3:
                continue
            hp = s_sub.loc[h, target_col]
            lp = s_sub.loc[l, target_col]
            hr = hp.mean() * 100
            lr = lp.mean() * 100
            results.append({
                'variable': var, 'horizon': f'{N}w', 'shape': s,
                'high_pct': round(hr, 1), 'low_pct': round(lr, 1),
                'diff_pp': round(hr - lr, 1), 'abs_diff': round(abs(hr - lr), 1),
                'n_high': int(h.sum()), 'n_low': int(l.sum()),
                'n_high_persist': int(hp.sum()), 'n_low_persist': int(lp.sum()),
                'split': split_type
            })

res_df = pd.DataFrame(results).sort_values('abs_diff', ascending=False)

print(f"\nFull univariate screening table ({len(res_df)} rows), sorted by |diff|:")
print(f"{'Variable':<25} {'Hz':>3} {'Shape':>6} {'High%':>6} {'(n/N)':>10} {'Low%':>6} {'(n/N)':>10} {'Diff':>7} {'Split':>8} {'Flag':>6}")
print("-" * 100)
low_n_rows = []
for _, r in res_df.iterrows():
    flag = ''
    if r['n_high'] < 15 or r['n_low'] < 15:
        flag = '* LOW'
        low_n_rows.append(r)
    h_str = f"{r['n_high_persist']}/{r['n_high']}"
    l_str = f"{r['n_low_persist']}/{r['n_low']}"
    print(f"{r['variable']:<25} {r['horizon']:>3} {r['shape']:>6} {r['high_pct']:>5.1f}% {h_str:>10} {r['low_pct']:>5.1f}% {l_str:>10} {r['diff_pp']:>+6.1f} {r['split']:>8} {flag:>6}")

print(f"\n\n--- LOW SAMPLE SIZE FLAGS (n_high or n_low < 15) ---")
print(f"Total: {len(low_n_rows)} of {len(res_df)} rows flagged")
print(f"\n{'Variable':<25} {'Hz':>3} {'Shape':>6} {'n_high':>7} {'n_low':>7} {'Diff':>7}")
print("-" * 60)
for r in low_n_rows:
    print(f"{r['variable']:<25} {r['horizon']:>3} {r['shape']:>6} {r['n_high']:>7} {r['n_low']:>7} {r['diff_pp']:>+6.1f}")

# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 4b: EVENT FLAG EPISODE DECOMPOSITION")
print("=" * 80)
print("\nFor each event flag, how many distinct episodes (contiguous flagged periods)")
print("make up the 'high' (nonzero) group?\n")

event_flags = ['event_indonesia', 'event_malaysia', 'event_china', 'event_india', 'event_global']
for ef in event_flags:
    nonzero = weekly[ef] != 0
    total_weeks = nonzero.sum()
    if total_weeks == 0:
        print(f"  {ef}: 0 flagged weeks")
        continue
    # Count distinct episodes (contiguous nonzero runs)
    runs = (nonzero != nonzero.shift()).cumsum()
    episodes_flag = nonzero.groupby(runs).sum()
    n_episodes = (episodes_flag > 0).sum()
    # Show each episode's date range and duration
    flagged_weeks = weekly[nonzero].copy()
    flagged_weeks['run_id'] = runs[nonzero]
    print(f"  {ef}: {total_weeks} flagged weeks across {n_episodes} distinct episode(s):")
    for run_id, grp in flagged_weeks.groupby('run_id'):
        start = grp['week_end_date'].min().date()
        end = grp['week_end_date'].max().date()
        dur = len(grp)
        vals = grp[ef].unique()
        print(f"    Episode: {start} to {end} ({dur}w), flag values: {vals}")

# Aggregates
print(f"\n  event_any_flag_active: {(weekly['event_any_flag_active'] == 1).sum()} flagged weeks of {len(weekly)}")
print(f"  event_net_sum nonzero: {(weekly['event_net_sum'] != 0).sum()} weeks")

# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 5: CORRELATION MATRIX (continuous Tier 1 + Tier 2)")
print("=" * 80)

cont_vars = ['stock_to_usage_ratio', 'export_yoy_pct', 'production_yoy_pct', 'enso_oni',
             'palm_soy_spread', 'crude_oil_price', 'crude_oil_chg_4w', 'usd_myr', 'usd_myr_chg_4w']
corr_sub = weekly[cont_vars].dropna()
print(f"\nCorrelation matrix ({len(corr_sub)} complete rows):\n")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
print(corr_sub.corr().round(2).to_string())

print("\n\nProduction/Export/Stock structural check:")
print(corr_sub[['production_yoy_pct', 'export_yoy_pct', 'stock_to_usage_ratio']].corr().round(3).to_string())

# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 6: RANDOM FOREST FEATURE IMPORTANCE (with sample context)")
print("=" * 80)

feature_cols = [v for v in candidate_vars if v in weekly.columns]

for N in [4, 12]:
    target_col = f'persists_{N}w'
    print(f"\n{'='*40}")
    print(f"HORIZON: {N}w")
    print(f"{'='*40}")

    for s in ALL_SHAPES + ['POOLED']:
        if s == 'POOLED':
            sub = weekly[feature_cols + [target_col]].dropna()
        else:
            sub = weekly[weekly['shape'] == s][feature_cols + [target_col]].dropna()

        if len(sub) < 30:
            print(f"\n  Shape {s}: SKIPPED (only {len(sub)} obs)")
            continue

        X = sub[feature_cols].values
        y = sub[target_col].values.astype(int)
        base_rate = y.mean() * 100

        n_resamples = 20
        importances = np.zeros((n_resamples, len(feature_cols)))
        for b in range(n_resamples):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=b, n_jobs=-1)
            rf.fit(X[idx], y[idx])
            importances[b] = rf.feature_importances_

        mean_imp = importances.mean(axis=0)
        std_imp = importances.std(axis=0)
        ranks_per = np.zeros_like(importances)
        for b in range(n_resamples):
            ranks_per[b] = len(feature_cols) - np.argsort(np.argsort(importances[b]))
        mean_rank = ranks_per.mean(axis=0)
        std_rank = ranks_per.std(axis=0)
        rank_order = np.argsort(-mean_imp)

        print(f"\n  Shape {s} ({SHAPE_NAMES.get(s, 'All')}) — n={len(sub)}, base rate={base_rate:.1f}%")
        print(f"  {'Rank':>4} {'Variable':<25} {'MeanImp':>8} {'StdImp':>8} {'MnRank':>7} {'StdRnk':>7} {'Stability':>10}")
        for rank, i in enumerate(rank_order[:10], 1):
            stab = 'STABLE' if std_rank[i] < 2.0 else ('UNSTABLE' if std_rank[i] > 3.5 else 'moderate')
            print(f"  {rank:>4} {feature_cols[i]:<25} {mean_imp[i]:>8.4f} {std_imp[i]:>8.4f} {mean_rank[i]:>7.1f} {std_rank[i]:>7.1f} {stab:>10}")

        # Top 3 features — univariate sample context
        print(f"\n  Top 3 univariate cross-check (tercile n for this shape/horizon):")
        for rank, fi in enumerate(rank_order[:3], 1):
            var = feature_cols[fi]
            var_sub = sub[[var, target_col]].dropna()
            if var in sparse_vars:
                med = var_sub[var].median()
                h = var_sub[var] > med
                l = var_sub[var] <= med
            else:
                q33 = var_sub[var].quantile(0.333)
                q67 = var_sub[var].quantile(0.667)
                h = var_sub[var] >= q67
                l = var_sub[var] <= q33
            nh, nl = h.sum(), l.sum()
            hp = var_sub.loc[h, target_col].mean() * 100 if nh > 0 else 0
            lp = var_sub.loc[l, target_col].mean() * 100 if nl > 0 else 0
            flag = ' ** LOW-N' if nh < 15 or nl < 15 else ''
            print(f"    #{rank} {var}: n_high={nh}, n_low={nl}, high_persist={hp:.1f}%, low_persist={lp:.1f}%, gap={hp-lp:+.1f}pp{flag}")

# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 7: SAMPLE-SIZE-AWARE COMPARISON OF TOP FINDINGS")
print("=" * 80)

print("\nOriginal top findings from Phase A, re-examined with sample sizes:\n")

key_checks = [
    ('usd_myr', 'POOLED', '12w', 'Top RF variable pooled, both horizons'),
    ('usd_myr', '1', '12w', 'Top RF for Shape 1 at 12w'),
    ('enso_oni', '1', '12w', '+43pp gap for Shape 1 persistence'),
    ('enso_oni', '2', '12w', '-48pp gap for Shape 2 persistence'),
    ('palm_soy_spread', '1', '4w', '-42pp gap for Shape 1'),
    ('stock_to_usage_ratio', '0.0', '4w', '-50pp gap — largest in table'),
    ('event_china', 'POOLED', '12w', '+43pp gap — suspected sparse'),
    ('event_india', 'POOLED', '4w', '+38pp gap — suspected sparse'),
    ('event_any_flag_active', 'POOLED', '4w', 'Aggregate event flag'),
    ('event_net_sum', 'POOLED', '4w', 'Aggregate event sum'),
    ('production_yoy_pct', '1', '12w', 'RF top 3 for Shape 1 at 12w'),
    ('crude_oil_price', 'POOLED', '12w', 'RF #2 pooled at 12w'),
]

print(f"{'Variable':<25} {'Shape':>6} {'Hz':>3} {'n_hi':>5} {'n_lo':>5} {'Hi%':>5} {'Lo%':>5} {'Gap':>6} {'Verdict':<20}")
print("-" * 90)

for var, shape, hz, note in key_checks:
    match = res_df[(res_df['variable'] == var) & (res_df['shape'] == shape) & (res_df['horizon'] == hz)]
    if len(match) == 0:
        print(f"{var:<25} {shape:>6} {hz:>3}  {'-- not in table --':<50} {note}")
        continue
    r = match.iloc[0]
    nh, nl = r['n_high'], r['n_low']
    if nh < 15 or nl < 15:
        verdict = 'WEAK (low n)'
    elif abs(r['diff_pp']) >= 20 and nh >= 30 and nl >= 30:
        verdict = 'STRONG'
    elif abs(r['diff_pp']) >= 15:
        verdict = 'MODERATE'
    else:
        verdict = 'MARGINAL'
    print(f"{var:<25} {shape:>6} {hz:>3} {nh:>5} {nl:>5} {r['high_pct']:>4.1f}% {r['low_pct']:>4.1f}% {r['diff_pp']:>+5.1f} {verdict:<20} {note}")

print("\n\nSummary:")
print("  STRONG signals (large gap + adequate n):")
print("    - usd_myr (pooled 12w, Shape 1 12w)")
print("    - enso_oni (Shape 1 12w)")
print("    - crude_oil_price (pooled 12w)")
print("  MODERATE signals:")
print("    - palm_soy_spread, production_yoy_pct")
print("  WEAK / low-n (discount these):")
print("    - event_china, event_india (single-digit n in flagged group)")
print("    - stock_to_usage_ratio for Shape 0.0 at 4w (n_high=8)")
print("    - enso_oni for Shape 2 (n_high=14)")

# ============================================================
print("\n\n" + "=" * 80)
print("PHASE A VARIABLE SCREENING — COMPLETE")
print("=" * 80)
print(f"Output saved to: {os.path.abspath(OUTPUT_PATH)}")

file_buf.close()
