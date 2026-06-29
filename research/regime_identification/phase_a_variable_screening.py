"""
Phase A: Regime Persistence Variable Screening
Run standalone first, then results get appended to FCPO_Spread_Research.ipynb
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Paths
BASE = os.path.join(os.path.dirname(__file__), '..', '..')
RAW = os.path.join(BASE, 'Raw Data')

print("=" * 70)
print("PHASE A: REGIME PERSISTENCE VARIABLE SCREENING")
print("=" * 70)

# ============================================================
# 1. LOAD ALL DATA SOURCES
# ============================================================
print("\n>>> 1. LOADING DATA SOURCES\n")

# --- 1a. Shape log ---
shape_log = pd.read_csv(
    os.path.join(RAW, 'Research', 'daily_shape_log.csv'),
    dtype={'shape': str}, parse_dates=['date']
)
print("1a. Shape log:")
print(f"    Columns: {list(shape_log.columns)}")
print(shape_log.head(3).to_string())
print(f"    dtype of shape: {shape_log['shape'].dtype}")

# --- 1b. MPOB Stock ---
stock_df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'FCPO Stock 3Y.xlsx'))
stock_df.columns = ['date', 'stock']
stock_df = stock_df.iloc[1:]  # skip "Close" header row
stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
stock_df = stock_df.dropna(subset=['date'])
stock_df['stock'] = pd.to_numeric(stock_df['stock'], errors='coerce')
stock_df = stock_df.dropna(subset=['stock']).sort_values('date').reset_index(drop=True)
print("\n1b. MPOB Stock:")
print(f"    Columns: {list(stock_df.columns)}")
print(stock_df.head(3).to_string())

# --- 1c. MPOB Export ---
export_df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'MPOB Export 3Y.xlsx'))
export_df.columns = ['date', 'export']
export_df = export_df.iloc[1:]
export_df['date'] = pd.to_datetime(export_df['date'], errors='coerce')
export_df = export_df.dropna(subset=['date'])
export_df['export'] = pd.to_numeric(export_df['export'], errors='coerce')
export_df = export_df.dropna(subset=['export']).sort_values('date').reset_index(drop=True)
print("\n1c. MPOB Export:")
print(f"    Columns: {list(export_df.columns)}")
print(export_df.head(3).to_string())

# --- 1d. MPOB Production ---
prod_df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'MPOB Production 3Y.xlsx'))
prod_df.columns = ['date', 'production']
prod_df = prod_df.iloc[1:]
prod_df['date'] = pd.to_datetime(prod_df['date'], errors='coerce')
prod_df = prod_df.dropna(subset=['date'])
prod_df['production'] = pd.to_numeric(prod_df['production'], errors='coerce')
prod_df = prod_df.dropna(subset=['production']).sort_values('date').reset_index(drop=True)
print("\n1d. MPOB Production:")
print(f"    Columns: {list(prod_df.columns)}")
print(prod_df.head(3).to_string())

# --- 1e. USD/MYR ---
fx_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'FX_IDC_USDMYR, 1D_1227e.csv'))
fx_df['date'] = pd.to_datetime(fx_df['time'], unit='s')
fx_df = fx_df[['date', 'close']].rename(columns={'close': 'usd_myr'}).sort_values('date')
print("\n1e. USD/MYR:")
print(f"    Columns: {list(fx_df.columns)}, Range: {fx_df['date'].min().date()} to {fx_df['date'].max().date()}")
print(fx_df.head(3).to_string())

# --- 1f. Palm-soy spread (already computed ratio from TradingView) ---
ps_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'MYX_DLY_FCPO1!_2_CBOT_DL_ZL1!, 1D_61912.csv'))
ps_df['date'] = pd.to_datetime(ps_df['time'], unit='s')
ps_df = ps_df[['date', 'close']].rename(columns={'close': 'palm_soy_spread'}).sort_values('date')
print("\n1f. Palm-soy spread (TV combined ratio export):")
print(f"    Columns: {list(ps_df.columns)}, Range: {ps_df['date'].min().date()} to {ps_df['date'].max().date()}")
print(f"    Value range: {ps_df['palm_soy_spread'].min():.2f} to {ps_df['palm_soy_spread'].max():.2f}")
print("    NOTE: This is a pre-computed ratio from TradingView (FCPO / ZL1!), not raw prices.")
print(ps_df.head(3).to_string())

# --- 1g. Crude oil ---
cl_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'NYMEX_DL_CL1!, 1D_84001.csv'))
cl_df['date'] = pd.to_datetime(cl_df['time'], unit='s')
cl_df = cl_df[['date', 'close']].rename(columns={'close': 'crude_oil_price'}).sort_values('date')
print("\n1g. Crude oil (WTI CL1!):")
print(f"    Columns: {list(cl_df.columns)}, Range: {cl_df['date'].min().date()} to {cl_df['date'].max().date()}")
print(cl_df.head(3).to_string())

# --- 1h. Macro Event Calendar ---
macro_df = pd.read_excel(
    os.path.join(RAW, 'Variable Analysis Extra Data', 'FCPO_Macro_Event_Calendar.xlsx'),
    sheet_name='Macro Event Calendar', header=4
)
print("\n1h. Macro Event Calendar:")
print(f"    Columns: {list(macro_df.columns)}")
print(macro_df.head(3).to_string())
# Clean flag columns
flag_cols = {
    'Indonesia Flag': 'event_indonesia',
    'Malaysia Flag': 'event_malaysia',
    'China Flag': 'event_china',
    'India Flag': 'event_india',
    'Global/Other Flag': 'event_global'
}
macro_clean = macro_df[['Year', 'Month']].copy()
for orig, new in flag_cols.items():
    macro_clean[new] = pd.to_numeric(macro_df[orig], errors='coerce').fillna(0).astype(int)
macro_clean['Year'] = pd.to_numeric(macro_clean['Year'], errors='coerce')
macro_clean['Month'] = pd.to_numeric(macro_clean['Month'], errors='coerce')
macro_clean = macro_clean.dropna(subset=['Year', 'Month'])
macro_clean['date'] = pd.to_datetime(
    macro_clean['Year'].astype(int).astype(str) + '-' + macro_clean['Month'].astype(int).astype(str) + '-01'
)
macro_clean = macro_clean.drop(columns=['Year', 'Month']).set_index('date').sort_index()
print("\n    Cleaned macro flags:")
print(macro_clean.head(5).to_string())

# --- 1i. ENSO ONI ---
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
# Map season to month (use middle month of the 3-month window)
season_to_month = {
    'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
    'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
}
oni_df['month'] = oni_df['season'].map(season_to_month)
oni_df['date'] = pd.to_datetime(oni_df['year'].astype(str) + '-' + oni_df['month'].astype(str) + '-01')
oni_df = oni_df[['date', 'oni']].set_index('date').sort_index()
print("\n1i. ENSO ONI:")
print(f"    Range: {oni_df.index.min().date()} to {oni_df.index.max().date()}")
print(oni_df.tail(3).to_string())

# ============================================================
# 2. BUILD WEEKLY PANEL
# ============================================================
print("\n\n>>> 2. BUILDING WEEKLY PANEL\n")

# Start with daily shape log, filter to trusted window
daily = shape_log[['date', 'shape', 'M1']].copy()
daily = daily.rename(columns={'M1': 'spot'})
daily = daily.set_index('date').sort_index()
daily = daily['2017-01-01':]  # trusted window

# Add ISO week key
daily['iso_year'] = daily.index.isocalendar().year.values
daily['iso_week'] = daily.index.isocalendar().week.values
daily['week_key'] = daily['iso_year'].astype(str) + '-W' + daily['iso_week'].astype(str).str.zfill(2)

# Resample to weekly: use last trading day of each ISO week
weekly = daily.groupby('week_key').agg(
    shape=('shape', 'last'),
    spot=('spot', 'last'),
    week_end_date=('spot', lambda x: x.index[-1])
).sort_values('week_end_date')

# Add shape_prev
weekly['shape_prev'] = weekly['shape'].shift(1)

# 5-day momentum sign (use last 5 trading days of each week's window)
daily['spot_5d_chg'] = daily['spot'].pct_change(5)
weekly['momentum_5d_sign'] = daily.groupby(
    daily['iso_year'].astype(str) + '-W' + daily['iso_week'].astype(str).str.zfill(2)
)['spot_5d_chg'].last().reindex(weekly.index)
weekly['momentum_5d_sign'] = np.sign(weekly['momentum_5d_sign'])

# Merge daily external data to weekly
# --- FX ---
fx_daily = fx_df.set_index('date')['usd_myr'].sort_index()
daily_fx = fx_daily.reindex(daily.index, method='ffill')
daily['usd_myr'] = daily_fx
weekly_fx = daily.groupby('week_key')['usd_myr'].mean()
weekly['usd_myr'] = weekly_fx.reindex(weekly.index)

# --- Palm-soy spread ---
ps_daily = ps_df.set_index('date')['palm_soy_spread'].sort_index()
daily['palm_soy'] = ps_daily.reindex(daily.index, method='ffill')
weekly['palm_soy_spread'] = daily.groupby('week_key')['palm_soy'].mean().reindex(weekly.index)

# --- Crude oil ---
cl_daily = cl_df.set_index('date')['crude_oil_price'].sort_index()
daily['crude'] = cl_daily.reindex(daily.index, method='ffill')
weekly['crude_oil_price'] = daily.groupby('week_key')['crude'].mean().reindex(weekly.index)

# 4-week changes
weekly['crude_oil_chg_4w'] = weekly['crude_oil_price'].pct_change(4) * 100
weekly['usd_myr_chg_4w'] = weekly['usd_myr'].pct_change(4) * 100

# --- MPOB monthly -> forward-fill to weekly ---
# Build monthly MPOB panel
mpob = stock_df.set_index('date')[['stock']].join(
    prod_df.set_index('date')[['production']], how='outer'
).join(
    export_df.set_index('date')[['export']], how='outer'
).sort_index()

# stock_to_usage_ratio: stock / production (fallback since no import data)
mpob['stock_to_usage_ratio'] = mpob['stock'] / mpob['production']
print("NOTE: stock_to_usage_ratio = stock / production (no import data available for full usage calc)")

# YoY %
mpob['export_yoy_pct'] = mpob['export'].pct_change(12) * 100
mpob['production_yoy_pct'] = mpob['production'].pct_change(12) * 100

# Forward-fill monthly MPOB to weekly via week_end_date
for col in ['stock_to_usage_ratio', 'export_yoy_pct', 'production_yoy_pct']:
    mpob_series = mpob[col].dropna()
    # For each week, find the most recent monthly observation
    vals = []
    for _, row in weekly.iterrows():
        dt = row['week_end_date']
        mask = mpob_series.index <= dt
        if mask.any():
            vals.append(mpob_series[mask].iloc[-1])
        else:
            vals.append(np.nan)
    weekly[col] = vals

# --- ENSO ONI monthly -> forward-fill to weekly ---
oni_series = oni_df['oni']
vals = []
for _, row in weekly.iterrows():
    dt = row['week_end_date']
    mask = oni_series.index <= dt
    if mask.any():
        vals.append(oni_series[mask].iloc[-1])
    else:
        vals.append(np.nan)
weekly['enso_oni'] = vals

# --- Macro event flags monthly -> forward-fill to weekly ---
for col in macro_clean.columns:
    series = macro_clean[col]
    vals = []
    for _, row in weekly.iterrows():
        dt = row['week_end_date']
        mask = series.index <= dt
        if mask.any():
            vals.append(series[mask].iloc[-1])
        else:
            vals.append(np.nan)
    weekly[col] = vals

weekly['event_any_flag_active'] = (weekly[['event_indonesia', 'event_malaysia', 'event_china',
                                           'event_india', 'event_global']].abs().sum(axis=1) > 0).astype(int)
weekly['event_net_sum'] = weekly[['event_indonesia', 'event_malaysia', 'event_china',
                                  'event_india', 'event_global']].sum(axis=1)

# Trim to trusted window: complete weeks only
weekly = weekly[weekly['week_end_date'] >= '2017-01-01']
weekly = weekly[weekly['week_end_date'] <= weekly['week_end_date'].max()]

print(f"\nWeekly panel shape: {weekly.shape}")
print(f"\nPer-column non-null counts:")
print(weekly.notna().sum().to_string())
print(f"\n.head(20):")
print(weekly.head(20).to_string())
print(f"\n.describe():")
print(weekly.describe().to_string())

# ============================================================
# 3. BUILD TARGET VARIABLES
# ============================================================
print("\n\n>>> 3. TARGET VARIABLES\n")

# --- 3a. Fixed-horizon persistence ---
print("--- 3a. Fixed-horizon persistence ---\n")

ALL_SHAPES = ['0.0', '0.1', '0.2', '1', '2']

for N in [4, 12]:
    weekly[f'shape_plus_{N}w'] = weekly['shape'].shift(-N)
    weekly[f'persists_{N}w'] = (weekly['shape'] == weekly[f'shape_plus_{N}w']).astype(int)
    # NaN where future shape is unavailable
    weekly.loc[weekly[f'shape_plus_{N}w'].isna(), f'persists_{N}w'] = np.nan

print("BASE PERSISTENCE RATES (naive baseline):")
print(f"{'Shape':<8} {'N=4w persist%':>15} {'N=4w count':>12} {'N=12w persist%':>16} {'N=12w count':>13}")
print("-" * 70)
for s in ALL_SHAPES:
    mask = weekly['shape'] == s
    for N in [4, 12]:
        col = f'persists_{N}w'
        sub = weekly.loc[mask, col].dropna()
        pct = sub.mean() * 100 if len(sub) > 0 else 0
        if N == 4:
            p4, c4 = pct, len(sub)
        else:
            p12, c12 = pct, len(sub)
    print(f"{s:<8} {p4:>14.1f}% {c4:>11} {p12:>15.1f}% {c12:>12}")

# Pooled
for N in [4, 12]:
    col = f'persists_{N}w'
    sub = weekly[col].dropna()
    print(f"{'POOLED':<8} N={N}w: {sub.mean()*100:.1f}% persist ({len(sub)} obs)")

# --- 3b. Survival-style ---
print("\n--- 3b. Survival-style (time-to-event) ---\n")

# Identify regime episodes
weekly_sorted = weekly.sort_values('week_end_date').copy()
weekly_sorted['regime_start'] = weekly_sorted['shape'] != weekly_sorted['shape_prev']

episodes = []
current_shape = None
start_idx = None
for i, (idx, row) in enumerate(weekly_sorted.iterrows()):
    if row['regime_start'] or current_shape is None:
        if current_shape is not None:
            episodes.append({
                'shape': current_shape,
                'start_week': start_idx,
                'duration_weeks': i - start_i,
                'censored': False
            })
        current_shape = row['shape']
        start_idx = idx
        start_i = i
# Last episode is censored (still ongoing)
if current_shape is not None:
    episodes.append({
        'shape': current_shape,
        'start_week': start_idx,
        'duration_weeks': len(weekly_sorted) - start_i,
        'censored': True
    })

ep_df = pd.DataFrame(episodes)
print(f"Total episodes: {len(ep_df)}")
print(f"\nEpisode counts and censoring rates per shape:")
print(f"{'Shape':<8} {'Episodes':>10} {'Censored':>10} {'Censor%':>10} {'Mean dur':>10} {'Median dur':>12}")
print("-" * 65)

from lifelines import KaplanMeierFitter

for s in ALL_SHAPES:
    sub = ep_df[ep_df['shape'] == s]
    n = len(sub)
    n_cens = sub['censored'].sum()
    cens_pct = 100 * n_cens / n if n > 0 else 0
    # Observed (non-censored) stats
    obs = sub[~sub['censored']]['duration_weeks']
    mean_d = obs.mean() if len(obs) > 0 else float('nan')
    # KM median
    if n > 0:
        kmf = KaplanMeierFitter()
        kmf.fit(sub['duration_weeks'], event_observed=~sub['censored'])
        km_median = kmf.median_survival_time_
    else:
        km_median = float('nan')
    print(f"{s:<8} {n:>10} {n_cens:>10} {cens_pct:>9.1f}% {mean_d:>9.1f}w {km_median:>11}w")

# ============================================================
# 4. UNIVARIATE SCREENING
# ============================================================
print("\n\n>>> 4. UNIVARIATE SCREENING\n")

candidate_vars = [
    # Tier 1
    'stock_to_usage_ratio', 'export_yoy_pct', 'production_yoy_pct', 'enso_oni', 'palm_soy_spread',
    # Tier 2
    'crude_oil_price', 'crude_oil_chg_4w', 'usd_myr', 'usd_myr_chg_4w', 'momentum_5d_sign',
    # Event flags
    'event_indonesia', 'event_malaysia', 'event_china', 'event_india', 'event_global',
    'event_any_flag_active', 'event_net_sum'
]

sparse_vars = {'event_indonesia', 'event_malaysia', 'event_china', 'event_india', 'event_global',
               'event_any_flag_active', 'momentum_5d_sign'}

results_persistence = []

for var in candidate_vars:
    for N in [4, 12]:
        target_col = f'persists_{N}w'
        sub = weekly[[var, target_col, 'shape']].dropna()
        if len(sub) < 20:
            continue

        # Use median split for sparse vars, tercile for continuous
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

        if high_mask.sum() < 5 or low_mask.sum() < 5:
            continue

        high_rate = sub.loc[high_mask, target_col].mean() * 100
        low_rate = sub.loc[low_mask, target_col].mean() * 100
        diff = high_rate - low_rate

        results_persistence.append({
            'variable': var,
            'horizon': f'{N}w',
            'shape': 'POOLED',
            'high_persist%': round(high_rate, 1),
            'low_persist%': round(low_rate, 1),
            'diff_pp': round(diff, 1),
            'abs_diff': round(abs(diff), 1),
            'n_high': int(high_mask.sum()),
            'n_low': int(low_mask.sum()),
            'split': split_type
        })

        # Per-shape
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
            hr = s_sub.loc[h, target_col].mean() * 100
            lr = s_sub.loc[l, target_col].mean() * 100
            results_persistence.append({
                'variable': var,
                'horizon': f'{N}w',
                'shape': s,
                'high_persist%': round(hr, 1),
                'low_persist%': round(lr, 1),
                'diff_pp': round(hr - lr, 1),
                'abs_diff': round(abs(hr - lr), 1),
                'n_high': int(h.sum()),
                'n_low': int(l.sum()),
                'split': split_type
            })

res_df = pd.DataFrame(results_persistence)
res_df = res_df.sort_values('abs_diff', ascending=False)

print("FIXED-HORIZON PERSISTENCE: UNIVARIATE SCREENING (sorted by |diff|)")
print("=" * 110)
print(f"{'Variable':<25} {'Horizon':>7} {'Shape':>7} {'High%':>7} {'Low%':>7} {'Diff(pp)':>9} {'n_high':>7} {'n_low':>7} {'Split':>8}")
print("-" * 110)
for _, r in res_df.head(60).iterrows():
    print(f"{r['variable']:<25} {r['horizon']:>7} {r['shape']:>7} {r['high_persist%']:>6.1f}% {r['low_persist%']:>6.1f}% {r['diff_pp']:>+8.1f} {r['n_high']:>7} {r['n_low']:>7} {r['split']:>8}")

# --- Survival univariate ---
print("\n\nSURVIVAL DURATION: UNIVARIATE SCREENING")
print("=" * 90)

# Map each episode to the variable value at its start week
ep_df_enriched = ep_df.copy()
for var in candidate_vars:
    vals = []
    for _, ep in ep_df_enriched.iterrows():
        wk = ep['start_week']
        if wk in weekly.index:
            vals.append(weekly.loc[wk, var])
        else:
            vals.append(np.nan)
    ep_df_enriched[var] = vals

surv_results = []
for var in candidate_vars:
    sub = ep_df_enriched[[var, 'duration_weeks', 'censored', 'shape']].dropna(subset=[var])
    if len(sub) < 10:
        continue
    if var in sparse_vars:
        med = sub[var].median()
        h = sub[var] > med
        l = sub[var] <= med
    else:
        q33 = sub[var].quantile(0.333)
        q67 = sub[var].quantile(0.667)
        h = sub[var] >= q67
        l = sub[var] <= q33

    if h.sum() < 3 or l.sum() < 3:
        continue

    # KM median for high and low groups
    for label, mask in [('high', h), ('low', l)]:
        grp = sub[mask]
        if len(grp) >= 3:
            kmf = KaplanMeierFitter()
            kmf.fit(grp['duration_weeks'], event_observed=~grp['censored'])
            km_med = kmf.median_survival_time_
        else:
            km_med = np.nan
        if label == 'high':
            h_med, h_n = km_med, len(grp)
        else:
            l_med, l_n = km_med, len(grp)

    surv_results.append({
        'variable': var,
        'high_KM_median': h_med,
        'low_KM_median': l_med,
        'diff': h_med - l_med if pd.notna(h_med) and pd.notna(l_med) else np.nan,
        'n_high': h_n,
        'n_low': l_n
    })

surv_df = pd.DataFrame(surv_results)
surv_df['abs_diff'] = surv_df['diff'].abs()
surv_df = surv_df.sort_values('abs_diff', ascending=False)

print(f"{'Variable':<25} {'High KM med':>12} {'Low KM med':>12} {'Diff(w)':>9} {'n_high':>7} {'n_low':>7}")
print("-" * 75)
for _, r in surv_df.iterrows():
    h_str = f"{r['high_KM_median']:.1f}" if pd.notna(r['high_KM_median']) else 'inf'
    l_str = f"{r['low_KM_median']:.1f}" if pd.notna(r['low_KM_median']) else 'inf'
    d_str = f"{r['diff']:+.1f}" if pd.notna(r['diff']) else 'N/A'
    print(f"{r['variable']:<25} {h_str:>12} {l_str:>12} {d_str:>9} {r['n_high']:>7} {r['n_low']:>7}")

# ============================================================
# 5. CORRELATION / STRUCTURAL CHECK
# ============================================================
print("\n\n>>> 5. CORRELATION MATRIX (continuous Tier 1 + Tier 2 variables)\n")

cont_vars = ['stock_to_usage_ratio', 'export_yoy_pct', 'production_yoy_pct', 'enso_oni',
             'palm_soy_spread', 'crude_oil_price', 'crude_oil_chg_4w', 'usd_myr', 'usd_myr_chg_4w']

corr_sub = weekly[cont_vars].dropna()
print(f"Correlation matrix ({len(corr_sub)} complete rows):\n")
corr_mat = corr_sub.corr()
# Print with 2 decimal places
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
print(corr_mat.round(2).to_string())

print("\n\nSPECIFIC CHECK: production/export/stock correlations:")
check_vars = ['production_yoy_pct', 'export_yoy_pct', 'stock_to_usage_ratio']
print(corr_sub[check_vars].corr().round(3).to_string())

# ============================================================
# 6. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================
print("\n\n>>> 6. RANDOM FOREST FEATURE IMPORTANCE (bootstrapped)\n")

from sklearn.ensemble import RandomForestClassifier

feature_cols = [v for v in candidate_vars if v in weekly.columns]

for N in [4, 12]:
    target_col = f'persists_{N}w'
    print(f"\n--- Horizon: {N}w ---")

    for s in ALL_SHAPES + ['POOLED']:
        if s == 'POOLED':
            sub = weekly[feature_cols + [target_col]].dropna()
        else:
            sub = weekly[weekly['shape'] == s][feature_cols + [target_col]].dropna()

        if len(sub) < 30:
            print(f"  Shape {s}: SKIPPED (only {len(sub)} obs)")
            continue

        X = sub[feature_cols].values
        y = sub[target_col].values.astype(int)

        n_resamples = 20
        importances = np.zeros((n_resamples, len(feature_cols)))
        for b in range(n_resamples):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=b, n_jobs=-1)
            rf.fit(X[idx], y[idx])
            importances[b] = rf.feature_importances_

        mean_imp = importances.mean(axis=0)
        std_imp = importances.std(axis=0)

        # Rank by mean importance
        rank_order = np.argsort(-mean_imp)
        # Also compute rank stability
        ranks_per_resample = np.zeros_like(importances)
        for b in range(n_resamples):
            ranks_per_resample[b] = len(feature_cols) - np.argsort(np.argsort(importances[b]))
        mean_rank = ranks_per_resample.mean(axis=0)
        std_rank = ranks_per_resample.std(axis=0)

        print(f"\n  Shape {s} ({len(sub)} obs, base rate {y.mean()*100:.1f}%):")
        print(f"  {'Rank':>4} {'Variable':<25} {'Mean Imp':>10} {'Std Imp':>10} {'Mean Rank':>10} {'Std Rank':>10} {'Stability':>10}")
        for rank, i in enumerate(rank_order[:10], 1):
            stability = 'STABLE' if std_rank[i] < 2.0 else 'UNSTABLE' if std_rank[i] > 3.5 else 'moderate'
            print(f"  {rank:>4} {feature_cols[i]:<25} {mean_imp[i]:>10.4f} {std_imp[i]:>10.4f} {mean_rank[i]:>10.1f} {std_rank[i]:>10.1f} {stability:>10}")

# ============================================================
# 7. DATA ISSUES SUMMARY
# ============================================================
print("\n\n>>> 7. DATA ISSUES SUMMARY\n")

print("1. stock_to_usage_ratio: Uses stock/production as fallback (no import data available).")
print("   Full usage = production + imports - exports; without imports, this is an approximation.")
print(f"2. MPOB data range: Stock {stock_df['date'].min().date()} to {stock_df['date'].max().date()}")
print(f"   Production {prod_df['date'].min().date()} to {prod_df['date'].max().date()}")
print(f"   Export {export_df['date'].min().date()} to {export_df['date'].max().date()}")
print(f"3. Palm-soy spread: TV ratio export, {ps_df['date'].min().date()} to {ps_df['date'].max().date()}")
print(f"   Pre-2017 data available but only 2017+ used in panel.")
print(f"4. USD/MYR: {fx_df['date'].min().date()} to {fx_df['date'].max().date()}")
print(f"5. Crude oil: {cl_df['date'].min().date()} to {cl_df['date'].max().date()}")
print(f"6. Macro calendar: {macro_clean.index.min().date()} to {macro_clean.index.max().date()}")
print(f"7. ENSO ONI: {oni_df.index.min().date()} to {oni_df.index.max().date()}")
print(f"8. Shape log trusted window: 2017-01-02 to {daily.index.max().date()}")
print(f"9. Weekly panel final shape: {weekly.shape}")

# Missing data by variable
print("\n10. Missing data in weekly panel:")
for col in candidate_vars:
    if col in weekly.columns:
        n_miss = weekly[col].isna().sum()
        pct = 100 * n_miss / len(weekly)
        if n_miss > 0:
            print(f"    {col}: {n_miss} missing ({pct:.1f}%)")

print("\n" + "=" * 70)
print("PHASE A COMPLETE")
print("=" * 70)
