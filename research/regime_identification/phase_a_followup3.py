"""
Phase A Follow-Up #3: Episode-Exclusion Robustness Check
Appends SECTION 9 to Phase_A_Variable_Screening.txt
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
RAW = os.path.join(BASE, 'Raw Data')
OUTPUT_PATH = os.path.join(BASE, 'research', 'outputs', 'Phase_A_Variable_Screening.txt')

# ---- Rebuild weekly panel (same as previous follow-ups) ----
shape_log = pd.read_csv(os.path.join(RAW, 'Research', 'daily_shape_log.csv'),
                         dtype={'shape': str}, parse_dates=['date'])
daily = shape_log[['date', 'shape', 'M1']].copy().rename(columns={'M1': 'spot'})
daily = daily.set_index('date').sort_index()['2017-01-01':]
daily['iso_year'] = daily.index.isocalendar().year.values
daily['iso_week'] = daily.index.isocalendar().week.values
daily['week_key'] = daily['iso_year'].astype(str) + '-W' + daily['iso_week'].astype(str).str.zfill(2)

weekly = daily.groupby('week_key').agg(
    shape=('shape', 'last'), spot=('spot', 'last'),
    week_end_date=('spot', lambda x: x.index[-1])
).sort_values('week_end_date')
weekly['shape_prev'] = weekly['shape'].shift(1)

# FX
fx_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'FX_IDC_USDMYR, 1D_1227e.csv'))
fx_df['date'] = pd.to_datetime(fx_df['time'], unit='s')
daily['usd_myr'] = fx_df.set_index('date')['close'].sort_index().reindex(daily.index, method='ffill')
weekly['usd_myr'] = daily.groupby('week_key')['usd_myr'].mean().reindex(weekly.index)

# Crude
cl_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'NYMEX_DL_CL1!, 1D_84001.csv'))
cl_df['date'] = pd.to_datetime(cl_df['time'], unit='s')
daily['crude'] = cl_df.set_index('date')['close'].sort_index().reindex(daily.index, method='ffill')
weekly['crude_oil_price'] = daily.groupby('week_key')['crude'].mean().reindex(weekly.index)

# Other features for RF
ps_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'MYX_DLY_FCPO1!_2_CBOT_DL_ZL1!, 1D_61912.csv'))
ps_df['date'] = pd.to_datetime(ps_df['time'], unit='s')
daily['palm_soy'] = ps_df.set_index('date')['close'].sort_index().reindex(daily.index, method='ffill')
weekly['palm_soy_spread'] = daily.groupby('week_key')['palm_soy'].mean().reindex(weekly.index)
weekly['crude_oil_chg_4w'] = weekly['crude_oil_price'].pct_change(4) * 100
weekly['usd_myr_chg_4w'] = weekly['usd_myr'].pct_change(4) * 100
daily['spot_5d_chg'] = daily['spot'].pct_change(5)
weekly['momentum_5d_sign'] = np.sign(daily.groupby('week_key')['spot_5d_chg'].last().reindex(weekly.index))

for fname, col in [('FCPO Stock 3Y.xlsx', 'stock'), ('MPOB Production 3Y.xlsx', 'production'),
                    ('MPOB Export 3Y.xlsx', 'export')]:
    df = pd.read_excel(os.path.join(RAW, 'Stock and Production', fname))
    df.columns = ['date', col]
    df = df.iloc[1:]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().set_index('date').sort_index()
    if col == 'stock':
        mpob = df
    else:
        mpob = mpob.join(df, how='outer')
mpob['stock_to_usage_ratio'] = mpob['stock'] / mpob['production']
mpob['export_yoy_pct'] = mpob['export'].pct_change(12) * 100
mpob['production_yoy_pct'] = mpob['production'].pct_change(12) * 100
for col in ['stock_to_usage_ratio', 'export_yoy_pct', 'production_yoy_pct']:
    s = mpob[col].dropna()
    weekly[col] = [s[s.index <= r['week_end_date']].iloc[-1] if (s.index <= r['week_end_date']).any() else np.nan
                   for _, r in weekly.iterrows()]

oni_lines = []
with open(os.path.join(RAW, 'ENSO', 'oni.ascii.txt')) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4:
            try: oni_lines.append({'season': parts[0], 'year': int(parts[1]), 'oni': float(parts[3])})
            except: continue
oni_df = pd.DataFrame(oni_lines)
sm = {'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
oni_df['date'] = pd.to_datetime(oni_df['year'].astype(str) + '-' + oni_df['season'].map(sm).astype(str) + '-01')
oni_s = oni_df.set_index('date')['oni'].sort_index()
weekly['enso_oni'] = [oni_s[oni_s.index <= r['week_end_date']].iloc[-1] if (oni_s.index <= r['week_end_date']).any() else np.nan
                      for _, r in weekly.iterrows()]

macro_df = pd.read_excel(os.path.join(RAW, 'Variable Analysis Extra Data', 'FCPO_Macro_Event_Calendar.xlsx'),
                          sheet_name='Macro Event Calendar', header=4)
flag_map = {'Indonesia Flag': 'event_indonesia', 'Malaysia Flag': 'event_malaysia',
            'China Flag': 'event_china', 'India Flag': 'event_india', 'Global/Other Flag': 'event_global'}
mc = macro_df[['Year', 'Month']].copy()
for orig, new in flag_map.items():
    mc[new] = pd.to_numeric(macro_df[orig], errors='coerce').fillna(0).astype(int)
mc['Year'] = pd.to_numeric(mc['Year'], errors='coerce')
mc['Month'] = pd.to_numeric(mc['Month'], errors='coerce')
mc = mc.dropna(subset=['Year', 'Month'])
mc['date'] = pd.to_datetime(mc['Year'].astype(int).astype(str) + '-' + mc['Month'].astype(int).astype(str) + '-01')
mc = mc.drop(columns=['Year', 'Month']).set_index('date').sort_index()
for col in mc.columns:
    s = mc[col]
    weekly[col] = [s[s.index <= r['week_end_date']].iloc[-1] if (s.index <= r['week_end_date']).any() else np.nan
                   for _, r in weekly.iterrows()]
weekly['event_any_flag_active'] = (weekly[['event_indonesia','event_malaysia','event_china',
                                           'event_india','event_global']].abs().sum(axis=1) > 0).astype(int)
weekly['event_net_sum'] = weekly[['event_indonesia','event_malaysia','event_china',
                                  'event_india','event_global']].sum(axis=1)

# Target
weekly['shape_plus_12w'] = weekly['shape'].shift(-12)
weekly['persists_12w'] = (weekly['shape'] == weekly['shape_plus_12w']).astype(int)
weekly.loc[weekly['shape_plus_12w'].isna(), 'persists_12w'] = np.nan

all_features = ['stock_to_usage_ratio', 'export_yoy_pct', 'production_yoy_pct', 'enso_oni',
                'palm_soy_spread', 'crude_oil_price', 'crude_oil_chg_4w', 'usd_myr', 'usd_myr_chg_4w',
                'momentum_5d_sign', 'event_indonesia', 'event_malaysia', 'event_china', 'event_india',
                'event_global', 'event_any_flag_active', 'event_net_sum']

# ============================================================
# SECTION 9: Episode-Exclusion Robustness Check
# ============================================================
out_lines = []
def pr(s=''):
    print(s)
    out_lines.append(s)

pr("\n\n" + "=" * 80)
pr("SECTION 9: EPISODE-EXCLUSION ROBUSTNESS CHECK")
pr("=" * 80)

# --- 1. Define exclusion windows ---
pr("\n--- 9.1 Exclusion Windows ---\n")

covid_start = pd.Timestamp('2020-03-01')
covid_end = pd.Timestamp('2020-06-30')
ban_start = pd.Timestamp('2022-01-01')
ban_end = pd.Timestamp('2022-10-31')

covid_mask = (weekly['week_end_date'] >= covid_start) & (weekly['week_end_date'] <= covid_end)
ban_mask = (weekly['week_end_date'] >= ban_start) & (weekly['week_end_date'] <= ban_end)
shock_mask = covid_mask | ban_mask

n_covid = covid_mask.sum()
n_ban = ban_mask.sum()
n_shock = shock_mask.sum()
n_remain = (~shock_mask).sum()

pr(f"  COVID window (2020-03 to 2020-06): {n_covid} weeks excluded")
pr(f"  Export-ban window (2022-01 to 2022-10): {n_ban} weeks excluded")
pr(f"  Total excluded: {n_shock} weeks")
pr(f"  Remaining panel: {n_remain} weeks (of {len(weekly)} total)")

# Quick sanity: what were USD/MYR and crude doing in those windows?
for label, mask in [('COVID', covid_mask), ('Export-ban', ban_mask)]:
    sub = weekly[mask]
    pr(f"\n  {label} window snapshot:")
    pr(f"    USD/MYR: {sub['usd_myr'].min():.3f} - {sub['usd_myr'].max():.3f} (range {sub['usd_myr'].max()-sub['usd_myr'].min():.3f})")
    pr(f"    Crude oil: {sub['crude_oil_price'].min():.1f} - {sub['crude_oil_price'].max():.1f}")
    pr(f"    Spot FCPO: {sub['spot'].min():.0f} - {sub['spot'].max():.0f}")
    pr(f"    Shapes present: {sorted(sub['shape'].unique())}")

# Create filtered panels
weekly_clean = weekly[~shock_mask].copy()
weekly_shock = weekly[shock_mask].copy()

# --- 2. Re-run screening on filtered panel ---
pr("\n\n--- 9.2 Filtered Panel (shocks excluded) — Univariate Screening ---\n")

def tercile_screening(panel, var, target_col, shape_filter=None, label=''):
    if shape_filter:
        sub = panel[panel['shape'] == shape_filter][[var, target_col]].dropna()
    else:
        sub = panel[[var, target_col]].dropna()
    if len(sub) < 10:
        return None
    q33 = sub[var].quantile(0.333)
    q67 = sub[var].quantile(0.667)
    h = sub[var] >= q67
    l = sub[var] <= q33
    if h.sum() < 3 or l.sum() < 3:
        return None
    hp = sub.loc[h, target_col]
    lp = sub.loc[l, target_col]
    return {
        'label': label,
        'high_pct': hp.mean() * 100,
        'low_pct': lp.mean() * 100,
        'gap': (hp.mean() - lp.mean()) * 100,
        'n_high': int(h.sum()),
        'n_low': int(l.sum()),
        'n_high_persist': int(hp.sum()),
        'n_low_persist': int(lp.sum()),
    }

cuts = [
    ('usd_myr', None, 'usd_myr / POOLED / 12w'),
    ('usd_myr', '1', 'usd_myr / Shape 1 / 12w'),
    ('crude_oil_price', None, 'crude_oil / POOLED / 12w'),
    ('crude_oil_price', '1', 'crude_oil / Shape 1 / 12w'),
]

# Original (all weeks)
orig_results = {}
for var, shape, label in cuts:
    r = tercile_screening(weekly, var, 'persists_12w', shape, label)
    if r:
        orig_results[label] = r

# Filtered (shocks excluded)
filt_results = {}
for var, shape, label in cuts:
    r = tercile_screening(weekly_clean, var, 'persists_12w', shape, label)
    if r:
        filt_results[label] = r

pr(f"  {'Cut':<30} {'Original':>30} {'Filtered (no shocks)':>30}")
pr(f"  {'':30} {'Hi%   Lo%   Gap   n_hi n_lo':>30} {'Hi%   Lo%   Gap   n_hi n_lo':>30}")
pr(f"  {'-'*95}")
for var, shape, label in cuts:
    o = orig_results.get(label)
    f = filt_results.get(label)
    if o and f:
        o_str = f"{o['high_pct']:5.1f}% {o['low_pct']:5.1f}% {o['gap']:+5.1f} {o['n_high']:>4} {o['n_low']:>4}"
        f_str = f"{f['high_pct']:5.1f}% {f['low_pct']:5.1f}% {f['gap']:+5.1f} {f['n_high']:>4} {f['n_low']:>4}"
        pr(f"  {label:<30} {o_str:>30} {f_str:>30}")

# RF on filtered panel
pr("\n  RF on filtered panel (pooled 12w, 20 bootstraps):")

def run_rf_report(panel, features, label):
    sub = panel[features + ['persists_12w']].dropna()
    X = sub[features].values
    y = sub['persists_12w'].values.astype(int)
    n_resamples = 20
    importances = np.zeros((n_resamples, len(features)))
    for b in range(n_resamples):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=b, n_jobs=-1)
        rf.fit(X[idx], y[idx])
        importances[b] = rf.feature_importances_
    mean_imp = importances.mean(axis=0)
    ranks_per = np.zeros_like(importances)
    for b in range(n_resamples):
        ranks_per[b] = len(features) - np.argsort(np.argsort(importances[b]))
    mean_rank = ranks_per.mean(axis=0)
    return {features[i]: {'mean_imp': mean_imp[i], 'mean_rank': mean_rank[i]} for i in range(len(features))}, len(sub)

rf_orig, n_orig = run_rf_report(weekly, all_features, 'original')
rf_filt, n_filt = run_rf_report(weekly_clean, all_features, 'filtered')

pr(f"\n  {'Variable':<25} {'Original (n={n_orig})':>25} {'Filtered (n={n_filt})':>25}")
pr(f"  {'':25} {'Imp     Rank':>25} {'Imp     Rank':>25}")
pr(f"  {'-'*78}")
for var in ['usd_myr', 'crude_oil_price', 'enso_oni', 'palm_soy_spread', 'production_yoy_pct']:
    o = rf_orig[var]
    f = rf_filt[var]
    pr(f"  {var:<25} {o['mean_imp']:>8.4f} {o['mean_rank']:>6.1f}        {f['mean_imp']:>8.4f} {f['mean_rank']:>6.1f}")

# --- 3. Shock windows only ---
pr("\n\n--- 9.3 Shock Windows Only (COVID + Export Ban) ---\n")

shock_results = {}
for var, shape, label in cuts:
    r = tercile_screening(weekly_shock, var, 'persists_12w', shape, label)
    if r:
        shock_results[label] = r
    else:
        shock_results[label] = None

pr(f"  {'Cut':<30} {'Hi%':>6} {'Lo%':>6} {'Gap':>7} {'n_hi':>5} {'n_lo':>5}")
pr(f"  {'-'*65}")
for var, shape, label in cuts:
    r = shock_results.get(label)
    if r:
        pr(f"  {label:<30} {r['high_pct']:>5.1f}% {r['low_pct']:>5.1f}% {r['gap']:>+6.1f} {r['n_high']:>5} {r['n_low']:>5}")
    else:
        pr(f"  {label:<30}  -- insufficient data for tercile split --")

# --- 4. Three-way comparison table ---
pr("\n\n--- 9.4 Three-Way Comparison Table ---\n")

pr(f"  {'Cut':<30} {'Gap (all)':>10} {'Gap (no shock)':>15} {'Gap (shock only)':>17}")
pr(f"  {'-'*75}")
for var, shape, label in cuts:
    o = orig_results.get(label)
    f = filt_results.get(label)
    s = shock_results.get(label)
    g_all = f"{o['gap']:+.1f}" if o else 'N/A'
    g_filt = f"{f['gap']:+.1f}" if f else 'N/A'
    g_shock = f"{s['gap']:+.1f}" if s else 'N/A'
    pr(f"  {label:<30} {g_all:>10} {g_filt:>15} {g_shock:>17}")

pr(f"\n  Note: 'Gap' = high_tercile_persist% minus low_tercile_persist% (pp).")
pr(f"  Positive gap = high variable value -> more persistence.")
pr(f"  Negative gap = high variable value -> less persistence (regime breaks sooner).")

# Append to file
with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
    for line in out_lines:
        f.write(line + '\n')

print(f"\nAppended SECTION 9 to: {OUTPUT_PATH}")
