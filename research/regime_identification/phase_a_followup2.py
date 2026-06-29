"""
Phase A Follow-Up #2: USD/MYR vs Crude Oil Independence Check
Appends SECTION 8 to Phase_A_Variable_Screening.txt
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

# ---- Rebuild weekly panel (same as phase_a_followup.py) ----
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
fx_daily = fx_df.set_index('date')['close'].sort_index()
daily['usd_myr'] = fx_daily.reindex(daily.index, method='ffill')
weekly['usd_myr'] = daily.groupby('week_key')['usd_myr'].mean().reindex(weekly.index)

# Crude
cl_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'NYMEX_DL_CL1!, 1D_84001.csv'))
cl_df['date'] = pd.to_datetime(cl_df['time'], unit='s')
cl_daily = cl_df.set_index('date')['close'].sort_index()
daily['crude'] = cl_daily.reindex(daily.index, method='ffill')
weekly['crude_oil_price'] = daily.groupby('week_key')['crude'].mean().reindex(weekly.index)

# Other vars needed for full RF
ps_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'MYX_DLY_FCPO1!_2_CBOT_DL_ZL1!, 1D_61912.csv'))
ps_df['date'] = pd.to_datetime(ps_df['time'], unit='s')
daily['palm_soy'] = ps_df.set_index('date')['close'].sort_index().reindex(daily.index, method='ffill')
weekly['palm_soy_spread'] = daily.groupby('week_key')['palm_soy'].mean().reindex(weekly.index)

weekly['crude_oil_chg_4w'] = weekly['crude_oil_price'].pct_change(4) * 100
weekly['usd_myr_chg_4w'] = weekly['usd_myr'].pct_change(4) * 100

daily['spot_5d_chg'] = daily['spot'].pct_change(5)
weekly['momentum_5d_sign'] = daily.groupby('week_key')['spot_5d_chg'].last().reindex(weekly.index)
weekly['momentum_5d_sign'] = np.sign(weekly['momentum_5d_sign'])

# MPOB
for fname, col in [('FCPO Stock 3Y.xlsx', 'stock'), ('MPOB Production 3Y.xlsx', 'production'),
                    ('MPOB Export 3Y.xlsx', 'export')]:
    df = pd.read_excel(os.path.join(RAW, 'Stock and Production', fname))
    df.columns = ['date', col]
    df = df.iloc[1:]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[col]).set_index('date').sort_index()
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

# ENSO
oni_lines = []
with open(os.path.join(RAW, 'ENSO', 'oni.ascii.txt')) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4:
            try:
                oni_lines.append({'season': parts[0], 'year': int(parts[1]), 'oni': float(parts[3])})
            except (ValueError, IndexError):
                continue
oni_df = pd.DataFrame(oni_lines)
sm = {'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
oni_df['date'] = pd.to_datetime(oni_df['year'].astype(str) + '-' + oni_df['season'].map(sm).astype(str) + '-01')
oni_s = oni_df.set_index('date')['oni'].sort_index()
weekly['enso_oni'] = [oni_s[oni_s.index <= r['week_end_date']].iloc[-1] if (oni_s.index <= r['week_end_date']).any() else np.nan
                      for _, r in weekly.iterrows()]

# Macro
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
# Now do the analysis and append to file
# ============================================================
out_lines = []
def pr(s=''):
    print(s)
    out_lines.append(s)

pr("\n\n" + "=" * 80)
pr("SECTION 8: USD/MYR vs CRUDE OIL INDEPENDENCE CHECK")
pr("=" * 80)

# --- 1. Partial correlation ---
pr("\n--- 8.1 Partial Correlation Check ---\n")
pr("Method: manual residualization (regress X on control, correlate residuals with target)")

sub = weekly[['crude_oil_price', 'usd_myr', 'persists_12w']].dropna()
y = sub['persists_12w'].values

# Simple correlations
r_crude = np.corrcoef(sub['crude_oil_price'].values, y)[0, 1]
r_usd = np.corrcoef(sub['usd_myr'].values, y)[0, 1]

# Partial: crude vs persist controlling for usd_myr
lr = LinearRegression()
lr.fit(sub[['usd_myr']].values, sub['crude_oil_price'].values)
crude_resid = sub['crude_oil_price'].values - lr.predict(sub[['usd_myr']].values)
r_crude_partial = np.corrcoef(crude_resid, y)[0, 1]

# Partial: usd_myr vs persist controlling for crude
lr2 = LinearRegression()
lr2.fit(sub[['crude_oil_price']].values, sub['usd_myr'].values)
usd_resid = sub['usd_myr'].values - lr2.predict(sub[['crude_oil_price']].values)
r_usd_partial = np.corrcoef(usd_resid, y)[0, 1]

pr(f"  n = {len(sub)} weeks (pooled, 12w horizon)")
pr(f"")
pr(f"  {'Measure':<45} {'Correlation':>12}")
pr(f"  {'-'*60}")
pr(f"  {'crude_oil_price vs persists_12w (simple)':<45} {r_crude:>12.4f}")
pr(f"  {'crude_oil_price vs persists_12w | usd_myr':<45} {r_crude_partial:>12.4f}")
pr(f"  {'usd_myr vs persists_12w (simple)':<45} {r_usd:>12.4f}")
pr(f"  {'usd_myr vs persists_12w | crude_oil_price':<45} {r_usd_partial:>12.4f}")
pr(f"")
crude_drop = abs(r_crude) - abs(r_crude_partial)
usd_drop = abs(r_usd) - abs(r_usd_partial)
pr(f"  Drop when controlling: crude_oil drops {crude_drop:+.4f}, usd_myr drops {usd_drop:+.4f}")

# --- 2. Nested RF comparison ---
pr("\n\n--- 8.2 Nested RF Comparison (20 bootstrap resamples each) ---\n")

def run_rf(features, label):
    sub_rf = weekly[features + ['persists_12w']].dropna()
    X = sub_rf[features].values
    y_rf = sub_rf['persists_12w'].values.astype(int)
    n_resamples = 20
    importances = np.zeros((n_resamples, len(features)))
    for b in range(n_resamples):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=b, n_jobs=-1)
        rf.fit(X[idx], y_rf[idx])
        importances[b] = rf.feature_importances_
    mean_imp = importances.mean(axis=0)
    std_imp = importances.std(axis=0)
    ranks_per = np.zeros_like(importances)
    for b in range(n_resamples):
        ranks_per[b] = len(features) - np.argsort(np.argsort(importances[b]))
    mean_rank = ranks_per.mean(axis=0)
    return {features[i]: {'mean_imp': mean_imp[i], 'std_imp': std_imp[i], 'mean_rank': mean_rank[i]}
            for i in range(len(features))}, len(sub_rf)

# Model A: all features
res_a, n_a = run_rf(all_features, 'A')
# Model B: no usd_myr
feats_b = [f for f in all_features if f != 'usd_myr']
res_b, n_b = run_rf(feats_b, 'B')
# Model C: no crude_oil_price
feats_c = [f for f in all_features if f != 'crude_oil_price']
res_c, n_c = run_rf(feats_c, 'C')

pr(f"  Model A: all 17 features (n={n_a})")
pr(f"  Model B: usd_myr removed (n={n_b})")
pr(f"  Model C: crude_oil_price removed (n={n_c})")

pr(f"\n  Comparison for crude_oil_price:")
pr(f"  {'Model':<12} {'Mean Imp':>10} {'Mean Rank':>10}")
pr(f"  {'-'*35}")
pr(f"  {'A (full)':<12} {res_a['crude_oil_price']['mean_imp']:>10.4f} {res_a['crude_oil_price']['mean_rank']:>10.1f}")
pr(f"  {'B (no USD)':<12} {res_b['crude_oil_price']['mean_imp']:>10.4f} {res_b['crude_oil_price']['mean_rank']:>10.1f}")

pr(f"\n  Comparison for usd_myr:")
pr(f"  {'Model':<12} {'Mean Imp':>10} {'Mean Rank':>10}")
pr(f"  {'-'*35}")
pr(f"  {'A (full)':<12} {res_a['usd_myr']['mean_imp']:>10.4f} {res_a['usd_myr']['mean_rank']:>10.1f}")
pr(f"  {'C (no CL)':<12} {res_c['usd_myr']['mean_imp']:>10.4f} {res_c['usd_myr']['mean_rank']:>10.1f}")

# Show full Model B top 5 to see what absorbs USD's share
pr(f"\n  Model B full ranking (top 5, after removing usd_myr):")
sorted_b = sorted(res_b.items(), key=lambda x: -x[1]['mean_imp'])
pr(f"  {'Rank':>4} {'Variable':<25} {'Mean Imp':>10} {'Mean Rank':>10}")
for rank, (var, vals) in enumerate(sorted_b[:5], 1):
    change = ''
    if var in res_a:
        old_rank = res_a[var]['mean_rank']
        new_rank = vals['mean_rank']
        change = f" (was rank {old_rank:.1f})"
    pr(f"  {rank:>4} {var:<25} {vals['mean_imp']:>10.4f} {vals['mean_rank']:>10.1f}{change}")

# --- 3. Two-variable horse race ---
pr("\n\n--- 8.3 Two-Variable Horse Race ---\n")

feats_2 = ['usd_myr', 'crude_oil_price']
res_2, n_2 = run_rf(feats_2, 'horse')
pr(f"  RF with ONLY usd_myr + crude_oil_price (n={n_2}):")
pr(f"  {'Variable':<25} {'Mean Imp':>10}")
pr(f"  {'-'*38}")
for var in feats_2:
    pr(f"  {var:<25} {res_2[var]['mean_imp']:>10.4f}")

ratio = res_2['usd_myr']['mean_imp'] / res_2['crude_oil_price']['mean_imp'] if res_2['crude_oil_price']['mean_imp'] > 0 else float('inf')
pr(f"\n  USD/MYR importance is {ratio:.2f}x that of crude_oil_price in head-to-head.")

# --- 4. Summary ---
pr("\n\n--- 8.4 One-Line Summary ---\n")
pr(f"  USD/MYR simple r={r_usd:.4f}, partial (controlling crude) r={r_usd_partial:.4f} — barely changes.")
pr(f"  Crude oil simple r={r_crude:.4f}, partial (controlling USD) r={r_crude_partial:.4f} — {'drops substantially' if crude_drop > 0.03 else 'modest change' if crude_drop > 0.01 else 'barely changes'}.")
pr(f"  RF: removing usd_myr pushes crude_oil from rank {res_a['crude_oil_price']['mean_rank']:.1f} to {res_b['crude_oil_price']['mean_rank']:.1f}.")
pr(f"  RF: removing crude_oil barely moves usd_myr (rank {res_a['usd_myr']['mean_rank']:.1f} -> {res_c['usd_myr']['mean_rank']:.1f}).")
pr(f"  Head-to-head: USD/MYR takes {res_2['usd_myr']['mean_imp']/(res_2['usd_myr']['mean_imp']+res_2['crude_oil_price']['mean_imp'])*100:.0f}% of importance.")
pr(f"  Conclusion: USD/MYR is the primary signal; crude oil adds {'some' if abs(r_crude_partial) > 0.05 else 'minimal'} independent information beyond it.")

# Append to file
with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
    for line in out_lines:
        f.write(line + '\n')

print(f"\nAppended to: {OUTPUT_PATH}")
