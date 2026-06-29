"""
Phase A Follow-Up #4: ENSO Episode-Exclusion Check
Appends SECTION 10 to Phase_A_Variable_Screening.txt
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
RAW = os.path.join(BASE, 'Raw Data')
OUTPUT_PATH = os.path.join(BASE, 'research', 'outputs', 'Phase_A_Variable_Screening.txt')

# ---- Rebuild weekly panel (same as prior follow-ups) ----
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

# All external vars
fx_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'FX_IDC_USDMYR, 1D_1227e.csv'))
fx_df['date'] = pd.to_datetime(fx_df['time'], unit='s')
daily['usd_myr'] = fx_df.set_index('date')['close'].sort_index().reindex(daily.index, method='ffill')
weekly['usd_myr'] = daily.groupby('week_key')['usd_myr'].mean().reindex(weekly.index)

cl_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'NYMEX_DL_CL1!, 1D_84001.csv'))
cl_df['date'] = pd.to_datetime(cl_df['time'], unit='s')
daily['crude'] = cl_df.set_index('date')['close'].sort_index().reindex(daily.index, method='ffill')
weekly['crude_oil_price'] = daily.groupby('week_key')['crude'].mean().reindex(weekly.index)

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
    df.columns = ['date', col]; df = df.iloc[1:]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().set_index('date').sort_index()
    if col == 'stock': mpob = df
    else: mpob = mpob.join(df, how='outer')
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

weekly['shape_plus_12w'] = weekly['shape'].shift(-12)
weekly['persists_12w'] = (weekly['shape'] == weekly['shape_plus_12w']).astype(int)
weekly.loc[weekly['shape_plus_12w'].isna(), 'persists_12w'] = np.nan

all_features = ['stock_to_usage_ratio', 'export_yoy_pct', 'production_yoy_pct', 'enso_oni',
                'palm_soy_spread', 'crude_oil_price', 'crude_oil_chg_4w', 'usd_myr', 'usd_myr_chg_4w',
                'momentum_5d_sign', 'event_indonesia', 'event_malaysia', 'event_china', 'event_india',
                'event_global', 'event_any_flag_active', 'event_net_sum']

# Shock masks
covid_mask = (weekly['week_end_date'] >= '2020-03-01') & (weekly['week_end_date'] <= '2020-06-30')
ban_mask = (weekly['week_end_date'] >= '2022-01-01') & (weekly['week_end_date'] <= '2022-10-31')
shock_mask = covid_mask | ban_mask
weekly_clean = weekly[~shock_mask].copy()
weekly_shock = weekly[shock_mask].copy()

# ============================================================
out_lines = []
def pr(s=''):
    print(s)
    out_lines.append(s)

pr("\n\n" + "=" * 80)
pr("SECTION 10: ENSO EPISODE-EXCLUSION CHECK")
pr("=" * 80)

# --- 1. Three-way comparison ---
pr("\n--- 10.1 Three-Way Comparison for enso_oni ---\n")

def tercile_screen(panel, var, target, shape_filter=None):
    sub = panel[panel['shape'] == shape_filter][[var, target]].dropna() if shape_filter else panel[[var, target]].dropna()
    if len(sub) < 10:
        return None
    q33, q67 = sub[var].quantile(0.333), sub[var].quantile(0.667)
    h, l = sub[var] >= q67, sub[var] <= q33
    if h.sum() < 3 or l.sum() < 3:
        return None
    hp, lp = sub.loc[h, target], sub.loc[l, target]
    return {'high_pct': hp.mean()*100, 'low_pct': lp.mean()*100, 'gap': (hp.mean()-lp.mean())*100,
            'n_high': int(h.sum()), 'n_low': int(l.sum()), 'n_high_persist': int(hp.sum()), 'n_low_persist': int(lp.sum())}

cuts = [
    ('enso_oni', '1', 'enso_oni / Shape 1 / 12w'),
    ('enso_oni', None, 'enso_oni / POOLED / 12w'),
]

pr(f"  {'Cut':<30} {'Gap (all)':>10} {'Gap (no shock)':>15} {'Gap (shock only)':>17}")
pr(f"  {'-'*75}")

for var, shape, label in cuts:
    r_all = tercile_screen(weekly, var, 'persists_12w', shape)
    r_filt = tercile_screen(weekly_clean, var, 'persists_12w', shape)
    r_shock = tercile_screen(weekly_shock, var, 'persists_12w', shape)

    g_all = f"{r_all['gap']:+.1f}" if r_all else 'N/A'
    g_filt = f"{r_filt['gap']:+.1f}" if r_filt else 'N/A'
    g_shock = f"{r_shock['gap']:+.1f}" if r_shock else 'N/A'
    pr(f"  {label:<30} {g_all:>10} {g_filt:>15} {g_shock:>17}")

# Print full detail for each panel
pr(f"\n  Detailed breakdown:")
for panel_name, panel in [('All weeks', weekly), ('Shocks excluded', weekly_clean), ('Shocks only', weekly_shock)]:
    pr(f"\n  --- {panel_name} ---")
    for var, shape, label in cuts:
        r = tercile_screen(panel, var, 'persists_12w', shape)
        if r:
            pr(f"    {label}: high={r['high_pct']:.1f}% ({r['n_high_persist']}/{r['n_high']}), "
               f"low={r['low_pct']:.1f}% ({r['n_low_persist']}/{r['n_low']}), gap={r['gap']:+.1f}pp")
        else:
            pr(f"    {label}: insufficient data for tercile split")

# Context: what was ENSO doing in those windows?
pr(f"\n  ENSO ONI values during shock windows:")
for label, mask in [('COVID (Mar-Jun 2020)', covid_mask), ('Export-ban (Jan-Oct 2022)', ban_mask)]:
    sub = weekly[mask]
    oni_vals = sub['enso_oni'].dropna()
    if len(oni_vals) > 0:
        pr(f"    {label}: ONI range {oni_vals.min():.2f} to {oni_vals.max():.2f}, "
           f"mean {oni_vals.mean():.2f}, n={len(oni_vals)} weeks")

# --- 2. RF importance on filtered panel ---
pr("\n\n--- 10.2 RF Importance: enso_oni on Filtered vs Original Panel ---\n")

def run_rf(panel, features, shape_filter=None):
    if shape_filter:
        sub = panel[panel['shape'] == shape_filter][features + ['persists_12w']].dropna()
    else:
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

rf_cuts = [
    (None, 'Pooled 12w'),
    ('1', 'Shape 1 12w'),
]

pr(f"  {'Cut':<20} {'Panel':<15} {'n':>5} {'enso_oni Imp':>13} {'enso_oni Rank':>14}")
pr(f"  {'-'*70}")

for shape, cut_label in rf_cuts:
    rf_orig, n_orig = run_rf(weekly, all_features, shape)
    rf_filt, n_filt = run_rf(weekly_clean, all_features, shape)

    o = rf_orig['enso_oni']
    f = rf_filt['enso_oni']
    pr(f"  {cut_label:<20} {'Original':<15} {n_orig:>5} {o['mean_imp']:>13.4f} {o['mean_rank']:>14.1f}")
    pr(f"  {'':<20} {'No shocks':<15} {n_filt:>5} {f['mean_imp']:>13.4f} {f['mean_rank']:>14.1f}")

    # Also show top 3 in filtered for context
    sorted_filt = sorted(rf_filt.items(), key=lambda x: -x[1]['mean_imp'])
    top3_str = ', '.join(f"{v} ({d['mean_rank']:.1f})" for v, d in sorted_filt[:3])
    pr(f"    Filtered top 3: {top3_str}")
    pr("")

# Append
with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
    for line in out_lines:
        f.write(line + '\n')

print(f"\nAppended SECTION 10 to: {OUTPUT_PATH}")
