"""
Phase B Follow-Up #3: Shape 0.0 Deep-Dive — Features + Model Diagnosis
Appends SECTION 10 to research/outputs/Phase_B_Persistence_Model.txt
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
RAW = os.path.join(BASE, 'Raw Data')
OUTPUT_PATH = os.path.join(BASE, 'research', 'outputs', 'Phase_B_Persistence_Model.txt')

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

file_buf = open(OUTPUT_PATH, 'a', encoding='utf-8')
tee = Tee(sys.stdout, file_buf)
_print = print
def print(*args, **kwargs):
    kwargs['file'] = tee
    _print(*args, **kwargs)

HORIZONS = list(range(1, 9))
ORIG_FEATURES = ['usd_myr', 'enso_oni']
EXPANDED_FEATURES = ['usd_myr', 'enso_oni', 'crude_oil_price', 'palm_soy_spread']

# Expected directions from Phase A
EXPECTED = {
    'usd_myr': 'negative',
    'enso_oni': 'positive',
    'crude_oil_price': 'negative',   # moderate -20.8pp for Shape 0.0
    'palm_soy_spread': 'positive',   # moderate +17.8pp for Shape 0.0
}

# ============================================================
# Rebuild weekly panel (same as prior scripts)
# ============================================================
shape_log = pd.read_csv(os.path.join(RAW, 'Research', 'daily_shape_log.csv'),
                         dtype={'shape': str}, parse_dates=['date'])

stock_df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'FCPO Stock 3Y.xlsx'))
stock_df.columns = ['date', 'stock']
stock_df = stock_df.iloc[1:]
stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
stock_df = stock_df.dropna(subset=['date'])
stock_df['stock'] = pd.to_numeric(stock_df['stock'], errors='coerce')
stock_df = stock_df.dropna(subset=['stock']).sort_values('date').reset_index(drop=True)

prod_df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'MPOB Production 3Y.xlsx'))
prod_df.columns = ['date', 'production']
prod_df = prod_df.iloc[1:]
prod_df['date'] = pd.to_datetime(prod_df['date'], errors='coerce')
prod_df = prod_df.dropna(subset=['date'])
prod_df['production'] = pd.to_numeric(prod_df['production'], errors='coerce')
prod_df = prod_df.dropna(subset=['production']).sort_values('date').reset_index(drop=True)

fx_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'FX_IDC_USDMYR, 1D_1227e.csv'))
fx_df['date'] = pd.to_datetime(fx_df['time'], unit='s')
fx_df = fx_df[['date', 'close']].rename(columns={'close': 'usd_myr'}).sort_values('date')

ps_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'MYX_DLY_FCPO1!_2_CBOT_DL_ZL1!, 1D_61912.csv'))
ps_df['date'] = pd.to_datetime(ps_df['time'], unit='s')
ps_df = ps_df[['date', 'close']].rename(columns={'close': 'palm_soy_spread'}).sort_values('date')

cl_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'NYMEX_DL_CL1!, 1D_84001.csv'))
cl_df['date'] = pd.to_datetime(cl_df['time'], unit='s')
cl_df = cl_df[['date', 'close']].rename(columns={'close': 'crude_oil_price'}).sort_values('date')

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

daily = shape_log[['date', 'shape', 'M1']].copy().rename(columns={'M1': 'spot'})
daily = daily.set_index('date').sort_index()
daily = daily['2017-01-01':]
daily['iso_year'] = daily.index.isocalendar().year.values
daily['iso_week'] = daily.index.isocalendar().week.values
daily['week_key'] = daily['iso_year'].astype(str) + '-W' + daily['iso_week'].astype(str).str.zfill(2)

weekly = daily.groupby('week_key').agg(
    shape=('shape', 'last'), spot=('spot', 'last'),
    week_end_date=('spot', lambda x: x.index[-1])
).sort_values('week_end_date')

fx_daily = fx_df.set_index('date')['usd_myr'].sort_index()
daily['usd_myr'] = fx_daily.reindex(daily.index, method='ffill')
weekly['usd_myr'] = daily.groupby('week_key')['usd_myr'].mean().reindex(weekly.index)

ps_daily = ps_df.set_index('date')['palm_soy_spread'].sort_index()
daily['palm_soy'] = ps_daily.reindex(daily.index, method='ffill')
weekly['palm_soy_spread'] = daily.groupby('week_key')['palm_soy'].mean().reindex(weekly.index)

cl_daily = cl_df.set_index('date')['crude_oil_price'].sort_index()
daily['crude'] = cl_daily.reindex(daily.index, method='ffill')
weekly['crude_oil_price'] = daily.groupby('week_key')['crude'].mean().reindex(weekly.index)

mpob = stock_df.set_index('date')[['stock']].join(
    prod_df.set_index('date')[['production']], how='outer'
).sort_index()
mpob['stock_to_usage_ratio'] = mpob['stock'] / mpob['production']

oni_series = oni_df['oni']
vals = []
for _, row in weekly.iterrows():
    dt = row['week_end_date']
    mask = oni_series.index <= dt
    vals.append(oni_series[mask].iloc[-1] if mask.any() else np.nan)
weekly['enso_oni'] = vals

for N in HORIZONS:
    weekly[f'shape_plus_{N}w'] = weekly['shape'].shift(-N)
    weekly[f'persists_{N}w'] = (weekly['shape'] == weekly[f'shape_plus_{N}w']).astype(int)
    weekly.loc[weekly[f'shape_plus_{N}w'].isna(), f'persists_{N}w'] = np.nan

weekly = weekly[weekly['week_end_date'] >= '2017-01-01']

train_cutoff = pd.Timestamp('2024-12-31')
train = weekly[weekly['week_end_date'] <= train_cutoff].copy()
test = weekly[weekly['week_end_date'] > train_cutoff].copy()

# Filter to Shape 0.0 only
train_00 = train[train['shape'] == '0.0'].copy()
test_00 = test[test['shape'] == '0.0'].copy()

# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 10: SHAPE 0.0 (CONTANGO) DEEP-DIVE — FEATURES + MODEL DIAGNOSIS")
print("=" * 80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Shape 0.0 training weeks: {len(train_00)}, test weeks: {len(test_00)}")

# ============================================================
# 10a: Within-shape correlation check
# ============================================================
print(f"\n\n{'='*80}")
print(f"10a. WITHIN-SHAPE CORRELATION CHECK")
print(f"{'='*80}")

corr_pooled = train[['usd_myr', 'enso_oni']].dropna().corr().iloc[0, 1]
corr_shape00 = train_00[['usd_myr', 'enso_oni']].dropna().corr().iloc[0, 1]

print(f"\n  Pooled correlation (all shapes, training set): {corr_pooled:.3f}")
print(f"  Within-Shape-0.0 correlation (Contango weeks only): {corr_shape00:.3f}")
diff = abs(corr_shape00) - abs(corr_pooled)
print(f"  Difference in |r|: {diff:+.3f}")

if abs(corr_shape00) > 0.5:
    print(f"\n  FINDING: Within-shape correlation is HIGH ({corr_shape00:.3f}). The two features")
    print(f"  are substantially entangled during Contango weeks. This directly explains why LR")
    print(f"  produces unstable/flipped coefficients — the model cannot reliably separate their")
    print(f"  individual contributions when they move together this closely.")
elif abs(corr_shape00) > abs(corr_pooled) + 0.1:
    print(f"\n  FINDING: Within-shape correlation ({corr_shape00:.3f}) is meaningfully higher than")
    print(f"  pooled ({corr_pooled:.3f}). Some multicollinearity contribution to coefficient")
    print(f"  instability, but not extreme.")
else:
    print(f"\n  FINDING: Within-shape correlation ({corr_shape00:.3f}) is similar to pooled")
    print(f"  ({corr_pooled:.3f}). Multicollinearity is NOT the primary explanation for")
    print(f"  coefficient direction problems.")

# Also check expanded features
corr_expanded = train_00[EXPANDED_FEATURES].dropna().corr()
print(f"\n  Expanded feature correlation matrix (Shape 0.0 training weeks only):")
print(f"  {corr_expanded.round(3).to_string()}")

# ============================================================
# 10b: Single-feature models
# ============================================================
print(f"\n\n{'='*80}")
print(f"10b. SINGLE-FEATURE MODELS (isolating each variable)")
print(f"{'='*80}")
print(f"\n  If a feature holds its correct direction alone but breaks when combined,")
print(f"  that confirms a multicollinearity/interaction problem.\n")

print(f"  {'Hz':>3}  {'usd_myr alone':>15} {'dir':>5} {'match':>6} {'acc%':>6}  "
      f"{'enso_oni alone':>15} {'dir':>5} {'match':>6} {'acc%':>6}")
print(f"  {'-'*80}")

usd_alone_dirs = {}
oni_alone_dirs = {}

for N in HORIZONS:
    target = f'persists_{N}w'
    sub = train_00[['usd_myr', 'enso_oni', target]].dropna()
    X_usd = sub[['usd_myr']].values
    X_oni = sub[['enso_oni']].values
    y = sub[target].values.astype(int)
    base = y.mean() * 100

    # usd_myr alone
    sc1 = StandardScaler()
    X1 = sc1.fit_transform(X_usd)
    lr1 = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr1.fit(X1, y)
    c1 = lr1.coef_[0][0]
    d1 = 'neg' if c1 < 0 else 'pos'
    m1 = 'YES' if d1 == 'neg' else 'NO'
    a1 = accuracy_score(y, lr1.predict(X1)) * 100
    usd_alone_dirs[N] = d1

    # enso_oni alone
    sc2 = StandardScaler()
    X2 = sc2.fit_transform(X_oni)
    lr2 = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr2.fit(X2, y)
    c2 = lr2.coef_[0][0]
    d2 = 'neg' if c2 < 0 else 'pos'
    m2 = 'YES' if d2 == 'pos' else 'NO'
    a2 = accuracy_score(y, lr2.predict(X2)) * 100
    oni_alone_dirs[N] = d2

    print(f"  {N:>3}w  coeff={c1:>+8.4f} {d1:>5} {m1:>6} {a1:>5.1f}%  "
          f"coeff={c2:>+8.4f} {d2:>5} {m2:>6} {a2:>5.1f}%")

# Compare to combined
print(f"\n  Comparison: direction when ALONE vs when COMBINED (from Section 9):")
print(f"  {'Hz':>3}  {'usd alone':>10} {'usd combined':>13} {'changed?':>9}  "
      f"{'oni alone':>10} {'oni combined':>13} {'changed?':>9}")
print(f"  {'-'*70}")

# Refit combined to get directions
for N in HORIZONS:
    target = f'persists_{N}w'
    sub = train_00[ORIG_FEATURES + [target]].dropna()
    X = sub[ORIG_FEATURES].values
    y = sub[target].values.astype(int)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(Xs, y)
    c_usd_comb = 'neg' if lr.coef_[0][0] < 0 else 'pos'
    c_oni_comb = 'neg' if lr.coef_[0][1] < 0 else 'pos'

    usd_chg = 'FLIPPED' if usd_alone_dirs[N] != c_usd_comb else 'same'
    oni_chg = 'FLIPPED' if oni_alone_dirs[N] != c_oni_comb else 'same'

    print(f"  {N:>3}w  {usd_alone_dirs[N]:>10} {c_usd_comb:>13} {usd_chg:>9}  "
          f"{oni_alone_dirs[N]:>10} {c_oni_comb:>13} {oni_chg:>9}")

# Summary
usd_alone_correct = sum(1 for d in usd_alone_dirs.values() if d == 'neg')
oni_alone_correct = sum(1 for d in oni_alone_dirs.values() if d == 'pos')
print(f"\n  usd_myr correct direction alone: {usd_alone_correct}/8 horizons")
print(f"  enso_oni correct direction alone: {oni_alone_correct}/8 horizons")

# ============================================================
# 10c: Expanded 4-feature models
# ============================================================
print(f"\n\n{'='*80}")
print(f"10c. EXPANDED 4-FEATURE MODELS — TRAINING SET")
print(f"{'='*80}")
print(f"  Features: {EXPANDED_FEATURES}")
print(f"  Adding crude_oil_price (Phase A: -20.8pp moderate) and palm_soy_spread (+17.8pp moderate)\n")

feat_hdr = '  '.join(f'{f[:8]:>10}' for f in EXPANDED_FEATURES)
print(f"  {'Hz':>3} {'Model':>5} {'Acc%':>6} {'Base%':>6} {'Lift':>7}  {feat_hdr}  {'Dirs':>10}")
print(f"  {'-'*(55 + 12*len(EXPANDED_FEATURES))}")

fitted_exp = {}

for N in HORIZONS:
    target = f'persists_{N}w'
    sub = train_00[EXPANDED_FEATURES + [target]].dropna()
    X = sub[EXPANDED_FEATURES].values
    y = sub[target].values.astype(int)
    base = y.mean() * 100

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # LR-W expanded
    lrw = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lrw.fit(Xs, y)
    acc_lrw = accuracy_score(y, lrw.predict(Xs)) * 100
    coeffs = lrw.coef_[0]
    dirs = ['pos' if c > 0 else 'neg' for c in coeffs]
    matches = [dirs[i][:3] == EXPECTED[f][:3] for i, f in enumerate(EXPANDED_FEATURES)]
    match_ct = sum(matches)
    match_str = f"{match_ct}/4"
    coeff_str = '  '.join(f'{c:>+10.4f}' for c in coeffs)
    print(f"  {N:>3}w {'LR-W':>5} {acc_lrw:>5.1f}% {base:>5.1f}% {acc_lrw-base:>+6.1f}pp  {coeff_str}  {match_str:>10}")

    # RF expanded
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    acc_rf = accuracy_score(y, rf.predict(X)) * 100

    n_resamples = 20
    rf_oob = []
    rf_imps_all = np.zeros((n_resamples, len(EXPANDED_FEATURES)))
    for b in range(n_resamples):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        oob = np.array([i for i in range(len(X)) if i not in idx])
        rf_b = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=b, n_jobs=-1)
        rf_b.fit(X[idx], y[idx])
        rf_imps_all[b] = rf_b.feature_importances_
        if len(oob) > 0:
            rf_oob.append(accuracy_score(y[oob], rf_b.predict(X[oob])) * 100)

    imp_str = '  '.join(f'{v:>10.3f}' for v in rf.feature_importances_)
    oob_str = f"OOB={np.mean(rf_oob):.1f}±{np.std(rf_oob):.1f}%" if rf_oob else ""
    print(f"  {N:>3}w {'RF':>5} {acc_rf:>5.1f}% {base:>5.1f}% {acc_rf-base:>+6.1f}pp  {imp_str}  {oob_str:>10}")

    fitted_exp[N] = {'lrw': lrw, 'rf': rf, 'scaler': sc, 'base_rate': base}

# --- Test set ---
print(f"\n{'='*80}")
print(f"10c (cont). EXPANDED 4-FEATURE — TEST SET RESULTS")
print(f"{'='*80}")

print(f"\n  {'Hz':>3} {'Model':>5} {'Acc%':>6} {'Prec1':>6} {'Rec1':>6} {'Prec0':>6} {'Rec0':>6} "
      f"{'Base%':>6} {'BaseAcc':>8} {'vs Base':>8} {'Note':<14}")
print(f"  {'-'*100}")

exp_test_results = []

for N in HORIZONS:
    if N not in fitted_exp:
        continue
    target = f'persists_{N}w'
    sub_te = test_00[EXPANDED_FEATURES + [target]].dropna()
    if len(sub_te) == 0:
        print(f"  {N:>3}w   — NO TEST DATA —")
        continue

    X_te = sub_te[EXPANDED_FEATURES].values
    y_te = sub_te[target].values.astype(int)
    te_base = y_te.mean() * 100
    n_te = len(y_te)

    tr_base = fitted_exp[N]['base_rate']
    baseline_pred = 1 if tr_base >= 50 else 0
    baseline_acc = accuracy_score(y_te, [baseline_pred] * n_te) * 100

    for mname, mkey in [('LR-W', 'lrw'), ('RF', 'rf')]:
        model = fitted_exp[N][mkey]
        if mname == 'LR-W':
            X_eval = fitted_exp[N]['scaler'].transform(X_te)
        else:
            X_eval = X_te

        y_pred = model.predict(X_eval)
        acc = accuracy_score(y_te, y_pred) * 100
        is_degen = len(np.unique(y_pred)) == 1

        if not is_degen:
            p1 = precision_score(y_te, y_pred, pos_label=1, zero_division=0) * 100
            r1 = recall_score(y_te, y_pred, pos_label=1, zero_division=0) * 100
            p0 = precision_score(y_te, y_pred, pos_label=0, zero_division=0) * 100
            r0 = recall_score(y_te, y_pred, pos_label=0, zero_division=0) * 100
        else:
            p1 = r1 = p0 = r0 = 0.0

        vs = acc - baseline_acc
        note = f"[all-{y_pred[0]:.0f}]" if is_degen else ""
        print(f"  {N:>3}w {mname:>5} {acc:>5.1f}% {p1:>5.1f}% {r1:>5.1f}% {p0:>5.1f}% {r0:>5.1f}% "
              f"{te_base:>5.1f}% {baseline_acc:>7.1f}% {vs:>+7.1f}pp {note:<14}")

        exp_test_results.append({
            'horizon': N, 'model': mname, 'acc': acc, 'baseline_acc': baseline_acc,
            'vs_base': vs, 'degenerate': is_degen, 'n_test': n_te, 'test_base': te_base,
            'feature_set': '4-feat'
        })

# Direction stability for expanded set
print(f"\n  Expanded-set coefficient direction stability:")
for feat in EXPANDED_FEATURES:
    exp_dir = EXPECTED[feat]
    print(f"\n  {feat} (expected: {exp_dir}):")
    print(f"  {'Hz':>3}  {'2-feat LR-W':>12} {'4-feat LR-W':>12} {'Test gap':>10}")
    print(f"  {'-'*45}")
    for N in HORIZONS:
        if N not in fitted_exp:
            continue
        target = f'persists_{N}w'
        # 2-feature direction
        sub2 = train_00[ORIG_FEATURES + [target]].dropna()
        sc2 = StandardScaler()
        Xs2 = sc2.fit_transform(sub2[ORIG_FEATURES].values)
        lr2 = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr2.fit(Xs2, sub2[target].values.astype(int))
        if feat in ORIG_FEATURES:
            fi2 = ORIG_FEATURES.index(feat)
            d2 = 'neg' if lr2.coef_[0][fi2] < 0 else 'pos'
        else:
            d2 = 'N/A'

        # 4-feature direction
        fi4 = EXPANDED_FEATURES.index(feat)
        c4 = fitted_exp[N]['lrw'].coef_[0][fi4]
        d4 = 'neg' if c4 < 0 else 'pos'

        # Test gap
        te_sub = test_00[[feat, target]].dropna()
        if len(te_sub) >= 4:
            med = te_sub[feat].median()
            hi = te_sub[te_sub[feat] > med][target].mean() * 100
            lo = te_sub[te_sub[feat] <= med][target].mean() * 100
            gap = hi - lo
            gap_str = f"{gap:+.1f}pp"
        else:
            gap_str = "insuf."

        print(f"  {N:>3}w  {d2:>12} {d4:>12} {gap_str:>10}")

# ============================================================
# 10d: LR vs RF coherence comparison
# ============================================================
print(f"\n\n{'='*80}")
print(f"10d. LR vs RF COHERENCE COMPARISON")
print(f"{'='*80}")
print(f"\n  Question: Does RF's feature-importance behavior tell a more coherent story")
print(f"  than LR's flipped coefficients?\n")

# Refit 2-feature models to compare alongside 4-feature
print(f"  2-FEATURE SET: LR-W vs RF, test-set comparison")
print(f"  {'Hz':>3}  {'LR-W acc':>9} {'vs base':>8} {'LR dirs':>8}  "
      f"{'RF acc':>7} {'vs base':>8} {'RF degen?':>10}  {'RF imp usd':>10} {'RF imp oni':>10}")
print(f"  {'-'*90}")

for N in HORIZONS:
    target = f'persists_{N}w'
    sub_tr = train_00[ORIG_FEATURES + [target]].dropna()
    sub_te = test_00[ORIG_FEATURES + [target]].dropna()
    if len(sub_tr) < 10 or len(sub_te) == 0:
        continue

    X_tr = sub_tr[ORIG_FEATURES].values
    y_tr = sub_tr[target].values.astype(int)
    X_te = sub_te[ORIG_FEATURES].values
    y_te = sub_te[target].values.astype(int)
    base = y_tr.mean() * 100
    bl_pred = 1 if base >= 50 else 0
    bl_acc = accuracy_score(y_te, [bl_pred] * len(y_te)) * 100

    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_te = sc.transform(X_te)

    lrw = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lrw.fit(Xs_tr, y_tr)
    y_lrw = lrw.predict(Xs_te)
    acc_lrw = accuracy_score(y_te, y_lrw) * 100
    vs_lrw = acc_lrw - bl_acc
    d_usd = 'neg' if lrw.coef_[0][0] < 0 else 'pos'
    d_oni = 'neg' if lrw.coef_[0][1] < 0 else 'pos'
    lr_dirs = f"{d_usd}/{d_oni}"

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_rf = rf.predict(X_te)
    acc_rf = accuracy_score(y_te, y_rf) * 100
    vs_rf = acc_rf - bl_acc
    rf_degen = "YES" if len(np.unique(y_rf)) == 1 else "no"

    print(f"  {N:>3}w  {acc_lrw:>8.1f}% {vs_lrw:>+7.1f}pp {lr_dirs:>8}  "
          f"{acc_rf:>6.1f}% {vs_rf:>+7.1f}pp {rf_degen:>10}  "
          f"{rf.feature_importances_[0]:>10.3f} {rf.feature_importances_[1]:>10.3f}")

print(f"\n  4-FEATURE SET: LR-W vs RF, test-set comparison")
print(f"  {'Hz':>3}  {'LR-W acc':>9} {'vs base':>8} {'dirs ok':>8}  "
      f"{'RF acc':>7} {'vs base':>8} {'RF degen?':>10}  RF importances (4 feats)")
print(f"  {'-'*100}")

for N in HORIZONS:
    if N not in fitted_exp:
        continue
    target = f'persists_{N}w'
    sub_te = test_00[EXPANDED_FEATURES + [target]].dropna()
    if len(sub_te) == 0:
        continue

    X_te = sub_te[EXPANDED_FEATURES].values
    y_te = sub_te[target].values.astype(int)
    bl_pred = 1 if fitted_exp[N]['base_rate'] >= 50 else 0
    bl_acc = accuracy_score(y_te, [bl_pred] * len(y_te)) * 100

    # LR-W
    y_lrw = fitted_exp[N]['lrw'].predict(fitted_exp[N]['scaler'].transform(X_te))
    acc_lrw = accuracy_score(y_te, y_lrw) * 100
    vs_lrw = acc_lrw - bl_acc
    coeffs = fitted_exp[N]['lrw'].coef_[0]
    dirs_ok = sum(1 for i, f in enumerate(EXPANDED_FEATURES)
                  if ('neg' if coeffs[i] < 0 else 'pos') == EXPECTED[f][:3])

    # RF
    y_rf = fitted_exp[N]['rf'].predict(X_te)
    acc_rf = accuracy_score(y_te, y_rf) * 100
    vs_rf = acc_rf - bl_acc
    rf_degen = "YES" if len(np.unique(y_rf)) == 1 else "no"
    imp_str = '  '.join(f'{v:.3f}' for v in fitted_exp[N]['rf'].feature_importances_)

    print(f"  {N:>3}w  {acc_lrw:>8.1f}% {vs_lrw:>+7.1f}pp {dirs_ok:>5}/4   "
          f"{acc_rf:>6.1f}% {vs_rf:>+7.1f}pp {rf_degen:>10}  {imp_str}")

# RF importance stability across horizons
print(f"\n  RF IMPORTANCE STABILITY (4-feature, across horizons):")
print(f"  Does RF concentrate importance on specific features or spread evenly?\n")
imp_matrix = []
for N in HORIZONS:
    if N in fitted_exp:
        imp_matrix.append(fitted_exp[N]['rf'].feature_importances_)
imp_matrix = np.array(imp_matrix)
print(f"  {'Feature':<20} {'Mean imp':>9} {'Std imp':>9} {'Min':>9} {'Max':>9} {'CV':>9}")
print(f"  {'-'*65}")
for i, f in enumerate(EXPANDED_FEATURES):
    mn = imp_matrix[:, i].mean()
    sd = imp_matrix[:, i].std()
    cv = sd / mn if mn > 0 else 0
    print(f"  {f:<20} {mn:>9.3f} {sd:>9.3f} {imp_matrix[:,i].min():>9.3f} {imp_matrix[:,i].max():>9.3f} {cv:>9.2f}")

# ============================================================
# 10e: Verdict
# ============================================================
print(f"\n\n{'='*80}")
print(f"10e. VERDICT — SHAPE 0.0 (CONTANGO) DIAGNOSIS")
print(f"{'='*80}")

# Gather evidence
print(f"\n  Evidence summary:")
print(f"  (a) Missing features?")
exp_beats = [r for r in exp_test_results if r['vs_base'] > 0 and not r['degenerate']]
if exp_beats:
    best_exp = max(exp_beats, key=lambda r: r['vs_base'])
    print(f"      Adding crude_oil + palm_soy: best test lift = {best_exp['model']} at "
          f"{best_exp['horizon']}w (+{best_exp['vs_base']:.1f}pp)")
else:
    print(f"      Adding crude_oil + palm_soy: no model beats baseline on test set")

# Check expanded direction correctness at beating horizons
exp_beats_correct = []
for r in exp_beats:
    N = r['horizon']
    coeffs = fitted_exp[N]['lrw'].coef_[0]
    ok = sum(1 for i, f in enumerate(EXPANDED_FEATURES)
             if ('neg' if coeffs[i] < 0 else 'pos') == EXPECTED[f][:3])
    exp_beats_correct.append((N, ok))
    print(f"      Horizon {N}w: {ok}/4 directions correct")

print(f"\n  (b) Multicollinearity within Shape 0.0?")
print(f"      Within-shape r(usd_myr, enso_oni) = {corr_shape00:.3f} vs pooled {corr_pooled:.3f}")
if abs(corr_shape00) > abs(corr_pooled) + 0.1:
    print(f"      Elevated within-shape correlation contributes to instability")
else:
    print(f"      Not materially different from pooled — NOT the primary explanation")

print(f"\n  (c) LR linear-assumption mismatch?")
print(f"      usd_myr correct direction ALONE: {usd_alone_correct}/8")
print(f"      enso_oni correct direction ALONE: {oni_alone_correct}/8")
if usd_alone_correct <= 4 and oni_alone_correct <= 4:
    print(f"      Both features show wrong direction even in isolation — problem is NOT just")
    print(f"      multicollinearity or LR combining them poorly. The univariate relationship")
    print(f"      itself does not hold consistently for Shape 0.0 across horizons.")
elif usd_alone_correct >= 6 and oni_alone_correct >= 6:
    print(f"      Both features hold correct direction alone but flip when combined —")
    print(f"      classic multicollinearity / interaction artifact in LR.")
else:
    print(f"      Mixed: one feature holds alone, the other doesn't. Partial interaction effect.")

# RF coherence
rf_degen_count_2feat = 0
rf_beats_2feat = 0
rf_degen_count_4feat = 0
rf_beats_4feat = 0
for N in HORIZONS:
    target = f'persists_{N}w'
    sub_te = test_00[ORIG_FEATURES + [target]].dropna()
    if len(sub_te) == 0:
        continue
    X_te = sub_te[ORIG_FEATURES].values
    y_te = sub_te[target].values.astype(int)
    base = train_00[target].dropna().mean() * 100
    bl_pred = 1 if base >= 50 else 0
    bl_acc = accuracy_score(y_te, [bl_pred] * len(y_te)) * 100

    rf_tmp = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    rf_tmp.fit(train_00[ORIG_FEATURES + [target]].dropna()[ORIG_FEATURES].values,
               train_00[ORIG_FEATURES + [target]].dropna()[target].values.astype(int))
    y_rf = rf_tmp.predict(X_te)
    if len(np.unique(y_rf)) == 1:
        rf_degen_count_2feat += 1
    elif accuracy_score(y_te, y_rf) * 100 > bl_acc:
        rf_beats_2feat += 1

for r in exp_test_results:
    if r['model'] == 'RF':
        if r['degenerate']:
            rf_degen_count_4feat += 1
        elif r['vs_base'] > 0:
            rf_beats_4feat += 1

print(f"\n  (d) RF performance (model-assumption check):")
print(f"      2-feature RF: degenerate {rf_degen_count_2feat}/8, beats baseline {rf_beats_2feat}/8")
print(f"      4-feature RF: degenerate {rf_degen_count_4feat}/8, beats baseline {rf_beats_4feat}/8")
if rf_beats_2feat == 0 and rf_beats_4feat == 0:
    print(f"      RF also fails — problem is NOT just LR's linear assumption.")
elif rf_beats_4feat > rf_beats_2feat:
    print(f"      RF improves with more features — some of the problem was missing features.")

# Final verdict
print(f"\n  {'='*60}")
print(f"  FINAL VERDICT:")

# Check: does ANY configuration beat baseline with correct directions?
clean_wins = []
for r in exp_test_results:
    if r['vs_base'] > 0 and not r['degenerate'] and r['model'] == 'LR-W':
        N = r['horizon']
        coeffs = fitted_exp[N]['lrw'].coef_[0]
        all_correct = all(
            ('neg' if coeffs[i] < 0 else 'pos') == EXPECTED[f][:3]
            for i, f in enumerate(EXPANDED_FEATURES)
        )
        if all_correct:
            clean_wins.append(r)

# Also check single-feature models on test
print(f"\n  Single-feature test check (usd_myr alone, LR-W balanced):")
single_feat_wins = []
for N in HORIZONS:
    target = f'persists_{N}w'
    sub_tr = train_00[['usd_myr', target]].dropna()
    sub_te = test_00[['usd_myr', target]].dropna()
    if len(sub_te) == 0 or len(sub_tr) < 10:
        continue
    X_tr = sub_tr[['usd_myr']].values
    y_tr = sub_tr[target].values.astype(int)
    X_te = sub_te[['usd_myr']].values
    y_te = sub_te[target].values.astype(int)
    base = y_tr.mean() * 100
    bl_pred = 1 if base >= 50 else 0
    bl_acc = accuracy_score(y_te, [bl_pred] * len(y_te)) * 100

    sc = StandardScaler()
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(sc.fit_transform(X_tr), y_tr)
    y_pred = lr.predict(sc.transform(X_te))
    acc = accuracy_score(y_te, y_pred) * 100
    vs = acc - bl_acc
    d = 'neg' if lr.coef_[0][0] < 0 else 'pos'
    degen = len(np.unique(y_pred)) == 1
    tag = f"[all-{y_pred[0]:.0f}]" if degen else ""
    print(f"    {N}w: acc={acc:.1f}%, bl={bl_acc:.1f}%, vs={vs:+.1f}pp, dir={d}, {tag}")
    if vs > 0 and not degen and d == 'neg':
        single_feat_wins.append({'horizon': N, 'acc': acc, 'vs_base': vs})

print(f"\n  Single-feature test check (enso_oni alone, LR-W balanced):")
for N in HORIZONS:
    target = f'persists_{N}w'
    sub_tr = train_00[['enso_oni', target]].dropna()
    sub_te = test_00[['enso_oni', target]].dropna()
    if len(sub_te) == 0 or len(sub_tr) < 10:
        continue
    X_tr = sub_tr[['enso_oni']].values
    y_tr = sub_tr[target].values.astype(int)
    X_te = sub_te[['enso_oni']].values
    y_te = sub_te[target].values.astype(int)
    base = y_tr.mean() * 100
    bl_pred = 1 if base >= 50 else 0
    bl_acc = accuracy_score(y_te, [bl_pred] * len(y_te)) * 100

    sc = StandardScaler()
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(sc.fit_transform(X_tr), y_tr)
    y_pred = lr.predict(sc.transform(X_te))
    acc = accuracy_score(y_te, y_pred) * 100
    vs = acc - bl_acc
    d = 'neg' if lr.coef_[0][0] < 0 else 'pos'
    degen = len(np.unique(y_pred)) == 1
    tag = f"[all-{y_pred[0]:.0f}]" if degen else ""
    print(f"    {N}w: acc={acc:.1f}%, bl={bl_acc:.1f}%, vs={vs:+.1f}pp, dir={d}, {tag}")

if clean_wins:
    best_cw = max(clean_wins, key=lambda r: r['vs_base'])
    print(f"\n  CANDIDATE RESULT: 4-feature LR-W at {best_cw['horizon']}w — {best_cw['acc']:.1f}% accuracy")
    print(f"  vs {best_cw['baseline_acc']:.1f}% baseline (+{best_cw['vs_base']:.1f}pp), all directions correct.")
    print(f"  CAVEAT: n={best_cw['n_test']} test observations. Treat as tentative, not confirmed.")
elif single_feat_wins:
    best_sf = max(single_feat_wins, key=lambda r: r['vs_base'])
    print(f"\n  PARTIAL RESULT: Single-feature (usd_myr) LR-W at {best_sf['horizon']}w — "
          f"{best_sf['acc']:.1f}% vs baseline (+{best_sf['vs_base']:.1f}pp), correct direction.")
    print(f"  Multi-feature models do not produce a clean result.")
else:
    print(f"\n  Shape 0.0 (Contango) has no trustworthy persistence model with current features")
    print(f"  and methods. No configuration (2-feature, 4-feature, single-feature, LR, LR-W, RF)")
    print(f"  beats baseline out-of-sample with correct feature directions. The Phase A")
    print(f"  screening signals for this shape do not translate into a working predictive model.")
    print(f"\n  Root cause: the features' univariate relationships with Contango persistence")
    print(f"  are weak and/or non-monotonic within this shape's observation range — both")
    print(f"  usd_myr and enso_oni show inconsistent direction even in single-feature models")
    print(f"  ({usd_alone_correct}/8 and {oni_alone_correct}/8 correct respectively),")
    print(f"  confirming this is a data/signal problem, not a modeling-technique problem.")

print(f"\n{'='*80}")
print(f"END OF SECTION 10")
print(f"{'='*80}")

file_buf.close()
