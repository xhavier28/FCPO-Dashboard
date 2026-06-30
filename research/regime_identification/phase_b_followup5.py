"""
Phase B Follow-Up #5: Refit with Independently-Confirmed Features Only
Appends SECTION 12 to research/outputs/Phase_B_Persistence_Model.txt

Shape 1 (Backwardation): drop stock_to_usage_ratio, keep usd_myr + enso_oni
Shape 2 (Flat): drop crude_oil_price + palm_soy_spread, keep enso_oni + production_yoy_pct
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

# NEW reduced feature sets
NEW_FEATURES = {
    '1': ['usd_myr', 'enso_oni'],
    '2': ['enso_oni', 'production_yoy_pct'],
}
# ORIGINAL feature sets (for comparison)
OLD_FEATURES = {
    '1': ['usd_myr', 'enso_oni', 'stock_to_usage_ratio'],
    '2': ['enso_oni', 'crude_oil_price', 'palm_soy_spread', 'production_yoy_pct'],
}
SHAPE_NAMES = {'1': 'Backwardation', '2': 'Flat'}

EXPECTED = {
    ('usd_myr', '1'): 'negative',
    ('enso_oni', '1'): 'positive',
    ('enso_oni', '2'): 'negative',
    ('production_yoy_pct', '2'): 'positive',
}

# ============================================================
# Rebuild weekly panel (same as all prior scripts)
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
mpob['production_yoy_pct'] = mpob['production'].pct_change(12) * 100

for col in ['stock_to_usage_ratio', 'production_yoy_pct']:
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

for N in HORIZONS:
    weekly[f'shape_plus_{N}w'] = weekly['shape'].shift(-N)
    weekly[f'persists_{N}w'] = (weekly['shape'] == weekly[f'shape_plus_{N}w']).astype(int)
    weekly.loc[weekly[f'shape_plus_{N}w'].isna(), f'persists_{N}w'] = np.nan

weekly = weekly[weekly['week_end_date'] >= '2017-01-01']

train_cutoff = pd.Timestamp('2024-12-31')
test_start = pd.Timestamp('2025-01-01')
train = weekly[weekly['week_end_date'] <= train_cutoff].copy()
test = weekly[weekly['week_end_date'] >= test_start].copy()

# ============================================================
# Helper: fit models and evaluate
# ============================================================
def fit_and_eval(shape, features, train_df, test_df, horizon, n_boot=20):
    """Fit LR-W and RF on train, evaluate on test. Returns dict of results."""
    target = f'persists_{horizon}w'

    tr = train_df[train_df['shape'] == shape][features + [target]].dropna()
    te = test_df[test_df['shape'] == shape][features + [target]].dropna()

    X_train = tr[features].values
    y_train = tr[target].values.astype(int)
    X_test = te[features].values if len(te) > 0 else np.array([]).reshape(0, len(features))
    y_test = te[target].values.astype(int) if len(te) > 0 else np.array([])

    sc = StandardScaler()
    Xs_train = sc.fit_transform(X_train)
    Xs_test = sc.transform(X_test) if len(X_test) > 0 else X_test

    base_rate = y_train.mean()
    base_acc_train = max(base_rate, 1 - base_rate)

    # LR-W (class-weighted)
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(Xs_train, y_train)
    lr_train_acc = accuracy_score(y_train, lr.predict(Xs_train))
    lr_coeffs = {f: lr.coef_[0][i] for i, f in enumerate(features)}
    lr_intercept = lr.intercept_[0]

    # RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3, min_samples_leaf=5)
    rf.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
    rf_importances = {f: rf.feature_importances_[i] for i, f in enumerate(features)}

    # RF bootstrap OOB
    oob_accs = []
    oob_imps = {f: [] for f in features}
    for b in range(n_boot):
        idx = np.random.RandomState(b).choice(len(X_train), len(X_train), replace=True)
        oob_idx = np.array(list(set(range(len(X_train))) - set(idx)))
        if len(oob_idx) < 5:
            continue
        rf_b = RandomForestClassifier(n_estimators=100, random_state=b, max_depth=3, min_samples_leaf=5)
        rf_b.fit(X_train[idx], y_train[idx])
        oob_accs.append(accuracy_score(y_train[oob_idx], rf_b.predict(X_train[oob_idx])))
        for i, f in enumerate(features):
            oob_imps[f].append(rf_b.feature_importances_[i])

    result = {
        'n_train': len(tr), 'n_test': len(te),
        'persist_train': int(y_train.sum()), 'break_train': int(len(y_train) - y_train.sum()),
        'base_rate': base_rate, 'base_acc_train': base_acc_train,
        'lr_train_acc': lr_train_acc, 'lr_coeffs': lr_coeffs, 'lr_intercept': lr_intercept,
        'rf_train_acc': rf_train_acc, 'rf_importances': rf_importances,
        'oob_mean': np.mean(oob_accs) if oob_accs else np.nan,
        'oob_std': np.std(oob_accs) if oob_accs else np.nan,
        'oob_imps': {f: (np.mean(v), np.std(v)) for f, v in oob_imps.items()},
    }

    # Test set evaluation
    if len(te) > 0:
        test_base_rate = y_test.mean()
        test_base_acc = max(test_base_rate, 1 - test_base_rate)

        for model_name, model, X_te in [('lr', lr, Xs_test), ('rf', rf, X_test)]:
            preds = model.predict(X_te)
            acc = accuracy_score(y_test, preds)

            # Check degenerate
            unique_preds = np.unique(preds)
            if len(unique_preds) == 1:
                degen = f'[all-{int(unique_preds[0])}]'
            else:
                degen = ''

            prec, rec, _, _ = precision_recall_fscore_support(y_test, preds, labels=[0, 1], zero_division=0)

            result[f'{model_name}_test_acc'] = acc
            result[f'{model_name}_test_prec1'] = prec[1]
            result[f'{model_name}_test_rec1'] = rec[1]
            result[f'{model_name}_test_prec0'] = prec[0]
            result[f'{model_name}_test_rec0'] = rec[0]
            result[f'{model_name}_test_degen'] = degen
            result[f'{model_name}_test_base_rate'] = test_base_rate
            result[f'{model_name}_test_base_acc'] = test_base_acc
            result[f'{model_name}_test_vs_base'] = acc - test_base_acc
            result[f'{model_name}_test_n'] = len(y_test)
            result[f'{model_name}_test_persist'] = int(y_test.sum())
            result[f'{model_name}_test_break'] = int(len(y_test) - y_test.sum())

    return result

# ============================================================
# Direction check on test set (tercile gap method)
# ============================================================
def direction_gap(shape, feature, horizon, train_df, test_df):
    """Compute persistence gap between high/low terciles on test set."""
    target = f'persists_{horizon}w'
    tr = train_df[train_df['shape'] == shape][[feature, target]].dropna()
    te = test_df[test_df['shape'] == shape][[feature, target]].dropna()

    if len(te) < 4:
        return None, None, None

    # Use train median to split test
    med = tr[feature].median()
    hi = te[te[feature] >= med]
    lo = te[te[feature] < med]

    if len(hi) < 2 or len(lo) < 2:
        return None, None, None

    gap = hi[target].mean() - lo[target].mean()
    return gap, len(hi), len(lo)

# ============================================================
# SECTION 12
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 12: REFIT WITH INDEPENDENTLY-CONFIRMED FEATURES")
print("=" * 80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print()
print("NOTE: This section's conclusions SUPERSEDE Section 8's Shape 1 conclusion")
print("and Section 9's Shape 2 conclusion. When referencing 'the' Shape 1 or Shape 2")
print("model going forward, use the results from this section.")
print()
print("Rationale: Section 11 found that some features in the original models only")
print("held their expected direction in the COMBINED model due to correlation with")
print("stronger partners, not because they carry independent signal:")
print("  Shape 1: stock_to_usage_ratio dropped (1/8 correct alone, r=-0.515 with usd_myr)")
print("  Shape 2: crude_oil_price + palm_soy_spread dropped (0/8 each, r=-0.729 together)")
print()
print("Retained features (independently confirmed):")
print("  Shape 1 (Backwardation): usd_myr (8/8 alone), enso_oni (8/8 alone)")
print("  Shape 2 (Flat):          enso_oni (6/8 alone), production_yoy_pct (8/8 alone)")

# ============================================================
# SECTION 1: Shape 1 (Backwardation) refit
# ============================================================
for shape in ['1', '2']:
    sname = SHAPE_NAMES[shape]
    new_feats = NEW_FEATURES[shape]
    old_feats = OLD_FEATURES[shape]
    dropped = [f for f in old_feats if f not in new_feats]

    print(f"\n\n{'='*80}")
    print(f"12{'a' if shape == '1' else 'b'}. SHAPE {shape} ({sname.upper()}) — REFIT WITH {len(new_feats)} FEATURES")
    print(f"{'='*80}")
    print(f"Features: {new_feats}")
    print(f"Dropped:  {dropped}")
    print(f"Train: {train['week_end_date'].min().strftime('%Y-%m-%d')} to {train['week_end_date'].max().strftime('%Y-%m-%d')}")
    print(f"Test:  {test['week_end_date'].min().strftime('%Y-%m-%d')} to {test['week_end_date'].max().strftime('%Y-%m-%d')}")

    # --- Sample sizes ---
    print(f"\n{'='*80}")
    print(f"SAMPLE SIZE & BASE RATE AT EACH HORIZON")
    print(f"{'='*80}")
    print(f"\n{'Hz':>3}   {'Train n':>7}  {'Persist':>7}    {'Break':>5}   {'Base%':>5}     {'Flag':>4}   {'Test n':>6}  {'Persist':>7}    {'Break':>5}   {'Base%':>5}     {'Flag':>4}")
    print(f"{'-'*100}")

    for N in HORIZONS:
        target = f'persists_{N}w'
        tr_s = train[train['shape'] == shape][[target] + new_feats].dropna()
        te_s = test[test['shape'] == shape][[target] + new_feats].dropna()
        tr_br = tr_s[target].mean() * 100
        te_br = te_s[target].mean() * 100 if len(te_s) > 0 else 0
        tr_flag = '  TRIVIAL' if tr_br > 90 else ''
        te_flag = '  TRIVIAL' if te_br > 90 else ''
        print(f"  {N}w  {len(tr_s):>7}  {int(tr_s[target].sum()):>7}    {int(len(tr_s) - tr_s[target].sum()):>5}   {tr_br:>5.1f}%{tr_flag:>9}  {len(te_s):>6}  {int(te_s[target].sum()) if len(te_s)>0 else 0:>7}    {int(len(te_s) - te_s[target].sum()) if len(te_s)>0 else 0:>5}   {te_br:>5.1f}%{te_flag:>9}")

    # --- Training set results ---
    print(f"\n{'='*80}")
    print(f"TRAINING SET RESULTS (all 8 horizons)")
    print(f"{'='*80}")

    feat_hdrs = '   '.join([f'{f[:8]:>8}' for f in new_feats])
    print(f"\n{'Hz':>3} {'Model':>5}   {'Acc%':>5}  {'Base%':>5}    {'Lift':>6}     {feat_hdrs}  {'Dirs match?':>12}")
    print(f"{'-'*95}")

    all_results = {}

    for N in HORIZONS:
        res = fit_and_eval(shape, new_feats, train, test, N)
        all_results[N] = res

        # LR-W row
        coeff_strs = []
        all_match = True
        for f in new_feats:
            c = res['lr_coeffs'][f]
            d = 'negative' if c < 0 else 'positive'
            exp = EXPECTED.get((f, shape), '?')
            match = d[:3] == exp[:3]
            if not match:
                all_match = False
            coeff_strs.append(f'{c:>+8.4f}')

        dir_status = 'ALL YES' if all_match else 'MIXED'
        coeff_str = '     '.join(coeff_strs)
        lift = res['lr_train_acc'] - res['base_acc_train']
        print(f"  {N}w    LR  {res['lr_train_acc']*100:>5.1f}%  {res['base_acc_train']*100:>5.1f}%   {lift*100:>+5.1f}pp     {coeff_str}  {dir_status:>12}")

        # RF row
        imp_strs = []
        for f in new_feats:
            imp_strs.append(f'imp={res["rf_importances"][f]:.3f}')
        imp_str = '     '.join(imp_strs)
        print(f"  {N}w    RF  {res['rf_train_acc']*100:>5.1f}%  {res['base_acc_train']*100:>5.1f}%  {(res['rf_train_acc'] - res['base_acc_train'])*100:>+5.1f}pp  {imp_str}  OOB={res['oob_mean']*100:.1f}+/-{res['oob_std']*100:.1f}%")

    # --- Test set results ---
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS (all 8 horizons)")
    print(f"{'='*80}")

    print(f"\n{'Hz':>3} {'Model':>5}   {'Acc%':>5}  {'Prec1':>5}   {'Rec1':>5}  {'Prec0':>5}   {'Rec0':>5}  {'Base%':>5}  {'BaseAcc':>7}  {'vs Base':>7} {'Note':>10}")
    print(f"{'-'*100}")

    for N in HORIZONS:
        res = all_results[N]
        if f'lr_test_acc' not in res:
            print(f"  {N}w    LR  {'N/A':>5}  (no test data)")
            print(f"  {N}w    RF  {'N/A':>5}  (no test data)")
            continue

        for mname in ['lr', 'rf']:
            acc = res[f'{mname}_test_acc']
            p1 = res[f'{mname}_test_prec1']
            r1 = res[f'{mname}_test_rec1']
            p0 = res[f'{mname}_test_prec0']
            r0 = res[f'{mname}_test_rec0']
            ba = res[f'{mname}_test_base_acc']
            vs = res[f'{mname}_test_vs_base']
            degen = res[f'{mname}_test_degen']
            label = 'LR' if mname == 'lr' else 'RF'

            print(f"  {N}w    {label}  {acc*100:>5.1f}%  {p1*100:>5.1f}%  {r1*100:>5.1f}%  {p0*100:>5.1f}%  {r0*100:>5.1f}%  {res[f'{mname}_test_base_rate']*100:>5.1f}%  {ba*100:>7.1f}%  {vs*100:>+6.1f}pp {degen:>10}")

    # --- Direction check on test set ---
    print(f"\n{'='*80}")
    print(f"FEATURE DIRECTION ON TEST SET (median-split gap)")
    print(f"{'='*80}")

    print(f"\n{'Hz':>3}", end="")
    for f in new_feats:
        print(f"   {f[:14]:>14} gap  {'dir':>4}  {'exp':>4}  {'ok':>4}", end="")
    print()
    print(f"  {'-'*80}")

    dir_results = {f: [] for f in new_feats}

    for N in HORIZONS:
        row = f"  {N}w"
        for f in new_feats:
            gap, n_hi, n_lo = direction_gap(shape, f, N, train, test)
            exp = EXPECTED.get((f, shape), '?')
            if gap is not None:
                d = 'pos' if gap > 0 else 'neg'
                match = d[:3] == exp[:3]
                dir_results[f].append(match)
                ok = 'YES' if match else 'NO'
                row += f"   {gap*100:>+14.1f}pp  {d:>4}  {exp[:3]:>4}  {ok:>4}"
            else:
                row += f"   {'insuf.':>14}     {'':>4}  {'':>4}  {'':>4}"
                dir_results[f].append(None)
        print(row)

    print(f"\n  Direction stability summary:")
    for f in new_feats:
        valid = [x for x in dir_results[f] if x is not None]
        holds = sum(valid)
        total = len(valid)
        exp = EXPECTED.get((f, shape), '?')
        print(f"    {f:<28} {holds}/{total} correct on test (expected: {exp})")

    # Find where direction flips
    for f in new_feats:
        valid = [(i+1, x) for i, x in enumerate(dir_results[f]) if x is not None]
        flip_point = None
        for hz, ok in valid:
            if not ok:
                flip_point = hz
                break
        if flip_point:
            print(f"    {f}: first test-set direction flip at {flip_point}w")

# ============================================================
# SECTION 3: Before/After Comparison Table
# ============================================================
print(f"\n\n{'='*80}")
print(f"12c. BEFORE/AFTER COMPARISON — ORIGINAL vs REDUCED FEATURE SETS")
print(f"{'='*80}")
print()
print("Side-by-side comparison of test-set performance.")
print("ORIGINAL = Phase B initial feature set; NEW = independently-confirmed features only.")

for shape in ['1', '2']:
    sname = SHAPE_NAMES[shape]
    old_feats = OLD_FEATURES[shape]
    new_feats = NEW_FEATURES[shape]

    print(f"\n  {'='*70}")
    print(f"  SHAPE {shape} ({sname})")
    print(f"  Original: {old_feats}")
    print(f"  New:      {new_feats}")
    print(f"  {'='*70}")

    print(f"\n  {'Hz':>3}  {'--- ORIGINAL (LR-W) ---':>28}  {'--- NEW (LR-W) ---':>24}  {'--- ORIGINAL (RF) ---':>28}  {'--- NEW (RF) ---':>24}")
    print(f"  {'':>3}  {'Acc%':>5} {'vs base':>7} {'degen':>8}  {'Acc%':>5} {'vs base':>7} {'degen':>8}  {'Acc%':>5} {'vs base':>7} {'degen':>8}  {'Acc%':>5} {'vs base':>7} {'degen':>8}  {'delta LR':>8} {'delta RF':>8}")
    print(f"  {'-'*145}")

    for N in HORIZONS:
        # Fit original
        old_res = fit_and_eval(shape, old_feats, train, test, N)
        # Fit new
        new_res = fit_and_eval(shape, new_feats, train, test, N)

        if 'lr_test_acc' not in old_res or 'lr_test_acc' not in new_res:
            print(f"  {N}w  insufficient data")
            continue

        o_lr_acc = old_res['lr_test_acc']
        o_lr_vs = old_res['lr_test_vs_base']
        o_lr_d = old_res['lr_test_degen'] or ''
        n_lr_acc = new_res['lr_test_acc']
        n_lr_vs = new_res['lr_test_vs_base']
        n_lr_d = new_res['lr_test_degen'] or ''

        o_rf_acc = old_res['rf_test_acc']
        o_rf_vs = old_res['rf_test_vs_base']
        o_rf_d = old_res['rf_test_degen'] or ''
        n_rf_acc = new_res['rf_test_acc']
        n_rf_vs = new_res['rf_test_vs_base']
        n_rf_d = new_res['rf_test_degen'] or ''

        d_lr = (n_lr_acc - o_lr_acc) * 100
        d_rf = (n_rf_acc - o_rf_acc) * 100

        print(f"  {N}w  {o_lr_acc*100:>5.1f}% {o_lr_vs*100:>+6.1f}pp {o_lr_d:>8}  {n_lr_acc*100:>5.1f}% {n_lr_vs*100:>+6.1f}pp {n_lr_d:>8}  {o_rf_acc*100:>5.1f}% {o_rf_vs*100:>+6.1f}pp {o_rf_d:>8}  {n_rf_acc*100:>5.1f}% {n_rf_vs*100:>+6.1f}pp {n_rf_d:>8}  {d_lr:>+7.1f}pp {d_rf:>+7.1f}pp")

# ============================================================
# SECTION 4: Updated Usable-Range Conclusion
# ============================================================
print(f"\n\n{'='*80}")
print(f"12d. UPDATED USABLE-RANGE CONCLUSION")
print(f"{'='*80}")

for shape in ['1', '2']:
    sname = SHAPE_NAMES[shape]
    new_feats = NEW_FEATURES[shape]

    print(f"\n  {'='*70}")
    print(f"  SHAPE {shape} ({sname}) — 2-feature model: {new_feats}")
    print(f"  {'='*70}")

    # Refit at all horizons and collect results
    lr_beats = []
    rf_beats = []
    lr_degen_list = []
    rf_degen_list = []
    dir_ok_at = {}

    for N in HORIZONS:
        res = fit_and_eval(shape, new_feats, train, test, N)

        if 'lr_test_acc' not in res:
            continue

        lr_vs = res['lr_test_vs_base']
        rf_vs = res['rf_test_vs_base']
        lr_d = res['lr_test_degen']
        rf_d = res['rf_test_degen']

        if lr_vs > 0 and not lr_d:
            lr_beats.append(N)
        if rf_vs > 0 and not rf_d:
            rf_beats.append(N)
        if lr_d:
            lr_degen_list.append(N)
        if rf_d:
            rf_degen_list.append(N)

        # Direction check
        all_ok = True
        for f in new_feats:
            gap, _, _ = direction_gap(shape, f, N, train, test)
            if gap is not None:
                exp = EXPECTED.get((f, shape), '?')
                d = 'pos' if gap > 0 else 'neg'
                if d[:3] != exp[:3]:
                    all_ok = False
        dir_ok_at[N] = all_ok

    # Report
    print(f"\n  LR-W STATUS:")
    if lr_degen_list:
        print(f"    Degenerate at horizons: {lr_degen_list}")
    if lr_beats:
        print(f"    Beats baseline at horizons: {lr_beats}")
        for N in lr_beats:
            r = fit_and_eval(shape, new_feats, train, test, N)
            print(f"      {N}w: {r['lr_test_acc']*100:.1f}% vs {r['lr_test_base_acc']*100:.1f}% baseline ({r['lr_test_vs_base']*100:+.1f}pp)")
    else:
        print(f"    Does not beat baseline at any horizon.")

    print(f"\n  RF STATUS:")
    if rf_degen_list:
        print(f"    Degenerate at horizons: {rf_degen_list}")
    if rf_beats:
        print(f"    Beats baseline at horizons: {rf_beats}")
        for N in rf_beats:
            r = fit_and_eval(shape, new_feats, train, test, N)
            print(f"      {N}w: {r['rf_test_acc']*100:.1f}% vs {r['rf_test_base_acc']*100:.1f}% baseline ({r['rf_test_vs_base']*100:+.1f}pp)")
    else:
        print(f"    Does not beat baseline at any horizon.")

    print(f"\n  DIRECTION STABILITY:")
    holds = [N for N in HORIZONS if dir_ok_at.get(N, False)]
    flips = [N for N in HORIZONS if not dir_ok_at.get(N, True)]
    print(f"    All directions hold at: {holds if holds else 'none'}")
    print(f"    Direction flips at:     {flips if flips else 'none'}")

    # Combined: beats baseline AND directions hold
    lr_usable = [N for N in lr_beats if dir_ok_at.get(N, False)]
    rf_usable = [N for N in rf_beats if dir_ok_at.get(N, False)]
    any_usable = sorted(set(lr_usable + rf_usable))

    print(f"\n  USABLE HORIZONS (beats baseline + correct directions):")
    print(f"    LR-W: {lr_usable if lr_usable else 'none'}")
    print(f"    RF:   {rf_usable if rf_usable else 'none'}")
    print(f"    Either: {any_usable if any_usable else 'none'}")

    # Best result
    best_acc = 0
    best_model = ''
    best_hz = 0
    best_vs = 0
    for N in any_usable:
        for mname in ['lr', 'rf']:
            r = fit_and_eval(shape, new_feats, train, test, N)
            acc = r[f'{mname}_test_acc']
            vs = r[f'{mname}_test_vs_base']
            if acc > best_acc:
                best_acc = acc
                best_model = mname.upper()
                best_hz = N
                best_vs = vs

    if best_hz > 0:
        print(f"\n  BEST RESULT: {best_model} at {best_hz}w — {best_acc*100:.1f}% accuracy ({best_vs*100:+.1f}pp vs baseline)")
    else:
        print(f"\n  NO USABLE HORIZON: the 2-feature model does not beat baseline with correct")
        print(f"  directions at any horizon on the 2025-2026 test set.")

# ============================================================
# OVERALL CONCLUSION
# ============================================================
print(f"\n\n{'='*80}")
print(f"12e. OVERALL CONCLUSION — PHASE B MODEL STATUS AFTER FEATURE PRUNING")
print(f"{'='*80}")

print(f"""
  This section completes the Phase B model build by stripping each shape's model
  down to only the features that independently confirmed their expected direction
  in single-feature testing (Section 11). The results below replace the earlier
  conclusions from Sections 8-9.

  SHAPE 0.0 (Contango):
    Status: NO MODEL. No reliable feature combination found (Sections 10-11).
    Prediction: unconditional base rate only.

  SHAPE 0.1 (Mild Contango):
    Status: NO MODEL. No validated driver (Phase A). Base rate only.

  SHAPE 0.2 (Steep Backwardation):
    Status: NO MODEL. Insufficient data (n=26). Base rate only.
""")

# Print dynamic conclusions for shapes 1 and 2
for shape in ['1', '2']:
    sname = SHAPE_NAMES[shape]
    new_feats = NEW_FEATURES[shape]

    # Quick refit summary
    usable = []
    best_acc = 0
    best_model = ''
    best_hz = 0
    best_vs = 0

    for N in HORIZONS:
        res = fit_and_eval(shape, new_feats, train, test, N)
        if 'lr_test_acc' not in res:
            continue

        # Check directions
        all_ok = True
        for f in new_feats:
            gap, _, _ = direction_gap(shape, f, N, train, test)
            if gap is not None:
                exp = EXPECTED.get((f, shape), '?')
                d = 'pos' if gap > 0 else 'neg'
                if d[:3] != exp[:3]:
                    all_ok = False

        for mname in ['lr', 'rf']:
            vs = res[f'{mname}_test_vs_base']
            degen = res[f'{mname}_test_degen']
            if vs > 0 and not degen and all_ok:
                usable.append((N, mname.upper()))
                acc = res[f'{mname}_test_acc']
                if acc > best_acc:
                    best_acc = acc
                    best_model = mname.upper()
                    best_hz = N
                    best_vs = vs

    print(f"  SHAPE {shape} ({sname}):")
    print(f"    Features: {new_feats}")
    if usable:
        hz_list = sorted(set([u[0] for u in usable]))
        print(f"    Usable horizons: {hz_list}")
        print(f"    Best: {best_model} at {best_hz}w — {best_acc*100:.1f}% ({best_vs*100:+.1f}pp vs baseline)")
    else:
        print(f"    Usable horizons: NONE")
        print(f"    The 2-feature model does not produce a usable result on the test set.")
    print()

print(f"{'='*80}")
print(f"END OF SECTION 12")
print(f"{'='*80}")

file_buf.close()
