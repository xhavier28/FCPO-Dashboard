"""
Phase B Follow-Up #2: Horizon Sweep for Shape 0.0 (Contango) and Shape 2 (Flat)
Adds class-weighted LR alongside plain LR and RF.
Appends SECTION 9 to research/outputs/Phase_B_Persistence_Model.txt
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

SHAPE_NAMES = {'0.0': 'Contango', '1': 'Backwardation', '2': 'Flat'}
SHAPE_FEATURES = {
    '0.0': ['usd_myr', 'enso_oni'],
    '2':   ['enso_oni', 'crude_oil_price', 'palm_soy_spread', 'production_yoy_pct'],
}
EXPECTED_DIRECTIONS = {
    ('usd_myr', '0.0'): 'negative',
    ('enso_oni', '0.0'): 'positive',
    ('enso_oni', '2'): 'negative',
    ('crude_oil_price', '2'): 'negative',
    ('palm_soy_spread', '2'): 'positive',
    ('production_yoy_pct', '2'): 'positive',
}
HORIZONS = list(range(1, 9))

# ============================================================
# Rebuild weekly panel (same as Phase B Section 0)
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
train = weekly[weekly['week_end_date'] <= train_cutoff].copy()
test = weekly[weekly['week_end_date'] > train_cutoff].copy()

# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 9: SHAPE 0.0 AND SHAPE 2 — HORIZON SWEEP (1w to 8w)")
print("=" * 80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Three model variants: Plain LR, Class-Weighted LR (balanced), Random Forest")
print(f"Train: {train['week_end_date'].min().date()} to {train['week_end_date'].max().date()} ({len(train)} weeks)")
print(f"Test:  {test['week_end_date'].min().date()} to {test['week_end_date'].max().date()} ({len(test)} weeks)")

# ============================================================
# Process each shape
# ============================================================
for shape in ['0.0', '2']:
    features = SHAPE_FEATURES[shape]
    sname = SHAPE_NAMES[shape]

    print(f"\n\n{'#'*80}")
    print(f"# SHAPE {shape} — {sname}")
    print(f"# Features: {features}")
    print(f"{'#'*80}")

    # --- 9a: Sample size & base rate ---
    print(f"\n{'='*80}")
    print(f"9a-{shape}. SAMPLE SIZE & BASE RATE AT EACH HORIZON — Shape {shape} ({sname})")
    print(f"{'='*80}")
    print(f"\n{'Hz':>3}  {'Train n':>8} {'Persist':>8} {'Break':>8} {'Base%':>7} {'Flag':>8}  "
          f"{'Test n':>7} {'Persist':>8} {'Break':>8} {'Base%':>7} {'Flag':>8}")
    print("-" * 100)

    trivial_horizons = set()
    for N in HORIZONS:
        target = f'persists_{N}w'
        tr = train[(train['shape'] == shape)][target].dropna()
        te = test[(test['shape'] == shape)][target].dropna()
        tr_p, tr_b = int(tr.sum()), len(tr) - int(tr.sum())
        tr_rate = tr.mean() * 100 if len(tr) > 0 else 0
        te_p, te_b = int(te.sum()) if len(te) > 0 else 0, len(te) - int(te.sum()) if len(te) > 0 else 0
        te_rate = te.mean() * 100 if len(te) > 0 else 0
        tr_flag = 'TRIVIAL' if tr_rate > 90 else ''
        te_flag = 'TRIVIAL' if te_rate > 90 else ''
        if tr_rate > 90 or te_rate > 90:
            trivial_horizons.add(N)
        print(f"{N:>3}w {len(tr):>8} {tr_p:>8} {tr_b:>8} {tr_rate:>6.1f}% {tr_flag:>8}  "
              f"{len(te):>7} {te_p:>8} {te_b:>8} {te_rate:>6.1f}% {te_flag:>8}")

    if trivial_horizons:
        print(f"\nTRIVIAL horizons (base rate >90%): {sorted(trivial_horizons)}")
    else:
        print(f"\nNo horizon flagged as trivial.")

    # --- 9b: Training results ---
    print(f"\n{'='*80}")
    print(f"9b-{shape}. TRAINING SET RESULTS — Shape {shape} ({sname})")
    print(f"{'='*80}")

    # Header for coefficient table
    feat_headers = '  '.join(f'{f[:10]:>10}' for f in features)
    print(f"\n{'Hz':>3} {'Model':>6} {'Acc%':>6} {'Base%':>6} {'Lift':>7}  {feat_headers}  {'Dirs':>8}")
    print("-" * (55 + 12 * len(features)))

    fitted = {}

    for N in HORIZONS:
        target = f'persists_{N}w'
        sub = train[(train['shape'] == shape)][features + [target]].dropna()
        X = sub[features].values
        y = sub[target].values.astype(int)
        base_rate = y.mean() * 100

        if len(y) < 10:
            print(f"{N:>3}w   SKIPPED — only {len(y)} training samples")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Plain LR
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, y)
        acc_lr = accuracy_score(y, lr.predict(X_scaled)) * 100
        coeffs_lr = lr.coef_[0]
        dirs_lr = ['pos' if c > 0 else 'neg' for c in coeffs_lr]
        matches_lr = [dirs_lr[i][:3] == EXPECTED_DIRECTIONS.get((f, shape), '?')[:3]
                      for i, f in enumerate(features)]
        match_str_lr = 'ALL YES' if all(matches_lr) else ' '.join('Y' if m else 'N' for m in matches_lr)
        coeff_str_lr = '  '.join(f'{c:>+10.4f}' for c in coeffs_lr)
        print(f"{N:>3}w {'LR':>6} {acc_lr:>5.1f}% {base_rate:>5.1f}% {acc_lr-base_rate:>+6.1f}pp  {coeff_str_lr}  {match_str_lr:>8}")

        # Class-weighted LR
        lr_w = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr_w.fit(X_scaled, y)
        acc_lrw = accuracy_score(y, lr_w.predict(X_scaled)) * 100
        coeffs_lrw = lr_w.coef_[0]
        dirs_lrw = ['pos' if c > 0 else 'neg' for c in coeffs_lrw]
        matches_lrw = [dirs_lrw[i][:3] == EXPECTED_DIRECTIONS.get((f, shape), '?')[:3]
                       for i, f in enumerate(features)]
        match_str_lrw = 'ALL YES' if all(matches_lrw) else ' '.join('Y' if m else 'N' for m in matches_lrw)
        coeff_str_lrw = '  '.join(f'{c:>+10.4f}' for c in coeffs_lrw)
        print(f"{N:>3}w {'LR-W':>6} {acc_lrw:>5.1f}% {base_rate:>5.1f}% {acc_lrw-base_rate:>+6.1f}pp  {coeff_str_lrw}  {match_str_lrw:>8}")

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        acc_rf = accuracy_score(y, rf.predict(X)) * 100

        n_resamples = 20
        rf_oob_accs = []
        rf_imps = np.zeros((n_resamples, len(features)))
        for b in range(n_resamples):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            oob = np.array([i for i in range(len(X)) if i not in idx])
            rf_b = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=b, n_jobs=-1)
            rf_b.fit(X[idx], y[idx])
            rf_imps[b] = rf_b.feature_importances_
            if len(oob) > 0:
                rf_oob_accs.append(accuracy_score(y[oob], rf_b.predict(X[oob])) * 100)

        imp_str = '  '.join(f'{"imp=":>4}{v:.3f}   ' for v in rf.feature_importances_)
        oob_str = f"OOB={np.mean(rf_oob_accs):.1f}±{np.std(rf_oob_accs):.1f}%" if rf_oob_accs else ""
        print(f"{N:>3}w {'RF':>6} {acc_rf:>5.1f}% {base_rate:>5.1f}% {acc_rf-base_rate:>+6.1f}pp  {imp_str}  {oob_str}")

        fitted[N] = {'lr': lr, 'lr_w': lr_w, 'rf': rf, 'scaler': scaler, 'base_rate': base_rate,
                      'train_n': len(y)}

    # --- 9c: Test results ---
    print(f"\n{'='*80}")
    print(f"9c-{shape}. TEST SET RESULTS — Shape {shape} ({sname})")
    print(f"{'='*80}")

    print(f"\n{'Hz':>3} {'Model':>6} {'Acc%':>6} {'Prec1':>6} {'Rec1':>6} {'Prec0':>6} {'Rec0':>6} "
          f"{'Base%':>6} {'BaseAcc':>8} {'vs Base':>8} {'Note':<14}")
    print("-" * 105)

    test_results = []
    # Track class-weight fix diagnosis
    plain_lr_degen_count = 0
    weighted_lr_degen_count = 0
    plain_lr_total = 0
    weighted_lr_total = 0

    for N in HORIZONS:
        if N not in fitted:
            continue
        target = f'persists_{N}w'
        sub_test = test[(test['shape'] == shape)][features + [target]].dropna()

        if len(sub_test) == 0:
            print(f"{N:>3}w   — NO TEST DATA —")
            continue

        X_test = sub_test[features].values
        y_test = sub_test[target].values.astype(int)
        test_base = y_test.mean() * 100
        n_test = len(y_test)

        train_base = fitted[N]['base_rate']
        baseline_pred = 1 if train_base >= 50 else 0
        baseline_acc = accuracy_score(y_test, [baseline_pred] * n_test) * 100

        for model_name, model_key in [('LR', 'lr'), ('LR-W', 'lr_w'), ('RF', 'rf')]:
            model = fitted[N][model_key]
            if model_name in ('LR', 'LR-W'):
                X_eval = fitted[N]['scaler'].transform(X_test)
            else:
                X_eval = X_test

            y_pred = model.predict(X_eval)
            acc = accuracy_score(y_test, y_pred) * 100

            if len(np.unique(y_pred)) > 1:
                p1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100
                r1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100
                p0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0) * 100
                r0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0) * 100
                is_degen = False
            else:
                p1 = r1 = p0 = r0 = 0.0
                is_degen = True

            vs_base = acc - baseline_acc
            trivial_flag = " TRIVIAL" if N in trivial_horizons else ""
            degen_flag = f" [all-{y_pred[0]:.0f}]" if is_degen else ""
            note = (trivial_flag + degen_flag).strip()

            print(f"{N:>3}w {model_name:>6} {acc:>5.1f}% {p1:>5.1f}% {r1:>5.1f}% {p0:>5.1f}% {r0:>5.1f}% "
                  f"{test_base:>5.1f}% {baseline_acc:>7.1f}% {vs_base:>+7.1f}pp {note:<14}")

            test_results.append({
                'horizon': N, 'model': model_name, 'acc': acc, 'baseline_acc': baseline_acc,
                'vs_base': vs_base, 'test_base': test_base, 'degenerate': is_degen,
                'trivial': N in trivial_horizons, 'n_test': n_test
            })

            if model_name == 'LR':
                plain_lr_total += 1
                if is_degen:
                    plain_lr_degen_count += 1
            elif model_name == 'LR-W':
                weighted_lr_total += 1
                if is_degen:
                    weighted_lr_degen_count += 1

    # Class-weight fix diagnosis
    print(f"\n  CLASS-WEIGHT FIX DIAGNOSIS — Shape {shape} ({sname}):")
    print(f"    Plain LR degenerate: {plain_lr_degen_count}/{plain_lr_total} horizons")
    print(f"    Class-weighted LR degenerate: {weighted_lr_degen_count}/{weighted_lr_total} horizons")
    if weighted_lr_degen_count < plain_lr_degen_count:
        fixed_count = plain_lr_degen_count - weighted_lr_degen_count
        print(f"    → YES, class weighting fixed degeneracy at {fixed_count} horizon(s).")
        # Check if the non-degenerate weighted LR actually beats baseline
        lrw_beats = [r for r in test_results if r['model'] == 'LR-W' and not r['degenerate']
                     and r['vs_base'] > 0 and not r['trivial']]
        if lrw_beats:
            print(f"    → And it beats baseline at: {[r['horizon'] for r in lrw_beats]}w")
        else:
            print(f"    → But it does NOT beat baseline at any non-degenerate, non-trivial horizon.")
    elif weighted_lr_degen_count == plain_lr_degen_count:
        if plain_lr_degen_count == plain_lr_total:
            print(f"    → NO, class weighting did NOT fix the collapse. Both are degenerate at every horizon.")
        else:
            print(f"    → Same degeneracy rate. Class weighting made no difference.")
    else:
        print(f"    → Class weighting actually INCREASED degeneracy (unexpected).")

    # --- 9d: Direction stability ---
    print(f"\n{'='*80}")
    print(f"9d-{shape}. FEATURE DIRECTION STABILITY — Shape {shape} ({sname})")
    print(f"{'='*80}")
    print(f"\nTracking LR coefficient direction (training) vs actual test-set gap direction.")
    print(f"Expected directions from Phase A shown for reference.\n")

    for feat in features:
        expected = EXPECTED_DIRECTIONS.get((feat, shape), '?')
        print(f"  {feat} (expected: {expected}):")
        print(f"  {'Hz':>3}  {'LR coeff':>10} {'LR dir':>8} {'Match':>6}  "
              f"{'LR-W coeff':>10} {'LR-W dir':>8} {'Match':>6}  {'Test gap':>10} {'Test dir':>10}")
        print(f"  {'-'*95}")

        feat_idx = features.index(feat)
        for N in HORIZONS:
            if N not in fitted:
                continue
            target = f'persists_{N}w'
            # LR coefficients
            c_lr = fitted[N]['lr'].coef_[0][feat_idx]
            d_lr = 'negative' if c_lr < 0 else 'positive'
            m_lr = 'YES' if d_lr == expected else 'NO'

            c_lrw = fitted[N]['lr_w'].coef_[0][feat_idx]
            d_lrw = 'negative' if c_lrw < 0 else 'positive'
            m_lrw = 'YES' if d_lrw == expected else 'NO'

            # Test gap
            te_sub = test[(test['shape'] == shape)][[feat, target]].dropna()
            if len(te_sub) >= 4:
                med = te_sub[feat].median()
                hi = te_sub[te_sub[feat] > med][target].mean() * 100
                lo = te_sub[te_sub[feat] <= med][target].mean() * 100
                gap = hi - lo
                d_test = 'negative' if gap < 0 else ('positive' if gap > 0 else 'zero')
                gap_str = f"{gap:+.1f}pp"
            else:
                gap_str = "insuf."
                d_test = "N/A"

            print(f"  {N:>3}w  {c_lr:>+10.4f} {d_lr:>8} {m_lr:>6}  "
                  f"{c_lrw:>+10.4f} {d_lrw:>8} {m_lrw:>6}  {gap_str:>10} {d_test:>10}")
        print()

    # --- 9e: Usable range ---
    print(f"\n{'='*80}")
    print(f"9e-{shape}. USABLE RANGE IDENTIFICATION — Shape {shape} ({sname})")
    print(f"{'='*80}")

    non_trivial = [N for N in HORIZONS if N not in trivial_horizons and N in fitted]
    floor_hz = min(non_trivial) if non_trivial else None
    print(f"\n  Practical floor (shortest non-trivial horizon): {floor_hz}w" if floor_hz else
          "\n  No non-trivial horizon available.")

    # Check each model variant
    for mname in ['LR', 'LR-W', 'RF']:
        beats = [r for r in test_results if r['model'] == mname and r['vs_base'] > 0
                 and not r['degenerate'] and not r['trivial']]
        degen = [r for r in test_results if r['model'] == mname and r['degenerate'] and not r['trivial']]
        beats_hz = sorted(set(r['horizon'] for r in beats))
        degen_hz = sorted(set(r['horizon'] for r in degen))
        print(f"\n  {mname}:")
        print(f"    Beats baseline at: {beats_hz if beats_hz else 'none'}")
        if beats_hz:
            for r in sorted(beats, key=lambda x: x['horizon']):
                print(f"      {r['horizon']}w: {r['acc']:.1f}% vs {r['baseline_acc']:.1f}% (+{r['vs_base']:.1f}pp, n={r['n_test']})")
        print(f"    Degenerate at: {degen_hz if degen_hz else 'none'}")

    # Overall assessment — any model beats baseline with correct feature directions?
    any_beats = [r for r in test_results if r['vs_base'] > 0 and not r['degenerate'] and not r['trivial']]
    beats_hz_all = sorted(set(r['horizon'] for r in any_beats))

    print(f"\n  ANY model beats baseline at: {beats_hz_all if beats_hz_all else 'none'}")

    # Direction check on those horizons
    if beats_hz_all:
        print(f"\n  Direction check on beating horizons:")
        for N in beats_hz_all:
            target = f'persists_{N}w'
            dir_ok = True
            dir_details = []
            for feat in features:
                expected = EXPECTED_DIRECTIONS.get((feat, shape), '?')
                te_sub = test[(test['shape'] == shape)][[feat, target]].dropna()
                if len(te_sub) >= 4:
                    med = te_sub[feat].median()
                    hi = te_sub[te_sub[feat] > med][target].mean() * 100
                    lo = te_sub[te_sub[feat] <= med][target].mean() * 100
                    gap = hi - lo
                    actual = 'negative' if gap < 0 else 'positive'
                    ok = actual == expected
                    dir_details.append(f"{feat[:12]}={'OK' if ok else 'FLIP'}")
                    if not ok:
                        dir_ok = False
                else:
                    dir_details.append(f"{feat[:12]}=insuf.")
            status = "ALL MATCH" if dir_ok else "SOME FLIPPED"
            print(f"    {N}w: {status} ({', '.join(dir_details)})")

    # Conclusion for this shape
    print(f"\n  {'='*60}")
    print(f"  CONCLUSION — Shape {shape} ({sname}):")

    if not any_beats:
        print(f"  No usable horizon found in 1-8 weeks. No model variant (plain LR, class-weighted")
        print(f"  LR, or RF) beats the naive baseline on the 2025-2026 test set at any non-trivial horizon.")
    else:
        best = max(any_beats, key=lambda r: r['vs_base'])
        print(f"  Best out-of-sample result: {best['model']} at {best['horizon']}w — {best['acc']:.1f}% accuracy")
        print(f"  vs {best['baseline_acc']:.1f}% baseline (+{best['vs_base']:.1f}pp lift, n={best['n_test']}).")
        # Check if this is a robust finding
        if best['n_test'] < 10:
            print(f"  WARNING: Very small test sample (n={best['n_test']}). Treat with caution.")
        if best['vs_base'] < 5:
            print(f"  NOTE: Lift is marginal (<5pp). May not be practically meaningful.")

print(f"\n\n{'='*80}")
print(f"END OF SECTION 9")
print(f"{'='*80}")

file_buf.close()
