"""
Phase B: Persistence Model Build
Builds predictive models for regime persistence using Phase A's validated drivers.
Outputs to research/outputs/Phase_B_Persistence_Model.txt
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
RAW = os.path.join(BASE, 'Raw Data')
OUTPUT_PATH = os.path.join(BASE, 'research', 'outputs', 'Phase_B_Persistence_Model.txt')

# Tee output to both console and file
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

# Validated feature sets from Phase A (Section 1 of instructions)
SHAPE_FEATURES = {
    '0.0': ['usd_myr', 'enso_oni'],
    '1':   ['usd_myr', 'enso_oni', 'stock_to_usage_ratio'],
    '2':   ['enso_oni', 'crude_oil_price', 'palm_soy_spread', 'production_yoy_pct'],
}
MODELABLE_SHAPES = ['0.0', '1', '2']

# Expected directions from Phase A
# (feature, shape) -> expected sign of logistic coefficient
# Negative = higher value -> lower persistence
EXPECTED_DIRECTIONS = {
    ('usd_myr', '0.0'): 'negative',     # higher FX -> lower persistence
    ('enso_oni', '0.0'): 'positive',     # higher ONI -> higher persistence (Contango)
    ('usd_myr', '1'): 'negative',       # higher FX -> lower persistence (-33.6pp gap)
    ('enso_oni', '1'): 'positive',       # higher ONI -> higher persistence (+19.8pp)
    ('stock_to_usage_ratio', '1'): 'negative',  # higher stock ratio -> lower persistence
    ('enso_oni', '2'): 'negative',       # higher ONI -> lower persistence for Flat
    ('crude_oil_price', '2'): 'negative',# higher crude -> lower persistence for Flat
    ('palm_soy_spread', '2'): 'positive',# direction from Phase A screening
    ('production_yoy_pct', '2'): 'positive',  # direction from Phase A screening
}

print("=" * 80)
print("PHASE B: PERSISTENCE MODEL BUILD — COMPLETE REPORT")
print("=" * 80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Training window: 2017-W01 through 2024-W52")
print(f"Test window: 2025-W01 through 2026-W26")
print(f"Model type: Per-shape logistic regression + Random Forest comparison")
print(f"Survival model: Cox proportional hazards (lifelines)")
print(f"Phase A inputs: Validated driver map (fixed, not re-derived)")

# ============================================================
# SECTION 0: REBUILD WEEKLY PANEL (same logic as Phase A)
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 0: DATA LOADING & WEEKLY PANEL RECONSTRUCTION")
print("=" * 80)

# --- Shape log ---
shape_log = pd.read_csv(os.path.join(RAW, 'Research', 'daily_shape_log.csv'),
                         dtype={'shape': str}, parse_dates=['date'])
print(f"\nShape log: {len(shape_log)} rows, {shape_log['date'].min().date()} to {shape_log['date'].max().date()}")

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

# --- FX ---
fx_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'FX_IDC_USDMYR, 1D_1227e.csv'))
fx_df['date'] = pd.to_datetime(fx_df['time'], unit='s')
fx_df = fx_df[['date', 'close']].rename(columns={'close': 'usd_myr'}).sort_values('date')

# --- Palm-soy spread ---
ps_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'MYX_DLY_FCPO1!_2_CBOT_DL_ZL1!, 1D_61912.csv'))
ps_df['date'] = pd.to_datetime(ps_df['time'], unit='s')
ps_df = ps_df[['date', 'close']].rename(columns={'close': 'palm_soy_spread'}).sort_values('date')

# --- Crude oil ---
cl_df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'NYMEX_DL_CL1!, 1D_84001.csv'))
cl_df['date'] = pd.to_datetime(cl_df['time'], unit='s')
cl_df = cl_df[['date', 'close']].rename(columns={'close': 'crude_oil_price'}).sort_values('date')

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

# MPOB monthly -> weekly
mpob = stock_df.set_index('date')[['stock']].join(
    prod_df.set_index('date')[['production']], how='outer'
).join(export_df.set_index('date')[['export']], how='outer').sort_index()
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

# Targets
for N in [4, 12]:
    weekly[f'shape_plus_{N}w'] = weekly['shape'].shift(-N)
    weekly[f'persists_{N}w'] = (weekly['shape'] == weekly[f'shape_plus_{N}w']).astype(int)
    weekly.loc[weekly[f'shape_plus_{N}w'].isna(), f'persists_{N}w'] = np.nan

weekly = weekly[weekly['week_end_date'] >= '2017-01-01']

print(f"\nWeekly panel: {weekly.shape[0]} weeks x {weekly.shape[1]} columns")
print(f"Date range: {weekly['week_end_date'].min().date()} to {weekly['week_end_date'].max().date()}")

# ============================================================
# SECTION 1: TRAIN/TEST SPLIT
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 1: TRAIN/TEST SPLIT")
print("=" * 80)

train_cutoff = pd.Timestamp('2024-12-31')
train = weekly[weekly['week_end_date'] <= train_cutoff].copy()
test = weekly[weekly['week_end_date'] > train_cutoff].copy()

print(f"\nTrain set: {len(train)} weeks")
print(f"  Date range: {train['week_end_date'].min().date()} to {train['week_end_date'].max().date()}")
print(f"Test set:  {len(test)} weeks")
print(f"  Date range: {test['week_end_date'].min().date()} to {test['week_end_date'].max().date()}")

# Overlap check
train_dates = set(train['week_end_date'])
test_dates = set(test['week_end_date'])
overlap = train_dates & test_dates
print(f"\nDate overlap check: {len(overlap)} overlapping dates {'(PASS — no overlap)' if len(overlap) == 0 else '*** FAIL ***'}")

# Save locked test set
test_lock_path = os.path.join(BASE, 'research', 'outputs', 'phase_b_test_set_LOCKED.csv')
test.to_csv(test_lock_path)
print(f"Test set saved to: {os.path.basename(test_lock_path)}")

# Per-shape counts
print(f"\nPer-shape train/test counts:")
print(f"{'Shape':<8} {'Name':<22} {'Train 4w':>10} {'Train 12w':>10} {'Test 4w':>10} {'Test 12w':>10}")
print("-" * 70)
for s in ALL_SHAPES:
    name = SHAPE_NAMES[s]
    tr4 = train[(train['shape'] == s) & train['persists_4w'].notna()]
    tr12 = train[(train['shape'] == s) & train['persists_12w'].notna()]
    te4 = test[(test['shape'] == s) & test['persists_4w'].notna()]
    te12 = test[(test['shape'] == s) & test['persists_12w'].notna()]
    print(f"{s:<8} {name:<22} {len(tr4):>10} {len(tr12):>10} {len(te4):>10} {len(te12):>10}")

# ============================================================
# SECTION 2: BUILD SURVIVAL EPISODES (for Cox models later)
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 2: SURVIVAL EPISODE CONSTRUCTION")
print("=" * 80)

weekly_sorted = weekly.sort_values('week_end_date').copy()
weekly_sorted['regime_start'] = weekly_sorted['shape'] != weekly_sorted['shape_prev']

episodes = []
current_shape = None
start_i = None
for i, (idx, row) in enumerate(weekly_sorted.iterrows()):
    if row['regime_start'] or current_shape is None:
        if current_shape is not None:
            episodes.append({
                'shape': current_shape, 'start_week': start_idx,
                'start_date': start_date, 'duration_weeks': i - start_i, 'censored': False
            })
        current_shape = row['shape']
        start_idx = idx
        start_date = row['week_end_date']
        start_i = i
if current_shape is not None:
    episodes.append({
        'shape': current_shape, 'start_week': start_idx,
        'start_date': start_date, 'duration_weeks': len(weekly_sorted) - start_i, 'censored': True
    })

ep_df = pd.DataFrame(episodes)

# Enrich episodes with covariate values at start
all_features_needed = set()
for feats in SHAPE_FEATURES.values():
    all_features_needed.update(feats)

for var in all_features_needed:
    vals = []
    for _, ep in ep_df.iterrows():
        wk = ep['start_week']
        if wk in weekly.index and pd.notna(weekly.loc[wk, var]):
            vals.append(weekly.loc[wk, var])
        else:
            vals.append(np.nan)
    ep_df[var] = vals

# Split episodes into train/test by start_date
ep_train = ep_df[ep_df['start_date'] <= train_cutoff].copy()
ep_test = ep_df[ep_df['start_date'] > train_cutoff].copy()

print(f"\nTotal episodes: {len(ep_df)} (train: {len(ep_train)}, test: {len(ep_test)})")
for s in MODELABLE_SHAPES:
    n_tr = len(ep_train[ep_train['shape'] == s])
    n_te = len(ep_test[ep_test['shape'] == s])
    print(f"  Shape {s} ({SHAPE_NAMES[s]}): {n_tr} train episodes, {n_te} test episodes")

# ============================================================
# SECTION 3: FIXED-HORIZON PERSISTENCE MODELS (Training)
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 3: FIXED-HORIZON PERSISTENCE MODELS — TRAINING SET RESULTS")
print("=" * 80)
print("\nFor each modelable shape, fitting Logistic Regression (primary) and")
print("Random Forest (comparison) on the 2017-2024 training set.")
print("Features per shape are fixed from Phase A's validated driver map.")

fitted_models = {}  # (shape, horizon) -> {'lr': model, 'rf': model, 'scaler': scaler, 'features': [...]}

for s in MODELABLE_SHAPES:
    features = SHAPE_FEATURES[s]
    print(f"\n{'='*70}")
    print(f"SHAPE {s} — {SHAPE_NAMES[s]}")
    print(f"Features: {features}")
    print(f"{'='*70}")

    for N in [4, 12]:
        target = f'persists_{N}w'
        sub = train[(train['shape'] == s)][features + [target]].dropna()
        X = sub[features].values
        y = sub[target].values.astype(int)
        base_rate = y.mean() * 100
        n_persist = y.sum()
        n_break = len(y) - n_persist

        print(f"\n  --- Horizon: {N}w ---")
        print(f"  Training samples: {len(y)} (persists: {n_persist}, breaks: {n_break})")
        print(f"  Base rate (majority class): {base_rate:.1f}%")

        if len(y) < 10:
            print(f"  SKIPPED — insufficient training data")
            continue

        # --- Logistic Regression ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, y)
        y_pred_lr = lr.predict(X_scaled)
        acc_lr = accuracy_score(y, y_pred_lr) * 100

        print(f"\n  LOGISTIC REGRESSION (training set):")
        print(f"    Accuracy: {acc_lr:.1f}%  (vs base rate {base_rate:.1f}%)")
        print(f"    {'Feature':<28} {'Coeff':>8} {'Direction':>10} {'Expected':>10} {'Match':>6}")
        print(f"    {'-'*65}")
        for fi, feat in enumerate(features):
            coeff = lr.coef_[0][fi]
            direction = 'positive' if coeff > 0 else 'negative'
            expected = EXPECTED_DIRECTIONS.get((feat, s), '?')
            match = 'YES' if direction == expected else ('NO' if expected != '?' else '?')
            print(f"    {feat:<28} {coeff:>+8.4f} {direction:>10} {expected:>10} {match:>6}")
        print(f"    Intercept: {lr.intercept_[0]:>+8.4f}")

        # Precision/recall
        if len(np.unique(y_pred_lr)) > 1:
            p1 = precision_score(y, y_pred_lr, pos_label=1, zero_division=0) * 100
            r1 = recall_score(y, y_pred_lr, pos_label=1, zero_division=0) * 100
            p0 = precision_score(y, y_pred_lr, pos_label=0, zero_division=0) * 100
            r0 = recall_score(y, y_pred_lr, pos_label=0, zero_division=0) * 100
            print(f"    Class 1 (persists): precision={p1:.1f}%, recall={r1:.1f}%")
            print(f"    Class 0 (breaks):   precision={p0:.1f}%, recall={r0:.1f}%")
        else:
            pred_class = y_pred_lr[0]
            print(f"    NOTE: Model predicts all class {pred_class} (trivial classifier)")

        # --- Random Forest with bootstrap stability ---
        n_resamples = 20
        rf_accs = []
        rf_importances = np.zeros((n_resamples, len(features)))
        for b in range(n_resamples):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            oob = np.array([i for i in range(len(X)) if i not in idx])
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=b, n_jobs=-1)
            rf.fit(X[idx], y[idx])
            rf_importances[b] = rf.feature_importances_
            if len(oob) > 0:
                rf_accs.append(accuracy_score(y[oob], rf.predict(X[oob])) * 100)

        # Fit final RF on full training data for later test evaluation
        rf_final = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
        rf_final.fit(X, y)
        y_pred_rf = rf_final.predict(X)
        acc_rf_train = accuracy_score(y, y_pred_rf) * 100

        mean_imp = rf_importances.mean(axis=0)
        std_imp = rf_importances.std(axis=0)

        print(f"\n  RANDOM FOREST (training set):")
        print(f"    In-sample accuracy: {acc_rf_train:.1f}%")
        if rf_accs:
            print(f"    Bootstrap OOB accuracy: {np.mean(rf_accs):.1f}% +/- {np.std(rf_accs):.1f}%")
        print(f"    Feature importances:")
        for fi, feat in enumerate(features):
            print(f"      {feat:<28} {mean_imp[fi]:.4f} +/- {std_imp[fi]:.4f}")

        # Store fitted models
        fitted_models[(s, N)] = {
            'lr': lr, 'rf': rf_final, 'scaler': scaler,
            'features': features, 'base_rate': base_rate,
            'train_n': len(y)
        }

# ============================================================
# SECTION 4: COX PROPORTIONAL HAZARDS MODELS
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 4: COX PROPORTIONAL HAZARDS MODELS — TRAINING SET")
print("=" * 80)
print("\nFitting Cox PH models on training-set episodes (2017-2024).")
print("Target: time-to-regime-break. Covariates: shape-specific validated features.")

cox_models = {}

for s in MODELABLE_SHAPES:
    features = SHAPE_FEATURES[s]
    sub = ep_train[ep_train['shape'] == s].copy()
    sub = sub[['duration_weeks', 'censored'] + features].dropna()

    if len(sub) < 10:
        print(f"\nShape {s} ({SHAPE_NAMES[s]}): SKIPPED — only {len(sub)} episodes")
        continue

    # Event observed = NOT censored (1 = regime broke, 0 = still ongoing)
    sub['event'] = (~sub['censored']).astype(int)

    print(f"\n{'='*60}")
    print(f"SHAPE {s} — {SHAPE_NAMES[s]}")
    print(f"Features: {features}")
    print(f"Episodes: {len(sub)} (events: {sub['event'].sum()}, censored: {sub['censored'].sum()})")
    print(f"{'='*60}")

    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(sub[['duration_weeks', 'event'] + features],
                duration_col='duration_weeks', event_col='event')

        print(f"\n  Cox PH Model Summary:")
        print(f"  {'Covariate':<28} {'Coeff':>8} {'HR':>8} {'p-value':>9} {'Direction':>10} {'Expected':>10} {'Match':>6}")
        print(f"  {'-'*82}")
        for feat in features:
            coeff = cph.params_[feat]
            hr = np.exp(coeff)
            p = cph.summary.loc[feat, 'p']
            # For Cox: positive coeff = higher hazard = shorter survival = LESS persistence
            # So for persistence direction: positive Cox coeff = negative persistence effect
            cox_direction = 'increases hazard' if coeff > 0 else 'decreases hazard'

            # Expected: if Phase A says higher value -> higher persistence,
            # then Cox should show NEGATIVE coeff (lower hazard = longer survival)
            expected_persist = EXPECTED_DIRECTIONS.get((feat, s), '?')
            if expected_persist == 'positive':
                expected_cox = 'decreases hazard'  # supports persistence -> lower hazard
            elif expected_persist == 'negative':
                expected_cox = 'increases hazard'   # hurts persistence -> higher hazard
            else:
                expected_cox = '?'

            match = 'YES' if cox_direction == expected_cox else ('NO' if expected_cox != '?' else '?')
            print(f"  {feat:<28} {coeff:>+8.4f} {hr:>8.3f} {p:>9.4f} {cox_direction:>18} {expected_cox:>18} {match:>6}")

        print(f"\n  Concordance index (train): {cph.concordance_index_:.3f}")
        cox_models[s] = cph

    except Exception as e:
        print(f"  Cox model failed: {e}")

# ============================================================
# SECTION 5: SHAPES 0.1 AND 0.2 — BASE RATE ONLY
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 5: NON-MODELABLE SHAPES — BASE RATE PREDICTIONS")
print("=" * 80)

print(f"""
Shape 0.1 — Mild Contango
  Status: No validated driver found in Phase A. No feature-based model built.
  Prediction method: Unconditional historical base rate from training set.
""")

for s, reason in [('0.1', 'No validated driver'), ('0.2', 'Insufficient data (n too small)')]:
    print(f"  Shape {s} — {SHAPE_NAMES[s]} (reason: {reason})")
    for N in [4, 12]:
        target = f'persists_{N}w'
        sub = train[(train['shape'] == s)][target].dropna()
        rate = sub.mean() * 100 if len(sub) > 0 else 0
        n = len(sub)
        print(f"    {N}w base rate: {rate:.1f}% (n={n})")
    print()

print("  These base rates ARE the standing prediction for these shapes.")
print("  Shape 0.2 (Steep Backwardation, n=26 total panel): flagged as a known gap")
print("  pending more data accumulation, not a finding.")

# ============================================================
# SECTION 6: FINAL EVALUATION ON HELD-OUT 2025-2026 TEST SET
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 6: FINAL EVALUATION ON HELD-OUT TEST SET (2025-2026)")
print("=" * 80)
print("\nLoading locked test set and running every fitted model exactly once.")
print("No iteration on models after seeing these results.")

# Reload test set from locked file to prove we use the saved version
test_locked = pd.read_csv(test_lock_path, index_col=0, dtype={'shape': str})
test_locked['week_end_date'] = pd.to_datetime(test_locked['week_end_date'])
# Ensure numeric columns are numeric
for col in ['persists_4w', 'persists_12w', 'usd_myr', 'enso_oni', 'stock_to_usage_ratio',
            'crude_oil_price', 'palm_soy_spread', 'production_yoy_pct']:
    if col in test_locked.columns:
        test_locked[col] = pd.to_numeric(test_locked[col], errors='coerce')
print(f"\nTest set loaded: {len(test_locked)} weeks, {test_locked['week_end_date'].min().date()} to {test_locked['week_end_date'].max().date()}")

print(f"\n{'='*90}")
print(f"{'Shape':<6} {'Hz':>3} {'Model':>5} {'Acc%':>6} {'Prec1':>6} {'Rec1':>6} {'Prec0':>6} {'Rec0':>6} {'Base%':>6} {'BaseAcc':>8} {'vs Base':>8}")
print(f"{'='*90}")

test_results = []

for s in MODELABLE_SHAPES:
    features = SHAPE_FEATURES[s]

    for N in [4, 12]:
        key = (s, N)
        if key not in fitted_models:
            continue

        model_info = fitted_models[key]
        lr = model_info['lr']
        rf = model_info['rf']
        scaler = model_info['scaler']
        train_base_rate = model_info['base_rate']

        target = f'persists_{N}w'
        sub_test = test_locked[(test_locked['shape'] == s)][features + [target]].dropna()

        if len(sub_test) == 0:
            print(f"{s:<6} {N:>3}w   — NO TEST DATA —")
            continue

        X_test = sub_test[features].values
        y_test = sub_test[target].values.astype(int)
        test_base_rate = y_test.mean() * 100
        n_test = len(y_test)

        # Baseline: always predict majority class from training
        baseline_pred = 1 if train_base_rate >= 50 else 0
        baseline_acc = accuracy_score(y_test, [baseline_pred] * n_test) * 100

        # Logistic Regression
        X_test_scaled = scaler.transform(X_test)
        y_pred_lr = lr.predict(X_test_scaled)
        acc_lr = accuracy_score(y_test, y_pred_lr) * 100

        if len(np.unique(y_pred_lr)) > 1:
            p1_lr = precision_score(y_test, y_pred_lr, pos_label=1, zero_division=0) * 100
            r1_lr = recall_score(y_test, y_pred_lr, pos_label=1, zero_division=0) * 100
            p0_lr = precision_score(y_test, y_pred_lr, pos_label=0, zero_division=0) * 100
            r0_lr = recall_score(y_test, y_pred_lr, pos_label=0, zero_division=0) * 100
        else:
            p1_lr = r1_lr = p0_lr = r0_lr = 0.0

        vs_base_lr = acc_lr - baseline_acc
        trivial_lr = " [all-1]" if np.all(y_pred_lr == 1) else (" [all-0]" if np.all(y_pred_lr == 0) else "")
        print(f"{s:<6} {N:>3}w {'LR':>5} {acc_lr:>5.1f}% {p1_lr:>5.1f}% {r1_lr:>5.1f}% {p0_lr:>5.1f}% {r0_lr:>5.1f}% {baseline_acc:>5.1f}% {baseline_acc:>7.1f}% {vs_base_lr:>+7.1f}pp{trivial_lr}")

        # Random Forest
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf) * 100

        if len(np.unique(y_pred_rf)) > 1:
            p1_rf = precision_score(y_test, y_pred_rf, pos_label=1, zero_division=0) * 100
            r1_rf = recall_score(y_test, y_pred_rf, pos_label=1, zero_division=0) * 100
            p0_rf = precision_score(y_test, y_pred_rf, pos_label=0, zero_division=0) * 100
            r0_rf = recall_score(y_test, y_pred_rf, pos_label=0, zero_division=0) * 100
        else:
            p1_rf = r1_rf = p0_rf = r0_rf = 0.0

        vs_base_rf = acc_rf - baseline_acc
        trivial_rf = " [all-1]" if np.all(y_pred_rf == 1) else (" [all-0]" if np.all(y_pred_rf == 0) else "")
        print(f"{s:<6} {N:>3}w {'RF':>5} {acc_rf:>5.1f}% {p1_rf:>5.1f}% {r1_rf:>5.1f}% {p0_rf:>5.1f}% {r0_rf:>5.1f}% {baseline_acc:>5.1f}% {baseline_acc:>7.1f}% {vs_base_rf:>+7.1f}pp{trivial_rf}")

        test_results.append({
            'shape': s, 'horizon': N,
            'n_test': n_test, 'test_base_rate': test_base_rate,
            'train_base_rate': train_base_rate,
            'baseline_acc': baseline_acc,
            'lr_acc': acc_lr, 'rf_acc': acc_rf,
            'lr_vs_base': vs_base_lr, 'rf_vs_base': vs_base_rf,
        })

# Base rate shapes on test set
print(f"\n{'='*60}")
print(f"BASE-RATE-ONLY SHAPES ON TEST SET")
print(f"{'='*60}")

for s in ['0.1', '0.2']:
    for N in [4, 12]:
        target = f'persists_{N}w'
        # Training base rate
        tr_sub = train[(train['shape'] == s)][target].dropna()
        tr_rate = tr_sub.mean() * 100 if len(tr_sub) > 0 else 0
        # Test actual
        te_sub = test_locked[(test_locked['shape'] == s)][target].dropna()
        if len(te_sub) == 0:
            print(f"  Shape {s} {N}w: no test data")
            continue
        te_rate = te_sub.mean() * 100
        baseline_pred = 1 if tr_rate >= 50 else 0
        baseline_acc = accuracy_score(te_sub.values.astype(int), [baseline_pred] * len(te_sub)) * 100
        print(f"  Shape {s} ({SHAPE_NAMES[s]}) {N}w: train base={tr_rate:.1f}%, test actual={te_rate:.1f}%, "
              f"baseline acc={baseline_acc:.1f}% (n={len(te_sub)})")

# --- Cox model test evaluation ---
print(f"\n{'='*60}")
print(f"COX MODEL — TEST SET EVALUATION")
print(f"{'='*60}")

for s in MODELABLE_SHAPES:
    if s not in cox_models:
        continue
    cph = cox_models[s]
    features = SHAPE_FEATURES[s]
    sub_test = ep_test[ep_test['shape'] == s].copy()
    sub_test = sub_test[['duration_weeks', 'censored'] + features].dropna()

    if len(sub_test) < 2:
        print(f"\n  Shape {s}: insufficient test episodes ({len(sub_test)})")
        continue

    sub_test['event'] = (~sub_test['censored']).astype(int)
    c_index = cph.score(sub_test[['duration_weeks', 'event'] + features],
                         scoring_method='concordance_index')

    # Predicted vs actual median survival
    # For training
    tr_sub = ep_train[ep_train['shape'] == s]
    tr_uncensored = tr_sub[~tr_sub['censored']]['duration_weeks']
    tr_median = tr_uncensored.median() if len(tr_uncensored) > 0 else np.nan

    te_uncensored = sub_test[sub_test['event'] == 1]['duration_weeks']
    te_median = te_uncensored.median() if len(te_uncensored) > 0 else np.nan

    print(f"\n  Shape {s} ({SHAPE_NAMES[s]}):")
    print(f"    Test episodes: {len(sub_test)} (events: {sub_test['event'].sum()}, censored: {(sub_test['event']==0).sum()})")
    print(f"    Concordance index (test): {c_index:.3f}")
    print(f"    Training median duration (uncensored): {tr_median:.1f}w")
    print(f"    Test median duration (uncensored): {te_median:.1f}w" if pd.notna(te_median) else f"    Test median duration: N/A (all censored or too few)")

# ============================================================
# SECTION 7: FEATURE DIRECTION CHECK ON TEST SET
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 7: FEATURE DIRECTION STABILITY — TRAIN vs TEST")
print("=" * 80)
print("\nDoes the validated direction of each feature hold up on 2025-2026 data?")
print("Method: same tercile/group split as Phase A, check if persistence gap direction matches.\n")

direction_flags = []

for s in MODELABLE_SHAPES:
    features = SHAPE_FEATURES[s]
    print(f"\n  Shape {s} ({SHAPE_NAMES[s]}):")

    for feat in features:
        for N in [4, 12]:
            target = f'persists_{N}w'
            # Training gap
            tr_sub = train[(train['shape'] == s)][[feat, target]].dropna()
            if len(tr_sub) < 10:
                continue
            q33_tr = tr_sub[feat].quantile(0.333)
            q67_tr = tr_sub[feat].quantile(0.667)
            tr_high = tr_sub[tr_sub[feat] >= q67_tr][target].mean() * 100
            tr_low = tr_sub[tr_sub[feat] <= q33_tr][target].mean() * 100
            tr_gap = tr_high - tr_low

            # Test gap — use test median split (training quantiles may leave empty groups
            # given the small test sample per shape)
            te_sub = test_locked[(test_locked['shape'] == s)][[feat, target]].dropna()
            if len(te_sub) < 4:
                te_gap = np.nan
                te_high = te_low = np.nan
                n_te_high = n_te_low = 0
            else:
                te_med = te_sub[feat].median()
                te_h_mask = te_sub[feat] > te_med
                te_l_mask = te_sub[feat] <= te_med
                n_te_high = te_h_mask.sum()
                n_te_low = te_l_mask.sum()
                te_high = te_sub.loc[te_h_mask, target].mean() * 100 if n_te_high > 0 else np.nan
                te_low = te_sub.loc[te_l_mask, target].mean() * 100 if n_te_low > 0 else np.nan
                te_gap = (te_high - te_low) if pd.notna(te_high) and pd.notna(te_low) else np.nan

            # Direction match
            if pd.notna(te_gap):
                same_sign = (tr_gap > 0 and te_gap > 0) or (tr_gap < 0 and te_gap < 0) or (tr_gap == 0 and te_gap == 0)
                status = 'HOLDS' if same_sign else 'FLIPPED'
                weakened = abs(te_gap) < abs(tr_gap) * 0.5 if same_sign and abs(tr_gap) > 5 else False
                if weakened:
                    status = 'WEAKENED'
            else:
                status = 'INSUFFICIENT DATA'

            flag = '***' if status in ['FLIPPED', 'WEAKENED'] else ''
            te_gap_str = f"{te_gap:+.1f}pp" if pd.notna(te_gap) else "N/A"
            print(f"    {feat:<28} {N:>2}w  train gap={tr_gap:>+6.1f}pp  test gap={te_gap_str:>10}  "
                  f"(n_hi={n_te_high}, n_lo={n_te_low})  {status} {flag}")

            if status in ['FLIPPED', 'WEAKENED']:
                direction_flags.append({
                    'shape': s, 'feature': feat, 'horizon': f'{N}w',
                    'train_gap': tr_gap, 'test_gap': te_gap, 'status': status
                })

if direction_flags:
    print(f"\n  *** FLAGGED DIRECTION ISSUES ({len(direction_flags)}) ***")
    for f in direction_flags:
        print(f"    Shape {f['shape']}, {f['feature']}, {f['horizon']}: "
              f"train={f['train_gap']:+.1f}pp -> test={f['test_gap']:+.1f}pp ({f['status']})")
else:
    print(f"\n  No direction flips or substantial weakenings detected.")

# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "=" * 80)
print("PHASE B SUMMARY")
print("=" * 80)

print(f"""
Models built:
  - 6 fixed-horizon classifiers (3 shapes x 2 horizons), each with LR + RF
  - 3 Cox PH survival models (one per modelable shape)
  - 2 base-rate-only shapes (0.1 Mild Contango, 0.2 Steep Backwardation)

Train/test split:
  - Train: 2017-W01 to 2024-W52 ({len(train)} weeks)
  - Test:  2025-W01 to 2026-W26 ({len(test_locked)} weeks), locked before model fitting

Feature sets (from Phase A, not re-derived):
  - Shape 0.0 (Contango):          usd_myr, enso_oni
  - Shape 1   (Backwardation):     usd_myr, enso_oni, stock_to_usage_ratio
  - Shape 2   (Flat):              enso_oni, crude_oil_price, palm_soy_spread, production_yoy_pct
  - Shape 0.1 (Mild Contango):     base rate only (no validated driver)
  - Shape 0.2 (Steep Backwardation): base rate only (insufficient data, n=26)
""")

print("=" * 80)
print("PHASE B: PERSISTENCE MODEL BUILD — COMPLETE")
print("=" * 80)
print(f"Output saved to: {os.path.abspath(OUTPUT_PATH)}")

file_buf.close()
