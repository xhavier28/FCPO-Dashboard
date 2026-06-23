"""
Part 0 — Extract centroids and stock terciles from the same data pipeline
as Curve_Regime_Research.ipynb. Reproduces Cells 2-7, 59, H3, S1, S3, N0
then computes mean z-score profiles per final_shape as centroids.
"""
import pandas as pd
import numpy as np
import glob
import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
months = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']

# --- Cell 2: Load daily settlement prices ---
csv_files = glob.glob('Raw Data/Term Structure/**/*.csv', recursive=True)
print(f"Found {len(csv_files)} contract CSV files")

frames = []
for f in csv_files:
    basename = os.path.basename(f)
    parts = basename.replace('FCPO ', '').replace('_Daily.csv', '').split()
    if not parts:
        continue
    token = parts[0]
    mmm = token[:3]
    yy = token[3:]
    if mmm not in MONTH_MAP or not yy.isdigit():
        continue
    month_num = MONTH_MAP[mmm]
    year_full = 2000 + int(yy)
    expiry_order = year_full * 100 + month_num
    contract_label = f"{mmm}{yy}"
    tmp = pd.read_csv(f)
    tmp['date'] = pd.to_datetime(tmp['Timestamp (UTC)']).dt.normalize()
    tmp = tmp[['date', 'Close']].dropna(subset=['Close']).copy()
    tmp = tmp.rename(columns={'Close': 'price'})
    tmp['contract'] = contract_label
    tmp['expiry_order'] = expiry_order
    frames.append(tmp)

raw = pd.concat(frames, ignore_index=True)
print(f"Total rows: {len(raw):,}, Contracts: {raw['contract'].nunique()}")

wide = raw.pivot_table(index='date', columns='contract', values='price', aggfunc='last').sort_index()
contract_order = raw.drop_duplicates('contract')[['contract', 'expiry_order']].set_index('contract')['expiry_order'].to_dict()

records = []
for date, row in wide.iterrows():
    avail = row.dropna()
    if len(avail) < 2:
        continue
    date_order = date.year * 100 + date.month
    future = {c: contract_order[c] for c in avail.index if c in contract_order and contract_order[c] >= date_order}
    if len(future) < 2:
        continue
    sorted_contracts = sorted(future.keys(), key=lambda c: contract_order[c])
    rec = {'date': date}
    for i, label in enumerate(months):
        if i < len(sorted_contracts):
            rec[label] = avail[sorted_contracts[i]]
        else:
            rec[label] = np.nan
    records.append(rec)

df = pd.DataFrame(records).set_index('date').sort_index()
df = df.dropna(subset=months, thresh=4)
print(f"Price df: {df.shape}")

# --- Cell 3: Stock data ---
stock_raw = pd.read_excel('Raw Data/Stock and Production/FCPO Stock 3Y.xlsx', skiprows=[1])
stock_raw['Date'] = pd.to_datetime(stock_raw['Date'])
stock = stock_raw.set_index('Date')[['MYPOMS-TPO (COMM_LAST)']].rename(
    columns={'MYPOMS-TPO (COMM_LAST)': 'stock_mpob'}
).sort_index()
stock = stock.resample('ME').last().resample('D').ffill()
df = df.join(stock, how='left')

# --- Cell 4: Spreads and spot return ---
df['spot'] = df['M1']
df['spot_ret_1d'] = df['spot'].pct_change() * 100
roll_mask = df['spot_ret_1d'].abs() > 5.0
roll_indices = df.index[roll_mask]
drop_indices = set()
all_idx = df.index.tolist()
for ri in roll_indices:
    pos = all_idx.index(ri)
    for offset in range(0, 4):
        if pos + offset < len(all_idx):
            drop_indices.add(all_idx[pos + offset])
df = df.drop(index=list(drop_indices), errors='ignore')

# --- Cell 6: Regime ---
df['G1'] = (df['M1'] + df['M2']) / 2
df['G3'] = (df['M5'] + df['M6']) / 2
df['G1_G3'] = df['G1'] - df['G3']
window = 756
df['G1_G3_pct'] = df['G1_G3'].rolling(window, min_periods=window // 2).rank() / window

def regime_label(pct):
    if pd.isna(pct):
        return 'Unknown'
    elif pct <= 0.33:
        return 'Contango'
    elif pct <= 0.67:
        return 'Neutral'
    else:
        return 'Backwardation'

df['regime'] = df['G1_G3_pct'].apply(regime_label)

# --- Cell 7: Stock tercile ---
df['stock_pct'] = df['stock_mpob'].rolling(window, min_periods=window // 2).rank() / window

def stock_tercile(pct):
    if pd.isna(pct):
        return 'Unknown'
    elif pct <= 0.33:
        return 'Low'
    elif pct <= 0.67:
        return 'Mid'
    else:
        return 'High'

df['stock_level'] = df['stock_pct'].apply(stock_tercile)

# --- Cell 8: Spot cat ---
bins = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
labels_cat = ['Large down', 'Small down', 'Flat', 'Small up', 'Large up']
df['spot_cat'] = pd.cut(df['spot_ret_1d'], bins=bins, labels=labels_cat)

# --- Cell H1: curve_raw and curve_norm ---
curve_raw = df[months + ['stock_mpob', 'stock_level', 'spot_cat', 'regime', 'spot_ret_1d']].dropna(subset=months)
curve_norm = curve_raw[months].apply(lambda row: (row - row.mean()) / row.std(), axis=1)
nan_mask = curve_norm.isna().any(axis=1)
if nan_mask.sum() > 0:
    print(f"Dropped {nan_mask.sum()} rows with identical prices (std=0)")
    curve_norm = curve_norm.dropna()
    curve_raw = curve_raw.loc[curve_norm.index]
print(f"curve_raw: {curve_raw.shape}, curve_norm: {curve_norm.shape}")

# --- Cell H3: KMeans (k=3, random_state=42) ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

sil_scores = {}
for k in range(3, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(curve_norm)
    sil_scores[k] = silhouette_score(curve_norm, labels)
best_k = max(sil_scores, key=sil_scores.get)
print(f"Best k: {best_k} (silhouette = {sil_scores[best_k]:.4f})")

km_best = KMeans(n_clusters=best_k, random_state=42, n_init=15)
shape_labels = km_best.fit_predict(curve_norm)
curve_raw = curve_raw.copy()
curve_raw['shape_id'] = shape_labels
print("Shape sizes:", curve_raw['shape_id'].value_counts().sort_index().to_dict())

# --- Cell S1: Sub-cluster Shape 0 ---
shape0_idx = curve_raw[curve_raw['shape_id'] == 0].index
shape0_norm = curve_norm.loc[shape0_idx]
sub_sil = {}
for k in range(2, 5):
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    sub_labels = km.fit_predict(shape0_norm)
    if len(set(sub_labels)) > 1:
        sub_sil[k] = silhouette_score(shape0_norm, sub_labels)
best_sub_k = max(sub_sil, key=sub_sil.get)
print(f"Shape 0 sub-k: {best_sub_k} (silhouette = {sub_sil[best_sub_k]:.4f})")

km_sub = KMeans(n_clusters=best_sub_k, random_state=42, n_init=15)
sub_labels = km_sub.fit_predict(shape0_norm)
curve_raw.loc[shape0_idx, 'sub_shape_id'] = sub_labels
curve_raw['sub_shape_id'] = curve_raw['sub_shape_id'].fillna(-1).astype(int)

# --- Cell S3: final_shape ---
curve_raw['final_shape'] = curve_raw['shape_id'].astype(str)
for sid in range(best_sub_k):
    mask = (curve_raw['shape_id'] == 0) & (curve_raw['sub_shape_id'] == sid)
    curve_raw.loc[mask, 'final_shape'] = f'0.{sid}'
print("Final shape distribution:", curve_raw['final_shape'].value_counts().sort_index().to_dict())

# --- Cell N0: trusted window ---
trusted_start = '2017-01-01'
curve_raw_t = curve_raw[curve_raw.index >= trusted_start].copy()
curve_norm_t = curve_norm.loc[curve_raw_t.index]
curve_raw_t['stock_pct'] = df.loc[curve_raw_t.index, 'stock_pct']
print(f"Trusted window: {len(curve_raw_t)} rows, "
      f"{curve_raw_t.index.min().date()} to {curve_raw_t.index.max().date()}")

# ============================================================
# Part 0A — Extract centroids
# ============================================================
shapes_expected = ['0.0', '0.1', '0.2', '1', '2']
actual_shapes = sorted(curve_raw_t['final_shape'].unique())
print(f"\nShape labels in curve_raw_t: {actual_shapes}")
assert actual_shapes == shapes_expected, f"Shape mismatch: {actual_shapes} vs {shapes_expected}"

centroid_rows = []
for shape in shapes_expected:
    mask = curve_raw_t['final_shape'] == shape
    shape_norm = curve_norm_t.loc[mask]
    n = len(shape_norm)
    mean_profile = shape_norm.mean().values
    row = {'shape': shape}
    for m, val in zip(months, mean_profile):
        row[m] = round(float(val), 6)
    centroid_rows.append(row)
    print(f"  Shape {shape} (n={n}): {[round(v, 3) for v in mean_profile]}")

centroids_df = pd.DataFrame(centroid_rows)
print("\nCentroid table:")
print(centroids_df.to_string(index=False))
os.makedirs('Raw Data/Research', exist_ok=True)
centroids_df.to_csv('Raw Data/Research/shape_centroids.csv', index=False)
print("Saved shape_centroids.csv")

# ============================================================
# Part 0B — Stock tercile boundaries
# ============================================================
tercile_rows = []
for shape in shapes_expected:
    shape_data = curve_raw_t[curve_raw_t['final_shape'] == shape]
    if 'stock_pct' not in shape_data.columns or shape_data['stock_pct'].dropna().shape[0] < 9:
        svals = curve_raw_t['stock_pct'].dropna()
        source = 'population-wide (insufficient per-shape data)'
    else:
        svals = shape_data['stock_pct'].dropna()
        source = 'per-shape'
    tercile_rows.append({
        'shape': shape,
        'low_max': round(svals.quantile(1 / 3), 6),
        'mid_max': round(svals.quantile(2 / 3), 6),
        'source': source,
        'n': len(svals)
    })
tercile_df = pd.DataFrame(tercile_rows)
print("\nTercile table:")
print(tercile_df.to_string(index=False))
tercile_df.to_csv('Raw Data/Research/shape_stock_terciles.csv', index=False)
print("Saved shape_stock_terciles.csv")

# ============================================================
# Part 0C — Verify
# ============================================================
check_c = pd.read_csv('Raw Data/Research/shape_centroids.csv')
check_t = pd.read_csv('Raw Data/Research/shape_stock_terciles.csv')
print(f"\nCentroids: {check_c.shape}")
print(check_c)
print(f"\nTerciles: {check_t.shape}")
print(check_t)
assert check_c.shape[0] == 5, f"Expected 5 centroid rows, got {check_c.shape[0]}"
assert check_t.shape[0] == 5, f"Expected 5 tercile rows, got {check_t.shape[0]}"
print("\nBoth files verified: 5 rows each, clean shape labels.")
