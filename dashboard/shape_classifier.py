import pandas as pd
import numpy as np
import os
from datetime import date, timedelta

MONTHS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']

SHAPE_NAMES = {
    '0.0': 'Steep Backwardation',
    '0.1': 'Mild Backwardation, Rising Back',
    '0.2': 'Mixed Curve',
    '1':   'Deep Uniform Contango',
    '2':   'Steep Front Contango, Flattening Back',
}

SPOT_MOM_DOWN_THRESHOLD = -1.0
SPOT_MOM_UP_THRESHOLD = 1.0

CENTROIDS_PATH = 'Raw Data/Research/shape_centroids.csv'
TERCILES_PATH = 'Raw Data/Research/shape_stock_terciles.csv'
LOG_PATH = 'Raw Data/Research/daily_shape_log.csv'


def load_centroids():
    if not os.path.exists(CENTROIDS_PATH):
        raise FileNotFoundError(
            f"{CENTROIDS_PATH} not found. Run Part 0 in the research notebook first."
        )
    df = pd.read_csv(CENTROIDS_PATH, dtype={'shape': str})
    return df.set_index('shape')[MONTHS]


def load_stock_terciles():
    if not os.path.exists(TERCILES_PATH):
        raise FileNotFoundError(
            f"{TERCILES_PATH} not found. Run Part 0 in the research notebook first."
        )
    return pd.read_csv(TERCILES_PATH, dtype={'shape': str}).set_index('shape')


def classify_shape(m1_to_m6, centroids_df):
    """
    m1_to_m6: array-like of 6 prices, M1 through M6, for a single day.
    centroids_df: output of load_centroids().
    Returns the shape label string of the nearest centroid.
    """
    vals = np.array(m1_to_m6, dtype=float)
    if len(vals) != 6 or np.any(np.isnan(vals)):
        return None
    mean = vals.mean()
    std = vals.std()
    if std == 0:
        return None
    normalized = (vals - mean) / std

    distances = {}
    for shape_label, row in centroids_df.iterrows():
        centroid_vec = row.values.astype(float)
        distances[shape_label] = np.linalg.norm(normalized - centroid_vec)

    return min(distances, key=distances.get)


def classify_stock_tercile(stock_pct, shape, terciles_df):
    if stock_pct is None or pd.isna(stock_pct):
        return None
    if shape not in terciles_df.index:
        return None
    row = terciles_df.loc[shape]
    if stock_pct <= row['low_max']:
        return 'Low'
    elif stock_pct <= row['mid_max']:
        return 'Mid'
    else:
        return 'High'


def classify_spot_momentum(m1_today, m1_5_days_ago):
    if m1_5_days_ago is None or m1_5_days_ago == 0 or pd.isna(m1_5_days_ago):
        return None
    mom_pct = (m1_today / m1_5_days_ago - 1) * 100
    if mom_pct < SPOT_MOM_DOWN_THRESHOLD:
        return 'Down'
    elif mom_pct > SPOT_MOM_UP_THRESHOLD:
        return 'Up'
    else:
        return 'Flat'


def update_shape_log(daily_curve_df, stock_pct_series, log_path=LOG_PATH):
    """
    daily_curve_df: DataFrame indexed by date, columns M1-M6, the full available
                     history (same source as the existing Term Structure tab).
    stock_pct_series: Series indexed by date, the rolling stock percentile
                       (same source as the existing S Calculator tab).
    """
    centroids_df = load_centroids()
    terciles_df = load_stock_terciles()

    daily_curve_df = daily_curve_df.sort_index()

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path, parse_dates=['date']).set_index('date')
        last_logged_date = existing_log.index.max()
        new_dates = daily_curve_df.index[daily_curve_df.index > last_logged_date]
    else:
        existing_log = pd.DataFrame()
        new_dates = daily_curve_df.index

    if len(new_dates) == 0:
        return existing_log.reset_index() if len(existing_log) > 0 else existing_log

    new_rows = []
    for d in new_dates:
        row_prices = daily_curve_df.loc[d, MONTHS]
        if row_prices.isna().any():
            continue

        shape = classify_shape(row_prices.values, centroids_df)
        if shape is None:
            continue

        stock_pct_today = stock_pct_series.get(d, np.nan) if isinstance(stock_pct_series, dict) else (
            stock_pct_series.loc[d] if d in stock_pct_series.index else np.nan
        )
        stock_tercile = classify_stock_tercile(stock_pct_today, shape, terciles_df)

        loc = daily_curve_df.index.get_loc(d)
        if loc >= 5:
            m1_5_days_ago = daily_curve_df.iloc[loc - 5]['M1']
            spot_mom_cat = classify_spot_momentum(row_prices['M1'], m1_5_days_ago)
        else:
            spot_mom_cat = None

        new_row = {'date': d, 'shape': shape,
                   'stock_pct': stock_pct_today, 'stock_tercile': stock_tercile,
                   'spot_mom_cat': spot_mom_cat, 'month': d.month}
        for m in MONTHS:
            new_row[m] = row_prices[m]
        new_rows.append(new_row)

    new_log_df = pd.DataFrame(new_rows)
    if len(existing_log) > 0:
        full_log = pd.concat([existing_log.reset_index(), new_log_df], ignore_index=True)
    else:
        full_log = new_log_df

    full_log = full_log.drop_duplicates(subset='date').sort_values('date')
    full_log.to_csv(log_path, index=False)
    return full_log


def force_full_reclassify(daily_curve_df, stock_pct_series, log_path=LOG_PATH):
    if os.path.exists(log_path):
        os.remove(log_path)
    return update_shape_log(daily_curve_df, stock_pct_series, log_path)
