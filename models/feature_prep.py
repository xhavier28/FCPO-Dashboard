"""
Feature preparation for PM and TM models.

PM (Persistence Model) and TM (Transition Model) use different transforms
of the same raw data sources. This is a deliberate modeling choice:
- PM operates on weekly ISO panels (shape=last, spot=last, weekly-mean externals)
- TM operates on daily panels with merge_asof for MPOB/ENSO/macro
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE = os.path.join(os.path.dirname(__file__), '..')
RAW = os.path.join(BASE, 'Raw Data')

# TM feature list (fixed from validated notebook)
TM_FEATURES = [
    'stock_pct', 'prod_mom_3m', 'prod_yoy',
    'export_yoy', 'oni', 'usd_myr_chg_4w',
    'palm_soy_chg_4w', 'days_in_shape',
    'prior_shape_enc', 'month',
    'stock_prod_interaction',
]

# ──────────────────────────────────────────────────────────────
# Section A: Raw Data Loaders (shared by PM and TM)
# ──────────────────────────────────────────────────────────────

def load_daily_shape_log():
    """Load daily shape log used by PM. Source: regime_identification outputs."""
    return pd.read_csv(
        os.path.join(BASE, 'research', '01. regime_identification', 'outputs', 'daily_shape_log.csv'),
        dtype={'shape': str}, parse_dates=['date'])


def load_enriched_shape_log():
    """Load enriched shape log used by TM. Has prior_shape, days_in_shape columns."""
    return pd.read_csv(
        os.path.join(BASE, 'research', '03. validation_analysis', 'shape_log_enriched.csv'),
        dtype={'shape': str, 'prior_shape': str}, parse_dates=['date'])


def load_mpob_stock():
    """Load MPOB stock data from Excel. Returns DataFrame[date, stock]."""
    df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'FCPO Stock 3Y.xlsx'))
    df.columns = ['date', 'stock']
    df = df.iloc[1:]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['stock'] = pd.to_numeric(df['stock'], errors='coerce')
    return df.dropna(subset=['stock']).sort_values('date').reset_index(drop=True)


def load_mpob_production():
    """Load MPOB production data from Excel. Returns DataFrame[date, production]."""
    df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'MPOB Production 3Y.xlsx'))
    df.columns = ['date', 'production']
    df = df.iloc[1:]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['production'] = pd.to_numeric(df['production'], errors='coerce')
    return df.dropna(subset=['production']).sort_values('date').reset_index(drop=True)


def load_mpob_export():
    """Load MPOB export data from Excel. Returns DataFrame[date, export]."""
    df = pd.read_excel(os.path.join(RAW, 'Stock and Production', 'MPOB Export 3Y.xlsx'))
    df.columns = ['date', 'export']
    df = df.iloc[1:]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['export'] = pd.to_numeric(df['export'], errors='coerce')
    return df.dropna(subset=['export']).sort_values('date').reset_index(drop=True)


def load_usd_myr():
    """Load USD/MYR FX rate. Unix timestamp CSV. Returns DataFrame[date, usd_myr]."""
    df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'FX_IDC_USDMYR, 1D_1227e.csv'))
    df['date'] = pd.to_datetime(df['time'], unit='s')
    return df[['date', 'close']].rename(columns={'close': 'usd_myr'}).sort_values('date')


def load_palm_soy_spread():
    """Load palm-soy spread. Unix timestamp CSV. Returns DataFrame[date, palm_soy_spread]."""
    df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data',
                                   'MYX_DLY_FCPO1!_2_CBOT_DL_ZL1!, 1D_61912.csv'))
    df['date'] = pd.to_datetime(df['time'], unit='s')
    return df[['date', 'close']].rename(columns={'close': 'palm_soy_spread'}).sort_values('date')


def load_crude_oil():
    """Load crude oil price. Unix timestamp CSV. Returns DataFrame[date, crude_oil_price]."""
    df = pd.read_csv(os.path.join(RAW, 'Variable Analysis Extra Data', 'NYMEX_DL_CL1!, 1D_84001.csv'))
    df['date'] = pd.to_datetime(df['time'], unit='s')
    return df[['date', 'close']].rename(columns={'close': 'crude_oil_price'}).sort_values('date')


def load_enso_oni():
    """Load ENSO ONI index from ASCII file. Season->month mapping. Returns DataFrame[date, oni]."""
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
    season_to_month = {
        'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
        'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12,
    }
    oni_df['month'] = oni_df['season'].map(season_to_month)
    oni_df['date'] = pd.to_datetime(oni_df['year'].astype(str) + '-' + oni_df['month'].astype(str) + '-01')
    return oni_df[['date', 'oni']].sort_values('date').reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Section B: PM-Specific Transforms (weekly aggregation)
# ──────────────────────────────────────────────────────────────

def build_pm_weekly_panel():
    """
    Build the weekly panel used by the Persistence Model.

    Returns a DataFrame indexed by week_key (e.g. '2025-W03') with columns:
      shape, spot, week_end_date, usd_myr, palm_soy_spread, crude_oil_price,
      stock_to_usage_ratio, production_yoy_pct, enso_oni,
      persists_4w, persists_12w
    """
    # Load raw data
    shape_log = load_daily_shape_log()
    stock_df = load_mpob_stock()
    prod_df = load_mpob_production()
    export_df = load_mpob_export()
    fx_df = load_usd_myr()
    ps_df = load_palm_soy_spread()
    cl_df = load_crude_oil()
    oni_df = load_enso_oni()

    # Build daily panel from shape log
    daily = shape_log[['date', 'shape', 'M1']].copy().rename(columns={'M1': 'spot'})
    daily = daily.set_index('date').sort_index()
    daily = daily['2017-01-01':]
    daily['iso_year'] = daily.index.isocalendar().year.values
    daily['iso_week'] = daily.index.isocalendar().week.values
    daily['week_key'] = daily['iso_year'].astype(str) + '-W' + daily['iso_week'].astype(str).str.zfill(2)

    # Weekly aggregation
    weekly = daily.groupby('week_key').agg(
        shape=('shape', 'last'), spot=('spot', 'last'),
        week_end_date=('spot', lambda x: x.index[-1])
    ).sort_values('week_end_date')
    weekly['shape_prev'] = weekly['shape'].shift(1)

    # Merge external daily -> weekly (weekly mean)
    # Deduplicate external series (keep last) before reindexing
    fx_daily = fx_df.set_index('date')['usd_myr'].sort_index()
    fx_daily = fx_daily[~fx_daily.index.duplicated(keep='last')]
    daily['usd_myr'] = fx_daily.reindex(daily.index, method='ffill')
    weekly['usd_myr'] = daily.groupby('week_key')['usd_myr'].mean().reindex(weekly.index)

    ps_daily = ps_df.set_index('date')['palm_soy_spread'].sort_index()
    ps_daily = ps_daily[~ps_daily.index.duplicated(keep='last')]
    daily['palm_soy'] = ps_daily.reindex(daily.index, method='ffill')
    weekly['palm_soy_spread'] = daily.groupby('week_key')['palm_soy'].mean().reindex(weekly.index)

    cl_daily = cl_df.set_index('date')['crude_oil_price'].sort_index()
    cl_daily = cl_daily[~cl_daily.index.duplicated(keep='last')]
    daily['crude'] = cl_daily.reindex(daily.index, method='ffill')
    weekly['crude_oil_price'] = daily.groupby('week_key')['crude'].mean().reindex(weekly.index)

    # MPOB monthly -> weekly (last-known <= week_end)
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

    # ENSO ONI -> weekly (last-known <= week_end)
    oni_series = oni_df.set_index('date')['oni']
    vals = []
    for _, row in weekly.iterrows():
        dt = row['week_end_date']
        mask = oni_series.index <= dt
        vals.append(oni_series[mask].iloc[-1] if mask.any() else np.nan)
    weekly['enso_oni'] = vals

    # Targets: does shape persist N weeks forward?
    for N in [4, 12]:
        weekly[f'shape_plus_{N}w'] = weekly['shape'].shift(-N)
        weekly[f'persists_{N}w'] = (weekly['shape'] == weekly[f'shape_plus_{N}w']).astype(int)
        weekly.loc[weekly[f'shape_plus_{N}w'].isna(), f'persists_{N}w'] = np.nan

    weekly = weekly[weekly['week_end_date'] >= '2017-01-01']
    return weekly


# ──────────────────────────────────────────────────────────────
# Section C: TM-Specific Transforms (daily, merge_asof)
# ──────────────────────────────────────────────────────────────

def build_tm_daily_panel():
    """
    Build the daily panel used by the Transition Model.

    STANDING RULE: stock_pct is computed on the full 2017-present panel
    internally (rank with pct=True). This function does NOT accept a
    pre-filtered slice — it loads the full panel, computes percentiles,
    then returns the complete result.

    Returns a DataFrame with columns matching TM_FEATURES plus:
      date, shape, prior_shape, days_in_shape, target_1w, target_2w
    """
    # Load enriched shape log (full history)
    df = load_enriched_shape_log()
    df = df[df['date'] >= '2017-01-01'].sort_values('date').reset_index(drop=True)

    # Load MPOB
    stock = load_mpob_stock().rename(columns={'stock': 'stock_raw'})
    prod = load_mpob_production().rename(columns={'production': 'production_raw'})
    exp = load_mpob_export().rename(columns={'export': 'export_raw'})

    # Load macro
    usdmyr = load_usd_myr().rename(columns={'usd_myr': 'usd_myr'})
    palmsoy = load_palm_soy_spread().rename(columns={'palm_soy_spread': 'palm_soy'})
    crude = load_crude_oil().rename(columns={'crude_oil_price': 'crude_oil'})

    # Load ENSO
    oni_df = load_enso_oni()
    enso_long = oni_df[['date', 'oni']].dropna().sort_values('date')

    # merge_asof backward for MPOB
    for src, col in [(stock, 'stock_raw'), (prod, 'production_raw'), (exp, 'export_raw')]:
        df = pd.merge_asof(
            df.sort_values('date'),
            src.sort_values('date'),
            on='date', direction='backward')

    # merge_asof backward for ENSO
    df = pd.merge_asof(
        df.sort_values('date'),
        enso_long.sort_values('date'),
        on='date', direction='backward')

    # merge_asof backward for macro
    for src, col in [(usdmyr, 'usd_myr'), (palmsoy, 'palm_soy'), (crude, 'crude_oil')]:
        df = pd.merge_asof(
            df.sort_values('date'),
            src.sort_values('date'),
            on='date', direction='backward')

    # Derived variables — computed on FULL panel (standing rule for stock_pct)
    df['stock_pct'] = df['stock_raw'].rank(pct=True)
    df['prod_mom_3m'] = df['production_raw'].pct_change(63) * 100
    df['prod_yoy'] = df['production_raw'].pct_change(252) * 100
    df['export_yoy'] = df['export_raw'].pct_change(252) * 100
    df['usd_myr_chg_4w'] = df['usd_myr'].pct_change(20) * 100
    df['palm_soy_chg_4w'] = df['palm_soy'].pct_change(20) * 100
    df['crude_chg_4w'] = df['crude_oil'].pct_change(20) * 100
    df['month'] = df['date'].dt.month

    # Prior shape encoding (fitted on full panel)
    le = LabelEncoder()
    df['prior_shape_enc'] = le.fit_transform(df['prior_shape'].fillna('unknown'))

    # Stock-production interaction quadrant
    df['stock_state'] = np.where(df['stock_pct'] >= df['stock_pct'].median(), 'High', 'Low')
    df['prod_state'] = np.where(df['prod_mom_3m'] >= 0, 'Rising', 'Falling')
    df['stock_prod_quad'] = df['stock_state'] + '_' + df['prod_state']
    quad_map = {'High_Falling': 3, 'Low_Rising': 2, 'Low_Falling': 1, 'High_Rising': 0}
    df['stock_prod_interaction'] = df['stock_prod_quad'].map(quad_map).fillna(0)

    # Targets
    df = df.sort_values('date').reset_index(drop=True)
    df['target_1w'] = df['shape'].shift(-5)   # +5 trading days
    df['target_2w'] = df['shape'].shift(-10)  # +10 trading days

    # Store the fitted LabelEncoder as an attribute for downstream use
    df.attrs['prior_shape_le'] = le

    return df
