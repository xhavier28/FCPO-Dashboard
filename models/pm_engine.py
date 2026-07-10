"""
Persistence Model (PM) Engine.

Predicts whether the current curve shape will persist at 4-week and 12-week horizons.
Uses per-shape Logistic Regression with StandardScaler, trained on 2017-2024 weekly panel.

Modelable shapes: 0.0 (Contango), 1 (Backwardation), 2 (Flat)
Non-modelable shapes (0.1, 0.2): return unconditional training base rates.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .feature_prep import build_pm_weekly_panel, load_daily_shape_log

warnings.filterwarnings('ignore')

ALL_SHAPES = ['0.0', '0.1', '0.2', '1', '2']
SHAPE_NAMES = {
    '0.0': 'Contango', '0.1': 'Mild Contango', '0.2': 'Steep Backwardation',
    '1': 'Backwardation', '2': 'Flat',
}

SHAPE_FEATURES = {
    '0.0': ['usd_myr', 'enso_oni'],
    '1':   ['usd_myr', 'enso_oni', 'stock_to_usage_ratio'],
    '2':   ['enso_oni', 'crude_oil_price', 'palm_soy_spread', 'production_yoy_pct'],
}
MODELABLE_SHAPES = ['0.0', '1', '2']

TRAIN_CUTOFF = pd.Timestamp('2024-12-31')

# Module-level cache
_fitted_models = {}   # (shape, horizon) -> {'lr': model, 'scaler': scaler, ...}
_base_rates = {}      # (shape, horizon) -> float
_weekly_panel = None
_shape_log = None


def _ensure_models_loaded():
    """Lazy init: build weekly panel, fit LR + scaler per (shape, horizon)."""
    global _fitted_models, _base_rates, _weekly_panel, _shape_log

    if _fitted_models:
        return

    _weekly_panel = build_pm_weekly_panel()
    train = _weekly_panel[_weekly_panel['week_end_date'] <= TRAIN_CUTOFF].copy()

    # Fit modelable shapes
    for s in MODELABLE_SHAPES:
        features = SHAPE_FEATURES[s]
        for N in [4, 12]:
            target = f'persists_{N}w'
            sub = train[(train['shape'] == s)][features + [target]].dropna()
            X = sub[features].values
            y = sub[target].values.astype(int)

            if len(y) < 10:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_scaled, y)

            _fitted_models[(s, N)] = {
                'lr': lr,
                'scaler': scaler,
                'features': features,
                'base_rate': y.mean() * 100,
                'train_n': len(y),
            }

    # Compute base rates for all shapes (including non-modelable)
    for s in ALL_SHAPES:
        for N in [4, 12]:
            target = f'persists_{N}w'
            sub = train[(train['shape'] == s)][target].dropna()
            _base_rates[(s, N)] = sub.mean() * 100 if len(sub) > 0 else 0.0

    # Load shape log for date->shape lookup
    _shape_log = load_daily_shape_log()
    _shape_log = _shape_log.set_index('date').sort_index()


def predict(date, shape=None):
    """
    Predict persistence probabilities for a given date.

    Args:
        date: str or datetime-like, the date to predict for
        shape: str or None. If None, looks up shape from shape log.

    Returns:
        dict with keys:
            predicted_persists_4w: float (probability 0-1)
            predicted_persists_12w: float (probability 0-1)
            shape: str
            features_used: list[str]
            method: 'lr' or 'base_rate'
    """
    _ensure_models_loaded()

    date = pd.Timestamp(date)

    # Look up shape if not provided
    if shape is None:
        mask = _shape_log.index <= date
        if not mask.any():
            raise ValueError(f"No shape data available for {date.date()}")
        shape = _shape_log.loc[mask, 'shape'].iloc[-1]

    result = {
        'shape': shape,
        'shape_name': SHAPE_NAMES.get(shape, shape),
        'date': str(date.date()),
    }

    # Non-modelable shapes: return base rate
    if shape not in MODELABLE_SHAPES:
        result['predicted_persists_4w'] = _base_rates.get((shape, 4), 0.0) / 100
        result['predicted_persists_12w'] = _base_rates.get((shape, 12), 0.0) / 100
        result['features_used'] = []
        result['method'] = 'base_rate'
        return result

    # Find the weekly row closest to this date
    features = SHAPE_FEATURES[shape]
    mask = _weekly_panel['week_end_date'] <= date
    if not mask.any():
        raise ValueError(f"No weekly data available for {date.date()}")

    row = _weekly_panel.loc[mask].iloc[-1]
    X = row[features].values.reshape(1, -1).astype(float)

    # Check for NaN features
    if np.any(np.isnan(X)):
        result['predicted_persists_4w'] = _base_rates.get((shape, 4), 0.0) / 100
        result['predicted_persists_12w'] = _base_rates.get((shape, 12), 0.0) / 100
        result['features_used'] = features
        result['method'] = 'base_rate (missing features)'
        return result

    preds = {}
    for N in [4, 12]:
        key = (shape, N)
        if key in _fitted_models:
            model_info = _fitted_models[key]
            X_scaled = model_info['scaler'].transform(X)
            prob = model_info['lr'].predict_proba(X_scaled)[0]
            # probability of class 1 (persists)
            class_idx = list(model_info['lr'].classes_).index(1)
            preds[f'predicted_persists_{N}w'] = float(prob[class_idx])
        else:
            preds[f'predicted_persists_{N}w'] = _base_rates.get(key, 0.0) / 100

    result.update(preds)
    result['features_used'] = features
    result['method'] = 'lr'

    return result
