"""
Transition Model (TM) Engine.

Predicts which shape the curve will transition to at 1-week and 2-week horizons.
- 1-week: Random Forest (per-shape, class_weight='balanced')
- 2-week: XGBoost regularised (per-shape, with LabelEncoder for targets)

Both trained on 2017-2024 daily panel, cached after first call.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .feature_prep import build_tm_daily_panel, TM_FEATURES

warnings.filterwarnings('ignore')

TRAIN_END = '2024-12-31'
TEST_START = '2025-01-01'

# Module-level cache
_models_1w = {}   # shape -> {'model': RF, 'classes': [...]}
_models_2w = {}   # shape -> {'model': XGB, 'le': LabelEncoder, 'classes': [...]}
_daily_panel = None
_loaded = False


def _ensure_models_loaded():
    """Lazy init: build daily panel, fit RF (1w) and XGBoost (2w) per shape."""
    global _models_1w, _models_2w, _daily_panel, _loaded

    if _loaded:
        return

    _daily_panel = build_tm_daily_panel()
    df_tm = _daily_panel.sort_values('date').reset_index(drop=True)
    df_tm = df_tm.dropna(subset=TM_FEATURES + ['target_1w', 'target_2w'])

    df_train = df_tm[df_tm['date'] <= TRAIN_END].copy()
    shapes = sorted(df_tm['shape'].unique())

    # Import XGBoost here (only needed at fit time)
    from xgboost import XGBClassifier

    for s in shapes:
        tr = df_train[df_train['shape'] == s].dropna(subset=TM_FEATURES + ['target_1w', 'target_2w'])

        if len(tr) < 30:
            continue

        X_tr = tr[TM_FEATURES].values

        # ── 1-week: Random Forest ──
        y_1w = tr['target_1w'].values
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=5,
            min_samples_leaf=8, random_state=42,
            class_weight='balanced')
        rf.fit(X_tr, y_1w)
        _models_1w[s] = {
            'model': rf,
            'classes': list(rf.classes_),
        }

        # ── 2-week: XGBoost regularised ──
        y_2w = tr['target_2w'].values
        le = LabelEncoder()
        y_2w_enc = le.fit_transform(y_2w)

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_weight=20,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0)
        xgb.fit(X_tr, y_2w_enc)
        _models_2w[s] = {
            'model': xgb,
            'le': le,
            'classes': list(le.classes_),
        }

    _loaded = True


def predict(date, horizon='1w'):
    """
    Predict the next shape transition for a given date and horizon.

    Args:
        date: str or datetime-like
        horizon: '1w' or '2w'

    Returns:
        dict with keys:
            top1_shape, top1_prob, top2_shape, top2_prob,
            ruled_out (list), all_probs (dict), current_shape, date, horizon
    """
    _ensure_models_loaded()

    date = pd.Timestamp(date)

    # Find the row for this date in the daily panel
    df = _daily_panel.sort_values('date')
    mask = df['date'] <= date
    if not mask.any():
        raise ValueError(f"No data available for {date.date()}")

    row = df.loc[mask].iloc[-1]
    current_shape = row['shape']

    # Check feature availability
    X = row[TM_FEATURES].values.reshape(1, -1).astype(float)
    if np.any(np.isnan(X)):
        return {
            'current_shape': current_shape,
            'date': str(date.date()),
            'horizon': horizon,
            'error': 'Missing features for this date',
        }

    if horizon == '1w':
        models = _models_1w
    elif horizon == '2w':
        models = _models_2w
    else:
        raise ValueError(f"horizon must be '1w' or '2w', got '{horizon}'")

    if current_shape not in models:
        return {
            'current_shape': current_shape,
            'date': str(date.date()),
            'horizon': horizon,
            'error': f'No model for shape {current_shape}',
        }

    model_info = models[current_shape]
    model = model_info['model']

    if horizon == '1w':
        # RF: predict_proba with string classes directly
        probs_raw = model.predict_proba(X)[0]
        classes = model.classes_
    else:
        # XGBoost: predict_proba returns encoded labels, decode
        le = model_info['le']
        probs_raw = model.predict_proba(X)[0]
        classes = le.classes_

    # Build probability dict
    all_probs = {str(cls): float(p) for cls, p in zip(classes, probs_raw)}

    # Sort by probability descending
    sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])

    # Ruled out: shapes with < 2% probability
    ruled_out = [s for s, p in sorted_probs if p < 0.02]

    result = {
        'current_shape': current_shape,
        'date': str(date.date()),
        'horizon': horizon,
        'top1_shape': sorted_probs[0][0],
        'top1_prob': round(sorted_probs[0][1], 4),
        'all_probs': {s: round(p, 4) for s, p in sorted_probs},
        'ruled_out': ruled_out,
    }

    if len(sorted_probs) >= 2:
        result['top2_shape'] = sorted_probs[1][0]
        result['top2_prob'] = round(sorted_probs[1][1], 4)

    return result
