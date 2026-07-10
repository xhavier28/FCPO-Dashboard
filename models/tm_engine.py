"""
Transition Model (TM) Engine.

Predicts which shape the curve will transition to at 1-week and 2-week horizons.
Both horizons use XGBoost regularised (Config B) with per-shape models:
- 1-week: XGBoost regularised (max_depth=3, reg_alpha=1.0, reg_lambda=2.0)
- 2-week: XGBoost regularised (same hyperparameter philosophy)

Both trained on 2017-2024 daily panel, cached after first call.

CONTAMINATION NOTE (July 2026):
  The 1w model was originally RF (46.6% test-only top-1). The historically-
  cited 70.7% top-1 / 93.6% top-2 were full-panel (train+test pooled, n=3,808)
  figures, not test-only. XGBoost Config B replaced RF at 65.7% top-1 / 90.3%
  top-2 on identical clean windows. See models/MODEL_REFERENCE.txt.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .feature_prep import build_tm_daily_panel, TM_FEATURES

warnings.filterwarnings('ignore')

TRAIN_END = '2024-12-31'
TEST_START = '2025-01-01'

# Module-level cache
_models_1w = {}   # shape -> {'model': XGB, 'le': LabelEncoder, 'classes': [...]}
_models_2w = {}   # shape -> {'model': XGB, 'le': LabelEncoder, 'classes': [...]}
_daily_panel = None
_loaded = False


def _ensure_models_loaded():
    """Lazy init: build daily panel, fit XGBoost regularised for 1w and 2w per shape."""
    global _models_1w, _models_2w, _daily_panel, _loaded

    if _loaded:
        return

    _daily_panel = build_tm_daily_panel()
    df_tm = _daily_panel.sort_values('date').reset_index(drop=True)
    df_tm = df_tm.dropna(subset=TM_FEATURES + ['target_1w', 'target_2w'])

    df_train = df_tm[df_tm['date'] <= TRAIN_END].copy()
    shapes = sorted(df_tm['shape'].unique())

    from xgboost import XGBClassifier

    for s in shapes:
        tr = df_train[df_train['shape'] == s].dropna(subset=TM_FEATURES + ['target_1w', 'target_2w'])

        if len(tr) < 30:
            continue

        X_tr = tr[TM_FEATURES].values

        # ── 1-week: XGBoost regularised (Config B) ──
        y_1w = tr['target_1w'].values
        le_1w = LabelEncoder()
        y_1w_enc = le_1w.fit_transform(y_1w)

        xgb_1w = XGBClassifier(
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
        xgb_1w.fit(X_tr, y_1w_enc)
        _models_1w[s] = {
            'model': xgb_1w,
            'le': le_1w,
            'classes': list(le_1w.classes_),
        }

        # ── 2-week: XGBoost regularised (Config B) ──
        y_2w = tr['target_2w'].values
        le_2w = LabelEncoder()
        y_2w_enc = le_2w.fit_transform(y_2w)

        xgb_2w = XGBClassifier(
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
        xgb_2w.fit(X_tr, y_2w_enc)
        _models_2w[s] = {
            'model': xgb_2w,
            'le': le_2w,
            'classes': list(le_2w.classes_),
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

    # Both horizons use XGBoost with LabelEncoder
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
