"""
PM Engine — Prediction Model (Model C).

Predicts CURRENT shape from today's fundamentals using XGBoost.
This is the model referenced by the backtest plan's disagreement signal:
compare predicted_shape against the shape classifier's OBSERVED shape
to detect disagreement.

NOT the same as the persistence engine (persistence_engine.py), which
predicts whether a shape will persist at 4w/12w horizons.

Source: research/03. validation_analysis/initial_backtest_and_calibration.ipynb
Validated at ~90.0% OOS accuracy on 2025-2026 test set.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from .feature_prep import build_tm_daily_panel, TM_FEATURES

warnings.filterwarnings('ignore')

TRAIN_END = '2024-12-31'

# Module-level cache
_model = None
_le = None
_daily_panel = None
_loaded = False


def _ensure_model_loaded():
    """Lazy init: build daily panel, fit XGBoost Model C on 2017-2024 data."""
    global _model, _le, _daily_panel, _loaded

    if _loaded:
        return

    from xgboost import XGBClassifier

    _daily_panel = build_tm_daily_panel()
    df = _daily_panel.dropna(subset=TM_FEATURES + ['shape']).copy()

    df_train = df[df['date'] <= TRAIN_END]

    X_tr = df_train[TM_FEATURES].values
    y_tr = df_train['shape'].values

    _le = LabelEncoder()
    y_tr_enc = _le.fit_transform(y_tr)

    _model = XGBClassifier(
        n_estimators=300, max_depth=5,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42, verbosity=0)
    _model.fit(X_tr, y_tr_enc)

    _loaded = True


def predict(date):
    """
    Predicts CURRENT shape from today's fundamentals.

    Args:
        date: str or datetime-like

    Returns:
        dict with keys:
            predicted_shape: str — model's top-1 shape prediction
            confidence: float — probability of the top-1 shape (0-1)
            shape_probs: dict — {shape: probability} for all shapes
            observed_shape: str — the actual classified shape for this date
            date: str
    """
    _ensure_model_loaded()

    date = pd.Timestamp(date)

    # Find the row for this date
    df = _daily_panel.sort_values('date')
    mask = df['date'] <= date
    if not mask.any():
        raise ValueError(f"No data available for {date.date()}")

    row = df.loc[mask].iloc[-1]
    observed_shape = row['shape']

    X = row[TM_FEATURES].values.reshape(1, -1).astype(float)
    if np.any(np.isnan(X)):
        return {
            'predicted_shape': None,
            'confidence': None,
            'shape_probs': {},
            'observed_shape': observed_shape,
            'date': str(date.date()),
            'error': 'Missing features for this date',
        }

    pred_enc = _model.predict(X)[0]
    predicted_shape = _le.inverse_transform([pred_enc])[0]

    probs_raw = _model.predict_proba(X)[0]
    shape_probs = {str(cls): float(p) for cls, p in zip(_le.classes_, probs_raw)}

    confidence = float(max(probs_raw))

    return {
        'predicted_shape': predicted_shape,
        'confidence': confidence,
        'shape_probs': shape_probs,
        'observed_shape': observed_shape,
        'date': str(date.date()),
    }


def validate_oos():
    """
    Run OOS validation on 2025-2026 test set.
    Returns accuracy percentage and detail DataFrame.
    """
    _ensure_model_loaded()

    df = _daily_panel.dropna(subset=TM_FEATURES + ['shape']).copy()
    df_test = df[df['date'] >= '2025-01-01']

    X_te = df_test[TM_FEATURES].values
    y_te = df_test['shape'].values

    y_te_enc = _le.transform(y_te)
    pred_enc = _model.predict(X_te)
    pred_labels = _le.inverse_transform(pred_enc)

    acc = accuracy_score(y_te, pred_labels) * 100
    return acc, pd.DataFrame({
        'date': df_test['date'].values,
        'actual': y_te,
        'predicted': pred_labels,
        'correct': pred_labels == y_te,
    })
