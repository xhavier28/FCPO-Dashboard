"""
PM Sensitivity Selector — interactive what-if tool for PM (Model C).

Select feature values at band boundaries and see how the predicted shape
changes in real time.

Run:  python -m streamlit run pm_sensitivity_app.py --server.port 8508 --server.headless true
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from models.feature_prep import build_tm_daily_panel, TM_FEATURES
import models.pm_engine as pm_mod
from sensitivity_sweep import (
    build_grid, sweep_feature_with_probs, find_bands,
    SHAPE_ABBREV, DISCRETE_FEATURES, GRID_SIZE,
)

# ── constants ──────────────────────────────────────────────────
SHAPE_COLORS = {
    'SB': '#1f77b4', 'MB': '#ff7f0e', 'Mixed': '#2ca02c',
    'C': '#d62728', 'F': '#9467bd',
}

SHAPE_NAMES_FROM_ENC = {0: 'SB', 1: 'MB', 2: 'Mixed', 3: 'C', 4: 'F'}
MONTH_NAMES = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',
}
QUADRANT_NAMES = {0: 'Low S / Low P', 1: 'Low S / High P', 2: 'High S / Low P', 3: 'High S / High P'}

TIER_1 = ['prior_shape_enc', 'stock_pct', 'palm_soy_chg_4w']
TIER_2 = ['prod_yoy', 'usd_myr_chg_4w', 'prod_mom_3m', 'oni']
TIER_3 = ['days_in_shape', 'export_yoy', 'month', 'stock_prod_interaction']

DARK_BG   = "#0e1117"
DARK_PLOT = "#262730"
DARK_TEXT = "#fafafa"


# ── caching ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data panel...")
def load_panel():
    panel = build_tm_daily_panel()
    return panel.dropna(subset=TM_FEATURES + ['shape']).copy()


@st.cache_resource(show_spinner="Training PM model...")
def load_model():
    pm_mod._ensure_model_loaded()
    return pm_mod._model, pm_mod._le


@st.cache_data(show_spinner="Computing sensitivity bands...")
def compute_bands(_panel_hash, today_vector, hist_mins, hist_maxs):
    """Sweep each feature to find shape-flip boundaries."""
    bands = {}
    for feat_idx, feat_name in enumerate(TM_FEATURES):
        grid = build_grid(feat_name, hist_mins[feat_idx], hist_maxs[feat_idx])
        results = sweep_feature_with_probs(feat_name, grid, today_vector, feat_idx)
        bands[feat_name] = find_bands([(r[0], r[1]) for r in results])
    return bands


# ── helpers ────────────────────────────────────────────────────
def abbrev(shape_code):
    return SHAPE_ABBREV.get(str(shape_code), str(shape_code))


def discrete_label(feat_name, val):
    v = int(round(val))
    if feat_name == 'prior_shape_enc':
        return SHAPE_NAMES_FROM_ENC.get(v, str(v))
    if feat_name == 'month':
        return MONTH_NAMES.get(v, str(v))
    if feat_name == 'stock_prod_interaction':
        return QUADRANT_NAMES.get(v, str(v))
    return str(v)


def build_selector(feat_name, bands, today_val):
    """Render a selector widget and return the selected value."""
    is_discrete = feat_name in DISCRETE_FEATURES

    if is_discrete:
        options = DISCRETE_FEATURES[feat_name]
        labels = [discrete_label(feat_name, v) for v in options]
        today_idx = 0
        for i, v in enumerate(options):
            if int(round(today_val)) == v:
                today_idx = i
                break
        # Check session state for reset
        key = f"sel_{feat_name}"
        selected_label = st.selectbox(
            feat_name.replace('_', ' ').title(),
            labels,
            index=today_idx,
            key=key,
            help=f"Today's value: {discrete_label(feat_name, today_val)}",
        )
        return float(options[labels.index(selected_label)])
    else:
        # Continuous: build options from band edges + today's value
        edges = set()
        for b in bands:
            edges.add(round(b['start'], 6))
            edges.add(round(b['end'], 6))
        edges.add(round(float(today_val), 6))
        options = sorted(edges)

        # Format labels
        labels = [f"{v:.4f}" for v in options]

        # Find today's position
        today_rounded = round(float(today_val), 6)
        today_idx = 0
        for i, v in enumerate(options):
            if abs(v - today_rounded) < 1e-8:
                today_idx = i
                break

        key = f"slider_{feat_name}"
        selected = st.select_slider(
            feat_name.replace('_', ' ').title(),
            options=options,
            value=options[today_idx],
            format_func=lambda x: f"{x:.4f}",
            key=key,
            help=f"Today's value: {today_val:.4f}",
        )
        return float(selected)


# ── main app ───────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="PM Sensitivity Selector", layout="wide")
    st.title("PM Sensitivity Selector")

    # Load data + model
    try:
        panel = load_panel()
        model, le = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    df = panel.sort_values('date')
    latest = df.iloc[-1]
    today_date = latest['date']
    today_vector = latest[TM_FEATURES].values.astype(float)
    today_shape = str(latest['shape'])

    if np.any(np.isnan(today_vector)):
        st.error("Latest row has NaN features — cannot run sensitivity.")
        return

    # Historical stats for grid bounds
    hist_mins = np.array([df[f].min() for f in TM_FEATURES])
    hist_maxs = np.array([df[f].max() for f in TM_FEATURES])

    # Compute bands (cached)
    panel_hash = len(df)
    all_bands = compute_bands(panel_hash, today_vector, tuple(hist_mins), tuple(hist_maxs))

    # Baseline prediction
    probs_baseline = model.predict_proba(today_vector.reshape(1, -1))[0]
    pred_enc_baseline = probs_baseline.argmax()
    pred_shape_baseline = str(le.inverse_transform([pred_enc_baseline])[0])
    conf_baseline = float(probs_baseline.max())

    # ── Anchor summary ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date", str(today_date.date()))
    c2.metric("Observed Shape", abbrev(today_shape))
    c3.metric("PM Predicted", abbrev(pred_shape_baseline))
    c4.metric("PM Confidence", f"{conf_baseline:.1%}")

    st.divider()

    # ── Reset button ──
    if st.button("Reset to today's values"):
        for key in list(st.session_state.keys()):
            if key.startswith("sel_") or key.startswith("slider_"):
                del st.session_state[key]
        st.rerun()

    # ── Feature selectors ──
    selected_values = {}

    # Tier 1
    st.subheader("Tier 1 — Primary Drivers")
    cols = st.columns(len(TIER_1))
    for col, feat in zip(cols, TIER_1):
        feat_idx = TM_FEATURES.index(feat)
        with col:
            selected_values[feat] = build_selector(feat, all_bands[feat], today_vector[feat_idx])

    # Tier 2
    with st.expander("Tier 2 — Secondary Drivers", expanded=True):
        cols = st.columns(len(TIER_2))
        for col, feat in zip(cols, TIER_2):
            feat_idx = TM_FEATURES.index(feat)
            with col:
                selected_values[feat] = build_selector(feat, all_bands[feat], today_vector[feat_idx])

    # Tier 3
    with st.expander("Tier 3 — Minor Drivers", expanded=False):
        cols = st.columns(len(TIER_3))
        for col, feat in zip(cols, TIER_3):
            feat_idx = TM_FEATURES.index(feat)
            with col:
                selected_values[feat] = build_selector(feat, all_bands[feat], today_vector[feat_idx])

    st.divider()

    # ── Build modified vector and predict ──
    modified_vector = today_vector.copy()
    for feat, val in selected_values.items():
        feat_idx = TM_FEATURES.index(feat)
        modified_vector[feat_idx] = val

    try:
        probs_new = model.predict_proba(modified_vector.reshape(1, -1))[0]
        pred_enc_new = probs_new.argmax()
        pred_shape_new = str(le.inverse_transform([pred_enc_new])[0])
        conf_new = float(probs_new.max())
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # ── Live prediction display ──
    st.subheader("Prediction Result")

    rc1, rc2, rc3 = st.columns(3)
    abbrev_new = abbrev(pred_shape_new)
    color = SHAPE_COLORS.get(abbrev_new, '#fafafa')

    rc1.markdown(f"### Predicted Shape: <span style='color:{color}'>{abbrev_new}</span>",
                 unsafe_allow_html=True)
    rc2.metric("Confidence", f"{conf_new:.1%}",
               delta=f"{conf_new - conf_baseline:+.1%}" if abs(conf_new - conf_baseline) > 0.001 else None)

    # Delta indicator
    if pred_shape_new != pred_shape_baseline:
        rc3.warning(f"Shape changed: {abbrev(pred_shape_baseline)} → {abbrev_new}")
    else:
        rc3.success("Same as baseline prediction")

    # Probability bar chart
    shape_labels = [str(c) for c in le.classes_]
    shape_abbrevs = [abbrev(s) for s in shape_labels]
    probs_list = [float(p) for p in probs_new]
    colors = [SHAPE_COLORS.get(a, '#888') for a in shape_abbrevs]

    fig = go.Figure(go.Bar(
        x=shape_abbrevs, y=probs_list,
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs_list],
        textposition='outside',
    ))
    fig.update_layout(
        title="Shape Probabilities",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=300,
        margin=dict(l=60, r=30, t=40, b=40),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT,
        font_color=DARK_TEXT,
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
