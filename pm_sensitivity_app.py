"""
PM Sensitivity Selector — interactive what-if tool for PM (Model C).

Select feature values at band boundaries and see how the predicted shape
changes in real time. Features shown with qualitative labels (Very Low
through Very High) derived from flip-point sensitivity bands.

Run:  python -m streamlit run pm_sensitivity_app.py --server.port 8508 --server.headless true
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date as dt_date

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
QUADRANT_NAMES = {
    0: 'Low S / Low P', 1: 'Low S / High P',
    2: 'High S / Low P', 3: 'High S / High P',
}

TIER_1 = ['prior_shape_enc', 'stock_pct', 'palm_soy_chg_4w']
TIER_2 = ['prod_yoy', 'usd_myr_chg_4w', 'prod_mom_3m', 'oni']
TIER_3 = ['days_in_shape', 'export_yoy', 'month', 'stock_prod_interaction']

FEATURE_TIER = {}
for f in TIER_1:
    FEATURE_TIER[f] = 1
for f in TIER_2:
    FEATURE_TIER[f] = 2
for f in TIER_3:
    FEATURE_TIER[f] = 3

FRIENDLY_NAMES = {
    'stock_pct':              'Stock Level (percentile)',
    'prod_mom_3m':            'Production Momentum (3-month change)',
    'prod_yoy':               'Production (Year-over-Year change)',
    'export_yoy':             'Exports (Year-over-Year change)',
    'oni':                    'ENSO Index (ONI)',
    'usd_myr_chg_4w':        'USD/MYR (4-week change)',
    'palm_soy_chg_4w':       'Palm-Soy Spread (4-week change)',
    'days_in_shape':          'Days in Current Regime',
    'prior_shape_enc':        'Previous Curve Shape',
    'month':                  'Calendar Month',
    'stock_prod_interaction': 'Stock/Production Quadrant',
}

FEATURE_DEFS = {
    'stock_pct':              "Where today's inventory ranks vs. all history, 0 = lowest ever, 1 = highest ever",
    'prod_mom_3m':            '% change in production over the trailing 3 months',
    'prod_yoy':               '% change in production vs. the same time last year',
    'export_yoy':             '% change in exports vs. the same time last year',
    'oni':                    'Ocean temperature anomaly indicating El Nino (positive) / La Nina (negative) strength',
    'usd_myr_chg_4w':        '% change in the USD/MYR exchange rate over the past 4 weeks',
    'palm_soy_chg_4w':       '% change in the price spread between palm oil and soybean oil over the past 4 weeks',
    'days_in_shape':          'Number of trading days the curve has stayed in its current shape',
    'prior_shape_enc':        'The shape the curve was in immediately before the current regime began',
    'month':                  'Seasonal month indicator',
    'stock_prod_interaction': 'Combined state of whether stock is high/low and production is rising/falling',
}

DARK_BG   = "#0e1117"
DARK_PLOT = "#262730"
DARK_TEXT = "#fafafa"

QUAL_LABELS = {
    2: ['Low', 'High'],
    3: ['Low', 'Med', 'High'],
    4: ['Low', 'Low-Med', 'Med-High', 'High'],
    5: ['Very Low', 'Low', 'Med', 'High', 'Very High'],
}

# 16 preset anchor dates from the sensitivity study
PRESET_DATES = [
    # High-conf
    '2026-06-26',
    # Low-conf (8)
    '2025-04-25', '2025-04-24', '2025-06-13', '2025-06-23',
    '2025-07-02', '2025-07-11', '2025-11-12', '2026-01-14',
    # Mid-conf (7)
    '2020-03-05', '2023-06-15', '2024-05-16', '2025-06-30',
    '2025-07-18', '2025-05-13', '2026-03-16',
]


# ── merge algorithm ────────────────────────────────────────────
def merge_bands_to_groups(bands):
    """
    Two-phase merge of flip-point bands into at most 5 qualitative groups.

    Phase 1: merge adjacent bands with the same predicted shape.
    Phase 2: iteratively merge the adjacent pair with the smallest
             combined width until <=5 remain.
    """
    if not bands:
        return []

    groups = []
    for b in bands:
        groups.append({
            'start': b['start'],
            'end': b['end'],
            'sub_bands': [b],
        })

    def _dominant(g):
        shapes = {}
        for sb in g['sub_bands']:
            w = sb['end'] - sb['start']
            shapes[sb['shape']] = shapes.get(sb['shape'], 0) + w
        return max(shapes, key=shapes.get)

    merged = True
    while merged:
        merged = False
        new_groups = [groups[0]]
        for g in groups[1:]:
            prev = new_groups[-1]
            if _dominant(prev) == _dominant(g):
                new_groups[-1] = {
                    'start': prev['start'],
                    'end': g['end'],
                    'sub_bands': prev['sub_bands'] + g['sub_bands'],
                }
                merged = True
            else:
                new_groups.append(g)
        groups = new_groups

    while len(groups) > 5:
        min_width = float('inf')
        min_idx = 0
        for i in range(len(groups) - 1):
            combined = (groups[i]['end'] - groups[i]['start']) + \
                       (groups[i + 1]['end'] - groups[i + 1]['start'])
            if combined < min_width:
                min_width = combined
                min_idx = i
        a, b = groups[min_idx], groups[min_idx + 1]
        merged_group = {
            'start': a['start'],
            'end': b['end'],
            'sub_bands': a['sub_bands'] + b['sub_bands'],
        }
        groups = groups[:min_idx] + [merged_group] + groups[min_idx + 2:]

    for g in groups:
        dom = _dominant(g)
        g['dominant_shape'] = dom
        g['dominant_abbrev'] = SHAPE_ABBREV.get(dom, dom)
        distinct = set(sb['shape'] for sb in g['sub_bands'])
        g['mixed'] = len(distinct) > 1
        g['all_shapes'] = distinct

    return groups


def build_group_tooltip(group, feat_name):
    """Build tooltip text for a merged group, flagging multi-shape merges."""
    is_discrete = feat_name in DISCRETE_FEATURES
    if is_discrete:
        rng = f"{group['start']:.0f} \u2013 {group['end']:.0f}"
    else:
        rng = f"{group['start']:.4f} \u2013 {group['end']:.4f}"

    tip = f"[{rng}] \u2192 {group['dominant_abbrev']}"

    if group['mixed']:
        minority = []
        for sb in group['sub_bands']:
            sb_abbrev = SHAPE_ABBREV.get(sb['shape'], sb['shape'])
            if sb['shape'] != group['dominant_shape']:
                if is_discrete:
                    minority.append(f"{sb_abbrev} near {sb['start']:.0f}")
                else:
                    minority.append(f"narrow {sb_abbrev} sub-band near {sb['start']:.4f}")
        if minority:
            tip += f" (contains {'; '.join(minority)})"

    return tip


def find_today_group_idx(groups, today_val):
    """Find which group index today's value falls in."""
    for i, g in enumerate(groups):
        if g['start'] <= today_val <= g['end']:
            return i
    dists = [min(abs(today_val - g['start']), abs(today_val - g['end'])) for g in groups]
    return int(np.argmin(dists))


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
def compute_bands(_panel_hash, anchor_vector_tuple, hist_mins, hist_maxs):
    """Sweep each feature to find shape-flip boundaries."""
    anchor_vector = np.array(anchor_vector_tuple)
    raw_bands = {}
    for feat_idx, feat_name in enumerate(TM_FEATURES):
        grid = build_grid(feat_name, hist_mins[feat_idx], hist_maxs[feat_idx])
        results = sweep_feature_with_probs(feat_name, grid, anchor_vector, feat_idx)
        raw_bands[feat_name] = find_bands([(r[0], r[1]) for r in results])
    return raw_bands


@st.cache_data(show_spinner="Scoring preset anchors...")
def build_preset_labels(_panel_hash):
    """Build display labels for the 16 preset anchor dates."""
    panel = load_panel()
    model, le = load_model()
    df = panel.sort_values('date').reset_index(drop=True)

    labels = []
    valid_dates = []
    for ds in PRESET_DATES:
        ts = pd.Timestamp(ds)
        match = df[df['date'] == ts]
        if len(match) == 0:
            # Try nearest before
            mask = df['date'] <= ts
            if not mask.any():
                continue
            match = df.loc[mask].iloc[[-1]]
        row = match.iloc[0]
        vec = row[TM_FEATURES].values.astype(float)
        if np.any(np.isnan(vec)):
            continue
        probs = model.predict_proba(vec.reshape(1, -1))[0]
        pred_enc = probs.argmax()
        pred_shape = str(le.inverse_transform([pred_enc])[0])
        obs_shape = str(row['shape'])
        conf = float(probs.max())
        actual_date = row['date'].date()

        obs_a = abbrev(obs_shape)
        pred_a = abbrev(pred_shape)
        labels.append(f"{actual_date}  |  obs={obs_a}, pred={pred_a}, conf={conf:.1%}")
        valid_dates.append(actual_date)

    return labels, valid_dates


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


def friendly(feat_name):
    return FRIENDLY_NAMES.get(feat_name, feat_name)


def build_selector(feat_name, raw_bands, anchor_val, hist_min, hist_max):
    """
    Render a selector widget for a feature and return the selected value.

    Discrete features keep their categorical selectbox.
    Continuous features with >=2 bands get qualitative-label select_slider.
    Features with 1 band (flat) get a disabled slider spanning the full
    historical range with the handle at the anchor value.
    """
    is_discrete = feat_name in DISCRETE_FEATURES
    label = friendly(feat_name)
    definition = FEATURE_DEFS.get(feat_name, '')
    tier = FEATURE_TIER.get(feat_name, '?')

    # ── Discrete features: categorical selectbox ──
    if is_discrete:
        st.caption(f"*{definition}*")
        options = DISCRETE_FEATURES[feat_name]
        labels = [discrete_label(feat_name, v) for v in options]
        today_idx = 0
        for i, v in enumerate(options):
            if int(round(anchor_val)) == v:
                today_idx = i
                break
        key = f"sel_{feat_name}"
        selected_label = st.selectbox(
            label,
            labels,
            index=today_idx,
            key=key,
            help=f"{definition}. Anchor value: {discrete_label(feat_name, anchor_val)}",
        )
        return float(options[labels.index(selected_label)])

    # ── Continuous features ──
    groups = merge_bands_to_groups(raw_bands)

    # Single band (flat): disabled slider spanning historical min/max
    if len(groups) <= 1:
        st.caption(f"*{definition}*")
        key = f"slider_{feat_name}"
        # Build options from hist range + anchor value, deduped and sorted
        opts = sorted(set([round(hist_min, 6), round(float(anchor_val), 6), round(hist_max, 6)]))
        # Find anchor position in options
        anchor_rounded = round(float(anchor_val), 6)
        default_val = anchor_rounded
        if default_val not in opts:
            # Snap to nearest
            default_val = min(opts, key=lambda x: abs(x - anchor_rounded))
        st.select_slider(
            label,
            options=opts,
            value=default_val,
            format_func=lambda x: f"{x:.4f}",
            key=key,
            help=f"{definition}. Anchor value: {anchor_val:.4f}. "
                 f"Range: [{hist_min:.4f}, {hist_max:.4f}]",
            disabled=True,
        )
        st.caption(
            f"Flat today (no flip in range) \u2014 historically sensitive "
            f"on other dates, see Tier {tier}."
        )
        return float(anchor_val)

    # Multiple groups: qualitative-label select_slider
    st.caption(f"*{definition}*")
    n = len(groups)
    qual = QUAL_LABELS.get(n, [f"G{i+1}" for i in range(n)])

    option_labels = []
    option_values = []
    tooltips = []
    today_group = find_today_group_idx(groups, anchor_val)

    for i, g in enumerate(groups):
        mid = (g['start'] + g['end']) / 2.0
        option_values.append(mid)
        tip = build_group_tooltip(g, feat_name)
        tooltips.append(tip)
        option_labels.append(f"{qual[i]} \u2192 {g['dominant_abbrev']}")

    key = f"slider_{feat_name}"
    selected_idx = st.select_slider(
        label,
        options=list(range(n)),
        value=today_group,
        format_func=lambda i: option_labels[i],
        key=key,
        help=f"{definition}. Anchor value: {anchor_val:.4f} ({qual[today_group]})",
    )

    g = groups[selected_idx]
    tip = tooltips[selected_idx]
    is_anchor = (selected_idx == today_group)
    marker = " *(anchor)*" if is_anchor else ""
    st.caption(f"{tip}{marker}")

    return option_values[selected_idx]


# ── date resolution ────────────────────────────────────────────
def resolve_anchor_row(df, selected_date):
    """
    Resolve a selected date to a panel row.
    Returns (row, error_message). If row is None, error_message explains why.
    """
    ts = pd.Timestamp(selected_date)
    panel_min = df['date'].iloc[0]
    panel_max = df['date'].iloc[-1]

    if ts < panel_min or ts > panel_max:
        return None, (
            f"Date {selected_date} is outside the panel range "
            f"({panel_min.date()} to {panel_max.date()})."
        )

    exact = df[df['date'] == ts]
    if len(exact) > 0:
        row = exact.iloc[0]
        vec = row[TM_FEATURES].values.astype(float)
        if np.any(np.isnan(vec)):
            return None, f"Date {selected_date} has missing feature values (NaN)."
        return row, None

    before = df[df['date'] < ts]
    after = df[df['date'] > ts]
    nearest_before = before['date'].iloc[-1].date() if len(before) > 0 else None
    nearest_after = after['date'].iloc[0].date() if len(after) > 0 else None
    parts = []
    if nearest_before:
        parts.append(str(nearest_before))
    if nearest_after:
        parts.append(str(nearest_after))
    return None, (
        f"No data for {selected_date} (non-trading day). "
        f"Nearest available dates: {', '.join(parts)}."
    )


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

    df = panel.sort_values('date').reset_index(drop=True)
    latest = df.iloc[-1]
    latest_date = latest['date'].date()
    panel_min_date = df['date'].iloc[0].date()

    # Historical stats for grid bounds (constant across all anchor dates)
    hist_mins = tuple(float(df[f].min()) for f in TM_FEATURES)
    hist_maxs = tuple(float(df[f].max()) for f in TM_FEATURES)

    # Build preset labels (cached)
    preset_labels, preset_dates = build_preset_labels(len(df))

    # ── Date selector ──
    st.subheader("Anchor Date")
    dc1, dc2, dc3 = st.columns([1, 1, 1])

    # Preset dropdown
    CUSTOM_OPTION = "Custom date..."
    LATEST_OPTION = f"Today (latest): {latest_date}"
    dropdown_options = [LATEST_OPTION] + preset_labels + [CUSTOM_OPTION]

    with dc1:
        preset_choice = st.selectbox(
            "Quick-select anchor",
            dropdown_options,
            index=0,
            key="preset_select",
            help="16 tested anchor dates from the sensitivity study, or pick any date.",
        )

    # Determine effective date
    if preset_choice == CUSTOM_OPTION:
        with dc2:
            selected_date = st.date_input(
                "Pick any date in panel",
                value=latest_date,
                min_value=panel_min_date,
                max_value=latest_date,
                key="anchor_date",
                help=f"Panel range: {panel_min_date} to {latest_date}.",
            )
    elif preset_choice == LATEST_OPTION:
        selected_date = latest_date
    else:
        # Extract date from preset label ("2025-07-11  |  obs=...")
        idx = preset_labels.index(preset_choice)
        selected_date = preset_dates[idx]

    # Resolve selected date to a panel row
    anchor_row, date_error = resolve_anchor_row(df, selected_date)
    if date_error:
        with dc3:
            st.error(date_error)
        return

    anchor_date = anchor_row['date']
    anchor_vector = anchor_row[TM_FEATURES].values.astype(float)
    anchor_shape = str(anchor_row['shape'])

    with dc3:
        st.info(f"Anchored to: {anchor_date.date()}")

    # Compute bands for this anchor (cached per unique vector)
    anchor_vector_tuple = tuple(float(v) for v in anchor_vector)
    all_bands = compute_bands(len(df), anchor_vector_tuple, hist_mins, hist_maxs)

    # Baseline prediction at anchor
    probs_baseline = model.predict_proba(anchor_vector.reshape(1, -1))[0]
    pred_enc_baseline = probs_baseline.argmax()
    pred_shape_baseline = str(le.inverse_transform([pred_enc_baseline])[0])
    conf_baseline = float(probs_baseline.max())

    # ── Anchor summary ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date", str(anchor_date.date()))
    c2.metric("Observed Shape", abbrev(anchor_shape))
    c3.metric("PM Predicted", abbrev(pred_shape_baseline))
    c4.metric("PM Confidence", f"{conf_baseline:.1%}")

    st.divider()

    # ── Reset button ──
    if st.button("Reset to anchor values"):
        for key in list(st.session_state.keys()):
            if key.startswith("sel_") or key.startswith("slider_"):
                del st.session_state[key]
        st.rerun()

    # ── Feature selectors ──
    selected_values = {}

    # Tier 1
    st.subheader("Tier 1 \u2014 Primary Drivers")
    cols = st.columns(len(TIER_1))
    for col, feat in zip(cols, TIER_1):
        feat_idx = TM_FEATURES.index(feat)
        with col:
            selected_values[feat] = build_selector(
                feat, all_bands[feat], anchor_vector[feat_idx],
                hist_mins[feat_idx], hist_maxs[feat_idx])

    # Tier 2
    with st.expander("Tier 2 \u2014 Secondary Drivers", expanded=True):
        cols = st.columns(len(TIER_2))
        for col, feat in zip(cols, TIER_2):
            feat_idx = TM_FEATURES.index(feat)
            with col:
                selected_values[feat] = build_selector(
                    feat, all_bands[feat], anchor_vector[feat_idx],
                    hist_mins[feat_idx], hist_maxs[feat_idx])

    # Tier 3
    with st.expander("Tier 3 \u2014 Minor Drivers", expanded=False):
        cols = st.columns(len(TIER_3))
        for col, feat in zip(cols, TIER_3):
            feat_idx = TM_FEATURES.index(feat)
            with col:
                selected_values[feat] = build_selector(
                    feat, all_bands[feat], anchor_vector[feat_idx],
                    hist_mins[feat_idx], hist_maxs[feat_idx])

    st.divider()

    # ── Build modified vector and predict ──
    modified_vector = anchor_vector.copy()
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

    rc1.markdown(
        f"### Predicted Shape: <span style='color:{color}'>{abbrev_new}</span>",
        unsafe_allow_html=True)
    rc2.metric("Confidence", f"{conf_new:.1%}",
               delta=f"{conf_new - conf_baseline:+.1%}"
               if abs(conf_new - conf_baseline) > 0.001 else None)

    if pred_shape_new != pred_shape_baseline:
        rc3.warning(f"Shape changed: {abbrev(pred_shape_baseline)} \u2192 {abbrev_new}")
    else:
        rc3.success("Same as baseline prediction")

    # Probability bar chart
    shape_labels = [str(c) for c in le.classes_]
    shape_abbrevs = [abbrev(s) for s in shape_labels]
    probs_list = [float(p) for p in probs_new]
    bar_colors = [SHAPE_COLORS.get(a, '#888') for a in shape_abbrevs]

    fig = go.Figure(go.Bar(
        x=shape_abbrevs, y=probs_list,
        marker_color=bar_colors,
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
