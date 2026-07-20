"""
FCPO Calendar Spread — Backtest Dashboard
==========================================
Interactive parameter exploration for the FCPO MR backtest engine.

Run: python -m streamlit run "MRBackTest/FCPO Backtest Dashboard/app.py" --server.port 8506
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path

# Import engine via importlib to avoid signal.py conflict
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'backtest_engine',
    str(Path(__file__).parent.parent / 'engine' / 'backtest_engine.py')
)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

DARK_BG   = "#0e1117"
DARK_PLOT = "#262730"
DARK_GRID = "#3a3a4a"
DARK_TEXT = "#fafafa"

CACHE_PATH = Path(__file__).parent / 'locked_config_cache.pkl'

SHAPE_COLORS = {
    '0.0': '#2ca02c',   # SB — green
    '1':   '#1f77b4',   # MB — blue
    '2':   '#ff7f0e',   # TB — orange
    '3':   '#d62728',   # C  — red
    '4':   '#9467bd',   # F  — purple
}
SHAPE_NAMES = {
    '0.0': 'SB (Super Backwardation)',
    '1':   'MB (Mild Backwardation)',
    '2':   'TB (Transitional)',
    '3':   'C (Contango)',
    '4':   'F (Full Contango)',
}

EXIT_COLORS = {
    'take_profit': '#2ca02c',
    'invalidated': '#ff7f0e',
    'time_stop': '#d62728',
    'regime_risk': '#9467bd',
    'stop_loss': '#e377c2',
}

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FCPO Backtest Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("FCPO Calendar Spread — Backtest Dashboard")


# ══════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Parameters")

    st.subheader("Entry / Exit")
    entry_z = st.number_input("Entry Z-score", min_value=0.5, max_value=4.0,
                               value=1.5, step=0.25, format="%.2f")
    exit_z = st.number_input("Exit Z-score", min_value=0.0, max_value=2.0,
                              value=0.5, step=0.1, format="%.2f")
    duration_threshold = st.number_input("Min days in shape", min_value=1, max_value=20,
                                          value=3)
    time_stop_days = st.number_input("Time stop (days)", min_value=5, max_value=60,
                                      value=20)

    st.subheader("Position Sizing")
    first_entry_lots = st.number_input("Base lots", min_value=1, max_value=10, value=1)

    enable_stop_loss = st.checkbox("Enable stop-loss")
    stop_loss_z = None
    if enable_stop_loss:
        stop_loss_z = st.number_input("Stop-loss Z", min_value=1.5, max_value=5.0,
                                       value=3.5, step=0.25, format="%.2f")

    st.subheader("Scale-In Tiers")
    n_tiers = st.selectbox("Number of scale-in tiers", [0, 1, 2, 3], index=0)
    scale_in_tiers = []
    for i in range(n_tiers):
        col1, col2 = st.columns(2)
        with col1:
            z_lvl = st.number_input(f"Tier {i+1} Z", min_value=1.5, max_value=4.0,
                                     value=1.75 + i * 0.25, step=0.25,
                                     format="%.2f", key=f"tier_z_{i}")
        with col2:
            lots = st.number_input(f"Tier {i+1} lots", min_value=1, max_value=5,
                                    value=1, key=f"tier_lots_{i}")
        scale_in_tiers.append({'z_level': z_lvl, 'lots': lots})

    st.subheader("Filters")
    pm_filter = st.selectbox("PM filter level", [0, 1, 2], index=0,
                              help="0=strict (top-1 match >=70%), 1=top-2, 2=any")
    tm_thresh = st.number_input("TM regime-risk threshold", min_value=0.0, max_value=1.0,
                                 value=0.50, step=0.05, format="%.2f",
                                 help="Exit if persistence prob < threshold (0=disabled)")
    if tm_thresh == 0.0:
        tm_thresh = None

    st.subheader("Instruments")
    instruments = st.multiselect(
        "Select instruments",
        options=engine.ALL_9,
        default=engine.ALL_9,
    )

    st.subheader("Cost Model")
    cost_spread = st.number_input("Spread cost (MYR/lot RT)", min_value=0.0, max_value=200.0,
                                   value=22.0, step=1.0)
    cost_butterfly = st.number_input("Butterfly cost (MYR/lot RT)", min_value=0.0, max_value=200.0,
                                      value=44.0, step=1.0)

    run_button = st.button("Run Backtest", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════
# BUILD CONFIG
# ══════════════════════════════════════════════════════════════

def build_config():
    return {
        'entry_z': entry_z,
        'exit_z': exit_z,
        'duration_threshold': duration_threshold,
        'first_entry_lots': first_entry_lots,
        'time_stop_days': time_stop_days,
        'stop_loss_z': stop_loss_z,
        'scale_in_tiers': scale_in_tiers,
        'instruments': instruments,
        'pm_filter_level': pm_filter,
        'tm_regime_risk_threshold': tm_thresh,
        'cost_spread_myr': cost_spread,
        'cost_butterfly_myr': cost_butterfly,
    }


def is_default_config():
    """Check if current params match the locked default (for cache use)."""
    return (entry_z == 1.5 and exit_z == 0.5 and duration_threshold == 3
            and first_entry_lots == 1 and time_stop_days == 20
            and stop_loss_z is None and len(scale_in_tiers) == 0
            and set(instruments) == set(engine.ALL_9) and pm_filter == 0
            and tm_thresh == 0.50 and cost_spread == 22.0 and cost_butterfly == 44.0)


# ══════════════════════════════════════════════════════════════
# LOAD / RUN RESULTS
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_cached():
    """Load pre-cached locked-config results."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_resource
def get_panel():
    """Load and cache the data panel."""
    return engine.build_panel()


def run_custom_backtest(config):
    """Run backtest with custom config."""
    df, tm_cache = get_panel()
    results = engine.run_backtest(config, df, tm_cache)

    # Compute metrics for each window
    output = {}
    for w in engine.WINDOWS:
        trades = results[w['name']]
        metrics = engine.compute_metrics(trades, df, w['test_start'], w['test_end'])
        adj_sharpe = engine.compute_daily_portfolio_sharpe_configurable(
            trades, w['test_start'], w['test_end'], df, config)
        metrics['adj_sharpe'] = adj_sharpe
        output[w['name']] = {
            'n': metrics['n_trades'],
            'pnl': metrics['total_pnl'],
            'adj_sharpe': adj_sharpe,
            'metrics': metrics,
            'trades': trades,
        }
    return output


# ══════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════

def display_results(results, label=""):
    """Display summary metrics, loss buckets, and trade chart."""

    if label:
        st.subheader(label)

    # --- Summary metrics (4 windows side by side) ---
    st.markdown("### Summary Metrics")
    cols = st.columns(4)
    worst_sharpe = min(r['adj_sharpe'] for r in results.values()
                       if not np.isnan(r.get('adj_sharpe', np.nan)))
    worst_window = None

    for i, w in enumerate(engine.WINDOWS):
        wname = w['name']
        r = results[wname]
        m = r['metrics']
        sharpe = r['adj_sharpe']
        is_worst = (not np.isnan(sharpe) and abs(sharpe - worst_sharpe) < 0.001)
        if is_worst:
            worst_window = wname

        with cols[i]:
            border = "border: 2px solid #d62728;" if is_worst else ""
            st.markdown(f"""
            <div style="background: {DARK_PLOT}; padding: 12px; border-radius: 8px; {border}">
                <h4 style="margin:0; color: {DARK_TEXT};">{w['label']}</h4>
                <p style="margin:2px 0; color: {DARK_TEXT};">Trades: <b>{m['n_trades']}</b></p>
                <p style="margin:2px 0; color: {DARK_TEXT};">Win Rate: <b>{m['win_rate']}%</b></p>
                <p style="margin:2px 0; color: {'#2ca02c' if m['total_pnl'] > 0 else '#d62728'};">
                    PnL: <b>{m['total_pnl']:+,.1f} pts</b></p>
                <p style="margin:2px 0; color: {DARK_TEXT};">Adj Sharpe: <b>{sharpe:.3f}</b></p>
                <p style="margin:2px 0; color: {DARK_TEXT};">Avg HP: <b>{m['avg_hp']} days</b></p>
                <p style="margin:2px 0; color: {DARK_TEXT};">Max DD: <b>{m['max_dd']:+,.1f}</b></p>
            </div>
            """, unsafe_allow_html=True)

    if worst_window:
        st.caption(f"Worst-case window: {worst_window} (adj Sharpe = {worst_sharpe:.3f})")

    # --- Exit reason breakdown ---
    st.markdown("### Exit Reason Breakdown")
    exit_data = []
    for w in engine.WINDOWS:
        r = results[w['name']]
        m = r['metrics']
        exit_data.append({
            'Window': w['label'],
            'Take Profit': f"{m['pct_take_profit']}%",
            'Invalidated': f"{m['pct_invalidated']}%",
            'Time Stop': f"{m['pct_time_stop']}%",
            'Regime Risk': f"{m['pct_regime_risk']}%",
            'Stop Loss': f"{m.get('pct_stop_loss', 0)}%",
        })
    st.dataframe(pd.DataFrame(exit_data).set_index('Window'), use_container_width=True)

    # --- Loss bucket breakdown ---
    st.markdown("### Loss Bucket Breakdown")
    bucket_data = []
    for w in engine.WINDOWS:
        trades = results[w['name']]['trades']
        if len(trades) == 0:
            continue
        losses = trades[trades['net_pnl'] <= 0].copy()
        if len(losses) == 0:
            bucket_data.append({
                'Window': w['label'], 'A (Regime)': 0, 'B/C (Time)': 0,
                'D (RR)': 0, 'TP-loss': 0, 'SL': 0,
            })
            continue
        losses['bucket'] = losses.apply(engine.classify_loss_bucket, axis=1)
        counts = losses['bucket'].value_counts()
        bucket_data.append({
            'Window': w['label'],
            'A (Regime)': counts.get('A', 0),
            'B/C (Time)': counts.get('B/C', 0),
            'D (RR)': counts.get('D', 0),
            'TP-loss': counts.get('TP-loss', 0),
            'SL': counts.get('SL', 0),
        })
    if bucket_data:
        st.dataframe(pd.DataFrame(bucket_data).set_index('Window'), use_container_width=True)

    # --- Per-instrument breakdown ---
    st.markdown("### Per-Instrument Summary")
    inst_rows = []
    for inst in instruments:
        inst_pnl = 0
        inst_n = 0
        inst_wins = 0
        for w in engine.WINDOWS:
            trades = results[w['name']]['trades']
            if len(trades) == 0:
                continue
            it = trades[trades['instrument'] == inst]
            inst_n += len(it)
            inst_wins += (it['net_pnl'] > 0).sum()
            inst_pnl += it['net_pnl'].sum()
        wr = round(inst_wins / inst_n * 100, 1) if inst_n > 0 else 0
        inst_rows.append({
            'Instrument': inst, 'Trades': inst_n, 'Win%': f"{wr}%",
            'Total PnL': round(inst_pnl, 1),
        })
    st.dataframe(pd.DataFrame(inst_rows).set_index('Instrument'), use_container_width=True)

    # --- Trade chart ---
    display_trade_chart(results)


def display_trade_chart(results):
    """Regime-labeled trade chart with entry/exit markers."""
    st.markdown("### Trade Chart")

    # Window + instrument selector
    col1, col2 = st.columns(2)
    with col1:
        selected_window = st.selectbox(
            "Window", [w['label'] for w in engine.WINDOWS],
            key="chart_window"
        )
    with col2:
        selected_inst = st.selectbox(
            "Instrument", instruments,
            key="chart_instrument"
        )

    # Find window
    w = next(w for w in engine.WINDOWS if w['label'] == selected_window)
    wname = w['name']
    trades = results[wname]['trades']
    inst_trades = trades[trades['instrument'] == selected_inst] if len(trades) > 0 else pd.DataFrame()

    # Get panel data for the chart
    try:
        df, _ = get_panel()
    except Exception:
        st.warning("Panel not loaded. Run a backtest first.")
        return

    ts = pd.Timestamp(w['test_start'])
    te = pd.Timestamp(w['test_end'])
    chart_df = df[(df['date'] >= ts) & (df['date'] <= te)].copy()

    if selected_inst not in chart_df.columns or chart_df[selected_inst].isna().all():
        st.info(f"No data for {selected_inst} in {selected_window}")
        return

    fig = make_subplots(rows=1, cols=1)

    # Spread price line
    fig.add_trace(go.Scatter(
        x=chart_df['date'], y=chart_df[selected_inst],
        mode='lines', name=selected_inst,
        line=dict(color='#fafafa', width=1.5),
    ))

    # Shape background bands
    shapes_list = []
    prev_shape = None
    band_start = None
    for _, row in chart_df.iterrows():
        s = str(row['shape'])
        if s != prev_shape:
            if prev_shape is not None and band_start is not None:
                shapes_list.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=band_start, x1=row['date'], y0=0, y1=1,
                    fillcolor=SHAPE_COLORS.get(prev_shape, '#555555'),
                    opacity=0.15, line_width=0,
                ))
            band_start = row['date']
            prev_shape = s
    # Close last band
    if prev_shape is not None and band_start is not None:
        shapes_list.append(dict(
            type="rect", xref="x", yref="paper",
            x0=band_start, x1=chart_df['date'].iloc[-1], y0=0, y1=1,
            fillcolor=SHAPE_COLORS.get(prev_shape, '#555555'),
            opacity=0.15, line_width=0,
        ))

    # Entry/exit markers
    if len(inst_trades) > 0:
        for _, t in inst_trades.iterrows():
            is_win = t['net_pnl'] > 0
            marker_color = '#2ca02c' if is_win else '#d62728'

            # Entry marker
            fig.add_trace(go.Scatter(
                x=[t['entry_date']], y=[t['entry_spread']],
                mode='markers', name='',
                marker=dict(symbol='triangle-up' if t['direction'] == 'LONG' else 'triangle-down',
                            size=10, color=marker_color, line=dict(width=1, color='white')),
                hovertext=f"ENTRY {t['direction']}<br>Z: {t['entry_z']:.2f}<br>Spread: {t['entry_spread']:.1f}",
                hoverinfo='text',
                showlegend=False,
            ))

            # Exit marker
            exit_color = EXIT_COLORS.get(t['exit_reason'], '#888888')
            fig.add_trace(go.Scatter(
                x=[t['exit_date']], y=[t['exit_spread']],
                mode='markers', name='',
                marker=dict(symbol='x', size=10, color=exit_color,
                            line=dict(width=2, color=exit_color)),
                hovertext=(f"EXIT: {t['exit_reason']}<br>"
                           f"PnL: {t['net_pnl']:+.1f}<br>"
                           f"Days: {t['days_held']}<br>"
                           f"Z: {t['exit_z']:.2f}" if not pd.isna(t['exit_z']) else
                           f"EXIT: {t['exit_reason']}<br>PnL: {t['net_pnl']:+.1f}<br>Days: {t['days_held']}"),
                hoverinfo='text',
                showlegend=False,
            ))

    fig.update_layout(
        shapes=shapes_list,
        height=500,
        plot_bgcolor=DARK_PLOT,
        paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        xaxis=dict(gridcolor=DARK_GRID, title='Date'),
        yaxis=dict(gridcolor=DARK_GRID, title=f'{selected_inst} Spread'),
        margin=dict(l=60, r=30, t=40, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Trade table
    if len(inst_trades) > 0:
        st.markdown(f"**{len(inst_trades)} trades** for {selected_inst} in {selected_window}")
        display_cols = ['entry_date', 'exit_date', 'direction', 'entry_spread', 'exit_spread',
                        'entry_z', 'exit_z', 'days_held', 'exit_reason', 'gross_pnl', 'net_pnl']
        display_cols = [c for c in display_cols if c in inst_trades.columns]
        st.dataframe(inst_trades[display_cols].reset_index(drop=True), use_container_width=True)

    # Shape legend
    st.markdown("**Shape Legend:** " + " | ".join(
        f"<span style='color:{c}'>{SHAPE_NAMES.get(k, k)}</span>"
        for k, c in SHAPE_COLORS.items()
    ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN FLOW
# ══════════════════════════════════════════════════════════════

# Load cached results on startup
if 'results' not in st.session_state:
    cached = load_cached()
    if cached:
        st.session_state['results'] = cached
        st.session_state['results_label'] = "Locked Config (spread=22, butterfly=44 MYR)"
    else:
        st.session_state['results'] = None
        st.session_state['results_label'] = ""

# Handle run button
if run_button:
    if not instruments:
        st.error("Select at least one instrument.")
    elif is_default_config():
        # Use cache
        cached = load_cached()
        if cached:
            st.session_state['results'] = cached
            st.session_state['results_label'] = "Locked Config (cached)"
        else:
            with st.spinner("Running backtest..."):
                config = build_config()
                st.session_state['results'] = run_custom_backtest(config)
                st.session_state['results_label'] = "Locked Config"
    else:
        with st.spinner("Running backtest with custom parameters..."):
            config = build_config()
            st.session_state['results'] = run_custom_backtest(config)
            st.session_state['results_label'] = "Custom Config"

# Display
if st.session_state.get('results'):
    display_results(st.session_state['results'], st.session_state.get('results_label', ''))
else:
    st.info("Click 'Run Backtest' or wait for cached results to load.")

# Cost model info
with st.sidebar:
    st.divider()
    st.caption(f"POINT_VALUE = {engine.POINT_VALUE} MYR/point")
    st.caption(f"Spread cost: {cost_spread/engine.POINT_VALUE:.2f} pts/lot")
    st.caption(f"Butterfly cost: {cost_butterfly/engine.POINT_VALUE:.2f} pts/lot")
