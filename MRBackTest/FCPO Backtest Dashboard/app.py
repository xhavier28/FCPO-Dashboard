"""
FCPO Calendar Spread — Backtest Dashboard
==========================================
Interactive parameter exploration for the FCPO MR backtest engine.
Two tabs: Daily (4 Walk-Forward Windows) and 60-Min Intraday (2024-2026).

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
    '0.0': '#2ca02c',   # SB
    '1':   '#1f77b4',   # MB
    '2':   '#ff7f0e',   # TB
    '3':   '#d62728',   # C
    '4':   '#9467bd',   # F
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
    'time_stop':   '#d62728',
    'regime_risk': '#9467bd',
    'stop_loss':   '#e377c2',
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
# SIDEBAR CONTROLS (shared across both tabs)
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
    scale_in_tiers = []
    for i in range(3):
        with st.container():
            enabled = st.checkbox(f"Tier {i+1}", value=False, key=f"tier_en_{i}")
            if enabled:
                col1, col2 = st.columns(2)
                with col1:
                    z_lvl = st.number_input(f"T{i+1} Z", min_value=1.5, max_value=4.0,
                                             value=1.75 + i * 0.25, step=0.25,
                                             format="%.2f", key=f"tier_z_{i}")
                with col2:
                    lots = st.number_input(f"T{i+1} lots", min_value=1, max_value=5,
                                            value=1, key=f"tier_lots_{i}")
                scale_in_tiers.append({'z_level': z_lvl, 'lots': lots, 'enabled': True})
            else:
                scale_in_tiers.append({'z_level': 1.75 + i * 0.25, 'lots': 1, 'enabled': False})

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

    st.divider()
    st.caption(f"POINT_VALUE = {engine.POINT_VALUE} MYR/point")
    st.caption(f"Spread cost: {cost_spread/engine.POINT_VALUE:.2f} pts/lot")
    st.caption(f"Butterfly cost: {cost_butterfly/engine.POINT_VALUE:.2f} pts/lot")


# ══════════════════════════════════════════════════════════════
# CONFIG BUILDER
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
        'data_resolution': 'daily',
        'date_range': 'all',
    }


def is_default_config():
    """Check if current params match the locked default (for cache use)."""
    any_tier_enabled = any(t.get('enabled', False) for t in scale_in_tiers)
    return (entry_z == 1.5 and exit_z == 0.5 and duration_threshold == 3
            and first_entry_lots == 1 and time_stop_days == 20
            and stop_loss_z is None and not any_tier_enabled
            and set(instruments) == set(engine.ALL_9) and pm_filter == 0
            and tm_thresh == 0.50 and cost_spread == 22.0 and cost_butterfly == 44.0)


# ══════════════════════════════════════════════════════════════
# DATA / COMPUTE
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_cached():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_resource
def get_panel():
    return engine.build_panel()


@st.cache_resource
def get_intraday():
    return engine.load_intraday_data()


def run_custom_daily(config):
    df, tm_cache = get_panel()
    results = engine.run_backtest(config, df, tm_cache)
    output = {}
    for w in engine.WINDOWS:
        trades = results[w['name']]
        metrics = engine.compute_metrics(trades, df, w['test_start'], w['test_end'])
        adj_sharpe = engine.compute_daily_portfolio_sharpe_configurable(
            trades, w['test_start'], w['test_end'], df, config)
        metrics['adj_sharpe'] = adj_sharpe
        output[w['name']] = {
            'n': metrics['n_trades'], 'pnl': metrics['total_pnl'],
            'adj_sharpe': adj_sharpe, 'metrics': metrics, 'trades': trades,
        }
    return output


def run_custom_intraday(config):
    df, tm_cache = get_panel()
    intraday_df = get_intraday()
    results = engine.run_backtest_intraday(config, df, tm_cache, intraday_df)
    trades = results['intraday']
    metrics = engine.compute_metrics(trades, df, engine.INTRADAY_START, engine.INTRADAY_END)
    adj_sharpe = engine.compute_daily_portfolio_sharpe_configurable(
        trades, engine.INTRADAY_START, engine.INTRADAY_END, df, config)
    metrics['adj_sharpe'] = adj_sharpe
    return {
        'intraday': {
            'n': metrics['n_trades'], 'pnl': metrics['total_pnl'],
            'adj_sharpe': adj_sharpe, 'metrics': metrics, 'trades': trades,
        }
    }


# ══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

def render_metric_card(label, m, sharpe, is_worst=False):
    """Render a single window's metric card."""
    border = "border: 2px solid #d62728;" if is_worst else ""
    naive_sh = m.get('naive_sharpe', np.nan)
    naive_str = f"{naive_sh:.3f}" if not (isinstance(naive_sh, float) and np.isnan(naive_sh)) else "N/A"
    adj_str = f"{sharpe:.3f}" if not (isinstance(sharpe, float) and np.isnan(sharpe)) else "N/A"
    pnl_color = '#2ca02c' if m['total_pnl'] > 0 else '#d62728'

    st.markdown(f"""
    <div style="background: {DARK_PLOT}; padding: 12px; border-radius: 8px; {border}">
        <h4 style="margin:0; color: {DARK_TEXT};">{label}</h4>
        <p style="margin:2px 0; color: {DARK_TEXT};">Trades: <b>{m['n_trades']}</b></p>
        <p style="margin:2px 0; color: {DARK_TEXT};">Win Rate: <b>{m['win_rate']}%</b></p>
        <p style="margin:2px 0; color: {pnl_color};">
            PnL: <b>{m['total_pnl']:+,.1f} pts</b></p>
        <p style="margin:2px 0; color: {DARK_TEXT};">Naive Sharpe: <b>{naive_str}</b></p>
        <p style="margin:2px 0; color: {DARK_TEXT};">Adj Sharpe: <b>{adj_str}</b></p>
        <p style="margin:2px 0; color: {DARK_TEXT};">Avg HP: <b>{m['avg_hp']} days</b></p>
        <p style="margin:2px 0; color: {DARK_TEXT};">Max DD: <b>{m['max_dd']:+,.1f}</b></p>
    </div>
    """, unsafe_allow_html=True)


def render_exit_table(results_dict, window_keys, window_labels):
    """Render exit reason breakdown table."""
    exit_data = []
    for wkey, wlabel in zip(window_keys, window_labels):
        m = results_dict[wkey]['metrics']
        exit_data.append({
            'Window': wlabel,
            'Take Profit': f"{m['pct_take_profit']}%",
            'Invalidated': f"{m['pct_invalidated']}%",
            'Time Stop': f"{m['pct_time_stop']}%",
            'Regime Risk': f"{m['pct_regime_risk']}%",
            'Stop Loss': f"{m.get('pct_stop_loss', 0)}%",
        })
    st.dataframe(pd.DataFrame(exit_data).set_index('Window'), use_container_width=True)


def render_loss_buckets(results_dict, window_keys, window_labels):
    """Render loss bucket breakdown with proper B/C separation."""
    bucket_data = []
    for wkey, wlabel in zip(window_keys, window_labels):
        trades = results_dict[wkey]['trades']
        if len(trades) == 0:
            bucket_data.append({
                'Window': wlabel, 'A (Regime)': 0, 'B (Adverse)': 0,
                'C (Stalled)': 0, 'D (RR)': 0, 'TP-loss': 0, 'SL': 0,
            })
            continue
        losses = trades[trades['net_pnl'] <= 0].copy()
        if len(losses) == 0:
            bucket_data.append({
                'Window': wlabel, 'A (Regime)': 0, 'B (Adverse)': 0,
                'C (Stalled)': 0, 'D (RR)': 0, 'TP-loss': 0, 'SL': 0,
            })
            continue
        losses['bucket'] = losses.apply(engine.classify_loss_bucket, axis=1)
        counts = losses['bucket'].value_counts()
        bucket_data.append({
            'Window': wlabel,
            'A (Regime)': counts.get('A', 0),
            'B (Adverse)': counts.get('B', 0),
            'C (Stalled)': counts.get('C', 0),
            'D (RR)': counts.get('D', 0),
            'TP-loss': counts.get('TP-loss', 0),
            'SL': counts.get('SL', 0),
        })
    if bucket_data:
        st.dataframe(pd.DataFrame(bucket_data).set_index('Window'), use_container_width=True)


def render_instrument_table(results_dict, window_keys, inst_list):
    """Render per-instrument summary across windows."""
    inst_rows = []
    for inst in inst_list:
        inst_pnl = 0
        inst_n = 0
        inst_wins = 0
        for wkey in window_keys:
            trades = results_dict[wkey]['trades']
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


def render_trade_chart(trades_df, inst, chart_df, date_col='date', title_suffix=''):
    """Regime-labeled trade chart with entry/exit markers.
    Works for both daily (date_col='date') and intraday (date_col='datetime')."""
    inst_trades = trades_df[trades_df['instrument'] == inst] if len(trades_df) > 0 else pd.DataFrame()

    if inst not in chart_df.columns or chart_df[inst].isna().all():
        st.info(f"No data for {inst}{title_suffix}")
        return

    fig = go.Figure()

    # Spread price line
    fig.add_trace(go.Scatter(
        x=chart_df[date_col], y=chart_df[inst],
        mode='lines', name=inst,
        line=dict(color='#fafafa', width=1.5),
    ))

    # Shape background bands
    shapes_list = []
    if 'shape' in chart_df.columns:
        prev_shape = None
        band_start = None
        for _, row in chart_df.iterrows():
            s = str(row['shape'])
            if s != prev_shape:
                if prev_shape is not None and band_start is not None:
                    shapes_list.append(dict(
                        type="rect", xref="x", yref="paper",
                        x0=band_start, x1=row[date_col], y0=0, y1=1,
                        fillcolor=SHAPE_COLORS.get(prev_shape, '#555555'),
                        opacity=0.15, line_width=0,
                    ))
                band_start = row[date_col]
                prev_shape = s
        if prev_shape is not None and band_start is not None:
            shapes_list.append(dict(
                type="rect", xref="x", yref="paper",
                x0=band_start, x1=chart_df[date_col].iloc[-1], y0=0, y1=1,
                fillcolor=SHAPE_COLORS.get(prev_shape, '#555555'),
                opacity=0.15, line_width=0,
            ))

    # Entry/exit markers
    if len(inst_trades) > 0:
        for _, t in inst_trades.iterrows():
            is_win = t['net_pnl'] > 0
            marker_color = '#2ca02c' if is_win else '#d62728'
            entry_x = t.get('entry_datetime', t['entry_date'])
            exit_x = t['exit_date']

            # Entry
            fig.add_trace(go.Scatter(
                x=[entry_x], y=[t['entry_spread']],
                mode='markers', name='',
                marker=dict(
                    symbol='triangle-up' if t['direction'] == 'LONG' else 'triangle-down',
                    size=10, color=marker_color,
                    line=dict(width=1, color='white')),
                hovertext=f"ENTRY {t['direction']}<br>Z: {t['entry_z']:.2f}<br>Spread: {t['entry_spread']:.1f}",
                hoverinfo='text', showlegend=False,
            ))

            # Scale-in markers
            n_tiers = t.get('n_tiers_fired', 0)
            if n_tiers > 0:
                fig.add_trace(go.Scatter(
                    x=[entry_x], y=[t['entry_spread']],
                    mode='markers', name='',
                    marker=dict(symbol='diamond', size=7, color='#ffff00',
                                line=dict(width=1, color='white')),
                    hovertext=f"Scale-in: {n_tiers} tier(s), {t['total_lots']} total lots",
                    hoverinfo='text', showlegend=False,
                ))

            # Exit
            exit_color = EXIT_COLORS.get(t['exit_reason'], '#888888')
            exit_z_str = f"Z: {t['exit_z']:.2f}" if not pd.isna(t.get('exit_z', np.nan)) else ""
            fig.add_trace(go.Scatter(
                x=[exit_x], y=[t['exit_spread']],
                mode='markers', name='',
                marker=dict(symbol='x', size=10, color=exit_color,
                            line=dict(width=2, color=exit_color)),
                hovertext=f"EXIT: {t['exit_reason']}<br>PnL: {t['net_pnl']:+.1f}<br>Days: {t['days_held']}<br>{exit_z_str}",
                hoverinfo='text', showlegend=False,
            ))

            # Stop-loss marker (distinct)
            if t['exit_reason'] == 'stop_loss':
                fig.add_trace(go.Scatter(
                    x=[exit_x], y=[t['exit_spread']],
                    mode='markers', name='',
                    marker=dict(symbol='hexagon', size=12, color='#e377c2',
                                line=dict(width=2, color='white')),
                    hovertext=f"STOP LOSS<br>PnL: {t['net_pnl']:+.1f}",
                    hoverinfo='text', showlegend=False,
                ))

    fig.update_layout(
        shapes=shapes_list,
        height=500,
        plot_bgcolor=DARK_PLOT,
        paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        xaxis=dict(gridcolor=DARK_GRID, title='Date'),
        yaxis=dict(gridcolor=DARK_GRID, title=f'{inst} Spread'),
        margin=dict(l=60, r=30, t=40, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Trade table
    if len(inst_trades) > 0:
        st.markdown(f"**{len(inst_trades)} trades** for {inst}")
        display_cols = ['entry_date', 'exit_date', 'direction', 'entry_spread', 'exit_spread',
                        'entry_z', 'exit_z', 'days_held', 'exit_reason', 'gross_pnl', 'net_pnl',
                        'total_lots', 'max_abs_z']
        display_cols = [c for c in display_cols if c in inst_trades.columns]
        st.dataframe(inst_trades[display_cols].reset_index(drop=True), use_container_width=True)

    # Shape legend
    st.markdown("**Shape Legend:** " + " | ".join(
        f"<span style='color:{c}'>{SHAPE_NAMES.get(k, k)}</span>"
        for k, c in SHAPE_COLORS.items()
    ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 1 — DAILY (4 WALK-FORWARD WINDOWS)
# ══════════════════════════════════════════════════════════════

def display_daily_tab(results):
    """Display daily walk-forward results across all 4 windows."""
    if results is None:
        st.info("Click 'Run Backtest' to compute results, or wait for cached default to load.")
        return

    # Summary metrics — all 4 windows side by side
    st.markdown("### Summary Metrics")

    sharpes = []
    for w in engine.WINDOWS:
        s = results[w['name']].get('adj_sharpe', np.nan)
        if not (isinstance(s, float) and np.isnan(s)):
            sharpes.append(s)
    worst_sharpe = min(sharpes) if sharpes else np.nan

    cols = st.columns(4)
    for i, w in enumerate(engine.WINDOWS):
        wname = w['name']
        r = results[wname]
        m = r['metrics']
        sharpe = r['adj_sharpe']
        is_worst = (not np.isnan(sharpe) and sharpes and abs(sharpe - worst_sharpe) < 0.001)
        with cols[i]:
            render_metric_card(w['label'], m, sharpe, is_worst)

    if sharpes:
        st.caption(f"Worst-case window: adj Sharpe = {worst_sharpe:.3f}")

    # Exit reason breakdown
    st.markdown("### Exit Reason Breakdown")
    wkeys = [w['name'] for w in engine.WINDOWS]
    wlabels = [w['label'] for w in engine.WINDOWS]
    render_exit_table(results, wkeys, wlabels)

    # Loss bucket breakdown
    st.markdown("### Loss Bucket Breakdown")
    st.caption("A=Regime break | B=Adverse extension (max|z|>entry|z|) | C=Stalled (max|z|<=entry|z|) | D=Regime-risk exit | TP-loss=Cost-induced")
    render_loss_buckets(results, wkeys, wlabels)

    # Per-instrument summary
    st.markdown("### Per-Instrument Summary")
    render_instrument_table(results, wkeys, instruments)

    # Trade chart
    st.markdown("### Regime-Labeled Trade Chart")
    col1, col2 = st.columns(2)
    with col1:
        sel_window = st.selectbox("Window", wlabels, key="daily_chart_window")
    with col2:
        sel_inst = st.selectbox("Instrument", instruments, key="daily_chart_inst")

    w = next(w for w in engine.WINDOWS if w['label'] == sel_window)
    trades = results[w['name']]['trades']

    try:
        df, _ = get_panel()
    except Exception:
        st.warning("Panel not loaded.")
        return

    ts = pd.Timestamp(w['test_start'])
    te = pd.Timestamp(w['test_end'])
    chart_df = df[(df['date'] >= ts) & (df['date'] <= te)].copy()
    render_trade_chart(trades, sel_inst, chart_df, date_col='date')


# ══════════════════════════════════════════════════════════════
# TAB 2 — 60-MIN INTRADAY (2024-2026)
# ══════════════════════════════════════════════════════════════

def display_intraday_tab(results):
    """Display 60-min intraday results for 2024-2026."""
    if results is None:
        st.info("Click 'Run Backtest' to compute intraday results.")
        return

    r = results['intraday']
    m = r['metrics']
    sharpe = r['adj_sharpe']

    # Summary metrics — single window
    st.markdown("### Summary Metrics (2024-01-01 to latest available)")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        render_metric_card("Intraday (2024-2026)", m, sharpe)
    with col2:
        st.markdown(f"""
        <div style="background: {DARK_PLOT}; padding: 12px; border-radius: 8px;">
            <h4 style="margin:0; color: {DARK_TEXT};">Exit Reasons</h4>
            <p style="margin:2px 0; color: {DARK_TEXT};">Take Profit: <b>{m['pct_take_profit']}%</b></p>
            <p style="margin:2px 0; color: {DARK_TEXT};">Invalidated: <b>{m['pct_invalidated']}%</b></p>
            <p style="margin:2px 0; color: {DARK_TEXT};">Time Stop: <b>{m['pct_time_stop']}%</b></p>
            <p style="margin:2px 0; color: {DARK_TEXT};">Regime Risk: <b>{m['pct_regime_risk']}%</b></p>
            <p style="margin:2px 0; color: {DARK_TEXT};">Stop Loss: <b>{m.get('pct_stop_loss', 0)}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        # Loss buckets
        trades = r['trades']
        if len(trades) > 0:
            losses = trades[trades['net_pnl'] <= 0].copy()
            if len(losses) > 0:
                losses['bucket'] = losses.apply(engine.classify_loss_bucket, axis=1)
                counts = losses['bucket'].value_counts()
                st.markdown(f"""
                <div style="background: {DARK_PLOT}; padding: 12px; border-radius: 8px;">
                    <h4 style="margin:0; color: {DARK_TEXT};">Loss Buckets</h4>
                    <p style="margin:2px 0; color: {DARK_TEXT};">A (Regime): <b>{counts.get('A', 0)}</b></p>
                    <p style="margin:2px 0; color: {DARK_TEXT};">B (Adverse): <b>{counts.get('B', 0)}</b></p>
                    <p style="margin:2px 0; color: {DARK_TEXT};">C (Stalled): <b>{counts.get('C', 0)}</b></p>
                    <p style="margin:2px 0; color: {DARK_TEXT};">D (RR): <b>{counts.get('D', 0)}</b></p>
                    <p style="margin:2px 0; color: {DARK_TEXT};">TP-loss: <b>{counts.get('TP-loss', 0)}</b></p>
                    <p style="margin:2px 0; color: {DARK_TEXT};">SL: <b>{counts.get('SL', 0)}</b></p>
                </div>
                """, unsafe_allow_html=True)

    # Per-instrument summary
    st.markdown("### Per-Instrument Summary")
    render_instrument_table(results, ['intraday'], instruments)

    # Trade chart — 60-min resolution
    st.markdown("### Regime-Labeled Trade Chart (60-Min)")
    sel_inst = st.selectbox("Instrument", instruments, key="intraday_chart_inst")

    try:
        df, _ = get_panel()
        intraday_df = get_intraday()
    except Exception:
        st.warning("Data not loaded.")
        return

    # Build chart df from intraday data with shape from daily panel
    ts = pd.Timestamp(engine.INTRADAY_START)
    intra_chart = intraday_df[intraday_df['date'] >= ts].copy()

    # Merge daily shape onto intraday bars
    daily_shapes = df[['date', 'shape']].copy()
    daily_shapes = daily_shapes.rename(columns={'date': 'date_key'})
    intra_chart['date_key'] = intra_chart['date']
    intra_chart = pd.merge(intra_chart, daily_shapes, on='date_key', how='left')
    intra_chart = intra_chart.drop(columns=['date_key'])

    render_trade_chart(r['trades'], sel_inst, intra_chart, date_col='datetime',
                       title_suffix=' (60-min)')


# ══════════════════════════════════════════════════════════════
# MAIN FLOW
# ══════════════════════════════════════════════════════════════

# Initialize session state
if 'daily_results' not in st.session_state:
    cached = load_cached()
    if cached:
        st.session_state['daily_results'] = cached
        st.session_state['daily_label'] = "Locked Config (spread=22, butterfly=44 MYR)"
    else:
        st.session_state['daily_results'] = None
        st.session_state['daily_label'] = ""

if 'intraday_results' not in st.session_state:
    st.session_state['intraday_results'] = None

# Handle run button
if run_button:
    if not instruments:
        st.error("Select at least one instrument.")
    else:
        config = build_config()

        # Daily
        if is_default_config():
            cached = load_cached()
            if cached:
                st.session_state['daily_results'] = cached
                st.session_state['daily_label'] = "Locked Config (cached)"
            else:
                with st.spinner("Running daily backtest..."):
                    st.session_state['daily_results'] = run_custom_daily(config)
                    st.session_state['daily_label'] = "Locked Config"
        else:
            with st.spinner("Running daily backtest with custom parameters..."):
                st.session_state['daily_results'] = run_custom_daily(config)
                st.session_state['daily_label'] = "Custom Config"

        # Intraday
        with st.spinner("Running 60-min intraday backtest..."):
            st.session_state['intraday_results'] = run_custom_intraday(config)

# Two tabs
tab_daily, tab_intraday = st.tabs([
    "Daily (4 Walk-Forward Windows)",
    "60-Min Intraday (2024-2026)",
])

with tab_daily:
    if st.session_state.get('daily_label'):
        st.subheader(st.session_state['daily_label'])
    display_daily_tab(st.session_state.get('daily_results'))

with tab_intraday:
    display_intraday_tab(st.session_state.get('intraday_results'))
