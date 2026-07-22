"""
FCPO Calendar Spread — Minute-Data Debug Dashboard (v2)
========================================================
Single-tab, minute-data-only dashboard for inspecting exit reasons
and interactively tweaking entry/exit z-score thresholds.

Reuses the validated tenor-ladder backtest engine from
MRBackTest/engine/backtest_engine.py — no duplicated logic.

Run: python -m streamlit run "MRBackTest/FCPO Backtest Dashboard/app_v2.py" --server.port 8507
v1 dashboard (app.py) runs on port 8506 — NOT modified by this file.
"""

import sys
sys.path.insert(0, r'C:/ClaudeCode')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import importlib.util

# Import engine (same approach as v1 to avoid signal.py conflict)
_spec = importlib.util.spec_from_file_location(
    'backtest_engine',
    str(Path(__file__).parent.parent / 'engine' / 'backtest_engine.py')
)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

# Import tenor mapping for calendar-month labeling
from MRBackTest.shared.tenor_mapping import (
    front_month, tenor_to_contract_month, contract_month_to_str, add_months
)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

DARK_BG   = "#0e1117"
DARK_PLOT = "#262730"
DARK_GRID = "#3a3a4a"
DARK_TEXT = "#fafafa"

EXIT_COLORS = {
    'take_profit':   '#2ca02c',
    'invalidated':   '#ff7f0e',
    'time_stop':     '#d62728',
    'regime_risk':   '#9467bd',
    'stop_loss':     '#e377c2',
    'expiry_buffer': '#17becf',
}

EXIT_LABELS = {
    'take_profit':   'Take Profit',
    'invalidated':   'Invalidated',
    'time_stop':     'Time Stop',
    'regime_risk':   'Regime Risk',
    'stop_loss':     'Stop Loss',
    'expiry_buffer': 'Expiry Buffer',
}

# Reference numbers from confirmed sensitivity sweep (2026-07-21)
REF_BASELINE = {
    'entry_z': 1.5, 'exit_z': 0.5,
    'n': 113, 'win_rate': 52.2, 'pnl': 408, 'adj_sharpe': 0.693, 'max_dd': -160,
}
REF_BEST = {
    'entry_z': 1.75, 'exit_z': 0.25,
    'n': 80, 'adj_sharpe': 0.974, 'pnl': 407, 'max_dd': -96,
}

# Calendar-month color palette (12 months cycling)
MONTH_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
]

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FCPO v2 — Minute Data Debug",
    page_icon="🔍",
    layout="wide",
)

st.title("FCPO Calendar Spread — Minute-Data Debug Dashboard (v2)")
st.caption("Single-purpose tool: inspect exit reasons + tweak thresholds against 60-min intraday data (2024-2026)")

# ══════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Parameters")

    st.subheader("Entry / Exit")
    entry_z = st.number_input("Entry Z-score", min_value=0.5, max_value=4.0,
                               value=1.5, step=0.25, format="%.2f")
    exit_z = st.number_input("Exit Z-score", min_value=0.0, max_value=2.0,
                              value=0.5, step=0.05, format="%.2f")
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
        enabled = st.checkbox(f"Tier {i+1}", value=False, key=f"v2_tier_en_{i}")
        if enabled:
            col1, col2 = st.columns(2)
            with col1:
                z_lvl = st.number_input(f"T{i+1} Z", min_value=1.5, max_value=4.0,
                                         value=1.75 + i * 0.25, step=0.25,
                                         format="%.2f", key=f"v2_tier_z_{i}")
            with col2:
                lots = st.number_input(f"T{i+1} lots", min_value=1, max_value=5,
                                        value=1, key=f"v2_tier_lots_{i}")
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

    st.divider()

    col_run, col_best = st.columns(2)
    with col_run:
        run_button = st.button("Run Backtest", type="primary", use_container_width=True)
    with col_best:
        jump_best = st.button("Best Known", use_container_width=True,
                               help="Jump to entry=1.75, exit=0.25 (adj Sharpe 0.974)")

    st.divider()
    st.caption(f"POINT_VALUE = {engine.POINT_VALUE} MYR/point")
    st.caption(f"Spread cost: {cost_spread/engine.POINT_VALUE:.2f} pts/lot")
    st.caption(f"Butterfly cost: {cost_butterfly/engine.POINT_VALUE:.2f} pts/lot")


# ══════════════════════════════════════════════════════════════
# CONFIG BUILDER
# ══════════════════════════════════════════════════════════════

def build_config(override_entry_z=None, override_exit_z=None):
    return {
        'entry_z': override_entry_z if override_entry_z is not None else entry_z,
        'exit_z': override_exit_z if override_exit_z is not None else exit_z,
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
        'data_resolution': '60min',
        'date_range': 'intraday',
    }


# ══════════════════════════════════════════════════════════════
# DATA / COMPUTE
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def get_panel():
    return engine.build_panel()


@st.cache_resource
def get_intraday():
    return engine.load_intraday_data()


def run_intraday(config):
    df, tm_cache = get_panel()
    intraday_df = get_intraday()
    results = engine.run_backtest_intraday(config, df, tm_cache, intraday_df)
    trades = results['intraday']
    metrics = engine.compute_metrics(trades, df, engine.INTRADAY_START, engine.INTRADAY_END)
    adj_sharpe = engine.compute_daily_portfolio_sharpe_configurable(
        trades, engine.INTRADAY_START, engine.INTRADAY_END, df, config)
    metrics['adj_sharpe'] = adj_sharpe
    return {
        'n': metrics['n_trades'], 'pnl': metrics['total_pnl'],
        'adj_sharpe': adj_sharpe, 'metrics': metrics, 'trades': trades,
        'config': config,
    }


# ══════════════════════════════════════════════════════════════
# PART 2 — CALENDAR-MONTH TENOR TRACKING CHART
# ══════════════════════════════════════════════════════════════

@st.cache_data
def build_calendar_month_data():
    """Build calendar-month price data from spread tenor wide CSV.
    Returns (cal_df, sorted list of contract labels)."""
    spread_wide = pd.read_csv(
        Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild/spread_tenor_close_wide.csv'),
        parse_dates=['datetime']
    )
    spread_wide = spread_wide[spread_wide['datetime'] >= engine.INTRADAY_START].copy()

    tenor_cols = [c for c in spread_wide.columns if c != 'datetime']

    records = []
    for _, row in spread_wide.iterrows():
        dt = row['datetime']
        d = dt.date() if hasattr(dt, 'date') else dt
        for tenor in tenor_cols:
            price = row[tenor]
            if pd.isna(price):
                continue
            if tenor == 'Current':
                offset = 0
            elif tenor.startswith('+') and tenor.endswith('M'):
                offset = int(tenor[1:-1])
            else:
                continue
            contract_ym = tenor_to_contract_month(d, offset, "spread")
            contract_label = contract_month_to_str(contract_ym)
            records.append({
                'datetime': dt,
                'contract': contract_label,
                'contract_ym': contract_ym,
                'price': price,
                'tenor_offset': offset,
            })

    if not records:
        return None, []

    cal_df = pd.DataFrame(records)

    # Filter out expiry buffer zone
    def is_in_buffer(row):
        d = row['datetime']
        cy, cm = row['contract_ym']
        return d.year == cy and d.month == cm and d.day >= engine.EXPIRY_FORCE_CLOSE_DAY

    cal_df['in_buffer'] = cal_df.apply(is_in_buffer, axis=1)
    cal_df = cal_df[~cal_df['in_buffer']].copy()

    # Sort contracts chronologically
    contracts_first = cal_df.groupby('contract').first().sort_values('contract_ym')
    sorted_contracts = contracts_first.index.tolist()

    return cal_df, sorted_contracts


def build_single_month_chart(cal_df, selected_contract, trades_df=None):
    """Build a chart for ONE selected delivery month with entry/exit markers."""
    cdata = cal_df[cal_df['contract'] == selected_contract].sort_values('datetime')
    if len(cdata) < 2:
        return None

    ym = cdata['contract_ym'].iloc[0]
    color = MONTH_COLORS[ym[1] - 1]

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=cdata['datetime'],
        y=cdata['price'],
        mode='lines',
        name=selected_contract,
        line=dict(color=color, width=2),
        hovertemplate=f"{selected_contract}<br>%{{x}}<br>Spread: %{{y:.1f}}<extra></extra>",
    ))

    # Overlay entry/exit markers from trades if available
    if trades_df is not None and len(trades_df) > 0:
        # Find trades on ALL spread instruments whose near-leg contract matches this month
        # A trade on M1-M2 with near_offset=0 → near_ym = front_month(entry_date)
        # A trade on M2-M3 with near_offset=1 → near_ym = front_month(entry_date) + 1
        # The selected contract is a M1-M2 spread for a specific calendar month.
        # Match trades where the instrument resolves to contracts involving this month.
        matched_trades = []
        for _, t in trades_df.iterrows():
            inst = t['instrument']
            if inst not in engine.INSTRUMENT_TENOR_OFFSETS:
                continue
            cfg = engine.INSTRUMENT_TENOR_OFFSETS[inst]
            entry_d = pd.Timestamp(t['entry_date'])
            resolved = engine.resolve_contracts(inst, entry_d)
            # Check if ANY leg matches the selected contract month
            if ym in resolved:
                matched_trades.append(t)

        for t in matched_trades:
            entry_dt = t.get('entry_datetime', t['entry_date'])
            exit_dt = t['exit_date']
            reason = t['exit_reason']

            # Entry marker — green triangle
            fig.add_trace(go.Scatter(
                x=[entry_dt],
                y=[t['entry_spread']],
                mode='markers',
                name='',
                marker=dict(
                    symbol='triangle-up' if t['direction'] == 'LONG' else 'triangle-down',
                    size=12, color='#00ff00',
                    line=dict(width=1.5, color='white'),
                ),
                hovertext=(
                    f"ENTRY {t['direction']}<br>"
                    f"Inst: {t['instrument']}<br>"
                    f"Z: {t['entry_z']:.2f}<br>"
                    f"Spread: {t['entry_spread']:.1f}"
                ),
                hoverinfo='text',
                showlegend=False,
            ))

            # Exit marker — color-coded by exit reason
            exit_color = EXIT_COLORS.get(reason, '#888888')
            exit_symbol = 'hexagon' if reason == 'expiry_buffer' else 'x'
            exit_size = 14 if reason == 'expiry_buffer' else 11
            exit_z_str = f"Z: {t['exit_z']:.2f}" if not pd.isna(t.get('exit_z', np.nan)) else ""
            fig.add_trace(go.Scatter(
                x=[exit_dt],
                y=[t['exit_spread']],
                mode='markers',
                name='',
                marker=dict(
                    symbol=exit_symbol, size=exit_size, color=exit_color,
                    line=dict(width=2, color='white'),
                ),
                hovertext=(
                    f"EXIT: {EXIT_LABELS.get(reason, reason)}<br>"
                    f"Inst: {t['instrument']}<br>"
                    f"PnL: {t['net_pnl']:+.1f}<br>"
                    f"Days: {t['days_held']}<br>{exit_z_str}"
                ),
                hoverinfo='text',
                showlegend=False,
            ))

        # Add legend entries for marker types (invisible traces for legend only)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', name='Entry',
            marker=dict(symbol='triangle-up', size=10, color='#00ff00'),
        ))
        for reason, color in EXIT_COLORS.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                name=f'Exit: {EXIT_LABELS.get(reason, reason)}',
                marker=dict(symbol='hexagon' if reason == 'expiry_buffer' else 'x',
                            size=10, color=color),
            ))

    fig.update_layout(
        title=f"{selected_contract} — Spread Price Through Tenor Lifecycle",
        height=500,
        plot_bgcolor=DARK_PLOT,
        paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        xaxis=dict(gridcolor=DARK_GRID, title='Date'),
        yaxis=dict(gridcolor=DARK_GRID, title='Spread (pts)'),
        margin=dict(l=60, r=30, t=50, b=80),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

def render_summary_metrics(result):
    """Render portfolio-level summary metrics."""
    m = result['metrics']
    sharpe = result['adj_sharpe']
    pnl_color = '#2ca02c' if m['total_pnl'] > 0 else '#d62728'

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trades", m['n_trades'])
        st.metric("Win Rate", f"{m['win_rate']}%")
    with col2:
        st.metric("Total PnL", f"{m['total_pnl']:+,.1f} pts")
        st.metric("Max DD", f"{m['max_dd']:+,.1f}")
    with col3:
        adj_str = f"{sharpe:.3f}" if not (isinstance(sharpe, float) and np.isnan(sharpe)) else "N/A"
        naive_str = f"{m.get('naive_sharpe', np.nan):.3f}" if not np.isnan(m.get('naive_sharpe', np.nan)) else "N/A"
        st.metric("Adj Sharpe", adj_str)
        st.metric("Naive Sharpe", naive_str)
    with col4:
        st.metric("Avg Holding Period", f"{m['avg_hp']} days")
        st.metric("Avg Win / Avg Loss", f"{m['avg_win']:+.1f} / {m['avg_loss']:+.1f}")


def render_exit_reason_summary(trades):
    """Render exit reason breakdown with expiry_buffer highlighted."""
    if len(trades) == 0:
        st.info("No trades to display.")
        return

    reasons = trades['exit_reason'].value_counts()
    total = len(trades)

    rows = []
    for reason in ['take_profit', 'invalidated', 'time_stop', 'regime_risk',
                    'stop_loss', 'expiry_buffer']:
        count = reasons.get(reason, 0)
        pct = round(count / total * 100, 1) if total > 0 else 0
        rows.append({
            'Exit Reason': EXIT_LABELS.get(reason, reason),
            'Count': count,
            '%': f"{pct}%",
        })

    df_exit = pd.DataFrame(rows)
    st.dataframe(df_exit.set_index('Exit Reason'), use_container_width=True)


def render_exit_reason_chart(trades):
    """Pie chart of exit reasons."""
    if len(trades) == 0:
        return
    reasons = trades['exit_reason'].value_counts()
    labels = [EXIT_LABELS.get(r, r) for r in reasons.index]
    colors = [EXIT_COLORS.get(r, '#888888') for r in reasons.index]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=reasons.values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hole=0.3,
    )])
    fig.update_layout(
        height=300,
        plot_bgcolor=DARK_PLOT,
        paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_per_instrument_table(trades):
    """Per-instrument summary with exit reason breakdown."""
    if len(trades) == 0:
        st.info("No trades.")
        return

    inst_rows = []
    for inst in engine.ALL_9:
        it = trades[trades['instrument'] == inst]
        n = len(it)
        if n == 0:
            continue
        wins = (it['net_pnl'] > 0).sum()
        wr = round(wins / n * 100, 1)
        pnl = round(it['net_pnl'].sum(), 1)
        eb_count = (it['exit_reason'] == 'expiry_buffer').sum()
        tp_count = (it['exit_reason'] == 'take_profit').sum()
        inv_count = (it['exit_reason'] == 'invalidated').sum()
        ts_count = (it['exit_reason'] == 'time_stop').sum()
        rr_count = (it['exit_reason'] == 'regime_risk').sum()
        sl_count = (it['exit_reason'] == 'stop_loss').sum()
        inst_rows.append({
            'Instrument': inst,
            'Trades': n,
            'Win%': f"{wr}%",
            'PnL': pnl,
            'TP': tp_count,
            'Inv': inv_count,
            'TS': ts_count,
            'RR': rr_count,
            'SL': sl_count,
            'Expiry': eb_count,
        })

    if inst_rows:
        df_inst = pd.DataFrame(inst_rows)
        st.dataframe(df_inst.set_index('Instrument'), use_container_width=True)


def render_trade_table(trades, highlight_expiry=True):
    """Per-trade detail table with exit reason inspection.
    Expiry buffer trades visually highlighted."""
    if len(trades) == 0:
        st.info("No trades.")
        return

    display_cols = ['instrument', 'entry_date', 'exit_date', 'direction',
                    'entry_spread', 'exit_spread', 'entry_z', 'exit_z',
                    'days_held', 'exit_reason', 'gross_pnl', 'net_pnl',
                    'total_lots', 'max_abs_z', 'pinned_contracts']
    display_cols = [c for c in display_cols if c in trades.columns]

    display_df = trades[display_cols].copy().reset_index(drop=True)

    # Format dates
    for col in ['entry_date', 'exit_date']:
        if col in display_df.columns:
            display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d')

    # Round numeric columns
    for col in ['entry_spread', 'exit_spread', 'entry_z', 'exit_z',
                'gross_pnl', 'net_pnl', 'max_abs_z']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    # Highlight expiry_buffer rows
    def highlight_expiry_rows(row):
        if row.get('exit_reason') == 'expiry_buffer':
            return ['background-color: rgba(23, 190, 207, 0.25)'] * len(row)
        return [''] * len(row)

    if highlight_expiry:
        styled = display_df.style.apply(highlight_expiry_rows, axis=1)
        st.dataframe(styled, use_container_width=True, height=400)
    else:
        st.dataframe(display_df, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════
# MAIN FLOW
# ══════════════════════════════════════════════════════════════

# Initialize session state
if 'v2_results' not in st.session_state:
    st.session_state['v2_results'] = None
    st.session_state['v2_label'] = ""

# Handle "Jump to Best Known" button
if jump_best:
    st.session_state['v2_jump_best'] = True

# Handle run button or jump-to-best
should_run = run_button
best_override = False

if st.session_state.get('v2_jump_best', False):
    should_run = True
    best_override = True
    st.session_state['v2_jump_best'] = False

if should_run:
    if not instruments:
        st.error("Select at least one instrument.")
    else:
        if best_override:
            config = build_config(override_entry_z=1.75, override_exit_z=0.25)
            label = "Best Known (entry=1.75, exit=0.25)"
        else:
            config = build_config()
            if entry_z == 1.5 and exit_z == 0.5:
                label = "Locked Baseline (entry=1.5, exit=0.5)"
            else:
                label = f"Custom (entry={config['entry_z']}, exit={config['exit_z']})"

        with st.spinner("Running 60-min intraday backtest..."):
            st.session_state['v2_results'] = run_intraday(config)
            st.session_state['v2_label'] = label

# ──────────────────────────────────────────────────────────────
# PART 2: Calendar-Month Tenor Tracking (Collapsible)
# ──────────────────────────────────────────────────────────────

with st.expander("Calendar-Month Tenor Tracking", expanded=False):
    st.caption(
        "Select a delivery month to see its spread price through the tenor lifecycle. "
        "Lines cut off at expiry-buffer deadline (day 8 of delivery month). "
        "Entry/exit markers shown when backtest results are available."
    )
    try:
        cal_data = build_calendar_month_data()
        cal_df, sorted_contracts = cal_data
        if cal_df is not None and len(sorted_contracts) > 0:
            # Default to most recent month with data
            sel_month = st.selectbox(
                "Delivery month",
                options=sorted_contracts,
                index=len(sorted_contracts) - 1,
                key='v2_cal_month',
            )
            trades_for_chart = result['trades'] if result is not None else None
            fig_cal = build_single_month_chart(cal_df, sel_month, trades_for_chart)
            if fig_cal:
                st.plotly_chart(fig_cal, use_container_width=True)
            else:
                st.warning(f"Not enough data for {sel_month}.")
        else:
            st.warning("No calendar-month data available.")
    except Exception as e:
        st.error(f"Error building calendar chart: {e}")

# ──────────────────────────────────────────────────────────────
# MAIN RESULTS DISPLAY
# ──────────────────────────────────────────────────────────────

result = st.session_state.get('v2_results')

if result is None:
    st.info("Click 'Run Backtest' to compute results with current parameters, "
            "or 'Best Known' to jump to entry=1.75/exit=0.25.")
else:
    label = st.session_state.get('v2_label', '')
    if label:
        st.subheader(label)

    # ── Summary Metrics ──
    st.markdown("### Portfolio Summary")
    render_summary_metrics(result)

    # ── Exit Reason Breakdown ──
    trades = result['trades']

    col_exit, col_chart = st.columns([1, 1])
    with col_exit:
        st.markdown("### Exit Reason Breakdown")
        render_exit_reason_summary(trades)
    with col_chart:
        st.markdown("### Exit Distribution")
        render_exit_reason_chart(trades)

    # ── Per-Instrument Summary ──
    st.markdown("### Per-Instrument Summary")
    render_per_instrument_table(trades)

    # ── Per-Trade Detail (Part 3) ──
    st.markdown("### Per-Trade Detail")
    st.caption("Rows highlighted in cyan = exited via expiry_buffer (the fix being inspected)")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_inst = st.multiselect("Filter instrument", options=['All'] + engine.ALL_9,
                                      default=['All'], key='v2_filter_inst')
    with col_f2:
        filter_reason = st.multiselect("Filter exit reason",
                                        options=['All'] + list(EXIT_LABELS.keys()),
                                        default=['All'], key='v2_filter_reason')
    with col_f3:
        date_range = st.date_input("Date range", value=[], key='v2_date_range')

    filtered = trades.copy()
    if 'All' not in filter_inst:
        filtered = filtered[filtered['instrument'].isin(filter_inst)]
    if 'All' not in filter_reason:
        filtered = filtered[filtered['exit_reason'].isin(filter_reason)]
    if len(date_range) == 2:
        d_start, d_end = date_range
        filtered = filtered[
            (pd.to_datetime(filtered['entry_date']).dt.date >= d_start) &
            (pd.to_datetime(filtered['exit_date']).dt.date <= d_end)
        ]

    st.caption(f"Showing {len(filtered)} of {len(trades)} trades")
    render_trade_table(filtered)

    # ── Filtered aggregate by exit reason ──
    if len(filtered) > 0:
        st.markdown("#### Filtered Exit Reason Aggregate")
        reasons = filtered['exit_reason'].value_counts()
        agg_rows = []
        for reason, count in reasons.items():
            pct = round(count / len(filtered) * 100, 1)
            pnl = round(filtered[filtered['exit_reason'] == reason]['net_pnl'].sum(), 1)
            agg_rows.append({
                'Exit Reason': EXIT_LABELS.get(reason, reason),
                'Count': count,
                '%': f"{pct}%",
                'Total PnL': pnl,
            })
        st.dataframe(pd.DataFrame(agg_rows).set_index('Exit Reason'), use_container_width=True)
