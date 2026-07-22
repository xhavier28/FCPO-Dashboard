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
# PART 2 — SPREAD-PAIR CHART (instrument × calendar-month pair)
# ══════════════════════════════════════════════════════════════

# Reverse maps: instrument → tenor column name
_SPREAD_COL_MAP = {v: k for k, v in engine.SPREAD_TENOR_MAP.items()}
_BF_COL_MAP = {v: k for k, v in engine.BUTTERFLY_TENOR_MAP.items()}


def _make_pair_label(inst, resolved):
    """Format resolved contracts as a label: 'Feb26-Mar26' (spread) or 'Feb26-Mar26-Apr26' (BF)."""
    return '-'.join(contract_month_to_str(ym) for ym in resolved)


@st.cache_data
def build_spread_pair_options():
    """Scan intraday data to find all instrument × calendar-month pair periods.
    Returns list of dicts: {instrument, pair_label, resolved, start_dt, end_dt, tenor_col}
    sorted by start date."""
    # Load both wide CSVs
    spread_wide = pd.read_csv(
        Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild/spread_tenor_close_wide.csv'),
        parse_dates=['datetime']
    )
    bf_wide = pd.read_csv(
        Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild/butterfly_tenor_close_wide.csv'),
        parse_dates=['datetime']
    )
    spread_wide = spread_wide[spread_wide['datetime'] >= engine.INTRADAY_START].copy()
    bf_wide = bf_wide[bf_wide['datetime'] >= engine.INTRADAY_START].copy()

    # Get unique trading dates (use daily resolution for option scanning)
    spread_dates = spread_wide['datetime'].dt.date.unique()
    bf_dates = bf_wide['datetime'].dt.date.unique()

    options = []

    # --- Spread instruments ---
    for inst in engine.SPREAD_INSTRUMENTS:
        tenor_col = _SPREAD_COL_MAP[inst]
        prev_resolved = None
        period_start = None
        for d in sorted(spread_dates):
            resolved = engine.resolve_contracts(inst, pd.Timestamp(d))
            if resolved != prev_resolved:
                if prev_resolved is not None:
                    options.append({
                        'instrument': inst,
                        'pair_label': _make_pair_label(inst, prev_resolved),
                        'resolved': prev_resolved,
                        'start_dt': period_start,
                        'end_dt': d,
                        'tenor_col': tenor_col,
                        'data_source': 'spread',
                    })
                prev_resolved = resolved
                period_start = d
        # Close last period
        if prev_resolved is not None:
            options.append({
                'instrument': inst,
                'pair_label': _make_pair_label(inst, prev_resolved),
                'resolved': prev_resolved,
                'start_dt': period_start,
                'end_dt': sorted(spread_dates)[-1],
                'tenor_col': tenor_col,
                'data_source': 'spread',
            })

    # --- Butterfly instruments ---
    for inst in engine.BUTTERFLY_INSTRUMENTS:
        tenor_col = _BF_COL_MAP[inst]
        prev_resolved = None
        period_start = None
        for d in sorted(bf_dates):
            resolved = engine.resolve_contracts(inst, pd.Timestamp(d))
            if resolved != prev_resolved:
                if prev_resolved is not None:
                    options.append({
                        'instrument': inst,
                        'pair_label': _make_pair_label(inst, prev_resolved),
                        'resolved': prev_resolved,
                        'start_dt': period_start,
                        'end_dt': d,
                        'tenor_col': tenor_col,
                        'data_source': 'butterfly',
                    })
                prev_resolved = resolved
                period_start = d
        if prev_resolved is not None:
            options.append({
                'instrument': inst,
                'pair_label': _make_pair_label(inst, prev_resolved),
                'resolved': prev_resolved,
                'start_dt': period_start,
                'end_dt': sorted(bf_dates)[-1],
                'tenor_col': tenor_col,
                'data_source': 'butterfly',
            })

    # Sort by start date, then instrument
    options.sort(key=lambda o: (o['start_dt'], o['instrument']))
    return options


@st.cache_data
def _load_wide_data(source):
    """Load and cache the wide CSV for a given source type."""
    if source == 'spread':
        df = pd.read_csv(
            Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild/spread_tenor_close_wide.csv'),
            parse_dates=['datetime']
        )
    else:
        df = pd.read_csv(
            Path(r'C:/ClaudeCode/research/06. stage3_minute_rebuild/butterfly_tenor_close_wide.csv'),
            parse_dates=['datetime']
        )
    return df[df['datetime'] >= engine.INTRADAY_START].copy()


def build_spread_pair_chart(option, trades_df=None):
    """Build chart for ONE instrument-pair-period with tenor-stage labels.

    Parameters
    ----------
    option : dict from build_spread_pair_options()
    trades_df : DataFrame of trades (or None)
    """
    inst = option['instrument']
    pair_label = option['pair_label']
    resolved = option['resolved']
    tenor_col = option['tenor_col']
    start_dt = pd.Timestamp(option['start_dt'])
    end_dt = pd.Timestamp(option['end_dt'])

    # Load price data
    wide_df = _load_wide_data(option['data_source'])

    # Filter to this period's date range
    mask = (wide_df['datetime'].dt.date >= start_dt.date()) & \
           (wide_df['datetime'].dt.date <= end_dt.date())
    period_df = wide_df.loc[mask, ['datetime', tenor_col]].copy()
    period_df = period_df.rename(columns={tenor_col: 'price'})
    period_df = period_df.dropna(subset=['price']).sort_values('datetime')

    if len(period_df) < 2:
        return None

    # Filter out expiry buffer zone
    near_ym = resolved[0]
    period_df = period_df[~(
        (period_df['datetime'].dt.year == near_ym[0]) &
        (period_df['datetime'].dt.month == near_ym[1]) &
        (period_df['datetime'].dt.day >= engine.EXPIRY_FORCE_CLOSE_DAY)
    )].copy()

    if len(period_df) < 2:
        return None

    fig = go.Figure()

    # Price line
    color = MONTH_COLORS[near_ym[1] - 1]
    fig.add_trace(go.Scatter(
        x=period_df['datetime'],
        y=period_df['price'],
        mode='lines',
        name=f'{inst}: {pair_label}',
        line=dict(color=color, width=2),
        hovertemplate=(
            f"{inst}: {pair_label}<br>"
            "%{x}<br>"
            "Spread: %{y:.1f}<extra></extra>"
        ),
    ))

    # ── Tenor-stage labels (months to near-leg expiry) ──
    # Compute months-to-expiry at each unique trading date
    unique_dates = period_df['datetime'].dt.date.unique()
    prev_mte = None
    tenor_boundaries = []  # list of (date, months_to_expiry)
    for d in sorted(unique_dates):
        mte = (near_ym[0] - d.year) * 12 + (near_ym[1] - d.month)
        if mte != prev_mte:
            tenor_boundaries.append((d, mte))
            prev_mte = mte

    # Add vertical lines and annotations at tenor transitions
    for i, (boundary_date, mte) in enumerate(tenor_boundaries):
        boundary_dt = pd.Timestamp(boundary_date)
        label = f"{mte}m" if mte > 0 else "Exp"

        # Vertical dashed line at each tenor boundary
        fig.add_vline(
            x=boundary_dt,
            line=dict(color=DARK_GRID, width=1, dash='dot'),
        )

        # Annotation at top of chart
        fig.add_annotation(
            x=boundary_dt,
            y=1.0,
            yref='paper',
            text=label,
            showarrow=False,
            font=dict(size=11, color='#aaaaaa'),
            yshift=10,
        )

    # ── Trade markers ──
    if trades_df is not None and len(trades_df) > 0:
        matched_trades = []
        for _, t in trades_df.iterrows():
            if t['instrument'] != inst:
                continue
            entry_d = pd.Timestamp(t['entry_date'])
            t_resolved = engine.resolve_contracts(inst, entry_d)
            if t_resolved == resolved:
                matched_trades.append(t)

        for t in matched_trades:
            entry_dt = t.get('entry_datetime', t['entry_date'])
            exit_dt = t['exit_date']
            reason = t['exit_reason']

            # Entry marker
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
                    f"Inst: {inst}<br>"
                    f"Z: {t['entry_z']:.2f}<br>"
                    f"Spread: {t['entry_spread']:.1f}"
                ),
                hoverinfo='text',
                showlegend=False,
            ))

            # Exit marker
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
                    f"Inst: {inst}<br>"
                    f"PnL: {t['net_pnl']:+.1f}<br>"
                    f"Days: {t['days_held']}<br>{exit_z_str}"
                ),
                hoverinfo='text',
                showlegend=False,
            ))

        # Legend entries (invisible traces)
        if matched_trades:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers', name='Entry',
                marker=dict(symbol='triangle-up', size=10, color='#00ff00'),
            ))
            for reason, clr in EXIT_COLORS.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    name=f'Exit: {EXIT_LABELS.get(reason, reason)}',
                    marker=dict(symbol='hexagon' if reason == 'expiry_buffer' else 'x',
                                size=10, color=clr),
                ))

    fig.update_layout(
        title=f"{inst}: {pair_label} — Spread Price Through Tenor Lifecycle",
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

with st.expander("Spread-Pair Tenor Tracking", expanded=False):
    st.caption(
        "Select an instrument × calendar-month pair to see its spread price. "
        "Tenor-stage labels (Xm) show months-to-near-leg-expiry. "
        "Entry/exit markers from backtest results (if available) are filtered to "
        "ONLY the trades matching this exact instrument-pair-period."
    )
    try:
        pair_options = build_spread_pair_options()
        if pair_options:
            # Count trades per option (from current backtest results if available)
            v2_res = st.session_state.get('v2_results')
            trades_for_chart = v2_res['trades'] if v2_res is not None else None

            # Count trades per option
            trade_counts = {}
            if trades_for_chart is not None and len(trades_for_chart) > 0:
                for _, t in trades_for_chart.iterrows():
                    inst_t = t['instrument']
                    entry_d = pd.Timestamp(t['entry_date'])
                    t_res = engine.resolve_contracts(inst_t, entry_d)
                    for i, opt in enumerate(pair_options):
                        if opt['instrument'] == inst_t and opt['resolved'] == t_res:
                            trade_counts[i] = trade_counts.get(i, 0) + 1
                            break

            # Filter toggle
            only_traded = st.checkbox("Show only pairs with trades", value=True,
                                       key='v2_only_traded')

            # Build filtered index + labels
            filtered_indices = []
            display_labels = []
            for i, opt in enumerate(pair_options):
                n_trades = trade_counts.get(i, 0)
                if only_traded and n_trades == 0:
                    continue
                trade_str = f" ({n_trades} trades)" if n_trades > 0 else ""
                filtered_indices.append(i)
                display_labels.append(
                    f"{opt['instrument']}: {opt['pair_label']}{trade_str}"
                )

            if not filtered_indices:
                st.info("No pairs with trades found. Uncheck filter or run backtest first.")
            else:
                sel_pos = st.selectbox(
                    "Instrument × Pair",
                    options=range(len(filtered_indices)),
                    format_func=lambda i: display_labels[i],
                    index=0,
                    key='v2_pair_sel',
                )
                sel_idx = filtered_indices[sel_pos]

                selected_option = pair_options[sel_idx]
                fig_pair = build_spread_pair_chart(selected_option, trades_for_chart)
                if fig_pair:
                    st.plotly_chart(fig_pair, use_container_width=True)
                else:
                    st.warning(f"Not enough data for {selected_option['instrument']}: {selected_option['pair_label']}.")
        else:
            st.warning("No spread-pair data available.")
    except Exception as e:
        st.error(f"Error building spread-pair chart: {e}")

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

    # ──────────────────────────────────────────────────────────────
    # PART C: Entry/Exit Z-Score Sensitivity Table
    # ──────────────────────────────────────────────────────────────

    st.markdown("### Entry/Exit Z-Score Sensitivity")
    st.caption(
        "Sweeps entry z-score (1.0–2.5) × exit z-score (0.0–0.5) "
        "using current dashboard settings for all other parameters. "
        "Recomputes live when parameters change."
    )

    LOW_N_THRESHOLD = 30

    if st.button("Run Sensitivity Sweep", key='v2_run_sweep'):
        with st.spinner("Running 42-cell sensitivity sweep..."):
            sweep_rows = []
            current_config = result.get('config', build_config())
            entry_vals = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
            exit_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

            df_panel, tm_cache_panel = get_panel()
            intraday_data = get_intraday()

            for ez in entry_vals:
                for xz in exit_vals:
                    sweep_config = current_config.copy()
                    sweep_config['entry_z'] = ez
                    sweep_config['exit_z'] = xz
                    sweep_res = engine.run_backtest_intraday(
                        sweep_config, df_panel, tm_cache_panel, intraday_data)
                    sweep_trades = sweep_res['intraday']
                    sweep_m = engine.compute_metrics(
                        sweep_trades, df_panel, engine.INTRADAY_START, engine.INTRADAY_END)
                    sweep_sharpe = engine.compute_daily_portfolio_sharpe_configurable(
                        sweep_trades, engine.INTRADAY_START, engine.INTRADAY_END,
                        df_panel, sweep_config)

                    is_baseline = (ez == 1.5 and xz == 0.5)
                    sweep_rows.append({
                        'Entry Z': ez,
                        'Exit Z': xz,
                        'Trades': sweep_m['n_trades'],
                        'Win%': sweep_m['win_rate'],
                        'PnL': sweep_m['total_pnl'],
                        'Adj Sharpe': sweep_sharpe if not (isinstance(sweep_sharpe, float) and np.isnan(sweep_sharpe)) else 0.0,
                        'Max DD': sweep_m['max_dd'],
                        'Avg HP': sweep_m['avg_hp'],
                        '_is_baseline': is_baseline,
                        '_low_n': sweep_m['n_trades'] < LOW_N_THRESHOLD,
                    })

            sweep_df = pd.DataFrame(sweep_rows)

            # Find best Sharpe row
            best_idx = sweep_df['Adj Sharpe'].idxmax()

            st.session_state['v2_sweep'] = sweep_df
            st.session_state['v2_sweep_best_idx'] = best_idx

    # Display sweep results if available
    sweep_df = st.session_state.get('v2_sweep')
    if sweep_df is not None:
        best_idx = st.session_state.get('v2_sweep_best_idx', -1)

        def style_sweep(row):
            styles = [''] * len(row)
            if row.name == best_idx:
                styles = ['background-color: rgba(44, 160, 44, 0.3)'] * len(row)
            elif row.get('_is_baseline', False):
                styles = ['background-color: rgba(31, 119, 180, 0.3)'] * len(row)
            if row.get('_low_n', False):
                styles = ['color: #888888; font-style: italic'] * len(row)
            return styles

        display_sweep = sweep_df.drop(columns=['_is_baseline', '_low_n'])
        styled_sweep = sweep_df.style.apply(style_sweep, axis=1).format({
            'Entry Z': '{:.2f}', 'Exit Z': '{:.1f}',
            'Win%': '{:.1f}', 'PnL': '{:+.1f}',
            'Adj Sharpe': '{:.3f}', 'Max DD': '{:+.1f}', 'Avg HP': '{:.1f}',
        })
        # Hide internal columns
        styled_sweep = styled_sweep.hide(subset=['_is_baseline', '_low_n'], axis='columns')
        st.dataframe(styled_sweep, use_container_width=True, height=500)

        best_row = sweep_df.iloc[best_idx]
        baseline_row = sweep_df[sweep_df['_is_baseline']]
        st.caption(
            f"Green = best Sharpe (entry={best_row['Entry Z']:.2f}, exit={best_row['Exit Z']:.1f}, "
            f"Sharpe={best_row['Adj Sharpe']:.3f}) | "
            f"Blue = baseline (1.5/0.5) | "
            f"Grey italic = low-N (<{LOW_N_THRESHOLD} trades)"
        )
