"""
MRBackTest — Streamlit app (3-page navigation via session_state["page"]).

Page 1: Configuration
Page 2: WF1 Results
Page 3: WF2 Results
"""

import datetime
import io
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── constants ────────────────────────────────────────────────────────────────
DARK_BG   = "#0e1117"
DARK_PLOT = "#262730"
DARK_GRID = "#3a3a4a"
DARK_TEXT = "#fafafa"

st.set_page_config(
    page_title="MRBackTest",
    page_icon="📊",
    layout="wide",
)

# ── session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "page":              "config",
        "wf1_bytes_y":       None,
        "wf1_bytes_x":       None,
        "wf2_bytes_y":       None,
        "wf2_bytes_x":       None,
        "rate_table_df":     None,
        "wf1_prep":          None,
        "wf1_breaks":        None,
        "wf1_splits":        None,
        "wf1_gating":        None,
        "wf1_rolling_coint": None,
        "wf1_rolling_hurst": None,
        "wf1_rolling_check": None,
        "wf1_grid_df":       None,
        "wf1_locked_params": None,
        "wf1_test_results":  None,
        "wf1_regime":        None,
        "wf1_kill_switches": None,
        "wf1_gate":          None,
        "wf2_prep":          None,
        "wf2_rate_warns":    None,
        "wf2_splits":        None,
        "wf2_grid_df":       None,
        "wf2_locked_params": None,
        "wf2_test_results":  None,
        "wf2_tt_settings":   None,
        "confirm_wf2":       False,
        "breaks_confirmed":  False,
        "lot_size":          1.0,
        "roundtrip_cost":    100.0,
        "mult_y":            1.0,
        "mult_x":            1.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("MRBackTest")
    st.markdown("---")

    if st.button("⚙ Configuration", use_container_width=True,
                 type="primary" if st.session_state.page == "config" else "secondary"):
        st.session_state.page = "config"
        st.rerun()

    if st.button("📈 WF1 Results", use_container_width=True,
                 type="primary" if st.session_state.page == "wf1" else "secondary",
                 disabled=st.session_state.wf1_test_results is None):
        st.session_state.page = "wf1"
        st.rerun()

    if st.button("⚡ WF2 Results", use_container_width=True,
                 type="primary" if st.session_state.page == "wf2" else "secondary",
                 disabled=st.session_state.wf2_test_results is None):
        st.session_state.page = "wf2"
        st.rerun()

    st.markdown("---")
    st.caption("FCPO vs SOY Mean Reversion")
    st.caption("Daily WF1 → 15-min WF2")


# ── helpers ──────────────────────────────────────────────────────────────────

def _badge(label, color):
    colors = {
        "green":  ("background:#198754", "color:#fff"),
        "orange": ("background:#fd7e14", "color:#000"),
        "red":    ("background:#dc3545", "color:#fff"),
        "blue":   ("background:#0d6efd", "color:#fff"),
        "gray":   ("background:#6c757d", "color:#fff"),
    }
    bg, fg = colors.get(color, ("background:#6c757d", "color:#fff"))
    return f'<span style="display:inline-block;padding:3px 10px;border-radius:4px;font-weight:bold;{bg};{fg}">{label}</span>'


def _gate_badge(gate):
    if gate == "GO":
        return _badge("GO", "green")
    elif gate == "MARGINAL":
        return _badge("MARGINAL", "orange")
    else:
        return _badge("NO-GO", "red")


def _pass_badge(passed):
    return _badge("PASS", "green") if passed else _badge("FAIL", "red")


def _plot_equity(equity_curve: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve.values,
        mode="lines",
        line=dict(color="#00d4ff", width=1.5),
        name="Equity",
    ))
    fig.add_hline(y=0, line_color="#888", line_dash="dash", line_width=0.8)
    fig.update_layout(
        title=title,
        plot_bgcolor=DARK_PLOT,
        paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        xaxis=dict(gridcolor=DARK_GRID, showgrid=True),
        yaxis=dict(gridcolor=DARK_GRID, showgrid=True, title="Net PnL (MYR)"),
        height=300,
        margin=dict(l=60, r=30, t=40, b=40),
    )
    return fig


def _plot_spread_z(spread, z_score, trades, dates, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(spread))), y=spread,
        mode="lines", line=dict(color="#aaddff", width=1),
        name="Spread", yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(z_score))), y=z_score,
        mode="lines", line=dict(color="#ffa500", width=1),
        name="Z-score", yaxis="y2",
    ))
    fig.update_layout(
        title=title,
        plot_bgcolor=DARK_PLOT,
        paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        yaxis=dict(title="Spread", gridcolor=DARK_GRID),
        yaxis2=dict(title="Z-score", overlaying="y", side="right", gridcolor=DARK_GRID),
        height=300,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    return fig


def _rate_table_has_nan(df: pd.DataFrame) -> bool:
    return df["USDMYR_Rate"].isna().any()


def _rate_table_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _rate_table_from_csv_bytes(b: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(b))
    df.columns = ["Month", "USDMYR_Rate"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def page_config():
    st.header("Configuration")

    # ── Section 1: CSV uploads ──────────────────────────────────────────────
    st.subheader("1. Data Upload")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**WF1 — Daily Data**")
        f_wf1_y = st.file_uploader("Asset A: FCPO Daily CSV", type="csv", key="up_wf1_y")
        f_wf1_x = st.file_uploader("Asset B: SOY Daily CSV",  type="csv", key="up_wf1_x")

        if f_wf1_y:
            st.session_state.wf1_bytes_y = f_wf1_y.read()
            st.success(f"FCPO: {f_wf1_y.name}")
        if f_wf1_x:
            st.session_state.wf1_bytes_x = f_wf1_x.read()
            st.success(f"SOY: {f_wf1_x.name}")

    with col2:
        st.markdown("**WF2 — 15-min Data**")
        f_wf2_y = st.file_uploader("Asset A: FCPO 15-min CSV", type="csv", key="up_wf2_y")
        f_wf2_x = st.file_uploader("Asset B: SOY 15-min CSV",  type="csv", key="up_wf2_x")

        if f_wf2_y:
            st.session_state.wf2_bytes_y = f_wf2_y.read()
            st.success(f"FCPO 15m: {f_wf2_y.name}")
        if f_wf2_x:
            st.session_state.wf2_bytes_x = f_wf2_x.read()
            st.success(f"SOY 15m: {f_wf2_x.name}")

    # ── Section 2: Params ──────────────────────────────────────────────────
    st.subheader("2. Trade Parameters")
    col3, col4, col5, col6 = st.columns(4)

    with col3:
        mult_y = st.number_input("Price multiplier A (FCPO)", value=1.0, min_value=0.001,
                                 format="%.4f", help="Multiplied to FCPO close before analysis")
    with col4:
        mult_x = st.number_input("Price multiplier B (SOY)",  value=1.0, min_value=0.001,
                                 format="%.4f", help="Multiplied to SOY close (cents/bushel) before FX")
    with col5:
        lot_size = st.number_input("Lot size (lots per trade)", value=1.0, min_value=0.01,
                                   format="%.2f")
    with col6:
        roundtrip_cost = st.number_input("Round-trip cost (MYR/trade)", value=100.0,
                                          min_value=0.0, format="%.2f")

    # ── Section 3: Monthly FX rate table (WF1) ────────────────────────────
    wf1_ready = (st.session_state.wf1_bytes_y is not None and
                 st.session_state.wf1_bytes_x is not None)

    if wf1_ready:
        st.subheader("3. Monthly USDMYR Rate Table (WF1)")
        st.caption(
            "SOY conversion: close × 0.01 × 2204.62 × USDMYR_rate. "
            "Enter one rate per month in the table below."
        )

        # Build empty table if not yet built
        if st.session_state.rate_table_df is None:
            try:
                from shared.fx_converter import build_empty_rate_table
                from engine.wf1 import _load_csv, _parse_price_series
                df_y = _load_csv(st.session_state.wf1_bytes_y)
                raw_y = _parse_price_series(df_y, "FCPO")
                df_x = _load_csv(st.session_state.wf1_bytes_x)
                raw_x = _parse_price_series(df_x, "SOY")
                common = raw_y.index.intersection(raw_x.index)
                dates  = pd.Series(common)
                st.session_state.rate_table_df = build_empty_rate_table(dates)
            except Exception as e:
                st.error(f"Could not parse CSV dates: {e}")

        if st.session_state.rate_table_df is not None:
            # Import / export buttons
            cexp, cimp = st.columns(2)
            with cexp:
                st.download_button(
                    "Export rate table CSV",
                    data=_rate_table_to_csv_bytes(st.session_state.rate_table_df),
                    file_name="usdmyr_rates.csv",
                    mime="text/csv",
                )
            with cimp:
                imported = st.file_uploader("Import rate table CSV", type="csv", key="up_rates")
                if imported:
                    try:
                        st.session_state.rate_table_df = _rate_table_from_csv_bytes(imported.read())
                        st.success("Rate table imported.")
                    except Exception as e:
                        st.error(f"Import failed: {e}")

            # Editable table
            edited_df = st.data_editor(
                st.session_state.rate_table_df,
                use_container_width=True,
                num_rows="fixed",
                key="rate_editor",
                column_config={
                    "Month":       st.column_config.TextColumn("Month (YYYY-MM)", disabled=True),
                    "USDMYR_Rate": st.column_config.NumberColumn(
                        "USDMYR Rate", min_value=0.1, max_value=20.0, format="%.4f"
                    ),
                },
            )
            # Sync back
            st.session_state.rate_table_df = edited_df

            n_missing = edited_df["USDMYR_Rate"].isna().sum()
            if n_missing > 0:
                st.warning(f"{n_missing} month(s) still missing a rate — fill all rows before running WF1.")

        # ── Section 4: WF2 inherited rate display ─────────────────────────
        wf2_ready = (st.session_state.wf2_bytes_y is not None and
                     st.session_state.wf2_bytes_x is not None)

        if wf2_ready and st.session_state.rate_table_df is not None:
            st.subheader("4. WF2 Inherited Rate Table (read-only)")
            st.info(
                "WF2 reuses the WF1 monthly rate table. "
                "Months outside the WF1 date range will use the mean fallback rate."
            )
            try:
                from shared.fx_converter import build_empty_rate_table
                from engine.wf2 import _load_csv as _load2, _parse_intraday
                df_y2  = _load2(st.session_state.wf2_bytes_y)
                raw_y2 = _parse_intraday(df_y2, "FCPO_15m")
                wf2_months_needed = build_empty_rate_table(pd.Series(raw_y2.index))
                st.dataframe(wf2_months_needed, use_container_width=True)

                from shared.fx_converter import rate_table_from_editor
                rt = rate_table_from_editor(edited_df)
                have   = set(rt.index)
                need   = set(pd.to_datetime(wf2_months_needed["Month"].astype(str) + "-01").dt.to_period("M"))
                missing_wf2 = need - have
                if missing_wf2:
                    st.warning(
                        f"WF2 months not covered by WF1 rates "
                        f"(fallback will apply): {sorted(str(m) for m in missing_wf2)}"
                    )
            except Exception as e:
                st.caption(f"(WF2 month preview error: {e})")

    # ── Run WF1 button ─────────────────────────────────────────────────────
    st.markdown("---")
    rate_ok = (
        st.session_state.rate_table_df is not None
        and not _rate_table_has_nan(st.session_state.rate_table_df)
    )
    can_run_wf1 = wf1_ready and rate_ok

    if not can_run_wf1:
        reasons = []
        if not wf1_ready:
            reasons.append("upload WF1 CSVs")
        if not rate_ok:
            reasons.append("fill all USDMYR rates")
        st.info(f"To run WF1: {' + '.join(reasons)}.")

    if st.button("Run WF1 Pipeline", disabled=not can_run_wf1, type="primary"):
        # Store params in session state so re-runs can access them
        st.session_state.lot_size       = lot_size
        st.session_state.roundtrip_cost = roundtrip_cost
        st.session_state.mult_y         = mult_y
        st.session_state.mult_x         = mult_x
        _run_wf1(mult_y=mult_y, mult_x=mult_x,
                 lot_size=lot_size, roundtrip_cost=roundtrip_cost)

    # ── WF2 gating ────────────────────────────────────────────────────────
    wf1_gate = st.session_state.wf1_gate
    if wf1_gate in ("GO", "MARGINAL") and st.session_state.wf1_test_results is not None:
        st.markdown("---")
        st.subheader("Proceed to WF2")
        st.session_state.confirm_wf2 = st.checkbox(
            "I confirm WF1 results and want to run WF2 intraday analysis",
            value=st.session_state.confirm_wf2,
        )

        wf2_ready_files = (st.session_state.wf2_bytes_y is not None and
                           st.session_state.wf2_bytes_x is not None)
        can_run_wf2 = (
            st.session_state.confirm_wf2
            and wf2_ready_files
            and rate_ok
        )

        if st.button("Run WF2 Pipeline", disabled=not can_run_wf2, type="primary"):
            _run_wf2(lot_size=lot_size, roundtrip_cost=roundtrip_cost,
                     mult_y=mult_y, mult_x=mult_x)


# ── Pipeline runners ─────────────────────────────────────────────────────────

def _run_wf1(mult_y, mult_x, lot_size, roundtrip_cost):
    from engine.wf1 import (
        load_and_prep_wf1, run_structural_break_step,
        apply_user_exclusions, split_wf1, run_gating_tests,
        run_rolling_coint, run_rolling_hurst, check_rolling_test_fail,
        run_validate_grid, run_test_slice, compute_regime_breakdown,
        detect_kill_switches,
    )
    import numpy as np

    with st.spinner("Running WF1 pipeline..."):
        try:
            # Load + prep
            prep = load_and_prep_wf1(
                st.session_state.wf1_bytes_y,
                st.session_state.wf1_bytes_x,
                st.session_state.rate_table_df,
                mult_y=mult_y, mult_x=mult_x,
            )
            st.session_state.wf1_prep = prep

            for w in prep.get("warnings", []):
                st.warning(w)

            # Break detection
            breaks = run_structural_break_step(prep["y"], prep["x"])
            st.session_state.wf1_breaks = breaks

            # Apply enabled exclusions (user can toggle on page 2, but run initially)
            y_clean, x_clean, _ = apply_user_exclusions(prep["y"], prep["x"], breaks)

            # Split
            splits = split_wf1(y_clean, x_clean)
            st.session_state.wf1_splits = splits

            # Gating
            gating = run_gating_tests(splits["train_y"], splits["train_x"])
            st.session_state.wf1_gating = gating

            gate_result = gating["gate_result"]

            if gate_result == "BLOCKED":
                st.session_state.wf1_gate = "BLOCKED"
                st.session_state.page = "wf1"
                st.rerun()
                return

            # Rolling tests
            rc = run_rolling_coint(splits["train_y"], splits["train_x"], window=120)
            beta_eg = gating.get("eg", {}).get("beta_static") or 1.0
            alpha_eg = gating.get("eg", {}).get("alpha_static") or 0.0
            rh_spread_vals = splits["train_y"].values - beta_eg * splits["train_x"].values - alpha_eg
            rh = run_rolling_hurst(pd.Series(rh_spread_vals, index=splits["train_y"].index), window=60)
            rolling_check = check_rolling_test_fail(rc, rh)
            st.session_state.wf1_rolling_coint = rc
            st.session_state.wf1_rolling_hurst = rh
            st.session_state.wf1_rolling_check  = rolling_check

            # Grid search
            grid_df = run_validate_grid(
                splits["val_y"], splits["val_x"],
                ou_std_log=gating.get("ou_std_log"),
                half_life_bars=gating.get("half_life_bars"),
                lot_size=lot_size,
                roundtrip_cost=roundtrip_cost,
            )
            st.session_state.wf1_grid_df = grid_df

            # Auto-select best row
            best = grid_df.iloc[0]
            locked = {
                "entry_z":       float(best["entry_z"]),
                "exit_z":        float(best["exit_z"]),
                "lookback":      int(best["lookback"]),
                "half_life_bars": gating.get("half_life_bars"),
            }
            st.session_state.wf1_locked_params = locked

            # Test slice
            test_results = run_test_slice(
                splits["test_y"], splits["test_x"],
                locked_params=locked,
                lot_size=lot_size,
                roundtrip_cost=roundtrip_cost,
            )
            st.session_state.wf1_test_results = test_results
            st.session_state.wf1_gate = test_results["gate"]

            # Regime + kill switches
            regime = compute_regime_breakdown(test_results["trades"], splits["test_dates"])
            st.session_state.wf1_regime = regime

            from shared.kalman import run_kalman as _rk
            k_train = _rk(
                np.log(splits["train_y"].values),
                np.log(splits["train_x"].values),
                delta=1e-4, Ve=0.1, space="raw"
            )
            train_beta_range = (float(np.nanmin(k_train["beta_t"])),
                                float(np.nanmax(k_train["beta_t"])))
            ks = detect_kill_switches(test_results, train_beta_range)
            st.session_state.wf1_kill_switches = ks

            st.session_state.page = "wf1"
            st.rerun()

        except Exception as e:
            st.error(f"WF1 pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _run_wf2(mult_y, mult_x, lot_size, roundtrip_cost):
    from engine.wf2 import (
        load_and_prep_wf2, inherit_rates, split_wf2,
        run_wf2_grid, run_wf2_test, compute_tt_settings,
    )
    from shared.kalman import run_kalman as _rk
    from shared.ou import fit_ou
    import numpy as np

    with st.spinner("Running WF2 pipeline..."):
        try:
            prep = load_and_prep_wf2(
                st.session_state.wf2_bytes_y,
                st.session_state.wf2_bytes_x,
                st.session_state.rate_table_df,
                mult_y=mult_y, mult_x=mult_x,
            )
            st.session_state.wf2_prep = prep

            for w in prep.get("warnings", []):
                st.warning(w)

            # Inherit rates
            wf1_rate_table = st.session_state.wf1_prep["rate_table"]
            _, rate_warns  = inherit_rates(prep["y"].index, wf1_rate_table)
            st.session_state.wf2_rate_warns = rate_warns

            # Split
            splits = split_wf2(prep["y"], prep["x"])
            st.session_state.wf2_splits = splits

            wf1_locked = st.session_state.wf1_locked_params

            # OU on 15-min training set for half-life
            k = _rk(
                np.log(splits["train_y"].values),
                np.log(splits["train_x"].values),
                delta=1e-4, Ve=0.1, space="raw"
            )
            ou = fit_ou(
                k["spread_reconstructed"],
                freq="intraday",
                bars_per_day=26.0,
                space="log",
            )
            half_life_bars_wf2 = ou.get("half_life_bars")

            # Grid search
            grid_df = run_wf2_grid(
                splits["train_y"], splits["train_x"],
                wf1_entry_z=wf1_locked["entry_z"],
                wf1_exit_z=wf1_locked["exit_z"],
                wf1_stop_z=4.0,
                half_life_bars=half_life_bars_wf2,
                lot_size=lot_size,
                roundtrip_cost=roundtrip_cost,
            )
            st.session_state.wf2_grid_df = grid_df

            # Auto-lock best
            best_wf2 = grid_df.iloc[0]
            wf2_locked = {
                "entry_z":       float(best_wf2["entry_z"]),
                "exit_z":        float(best_wf2["exit_z"]),
                "lookback":      int(best_wf2["lookback"]),
                "half_life_bars": half_life_bars_wf2,
            }
            st.session_state.wf2_locked_params = wf2_locked

            # Test
            test_results = run_wf2_test(
                splits["test_y"], splits["test_x"],
                locked_params=wf2_locked,
                lot_size=lot_size,
                roundtrip_cost=roundtrip_cost,
            )
            st.session_state.wf2_test_results = test_results

            # TT settings — use full-dataset Kalman for final beta
            k_full = _rk(
                np.log(prep["y"].values),
                np.log(prep["x"].values),
                delta=1e-4, Ve=0.1, space="raw"
            )
            tt = compute_tt_settings(
                kalman_result=k_full,
                ou_result=ou,
                wf1_locked_params=wf1_locked,
                wf2_locked_params=wf2_locked,
                last_fcpo_price=float(prep["y"].iloc[-1]),
                last_soy_price=float(prep["x"].iloc[-1]),
                lot_size=lot_size,
            )
            st.session_state.wf2_tt_settings = tt

            st.session_state.page = "wf2"
            st.rerun()

        except Exception as e:
            st.error(f"WF2 pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WF1 RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def page_wf1():
    st.header("WF1 Results — Daily Walk-Forward")

    gating = st.session_state.wf1_gating
    if gating is None:
        st.info("Run the WF1 pipeline from the Configuration page first.")
        return

    # ── Break detection table ──────────────────────────────────────────────
    with st.expander("Structural Break Detection", expanded=True):
        breaks = st.session_state.wf1_breaks or []

        if not breaks:
            st.success("No structural breaks detected.")
        else:
            st.markdown(f"**{len(breaks)} break period(s) detected.**  "
                        "Toggle 'Excluded' to include/exclude each period.")

            breaks_data = []
            for i, b in enumerate(breaks):
                breaks_data.append({
                    "#":        i + 1,
                    "Start":    b.get("date_start", f"bar {b['bar_start']}"),
                    "End":      b.get("date_end",   f"bar {b['bar_end']}"),
                    "Bars":     b["n_bars"],
                    "Peak |z|": f"{b['peak_z']:.2f}" if b.get("peak_z") else "N/A",
                    "Trigger":  b.get("trigger_label", ""),
                    "Excluded": b.get("enabled", True),
                })

            edited_breaks = st.data_editor(
                pd.DataFrame(breaks_data),
                use_container_width=True,
                disabled=["#", "Start", "End", "Bars", "Peak |z|", "Trigger"],
                column_config={
                    "Excluded": st.column_config.CheckboxColumn("Exclude?"),
                },
                key="break_editor",
            )

            # Manual add
            st.markdown("**Add manual exclusion period:**")
            cm1, cm2, cm3 = st.columns(3)
            with cm1:
                man_start = st.text_input("Start date (YYYY-MM-DD)", key="man_start")
            with cm2:
                man_end   = st.text_input("End date (YYYY-MM-DD)",   key="man_end")
            with cm3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add manual period"):
                    if man_start and man_end:
                        breaks.append({
                            "bar_start":     0,
                            "bar_end":       0,
                            "n_bars":        0,
                            "peak_z":        None,
                            "source":        "manual",
                            "trigger_label": "Manual",
                            "date_start":    man_start,
                            "date_end":      man_end,
                            "enabled":       True,
                        })
                        st.session_state.wf1_breaks = breaks
                        st.rerun()

            # Sync toggles back
            for i, row in edited_breaks.iterrows():
                if i < len(breaks):
                    breaks[i]["enabled"] = bool(row["Excluded"])
            st.session_state.wf1_breaks = breaks

        if st.button("Confirm exclusions and re-run gating", type="primary"):
            _rerun_wf1_with_exclusions()

    # ── Gating tests ───────────────────────────────────────────────────────
    st.subheader("Gating Tests — Training Set")
    passes = gating.get("passes", {})
    gate_result = gating.get("gate_result", "BLOCKED")

    test_rows = [
        ("ADF/KPSS — FCPO is I(1)", gating.get("adf_y", {}).get("verdict",""), "adf_y"),
        ("ADF/KPSS — SOY is I(1)",  gating.get("adf_x", {}).get("verdict",""), "adf_x"),
        ("Engle-Granger Coint",
         f"p={gating.get('eg', {}).get('eg_pvalue','?')}", "eg"),
        ("Johansen Coint",
         gating.get("johansen", {}).get("verdict",""), "johansen"),
        ("Hurst Exponent",
         f"H={gating.get('hurst', {}).get('hurst','?')}", "hurst"),
        ("OU Half-Life",
         f"{gating.get('ou', {}).get('half_life_days','?')}d", "ou"),
    ]

    tbl_data = []
    for name, stat, key in test_rows:
        passed = passes.get(key, False)
        tbl_data.append({"Test": name, "Result": stat,
                          "Pass": "✅ PASS" if passed else "❌ FAIL"})
    st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

    gate_color = {"PASS": "green", "MARGINAL": "orange", "BLOCKED": "red"}.get(gate_result, "gray")
    st.markdown(f"**Gate: {_badge(gate_result, gate_color)}**", unsafe_allow_html=True)

    # Rolling test stability
    rc  = st.session_state.wf1_rolling_coint
    rh  = st.session_state.wf1_rolling_hurst
    rck = st.session_state.wf1_rolling_check
    if rck:
        with st.expander("Rolling Stability Tests"):
            ci1, ci2 = st.columns(2)
            with ci1:
                st.metric("Coint bad windows",
                          f"{rck.get('coint_bad_pct',0)*100:.1f}%",
                          delta="FAIL" if rck.get("fail_coint") else "PASS",
                          delta_color="inverse")
            with ci2:
                st.metric("Hurst bad windows",
                          f"{rck.get('hurst_bad_pct',0)*100:.1f}%",
                          delta="FAIL" if rck.get("fail_hurst") else "PASS",
                          delta_color="inverse")

            if rc is not None:
                fig_rc = go.Figure()
                fig_rc.add_trace(go.Scatter(y=rc.values, mode="lines",
                                            line=dict(color="#00d4ff", width=1),
                                            name="Coint p-value"))
                fig_rc.add_hline(y=0.20, line_dash="dash", line_color="#ffa500")
                fig_rc.update_layout(
                    title="Rolling Cointegration p-value (120-day windows)",
                    plot_bgcolor=DARK_PLOT, paper_bgcolor=DARK_BG,
                    font=dict(color=DARK_TEXT), height=220,
                    margin=dict(l=60, r=30, t=30, b=30),
                )
                st.plotly_chart(fig_rc, use_container_width=True)

    if gate_result == "BLOCKED":
        st.error("BLOCKED: Too many gating tests failed. Review the pair or data quality before proceeding.")
        return

    # ── Grid search ────────────────────────────────────────────────────────
    st.subheader("Validation Grid — 36 Parameter Combinations")
    grid_df = st.session_state.wf1_grid_df

    if grid_df is not None:
        # Color warning rows
        def _style_row(row):
            if not row.get("cost_ok", True):
                return ["background-color: #2a1a00"] * len(row)
            if row["rank"] == 1:
                return ["background-color: #0d2a1a"] * len(row)
            return [""] * len(row)

        display_df = grid_df.copy()
        display_df["cost_ok"] = display_df["cost_ok"].map({True: "✅", False: "⚠ Cost risk"})

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "sharpe":   st.column_config.NumberColumn("Sharpe", format="%.3f"),
                "calmar":   st.column_config.NumberColumn("Calmar", format="%.3f"),
                "win_rate": st.column_config.NumberColumn("Win Rate", format="%.3f"),
                "net_pnl":  st.column_config.NumberColumn("Net PnL", format="%.0f"),
            },
        )
        st.caption("⚠ Cost risk = entry_z may not overcome round-trip costs at current ou_std.")

        # Row selector
        st.markdown("**Select parameter row to lock:**")
        grid_idx = st.selectbox(
            "Lock row (rank)",
            options=grid_df["rank"].tolist(),
            index=0,
            format_func=lambda r: (
                f"Rank {r} | entry_z={grid_df.loc[grid_df.rank==r, 'entry_z'].iloc[0]} "
                f"| exit_z={grid_df.loc[grid_df.rank==r, 'exit_z'].iloc[0]} "
                f"| lookback={grid_df.loc[grid_df.rank==r, 'lookback'].iloc[0]} "
                f"| Sharpe={grid_df.loc[grid_df.rank==r, 'sharpe'].iloc[0]:.3f}"
            ),
        )
        selected_row = grid_df[grid_df["rank"] == grid_idx].iloc[0]

        if not selected_row.get("cost_ok", True):
            st.warning("Selected row may not overcome round-trip costs. Consider a higher entry_z.")

        if st.button("Lock selected parameters", type="secondary"):
            locked = {
                "entry_z":       float(selected_row["entry_z"]),
                "exit_z":        float(selected_row["exit_z"]),
                "lookback":      int(selected_row["lookback"]),
                "half_life_bars": st.session_state.wf1_gating.get("half_life_bars"),
            }
            st.session_state.wf1_locked_params = locked
            st.success(f"Locked: entry_z={locked['entry_z']} | exit_z={locked['exit_z']} | lookback={locked['lookback']}")

    # ── Test results ───────────────────────────────────────────────────────
    locked = st.session_state.wf1_locked_params
    test_results = st.session_state.wf1_test_results

    if locked:
        st.markdown(f"**Locked params:** entry_z={locked['entry_z']} | exit_z={locked['exit_z']} | lookback={locked['lookback']}")

    if test_results is None:
        return

    st.subheader("Test Set Results")
    m = test_results["metrics"]
    gate = test_results["gate"]

    cols = st.columns(6)
    cols[0].metric("Trades",     m.get("n_trades", 0))
    cols[1].metric("Win Rate",   f"{m.get('win_rate',0)*100:.1f}%" if m.get("win_rate") else "N/A")
    cols[2].metric("Sharpe",     f"{m.get('sharpe',0):.3f}"        if m.get("sharpe")   else "N/A")
    cols[3].metric("Net PnL",    f"{m.get('total_net_pnl',0):,.0f}")
    cols[4].metric("Max DD",     f"{m.get('max_drawdown',0):,.0f}")
    cols[5].metric("Years",      f"{m.get('years',0):.1f}")

    gate_color = {"GO": "green", "MARGINAL": "orange", "NO-GO": "red"}.get(gate, "gray")
    st.markdown(f"### Verdict: {_gate_badge(gate)}", unsafe_allow_html=True)

    # Kill switches
    ks = st.session_state.wf1_kill_switches or []
    if ks:
        st.subheader("Kill Switch Warnings")
        for msg in ks:
            st.error(msg)

    # Regime breakdown
    regime = st.session_state.wf1_regime or {}
    if regime:
        st.subheader("Regime Breakdown")
        regime_rows = []
        for band, v in regime.items():
            regime_rows.append({
                "Period":   band,
                "Trades":   v["n_trades"],
                "Net PnL":  f"{v['net_pnl']:,.0f}",
                "Win Rate": f"{v['win_rate']*100:.1f}%" if v.get("win_rate") else "N/A",
            })
        st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

    # Equity curve
    ec = test_results.get("equity_curve")
    if ec is not None:
        st.plotly_chart(_plot_equity(ec, "WF1 Test Equity Curve"), use_container_width=True)

    # Spread + z-score
    spread  = test_results.get("spread",  [])
    z_score = test_results.get("z_score", [])
    if len(spread) > 0:
        st.plotly_chart(
            _plot_spread_z(spread, z_score, test_results.get("trades",[]), None,
                           "WF1 Spread & Z-score"),
            use_container_width=True,
        )

    # Trade log
    trades_list = test_results.get("trades", [])
    if trades_list:
        st.subheader("Trade Log")
        trade_df = pd.DataFrame(trades_list)
        st.dataframe(trade_df, use_container_width=True, hide_index=True)

    # HTML report download
    st.markdown("---")
    if st.button("Generate & Download HTML Report (WF1)"):
        try:
            from report.html_report import generate_report
            report_bytes = generate_report(
                wf1_results={
                    "gating":          gating,
                    "grid_df":         st.session_state.wf1_grid_df,
                    "locked_params":   locked,
                    "test_results":    test_results,
                    "regime_breakdown": regime,
                    "kill_switches":   ks,
                    "breaks":          st.session_state.wf1_breaks or [],
                },
                config={
                    "Date range": str(st.session_state.wf1_prep["date_range"]),
                    "N bars":     st.session_state.wf1_prep["n_bars"],
                    "Lot size":   locked.get("lot_size", "—"),
                },
            )
            fname = f"MRBackTest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            st.download_button("Download Report", data=report_bytes, file_name=fname, mime="text/html")
        except Exception as e:
            st.error(f"Report generation error: {e}")


def _rerun_wf1_with_exclusions():
    """Re-run gating + downstream with updated exclusion toggles."""
    from engine.wf1 import (
        apply_user_exclusions, split_wf1, run_gating_tests,
        run_rolling_coint, run_rolling_hurst, check_rolling_test_fail,
        run_validate_grid, run_test_slice, compute_regime_breakdown,
        detect_kill_switches,
    )
    from shared.kalman import run_kalman as _rk
    import numpy as np

    prep   = st.session_state.wf1_prep
    breaks = st.session_state.wf1_breaks or []

    lot_size       = st.session_state.lot_size
    roundtrip_cost = st.session_state.roundtrip_cost

    with st.spinner("Re-running with updated exclusions..."):
        try:
            y_clean, x_clean, _ = apply_user_exclusions(prep["y"], prep["x"], breaks)
            splits  = split_wf1(y_clean, x_clean)
            st.session_state.wf1_splits = splits

            gating = run_gating_tests(splits["train_y"], splits["train_x"])
            st.session_state.wf1_gating = gating

            rc = run_rolling_coint(splits["train_y"], splits["train_x"])
            beta_eg2  = gating.get("eg", {}).get("beta_static") or 1.0
            alpha_eg2 = gating.get("eg", {}).get("alpha_static") or 0.0
            rh_spread2 = (splits["train_y"].values
                          - beta_eg2 * splits["train_x"].values - alpha_eg2)
            rh = run_rolling_hurst(pd.Series(rh_spread2, index=splits["train_y"].index))
            st.session_state.wf1_rolling_coint = rc
            st.session_state.wf1_rolling_hurst = rh
            st.session_state.wf1_rolling_check = check_rolling_test_fail(rc, rh)

            grid_df = run_validate_grid(
                splits["val_y"], splits["val_x"],
                ou_std_log=gating.get("ou_std_log"),
                half_life_bars=gating.get("half_life_bars"),
                lot_size=lot_size,
                roundtrip_cost=roundtrip_cost,
            )
            st.session_state.wf1_grid_df = grid_df

            best   = grid_df.iloc[0]
            locked = {
                "entry_z":       float(best["entry_z"]),
                "exit_z":        float(best["exit_z"]),
                "lookback":      int(best["lookback"]),
                "half_life_bars": gating.get("half_life_bars"),
            }
            st.session_state.wf1_locked_params = locked

            test_results = run_test_slice(
                splits["test_y"], splits["test_x"],
                locked_params=locked,
                lot_size=lot_size,
                roundtrip_cost=roundtrip_cost,
            )
            st.session_state.wf1_test_results = test_results
            st.session_state.wf1_gate         = test_results["gate"]

            regime = compute_regime_breakdown(test_results["trades"], splits["test_dates"])
            st.session_state.wf1_regime = regime

            k_train = _rk(
                np.log(splits["train_y"].values),
                np.log(splits["train_x"].values),
                delta=1e-4, Ve=0.1, space="raw"
            )
            train_beta_range = (float(np.nanmin(k_train["beta_t"])),
                                float(np.nanmax(k_train["beta_t"])))
            st.session_state.wf1_kill_switches = detect_kill_switches(
                test_results, train_beta_range
            )
            st.rerun()

        except Exception as e:
            st.error(f"Re-run error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WF2 RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def page_wf2():
    st.header("WF2 Results — 15-Minute Intraday")

    st.info(
        "⚠ **Execution Caveat:** All Sharpe and PnL figures are pre-execution-cost estimates. "
        "Intraday auto-spread trading involves bid-ask slippage, leg-timing risk, and queue position "
        "uncertainty not fully captured in simulation. Use as signal quality indicators only."
    )

    test_results = st.session_state.wf2_test_results
    if test_results is None:
        st.info("Run the WF2 pipeline from the Configuration page first.")
        return

    # Rate warnings
    rate_warns = st.session_state.wf2_rate_warns or []
    for w in rate_warns:
        st.warning(w)

    # Grid
    st.subheader("WF2 Intraday Grid Search (12 rows)")
    grid_df = st.session_state.wf2_grid_df
    if grid_df is not None:
        st.dataframe(
            grid_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "sharpe":   st.column_config.NumberColumn("Sharpe (indicative)", format="%.3f"),
                "win_rate": st.column_config.NumberColumn("Win Rate", format="%.3f"),
                "net_pnl":  st.column_config.NumberColumn("Net PnL", format="%.0f"),
            },
        )

        # Row selector
        grid_idx2 = st.selectbox(
            "Lock WF2 row (rank)",
            options=grid_df["rank"].tolist(),
            index=0,
            format_func=lambda r: (
                f"Rank {r} | entry_z={grid_df.loc[grid_df.rank==r, 'entry_z'].iloc[0]} "
                f"| exit_z={grid_df.loc[grid_df.rank==r, 'exit_z'].iloc[0]} "
                f"| lookback={grid_df.loc[grid_df.rank==r, 'lookback'].iloc[0]}"
            ),
        )
        if st.button("Lock selected WF2 parameters", type="secondary"):
            sel = grid_df[grid_df["rank"] == grid_idx2].iloc[0]
            wf2_locked = {
                "entry_z":  float(sel["entry_z"]),
                "exit_z":   float(sel["exit_z"]),
                "lookback": int(sel["lookback"]),
                "half_life_bars": st.session_state.wf2_locked_params.get("half_life_bars"),
            }
            st.session_state.wf2_locked_params = wf2_locked
            st.success("WF2 params locked.")

    wf2_locked = st.session_state.wf2_locked_params
    if wf2_locked:
        st.markdown(
            f"**Locked WF2 params:** entry_z={wf2_locked.get('entry_z')} "
            f"| exit_z={wf2_locked.get('exit_z')} "
            f"| lookback={wf2_locked.get('lookback')}"
        )

    # Test results
    st.subheader("WF2 Test Results")
    m    = test_results["metrics"]
    gate = test_results["gate"]

    cols = st.columns(5)
    cols[0].metric("Trades",           m.get("n_trades", 0))
    cols[1].metric("Win Rate",         f"{m.get('win_rate',0)*100:.1f}%" if m.get("win_rate") else "N/A")
    cols[2].metric("Sharpe (indicative)", f"{m.get('sharpe',0):.3f}"    if m.get("sharpe")   else "N/A")
    cols[3].metric("Net PnL (MYR)",    f"{m.get('total_net_pnl',0):,.0f}")
    cols[4].metric("Max DD",           f"{m.get('max_drawdown',0):,.0f}")

    st.markdown(f"### Verdict: {_gate_badge(gate)}", unsafe_allow_html=True)

    # Calibration verdict
    if gate == "GO":
        st.success("Execution Calibration: **Complete** — intraday signal confirmed.")
    elif gate == "MARGINAL":
        st.warning("Execution Calibration: **Inconclusive** — signal marginal intraday; paper-trade before live.")
    else:
        st.error("Execution Calibration: **Inconclusive** — intraday signal weak; do not proceed live.")

    # Equity curve
    ec2 = test_results.get("equity_curve")
    if ec2 is not None:
        st.plotly_chart(_plot_equity(ec2, "WF2 Test Equity Curve (Intraday)"), use_container_width=True)

    # Spread + z-score
    spread2  = test_results.get("spread",  [])
    z_score2 = test_results.get("z_score", [])
    if len(spread2) > 0:
        st.plotly_chart(
            _plot_spread_z(spread2, z_score2, test_results.get("trades", []), None,
                           "WF2 Spread & Z-score"),
            use_container_width=True,
        )

    # TT Auto Spreader settings
    tt = st.session_state.wf2_tt_settings
    if tt:
        st.subheader("TT Auto Spreader Settings")
        col_tt1, col_tt2 = st.columns(2)
        with col_tt1:
            st.metric("Hedge Ratio (beta)", f"{tt.get('hedge_ratio', '?'):.4f}")
            st.metric("Lot Ratio",          tt.get("lot_ratio_display", "?"))
            st.metric("Half-Life (hours)",  f"{tt.get('half_life_hours', '?')}")
        with col_tt2:
            st.metric("Entry Z (WF1)",  tt.get("entry_z_wf1", "?"))
            st.metric("Exit Z (WF1)",   tt.get("exit_z_wf1", "?"))
            st.metric("Stop Z",         tt.get("stop_z", "?"))
            st.metric("Entry Z (WF2)",  tt.get("entry_z_wf2", "?"))
            st.metric("Exit Z (WF2)",   tt.get("exit_z_wf2", "?"))

        with st.expander("Full TT Settings Table"):
            tt_rows = [{"Setting": k, "Value": str(v)} for k, v in tt.items()]
            st.dataframe(pd.DataFrame(tt_rows), use_container_width=True, hide_index=True)

    # HTML report download (WF1 + WF2 combined)
    st.markdown("---")
    if st.button("Generate & Download HTML Report (WF1 + WF2)"):
        try:
            from report.html_report import generate_report
            gating = st.session_state.wf1_gating
            locked = st.session_state.wf1_locked_params
            wf1_test = st.session_state.wf1_test_results

            report_bytes = generate_report(
                wf1_results={
                    "gating":           gating,
                    "grid_df":          st.session_state.wf1_grid_df,
                    "locked_params":    locked,
                    "test_results":     wf1_test,
                    "regime_breakdown": st.session_state.wf1_regime or {},
                    "kill_switches":    st.session_state.wf1_kill_switches or [],
                    "breaks":           st.session_state.wf1_breaks or [],
                },
                wf2_results={
                    "grid_df":       st.session_state.wf2_grid_df,
                    "locked_params": wf2_locked,
                    "test_results":  test_results,
                    "tt_settings":   tt or {},
                },
                config={
                    "WF1 date range": str(st.session_state.wf1_prep.get("date_range", "")),
                    "WF2 date range": str(st.session_state.wf2_prep.get("date_range", "")) if st.session_state.wf2_prep else "",
                },
            )
            fname = f"MRBackTest_Full_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            st.download_button("Download Full Report", data=report_bytes,
                               file_name=fname, mime="text/html")
        except Exception as e:
            st.error(f"Report generation error: {e}")


# ── Router ────────────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "config":
    page_config()
elif page == "wf1":
    page_wf1()
elif page == "wf2":
    page_wf2()
else:
    page_config()
