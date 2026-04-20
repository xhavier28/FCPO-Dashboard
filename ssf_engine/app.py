# app.py
"""
Streamlit dashboard. Run in Terminal 2:
  streamlit run app.py

Reads all state from state.db. Never touches FIX directly.
Writes only: dividend yield config, spot_order toggles, position actions.
"""
import datetime
import streamlit as st
import pandas as pd
import time
import config
from db import database as db
from engine.fair_value import calc_fair_value, round_to_tick, get_tte

st.set_page_config(
    page_title="SSF Auto-Quote Engine",
    page_icon="[SSF]",
    layout="wide",
)

db.init_db()

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
if "div_preview" not in st.session_state:
    st.session_state.div_preview = None
if "div_drafts" not in st.session_state:
    drafts = {}
    for s in config.STOCKS:
        key   = f"div_{s['sym']}"
        saved = db.get_config(key)
        drafts[s["sym"]] = saved if saved else {
            str(m): s["div"].get(m, 0.0)
            for m in [4, 5, 6]
        }
    st.session_state.div_drafts = drafts


# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_div(sym: str, month: int) -> float:
    return st.session_state.div_drafts.get(sym, {}).get(str(month), 0.0)


def affected_months(changed_month: int) -> list:
    """April change -> [4,5,6]. Mei -> [5,6]. Juni -> [6]."""
    return [m for m in [4, 5, 6] if m >= changed_month]


def contract_name(month: int) -> str:
    return {4: "April", 5: "Mei", 6: "Juni"}.get(month, str(month))


# ── HEADER ────────────────────────────────────────────────────────────────────
col_title, col_status, col_mode, col_feed = st.columns([3, 1, 1, 1])
with col_title:
    st.markdown("## SSF Auto-Quote Engine")
with col_status:
    status = db.get_engine_state("status", "UNKNOWN")
    color  = "green" if status == "RUNNING" else "red"
    st.markdown(f"**Status:** :{color}[{status}]")
with col_mode:
    mode = db.get_engine_state("mode", "PAPER")
    st.markdown(f"**Mode:** `{mode}`")
with col_feed:
    feed = db.get_engine_state("feed", config.FEED_SOURCE)
    st.markdown(f"**Feed:** `{feed.upper()}`")

st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_quotes, tab_positions, tab_div, tab_config = st.tabs([
    "Quote Board",
    "Position Tracker",
    "Dividend Yields",
    "Config",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUOTE BOARD
# ════════════════════════════════════════════════════════════════════════════
with tab_quotes:
    selected_month = st.radio(
        "Contract",
        options=[c["month"] for c in config.CONTRACTS],
        format_func=lambda m: (
            f"{contract_name(m)} "
            f"({get_tte(next(c for c in config.CONTRACTS if c['month']==m)['expiry'])}d)"
        ),
        horizontal=True,
    )

    quotes       = db.get_all_quotes()
    month_quotes = [q for q in quotes if q["month"] == selected_month]

    if not month_quotes:
        st.info("No quotes yet — start fix_engine.py in Terminal 1.")
    else:
        rows = []
        for q in month_quotes:
            bids = {l["layer_num"]: l for l in q["bid_layers"]}
            asks = {l["layer_num"]: l for l in q["ask_layers"]}

            def layer_cell(layers_dict, n):
                l = layers_dict.get(n)
                if not l:
                    return "-"
                return f"{l['price']:,} x {l['lots']}L"

            best_bid = bids.get(1, {}).get("price", 0)
            best_ask = asks.get(1, {}).get("price", 0)
            rows.append({
                "Symbol": q["sym"],
                "Spot":   f"{int(q['spot']):,}",
                "FV":     f"{q['fv']:.0f}",
                "Bid L1": layer_cell(bids, 1),
                "Bid L2": layer_cell(bids, 2),
                "Bid L3": layer_cell(bids, 3),
                "Ask L1": layer_cell(asks, 1),
                "Ask L2": layer_cell(asks, 2),
                "Ask L3": layer_cell(asks, 3),
                "Spread": best_ask - best_bid if best_bid and best_ask else 0,
                "TTE":    f"{q['tte']}d",
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Bid L1": st.column_config.TextColumn("Bid L1 (5L)", help="Closest to FV"),
                "Bid L2": st.column_config.TextColumn("Bid L2 (7L)"),
                "Bid L3": st.column_config.TextColumn("Bid L3 (10L)", help="Deepest"),
                "Ask L1": st.column_config.TextColumn("Ask L1 (5L)", help="Closest to FV"),
                "Ask L2": st.column_config.TextColumn("Ask L2 (7L)"),
                "Ask L3": st.column_config.TextColumn("Ask L3 (10L)", help="Deepest"),
            }
        )
        st.caption(f"Last updated: {quotes[0]['updated_at'][:19] if quotes else '-'}")

    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh (5s)", value=True)
    if auto_refresh:
        time.sleep(5)
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — POSITION TRACKER
# ════════════════════════════════════════════════════════════════════════════
with tab_positions:
    positions = db.get_open_positions()

    if not positions:
        st.info("No open positions.")
    else:
        rows = []
        for p in positions:
            rows.append({
                "ID":           p["id"],
                "Symbol":       p["sym"],
                "Contract":     p["contract"],
                "SSF Side":     p["ssf_side"],
                "SSF Entry":    f"{p['ssf_entry_price']:,}",
                "SSF Current":  f"{p['ssf_current']:,}",
                "Spot Side":    p["spot_side"],
                "Spot Entry":   f"{p['spot_entry_price']:,}",
                "Spot Current": f"{p['spot_current']:,}",
                "Lots":         p["lots"],
                "Leverage":     f"{p['spot_leverage']}x",
                "TTE":          f"{p['tte']}d",
                "SSF PnL":      f"{p['ssf_pnl']:+,.0f}",
                "Spot PnL":     f"{p['spot_pnl']:+,.0f}",
                "Total PnL":    f"{p['total_pnl']:+,.0f}",
                "Action":       p["action"],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### Manage Positions")
        for p in positions:
            col_sym, col_pnl, col_hold, col_exit, col_close = st.columns([2, 2, 2, 2, 1])
            with col_sym:
                st.write(f"**{p['sym']}** {p['contract']} | {p['ssf_side']} SSF")
            with col_pnl:
                color = "green" if p["total_pnl"] >= 0 else "red"
                st.markdown(f"PnL: :{color}[{p['total_pnl']:+,.0f} IDR]")
            with col_hold:
                if st.button("Hold -> Expire", key=f"hold_{p['id']}"):
                    db.set_position_action(p["id"], "HOLD_EXPIRE")
                    st.rerun()
            with col_exit:
                if st.button("Initiate Exit", key=f"exit_{p['id']}"):
                    db.set_position_action(p["id"], "INITIATE_EXIT")
                    st.rerun()
            with col_close:
                if st.button("Close", key=f"close_{p['id']}"):
                    db.close_position(p["id"])
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DIVIDEND YIELD INPUT WITH PREVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_div:
    st.markdown("### Dividend Yield Input")
    st.caption("Change a month's yield -> preview affected contracts -> confirm to apply. "
               "Changing April affects April+Mei+Juni. Changing Mei affects Mei+Juni only.")

    if st.session_state.div_preview:
        prev    = st.session_state.div_preview
        sym     = prev["sym"]
        month   = prev["month"]
        old_div = prev["old_yield"]
        new_div = prev["new_yield"]
        months_affected = affected_months(month)
        stock   = next(s for s in config.STOCKS if s["sym"] == sym)

        # try to use last known spot from DB
        all_quotes = db.get_all_quotes()
        q_spot = next((q["spot"] for q in all_quotes if q["sym"] == sym), 6750)
        spot = int(q_spot)

        st.warning(f"Preview: **{sym}** {contract_name(month)} yield "
                   f"{old_div:.2f}% -> {new_div:.2f}%")

        preview_rows = []
        for m in months_affected:
            c      = next(c for c in config.CONTRACTS if c["month"] == m)
            tte    = get_tte(c["expiry"])
            fv_old = calc_fair_value(spot, tte, old_div if m == month else get_div(sym, m))
            fv_new = calc_fair_value(spot, tte, new_div if m == month else get_div(sym, m))
            tick   = stock["tick"]
            for layer_num, (offset, lots) in enumerate(config.BID_LAYERS, 1):
                preview_rows.append({
                    "Contract": contract_name(m),
                    "Layer":    f"Bid L{layer_num}",
                    "Old FV":   f"{fv_old:.0f}",
                    "New FV":   f"{fv_new:.0f}",
                    "Old Bid":  f"{round_to_tick(fv_old - offset * tick, tick):,}",
                    "New Bid":  f"{round_to_tick(fv_new - offset * tick, tick):,}",
                    "Old Ask":  f"{round_to_tick(fv_old + offset * tick, tick):,}",
                    "New Ask":  f"{round_to_tick(fv_new + offset * tick, tick):,}",
                })

        st.dataframe(pd.DataFrame(preview_rows),
                     use_container_width=True, hide_index=True)

        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("Confirm — Apply Changes", type="primary"):
                drafts = st.session_state.div_drafts
                for m in months_affected:
                    drafts[sym][str(m)] = new_div
                db.set_config(f"div_{sym}", drafts[sym])
                st.session_state.div_preview = None
                st.success(f"Dividend yield updated for {sym}.")
                st.rerun()
        with col_cancel:
            if st.button("Cancel"):
                st.session_state.div_preview = None
                st.rerun()

        st.divider()

    for s in config.STOCKS:
        sym = s["sym"]
        with st.expander(f"{sym} — Dividend Yield"):
            cols = st.columns(3)
            for i, (m, month_label) in enumerate([(4, "April"), (5, "Mei"), (6, "Juni")]):
                with cols[i]:
                    current_val = float(
                        st.session_state.div_drafts.get(sym, {}).get(str(m), 0.0))
                    new_val = st.number_input(
                        f"{month_label} (%)",
                        value=current_val,
                        min_value=0.0,
                        max_value=50.0,
                        step=0.01,
                        format="%.2f",
                        key=f"div_{sym}_{m}",
                    )
                    if new_val != current_val:
                        st.session_state.div_preview = {
                            "sym":       sym,
                            "month":     m,
                            "old_yield": current_val,
                            "new_yield": new_val,
                        }
                        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONFIG
# ════════════════════════════════════════════════════════════════════════════
with tab_config:
    st.markdown("### Per-Stock Settings")
    st.caption("Changes here update config at runtime. "
               "Restart fix_engine.py to apply max_lots or leverage changes.")

    for s in config.STOCKS:
        sym = s["sym"]
        with st.expander(f"{sym}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.number_input(
                    "Max Lots",
                    value=s["max_lots"],
                    min_value=1,
                    key=f"max_{sym}",
                )
            with col2:
                st.number_input(
                    "Spot Leverage",
                    value=float(s["spot_leverage"]),
                    min_value=1.0,
                    step=1.0,
                    key=f"lev_{sym}",
                )
            with col3:
                st.selectbox(
                    "Spot Order Type",
                    options=["LIMIT_PLUS_1", "MARKET"],
                    index=0 if s["spot_order"] == "LIMIT_PLUS_1" else 1,
                    key=f"ord_{sym}",
                )

    st.divider()
    st.markdown("### Global Market Parameters")
    col_j, col_b, col_sb, col_ss = st.columns(4)
    with col_j:
        st.number_input("JIBOR 1M (%)",      value=config.JIBOR_1M,       step=0.01,  key="g_jibor")
    with col_b:
        st.number_input("Borrow Cost (%)",   value=config.BORROWING_COST, step=0.01,  key="g_borrow")
    with col_sb:
        st.number_input("Spot Buy Cost (%)", value=config.SPOT_BUY_COST,  step=0.005, key="g_spotbuy")
    with col_ss:
        st.number_input("Spot Sell Cost (%)",value=config.SPOT_SELL_COST, step=0.005, key="g_spotsell")
