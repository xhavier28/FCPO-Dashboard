# engine/spot_executor.py
"""
Executes spot hedge leg via FIX AFTER SSF fill is confirmed.
Never called unless SSF fill is received first.

spot_order type per stock (from config):
  "MARKET"       -> pure market order (OrdType=1)
  "LIMIT_PLUS_1" -> best price + 1 tick limit (OrdType=2)
                    BUY:  best ask + 1 tick
                    SELL: best bid - 1 tick
"""
import config
from utils.logger import get_logger

log = get_logger("spot_executor")


def execute_spot_hedge(sym: str, side: str, lots: int):
    """
    Place spot market/limit order for hedge.
    side: "LONG" (buy spot) or "SHORT" (sell spot)
    lots: SSF lots filled -- shares = lots x 100

    Returns fill price (int) or None if failed.
    In paper trading mode returns simulated price.
    """
    stock      = next(s for s in config.STOCKS if s["sym"] == sym)
    shares     = lots * 100
    order_type = stock["spot_order"]

    if config.PAPER_TRADING:
        from db import database as db
        quotes = db.get_all_quotes()
        quote  = next((q for q in quotes if q["sym"] == sym), None)
        if not quote:
            return None
        spot_price = int(quote["spot"])
        tick       = stock["tick"]

        if order_type == "LIMIT_PLUS_1":
            if side == "LONG":
                sim_price = spot_price + tick
            else:
                sim_price = spot_price - tick
        else:
            sim_price = spot_price

        log.info(f"PAPER SPOT HEDGE: {sym} {side} {shares}shares @ {sim_price:,} "
                 f"({order_type})")
        return sim_price

    else:
        # ── LIVE: send via FIX spot session ──────────────────────────────────
        # from fix.fix_session import spot_session
        # fix_side = "BUY" if side == "LONG" else "SELL"
        # result = spot_session.place_order(
        #     sym      = sym,
        #     side     = fix_side,
        #     ord_type = order_type,
        #     shares   = shares,
        # )
        # return result.get("fill_price")
        log.error("Live spot FIX not configured.")
        return None
