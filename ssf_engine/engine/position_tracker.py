# engine/position_tracker.py
"""
Manages position state per stock:
- Tracks total lots (SSF + spot combined)
- Determines which layers are active based on fill history and position size
- Triggers spot hedge leg after SSF fill confirmed
- Layer resume logic based on configurable thresholds
"""
import datetime
import config
from db import database as db
from utils.logger import get_logger

log = get_logger("position_tracker")


def get_active_layers(sym: str, side: str) -> list:
    """
    Returns list of active layer numbers [1,2,3] for a symbol+side.

    Resume logic:
      total_lots == 0              -> all 3 layers active
      total_lots < resume_pct[1]% -> layers 2 + 3 active
      total_lots < resume_pct[0]% -> layer 3 only
      total_lots >= max_lots      -> no layers active (hard stop)
    """
    stock      = next(s for s in config.STOCKS if s["sym"] == sym)
    max_lots   = stock["max_lots"]
    pct_l3, pct_l2 = stock["resume_pct"]
    total_lots = db.get_total_lots(sym)

    if total_lots >= max_lots:
        log.warning(f"{sym} at max position ({total_lots}/{max_lots} lots) — no quoting")
        return []

    thresh_l3 = max_lots * pct_l3 / 100
    thresh_l2 = max_lots * pct_l2 / 100

    if total_lots == 0:
        return [1, 2, 3]
    elif total_lots < thresh_l2:
        return [2, 3]
    elif total_lots < thresh_l3:
        return [3]
    else:
        return []


def on_ssf_fill(sym: str, contract: str, month: int,
                side: str, layer_num: int,
                fill_price: int, lots: int, fv: float) -> None:
    """
    Called immediately when FIX confirms an SSF fill.
    1. Records the SSF fill
    2. Triggers spot hedge leg
    3. Opens position record in DB
    """
    from engine.spot_executor import execute_spot_hedge

    log.info(f"SSF FILL: {sym}-{contract} {side} L{layer_num} "
             f"{lots}lot @ {fill_price:,} | FV={fv:.0f}")

    db.record_fill(sym, contract, "SSF", side,
                   layer_num, fill_price, lots, fv)

    # SSF BID filled (long SSF) -> short spot
    # SSF ASK filled (short SSF) -> long spot
    spot_side = "SHORT" if side == "BID" else "LONG"
    ssf_side  = "LONG"  if side == "BID" else "SHORT"

    spot_fill_price = execute_spot_hedge(sym, spot_side, lots)

    if spot_fill_price is None:
        log.error(f"Spot hedge FAILED for {sym} — position is UNHEDGED. Manual action required.")
        return

    db.record_fill(sym, contract, "SPOT", spot_side,
                   0, spot_fill_price, lots, fv)

    stock = next(s for s in config.STOCKS if s["sym"] == sym)
    tte   = (next(c for c in config.CONTRACTS
                  if c["month"] == month)["expiry"]
             - datetime.date.today()).days

    pos_id = db.insert_position(
        sym           = sym,
        contract      = contract,
        ssf_side      = ssf_side,
        ssf_entry     = fill_price,
        spot_side     = spot_side,
        spot_entry    = spot_fill_price,
        lots          = lots,
        spot_leverage = stock["spot_leverage"],
        tte           = tte,
    )
    log.info(f"Position opened: id={pos_id} {sym} "
             f"SSF {ssf_side}@{fill_price:,} + SPOT {spot_side}@{spot_fill_price:,}")
