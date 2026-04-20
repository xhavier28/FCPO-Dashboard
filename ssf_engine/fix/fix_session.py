# fix/fix_session.py
"""
FIX 4.4 session manager -- PLACEHOLDER.
Uses QuickFIX (pip install quickfix).

When broker provides FIX credentials:
1. Fill FIX_CONFIG in config.py
2. Update ssf.cfg and spot.cfg with correct session params
3. Uncomment the QuickFIX blocks below

Message types used:
  35=D  NewOrderSingle             -- place new order
  35=G  OrderCancelReplaceRequest  -- cancel+replace on requote
  35=F  OrderCancelRequest         -- cancel existing order
  35=8  ExecutionReport            -- fill confirmation (incoming)
"""
from utils.logger import get_logger

log = get_logger("fix_session")


class FIXSession:
    """Placeholder FIX session. Replace body with QuickFIX calls."""

    def __init__(self, session_name: str):
        self.name        = session_name
        self.connected   = False
        self.open_orders = {}   # {client_order_id: order_dict}

    def connect(self) -> None:
        # import quickfix as fix
        # settings  = fix.SessionSettings(f"fix/{self.name}.cfg")
        # store     = fix.FileStoreFactory(settings)
        # log_f     = fix.FileLogFactory(settings)
        # self.app  = SSFFixApp(self)
        # self.init = fix.SocketInitiator(self.app, store, settings, log_f)
        # self.init.start()
        log.info(f"[{self.name}] FIX session connect placeholder")
        self.connected = True

    def place_order(self, sym: str, side: str, ord_type: str,
                    price: int, qty: int, contract: str = "") -> dict:
        """
        Place new limit or market order.
        side:     "BUY" or "SELL"
        ord_type: "MARKET" or "LIMIT_PLUS_1"
        Returns:  {"client_order_id": "...", "status": "PENDING"}
        """
        import uuid
        cl_ord_id = str(uuid.uuid4())[:8].upper()

        # ── REPLACE WITH QUICKFIX NewOrderSingle ────────────────────────────
        # msg = fix.Message()
        # msg.getHeader().setField(fix.MsgType(fix.MsgType_NewOrderSingle))
        # msg.setField(fix.ClOrdID(cl_ord_id))
        # msg.setField(fix.Symbol(f"{sym}-SSF-{contract}" if contract else sym))
        # msg.setField(fix.Side(fix.Side_Buy if side=="BUY" else fix.Side_Sell))
        # msg.setField(fix.OrdType(fix.OrdType_Market if ord_type=="MARKET"
        #                          else fix.OrdType_Limit))
        # if ord_type != "MARKET":
        #     msg.setField(fix.Price(price))
        # msg.setField(fix.OrderQty(qty))
        # msg.setField(fix.TimeInForce(fix.TimeInForce_GoodTillCancel))
        # fix.Session.sendToTarget(msg, self.session_id)
        # ────────────────────────────────────────────────────────────────────

        log.info(f"[{self.name}] PAPER ORDER: {side} {sym} "
                 f"{qty} @ {price} ({ord_type}) | id={cl_ord_id}")
        self.open_orders[cl_ord_id] = {
            "sym": sym, "side": side, "price": price,
            "qty": qty, "status": "PENDING"
        }
        return {"client_order_id": cl_ord_id, "status": "PENDING"}

    def cancel_replace(self, orig_cl_ord_id: str, new_price: int) -> dict:
        """Cancel existing order and replace at new price (35=G)."""
        import uuid
        new_cl_ord_id = str(uuid.uuid4())[:8].upper()

        # ── REPLACE WITH QUICKFIX OrderCancelReplaceRequest ─────────────────
        # msg = fix.Message()
        # msg.getHeader().setField(
        #     fix.MsgType(fix.MsgType_OrderCancelReplaceRequest))
        # msg.setField(fix.OrigClOrdID(orig_cl_ord_id))
        # msg.setField(fix.ClOrdID(new_cl_ord_id))
        # msg.setField(fix.Price(new_price))
        # fix.Session.sendToTarget(msg, self.session_id)
        # ────────────────────────────────────────────────────────────────────

        log.info(f"[{self.name}] CANCEL-REPLACE {orig_cl_ord_id} "
                 f"-> new price {new_price:,} | new_id={new_cl_ord_id}")
        return {"client_order_id": new_cl_ord_id, "status": "PENDING"}

    def cancel(self, cl_ord_id: str) -> bool:
        log.info(f"[{self.name}] CANCEL {cl_ord_id}")
        self.open_orders.pop(cl_ord_id, None)
        return True


# Global sessions — one for SSF, one for spot
ssf_session  = FIXSession("ssf")
spot_session = FIXSession("spot")
