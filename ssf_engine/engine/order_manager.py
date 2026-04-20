# engine/order_manager.py
"""
Paper mode:  logs every layer to console + CSV. No real orders.
Live mode:   cancel all existing layers for the symbol, then
             place new bid and ask layers via broker API.

Cancel-replace is done atomically per symbol:
  1. Cancel all 6 open orders (3 bid + 3 ask) for that symbol+contract
  2. Place 6 new orders at the shifted prices
"""
import config
from engine.quoter import QuoteSet, Layer
from utils.logger import get_logger

log = get_logger("order_manager")


class OrderManager:
    def __init__(self):
        # open_orders[sym][month][side][layer_num] = order_id
        self.open_orders: dict = {}
        self.fill_log:    list = []
        self.paper_pnl:   float = 0.0

    def send_quote_set(self, qs: QuoteSet) -> None:
        """Entry point: paper or live depending on config."""
        if config.PAPER_TRADING:
            self._paper_send(qs)
        else:
            self._live_cancel_replace(qs)

    # ── PAPER MODE ────────────────────────────────────────────────────────────

    def _paper_send(self, qs: QuoteSet) -> None:
        """Log all layers. No real orders."""
        log.info(qs.display())

    # ── LIVE MODE ─────────────────────────────────────────────────────────────

    def _live_cancel_replace(self, qs: QuoteSet) -> None:
        """
        Cancel all existing layers for this sym+contract,
        then place new shifted layers.
        """
        from feed import broker_feed

        sym   = qs.sym
        month = qs.month

        # ── Step 1: Cancel all existing open orders ───────────────────────────
        existing = (self.open_orders
                    .get(sym, {})
                    .get(month, {}))
        for side, layers in existing.items():
            for layer_num, order_id in layers.items():
                if order_id:
                    try:
                        broker_feed.cancel_order(order_id)
                        log.debug(f"Cancelled {sym}-{qs.contract_name} "
                                  f"{side} L{layer_num} order {order_id}")
                    except Exception as e:
                        log.warning(f"Cancel failed for {order_id}: {e}")

        # ── Step 2: Place new layered orders ─────────────────────────────────
        if sym not in self.open_orders:
            self.open_orders[sym] = {}
        self.open_orders[sym][month] = {"BID": {}, "ASK": {}}

        for layer in qs.bid_layers + qs.ask_layers:
            try:
                result = broker_feed.place_order(
                    sym          = sym,
                    contract     = qs.contract_name,
                    side         = layer.side,       # "BID" or "ASK"
                    price        = layer.price,
                    qty          = layer.lots,
                )
                order_id = result.get("order_id")
                self.open_orders[sym][month][layer.side][layer.layer_num] = order_id
                log.info(f"PLACED {layer} → order_id={order_id}")
            except Exception as e:
                log.error(f"Place order failed {layer}: {e}")

    # ── FILL RECORDING ────────────────────────────────────────────────────────

    def record_fill(self, sym: str, contract: str, side: str,
                    fill_price: int, fv: float, lots: int) -> None:
        """
        Record a fill and calculate PnL vs fair value.
        BID fill (bought long):  PnL = (FV - fill_price) × lots × 100
        ASK fill (sold short):   PnL = (fill_price - FV) × lots × 100
        """
        if side == "BID":
            pnl = (fv - fill_price) * lots * 100
        else:
            pnl = (fill_price - fv) * lots * 100
        self.paper_pnl += pnl

        record = dict(sym=sym, contract=contract, side=side,
                      fill_price=fill_price, fv=fv, lots=lots, pnl=pnl)
        self.fill_log.append(record)
        log.info(f"FILL {sym}-{contract} {side} {lots}L "
                 f"@ {fill_price:,} | PnL={pnl:+,.0f} IDR | "
                 f"total={self.paper_pnl:+,.0f} IDR")
