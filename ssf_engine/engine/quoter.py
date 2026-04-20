# engine/quoter.py
"""
Generates layered bid/ask quotes respecting position limits.
On each spot tick:
  1. Check which layers are active (based on position tracker)
  2. Build QuoteSet with only active layers
  3. Return requoted symbols for order manager to cancel-replace
"""
import datetime
import config
from engine.fair_value       import calc_fair_value, round_to_tick, get_tte, calc_cost_per_lot
from engine.position_tracker import get_active_layers
from db import database as db


class Layer:
    def __init__(self, side: str, price: int, lots: int,
                 tick_offset: int, layer_num: int, filled: bool = False):
        self.side        = side
        self.price       = price
        self.lots        = lots
        self.tick_offset = tick_offset
        self.layer_num   = layer_num
        self.filled      = filled
        self.order_id    = None

    def to_dict(self) -> dict:
        return {
            "side":        self.side,
            "price":       self.price,
            "lots":        self.lots,
            "tick_offset": self.tick_offset,
            "layer_num":   self.layer_num,
            "filled":      self.filled,
        }

    def __repr__(self):
        status = "FILLED" if self.filled else "ACTIVE"
        return f"{self.side} L{self.layer_num} {self.price:,}x{self.lots}lot [{status}]"


class QuoteSet:
    def __init__(self, sym, contract_name, month, spot, fv, tte, tick, cost,
                 bid_layers, ask_layers):
        self.sym           = sym
        self.contract_name = contract_name
        self.month         = month
        self.spot          = spot
        self.fv            = fv
        self.tte           = tte
        self.tick          = tick
        self.cost          = cost
        self.bid_layers    = bid_layers
        self.ask_layers    = ask_layers
        self.timestamp     = datetime.datetime.now()

    @property
    def best_bid(self):
        return self.bid_layers[0].price if self.bid_layers else 0

    @property
    def best_ask(self):
        return self.ask_layers[0].price if self.ask_layers else 0

    def display(self) -> str:
        bids = " | ".join(str(l) for l in self.bid_layers) or "-- none active --"
        asks = " | ".join(str(l) for l in self.ask_layers) or "-- none active --"
        return (f"[{self.sym}-{self.contract_name}] "
                f"spot={self.spot:,} fv={self.fv:.0f} | "
                f"BIDS: {bids} | ASKS: {asks}")

    def persist(self) -> None:
        """Write current quote state to SQLite."""
        db.upsert_quote(
            sym        = self.sym,
            month      = self.month,
            contract   = self.contract_name,
            spot       = self.spot,
            fv         = self.fv,
            tte        = self.tte,
            bid_layers = [l.to_dict() for l in self.bid_layers],
            ask_layers = [l.to_dict() for l in self.ask_layers],
        )


def _build_active_layers(fv: float, tick: int,
                         layer_config: list, side: str,
                         active_nums: list) -> list:
    layers = []
    for i, (offset_ticks, lots) in enumerate(layer_config):
        layer_num = i + 1
        if layer_num not in active_nums:
            continue
        if side == "BID":
            price = round_to_tick(fv - offset_ticks * tick, tick)
        else:
            price = round_to_tick(fv + offset_ticks * tick, tick)
        layers.append(Layer(side, price, lots, offset_ticks, layer_num))
    return layers


class Quoter:
    def __init__(self):
        self.prev_spots:  dict = {}
        self.live_quotes: dict = {}
        self.requote_count = 0
        for s in config.STOCKS:
            self.prev_spots[s["sym"]]  = 0
            self.live_quotes[s["sym"]] = {}

    def needs_requote(self, sym: str, new_spot: float) -> bool:
        tick  = next(s["tick"] for s in config.STOCKS if s["sym"] == sym)
        moved = abs(new_spot - self.prev_spots.get(sym, 0))
        return moved >= tick

    def build_quote_set(self, sym: str, spot: float, contract: dict) -> QuoteSet:
        stock      = next(s for s in config.STOCKS if s["sym"] == sym)
        tick       = stock["tick"]
        tte        = get_tte(contract["expiry"])
        div        = self._get_div(sym, contract["month"])
        fv         = calc_fair_value(spot, tte, div)
        cost       = calc_cost_per_lot(spot)
        active_bid = get_active_layers(sym, "BID")
        active_ask = get_active_layers(sym, "ASK")
        bid_layers = _build_active_layers(fv, tick, config.BID_LAYERS, "BID", active_bid)
        ask_layers = _build_active_layers(fv, tick, config.ASK_LAYERS, "ASK", active_ask)
        return QuoteSet(sym, contract["name"], contract["month"],
                        spot, fv, tte, tick, cost, bid_layers, ask_layers)

    def _get_div(self, sym: str, month: int) -> float:
        key  = f"div_{sym}"
        divs = db.get_config(key)
        if divs and str(month) in divs:
            return divs[str(month)]
        stock = next(s for s in config.STOCKS if s["sym"] == sym)
        return stock["div"].get(month, 0.0)

    def process_spot_update(self, spots: dict) -> dict:
        requoted = {}
        for sym, spot in spots.items():
            if not self.needs_requote(sym, spot):
                continue
            quote_sets = []
            for contract in config.CONTRACTS:
                qs = self.build_quote_set(sym, spot, contract)
                qs.persist()
                self.live_quotes[sym][contract["month"]] = qs
                quote_sets.append(qs)
            self.prev_spots[sym] = spot
            self.requote_count  += 1
            requoted[sym]        = quote_sets
        return requoted

    def force_requote_all(self, spots: dict) -> dict:
        for sym in self.prev_spots:
            self.prev_spots[sym] = 0
        return self.process_spot_update(spots)
