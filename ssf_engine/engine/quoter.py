# engine/quoter.py
"""
Generates layered bid/ask quotes for all symbols × all contracts.

For each symbol, produces:
  - N bid layers: prices going DOWN from FV, lots increasing with depth
  - N ask layers: prices going UP from FV, lots increasing with depth

All layers shift together on every requote (cancel-all → replace-all).
"""
import datetime
import config
from engine.fair_value import calc_fair_value, round_to_tick, get_tte, calc_cost_per_lot


class Layer:
    """A single price level in the order book."""
    def __init__(self, side: str, price: int, lots: int,
                 tick_offset: int, layer_num: int):
        self.side        = side        # "BID" or "ASK"
        self.price       = price
        self.lots        = lots
        self.tick_offset = tick_offset # how many ticks from FV
        self.layer_num   = layer_num   # 1 = closest to FV, 3 = deepest
        self.order_id    = None        # filled by order_manager when live

    def __repr__(self):
        return f"{self.side} L{self.layer_num} {self.price:,} × {self.lots}lot"


class QuoteSet:
    """All layers for one symbol + one contract, at one point in time."""
    def __init__(self, sym: str, contract_name: str, month: int,
                 spot: float, fv: float, tte: int, tick: int, cost: float,
                 bid_layers: list[Layer], ask_layers: list[Layer]):
        self.sym           = sym
        self.contract_name = contract_name
        self.month         = month
        self.spot          = spot
        self.fv            = fv
        self.tte           = tte
        self.tick          = tick
        self.cost          = cost
        self.bid_layers    = bid_layers   # sorted best→deep (price high→low)
        self.ask_layers    = ask_layers   # sorted best→deep (price low→high)
        self.timestamp     = datetime.datetime.now()

    @property
    def best_bid(self) -> int:
        return self.bid_layers[0].price if self.bid_layers else 0

    @property
    def best_ask(self) -> int:
        return self.ask_layers[0].price if self.ask_layers else 0

    @property
    def spread(self) -> int:
        return self.best_ask - self.best_bid

    def display(self) -> str:
        """One-line summary for logging."""
        bids = " | ".join(f"{l.price:,}×{l.lots}" for l in self.bid_layers)
        asks = " | ".join(f"{l.price:,}×{l.lots}" for l in self.ask_layers)
        return (f"[{self.sym}-{self.contract_name}] "
                f"spot={self.spot:,} fv={self.fv:.0f} "
                f"BID[{bids}]  ASK[{asks}]")


def _build_layers(fv: float, tick: int, layer_config: list[tuple],
                  side: str) -> list[Layer]:
    """
    Build Layer objects from config tuples (tick_offset, lots).
    BID: price = FV - offset*tick  (goes down)
    ASK: price = FV + offset*tick  (goes up)
    """
    layers = []
    for i, (offset_ticks, lots) in enumerate(layer_config):
        if side == "BID":
            price = round_to_tick(fv - offset_ticks * tick, tick)
        else:
            price = round_to_tick(fv + offset_ticks * tick, tick)
        layers.append(Layer(side, price, lots, offset_ticks, i + 1))
    return layers


class Quoter:
    def __init__(self):
        # last spot price that triggered a requote, per symbol
        self.prev_spots:  dict[str, float] = {}
        # current live QuoteSets: {sym: {month: QuoteSet}}
        self.live_quotes: dict[str, dict]  = {}
        self.requote_count = 0

        for s in config.STOCKS:
            self.prev_spots[s["sym"]]  = 0   # 0 forces first requote
            self.live_quotes[s["sym"]] = {}

    def needs_requote(self, sym: str, new_spot: float) -> bool:
        tick   = next(s["tick"] for s in config.STOCKS if s["sym"] == sym)
        moved  = abs(new_spot - self.prev_spots.get(sym, 0))
        return moved >= tick

    def build_quote_set(self, sym: str, spot: float,
                        contract: dict) -> QuoteSet:
        """Calculate FV and build all bid/ask layers for one contract."""
        stock = next(s for s in config.STOCKS if s["sym"] == sym)
        tick  = stock["tick"]
        tte   = get_tte(contract["expiry"])
        div   = stock["div"].get(contract["month"], 0)
        fv    = calc_fair_value(spot, tte, div)
        cost  = calc_cost_per_lot(spot)

        bid_layers = _build_layers(fv, tick, config.BID_LAYERS, "BID")
        ask_layers = _build_layers(fv, tick, config.ASK_LAYERS, "ASK")

        return QuoteSet(
            sym           = sym,
            contract_name = contract["name"],
            month         = contract["month"],
            spot          = spot,
            fv            = fv,
            tte           = tte,
            tick          = tick,
            cost          = cost,
            bid_layers    = bid_layers,
            ask_layers    = ask_layers,
        )

    def process_spot_update(self, spots: dict[str, float]) -> dict[str, list[QuoteSet]]:
        """
        Called on every price feed tick.
        Returns {sym: [QuoteSet × 3 contracts]} ONLY for symbols
        that moved >= 1 tick. Empty dict = no requotes needed.
        """
        requoted = {}
        for sym, spot in spots.items():
            if not self.needs_requote(sym, spot):
                continue

            quote_sets = []
            for contract in config.CONTRACTS:
                qs = self.build_quote_set(sym, spot, contract)
                self.live_quotes[sym][contract["month"]] = qs
                quote_sets.append(qs)

            self.prev_spots[sym] = spot
            self.requote_count  += 1
            requoted[sym]        = quote_sets

        return requoted

    def force_requote_all(self, spots: dict[str, float]) -> dict[str, list[QuoteSet]]:
        """Reset all prev_spots to force a full requote regardless of movement."""
        for sym in self.prev_spots:
            self.prev_spots[sym] = 0
        return self.process_spot_update(spots)
