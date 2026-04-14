# feed/sim_feed.py
import random
import config

BASE = {
    "BBCA": 6750, "BBRI": 3450, "ASII": 6225, "TLKM": 3180,
    "MDKA": 3190, "AMRT": 1525, "ANTM": 3770, "BRPT": 2350,
    "BMRI": 4670, "INDF": 6800,
}
_current = dict(BASE)

def fetch_spots() -> dict:
    tick_map = {s["sym"]: s["tick"] for s in config.STOCKS}
    for sym in _current:
        tick   = tick_map[sym]
        drift  = random.gauss(0, tick * 1.5)
        revert = (BASE[sym] - _current[sym]) * 0.002  # mean reversion
        raw    = _current[sym] + drift + revert
        _current[sym] = max(tick, round(raw / tick) * tick)
    return dict(_current)
