# engine/fair_value.py
import datetime
import config


def get_tte(expiry_date: datetime.date) -> int:
    """Calendar days to expiry. Returns 0 if already expired."""
    return max(0, (expiry_date - datetime.date.today()).days)


def get_near_contract() -> dict:
    """Return nearest non-expired contract. This is the auto-roll logic."""
    today = datetime.date.today()
    for c in config.CONTRACTS:
        if c["expiry"] >= today:
            return c
    return config.CONTRACTS[-1]


def calc_fair_value(spot: float, tte: int, div_yield_pct: float) -> float:
    """
    SSF theoretical fair value.
    FV = Spot × (1 + (JIBOR + BorrowCost - DivYield) × TTE / 365)
    """
    rate = (config.JIBOR_1M + config.BORROWING_COST) / 100
    div  = div_yield_pct / 100
    return spot * (1 + (rate - div) * tte / 365)


def round_to_tick(price: float, tick: int) -> int:
    """Round any price to the nearest valid tick."""
    return int(round(price / tick) * tick)


def calc_cost_per_lot(spot: float) -> float:
    """Estimated round-trip transaction cost per lot (IDR)."""
    spot_cost = spot * (config.SPOT_BUY_COST + config.SPOT_SELL_COST) / 100
    fut_cost  = config.FUTURE_BUY_COST / 100   # per lot approx
    return spot_cost + fut_cost
