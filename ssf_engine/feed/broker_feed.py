# feed/broker_feed.py
"""
PLACEHOLDER — fill in when you have broker API credentials.
To activate:
  1. Set FEED_SOURCE = "broker" in config.py
  2. Fill BROKER_CONFIG in config.py
  3. Uncomment the requests blocks below and adjust to your broker's schema
"""
import requests
import config
from utils.logger import get_logger

log = get_logger("broker_feed")

def fetch_spots() -> dict:
    """GET spot prices for all symbols from broker API."""
    base    = config.BROKER_CONFIG["base_url"]
    key     = config.BROKER_CONFIG["api_key"]
    headers = {"Authorization": f"Bearer {key}"}
    prices  = {}
    tick_map = {s["sym"]: s["tick"] for s in config.STOCKS}

    for stock in config.STOCKS:
        sym = stock["sym"]
        try:
            # ── REPLACE WITH YOUR BROKER ENDPOINT ───────────────────────────
            # r = requests.get(f"{base}/quotes",
            #     params={"symbol": sym, "type": "spot"},
            #     headers=headers, timeout=5)
            # r.raise_for_status()
            # price = r.json()["last"]
            # tick  = tick_map[sym]
            # prices[sym] = int(round(price / tick) * tick)
            # ────────────────────────────────────────────────────────────────
            raise NotImplementedError("Broker feed not configured. Edit feed/broker_feed.py")
        except NotImplementedError as e:
            log.error(str(e))
            return {}
        except Exception as e:
            log.warning(f"{sym} fetch error: {e}")

    return prices


def place_order(sym: str, contract: str, side: str,
                price: int, qty: int) -> dict:
    """
    Place one limit order.
    side: "BID" (buy) or "ASK" (sell)
    Returns dict with at least {"order_id": "..."}
    """
    base    = config.BROKER_CONFIG["base_url"]
    key     = config.BROKER_CONFIG["api_key"]
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type":  "application/json"}
    body = {
        "symbol":        f"{sym}-SSF-{contract}",
        "side":          "BUY" if side == "BID" else "SELL",
        "type":          "LIMIT",
        "price":         price,
        "quantity":      qty,
        "time_in_force": "GTC",
        "account_type":  "SSF",
    }
    # ── UNCOMMENT WHEN READY ─────────────────────────────────────────────────
    # r = requests.post(f"{base}/orders", json=body, headers=headers, timeout=5)
    # r.raise_for_status()
    # return r.json()
    raise NotImplementedError("place_order not configured.")


def cancel_order(order_id: str) -> bool:
    """Cancel one open order by ID. Returns True if successful."""
    base    = config.BROKER_CONFIG["base_url"]
    key     = config.BROKER_CONFIG["api_key"]
    headers = {"Authorization": f"Bearer {key}"}
    # ── UNCOMMENT WHEN READY ─────────────────────────────────────────────────
    # r = requests.delete(f"{base}/orders/{order_id}",
    #     headers=headers, timeout=5)
    # return r.status_code == 200
    raise NotImplementedError("cancel_order not configured.")
