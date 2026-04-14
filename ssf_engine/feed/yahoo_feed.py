# feed/yahoo_feed.py
import yfinance as yf
import config
from utils.logger import get_logger

log = get_logger("yahoo_feed")

def fetch_spots() -> dict:
    """Batch-fetch all symbols in one yfinance call."""
    tickers  = [s["yahoo"] for s in config.STOCKS]
    sym_map  = {s["yahoo"]: s["sym"] for s in config.STOCKS}
    tick_map = {s["sym"]: s["tick"] for s in config.STOCKS}
    try:
        data  = yf.download(tickers, period="1d", interval="1m",
                             progress=False, auto_adjust=True)
        close = data["Close"]
        prices = {}
        for yahoo_sym, sym in sym_map.items():
            try:
                price = float(close[yahoo_sym].dropna().iloc[-1])
                tick  = tick_map[sym]
                prices[sym] = int(round(price / tick) * tick)
            except Exception as e:
                log.warning(f"Skip {yahoo_sym}: {e}")
        return prices
    except Exception as e:
        log.error(f"Yahoo fetch failed: {e}")
        return {}
