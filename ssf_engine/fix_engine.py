# fix_engine.py
"""
Main FIX engine process. Run this in Terminal 1.
  python fix_engine.py

Connects FIX sessions, runs quote loop, writes all state to state.db.
Streamlit (app.py) reads from state.db independently.
"""
import time
import config
from db import database as db
from engine.quoter import Quoter
from engine.order_manager import OrderManager
from fix.fix_session import ssf_session, spot_session
from utils.logger import get_logger

log = get_logger("fix_engine")

# select feed
if config.FEED_SOURCE == "sim":
    from feed.sim_feed   import fetch_spots; INTERVAL = config.SIM_INTERVAL
elif config.FEED_SOURCE == "yahoo":
    from feed.yahoo_feed import fetch_spots; INTERVAL = config.YAHOO_INTERVAL
else:
    from feed.sim_feed   import fetch_spots; INTERVAL = config.SIM_INTERVAL

quoter = Quoter()
orders = OrderManager()


def main():
    db.init_db()
    db.set_engine_state("status", "RUNNING")
    db.set_engine_state("feed",   config.FEED_SOURCE)
    db.set_engine_state("mode",   "PAPER" if config.PAPER_TRADING else "LIVE")

    ssf_session.connect()
    spot_session.connect()

    log.info("=" * 60)
    log.info("  FIX Engine started")
    log.info(f"  Feed: {config.FEED_SOURCE.upper()}")
    log.info(f"  Mode: {'PAPER' if config.PAPER_TRADING else 'LIVE'}")
    log.info("=" * 60)

    # initial full requote
    spots = fetch_spots()
    if spots:
        for sym, qsets in quoter.force_requote_all(spots).items():
            for qs in qsets:
                orders.send_quote_set(qs)

    while True:
        try:
            spots    = fetch_spots()
            requoted = quoter.process_spot_update(spots)
            for sym, qsets in requoted.items():
                for qs in qsets:
                    orders.send_quote_set(qs)
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            log.info("Engine stopped by user.")
            db.set_engine_state("status", "STOPPED")
            break
        except Exception as e:
            log.error(f"Engine error: {e}", exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
