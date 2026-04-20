# main.py
"""
SSF Layered Auto-Quote Engine.

Usage:   python main.py
Keyboard (while running):
  Q  → quit
  R  → force requote all symbols
  P  → print current quote table
"""
import time, threading, datetime
import config
from engine.quoter        import Quoter
from engine.order_manager import OrderManager
from engine.fair_value    import get_near_contract, get_tte
from utils.logger         import get_logger, QuoteCSVLogger

log        = get_logger("main")
csv_logger = QuoteCSVLogger()

# ── select feed ───────────────────────────────────────────────────────────────
if   config.FEED_SOURCE == "sim":
    from feed.sim_feed    import fetch_spots;  INTERVAL = config.SIM_INTERVAL
elif config.FEED_SOURCE == "yahoo":
    from feed.yahoo_feed  import fetch_spots;  INTERVAL = config.YAHOO_INTERVAL
elif config.FEED_SOURCE == "broker":
    from feed.broker_feed import fetch_spots;  INTERVAL = config.BROKER_CONFIG["poll_interval_sec"]
else:
    raise ValueError(f"Unknown FEED_SOURCE: {config.FEED_SOURCE}")

quoter  = Quoter()
orders  = OrderManager()
running = True


def print_quote_table() -> None:
    """Print layered quote table for the near contract."""
    near  = get_near_contract()
    month = near["month"]
    now   = datetime.datetime.now().strftime("%H:%M:%S")
    w     = 90

    print(f"\n{'-'*w}")
    print(f"  SSF LAYERED QUOTES  |  {near['name']} contract  |  "
          f"TTE {get_tte(near['expiry'])}d  |  {now}  |  "
          f"{'PAPER' if config.PAPER_TRADING else 'LIVE'}")
    print(f"{'-'*w}")
    print(f"  {'SYM':<6} {'SPOT':>7} {'FV':>8}  "
          f"{'BID L1':>8} {'BID L2':>8} {'BID L3':>8}  "
          f"{'ASK L1':>8} {'ASK L2':>8} {'ASK L3':>8}")
    print(f"  {'':6} {'':7} {'':8}  "
          f"{'5 lot':>8} {'7 lot':>8} {'10 lot':>8}  "
          f"{'5 lot':>8} {'7 lot':>8} {'10 lot':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*8}  "
          f"{'-'*8} {'-'*8} {'-'*8}  "
          f"{'-'*8} {'-'*8} {'-'*8}")

    for sym, contracts in quoter.live_quotes.items():
        if month not in contracts:
            continue
        qs = contracts[month]
        b  = [l.price for l in qs.bid_layers]
        a  = [l.price for l in qs.ask_layers]
        # pad if layers < 3
        while len(b) < 3: b.append(0)
        while len(a) < 3: a.append(0)
        print(f"  {qs.sym:<6} {qs.spot:>7,} {qs.fv:>8.0f}  "
              f"{b[0]:>8,} {b[1]:>8,} {b[2]:>8,}  "
              f"{a[0]:>8,} {a[1]:>8,} {a[2]:>8,}")

    print(f"{'-'*w}")
    print(f"  Requotes: {quoter.requote_count}  |  "
          f"Feed: {config.FEED_SOURCE.upper()}  |  "
          f"Trigger: spot move >= 1 tick")
    print()


def check_auto_roll() -> None:
    for c in config.CONTRACTS:
        tte = get_tte(c["expiry"])
        if tte == 0:
            log.warning(f"EXPIRED: {c['name']} — roll positions now!")
        elif tte <= 3:
            log.warning(f"NEAR EXPIRY: {c['name']} — {tte}d left")


def run_engine() -> None:
    log.info("=" * 60)
    log.info("  SSF Layered Auto-Quote Engine")
    log.info(f"  Feed     : {config.FEED_SOURCE.upper()}")
    log.info(f"  Mode     : {'PAPER TRADING' if config.PAPER_TRADING else '⚠  LIVE'}")
    log.info(f"  Symbols  : {len(config.STOCKS)}")
    log.info(f"  Bid layers: " +
             ", ".join(f"L{i+1} -{t}t×{l}lot"
                       for i,(t,l) in enumerate(config.BID_LAYERS)))
    log.info(f"  Ask layers: " +
             ", ".join(f"L{i+1} +{t}t×{l}lot"
                       for i,(t,l) in enumerate(config.ASK_LAYERS)))
    log.info("=" * 60)

    # initial full requote
    spots = fetch_spots()
    if spots:
        for sym, qsets in quoter.force_requote_all(spots).items():
            for qs in qsets:
                orders.send_quote_set(qs)
                csv_logger.log(qs)
        if config.PRINT_QUOTES:
            print_quote_table()

    while running:
        try:
            spots = fetch_spots()
            if not spots:
                log.warning("No prices received — retrying...")
                time.sleep(INTERVAL)
                continue

            requoted = quoter.process_spot_update(spots)

            if requoted:
                for sym, qsets in requoted.items():
                    for qs in qsets:
                        orders.send_quote_set(qs)
                        csv_logger.log(qs)
                if config.PRINT_QUOTES:
                    print_quote_table()

            check_auto_roll()
            time.sleep(INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Engine error: {e}", exc_info=True)
            time.sleep(5)

    log.info("Engine stopped.")


def keyboard_listener():
    global running
    while running:
        try:
            cmd = input().strip().upper()
            if cmd == "Q":
                running = False
            elif cmd == "R":
                spots = fetch_spots()
                if spots:
                    for sym, qsets in quoter.force_requote_all(spots).items():
                        for qs in qsets:
                            orders.send_quote_set(qs)
                print_quote_table()
            elif cmd == "P":
                print_quote_table()
        except EOFError:
            break


if __name__ == "__main__":
    threading.Thread(target=keyboard_listener, daemon=True).start()
    run_engine()
