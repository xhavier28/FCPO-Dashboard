# utils/logger.py
import logging, csv, os, datetime
import config

os.makedirs("logs", exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(name)-14s] %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if config.LOG_TO_FILE:
        fh = logging.FileHandler(
            f"logs/engine_{datetime.date.today():%Y%m%d}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class QuoteCSVLogger:
    """Append every requote (all layers) to a daily CSV."""
    def __init__(self):
        self.path = f"logs/quotes_{datetime.date.today():%Y%m%d}.csv"
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "time", "sym", "contract", "tte", "spot", "fv",
                    "side", "layer", "price", "lots", "tick_offset"
                ])

    def log(self, qs) -> None:
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            t = qs.timestamp.strftime("%H:%M:%S.%f")
            for layer in qs.bid_layers + qs.ask_layers:
                w.writerow([
                    t, qs.sym, qs.contract_name, qs.tte,
                    qs.spot, f"{qs.fv:.2f}",
                    layer.side, layer.layer_num,
                    layer.price, layer.lots, layer.tick_offset
                ])
