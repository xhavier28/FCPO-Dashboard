# config.py — ONLY FILE YOU NEED TO EDIT
import datetime

# ── FEED SOURCE ───────────────────────────────────────────────────────────────
# "sim"    → random walk (no API, always works)
# "yahoo"  → yfinance real prices (~15 min delay)
# "broker" → broker REST API (fill BROKER_CONFIG below when ready)
FEED_SOURCE = "yahoo"

# ── MARKET PARAMETERS ────────────────────────────────────────────────────────
JIBOR_1M        = 4.23   # % — 1 month JIBOR
BORROWING_COST  = 2.00   # % — short borrowing cost
SPOT_BUY_COST   = 0.095  # % — spot buy transaction cost
SPOT_SELL_COST  = 0.195  # % — spot sell transaction cost
FUTURE_BUY_COST = 1000   # IDR — fixed futures transaction cost

# ── LAYERED QUOTE PARAMETERS ─────────────────────────────────────────────────
# Each entry = (offset_in_ticks_from_FV, lots)
# Bid layers go DOWN from FV, Ask layers go UP from FV.
# First entry = closest to FV (best price), last = furthest (deepest).
BID_LAYERS = [
    (3, 5),    # Layer 1: FV - 3 ticks,  5 lots  ← profit target
    (4, 7),    # Layer 2: FV - 4 ticks,  7 lots
    (5, 10),   # Layer 3: FV - 5 ticks, 10 lots
]
ASK_LAYERS = [
    (3, 5),    # Layer 1: FV + 3 ticks,  5 lots  ← profit target
    (4, 7),    # Layer 2: FV + 4 ticks,  7 lots
    (5, 10),   # Layer 3: FV + 5 ticks, 10 lots
]

# ── CONTRACTS ────────────────────────────────────────────────────────────────
CONTRACTS = [
    {"month": 4, "name": "April", "expiry": datetime.date(2026, 4, 30)},
    {"month": 5, "name": "Mei",   "expiry": datetime.date(2026, 5, 29)},
    {"month": 6, "name": "Juni",  "expiry": datetime.date(2026, 6, 30)},
]

# ── STOCKS ───────────────────────────────────────────────────────────────────
# div: dividend yield % per contract month — update when dividend announced
STOCKS = [
    {"sym": "BBCA", "yahoo": "BBCA.JK", "tick": 25, "div": {4: 0,    5: 0,    6: 0   }},
    {"sym": "BBRI", "yahoo": "BBRI.JK", "tick": 10, "div": {4: 9.65, 5: 0,    6: 0   }},
    {"sym": "ASII", "yahoo": "ASII.JK", "tick": 25, "div": {4: 0,    5: 6.82, 6: 0   }},
    {"sym": "TLKM", "yahoo": "TLKM.JK", "tick": 10, "div": {4: 0,    5: 0,    6: 0   }},
    {"sym": "MDKA", "yahoo": "MDKA.JK", "tick": 10, "div": {4: 0,    5: 0,    6: 0   }},
    {"sym": "AMRT", "yahoo": "AMRT.JK", "tick":  5, "div": {4: 0,    5: 0,    6: 0   }},
    {"sym": "ANTM", "yahoo": "ANTM.JK", "tick": 10, "div": {4: 0,    5: 0,    6: 0   }},
    {"sym": "BRPT", "yahoo": "BRPT.JK", "tick":  5, "div": {4: 0,    5: 0,    6: 0   }},
    {"sym": "BMRI", "yahoo": "BMRI.JK", "tick": 10, "div": {4: 11.6, 5: 0,    6: 0   }},
    {"sym": "INDF", "yahoo": "INDF.JK", "tick": 25, "div": {4: 0,    5: 0,    6: 0   }},
]

# ── BROKER API CONFIG (fill when you have credentials) ───────────────────────
BROKER_CONFIG = {
    "base_url":          "https://api.YOUR-BROKER.co.id/v1",
    "api_key":           "YOUR_API_KEY_HERE",
    "poll_interval_sec": 2,
}

# ── ENGINE SETTINGS ──────────────────────────────────────────────────────────
PAPER_TRADING  = True    # True = log only, never send real orders
YAHOO_INTERVAL = 30      # seconds between Yahoo fetches
SIM_INTERVAL   = 0.9     # seconds between sim ticks
LOG_TO_FILE    = True    # write logs/quotes_YYYYMMDD.csv
PRINT_QUOTES   = True    # print layered quote table to console on requote
