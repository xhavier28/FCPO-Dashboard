# config.py
import datetime

# ── FEED SOURCE ───────────────────────────────────────────────────────────────
# "sim"    → random walk simulation
# "yahoo"  → yfinance (~15 min delay)
# "broker" → broker REST API placeholder
# "fix"    → live prices via FIX market data session
FEED_SOURCE = "yahoo"

# ── MARKET PARAMETERS ────────────────────────────────────────────────────────
JIBOR_1M        = 4.23
BORROWING_COST  = 2.00
SPOT_BUY_COST   = 0.095
SPOT_SELL_COST  = 0.195
FUTURE_BUY_COST = 1000

# ── LAYERED QUOTE PARAMETERS ─────────────────────────────────────────────────
# (tick_offset_from_FV, lots)
# Layer 1 = closest to FV (best price, shallowest)
# Layer 3 = furthest from FV (deepest, most lots)
BID_LAYERS = [
    (3,  5),   # Layer 1: FV - 3 ticks,  5 lots
    (4,  7),   # Layer 2: FV - 4 ticks,  7 lots
    (5, 10),   # Layer 3: FV - 5 ticks, 10 lots
]
ASK_LAYERS = [
    (3,  5),   # Layer 1: FV + 3 ticks,  5 lots
    (4,  7),   # Layer 2: FV + 4 ticks,  7 lots
    (5, 10),   # Layer 3: FV + 5 ticks, 10 lots
]

# ── CONTRACTS ────────────────────────────────────────────────────────────────
CONTRACTS = [
    {"month": 4, "name": "April", "expiry": datetime.date(2026, 4, 30)},
    {"month": 5, "name": "Mei",   "expiry": datetime.date(2026, 5, 29)},
    {"month": 6, "name": "Juni",  "expiry": datetime.date(2026, 6, 30)},
]

# ── STOCKS ───────────────────────────────────────────────────────────────────
# max_lots      : hard position cap per stock (combined SSF + spot legs)
# resume_pct    : (layer3_resume, layer2_resume) as % of max_lots
#                 layer1 only resumes at 0 (fully flat)
# spot_leverage : broker margin leverage on spot leg (e.g. 10 = 10x)
# spot_order    : "MARKET" or "LIMIT_PLUS_1" for spot hedge execution
# div           : dividend yield % per contract month — editable in Streamlit
STOCKS = [
    {
        "sym": "BBCA", "yahoo": "BBCA.JK", "tick": 25,
        "max_lots": 50, "resume_pct": (80, 50),
        "spot_leverage": 10, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 0.0, 5: 0.0, 6: 0.0},
    },
    {
        "sym": "BBRI", "yahoo": "BBRI.JK", "tick": 10,
        "max_lots": 50, "resume_pct": (80, 50),
        "spot_leverage": 10, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 9.65, 5: 0.0, 6: 0.0},
    },
    {
        "sym": "ASII", "yahoo": "ASII.JK", "tick": 25,
        "max_lots": 40, "resume_pct": (80, 50),
        "spot_leverage": 10, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 0.0, 5: 6.82, 6: 0.0},
    },
    {
        "sym": "TLKM", "yahoo": "TLKM.JK", "tick": 10,
        "max_lots": 40, "resume_pct": (80, 50),
        "spot_leverage": 10, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 0.0, 5: 0.0, 6: 0.0},
    },
    {
        "sym": "MDKA", "yahoo": "MDKA.JK", "tick": 10,
        "max_lots": 30, "resume_pct": (80, 50),
        "spot_leverage": 8, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 0.0, 5: 0.0, 6: 0.0},
    },
    {
        "sym": "AMRT", "yahoo": "AMRT.JK", "tick": 5,
        "max_lots": 30, "resume_pct": (80, 50),
        "spot_leverage": 5, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 0.0, 5: 0.0, 6: 0.0},
    },
    {
        "sym": "ANTM", "yahoo": "ANTM.JK", "tick": 10,
        "max_lots": 30, "resume_pct": (80, 50),
        "spot_leverage": 8, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 0.0, 5: 0.0, 6: 0.0},
    },
    {
        "sym": "BRPT", "yahoo": "BRPT.JK", "tick": 5,
        "max_lots": 20, "resume_pct": (80, 50),
        "spot_leverage": 5, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 0.0, 5: 0.0, 6: 0.0},
    },
    {
        "sym": "BMRI", "yahoo": "BMRI.JK", "tick": 10,
        "max_lots": 50, "resume_pct": (80, 50),
        "spot_leverage": 10, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 11.6, 5: 0.0, 6: 0.0},
    },
    {
        "sym": "INDF", "yahoo": "INDF.JK", "tick": 25,
        "max_lots": 40, "resume_pct": (80, 50),
        "spot_leverage": 8, "spot_order": "LIMIT_PLUS_1",
        "div": {4: 0.0, 5: 0.0, 6: 0.0},
    },
]

# ── FIX SESSION CONFIG ────────────────────────────────────────────────────────
# Fill these when broker provides FIX credentials
FIX_CONFIG = {
    "ssf": {
        "config_file":    "fix/ssf.cfg",
        "sender_comp_id": "YOUR_SENDER_ID",     # <- replace
        "target_comp_id": "YOUR_TARGET_ID",     # <- replace
        "host":           "fix.YOUR-BROKER.co.id",
        "port":           9880,
    },
    "spot": {
        "config_file":    "fix/spot.cfg",
        "sender_comp_id": "YOUR_SENDER_ID_SPOT",
        "target_comp_id": "YOUR_TARGET_ID_SPOT",
        "host":           "fix.YOUR-BROKER.co.id",
        "port":           9881,
    },
}

# ── ENGINE SETTINGS ──────────────────────────────────────────────────────────
PAPER_TRADING  = True    # True = never send real orders
YAHOO_INTERVAL = 30      # seconds between Yahoo fetches
SIM_INTERVAL   = 0.9     # seconds between sim ticks
LOG_TO_FILE    = True
PRINT_QUOTES   = True
