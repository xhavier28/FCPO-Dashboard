# db/database.py
"""
Shared SQLite database — single communication layer between
fix_engine.py and app.py (Streamlit).

Tables:
  quotes       — current live layered quotes per sym+contract
  positions    — open spread positions (SSF leg + spot leg)
  fills        — fill history
  config       — runtime config overrides (div yields, spot_order toggles)
  engine_state — engine status (running, stopped, feed source)
"""
import sqlite3
import json
import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "state.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS quotes (
            sym          TEXT,
            contract     TEXT,
            month        INTEGER,
            spot         REAL,
            fv           REAL,
            tte          INTEGER,
            bid_layers   TEXT,
            ask_layers   TEXT,
            updated_at   TEXT,
            PRIMARY KEY (sym, month)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            sym              TEXT,
            contract         TEXT,
            ssf_side         TEXT,
            ssf_entry_price  INTEGER,
            ssf_current      INTEGER,
            spot_side        TEXT,
            spot_entry_price INTEGER,
            spot_current     INTEGER,
            lots             INTEGER,
            spot_leverage    REAL,
            tte              INTEGER,
            ssf_pnl          REAL,
            spot_pnl         REAL,
            total_pnl        REAL,
            action           TEXT,
            status           TEXT,
            opened_at        TEXT,
            closed_at        TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS fills (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            sym        TEXT,
            contract   TEXT,
            leg        TEXT,
            side       TEXT,
            layer_num  INTEGER,
            price      INTEGER,
            lots       INTEGER,
            fv         REAL,
            filled_at  TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS engine_state (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()


# ── QUOTES ────────────────────────────────────────────────────────────────────

def upsert_quote(sym: str, month: int, contract: str,
                 spot: float, fv: float, tte: int,
                 bid_layers: list, ask_layers: list) -> None:
    conn = get_conn()
    conn.execute("""
        INSERT INTO quotes (sym, contract, month, spot, fv, tte,
                            bid_layers, ask_layers, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(sym, month) DO UPDATE SET
            spot=excluded.spot, fv=excluded.fv, tte=excluded.tte,
            bid_layers=excluded.bid_layers, ask_layers=excluded.ask_layers,
            updated_at=excluded.updated_at
    """, (sym, contract, month, spot, fv, tte,
          json.dumps(bid_layers), json.dumps(ask_layers),
          datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_all_quotes() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM quotes ORDER BY sym, month").fetchall()
    conn.close()
    result = []
    for r in rows:
        row = dict(r)
        row["bid_layers"] = json.loads(row["bid_layers"])
        row["ask_layers"] = json.loads(row["ask_layers"])
        result.append(row)
    return result


# ── POSITIONS ─────────────────────────────────────────────────────────────────

def insert_position(sym: str, contract: str,
                    ssf_side: str, ssf_entry: int,
                    spot_side: str, spot_entry: int,
                    lots: int, spot_leverage: float, tte: int) -> int:
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO positions
        (sym, contract, ssf_side, ssf_entry_price, ssf_current,
         spot_side, spot_entry_price, spot_current,
         lots, spot_leverage, tte, ssf_pnl, spot_pnl, total_pnl,
         action, status, opened_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,0,0,0,'HOLD_EXPIRE','OPEN',?)
    """, (sym, contract, ssf_side, ssf_entry, ssf_entry,
          spot_side, spot_entry, spot_entry,
          lots, spot_leverage, tte,
          datetime.datetime.now().isoformat()))
    pos_id = cur.lastrowid
    conn.commit()
    conn.close()
    return pos_id


def update_position_prices(pos_id: int, ssf_current: int,
                            spot_current: int, tte: int) -> None:
    conn = get_conn()
    pos = conn.execute(
        "SELECT * FROM positions WHERE id=?", (pos_id,)).fetchone()
    if not pos:
        conn.close()
        return

    if pos["ssf_side"] == "LONG":
        ssf_pnl = (ssf_current - pos["ssf_entry_price"]) * pos["lots"] * 100
    else:
        ssf_pnl = (pos["ssf_entry_price"] - ssf_current) * pos["lots"] * 100

    if pos["spot_side"] == "LONG":
        spot_pnl = (spot_current - pos["spot_entry_price"]) * pos["lots"] * 100
    else:
        spot_pnl = (pos["spot_entry_price"] - spot_current) * pos["lots"] * 100

    total_pnl = ssf_pnl + spot_pnl

    conn.execute("""
        UPDATE positions SET
            ssf_current=?, spot_current=?, tte=?,
            ssf_pnl=?, spot_pnl=?, total_pnl=?
        WHERE id=?
    """, (ssf_current, spot_current, tte,
          ssf_pnl, spot_pnl, total_pnl, pos_id))
    conn.commit()
    conn.close()


def set_position_action(pos_id: int, action: str) -> None:
    conn = get_conn()
    conn.execute("UPDATE positions SET action=? WHERE id=?", (action, pos_id))
    conn.commit()
    conn.close()


def close_position(pos_id: int) -> None:
    conn = get_conn()
    conn.execute("""
        UPDATE positions SET status='CLOSED', closed_at=?
        WHERE id=?
    """, (datetime.datetime.now().isoformat(), pos_id))
    conn.commit()
    conn.close()


def get_open_positions() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM positions WHERE status='OPEN' ORDER BY opened_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_total_lots(sym: str) -> int:
    conn = get_conn()
    row = conn.execute("""
        SELECT COALESCE(SUM(lots), 0) as total
        FROM positions WHERE sym=? AND status='OPEN'
    """, (sym,)).fetchone()
    conn.close()
    return int(row["total"])


# ── FILLS ─────────────────────────────────────────────────────────────────────

def record_fill(sym: str, contract: str, leg: str, side: str,
                layer_num: int, price: int, lots: int, fv: float) -> None:
    conn = get_conn()
    conn.execute("""
        INSERT INTO fills
        (sym, contract, leg, side, layer_num, price, lots, fv, filled_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (sym, contract, leg, side, layer_num, price, lots, fv,
          datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()


# ── CONFIG ────────────────────────────────────────────────────────────────────

def get_config(key: str, default=None):
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM config WHERE key=?", (key,)).fetchone()
    conn.close()
    if row:
        return json.loads(row["value"])
    return default


def set_config(key: str, value) -> None:
    conn = get_conn()
    conn.execute("""
        INSERT INTO config (key, value) VALUES (?,?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, json.dumps(value)))
    conn.commit()
    conn.close()


# ── ENGINE STATE ──────────────────────────────────────────────────────────────

def get_engine_state(key: str, default=None):
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM engine_state WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_engine_state(key: str, value: str) -> None:
    conn = get_conn()
    conn.execute("""
        INSERT INTO engine_state (key, value) VALUES (?,?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, value))
    conn.commit()
    conn.close()
