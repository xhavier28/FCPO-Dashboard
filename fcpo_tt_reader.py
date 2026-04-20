# fcpo_tt_reader.py
# Reads FCPO prices from SharePoint Excel — three zones per sheet.
# Zone A: outright M1-M12 prices (paste from TT outright grid)
# Zone B: listed calendar spread contract prices (paste from TT FCPOS grid)
# Zone C: listed butterfly contract prices (paste from TT FCPOB grid)
# Rolling MA and Z-scores come from Raw Data CSVs — NOT this file.

import openpyxl
from pathlib import Path
from datetime import datetime

# ── UPDATE THIS PATH ──────────────────────────────────────────────────────────
SHAREPOINT_FILE = r"Z:\YourSharePointFolder\FCPO_Curve_Input.xlsx"

CURVE_SHEET    = "Curve Input"
PRICE_COL      = 3       # column C in all three zones
ZONE_A_START   = 7       # M1 outright price
ZONE_B_START   = 24      # M1/M2 listed spread contract
ZONE_C_START   = 39      # M1/M2/M3 listed butterfly contract


def _read_zone(ws, start_row, n_rows, col=PRICE_COL):
    """Read n_rows prices from col starting at start_row. Returns list of float|None."""
    out = []
    for i in range(n_rows):
        val = ws.cell(start_row + i, col).value
        try:
            out.append(float(val) if val not in (None, "", "—") else None)
        except (ValueError, TypeError):
            out.append(None)
    return out


def read_all(filepath=SHAREPOINT_FILE):
    """
    Reads all three zones from the SharePoint Excel file.
    Returns:
      {
        'outrights':   {1: price, 2: price, ... 12: price},
        'spreads':     {(1,2): price, (2,3): price, ... (11,12): price},
        'butterflies': {(1,2,3): price, ... (10,11,12): price}
      }
    Any value is None if that cell was not filled in.
    Returns None if file is unavailable.
    """
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        ws = wb[CURVE_SHEET]
        a  = _read_zone(ws, ZONE_A_START, 12)
        b  = _read_zone(ws, ZONE_B_START, 11)
        c_ = _read_zone(ws, ZONE_C_START, 10)
        wb.close()
        return {
            "outrights":   {i+1: p for i, p in enumerate(a)},
            "spreads":     {(i+1, i+2): p for i, p in enumerate(b)},
            "butterflies": {(i+1, i+2, i+3): p for i, p in enumerate(c_)},
        }
    except Exception:
        return None


def get_outrights(filepath=SHAREPOINT_FILE):
    """Returns {offset: price} for M1-M12 outrights only, or None."""
    data = read_all(filepath)
    return data["outrights"] if data else None


def is_available(filepath=SHAREPOINT_FILE):
    """True if file is readable and M1 outright price exists."""
    try:
        d = read_all(filepath)
        return d is not None and d["outrights"].get(1) is not None
    except Exception:
        return False


def get_last_update_time(filepath=SHAREPOINT_FILE):
    """Returns file last-modified datetime, or None."""
    try:
        return datetime.fromtimestamp(Path(filepath).stat().st_mtime)
    except Exception:
        return None


def compute_gaps(data):
    """
    Computes gap between listed contract price and value calculated from outrights.
    Gap = listed price − calculated value.
    Positive gap → listed is RICH vs outrights (sell listed, buy outrights).
    Negative gap → listed is CHEAP vs outrights (buy listed, sell outrights).
    Returns dict with 'spread_gaps' and 'butterfly_gaps'.
    """
    if not data:
        return None
    oc = data["outrights"]

    spread_gaps = {}
    for (near, far), listed in data["spreads"].items():
        p_near = oc.get(near)
        p_far  = oc.get(far)
        if listed is not None and p_near and p_far:
            calc = p_near - p_far   # near minus far — matches existing convention
            spread_gaps[(near, far)] = {
                "listed":     listed,
                "calculated": round(calc, 2),
                "gap":        round(listed - calc, 2),
                "signal":     "RICH"  if listed - calc >  5 else
                              "CHEAP" if listed - calc < -5 else "FAIR"
            }

    butter_gaps = {}
    for (fr, mi, bk), listed in data["butterflies"].items():
        p_fr = oc.get(fr); p_mi = oc.get(mi); p_bk = oc.get(bk)
        if listed is not None and p_fr and p_mi and p_bk:
            calc = p_mi - 0.5 * (p_fr + p_bk)
            butter_gaps[(fr, mi, bk)] = {
                "listed":     listed,
                "calculated": round(calc, 2),
                "gap":        round(listed - calc, 2),
                "signal":     "RICH"  if listed - calc >  3 else
                              "CHEAP" if listed - calc < -3 else "FAIR"
            }

    return {"spread_gaps": spread_gaps, "butterfly_gaps": butter_gaps}


if __name__ == "__main__":
    data = read_all()
    if data:
        print(f"Outrights:   {sum(1 for v in data['outrights'].values() if v)}/12")
        print(f"Spreads:     {sum(1 for v in data['spreads'].values() if v)}/11")
        print(f"Butterflies: {sum(1 for v in data['butterflies'].values() if v)}/10")
        gaps = compute_gaps(data)
        if gaps:
            print("\nSpread gaps (listed vs calculated):")
            for k, v in gaps["spread_gaps"].items():
                print(f"  M{k[0]}/M{k[1]}: listed={v['listed']:+.1f}  "
                      f"calc={v['calculated']:+.1f}  gap={v['gap']:+.1f}  [{v['signal']}]")
    else:
        print("File unavailable. Update SHAREPOINT_FILE path.")
