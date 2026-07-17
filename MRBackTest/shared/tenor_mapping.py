"""
Shared tenor mapping utilities for FCPO minute data.

Provides front_month() and tenor_to_contract_month() for converting
between rolling tenor labels (Current, +1M, ..., +9M) and specific
calendar contract months, based on the observation date.

Roll rules (instrument-specific):
  Spread:    day 1-15  -> front month = current calendar month
             day 16+   -> front month = next calendar month
  Butterfly: day 1     -> front month = current calendar month
             day 2+    -> front month = next calendar month

Butterfly rolls earlier because butterfly contracts expire much earlier
in the month (day 2-14 observed) vs spreads (day 9-15 observed).
"""

from datetime import date

# Roll day thresholds: if date.day <= ROLL_DAY, front_month = current month
ROLL_DAY_SPREAD = 15
ROLL_DAY_BUTTERFLY = 1


def front_month(d, instrument_type="spread"):
    """
    Determine the front-month contract for a given date.

    Parameters
    ----------
    d : date or datetime-like with .day, .month, .year attributes
    instrument_type : str, "spread" or "butterfly"
        Spread rolls on day 16 (roll_day=15).
        Butterfly rolls on day 2 (roll_day=1).

    Returns
    -------
    tuple (year, month) representing the front-month contract
    """
    roll_day = ROLL_DAY_BUTTERFLY if instrument_type == "butterfly" else ROLL_DAY_SPREAD

    if d.day <= roll_day:
        return (d.year, d.month)
    else:
        m = d.month + 1
        y = d.year + (1 if m > 12 else 0)
        return (y, m if m <= 12 else m - 12)


def add_months(ym, n):
    """
    Add n months to a (year, month) tuple.

    Parameters
    ----------
    ym : tuple (year, month)
    n  : int, number of months to add

    Returns
    -------
    tuple (year, month)
    """
    y, m = ym
    m += n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return (y, m)


def tenor_to_contract_month(d, tenor_offset, instrument_type="spread"):
    """
    Map an observation date + tenor offset to the specific calendar
    contract month.

    Parameters
    ----------
    d : date or datetime-like
    tenor_offset : int (0=Current, 1=+1M, ..., 9=+9M)
    instrument_type : str, "spread" or "butterfly"

    Returns
    -------
    tuple (year, month) of the calendar contract
    """
    fm = front_month(d, instrument_type=instrument_type)
    return add_months(fm, tenor_offset)


def contract_month_to_str(ym):
    """Convert (year, month) to 'Mmm{YY}' format, e.g. (2026, 4) -> 'Apr26'."""
    MONTH_ABBRS = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    y, m = ym
    return f"{MONTH_ABBRS[m - 1]}{str(y)[2:]}"
