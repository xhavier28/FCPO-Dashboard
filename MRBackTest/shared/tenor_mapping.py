"""
Shared tenor mapping utilities for FCPO minute data.

Provides front_month() and tenor_to_contract_month() for converting
between rolling tenor labels (Current, +1M, ..., +9M) and specific
calendar contract months, based on the observation date.

Roll rule: day 1-15 -> front month = current calendar month
           day 16+  -> front month = next calendar month
"""

from datetime import date


def front_month(d):
    """
    Determine the front-month contract for a given date.

    Parameters
    ----------
    d : date or datetime-like with .day, .month, .year attributes

    Returns
    -------
    tuple (year, month) representing the front-month contract
    """
    if d.day <= 15:
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


def tenor_to_contract_month(d, tenor_offset):
    """
    Map an observation date + tenor offset to the specific calendar
    contract month.

    Parameters
    ----------
    d : date or datetime-like
    tenor_offset : int (0=Current, 1=+1M, ..., 9=+9M)

    Returns
    -------
    tuple (year, month) of the calendar contract
    """
    fm = front_month(d)
    return add_months(fm, tenor_offset)


def contract_month_to_str(ym):
    """Convert (year, month) to 'Mmm{YY}' format, e.g. (2026, 4) -> 'Apr26'."""
    MONTH_ABBRS = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    y, m = ym
    return f"{MONTH_ABBRS[m - 1]}{str(y)[2:]}"
