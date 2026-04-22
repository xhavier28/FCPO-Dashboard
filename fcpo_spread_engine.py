# fcpo_spread_engine.py
# Pure calculation module — no Streamlit imports.
# All functions return plain Python objects or DataFrames.

import math
import datetime
import pandas as pd
import numpy as np


# ── Shared helpers (mirrored from app.py to avoid circular import) ─────────────

def _front_month(date):
    if date.day <= 15:
        return (date.year, date.month)
    else:
        m = date.month + 1
        y = date.year + (1 if m > 12 else 0)
        return (y, m % 12 or 12)


def _add_months(ym, n):
    y, m = ym
    m += n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return (y, m)


def get_active_curve(contracts_dict, as_of_date):
    """
    Returns {offset: price} where offset 1=M1 (front month), 2=M2, ... 12=M12.
    Returns None for any month not found.
    as_of_date: datetime.date
    """
    if isinstance(as_of_date, datetime.datetime):
        as_of_date = as_of_date.date()
    as_of_ts = pd.Timestamp(as_of_date)
    fm = _front_month(as_of_ts)
    result = {}
    for offset in range(1, 13):
        ym = _add_months(fm, offset - 1)
        series = contracts_dict.get(ym)
        if series is None:
            result[offset] = None
            continue
        # Get the most recent price on or before as_of_date
        available = series[series.index <= as_of_ts]
        result[offset] = float(available.iloc[-1]) if not available.empty else None
    return result


def _build_aligned_prices(contracts_dict, offsets, lookback_days=180):
    """
    Internal helper: build a DataFrame with one column per offset, aligned by date.
    offset 1 = M1 (front month), etc.
    Returns DataFrame indexed by date with columns [off1, off2, ...] and date column.
    """
    # Collect all dates across all contracts
    all_dates = sorted(set(d for s in contracts_dict.values() for d in s.index))
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=lookback_days + 90)  # extra buffer for rolling
    all_dates = [d for d in all_dates if d >= cutoff and d <= today]

    rows = []
    for date in all_dates:
        fm = _front_month(date)
        row = {'date': date}
        for offset in offsets:
            ym = _add_months(fm, offset - 1)
            s = contracts_dict.get(ym)
            if s is not None and date in s.index:
                row[f'off{offset}'] = float(s[date])
            else:
                row[f'off{offset}'] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def _add_rolling_stats(df, value_col):
    """Add ma20/45/90, std20/45/90, z20/45/90 columns for value_col."""
    for w in [20, 45, 90]:
        df[f'ma{w}']  = df[value_col].rolling(w, min_periods=max(w//2, 5)).mean()
        df[f'std{w}'] = df[value_col].rolling(w, min_periods=max(w//2, 5)).std()
        df[f'z{w}']   = (df[value_col] - df[f'ma{w}']) / df[f'std{w}'].replace(0, np.nan)
    return df


def build_spread_history(contracts_dict, near_offset, far_offset, lookback_days=180):
    """
    Returns DataFrame: [date, F_near, F_far, spread, ma20, ma45, ma90,
                         std20, std45, std90, z20, z45, z90]
    spread = F_near - F_far (positive = backwardation)
    Last lookback_days rows only.
    """
    df = _build_aligned_prices(contracts_dict, [near_offset, far_offset],
                                lookback_days=lookback_days)
    df = df.rename(columns={f'off{near_offset}': 'F_near', f'off{far_offset}': 'F_far'})
    df = df.dropna(subset=['F_near', 'F_far']).reset_index(drop=True)
    df['spread'] = df['F_near'] - df['F_far']
    df = _add_rolling_stats(df, 'spread')
    return df.tail(lookback_days).reset_index(drop=True)


def build_butterfly_history(contracts_dict, front_offset, mid_offset, back_offset, lookback_days=180):
    """
    Returns DataFrame: [date, F_front, F_mid, F_back, butterfly, ma20..z90]
    butterfly = F_mid - 0.5*(F_front + F_back)
    """
    df = _build_aligned_prices(contracts_dict,
                                [front_offset, mid_offset, back_offset],
                                lookback_days=lookback_days)
    df = df.rename(columns={
        f'off{front_offset}': 'F_front',
        f'off{mid_offset}':   'F_mid',
        f'off{back_offset}':  'F_back',
    })
    df = df.dropna(subset=['F_front', 'F_mid', 'F_back']).reset_index(drop=True)
    df['butterfly'] = df['F_mid'] - 0.5 * (df['F_front'] + df['F_back'])
    df = _add_rolling_stats(df, 'butterfly')
    return df.tail(lookback_days).reset_index(drop=True)


def fair_spread_value(F_near, r_annual, s_myr_per_tonne, c_annual, dt=1/12):
    """
    Fair value of calendar spread in MYR/tonne.
    Formula: F_near * (exp((r_annual/12 + s_myr_per_tonne/F_near - c_annual/12) * dt) - 1)
    """
    exponent = (r_annual / 12 + s_myr_per_tonne / F_near - c_annual / 12) * dt
    return F_near * (math.exp(exponent) - 1)


def implied_s_backsolve(F_near, F_far, r_annual=0.03, dt=1/12):
    """
    Back-solves implied storage cost from M1/M2 prices.
    Returns dict with s_implied_rate, s_implied_myr, c_implied.
    """
    if F_near <= 0 or F_far <= 0:
        return {'s_implied_rate': 0.0, 's_implied_myr': 0.0, 'c_implied': 0.0}
    s_implied_rate = (1 / dt) * math.log(F_far / F_near) - r_annual / 12
    s_implied_myr  = s_implied_rate * F_near
    # Negative values are valid: backwardation implies convenience yield > storage + financing
    return {
        's_implied_rate': s_implied_rate,
        's_implied_myr':  s_implied_myr,
        'c_implied':      0.0,
    }


def implied_c(F_near, F_far, s_myr, r_annual=0.03, dt=1/12):
    """
    Returns convenience yield annualised (e.g. 0.04 = 4%).
    c_monthly = r_annual/12 + s_myr/F_near - (1/dt)*ln(F_far/F_near)
    """
    if F_near <= 0 or F_far <= 0:
        return 0.0
    c_monthly = r_annual / 12 + s_myr / F_near - (1 / dt) * math.log(F_far / F_near)
    return c_monthly * 12


def conviction_score(z_vs_ma, z_vs_fv, c_annual, z_short, z_medium):
    """
    Returns dict: score (0-7), direction, size, breakdown.
    """
    direction = 'sell' if z_vs_ma > 0 else ('buy' if z_vs_ma < 0 else 'none')
    score = 0
    breakdown = []

    if abs(z_vs_ma) > 2.0:
        score += 1
        breakdown.append(("Z vs MA > 2.0", 1))
    if abs(z_vs_ma) > 2.5:
        score += 1
        breakdown.append(("Z vs MA > 2.5", 1))
    if abs(z_vs_fv) > 1.5:
        score += 1
        breakdown.append(("Z vs FV > 1.5", 1))
    if abs(z_vs_fv) > 2.0:
        score += 1
        breakdown.append(("Z vs FV > 2.0", 1))
    if z_vs_ma != 0 and z_vs_fv != 0 and (z_vs_ma > 0) == (z_vs_fv > 0):
        score += 1
        breakdown.append(("Z vs MA and Z vs FV same sign", 1))
    if direction == 'buy' and c_annual > 0.04:
        score += 1
        breakdown.append(("c > 4% — favours buy", 1))
    elif direction == 'sell' and c_annual < 0.01:
        score += 1
        breakdown.append(("c < 1% — favours sell", 1))
    if z_short != 0 and z_medium != 0 and (z_short > 0) == (z_medium > 0):
        score += 1
        breakdown.append(("Z_short and Z_medium same sign", 1))

    if score >= 6:
        size = '1.0x'
    elif score >= 4:
        size = '0.75x'
    elif score >= 2:
        size = '0.50x'
    else:
        size = 'no trade'

    return {'score': score, 'direction': direction, 'size': size, 'breakdown': breakdown}


def scenario_interpretation(z_calendar, z_butterfly):
    """Returns one-sentence interpretation string."""
    ac = abs(z_calendar)
    ab = abs(z_butterfly)

    def _rich_cheap(z):
        return 'rich' if z > 0 else 'cheap'

    if ac > 2 and ab > 2 and (z_calendar > 0) == (z_butterfly > 0):
        mid = 'M2' if z_butterfly > 0 else 'M2'
        return (f"Both calendar and butterfly {_rich_cheap(z_calendar)} — "
                f"M2 is likely the culprit.")
    if ac > 2 and ab < 1:
        direction = 'steepening' if z_calendar > 0 else 'flattening'
        return (f"Calendar {_rich_cheap(z_calendar)} but butterfly neutral — "
                f"curve {direction}, not a kink.")
    if ab > 2 and ac < 1:
        mid_status = 'rich' if z_butterfly > 0 else 'cheap'
        return (f"Butterfly {mid_status} but calendar neutral — "
                f"pure mid-month kink.")
    if ac < 1 and ab < 1:
        return "Both instruments within normal range — no trade."
    return (f"Calendar Z={z_calendar:.2f}, butterfly Z={z_butterfly:.2f} — "
            f"mixed signals, monitor closely.")


def entry_conditions_checklist(z_short, z_medium, spread, fair_value, c_annual, direction):
    """
    Returns list of dicts: {rule, condition, met, warning}
    """
    checklist = []
    d = direction if direction in ('buy', 'sell') else 'sell'  # default for display

    # 1. z_short threshold
    if d == 'sell':
        met = z_short > 2.0
        warn = 2.0 < z_short < 2.5
        checklist.append({'rule': 'Z_short threshold', 'condition': 'Z_20d > 2.0 (sell)',
                           'met': met, 'warning': warn and met})
    else:
        met = z_short < -2.0
        warn = -2.5 < z_short < -2.0
        checklist.append({'rule': 'Z_short threshold', 'condition': 'Z_20d < -2.0 (buy)',
                           'met': met, 'warning': warn and met})

    # 2. z_medium confirms
    if d == 'sell':
        met = z_medium > 0.5
        checklist.append({'rule': 'Z_medium confirms', 'condition': 'Z_45d > 0.5 (sell)',
                           'met': met, 'warning': 0.5 < z_medium < 1.0})
    else:
        met = z_medium < -0.5
        checklist.append({'rule': 'Z_medium confirms', 'condition': 'Z_45d < -0.5 (buy)',
                           'met': met, 'warning': -1.0 < z_medium < -0.5})

    # 3. spread vs fair value
    if d == 'sell':
        met = spread > fair_value
        checklist.append({'rule': 'Spread vs FV', 'condition': 'Spread > FV (sell)',
                           'met': met, 'warning': False})
    else:
        met = spread < fair_value
        checklist.append({'rule': 'Spread vs FV', 'condition': 'Spread < FV (buy)',
                           'met': met, 'warning': False})

    # 4. convenience yield regime
    if d == 'sell':
        met = c_annual < 0.01
        warn = 0.01 <= c_annual < 0.025
        checklist.append({'rule': 'Convenience yield regime', 'condition': 'c < 1% (sell favoured)',
                           'met': met, 'warning': warn})
    else:
        met = c_annual > 0.04
        warn = 0.025 <= c_annual <= 0.04
        checklist.append({'rule': 'Convenience yield regime', 'condition': 'c > 4% (buy favoured)',
                           'met': met, 'warning': warn})

    # 5. cross-window agreement
    same_sign = (z_short > 0 and z_medium > 0) or (z_short < 0 and z_medium < 0)
    checklist.append({'rule': 'Cross-window agreement', 'condition': 'Z_short and Z_medium same sign',
                       'met': same_sign, 'warning': False})

    # 6. distress flag (placeholder — producer log checked upstream)
    checklist.append({'rule': 'No distress flag', 'condition': 'No distress selling reported (last 7d)',
                       'met': True, 'warning': False})

    return checklist
