# fcpo_s_calculator.py
# Three-source storage cost (S) engine — no Streamlit imports.

import math
import datetime
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


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


def load_mpob_history(filepath):
    """
    Reads MPOB stock history from Excel.
    Sheet: "Table Data", data from row 2 (index 1 after header), columns [Date, Value].
    Value already in tonnes (matches existing load_supply_demand() convention).
    Returns DataFrame [date (datetime), mpob_stocks (float, in tonnes)].
    """
    df = pd.read_excel(filepath, sheet_name="Table Data")
    df = df.iloc[1:].reset_index(drop=True)   # drop the "Close" text row
    df.columns = ["Date", "Value"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["mpob_stocks"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["mpob_stocks"])
    df = df.rename(columns={"Date": "date"})[["date", "mpob_stocks"]]
    return df.sort_values("date").reset_index(drop=True)


def estimate_capacity(mpob_stocks_series, assumed_max_util=0.875):
    """
    Returns dict: max_observed, implied_capacity, working_estimate.
    """
    max_obs = float(mpob_stocks_series.max())
    implied = max_obs / assumed_max_util
    # Round to nearest 250,000
    working = round(implied / 250_000) * 250_000
    return {
        'max_observed':    max_obs,
        'implied_capacity': implied,
        'working_estimate': working,
    }


def build_regression_dataset(mpob_df, contracts_dict, r_annual=0.03, capacity=3_750_000):
    """
    Returns DataFrame [date, utilisation, s_implied, F_M1, F_M2, mpob_stocks].
    Filters: 0.30 < utilisation < 0.98 AND 5 < s_implied_myr < 40.
    """
    from fcpo_spread_engine import implied_s_backsolve

    rows = []
    for _, row in mpob_df.iterrows():
        date = pd.Timestamp(row['date'])
        stocks = float(row['mpob_stocks'])
        util = stocks / capacity

        # MPOB date is month-start (e.g. 2023-01-01) but represents end-of-month stocks.
        # Use the last calendar day of that month for both front-month calc and price lookup.
        eom = (date + pd.offsets.MonthEnd(0))
        fm = _front_month(eom)
        m1_ym = fm
        m2_ym = _add_months(fm, 1)

        s1 = contracts_dict.get(m1_ym)
        s2 = contracts_dict.get(m2_ym)
        if s1 is None or s2 is None:
            continue

        # Get last available price on or before end of this month
        s1_avail = s1[s1.index <= eom]
        s2_avail = s2[s2.index <= eom]
        if s1_avail.empty or s2_avail.empty:
            continue

        F_M1 = float(s1_avail.iloc[-1])
        F_M2 = float(s2_avail.iloc[-1])

        s_res = implied_s_backsolve(F_M1, F_M2, r_annual)
        s_impl = s_res['s_implied_myr']

        rows.append({
            'date':        date,
            'utilisation': util,
            's_implied':   s_impl,
            'F_M1':        F_M1,
            'F_M2':        F_M2,
            'mpob_stocks': stocks,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Filter: remove extreme utilisation and implausible s values.
    # s_implied is monthly MYR/t. Negative = backwardation (valid).
    # Normal range: -300 to +150. Cap at ±500 to exclude only extreme outlier months.
    df = df[(df['utilisation'] > 0.30) & (df['utilisation'] < 0.98)]
    df = df[(df['s_implied'] > -500) & (df['s_implied'] < 150)]
    return df.reset_index(drop=True)


def fit_s_regression(reg_df):
    """
    Fits linear and exponential models of utilisation → s_implied.
    Returns dict with util_to_s_function, best_model, best_r2, params, etc.
    """
    if len(reg_df) < 12:
        # Fallback: return flat function
        def flat_fn(u):
            return 12.0
        return {
            'util_to_s_function': flat_fn,
            'best_model': 'linear',
            'best_r2': 0.0,
            'linear_params': {'alpha': 12.0, 'beta': 0.0},
            'linear_r2': 0.0,
            'exp_params': {'a': 5.0, 'b': 1.0, 'c': 5.0},
            'exp_r2': 0.0,
            'n_observations': len(reg_df),
            'util_range': (0.5, 0.9),
            's_range_observed': (5.0, 25.0),
            'warning': 'Insufficient data — fewer than 12 observations',
        }

    x = reg_df['utilisation'].values
    y = reg_df['s_implied'].values

    # Linear
    slope, intercept, r_lin, _, _ = linregress(x, y)
    r2_lin = r_lin ** 2

    def linear_fn(u, a=intercept, b=slope):
        return max(5.0, a + b * u)

    # Exponential: y = a*exp(b*x) + c
    r2_exp = 0.0
    exp_params = {'a': 5.0, 'b': 1.0, 'c': 5.0}
    exp_fn = None
    try:
        def exp_model(x_val, a, b, c):
            return a * np.exp(b * x_val) + c

        p0 = [1.0, 2.0, 5.0]
        popt, _ = curve_fit(exp_model, x, y, p0=p0, maxfev=5000)
        y_pred_exp = exp_model(x, *popt)
        ss_res = np.sum((y - y_pred_exp) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_exp = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        exp_params = {'a': float(popt[0]), 'b': float(popt[1]), 'c': float(popt[2])}

        def exp_fn(u, a=popt[0], b=popt[1], c=popt[2]):
            return max(5.0, a * math.exp(b * u) + c)
    except Exception:
        pass

    # Pick best model
    if r2_exp > r2_lin and exp_fn is not None:
        best_model = 'exponential'
        best_r2 = r2_exp
        best_fn = exp_fn
    else:
        best_model = 'linear'
        best_r2 = r2_lin
        best_fn = linear_fn

    warning = None
    if best_r2 < 0.45:
        warning = "Low R² — consider extending MPOB history"

    return {
        'util_to_s_function': best_fn,
        'best_model':         best_model,
        'best_r2':            best_r2,
        'linear_params':      {'alpha': float(intercept), 'beta': float(slope)},
        'linear_r2':          r2_lin,
        'exp_params':         exp_params,
        'exp_r2':             r2_exp,
        'n_observations':     len(reg_df),
        'util_range':         (float(x.min()), float(x.max())),
        's_range_observed':   (float(y.min()), float(y.max())),
        'warning':            warning,
    }


def fit_seasonal_regression(reg_df):
    """
    OLS with 11 month dummies (drop January as baseline) + utilisation.
    Returns dict: function, r2, beta_util, alpha, month_coefs, note.
    """
    if len(reg_df) < 12:
        def flat_fn(u, m):
            return 12.0
        return {'function': flat_fn, 'r2': 0.0, 'beta_util': 0.0,
                'alpha': 12.0, 'month_coefs': {}, 'note': 'Insufficient data'}

    df = reg_df.copy()
    df['month'] = pd.to_datetime(df['date']).dt.month

    # 11 month dummies (February through December; January is baseline)
    for m in range(2, 13):
        df[f'm{m}'] = (df['month'] == m).astype(float)

    feature_cols = ['utilisation'] + [f'm{m}' for m in range(2, 13)]
    X = df[feature_cols].values
    y = df['s_implied'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    alpha     = float(model.intercept_)
    beta_util = float(model.coef_[0])
    month_coefs = {MONTH_ABBRS[0]: 0.0}  # January baseline
    for i, m in enumerate(range(2, 13)):
        month_coefs[MONTH_ABBRS[m - 1]] = float(model.coef_[i + 1])

    def seasonal_fn(utilisation, month_int,
                    _alpha=alpha, _beta=beta_util, _mc=month_coefs):
        m_abbr = MONTH_ABBRS[month_int - 1]
        m_coef = _mc.get(m_abbr, 0.0)
        return max(5.0, _alpha + _beta * utilisation + m_coef)

    return {
        'function':   seasonal_fn,
        'r2':         r2,
        'beta_util':  beta_util,
        'alpha':      alpha,
        'month_coefs': month_coefs,
        'note':       f'Seasonal OLS (n={len(reg_df)})',
    }


def build_seasonal_s_table(mpob_df, regression_result, capacity=3_750_000):
    """
    Returns dict {month_int: {month_name, util_mean, util_p25, util_p75,
                               s_mean, s_low, s_high, n_years}}.
    """
    df = mpob_df.copy()
    df['month'] = df['date'].dt.month
    df['year']  = df['date'].dt.year
    df['util']  = df['mpob_stocks'] / capacity

    fn = regression_result['util_to_s_function']
    df['s_est'] = df['util'].apply(fn)

    result = {}
    for m in range(1, 13):
        mdf = df[df['month'] == m]
        if mdf.empty:
            result[m] = {
                'month_name': MONTH_ABBRS[m - 1],
                'util_mean': 0.65, 'util_p25': 0.55, 'util_p75': 0.75,
                's_mean': 12.0, 's_low': 8.0, 's_high': 16.0, 'n_years': 0
            }
            continue
        u = mdf['util']
        s = mdf['s_est']
        result[m] = {
            'month_name': MONTH_ABBRS[m - 1],
            'util_mean':  float(u.mean()),
            'util_p25':   float(u.quantile(0.25)),
            'util_p75':   float(u.quantile(0.75)),
            's_mean':     float(s.mean()),
            's_low':      float(s.quantile(0.25)),
            's_high':     float(s.quantile(0.75)),
            'n_years':    int(mdf['year'].nunique()),
        }
    return result


def get_s_mpob(current_stocks, regression_result, capacity=3_750_000):
    """
    Returns dict: utilisation, s_mpob_myr, regime.
    """
    util = current_stocks / capacity
    s_myr = regression_result['util_to_s_function'](util)

    if util < 0.50:
        regime = 'VERY LOOSE'
    elif util < 0.60:
        regime = 'LOOSE'
    elif util < 0.68:
        regime = 'BALANCED'
    elif util < 0.76:
        regime = 'TIGHTENING'
    elif util < 0.84:
        regime = 'TIGHT'
    else:
        regime = 'CRITICAL'

    return {'utilisation': util, 's_mpob_myr': s_myr, 'regime': regime}


def producer_s_composite(rel_pos, buyer_lifting, discount_pressure,
                         production_outlook, seasonal_table, current_month):
    """
    Returns dict: s_current, s_forward, s_monthly_mean/low/high, rel_pos,
                  conviction_bonus, signal, interpretation.
    rel_pos: float 0-1 (relative position within contact's historical range)
    seasonal_table: dict from build_seasonal_s_table() — has s_mean, s_low, s_high per month
    current_month: int 1-12
    """
    # Step 1: Get this month's S baseline from MPOB history
    month_data = seasonal_table.get(current_month, {})
    s_mean  = month_data.get('s_mean', 15.0)  # fallback if no data
    s_low   = month_data.get('s_low',  8.0)
    s_high  = month_data.get('s_high', 28.0)
    s_range = s_high - s_low

    # Step 2: Place producer within monthly range using relative position
    s_base = s_low + rel_pos * s_range

    # Step 3: Qualitative adjustments as % of monthly range (not fixed MYR)
    lifting_adj = {
        'rushing':      -0.20,   # -20% of range
        'on_time':       0.00,
        'slight_delay': +0.15,   # +15% of range
        'major_delay':  +0.35,   # +35% of range
    }.get(buyer_lifting, 0.0) * s_range

    discount_adj = {
        'none':      0.00,
        'small':    +0.10,   # +10% of range
        'large':    +0.30,   # +30% of range
        'distress': +0.60,   # +60% of range — distress is severe
    }.get(discount_pressure, 0.0) * s_range

    # Step 4: S current (front month)
    s_current = max(s_low, min(s_high * 1.2, s_base + lifting_adj + discount_adj))

    # Step 5: S forward (back month) — production outlook shifts from mean
    production_adj = {
        'light':  -0.25 * s_range,  # supply tightening → lower future storage
        'normal':  0.0,
        'heavy':  +0.30 * s_range,  # supply coming → stocks build → higher future S
    }.get(production_outlook, 0.0)

    s_forward = max(s_low, min(s_high * 1.2, s_mean + production_adj))

    # Conviction bonus
    conviction_bonus = 0
    if discount_pressure == 'distress':   conviction_bonus += 3
    if buyer_lifting == 'major_delay':    conviction_bonus += 2
    if buyer_lifting == 'rushing':        conviction_bonus += 2
    if production_outlook == 'heavy':     conviction_bonus += 2
    if rel_pos > 0.75:                    conviction_bonus += 1
    if rel_pos < 0.25:                    conviction_bonus += 1

    # Signal
    if discount_pressure == 'distress':
        signal = 'DISTRESS — front month selling likely'
    elif buyer_lifting in ('major_delay', 'slight_delay') and production_outlook == 'heavy':
        signal = 'BEARISH — rising stocks expected'
    elif buyer_lifting == 'rushing' and production_outlook == 'light':
        signal = 'BULLISH — drawdown expected'
    else:
        signal = 'NEUTRAL'

    month_name = MONTH_ABBRS[current_month - 1]
    interpretation = (
        f"Relative position: {rel_pos*100:.0f}th percentile of range. "
        f"Buyer lifting: {buyer_lifting.replace('_', ' ')}. "
        f"Discount: {discount_pressure}. "
        f"Production: {production_outlook}. "
        f"{month_name} S range (MPOB): {s_low:.1f}–{s_high:.1f} MYR/t."
    )

    return {
        's_current':        round(s_current, 1),
        's_forward':        round(s_forward, 1),
        's_monthly_mean':   round(s_mean, 1),
        's_monthly_low':    round(s_low, 1),
        's_monthly_high':   round(s_high, 1),
        'rel_pos':          round(rel_pos, 3),
        'conviction_bonus': conviction_bonus,
        'signal':           signal,
        'interpretation':   interpretation,
    }


def build_forward_s_curve(current_month, seasonal_table, current_mpob_stocks,
                           capacity, s_producer_current, s_producer_forward,
                           enso_factor=1.0):
    """
    Returns dict {pair_label: {s_value, source, confidence, trade_role,
                                seasonal_mean, gap_vs_seasonal}}.
    Pairs: M1/M2 through M11/M12.
    """
    current_util = current_mpob_stocks / capacity
    result = {}

    for offset in range(1, 12):  # M1/M2=1, M2/M3=2, ... M11/M12=11
        pair_label = f"M{offset}/M{offset+1}"

        # Near and far month for seasonal lookup (both sides of the spread)
        near_month   = ((current_month - 1 + offset - 1) % 12) + 1
        far_month    = ((current_month - 1 + offset)     % 12) + 1
        target_month = far_month  # kept for back-compat with seasonal logic below

        # Seasonal mean: average of near and far month baselines
        near_s_mean = seasonal_table.get(near_month, {}).get('s_mean', None)
        far_s_mean  = seasonal_table.get(far_month,  {}).get('s_mean', None)
        valid_means = [x for x in (near_s_mean, far_s_mean) if x is not None]
        pair_seasonal_mean = round(sum(valid_means) / len(valid_means), 1) if valid_means else None

        if offset == 1:
            s_val = s_producer_current
            source = 'producer'
            confidence = 'HIGH'
            trade_role = 'Intel only — do not trade'

        elif offset == 2:
            s_val = s_producer_forward
            source = 'producer'
            confidence = 'MED-HIGH'
            trade_role = 'Direction — approach with care'

        elif offset in (3, 4):
            seas = seasonal_table.get(target_month, {})
            s_seas = seas.get('s_mean', 12.0)
            seas_util_mean = seas.get('util_mean', 0.65)
            year_ratio = current_util / seas_util_mean if seas_util_mean > 0 else 1.0
            dampen = max(0, 1 - offset * 0.08)
            s_val = s_seas * (1 + (year_ratio - 1) * dampen)
            s_val = max(5.0, s_val)
            source = 'seasonal'
            confidence = 'MEDIUM'
            trade_role = 'Primary trading zone'

        elif offset == 5:
            seas = seasonal_table.get(target_month, {})
            s_seas = seas.get('s_mean', 12.0)
            seas_util_mean = seas.get('util_mean', 0.65)
            year_ratio = current_util / seas_util_mean if seas_util_mean > 0 else 1.0
            dampen = max(0, 1 - offset * 0.08)
            s_val = s_seas * (1 + (year_ratio - 1) * dampen)
            s_val = max(5.0, s_val)
            source = 'seasonal'
            confidence = 'LOW-MED'
            trade_role = 'Primary trading zone — wider stops'

        elif offset in (6, 7, 8):
            seas = seasonal_table.get(target_month, {})
            s_val = max(5.0, seas.get('s_mean', 12.0) * enso_factor)
            source = 'seasonal_enso'
            confidence = 'LOW'
            trade_role = 'Selective trading'

        else:  # 9–11
            seas = seasonal_table.get(target_month, {})
            s_val = max(5.0, seas.get('s_mean', 12.0) * enso_factor)
            source = 'seasonal_enso'
            confidence = 'VERY LOW'
            trade_role = 'Regime context only'

        gap_vs_seasonal = (
            round(round(s_val, 2) - pair_seasonal_mean, 1)
            if pair_seasonal_mean is not None else None
        )

        result[pair_label] = {
            's_value':          round(s_val, 2),
            'source':           source,
            'confidence':       confidence,
            'trade_role':       trade_role,
            'seasonal_mean':    pair_seasonal_mean,
            'gap_vs_seasonal':  gap_vs_seasonal,
        }

    return result


def three_source_gaps(s_implied, s_mpob, s_producer, threshold=8.0):
    """
    Returns dict with gap1, gap2, gap3 and their signals + story.
    """
    gap1 = s_implied - s_mpob
    gap2 = s_mpob - s_producer
    gap3 = s_implied - s_producer

    def _gap_signal(g, thr=5.0):
        if g > thr:   return 'SELL SPREAD'
        if g < -thr:  return 'BUY SPREAD'
        return 'NEUTRAL'

    gap3_signal = 'SELL SPREAD' if gap3 > threshold else ('BUY SPREAD' if gap3 < -threshold else 'NEUTRAL')
    triggered = abs(gap3) >= threshold

    if gap3 < -threshold:
        story = ("Producer tanks filling fast. Market hasn't priced it. "
                 "Buy spread before MPOB confirms.")
    elif gap3 > threshold:
        story = ("Market overstating tightness vs physical reality. Sell spread.")
    else:
        story = "All three sources broadly aligned. No gap signal."

    return {
        'gap1':            round(gap1, 2),
        'gap2':            round(gap2, 2),
        'gap3':            round(gap3, 2),
        'gap1_signal':     _gap_signal(gap1),
        'gap2_signal':     _gap_signal(gap2),
        'gap3_signal':     gap3_signal,
        'gap3_triggered':  triggered,
        'story':           story,
    }
