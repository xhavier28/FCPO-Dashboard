"""
DIAGNOSTIC — TM Lead Time Before SB/C Transitions
===================================================
Measures whether TM 1-week's persistence probability shows genuine
lead time before resting states (SB, C) break into transitional shapes.

Run:  python "research/04. backtest_analysis/diagnostic_tm_lead_time.py"
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime
from models.tm_engine import predict as tm_predict

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_csv('research/03. validation_analysis/shape_log_enriched.csv',
                 dtype={'shape': str, 'prior_shape': str})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Date index for lookups
date_to_idx = {d: i for i, d in enumerate(df['date'])}
dates = df['date'].values

print("=" * 70)
print("DIAGNOSTIC — TM LEAD TIME BEFORE SB/C TRANSITIONS")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# PART 1 — IDENTIFY TRANSITION EVENTS
# ══════════════════════════════════════════════════════════════════
RESTING = {'0.0', '1'}
WINDOW_BEFORE = 10
WINDOW_AFTER = 3

events = []
for i in range(1, len(df)):
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    if prev['shape'] in RESTING and curr['shape'] != prev['shape']:
        events.append({
            'flip_date': curr['date'],
            'flip_idx': i,
            'origin': prev['shape'],
            'destination': curr['shape'],
            'days_in_resting': int(prev['days_in_shape']),
        })

ev = pd.DataFrame(events)
print(f"\nPart 1 — Transition events: {len(ev)}")
print(f"  SB-origin: {(ev['origin'] == '0.0').sum()}")
print(f"  C-origin:  {(ev['origin'] == '1').sum()}")

# Crosstab
ct = pd.crosstab(ev['origin'], ev['destination'], margins=True)
print(f"\n  Origin x Destination:")
for line in ct.to_string().split('\n'):
    print(f"    {line}")

# Flag thin cells
thin = []
for o in ct.index[:-1]:
    for d in ct.columns[:-1]:
        n = ct.loc[o, d]
        if 0 < n < 5:
            thin.append(f"{o}->{d} (n={n})")
if thin:
    print(f"  Thin cells (n<5): {', '.join(thin)}")

# ══════════════════════════════════════════════════════════════════
# PART 2 — MEASURE TM PERSISTENCE PROBABILITY IN THE RUN-UP
# ══════════════════════════════════════════════════════════════════
print(f"\nPart 2 — Pulling TM persistence probabilities...")
print(f"  Window: t-{WINDOW_BEFORE} to t+{WINDOW_AFTER} around each flip")

# Pre-compute all TM 1w predictions for efficiency
# Collect all unique dates we need
needed_dates = set()
for _, e in ev.iterrows():
    idx = e['flip_idx']
    for offset in range(-WINDOW_BEFORE, WINDOW_AFTER + 1):
        target_idx = idx + offset
        if 0 <= target_idx < len(df):
            needed_dates.add(df.iloc[target_idx]['date'])

print(f"  Unique dates to query: {len(needed_dates)}")

# Cache TM predictions
tm_cache = {}
for i, dt in enumerate(sorted(needed_dates)):
    if i % 200 == 0:
        print(f"    Processing {dt.date()} ({i}/{len(needed_dates)})...")
    try:
        result = tm_predict(dt, '1w')
        tm_cache[dt] = result.get('all_probs', {})
    except Exception:
        tm_cache[dt] = {}

print(f"  Cached {len(tm_cache)} TM predictions")

# Build day-by-day series for each event
event_series = []
for _, e in ev.iterrows():
    flip_idx = e['flip_idx']
    origin = e['origin']
    series = {}

    for offset in range(-WINDOW_BEFORE, WINDOW_AFTER + 1):
        target_idx = flip_idx + offset
        if 0 <= target_idx < len(df):
            dt = df.iloc[target_idx]['date']
            observed_shape = df.iloc[target_idx]['shape']
            probs = tm_cache.get(dt, {})

            # persistence_prob = TM's probability for the RESTING shape
            # (origin), regardless of what shape is currently observed
            # Before flip: observed == origin, so this is P(stay)
            # After flip: observed != origin, so this shows how fast
            # TM moves away from the old shape
            persistence_prob = probs.get(origin, np.nan)
            series[offset] = persistence_prob
        else:
            series[offset] = np.nan

    event_series.append(series)

ev['series'] = event_series

# Aggregate trajectories
offsets = list(range(-WINDOW_BEFORE, WINDOW_AFTER + 1))
offset_labels = [f"t{o:+d}" if o != 0 else "t0" for o in offsets]

def aggregate_trajectory(subset):
    """Compute mean persistence_prob at each offset."""
    traj = {}
    for o in offsets:
        vals = [s[o] for s in subset['series'] if o in s and not np.isnan(s[o])]
        traj[o] = np.mean(vals) if vals else np.nan
    return traj

traj_all = aggregate_trajectory(ev)
traj_sb = aggregate_trajectory(ev[ev['origin'] == '0.0'])
traj_c = aggregate_trajectory(ev[ev['origin'] == '1'])

print(f"\nPart 2.3 — Average persistence_prob trajectory:")
print(f"  {'Day':<6} {'Pooled':>8} {'SB-orig':>8} {'C-orig':>8}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
for o in offsets:
    label = f"t{o:+d}" if o != 0 else "t0"
    print(f"  {label:<6} {traj_all.get(o, np.nan):>7.1%} {traj_sb.get(o, np.nan):>7.1%} {traj_c.get(o, np.nan):>7.1%}")

# ══════════════════════════════════════════════════════════════════
# PART 3 — LEAD TIME MEASUREMENT
# ══════════════════════════════════════════════════════════════════
print(f"\nPart 3 — Lead time (threshold-crossing at 50%)...")

THRESHOLD = 0.50

def compute_lead_time(series_dict):
    """Find first day going backward from t0 where persistence_prob < 50%.
    Returns lead time in days (positive = genuine lead).
    Returns -1 if never dropped below threshold in window."""
    # Check days t-10 through t-1
    for offset in range(-1, -WINDOW_BEFORE - 1, -1):
        val = series_dict.get(offset, np.nan)
        if np.isnan(val):
            continue
        if val < THRESHOLD:
            # Found a crossing — but check if it was the first one
            # going backward. Actually we want the EARLIEST crossing.
            pass

    # Find the earliest day where persistence_prob < threshold
    first_below = None
    for offset in range(-WINDOW_BEFORE, 0):
        val = series_dict.get(offset, np.nan)
        if np.isnan(val):
            continue
        if val < THRESHOLD:
            if first_below is None:
                first_below = offset

    # Also check t0 itself
    t0_val = series_dict.get(0, np.nan)

    if first_below is not None:
        # Lead time = -first_below (e.g., first_below=-3 means 3 days lead)
        return -first_below
    elif not np.isnan(t0_val) and t0_val < THRESHOLD:
        return 0  # Only crossed on flip day
    else:
        return -1  # Never crossed

ev['lead_time'] = ev['series'].apply(compute_lead_time)

# Distribution
lt = ev['lead_time']
n_total = len(lt)
n_zero = (lt == 0).sum()
n_negative = (lt == -1).sum()
n_positive = (lt > 0).sum()
n_3plus = (lt >= 3).sum()

print(f"\n  Lead time distribution (all events, n={n_total}):")
print(f"    Mean lead time (excl never-crossed): {lt[lt >= 0].mean():.1f} days")
print(f"    Median lead time (excl never-crossed): {lt[lt >= 0].median():.1f} days")
print(f"    0 days (flip day only):    {n_zero:>4d} ({n_zero/n_total:.1%})")
print(f"    Never crossed (<0):        {n_negative:>4d} ({n_negative/n_total:.1%})")
print(f"    1-2 days lead:             {((lt >= 1) & (lt <= 2)).sum():>4d} ({((lt >= 1) & (lt <= 2)).sum()/n_total:.1%})")
print(f"    3+ days lead:              {n_3plus:>4d} ({n_3plus/n_total:.1%})")
print(f"    Lead time value counts:")
for v in sorted(lt.unique()):
    label = f"{v}d" if v >= 0 else "never"
    print(f"      {label}: {(lt == v).sum()}")

# Split by origin
for origin_label, origin_code in [("SB-origin (0.0)", "0.0"), ("C-origin (1)", "1")]:
    sub = ev[ev['origin'] == origin_code]
    lt_sub = sub['lead_time']
    n = len(lt_sub)
    if n == 0:
        continue
    print(f"\n  {origin_label} (n={n}):")
    valid = lt_sub[lt_sub >= 0]
    print(f"    Mean: {valid.mean():.1f}d, Median: {valid.median():.1f}d")
    print(f"    0 days: {(lt_sub == 0).sum()} ({(lt_sub == 0).sum()/n:.1%})")
    print(f"    Never:  {(lt_sub == -1).sum()} ({(lt_sub == -1).sum()/n:.1%})")
    print(f"    3+ days: {(lt_sub >= 3).sum()} ({(lt_sub >= 3).sum()/n:.1%})")

# Split by destination
print(f"\n  By destination shape:")
for dest in sorted(ev['destination'].unique()):
    sub = ev[ev['destination'] == dest]
    lt_sub = sub['lead_time']
    n = len(lt_sub)
    valid = lt_sub[lt_sub >= 0]
    thin_tag = " [THIN]" if n < 5 else ""
    print(f"    -> {dest} (n={n}){thin_tag}: ", end="")
    if len(valid) > 0:
        print(f"mean={valid.mean():.1f}d, median={valid.median():.1f}d, "
              f"0d={((lt_sub == 0).sum())}, never={((lt_sub == -1).sum())}, "
              f"3+d={((lt_sub >= 3).sum())}")
    else:
        print(f"all never-crossed")

# Split by origin x destination
print(f"\n  By origin x destination:")
for origin_code in ['0.0', '1']:
    for dest in sorted(ev['destination'].unique()):
        sub = ev[(ev['origin'] == origin_code) & (ev['destination'] == dest)]
        if len(sub) == 0:
            continue
        lt_sub = sub['lead_time']
        n = len(lt_sub)
        valid = lt_sub[lt_sub >= 0]
        thin_tag = " [THIN]" if n < 5 else ""
        print(f"    {origin_code}->{dest} (n={n}){thin_tag}: ", end="")
        if len(valid) > 0:
            print(f"mean={valid.mean():.1f}d, median={valid.median():.1f}d, "
                  f"0d={((lt_sub == 0).sum())}, never={((lt_sub == -1).sum())}, "
                  f"3+d={((lt_sub >= 3).sum())}")
        else:
            print(f"all never-crossed")

# ══════════════════════════════════════════════════════════════════
# PART 4 — SANITY CHECKS
# ══════════════════════════════════════════════════════════════════
print(f"\nPart 4.1 — Lead time vs days_in_resting:")

# Bucket days_in_resting
ev['resting_bucket'] = pd.cut(ev['days_in_resting'],
                               bins=[0, 5, 10, 20, 50, 999],
                               labels=['1-5d', '6-10d', '11-20d', '21-50d', '51+d'])

for bucket in ['1-5d', '6-10d', '11-20d', '21-50d', '51+d']:
    sub = ev[ev['resting_bucket'] == bucket]
    if len(sub) == 0:
        continue
    lt_sub = sub['lead_time']
    valid = lt_sub[lt_sub >= 0]
    n = len(sub)
    print(f"  Resting {bucket} (n={n}): ", end="")
    if len(valid) > 0:
        print(f"mean lead={valid.mean():.1f}d, 3+d={((lt_sub >= 3).sum())}/{n}")
    else:
        print(f"all never-crossed")

# Correlation (for events with valid lead time)
valid_ev = ev[ev['lead_time'] >= 0]
if len(valid_ev) > 5:
    corr = valid_ev['days_in_resting'].corr(valid_ev['lead_time'])
    print(f"  Correlation (days_in_resting vs lead_time, n={len(valid_ev)}): {corr:.3f}")

# Part 4.2 — Spot-check 3 events
print(f"\nPart 4.2 — Spot-check events:")

# Filter to events with actual TM data (post 2018-04-01)
has_data = ev[ev['flip_date'] >= '2018-04-01'].copy()

# Pick: best lead time (with 5+ days resting to avoid edge cases),
# one median, one never-crossed with actual data
valid_events = has_data[has_data['lead_time'] >= 0].sort_values('lead_time', ascending=False)
valid_established = valid_events[valid_events['days_in_resting'] >= 5]
never_events = has_data[has_data['lead_time'] == -1]
# For never-crossed, pick one where we can see actual probabilities (not all NaN)
never_with_data = never_events[never_events['series'].apply(
    lambda s: any(not np.isnan(s.get(o, np.nan)) for o in range(-WINDOW_BEFORE, 0)))]

spot_checks = []
if len(valid_established) >= 1:
    spot_checks.append(("CLEAREST lead time (resting 5+d)", valid_established.iloc[0]))
elif len(valid_events) >= 1:
    spot_checks.append(("CLEAREST lead time", valid_events.iloc[0]))
if len(valid_events) >= 2:
    mid_idx = len(valid_events) // 2
    spot_checks.append(("MEDIAN lead time", valid_events.iloc[mid_idx]))
if len(never_with_data) >= 1:
    spot_checks.append(("NO crossing (never dropped <50%)", never_with_data.iloc[0]))
elif len(never_events) >= 1:
    spot_checks.append(("NO crossing (never dropped <50%)", never_events.iloc[0]))

for label, row in spot_checks:
    print(f"\n  [{label}]")
    print(f"  Event: {row['origin']} -> {row['destination']} on {row['flip_date'].date()}")
    print(f"  Days in resting: {row['days_in_resting']}, Lead time: {row['lead_time']}d")
    series = row['series']
    print(f"  {'Day':<6} {'Persist%':>8}")
    for o in offsets:
        val = series.get(o, np.nan)
        marker = " <-- FLIP" if o == 0 else ""
        below = " *" if not np.isnan(val) and val < THRESHOLD and o < 0 else ""
        print(f"  {offset_labels[o + WINDOW_BEFORE]:<6} {val:>7.1%}{below}{marker}")

# ══════════════════════════════════════════════════════════════════
# PART 5 — LOGGING
# ══════════════════════════════════════════════════════════════════
print(f"\nWriting to backtest_analysis.txt...")

log_lines = []
log_lines.append("")
log_lines.append("=" * 70)
log_lines.append(f"DIAGNOSTIC — TM LEAD TIME BEFORE SB/C TRANSITIONS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log_lines.append("=" * 70)
log_lines.append("")

# Part 1
log_lines.append("--- PART 1: TRANSITION EVENTS (2017-2026) ---")
log_lines.append(f"Total events (resting shape breaks): {len(ev)}")
log_lines.append(f"  SB-origin (0.0): {(ev['origin'] == '0.0').sum()}")
log_lines.append(f"  C-origin  (1):   {(ev['origin'] == '1').sum()}")
log_lines.append("")
log_lines.append("  Origin x Destination count:")
for line in ct.to_string().split('\n'):
    log_lines.append(f"    {line}")
if thin:
    log_lines.append(f"  Thin cells (n<5): {', '.join(thin)}")
log_lines.append("")

# Part 2
log_lines.append("--- PART 2: AVERAGE PERSISTENCE PROBABILITY TRAJECTORY ---")
log_lines.append(f"  persistence_prob = TM 1w's predicted P(resting shape continues)")
log_lines.append("")
log_lines.append(f"  {'Day':<6} {'Pooled':>8} {'SB-orig':>8} {'C-orig':>8}")
log_lines.append(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
for o in offsets:
    label = f"t{o:+d}" if o != 0 else "t0"
    p = traj_all.get(o, np.nan)
    s = traj_sb.get(o, np.nan)
    c = traj_c.get(o, np.nan)
    log_lines.append(f"  {label:<6} {p:>7.1%} {s:>7.1%} {c:>7.1%}")
log_lines.append("")

# Part 3
log_lines.append("--- PART 3: LEAD TIME DISTRIBUTION ---")
log_lines.append(f"  Threshold: persistence_prob < {THRESHOLD:.0%}")
log_lines.append(f"  Lead time = first day before flip where P(stay) < {THRESHOLD:.0%}")
log_lines.append("")
valid_all = lt[lt >= 0]
log_lines.append(f"  Overall (n={n_total}):")
if len(valid_all) > 0:
    log_lines.append(f"    Mean lead time (excl never-crossed): {valid_all.mean():.1f} days")
    log_lines.append(f"    Median lead time (excl never-crossed): {valid_all.median():.1f} days")
log_lines.append(f"    0 days (flip day only):    {n_zero:>4d} ({n_zero/n_total:.1%})")
log_lines.append(f"    Never crossed:             {n_negative:>4d} ({n_negative/n_total:.1%})")
log_lines.append(f"    1-2 days lead:             {((lt >= 1) & (lt <= 2)).sum():>4d} ({((lt >= 1) & (lt <= 2)).sum()/n_total:.1%})")
log_lines.append(f"    3+ days lead:              {n_3plus:>4d} ({n_3plus/n_total:.1%})")
log_lines.append(f"    Value counts:")
for v in sorted(lt.unique()):
    vlabel = f"{v}d" if v >= 0 else "never"
    log_lines.append(f"      {vlabel}: {(lt == v).sum()}")
log_lines.append("")

for origin_label, origin_code in [("SB-origin (0.0)", "0.0"), ("C-origin (1)", "1")]:
    sub = ev[ev['origin'] == origin_code]
    lt_sub = sub['lead_time']
    n = len(lt_sub)
    valid = lt_sub[lt_sub >= 0]
    log_lines.append(f"  {origin_label} (n={n}):")
    if len(valid) > 0:
        log_lines.append(f"    Mean: {valid.mean():.1f}d, Median: {valid.median():.1f}d")
    log_lines.append(f"    0 days: {(lt_sub == 0).sum()} ({(lt_sub == 0).sum()/n:.1%})")
    log_lines.append(f"    Never:  {(lt_sub == -1).sum()} ({(lt_sub == -1).sum()/n:.1%})")
    log_lines.append(f"    3+ days: {(lt_sub >= 3).sum()} ({(lt_sub >= 3).sum()/n:.1%})")
    log_lines.append("")

log_lines.append("  By destination shape:")
for dest in sorted(ev['destination'].unique()):
    sub = ev[ev['destination'] == dest]
    lt_sub = sub['lead_time']
    n = len(lt_sub)
    valid = lt_sub[lt_sub >= 0]
    thin_tag = " [THIN]" if n < 5 else ""
    if len(valid) > 0:
        log_lines.append(f"    -> {dest} (n={n}){thin_tag}: mean={valid.mean():.1f}d, "
                         f"median={valid.median():.1f}d, 0d={((lt_sub == 0).sum())}, "
                         f"never={((lt_sub == -1).sum())}, 3+d={((lt_sub >= 3).sum())}")
    else:
        log_lines.append(f"    -> {dest} (n={n}){thin_tag}: all never-crossed")
log_lines.append("")

log_lines.append("  By origin x destination:")
for origin_code in ['0.0', '1']:
    for dest in sorted(ev['destination'].unique()):
        sub = ev[(ev['origin'] == origin_code) & (ev['destination'] == dest)]
        if len(sub) == 0:
            continue
        lt_sub = sub['lead_time']
        n = len(lt_sub)
        valid = lt_sub[lt_sub >= 0]
        thin_tag = " [THIN]" if n < 5 else ""
        if len(valid) > 0:
            log_lines.append(f"    {origin_code}->{dest} (n={n}){thin_tag}: mean={valid.mean():.1f}d, "
                             f"median={valid.median():.1f}d, 0d={((lt_sub == 0).sum())}, "
                             f"never={((lt_sub == -1).sum())}, 3+d={((lt_sub >= 3).sum())}")
        else:
            log_lines.append(f"    {origin_code}->{dest} (n={n}){thin_tag}: all never-crossed")
log_lines.append("")

# Part 4.1
log_lines.append("--- PART 4.1: LEAD TIME vs DAYS IN RESTING ---")
for bucket in ['1-5d', '6-10d', '11-20d', '21-50d', '51+d']:
    sub = ev[ev['resting_bucket'] == bucket]
    if len(sub) == 0:
        continue
    lt_sub = sub['lead_time']
    valid = lt_sub[lt_sub >= 0]
    n = len(sub)
    if len(valid) > 0:
        log_lines.append(f"  Resting {bucket} (n={n}): mean lead={valid.mean():.1f}d, "
                         f"3+d={((lt_sub >= 3).sum())}/{n}")
    else:
        log_lines.append(f"  Resting {bucket} (n={n}): all never-crossed")

if len(valid_ev) > 5:
    log_lines.append(f"  Correlation (days_in_resting vs lead_time, n={len(valid_ev)}): {corr:.3f}")
log_lines.append("")

# Part 4.2
log_lines.append("--- PART 4.2: SPOT-CHECK EVENTS ---")
for label, row in spot_checks:
    log_lines.append(f"")
    log_lines.append(f"  [{label}]")
    log_lines.append(f"  Event: {row['origin']} -> {row['destination']} on {row['flip_date'].date()}")
    log_lines.append(f"  Days in resting: {row['days_in_resting']}, Lead time: {row['lead_time']}d")
    series = row['series']
    log_lines.append(f"  {'Day':<6} {'Persist%':>8}")
    for o in offsets:
        val = series.get(o, np.nan)
        marker = " <-- FLIP" if o == 0 else ""
        below = " *" if not np.isnan(val) and val < THRESHOLD and o < 0 else ""
        log_lines.append(f"  {offset_labels[o + WINDOW_BEFORE]:<6} {val:>7.1%}{below}{marker}")

log_lines.append("")

# Write
log_path = 'research/04. backtest_analysis/backtest_analysis.txt'
with open(log_path, 'a', encoding='utf-8') as f:
    f.write('\n'.join(log_lines) + '\n')

print(f"Done — appended {len(log_lines)} lines to {log_path}")
