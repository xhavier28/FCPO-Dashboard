# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow

After completing any meaningful piece of work, commit and push to GitHub so progress is never lost.

```bash
git add app.py                # stage specific files, not git add -A
git commit -m "feat: short description of what changed"
git push origin main
```

Use conventional commit prefixes: `feat:` for new features, `fix:` for bug fixes, `refactor:` for restructuring. Keep messages concise and descriptive of the change, not the process (e.g. `feat: add dark mode to all charts` not `updated the file`).

Push after every logical unit of work — do not batch multiple unrelated changes into one commit.

## Running the App

```bash
python -m streamlit run app.py --server.port 8505 --server.headless true
```

Open at http://localhost:8505. To kill and restart cleanly on Windows:
```bash
taskkill //F //IM python.exe && python -m streamlit run app.py --server.port 8505 --server.headless true
```

## Installing Dependencies

```bash
pip install -r requirements.txt
```

Requirements: `streamlit>=1.32.0`, `pandas>=2.1.0`, `plotly>=5.20.0`

## Architecture

Single-file app (`app.py`) with two Streamlit tabs built on Plotly + Pandas.

### Data Sources

- **Spot price**: `Raw Data/MYX_DLY_FCPO1!, D_59dbd.csv` — Unix timestamp + OHLCV, filtered to 2023–2026.
- **Term structure contracts**: `Raw Data/Term Structure/{year}/FCPO {Mmm}{yy}_Daily.csv` — one CSV per contract month. Auto-scanned by `load_contracts()`. Files may be missing for some months.

### Key Design Decisions

**X-axis is day-of-year (doy), not calendar date.** All charts use doy (1–366) on the x-axis so multiple years overlay on the same axis. Month labels are mapped via `TICKVALS`/`TICKTEXT` constants.

**Front-month rolling rule**: `front_month(date)` rolls to the next month on the 16th of each month (day ≤ 15 → current month, day ≥ 16 → next month). This drives which contract is "Current" in the term table.

**Week label format**: `"W1 Jan 2025"` — week number is `(day-1)//7` clamped at W4 for days 29–31. Used as the primary key in `build_term_table`.

### Tab 1 — Year-over-Year

- Year selector is a `st.multiselect` inside the tab (not sidebar).
- 5-day centred rolling mean smooths the close price.
- Most-recent year = full opacity; older years dimmed at alpha 0.35 via `hex_to_rgba()`.
- `YEAR_COLORS`: 2023=#1f77b4, 2024=#ff7f0e, 2025=#2ca02c, 2026=#d62728

### Tab 2 — Term Structure

- `build_term_table(contracts)` → DataFrame of week × tenor (Current, +1M … +11M) averaged daily closes.
- `build_combined_chart(df, year, df_term)` → 2-row shared-x subplot:
  - **Row 1 (70%)**: smoothed spot price on actual MYR scale.
  - **Row 2 (30%)**: 48 normalised forward-curve slivers, one per week, positioned in their doy window (6-day wide, 1-day gap). Y normalised [0,1] — shape only, no absolute scale.
- Colorscale: **Turbo [0.05, 0.95]** — vivid cyan → red progression across weeks Jan→Dec.
- Shared x-axis uses `make_subplots(shared_xaxes=True)`: `xaxis` is the hidden top axis (row 1); `xaxis2` is the visible bottom axis (row 2) — **all tick, range, and pan settings must go on `xaxis2`**.
- Pan is locked to 4-month window (`range=[1, 122]`) with `dragmode='pan'` and `minallowed=1, maxallowed=366`.

### Dark Mode

Charts use dark theme constants defined at module level:
```python
DARK_BG   = "#0e1117"
DARK_PLOT = "#262730"
DARK_GRID = "#3a3a4a"
DARK_TEXT = "#fafafa"
```
Streamlit theme is set via `.streamlit/config.toml` (`base = "dark"`).
