"""
report/html_report.py — Jinja2-based single-file HTML report generator.
"""

import base64
import datetime
import io
import json

import numpy as np
import pandas as pd

try:
    from jinja2 import Template
except ImportError:
    Template = None

_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MRBackTest Report — {{ generated_at }}</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background:#0e1117; color:#fafafa; margin:24px; }
  h1   { color:#00d4ff; border-bottom:2px solid #262730; padding-bottom:8px; }
  h2   { color:#aaddff; margin-top:32px; }
  h3   { color:#cccccc; margin-top:20px; }
  .badge       { display:inline-block; padding:4px 12px; border-radius:4px; font-weight:bold; margin:4px; }
  .badge-go    { background:#198754; color:#fff; }
  .badge-marginal { background:#fd7e14; color:#000; }
  .badge-nogo  { background:#dc3545; color:#fff; }
  .badge-pass  { background:#198754; color:#fff; }
  .badge-fail  { background:#dc3545; color:#fff; }
  .badge-warn  { background:#ffc107; color:#000; }
  table  { border-collapse:collapse; width:100%; margin:12px 0; }
  th, td { border:1px solid #3a3a4a; padding:6px 10px; text-align:right; font-size:13px; }
  th     { background:#262730; text-align:center; }
  td:first-child { text-align:left; }
  .section { background:#1a1d24; border-radius:8px; padding:16px 20px; margin:16px 0; }
  .kill-banner { background:#4a1010; border-left:4px solid #dc3545; padding:10px 16px; margin:8px 0; border-radius:4px; }
  .caveat { background:#2a1a00; border-left:4px solid #ffc107; padding:10px 16px; margin:8px 0; border-radius:4px; font-size:13px; }
  pre { background:#262730; padding:12px; border-radius:4px; font-size:12px; overflow-x:auto; }
  img { max-width:100%; border-radius:4px; margin:8px 0; }
  .metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:12px 0; }
  .metric-card { background:#262730; border-radius:6px; padding:12px; text-align:center; }
  .metric-val  { font-size:22px; font-weight:bold; color:#00d4ff; }
  .metric-lbl  { font-size:11px; color:#aaa; margin-top:4px; }
</style>
</head>
<body>

<h1>MRBackTest Report</h1>
<p style="color:#888">Generated: {{ generated_at }} &nbsp;|&nbsp; FCPO vs SOY Cross-Asset Mean Reversion</p>

<!-- ── WF1 ──────────────────────────────────────────────────────────────── -->
<h2>Walk-Forward 1 — Daily</h2>

<div class="section">
  <h3>Configuration</h3>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    {% for k, v in wf1_config.items() %}
    <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
    {% endfor %}
  </table>
</div>

{% if wf1_breaks %}
<div class="section">
  <h3>Structural Breaks Detected</h3>
  <table>
    <tr><th>Start</th><th>End</th><th>Bars</th><th>Peak |z|</th><th>Trigger</th><th>Status</th></tr>
    {% for b in wf1_breaks %}
    <tr>
      <td>{{ b.get('date_start', b.bar_start) }}</td>
      <td>{{ b.get('date_end', b.bar_end) }}</td>
      <td>{{ b.n_bars }}</td>
      <td>{{ "%.2f"|format(b.peak_z) if b.peak_z else 'N/A' }}</td>
      <td>{{ b.get('trigger_label', '') }}</td>
      <td>{{ 'Excluded' if b.enabled else 'Retained' }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

<div class="section">
  <h3>Gating Tests — Training Set</h3>
  <table>
    <tr><th>Test</th><th>Key Statistic</th><th>Result</th></tr>
    {% for row in gating_rows %}
    <tr>
      <td>{{ row.name }}</td>
      <td>{{ row.stat }}</td>
      <td><span class="badge {{ 'badge-pass' if row.passed else 'badge-fail' }}">{{ 'PASS' if row.passed else 'FAIL' }}</span></td>
    </tr>
    {% endfor %}
  </table>
  <p><strong>Gate: <span class="badge {{ 'badge-go' if gate_result == 'PASS' else ('badge-marginal' if gate_result == 'MARGINAL' else 'badge-nogo') }}">{{ gate_result }}</span></strong></p>
</div>

<div class="section">
  <h3>Validation Grid — Best Row Selected</h3>
  {{ grid_html | safe }}
  <p>Locked params: entry_z={{ locked.entry_z }} | exit_z={{ locked.exit_z }} | lookback={{ locked.lookback }}</p>
</div>

<div class="section">
  <h3>Test Set Results</h3>
  <div class="metric-grid">
    <div class="metric-card"><div class="metric-val">{{ wf1_metrics.n_trades }}</div><div class="metric-lbl">Trades</div></div>
    <div class="metric-card"><div class="metric-val">{{ "%.1f%%"|format(wf1_metrics.win_rate * 100) if wf1_metrics.win_rate else 'N/A' }}</div><div class="metric-lbl">Win Rate</div></div>
    <div class="metric-card"><div class="metric-val">{{ "%.2f"|format(wf1_metrics.sharpe) if wf1_metrics.sharpe else 'N/A' }}</div><div class="metric-lbl">Sharpe</div></div>
    <div class="metric-card"><div class="metric-val">{{ "%.0f"|format(wf1_metrics.total_net_pnl) }}</div><div class="metric-lbl">Net PnL (MYR)</div></div>
    <div class="metric-card"><div class="metric-val">{{ "%.0f"|format(wf1_metrics.max_drawdown) }}</div><div class="metric-lbl">Max Drawdown</div></div>
    <div class="metric-card"><div class="metric-val">{{ wf1_metrics.n_bars }}</div><div class="metric-lbl">Test Bars</div></div>
  </div>
  <p><strong>Verdict: <span class="badge {{ 'badge-go' if wf1_gate == 'GO' else ('badge-marginal' if wf1_gate == 'MARGINAL' else 'badge-nogo') }}">{{ wf1_gate }}</span></strong></p>
</div>

{% if kill_switches %}
<div class="section">
  <h3>Kill Switch Warnings</h3>
  {% for ks in kill_switches %}
  <div class="kill-banner">⚠ {{ ks }}</div>
  {% endfor %}
</div>
{% endif %}

{% if regime_breakdown %}
<div class="section">
  <h3>Regime Breakdown</h3>
  <table>
    <tr><th>Period</th><th>Trades</th><th>Net PnL</th><th>Win Rate</th></tr>
    {% for band, v in regime_breakdown.items() %}
    <tr>
      <td>{{ band }}</td>
      <td>{{ v.n_trades }}</td>
      <td>{{ "%.0f"|format(v.net_pnl) }}</td>
      <td>{{ "%.1f%%"|format(v.win_rate * 100) if v.win_rate else 'N/A' }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

{% if equity_img_wf1 %}
<div class="section">
  <h3>WF1 Equity Curve</h3>
  <img src="data:image/png;base64,{{ equity_img_wf1 }}" alt="WF1 Equity Curve">
</div>
{% endif %}

<div class="section">
  <h3>WF1 Trade Log (first 100)</h3>
  {{ trade_log_html_wf1 | safe }}
</div>

<!-- ── WF2 ──────────────────────────────────────────────────────────────── -->
{% if wf2_results %}
<h2>Walk-Forward 2 — 15-Minute Intraday</h2>

<div class="caveat">
  ⚠ <strong>Execution Caveat:</strong> All Sharpe and PnL figures below are pre-execution-cost estimates.
  Intraday auto-spread execution involves bid-ask slippage, leg-timing risk, and queue position uncertainty
  that are not fully captured in simulation. Treat as signal quality indicators only, not realized performance.
</div>

<div class="section">
  <h3>WF2 Grid Search Results</h3>
  {{ wf2_grid_html | safe }}
  <p>Locked params: entry_z={{ wf2_locked.entry_z }} | exit_z={{ wf2_locked.exit_z }} | lookback={{ wf2_locked.lookback }}</p>
</div>

<div class="section">
  <h3>WF2 Test Set Results</h3>
  <div class="metric-grid">
    <div class="metric-card"><div class="metric-val">{{ wf2_metrics.n_trades }}</div><div class="metric-lbl">Trades</div></div>
    <div class="metric-card"><div class="metric-val">{{ "%.1f%%"|format(wf2_metrics.win_rate * 100) if wf2_metrics.win_rate else 'N/A' }}</div><div class="metric-lbl">Win Rate</div></div>
    <div class="metric-card"><div class="metric-val">{{ "%.2f"|format(wf2_metrics.sharpe) if wf2_metrics.sharpe else 'N/A' }}</div><div class="metric-lbl">Sharpe (indicative)</div></div>
    <div class="metric-card"><div class="metric-val">{{ "%.0f"|format(wf2_metrics.total_net_pnl) }}</div><div class="metric-lbl">Net PnL (MYR)</div></div>
    <div class="metric-card"><div class="metric-val">{{ "%.0f"|format(wf2_metrics.max_drawdown) }}</div><div class="metric-lbl">Max Drawdown</div></div>
  </div>
  <p><strong>Verdict: <span class="badge {{ 'badge-go' if wf2_gate == 'GO' else ('badge-marginal' if wf2_gate == 'MARGINAL' else 'badge-nogo') }}">{{ wf2_gate }}</span></strong></p>
</div>

{% if equity_img_wf2 %}
<div class="section">
  <h3>WF2 Equity Curve</h3>
  <img src="data:image/png;base64,{{ equity_img_wf2 }}" alt="WF2 Equity Curve">
</div>
{% endif %}

{% if tt_settings %}
<div class="section">
  <h3>TT Auto Spreader Settings</h3>
  <table>
    <tr><th>Setting</th><th>Value</th></tr>
    {% for k, v in tt_settings.items() %}
    <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
    {% endfor %}
  </table>
</div>
{% endif %}

{% endif %}  {# end wf2 #}

<hr style="border-color:#3a3a4a; margin:32px 0">
<p style="color:#555; font-size:12px">Generated by MRBackTest &nbsp;|&nbsp; {{ generated_at }}</p>
</body>
</html>
"""


def _equity_to_png_b64(equity_curve: pd.Series) -> str:
    """Convert equity curve to base64 PNG string (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0e1117")
        ax.set_facecolor("#262730")
        ax.plot(equity_curve.values, color="#00d4ff", linewidth=1.2)
        ax.axhline(0, color="#888", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Bar", color="#aaa")
        ax.set_ylabel("Net PnL", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a4a")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0e1117")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def _df_to_html(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Convert DataFrame to styled HTML table."""
    if df is None or len(df) == 0:
        return "<p><em>No data</em></p>"
    sub = df.head(max_rows)
    html = sub.to_html(index=False, border=0, classes="")
    return html


def generate_report(
    wf1_results: dict,
    wf2_results: dict = None,
    config: dict = None,
) -> bytes:
    """
    Generate a single-file HTML report from WF1 (+ optional WF2) results.

    Parameters
    ----------
    wf1_results : dict from run_test_slice + gating + grid
    wf2_results : dict from run_wf2_test (optional)
    config      : dict of user-level config params to display

    Returns
    -------
    bytes: UTF-8 encoded HTML
    """
    if Template is None:
        raise ImportError("jinja2 is required for HTML report generation. pip install jinja2")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # WF1 config section
    wf1_config = config or {}

    # Gating rows
    gating = wf1_results.get("gating", {})
    passes = gating.get("passes", {})
    gating_rows = [
        {"name": "ADF/KPSS — Y is I(1)", "stat": str(gating.get("adf_y", {}).get("verdict", "")), "passed": passes.get("adf_y", False)},
        {"name": "ADF/KPSS — X is I(1)", "stat": str(gating.get("adf_x", {}).get("verdict", "")), "passed": passes.get("adf_x", False)},
        {"name": "Engle-Granger Coint",   "stat": f"p={gating.get('eg', {}).get('eg_pvalue', 'N/A')}", "passed": passes.get("eg", False)},
        {"name": "Johansen Coint",        "stat": gating.get("johansen", {}).get("verdict", ""), "passed": passes.get("johansen", False)},
        {"name": "Hurst Exponent",        "stat": f"H={gating.get('hurst', {}).get('hurst', 'N/A')}", "passed": passes.get("hurst", False)},
        {"name": "OU Half-Life",          "stat": f"{gating.get('ou', {}).get('half_life_days', 'N/A')}d", "passed": passes.get("ou", False)},
    ]
    gate_result = gating.get("gate_result", "BLOCKED")

    # Grid
    grid_df   = wf1_results.get("grid_df")
    grid_html = _df_to_html(grid_df, max_rows=36)

    locked    = wf1_results.get("locked_params", {})
    locked    = type("LP", (), locked)()  # make attribute-accessible

    # Test metrics
    wf1_test    = wf1_results.get("test_results", {})
    wf1_metrics = type("M", (), wf1_test.get("metrics", {}))()
    wf1_gate    = wf1_test.get("gate", "NO-GO")

    # Kill switches
    kill_switches    = wf1_results.get("kill_switches", [])
    regime_breakdown = wf1_results.get("regime_breakdown", {})

    # Equity images
    equity_wf1      = wf1_test.get("equity_curve")
    equity_img_wf1  = _equity_to_png_b64(equity_wf1) if equity_wf1 is not None else ""

    # Trade log
    trades_wf1      = wf1_test.get("trades", [])
    trade_df_wf1    = pd.DataFrame(trades_wf1) if trades_wf1 else pd.DataFrame()
    trade_log_html_wf1 = _df_to_html(trade_df_wf1)

    # WF2
    wf2_grid_html = ""
    wf2_metrics   = type("M", (), {})()
    wf2_locked    = type("LP", (), {})()
    wf2_gate      = ""
    equity_img_wf2 = ""
    tt_settings   = {}

    if wf2_results:
        wf2_grid_df   = wf2_results.get("grid_df")
        wf2_grid_html = _df_to_html(wf2_grid_df, max_rows=12)

        wf2_test_r    = wf2_results.get("test_results", {})
        wf2_metrics   = type("M", (), wf2_test_r.get("metrics", {}))()
        wf2_gate      = wf2_test_r.get("gate", "NO-GO")

        wf2_lp        = wf2_results.get("locked_params", {})
        wf2_locked    = type("LP", (), wf2_lp)()

        equity_wf2    = wf2_test_r.get("equity_curve")
        equity_img_wf2 = _equity_to_png_b64(equity_wf2) if equity_wf2 is not None else ""

        tt_settings   = wf2_results.get("tt_settings", {})

    # Breaks
    wf1_breaks = wf1_results.get("breaks", [])

    template = Template(_REPORT_TEMPLATE)
    html = template.render(
        generated_at=now,
        wf1_config=wf1_config,
        wf1_breaks=wf1_breaks,
        gating_rows=gating_rows,
        gate_result=gate_result,
        grid_html=grid_html,
        locked=locked,
        wf1_metrics=wf1_metrics,
        wf1_gate=wf1_gate,
        kill_switches=kill_switches,
        regime_breakdown=regime_breakdown,
        equity_img_wf1=equity_img_wf1,
        trade_log_html_wf1=trade_log_html_wf1,
        wf2_results=wf2_results,
        wf2_grid_html=wf2_grid_html,
        wf2_metrics=wf2_metrics,
        wf2_locked=wf2_locked,
        wf2_gate=wf2_gate,
        equity_img_wf2=equity_img_wf2,
        tt_settings=tt_settings,
    )

    return html.encode("utf-8")
