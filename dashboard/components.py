"""Reusable UI components. `*_html` builders are pure (unit-testable); `render_*` write them to Streamlit.

KPI cards carry data-kpi / data-value attributes so tests can read the displayed values.
All text that can come from data is HTML-escaped.
"""
from __future__ import annotations

from html import escape

import streamlit as st

from theme import STATE_SLUG, STATUS, STATUS_ICON


def _md(html: str):
    st.markdown(html, unsafe_allow_html=True)


# ---------------- small pieces ----------------
def status_badge_html(status: str | None, large: bool = False, label: str | None = None) -> str:
    """Health-state colour + icon + text label (never colour alone)."""
    if status not in STATUS:
        return f"<span class='badge{' lg' if large else ''}' style='color:var(--muted)'>N/A</span>"
    return (f"<span class='badge{' lg' if large else ''}' style='color:{STATUS[status]}'>"
            f"{STATUS_ICON[status]} {escape(label or status)}</span>")


def state_badge_label(state: str, latched: bool) -> str:
    return "Near Failure — latched" if state == "Near Failure" and latched else state


def tone_for(state: str | None) -> str | None:
    return STATE_SLUG.get(state, state)


def help_html(text: str, tip: str) -> str:
    return f"<span class='help' title='{escape(tip, quote=True)}'>{escape(text)}</span>"


def section_header_html(title: str, subtitle: str | None = None) -> str:
    sub = f"<span class='sec-sub'>{escape(subtitle)}</span>" if subtitle else ""
    return f"<div class='sec'><span class='sec-title'>{escape(title)}</span>{sub}</div>"


def render_section_header(title: str, subtitle: str | None = None):
    _md(section_header_html(title, subtitle))


def page_header_html(title: str, subtitle: str, meta: dict[str, str] | None = None) -> str:
    items = "".join(f"<span>{escape(k)}<b>{escape(v)}</b></span>" for k, v in (meta or {}).items())
    return (f"<div class='page-head'><div><div class='page-title'>{escape(title)}</div>"
            f"<div class='page-sub'>{escape(subtitle)}</div></div><div class='head-meta'>{items}</div></div>")


def render_page_header(title: str, subtitle: str, meta: dict[str, str] | None = None):
    _md(page_header_html(title, subtitle, meta))


# ---------------- cards ----------------
def metric_card_html(label: str, value: str, unit: str | None = None, sub: str | None = None,
                     tone: str | None = None, dominant: bool = False, simulated: bool = False,
                     value_html: str | None = None, sub_html: str | None = None) -> str:
    """tone: 'info', a health-state name (see theme.STATUS) or None."""
    tone = tone_for(tone)
    cls = "card kpi" + (" dominant" if dominant else "") + (f" tone-{tone}" if tone else "") + (" sim" if simulated else "")
    unit_html = f"<span class='kpi-unit'>{escape(unit)}</span>" if unit else ""
    body = value_html if value_html is not None else f"{escape(value)}{unit_html}"
    sub_part = sub_html if sub_html is not None else (escape(sub) if sub else "")
    return (f"<div class='{cls}' data-kpi='{escape(label, quote=True)}' data-value='{escape(value, quote=True)}'>"
            f"<div class='kpi-label'>{escape(label)}</div><div class='kpi-value'>{body}</div>"
            f"<div class='kpi-sub'>{sub_part}</div></div>")


def render_metric_card(*args, **kwargs):
    _md(metric_card_html(*args, **kwargs))


def prediction_card_html(rul: float, status: str, uncertainty: float, model_label: str, simulated: bool = False,
                         latched: bool = False, latch_cycle: int | None = None, above_boundary: bool = False,
                         boundary: float = 45.0) -> str:
    """Current prediction ALWAYS shown; the health state (possibly latched) is shown next to it."""
    sim = "<div class='pred-row' style='color:var(--warn)'><b style='color:var(--warn)'>SIMULATED INPUT</b></div>" if simulated else ""
    latch = (f"<div class='pred-row'>Latched at cycle <b>{latch_cycle}</b></div>" if latched and latch_cycle is not None else "")
    above = (f"<div class='pred-row' style='color:var(--serious)'><b style='color:var(--serious)'>Latched Near Failure — "
             f"current prediction is above boundary ({boundary:.0f})</b></div>" if above_boundary else "")
    return (f"<div class='card kpi tone-{tone_for(status)}{' sim' if simulated else ''}' data-kpi='Current RUL' "
            f"data-value='{rul:.0f}' data-state='{escape(status, quote=True)}'>"
            f"<div class='kpi-label'>Current predicted remaining useful life</div>"
            f"<div class='pred'><div><div class='pred-num'>{rul:.0f}</div><div class='pred-unit'>CYCLES REMAINING</div></div>"
            f"<div class='pred-side'>{status_badge_html(status, large=True, label=state_badge_label(status, latched))}"
            f"<div class='pred-row'>Uncertainty <b>±{uncertainty:.1f}</b> cycles</div>{latch}{above}"
            f"<div class='pred-row'>Model <b>{escape(model_label)}</b></div>{sim}</div></div></div>")


def render_prediction_card(*args, **kwargs):
    _md(prediction_card_html(*args, **kwargs))


# ---------------- uncertainty + health timeline ----------------
def uncertainty_range_html(pred: float, unc: float, axis_max: float, boundary: float | None = None) -> str:
    """Lower bound - prediction - upper bound on a 0..axis_max cycle axis (bounds clipped at 0); optional dashed
    marker at the fleet-level Near-Failure boundary."""
    lo, hi = max(pred - unc, 0.0), pred + unc
    axis_max = max(axis_max, hi) or 1.0
    pos = lambda v: 100.0 * v / axis_max  # noqa: E731
    bmark = (f"<div class='range-boundary' style='left:{pos(boundary):.2f}%' title='Near-Failure boundary "
             f"{boundary:.0f} cycles (fleet-level)'></div>" if boundary is not None else "")
    return (f"<div class='range' title='Prediction ± 1 standard deviation of the MC-dropout samples'>"
            f"<div class='range-axis'></div>"
            f"<div class='range-band' style='left:{pos(lo):.2f}%;width:{pos(hi) - pos(lo):.2f}%'></div>{bmark}"
            f"<div class='range-mark' style='left:{pos(pred):.2f}%'></div></div>"
            f"<div class='range-labels'><span>Lower <b>{lo:.1f}</b></span><span>Prediction <b>{pred:.1f}</b> cycles</span>"
            f"<span>Upper <b>{hi:.1f}</b></span></div>")


def state_strip_html(states: list[str], cycles: list[int]) -> str:
    """Actual health-state history of one engine: one coloured segment per run of identical states."""
    if not states:
        return "<div class='kpi-sub'>No state history.</div>"
    runs, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            runs.append((states[start], start, i - 1)); start = i
    n = len(states)
    segs = "".join(
        f"<div class='strip-seg' style='width:{100.0 * (b - a + 1) / n:.3f}%;background:{STATUS.get(s, '#555')}' "
        f"title='{escape(s, quote=True)}: cycles {cycles[a]}–{cycles[b]}'>{STATUS_ICON.get(s, '') if (b - a + 1) / n > 0.04 else ''}</div>"
        for s, a, b in runs)
    legend = "".join(f"<span><i style='background:{c}'></i>{escape(s)}</span>" for s, c in STATUS.items() if s in states)
    return (f"<div class='strip'>{segs}</div><div class='strip-scale'><span>cycle {cycles[0]}</span>"
            f"<span>cycle {cycles[-1]} (now)</span></div><div class='legend'>{legend}</div>")


def evidence_html(r: dict) -> str:
    """Model evidence and sensor evidence side by side (from the API's health fields)."""
    R = r.get("near_failure_boundary") or 45
    si, sb = r.get("sensor_index"), r.get("sensor_boundary")
    run, need = r.get("sensor_consecutive_cycles_above"), r.get("sensor_persistence_required")
    sens_rows = (f"<div class='evid-row'>Sensor index <b>{si:.2f}</b> vs boundary <b>{sb:.3f}</b></div>" if si is not None
                 else "<div class='evid-row'>Sensor index not yet available (baseline forming)</div>")
    if r.get("degrading_since_cycle") is not None:
        sens_rows += (f"<div class='evid-row'>Degrading since cycle <b>{r.get('degrading_since_cycle_number')}</b> "
                      f"(latched)</div>")
    elif run is not None and need:
        sens_rows += f"<div class='evid-row'>Consecutive cycles above boundary: <b>{run}</b> / {need}</div>"
    model_rows = (f"<div class='evid-row'>Current prediction <b>{r['predicted_rul']:.1f} ± {r['uncertainty']:.1f}</b> "
                  f"vs Near-Failure boundary <b>{R:.0f}</b> (fleet-level)</div>")
    if r.get("state_latched"):
        model_rows += f"<div class='evid-row'>Near Failure latched at cycle <b>{r.get('latch_cycle_number')}</b></div>"
    return (f"<div class='evid'><div class='card'><div class='evid-title'>Model evidence</div>"
            f"<div class='evid-value'>{escape(str(r.get('model_evidence')))}</div>{model_rows}</div>"
            f"<div class='card'><div class='evid-title'>Sensor evidence</div>"
            f"<div class='evid-value'>{escape(str(r.get('sensor_evidence')))}</div>{sens_rows}</div></div>")


STATE_EXPLANATION = {
    "Insufficient History": "INSUFFICIENT HISTORY — Full 30-cycle observation window not yet available.",
    "Healthy / Baseline-consistent": "The sensor degradation index is consistent with this engine's early-life "
                                     "baseline, and the current prediction is above the Near-Failure boundary.",
    "Degrading": "Sensor degradation index has remained above the learned boundary for 20 consecutive cycles.",
    "Boundary / Uncertain": "The current prediction is close to the fleet-level Near-Failure boundary (45 cycles) "
                            "considering model uncertainty.",
    "Near Failure": "The predicted RUL plus its uncertainty reached the fleet-level Near-Failure boundary (45 cycles); "
                    "the state is latched. The current prediction continues to be shown.",
}


# ---------------- sensors ----------------
def trend_arrow(drift: float, stable_below: float) -> tuple[str, str]:
    if abs(drift) < stable_below:
        return "→", "Stable"
    return ("↑", "Increasing") if drift > 0 else ("↓", "Decreasing")


def sensor_list_html(trends: list[dict], stable_below: float = 0.5, names: dict | None = None) -> str:
    """Rows from the API's sensor_trends (drift of recent level vs the engine's early baseline, training-σ units)."""
    rows = []
    for t in trends:
        arrow, word = trend_arrow(float(t["drift_std_units"]), stable_below)
        name = (names or {}).get(t["sensor"], t["sensor"])
        rows.append(f"<div class='sensor-row'><span class='sensor-name'>{escape(str(name))}</span>"
                    f"<span class='sensor-trend'><span class='arrow'>{arrow}</span>{word} "
                    f"({float(t['drift_std_units']):+.1f}σ)</span></div>")
    return "<div class='card'>" + "".join(rows) + "</div>" if rows else "<div class='card kpi-sub'>No sensor trend data.</div>"


def render_sensor_list(trends, stable_below: float = 0.5, names: dict | None = None):
    _md(sensor_list_html(trends, stable_below, names))


def banner_html(label: str, text: str) -> str:
    return f"<div class='banner'><b>{escape(label)}</b> &nbsp;{escape(text)}</div>"


def note_html(text: str) -> str:
    return f"<div class='note'>{escape(text)}</div>"
