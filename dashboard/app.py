"""Engine Health Monitor - Streamlit dashboard; an HTTP client of the FastAPI backend (it runs no model code).

Run the backend first:   .\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000
Then the dashboard:      .\\.venv\\Scripts\\python.exe -m streamlit run dashboard/app.py   (or .\\start_demo.ps1)
Backend address:         environment variable RUL_API_URL (default http://127.0.0.1:8000)

Live predictions come from the LSTM baseline served by the API. DATF-Net is a research model evaluated offline;
its results are read from outputs/experiments/summary.json (a local file, not served by the API).
pandas-free: tables are rendered as HTML (table_view.py) because pandas' DLLs are blocked by Windows Application
Control on this machine and st.dataframe/st.table use pandas.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import ApiClient, ApiError, api_url_from_env  # noqa: E402
from components import (STATE_EXPLANATION, banner_html, evidence_html, help_html, note_html,  # noqa: E402
                        render_metric_card, render_page_header, render_prediction_card, render_section_header,
                        render_sensor_list, state_badge_label, state_strip_html, status_badge_html, trend_arrow,
                        uncertainty_range_html)
from table_view import TABLE_CSS, html_table, records_to_columns  # noqa: E402
from theme import CSS, INFO, NEUTRAL_LINE, OFFLINE, ONLINE, SERIES, STATUS, style_fig  # noqa: E402
from adaptation_ui import (render_mode2_upload, render_mode2_predict, render_mode3_mapping,  # noqa: E402
                           render_mode3_profiling, render_mode3_training, render_mode3_registry,
                           render_mode3_predict)

API_URL = api_url_from_env()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SUMMARY = PROJECT_ROOT / "outputs" / "experiments" / "summary.json"
PAGES = ["Overview", "Engine Monitor", "RUL Analysis", "Simulation", "Model Performance", "About"]
MODES = {"real": "Real dataset (FD001 test)", "sim": "Simulation"}
SIM_STAGES = ["normal", "warning", "critical"]
# Simulation scenarios are defined by a scenario RUL band only (API stage keys unchanged); they are not health states.
SIM_STAGE_LABELS = {"normal": "Early-degradation scenario (RUL 80–110)", "warning": "Late-degradation scenario (RUL 33–42)",
                    "critical": "End-of-life scenario (RUL 5–16)"}
METHODOLOGY_TEXT = (
    "Health states are derived from FD001 training-data analysis. Degrading is identified using an engine-relative "
    "sensor degradation index. Near Failure uses a fleet-level RUL acceleration boundary of 45 cycles. The 45-cycle "
    "boundary is not an individual-engine failure threshold or safety limit.")
PERSISTENCE_TEXT = "The 20-cycle persistence requirement is an implementation policy chosen from validation analysis."
DEFAULT_SIM = {"stage": "normal", "seed": 0, "n_cycles": 180}  # sample shown before the first "Run simulation"
MODEL_LABEL = "LSTM baseline"
STABLE_DRIFT = 0.5  # |drift| below this (training-σ units) is shown as "Stable" in the sensor list
# Channel descriptions per the C-MAPSS documentation (Saxena et al., 2008)
SENSOR_INFO = {
    "sensor_2": ("T24", "LPC outlet temperature", "°R"), "sensor_3": ("T30", "HPC outlet temperature", "°R"),
    "sensor_4": ("T50", "LPT outlet temperature", "°R"), "sensor_6": ("P15", "Bypass-duct pressure", "psia"),
    "sensor_7": ("P30", "HPC outlet pressure", "psia"), "sensor_8": ("Nf", "Physical fan speed", "rpm"),
    "sensor_9": ("Nc", "Physical core speed", "rpm"), "sensor_11": ("Ps30", "HPC outlet static pressure", "psia"),
    "sensor_12": ("phi", "Fuel flow / Ps30", "pps/psi"), "sensor_13": ("NRf", "Corrected fan speed", "rpm"),
    "sensor_14": ("NRc", "Corrected core speed", "rpm"), "sensor_15": ("BPR", "Bypass ratio", ""),
    "sensor_17": ("htBleed", "Bleed enthalpy", ""), "sensor_20": ("W31", "HPT coolant bleed", "lbm/s"),
    "sensor_21": ("W32", "LPT coolant bleed", "lbm/s"),
}
RESEARCH_MODELS = [("lstm_baseline", "LSTM"), ("lstm_info_matched", "LSTM + baseline control"),
                   ("datf_full", "DATF-Net"), ("abl_no_baseline", "DATF-Net without baseline")]

st.set_page_config(page_title="Engine Health Monitor", page_icon="⚙️", layout="wide")
st.markdown(CSS + TABLE_CSS, unsafe_allow_html=True)


# ---------------- API access (fresh client with latest methods) ----------------
def get_client(url: str) -> ApiClient:
    return ApiClient(url)


api = ApiClient(API_URL)


@st.cache_data(ttl=300, show_spinner=False)
def cached_engines(url):
    return api.engines()


@st.cache_data(ttl=300, show_spinner="Loading engine data…")
def cached_engine(url, engine_id):
    return api.engine(engine_id)


@st.cache_data(ttl=300, show_spinner="Running model inference on the backend…")
def cached_real_prediction(url, engine_id):
    return api.predict(engine_id=engine_id), api.trajectory(engine_id=engine_id), time.strftime("%H:%M:%S")


@st.cache_data(ttl=300, show_spinner="Generating simulated engine…")
def cached_simulation(url, stage, seed, n_cycles):
    return api.simulate(seed, n_cycles, stage)


@st.cache_data(ttl=300, show_spinner="Running model inference on the simulated engine…")
def cached_sim_prediction(url, stage, seed, n_cycles):
    readings = cached_simulation(url, stage, seed, n_cycles)["readings"]
    return api.predict(readings=readings), api.trajectory(readings=readings), time.strftime("%H:%M:%S")


@st.cache_data(ttl=60, show_spinner="Loading evaluation results…")
def cached_performance(url):
    return api.metrics(), api.training_history(), api.test_predictions()


@st.cache_data(ttl=60, show_spinner=False)
def load_experiment_summary(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def readings_to_columns(readings: list[dict]) -> dict:
    """[{col: value}] from the API -> {col: NumPy array}, ordered by cycle."""
    cols = {c: np.array([r[c] for r in readings]) for c in readings[0]}
    order = np.argsort(cols["cycle"], kind="stable")
    return {c: v[order] for c, v in cols.items()}


def fail(e: ApiError, what: str):
    (st.error if e.kind in ("unavailable", "timeout", "invalid") else st.warning)(f"**{what}:** {e}")


def md(html: str, container=st):
    container.markdown(html, unsafe_allow_html=True)


# ---------------- shared state ----------------
def sync(src: str, dst: str):
    """Copy a widget's value into a non-widget key so it survives page switches (widget state is dropped when a
    widget is not rendered)."""
    st.session_state[dst] = st.session_state[src]


def engine_selector(container, engine_ids: list[int], key: str, disabled: bool = False):
    ss = st.session_state
    container.selectbox("Engine", engine_ids, index=engine_ids.index(ss.engine_id), key=key,
                        format_func=lambda i: f"Engine {i}", on_change=sync, args=(key, "engine_id"),
                        disabled=disabled, help="Engines of the NASA C-MAPSS FD001 test set")


def load_context() -> dict | None:
    """Data and prediction for the current engine: the selected real engine or the loaded simulated sample."""
    ss = st.session_state
    ctx = {"mode": ss.data_mode, "real": ss.data_mode == "real", "pred_error": None, "result": None, "traj": None,
           "fetched_at": None}
    try:
        if ctx["real"]:
            info = cached_engine(API_URL, ss.engine_id)
            ctx.update(eid=ss.engine_id, label=f"Engine {ss.engine_id}", df=readings_to_columns(info["recent_readings"]),
                       actual=info["true_rul_at_last_cycle"])
        else:
            s = ss.sim_loaded
            sim = cached_simulation(API_URL, s["stage"], s["seed"], s["n_cycles"])
            ctx.update(eid="SIM", label="Simulated engine", df=readings_to_columns(sim["readings"]), sim=sim,
                       actual=None)
    except ApiError as e:
        fail(e, "Could not load engine data from the backend")
        st.stop()
    try:
        if ctx["real"]:
            ctx["result"], ctx["traj"], ctx["fetched_at"] = cached_real_prediction(API_URL, ss.engine_id)
        else:
            s = ss.sim_loaded
            ctx["result"], ctx["traj"], ctx["fetched_at"] = cached_sim_prediction(API_URL, s["stage"], s["seed"], s["n_cycles"])
    except ApiError as e:
        ctx["pred_error"] = e
    return ctx


# ---------------- derived displays (only from API data) ----------------
def degradation_trend(traj: dict, clip: float, window: int = 30) -> tuple[str, str, str]:
    """(arrow, label, detail) from the slope of the predicted RUL over the last `window` cycles."""
    y = np.asarray(traj["predicted_rul"], dtype=float)[-window:]
    x = np.asarray(traj["cycles"], dtype=float)[-window:]
    if len(y) < 5:
        return "–", "Not enough history", "Fewer than 5 predicted cycles available."
    slope = float(np.polyfit(x, y, 1)[0])
    detail = f"Predicted RUL changed by {slope:+.2f} cycles per cycle over the last {len(y)} cycles."
    if y[-1] >= 0.95 * clip:
        return "→", "Stable (healthy plateau)", detail + f" Predictions are near the {clip:.0f}-cycle cap."
    if slope <= -0.3:
        return "↑", "Increasing", detail
    if slope >= 0.3:
        return "↓", "Decreasing", detail
    return "→", "Stable", detail


def sensor_label(c: str) -> str:
    return f"S{c.split('_')[1]}" if c.startswith("sensor_") else c


def sensor_display(c: str) -> str:
    """e.g. 'S11 · Ps30' (short code + C-MAPSS symbol)."""
    return f"{sensor_label(c)} · {SENSOR_INFO[c][0]}" if c in SENSOR_INFO else c


SENSOR_NAMES = {c: sensor_display(c) for c in SENSOR_INFO}


def trajectory_figure(ctx: dict, thr: dict, height: int) -> go.Figure:
    traj, result = ctx["traj"], ctx["result"]
    cycles = np.asarray(traj["cycles"])
    pred_series = np.asarray(traj["predicted_rul"])
    clip = traj["rul_clip"]
    now, rul = int(cycles.max()), float(result["predicted_rul"])
    fig = go.Figure()
    fig.add_scatter(x=cycles, y=pred_series, name=f"Predicted RUL ({MODEL_LABEL})", mode="lines",
                    line=dict(color=INFO, width=2), hovertemplate="Cycle %{x}<br>Predicted RUL %{y:.1f}<extra></extra>")
    if ctx["real"]:
        ref = np.minimum(ctx["actual"] + (now - cycles), clip)
        fig.add_scatter(x=cycles, y=ref, name=f"Actual RUL (derived from test label, capped at {clip})", mode="lines",
                        line=dict(color=NEUTRAL_LINE, width=2, dash="dash"),
                        hovertemplate="Cycle %{x}<br>Actual RUL %{y:.0f}<extra></extra>")
    else:
        ref = np.minimum(np.asarray(ctx["sim"]["scenario_rul"]), clip)
        fig.add_scatter(x=cycles, y=ref, name=f"Scenario RUL (simulated, capped at {clip})", mode="lines",
                        line=dict(color=NEUTRAL_LINE, width=2, dash="dash"),
                        hovertemplate="Cycle %{x}<br>Scenario RUL %{y:.0f}<extra></extra>")
    fig.add_scatter(x=[now, now + rul], y=[rul, 0], name="Projection to failure (−1 RUL per cycle)", mode="lines",
                    line=dict(color=INFO, width=2, dash="dot"), hoverinfo="skip")
    fig.add_scatter(x=[now], y=[rul], name="Current prediction", mode="markers",
                    marker=dict(color=INFO, size=11, line=dict(color="#f4f4f1", width=2)),
                    hovertemplate="Current cycle %{x}<br>Predicted RUL %{y:.1f}<extra></extra>")
    fig.add_scatter(x=[now + rul], y=[0], name="Projected failure point (RUL = 0)", mode="markers",
                    marker=dict(color=STATUS["Near Failure"], size=11, symbol="x", line=dict(color="#f4f4f1", width=1)),
                    hovertemplate="Projected failure at cycle %{x:.0f}<extra></extra>")
    add_boundary_line(fig, thr)
    if traj.get("latch_cycle") is not None:
        lc = int(cycles[traj["latch_cycle"]])
        fig.add_vline(x=lc, line=dict(color=STATUS["Near Failure"], width=1, dash="dash"),
                      annotation_text=f"Near Failure latched (cycle {lc})", annotation_position="bottom left",
                      annotation_font=dict(color=STATUS["Near Failure"], size=10))
    fig.add_vline(x=now, line=dict(color="#8d8c85", width=1, dash="dot"),
                  annotation_text=f"Current cycle {now}", annotation_position="top right",
                  annotation_font=dict(color="#c3c2b7", size=10))
    style_fig(fig, height, "Cycle", "Remaining useful life (cycles)")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(hovermode="closest")
    return fig


def add_boundary_line(fig, thr):
    R = thr["near_failure_boundary"]
    fig.add_hline(y=R, line=dict(color=STATUS["Near Failure"], width=1, dash="dot"), opacity=0.8,
                  annotation_text=f"Near-Failure boundary {R:.0f} (fleet-level, not an engine-specific limit)",
                  annotation_position="top left", annotation_font=dict(color=STATUS["Near Failure"], size=10))


def health_flags(r: dict):
    """State explanation, latched-above-boundary notice and model-sensor disagreement (never hidden)."""
    state = r.get("health_state")
    if state is None:
        st.warning("Health state unavailable: the backend has no health-state parameters "
                   "(run `python -m src.health_state_calibrate`).")
        return
    text = STATE_EXPLANATION.get(state, "")
    if state == "Insufficient History":
        md(banner_html("INSUFFICIENT HISTORY", "Full 30-cycle observation window not yet available."))
    else:
        md(note_html(text))
    if r.get("current_prediction_above_boundary"):
        md(f"<div class='flag'><b>Latched Near Failure — current prediction is above boundary.</b> The current "
           f"prediction is {r['predicted_rul']:.1f} ± {r['uncertainty']:.1f} cycles (boundary "
           f"{r['near_failure_boundary']:.0f}); Near Failure stays latched since cycle {r['latch_cycle_number']}.</div>")
    d = r.get("disagreement") or {}
    if d.get("present"):
        md(f"<div class='flag'><b>Model–sensor {'conflict' if d.get('kind') == 'conflict' else 'difference'}:</b> "
           f"{d.get('message', '')}</div>")


def methodology_expander():
    with st.expander("How health states are determined"):
        st.markdown(f"{METHODOLOGY_TEXT}\n\n{PERSISTENCE_TEXT}\n\n"
                    f"- **Insufficient History:** cycles 0–{thr['history_gate_cycle'] - 1} (the first complete "
                    f"30-cycle window is at cycle {thr['history_gate_cycle']}).\n"
                    f"- **Degrading:** sensor degradation index > {thr['sensor_boundary']} for "
                    f"{thr['sensor_persistence_cycles']} consecutive cycles (relative to the engine's first "
                    f"{thr['baseline_cycles']} cycles).\n"
                    f"- **Boundary / Uncertain:** the current prediction is within its uncertainty of "
                    f"{thr['near_failure_boundary']:.0f} cycles.\n"
                    f"- **Near Failure:** first cycle ≥ {thr['history_gate_cycle']} with predicted RUL + uncertainty ≤ "
                    f"{thr['near_failure_boundary']:.0f}; latched afterwards, while the current prediction stays visible.\n"
                    f"- Model evidence and sensor evidence are shown separately; differences are flagged, not resolved.")


def status_meta(ctx) -> dict:
    return {"System": "ONLINE", "Selected": ctx["label"], "Data": MODES[ctx["mode"]],
            "Last prediction": ctx["fetched_at"] or "–"}


def simulated_banner(ctx):
    if not ctx["real"]:
        md(banner_html("SIMULATED DATA", f"{ctx['sim']['label']}. Predictions come from the real trained model "
                       "applied to synthetic input; they are not evaluation results and are never mixed with real "
                       "metrics. Change or reset the sample on the Simulation page."))


def prediction_unavailable(ctx):
    if ctx["pred_error"] is not None:
        fail(ctx["pred_error"], "Prediction unavailable")
    else:
        st.info("No prediction available.")
    st.caption("If the model is not trained yet, run `python -m src.train` and `python -m src.evaluate` on the "
               "backend machine, then restart the backend.")


# ================= sidebar + backend connection =================
md("<div class='brand'><div class='brand-title'>INDUSTRIAL AI PLATFORM</div>"
   "<div class='brand-sub'>Multi-Industry Predictive Maintenance</div></div>", st.sidebar)

OPERATING_MODES = [
    "✈️ Mode 1: NASA C-MAPSS Demo",
    "📁 Mode 2: External Dataset Upload",
    "🏭 Mode 3: Industry Adaptation & Training",
]
selected_mode = st.sidebar.selectbox("Platform Operation Mode", OPERATING_MODES, index=0, key="platform_mode")

page = "Overview"
page_m2 = "Dataset Upload & Validation"
page_m3 = "Industry & Sensor Mapping"

if selected_mode == "✈️ Mode 1: NASA C-MAPSS Demo":
    page = st.sidebar.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
elif selected_mode == "📁 Mode 2: External Dataset Upload":
    page_m2 = st.sidebar.radio("Mode 2 Navigation", ["Dataset Upload & Validation", "Model Safety Shield & Predict"], key="nav_mode2")
else:
    page_m3 = st.sidebar.radio("Mode 3 Navigation", ["Industry & Sensor Mapping", "Dataset Profiling & Heatmap", "Model Adaptation Training", "Model Registry & Comparison", "Universal RUL Prediction"], key="nav_mode3")

try:
    health = api.health()
except ApiError as e:
    md(f"<div class='sys-status'><span class='dot' style='background:{OFFLINE}'></span>"
       f"<b>Backend offline</b><br>{API_URL}</div>", st.sidebar)
    render_page_header("ENGINE HEALTH MONITOR", "Predictive Maintenance Center", {"System": "OFFLINE"})
    st.error(f"**Backend unavailable.** {e}")
    st.info("The dashboard gets all data and predictions from the FastAPI backend. Start the backend, then reload "
            "this page. To use a different address, set the `RUL_API_URL` environment variable before starting "
            "Streamlit.")
    st.stop()

thr, model, has_data = health["thresholds"], health.get("model"), health["dataset_available"]
if not thr:
    st.error("The backend has no health-state parameters. Run `python -m src.health_state_calibrate` and restart it.")
    st.stop()
ss = st.session_state
ss.setdefault("data_mode", "real" if has_data else "sim")
ss.setdefault("sim_loaded", dict(DEFAULT_SIM))
engine_ids: list[int] = []
if has_data:
    try:
        engine_ids = cached_engines(API_URL)["engine_ids"]
    except ApiError as e:
        fail(e, "Could not load the engine list")
        st.stop()
    ss.setdefault("engine_id", engine_ids[0])
if not has_data:
    ss.data_mode = "sim"

model_line = (f"Model <b>{MODEL_LABEL}</b> · {model['device']}" if model
              else f"Model <b>not loaded</b>")
context_line = (f"Engine <b>{ss.engine_id}</b> · {MODES['real']}" if ss.data_mode == "real"
                else f"<b>Simulated engine</b> · {SIM_STAGE_LABELS[ss.sim_loaded['stage']]}")
md(f"<div class='ctx'>{context_line}</div>"
   f"<div class='sys-status'><span class='dot' style='background:{ONLINE}'></span><b>System Online</b><br>"
   f"Backend {API_URL.replace('http://', '')}<br>{model_line}<br>"
   f"Dataset {'available' if has_data else 'missing'}<br>"
   f"Near-Failure boundary {thr['near_failure_boundary']:.0f} (fleet-level) · sensor boundary "
   f"{thr['sensor_boundary']} / {thr['sensor_persistence_cycles']} cycles</div>", st.sidebar)
st.sidebar.caption("Health states are data-driven from FD001 training data. The 45-cycle boundary is fleet-level, "
                   "not an individual-engine failure threshold or safety limit.")


# ================= pages =================
def page_overview():
    ctx = load_context()
    render_page_header("ENGINE HEALTH MONITOR", "Predictive Maintenance Center", status_meta(ctx))
    simulated_banner(ctx)
    r = ctx["result"]
    if r is None:
        prediction_unavailable(ctx)
        return
    status = r["health_state"]
    sub = (f"Ground truth (test label): {ctx['actual']} cycles" if ctx["real"]
           else f"Scenario RUL (simulated): {ctx['sim']['scenario_rul_at_last_cycle']} cycles")
    c = st.columns([1.6, 1, 1, 1])
    with c[0]:
        render_metric_card("Remaining useful life", f"{r['predicted_rul']:.0f}", "cycles", sub, tone="info",
                           dominant=True, simulated=not ctx["real"])
    with c[1]:
        render_metric_card("Health state", status,
                           value_html=status_badge_html(status, label=state_badge_label(status, r["state_latched"])),
                           sub=(f"Latched at cycle {r['latch_cycle_number']}" if r["state_latched"] else
                                f"Sensor: {r['sensor_evidence']} · Model: {r['model_evidence']}"),
                           tone=status)
    with c[2]:
        render_metric_card("Uncertainty", f"±{r['uncertainty']:.1f}", "cycles",
                           sub_html=help_html("MC-dropout estimate", r["uncertainty_method"]))
    with c[3]:
        render_metric_card("Engine", ctx["label"], sub=f"Latest cycle {int(ctx['df']['cycle'].max())} · "
                           f"{len(engine_ids) if ctx['real'] else 1} engine(s) in dataset")

    health_flags(r)
    render_section_header("RUL trajectory", "Predicted remaining useful life over the engine's recorded cycles")
    st.plotly_chart(trajectory_figure(ctx, thr, 360), width="stretch", key="traj_overview")
    if ctx["traj"].get("health_states"):
        md("<div class='card'>" + state_strip_html(ctx["traj"]["health_states"], ctx["traj"]["cycles"]) + "</div>")

    left, right = st.columns(2)
    arrow, label, detail = degradation_trend(ctx["traj"], ctx["traj"]["rul_clip"])
    with left:
        render_section_header("Degradation status", "Model and sensor evidence, shown separately")
        md(evidence_html(r))
        render_metric_card("Degradation trend", label, value_html=f"{arrow} {label}",
                           sub_html=help_html("from predicted RUL slope", detail))
    methodology_expander()
    with right:
        render_section_header("Sensor status", "Drift of recent level vs the engine's early-life baseline")
        render_sensor_list(r["sensor_trends"], STABLE_DRIFT, SENSOR_NAMES)
        st.caption(f"From the API's sensor trends (top drifting channels, training-σ units). "
                   f"|drift| < {STABLE_DRIFT}σ is shown as stable. Correlation only, not causal attribution.")


def page_engine_monitor():
    left, main = st.columns([1, 3], gap="large")
    with left:
        with st.container(border=True):
            render_section_header("Engine")
            if engine_ids:
                engine_selector(st, engine_ids, "engine_select", disabled=ss.data_mode != "real")
            render_section_header("Data mode")
            modes = list(MODES) if has_data else ["sim"]
            st.radio("Data mode", modes, index=modes.index(ss.data_mode), key="data_mode_select",
                     format_func=MODES.get, on_change=sync, args=("data_mode_select", "data_mode"),
                     label_visibility="collapsed")
            if st.button("Run prediction", type="primary", width="stretch", key="run_prediction"):
                if ss.data_mode == "real":
                    cached_real_prediction.clear(API_URL, ss.engine_id)
                else:
                    s = ss.sim_loaded
                    cached_sim_prediction.clear(API_URL, s["stage"], s["seed"], s["n_cycles"])
    ctx = load_context()
    df = ctx["df"]
    with left:
        info = {"Cycles recorded": str(len(df["cycle"])), "Latest cycle": str(int(df["cycle"].max()))}
        if ctx["real"]:
            info["Test label (true RUL)"] = f"{ctx['actual']} cycles"
        else:
            info["Scenario"] = SIM_STAGE_LABELS[ctx["sim"]["stage"]]
            info["Scenario RUL"] = f"{ctx['sim']['scenario_rul_at_last_cycle']} cycles"
        md("<div class='card'>" + "".join(f"<div class='pred-row' style='margin:.25rem 0'>{k} <b style='white-space:nowrap'>{v}</b></div>"
                                          for k, v in info.items()) + "</div>")
        if not ctx["real"]:
            st.caption("Simulation mode shows the sample loaded on the Simulation page.")
    with main:
        render_page_header("ENGINE MONITOR", ctx["label"], status_meta(ctx))
        simulated_banner(ctx)
        r = ctx["result"]
        render_section_header("Current engine status")
        if r is None:
            prediction_unavailable(ctx)
        else:
            render_prediction_card(r["predicted_rul"], r["health_state"], r["uncertainty"], MODEL_LABEL,
                                   simulated=not ctx["real"], latched=r["state_latched"], latch_cycle=r["latch_cycle_number"],
                                   above_boundary=r["current_prediction_above_boundary"],
                                   boundary=r["near_failure_boundary"])
            health_flags(r)
            if r["window_padded"]:
                st.caption("Fewer cycles than the model window; the start of the window was padded.")
            md(evidence_html(r))

        traj = ctx["traj"]
        if traj and traj.get("sensor_index"):
            render_section_header("Sensor degradation index", "Engine-relative; drives the Degrading state")
            si = np.array([np.nan if v is None else v for v in traj["sensor_index"]], dtype=float)
            fig = go.Figure()
            fig.add_scatter(x=traj["cycles"], y=si, mode="lines", line=dict(color=INFO, width=2), name="Sensor index",
                            hovertemplate="Cycle %{x}<br>Sensor index %{y:.2f}<extra></extra>")
            fig.add_hline(y=thr["sensor_boundary"], line=dict(color=STATUS["Degrading"], width=1, dash="dot"),
                          annotation_text=f"Sensor boundary {thr['sensor_boundary']} (learned from training engines)",
                          annotation_position="top left", annotation_font=dict(color=STATUS["Degrading"], size=10))
            if traj.get("degrading_since_cycle") is not None:
                dc = int(traj["cycles"][traj["degrading_since_cycle"]])
                fig.add_vline(x=dc, line=dict(color=STATUS["Degrading"], width=1, dash="dash"),
                              annotation_text=f"Degrading declared (cycle {dc}, after {thr['sensor_persistence_cycles']} "
                                              f"consecutive cycles)", annotation_position="bottom right",
                              annotation_font=dict(color=STATUS["Degrading"], size=10))
            style_fig(fig, 300, "Cycle", "Sensor index (training-σ units)", legend=False)
            st.plotly_chart(fig, width="stretch", key="sensor_index_chart")
            st.caption(PERSISTENCE_TEXT)

        render_section_header("Engine sensor monitoring", "Select a channel")
        sensors = [c for c in SENSOR_INFO if c in df and len(np.unique(df[c])) > 1]
        top = r["sensor_trends"][0]["sensor"] if r and r["sensor_trends"] else None
        default = top if top in sensors else (sensors[0] if sensors else None)
        sel = st.pills("Sensor", sensors, default=default, format_func=sensor_label, key="sensor_pill",
                       label_visibility="collapsed") or default
        if sel:
            code, desc, unit = SENSOR_INFO[sel]
            y = np.asarray(df[sel], dtype=float)
            base = float(y[: min(10, len(y))].mean())
            fig = go.Figure()
            fig.add_scatter(x=df["cycle"], y=y, mode="lines", line=dict(color=INFO, width=2), name=sensor_label(sel),
                            hovertemplate="Cycle %{x}<br>%{y:.4g} " + unit + "<extra></extra>")
            fig.add_hline(y=base, line=dict(color=NEUTRAL_LINE, width=1, dash="dash"),
                          annotation_text="Early-life baseline (mean of first 10 cycles)",
                          annotation_position="bottom right", annotation_font=dict(color="#c3c2b7", size=10))
            style_fig(fig, 340, "Cycle", f"{code} ({unit})" if unit else code,
                      title=f"{sensor_label(sel)} · {code} – {desc}", legend=False)
            st.plotly_chart(fig, width="stretch", key="sensor_chart")
        with st.expander("Compare multiple channels"):
            avail = [c for c in df if (c.startswith("sensor_") or c.startswith("op_setting")) and len(np.unique(df[c])) > 1]
            dflt = [c for c in (model["feature_cols"] if model else avail) if c in avail][:3]
            picked = st.multiselect("Channels", avail, default=dflt, key="sensor_compare",
                                    help="Up to 3 channels are colour-coded; more are harder to tell apart.")
            normalize = st.checkbox("Normalise each channel (z-score) for comparison", value=len(picked) > 1,
                                    key="sensor_normalize")
            fig = go.Figure()
            for i, s in enumerate(picked):
                y = np.asarray(df[s], dtype=float)
                y = (y - y.mean()) / (y.std(ddof=1) or 1) if normalize else y
                fig.add_scatter(x=df["cycle"], y=y, name=sensor_label(s), mode="lines",
                                line=dict(color=SERIES[i % len(SERIES)] if i < len(SERIES) else NEUTRAL_LINE, width=2))
            style_fig(fig, 380, "Cycle", "z-score" if normalize else "Sensor value", legend=len(picked) > 1)
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, width="stretch", key="sensor_compare_chart")
        with st.expander("Raw sensor readings (latest 15 cycles)"):
            md(html_table({c: v[-15:].tolist() for c, v in df.items() if c != "unit"}))


def page_rul_analysis():
    ctx = load_context()
    render_page_header("RUL ANALYSIS", ctx["label"], status_meta(ctx))
    simulated_banner(ctx)
    r = ctx["result"]
    if r is None:
        prediction_unavailable(ctx)
        return
    clip = ctx["traj"]["rul_clip"]
    render_section_header("Remaining useful life")
    a, b = st.columns([1, 1.4], gap="large")
    with a:
        render_prediction_card(r["predicted_rul"], r["health_state"], r["uncertainty"], MODEL_LABEL,
                               simulated=not ctx["real"], latched=r["state_latched"], latch_cycle=r["latch_cycle_number"],
                               above_boundary=r["current_prediction_above_boundary"],
                               boundary=r["near_failure_boundary"])
    with b:
        md(f"<div class='card'><div class='kpi-label'>Prediction uncertainty</div>"
           f"<div class='kpi-value'>{r['predicted_rul']:.0f} <span class='kpi-unit'>± {r['uncertainty']:.1f} cycles"
           f"</span></div>{uncertainty_range_html(r['predicted_rul'], r['uncertainty'], clip, r['near_failure_boundary'])}"
           f"<div class='kpi-sub'>Estimated model uncertainty: ±1 standard deviation of 30 MC-dropout passes. "
           f"Not a calibrated confidence interval. Dashed red line: fleet-level Near-Failure boundary "
           f"({r['near_failure_boundary']:.0f} cycles).</div></div>")
    health_flags(r)
    if ctx["real"]:
        g = st.columns(3)
        with g[0]:
            render_metric_card("Actual RUL (test label)", str(ctx["actual"]), "cycles", "RUL_FD001.txt ground truth")
        with g[1]:
            render_metric_card("Predicted RUL (model)", f"{r['predicted_rul']:.1f}", "cycles", MODEL_LABEL, tone="info")
        with g[2]:
            render_metric_card("Error (pred − actual)", f"{r['predicted_rul'] - ctx['actual']:+.1f}", "cycles",
                               "Positive = late (over-estimated) prediction")
    else:
        sim = ctx["sim"]
        g = st.columns(3)
        with g[0]:
            render_metric_card("Scenario RUL (simulated)", str(sim["scenario_rul_at_last_cycle"]), "cycles",
                               f"Stage band {sim['stage_band'][0]}–{sim['stage_band'][1]} cycles", simulated=True)
        with g[1]:
            render_metric_card("Predicted RUL (model)", f"{r['predicted_rul']:.1f}", "cycles", MODEL_LABEL, tone="info")
        with g[2]:
            render_metric_card("Difference (pred − scenario)", f"{r['predicted_rul'] - sim['scenario_rul_at_last_cycle']:+.1f}",
                               "cycles", "Model error on simulated input")

    render_section_header("RUL trajectory", "Model prediction per cycle, reference line and projection to failure")
    st.plotly_chart(trajectory_figure(ctx, thr, 460), width="stretch", key="traj_analysis")

    render_section_header("Health-state history", "Data-driven state at every recorded cycle")
    if ctx["traj"].get("health_states"):
        md("<div class='card'>" + state_strip_html(ctx["traj"]["health_states"], ctx["traj"]["cycles"]) + "</div>")
    md(evidence_html(r))
    methodology_expander()

    render_section_header("Degradation analysis")
    arrow, label, detail = degradation_trend(ctx["traj"], clip)
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        render_metric_card("Degradation trend", label, value_html=f"{arrow} {label}", sub=detail)
        md(note_html(r["explanation"]))
    with c2:
        if r["sensor_trends"]:
            st.caption("Channels drifting most from the engine's early-life baseline (API sensor trends)")
            rows = [{"Channel": sensor_display(t["sensor"]), "Description": SENSOR_INFO.get(t["sensor"], ("", "", ""))[1],
                     "Drift (training σ)": f"{t['drift_std_units']:+.2f}",
                     "Trend": " ".join(trend_arrow(t["drift_std_units"], STABLE_DRIFT))} for t in r["sensor_trends"]]
            md(html_table(records_to_columns(rows), max_height=None))
    with st.expander("Technical details"):
        st.markdown(f"- Uncertainty method: {r['uncertainty_method']}\n- Cycles used: {r['cycles_used']} "
                    f"(model window {model['seq_len'] if model else 30})\n- RUL cap in training: {clip} cycles\n"
                    f"- Health states: {r.get('thresholds_note', '')}\n"
                    f"- History status: {r.get('history_status')} (cycle {r.get('cycle_number')}; history index {r.get('cycle_index')}, gate from index {thr['history_gate_cycle']})\n"
                    f"- Prediction time: {ctx['fetched_at']}")


def page_simulation():
    render_page_header("WHAT-IF ENGINE SIMULATION", "Synthetic engines at a chosen degradation stage")
    md(banner_html("SIMULATED SCENARIO", "Simulated engines follow the mean FD001 training-set degradation profile "
                   "at a scenario RUL chosen from the stage. They are not real engines and never enter model metrics."))
    loaded = ss.sim_loaded
    left, main = st.columns([1, 2.6], gap="large")
    with left:
        with st.container(border=True):
            if engine_ids:
                render_section_header("Reference engine")
                engine_selector(st, engine_ids, "engine_select_sim")
            # The loaded sample changes ONLY when "Run simulation" is clicked. The controls sit in a form (no rerun
            # while editing). Stable keys keep widget IDs fixed; defaults mirror the loaded sample.
            with st.form("sim_controls"):
                render_section_header("Scenario")
                stage_in = st.radio("Degradation stage (scenario)", SIM_STAGES, index=SIM_STAGES.index(loaded["stage"]),
                                    format_func=SIM_STAGE_LABELS.get, key="sim_stage_in", label_visibility="collapsed")
                seed_in = st.number_input("Simulation seed", 0, 9999, loaded["seed"], key="sim_seed_in")
                cycles_in = st.slider("Simulated cycles", 60, 300, loaded["n_cycles"], key="sim_cycles_in")
                if st.form_submit_button("Run simulation", type="primary", width="stretch"):
                    loaded = ss.sim_loaded = {"stage": stage_in, "seed": int(seed_in), "n_cycles": int(cycles_in)}
        st.caption(f"Loaded sample: **{SIM_STAGE_LABELS[loaded['stage']]}** scenario · seed {loaded['seed']} · "
                   f"{loaded['n_cycles']} cycles")
    with main:
        try:
            sim = cached_simulation(API_URL, loaded["stage"], loaded["seed"], loaded["n_cycles"])
            sim_pred, sim_traj, _ = cached_sim_prediction(API_URL, loaded["stage"], loaded["seed"], loaded["n_cycles"])
        except ApiError as e:
            fail(e, "Simulation unavailable")
            return
        real_pred = real_traj = None
        if engine_ids:
            try:
                real_pred, real_traj, _ = cached_real_prediction(API_URL, ss.engine_id)
            except ApiError as e:
                fail(e, "Prediction for the reference engine unavailable")
        c = st.columns(3)
        with c[0]:
            if real_pred:
                render_metric_card(f"Current RUL · Engine {ss.engine_id}", f"{real_pred['predicted_rul']:.0f}", "cycles",
                                   value_html=f"{real_pred['predicted_rul']:.0f}<span class='kpi-unit'>cycles</span>",
                                   sub_html=status_badge_html(real_pred["health_state"], label=state_badge_label(
                                       real_pred["health_state"], real_pred["state_latched"])) + " real engine",
                                   tone=real_pred["health_state"])
        with c[1]:
            render_metric_card("Scenario RUL (simulated)", str(sim["scenario_rul_at_last_cycle"]), "cycles",
                               sub=f"{SIM_STAGE_LABELS[sim['stage']]} (set by the simulation)", simulated=True)
            md(f"<span data-kpi='Scenario stage (simulated)' data-value='{sim['stage']}'></span>")
        with c[2]:
            render_metric_card("Simulated RUL (model prediction)", f"{sim_pred['predicted_rul']:.0f}", "cycles",
                               value_html=f"{sim_pred['predicted_rul']:.0f}<span class='kpi-unit'>cycles</span>",
                               sub_html=status_badge_html(sim_pred["health_state"], label=state_badge_label(
                                   sim_pred["health_state"], sim_pred["state_latched"])) + f" ±{sim_pred['uncertainty']:.1f} cycles",
                               tone=sim_pred["health_state"], simulated=True)
            md(f"<span data-kpi='Simulated health state' data-value='{sim_pred['health_state']}'></span>")
        health_flags(sim_pred)
        st.caption("The simulated engine's health state is computed with the same data-driven rules as real engines. "
                   "Scenario names describe the scenario RUL only; they are not health states.")

        render_section_header("RUL comparison", "Aligned at each engine's latest observed cycle")
        fig = go.Figure()
        if real_traj:
            rc = np.asarray(real_traj["cycles"])
            fig.add_scatter(x=rc - rc.max(), y=real_traj["predicted_rul"], mode="lines", line=dict(color=SERIES[0], width=2),
                            name=f"Engine {ss.engine_id} – predicted RUL (real data)")
        sc = np.asarray(sim_traj["cycles"])
        fig.add_scatter(x=sc - sc.max(), y=sim_traj["predicted_rul"], mode="lines", line=dict(color=SERIES[1], width=2),
                        name="Simulated engine – predicted RUL")
        fig.add_scatter(x=sc - sc.max(), y=np.minimum(np.asarray(sim["scenario_rul"]), sim_traj["rul_clip"]), mode="lines",
                        line=dict(color=NEUTRAL_LINE, width=2, dash="dash"), name="Simulated engine – scenario RUL")
        add_boundary_line(fig, thr)
        style_fig(fig, 420, "Cycles relative to latest observation (0 = now)", "Remaining useful life (cycles)")
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, width="stretch", key="sim_compare_chart")

        in_monitor = ss.data_mode == "sim"
        if st.button("Stop showing simulated engine in monitor pages" if in_monitor
                     else "Show this simulated engine in the monitor pages", key="toggle_sim_mode",
                     disabled=not has_data and in_monitor):
            ss.data_mode = "real" if in_monitor else "sim"
            st.rerun()
        with st.expander("How the simulation works"):
            st.markdown(f"- Method: {sim['method']}\n- Scenario RUL band for this stage: {sim['stage_band'][0]}–"
                        f"{sim['stage_band'][1]} cycles, chosen reproducibly from the seed\n- {sim['note']}")


def page_model_performance():
    render_page_header("MODEL PERFORMANCE", "Evaluation on NASA C-MAPSS FD001 (research results and deployed model)")
    md(note_html("DATF-Net achieved lower average test error than the evaluated LSTM variants on FD001. Paired "
                 "bootstrap intervals include zero, so the observed improvement is statistically inconclusive."))
    summary = None
    if EXPERIMENT_SUMMARY.exists():
        summary = load_experiment_summary(str(EXPERIMENT_SUMMARY), EXPERIMENT_SUMMARY.stat().st_mtime)
    if summary is None:
        st.info("No research results found (outputs/experiments/summary.json). Run `python -m src.experiments tune`, "
                "`final`, `control` and `report`.")
    else:
        v = summary["variants"]
        models = [(k, lab) for k, lab in RESEARCH_MODELS if k in v]
        seeds = summary["protocol"]["seeds"]
        render_section_header("Model comparison", f"Mean ± SD over {len(seeds)} seeds {seeds}; test = 100 engines")
        f = lambda t: f"{t[0]:.2f} ± {t[1]:.2f}"  # noqa: E731
        md(html_table({"Model": [lab for _, lab in models],
                       "Parameters": [f"{v[k]['n_parameters']:,}" for k, _ in models],
                       "Val RMSE": [f(v[k]["val_rmse"]) for k, _ in models],
                       "Test MAE": [f(v[k]["test_mae"]) for k, _ in models],
                       "Test RMSE": [f(v[k]["test_rmse"]) for k, _ in models],
                       "NASA score": [f"{v[k]['test_nasa'][0]:.0f} ± {v[k]['test_nasa'][1]:.0f}" for k, _ in models]},
                      max_height=None))
        st.caption("Lower is better for every metric. NASA score penalises late predictions more than early ones. "
                   "Source: outputs/experiments/summary.json (local file, not served by the API).")
        cols = st.columns(3)
        for col, (key, title, unit) in zip(cols, (("test_rmse", "Test RMSE", "cycles"), ("test_mae", "Test MAE", "cycles"),
                                                   ("test_nasa", "NASA score", "score"))):
            means = [v[k][key][0] for k, _ in models]
            sds = [v[k][key][1] for k, _ in models]
            fig = go.Figure(go.Bar(x=means, y=[lab for _, lab in models], orientation="h",
                                   marker=dict(color=INFO, line=dict(width=0)), width=0.55,
                                   error_x=dict(type="data", array=sds, color="#c3c2b7", thickness=1.5, width=4),
                                   text=[f"{m:.1f}" if unit != "score" else f"{m:.0f}" for m in means],
                                   textposition="inside", insidetextanchor="start", textfont=dict(color="#f4f4f1"),
                                   hovertemplate="%{y}<br>%{x:.2f} ± %{error_x.array:.2f}<extra></extra>"))
            style_fig(fig, 280, f"{title} ({unit}) – mean ± SD", None, legend=False)
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(rangemode="tozero")
            with col:
                st.plotly_chart(fig, width="stretch", key=f"perf_{key}")

        render_section_header("Paired comparisons on the test set", "Paired bootstrap over 100 engines; negative ΔRMSE "
                              "= first model lower")
        labels = {"datf_full_minus_lstm_info_matched": ("DATF-Net vs LSTM + baseline control", "window + baseline (matched)"),
                  "abl_no_baseline_minus_lstm_baseline": ("DATF-Net without baseline vs LSTM", "window only (matched)"),
                  "datf_full_minus_lstm_baseline": ("DATF-Net vs LSTM", "not matched (DATF-Net also sees baseline)"),
                  "lstm_info_matched_minus_lstm_baseline": ("LSTM + baseline control vs LSTM", "not matched")}
        comp = summary["paired_bootstrap_test_rmse"]
        rows = [(labels[k], comp[k]) for k in labels if k in comp]
        md(html_table({"Comparison": [l[0] for l, _ in rows], "Information seen by both": [l[1] for l, _ in rows],
                       "ΔRMSE": [f"{c['rmse_diff']:+.2f}" for _, c in rows],
                       "95% CI": [f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]" for _, c in rows],
                       "CI includes 0": ["yes" if c["ci95"][0] <= 0 <= c["ci95"][1] else "no" for _, c in rows]},
                      max_height=None))
        with st.expander("DATF-Net ablations (5 seeds each)"):
            abl = {k: val for k, val in v.items() if k.startswith("abl_") or k == "datf_full"}
            md(html_table({"Variant": list(abl), "Val RMSE": [f(a["val_rmse"]) for a in abl.values()],
                           "Test RMSE": [f(a["test_rmse"]) for a in abl.values()],
                           "Test MAE": [f(a["test_mae"]) for a in abl.values()]}, max_height=None))

    render_section_header("Deployed model", f"{MODEL_LABEL} (seed 42) served by the API – the model behind live predictions")
    try:
        metrics, hist, preds = cached_performance(API_URL)
    except ApiError as e:
        if e.status == 404:
            st.info(f"No results yet ({e}). Run `python -m src.train` then `python -m src.evaluate`.")
        else:
            fail(e, "Could not load model performance")
        return
    t, val = metrics["test"], metrics["validation"]
    c = st.columns(4)
    with c[0]:
        render_metric_card("Test MAE (cycles)", f"{t['mae']:.2f}", sub=f"{t['n_engines']} test engines")
    with c[1]:
        render_metric_card("Test RMSE (cycles)", f"{t['rmse']:.2f}", sub="vs unclipped RUL_FD001 labels")
    with c[2]:
        render_metric_card("Val MAE / RMSE", f"{val['mae']:.1f} / {val['rmse']:.1f}", sub=f"{val['n_engines']} held-out engines")
    with c[3]:
        render_metric_card("NASA score (test)", f"{t['nasa_score']:.0f}", sub=f"Checkpoint epoch {metrics['checkpoint_epoch']}")
    render_section_header("Training diagnostics")
    l, rcol = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_scatter(y=hist["train_loss"], name="Train loss", mode="lines", line=dict(color=SERIES[0], width=2))
        fig.add_scatter(y=hist["val_loss"], name="Validation loss", mode="lines", line=dict(color=SERIES[1], width=2))
        style_fig(fig, 360, "Epoch", "MSE on RUL / 125", title="Training and validation loss")
        st.plotly_chart(fig, width="stretch", key="perf_loss")
    with rcol:
        rows = preds["rows"]
        true = np.array([x["true_rul"] for x in rows], dtype=float)
        pred = np.array([x["predicted_rul"] for x in rows])
        mx = float(max(true.max(), pred.max()))
        fig = go.Figure()
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color=NEUTRAL_LINE, width=1),
                        name="Perfect prediction")
        fig.add_scatter(x=true, y=pred, mode="markers", text=[x["engine_id"] for x in rows], name="Test engines",
                        marker=dict(color=pred - true, colorscale="RdBu_r", cmid=0, size=8, showscale=True,
                                    line=dict(color="#1a1a19", width=1), colorbar=dict(title="pred − actual")),
                        hovertemplate="Engine %{text}<br>Actual %{x:.0f}<br>Predicted %{y:.1f}<extra></extra>")
        style_fig(fig, 360, "Actual RUL (cycles)", "Predicted RUL (cycles)", title="Predicted vs actual RUL (test set)")
        st.plotly_chart(fig, width="stretch", key="perf_scatter")


def page_about():
    render_page_header("ABOUT", "Project, data, models and technology")
    render_section_header("Project")
    md("<div class='card'><div class='kpi-value' style='font-size:1.25rem'>Degradation-Aware Temporal Learning for "
       "Remaining Useful Life Prediction Under Variable Operating Conditions</div>"
       "<div class='kpi-sub'>Predicts the remaining useful life (in operating cycles) of turbofan engines from "
       "multivariate sensor histories and reports an illustrative health status.</div></div>")
    a, b = st.columns(2, gap="large")
    with a:
        render_section_header("Data and task")
        st.markdown("- **Dataset:** NASA C-MAPSS FD001 (100 training and 100 test engines, one operating condition, "
                    "one fault mode)\n- **Input:** 30-cycle window of 17 standardised channels (2 operating settings, "
                    "15 sensors)\n- **Output:** remaining useful life in cycles (training target capped at 125)\n"
                    "- **Health state:** data-driven (sensor degradation index + fleet-level Near-Failure boundary)")
        render_section_header("Models")
        st.markdown(f"- **LSTM baseline** – serves all live predictions in this dashboard\n"
                    f"- **LSTM information-matched control** – same LSTM with the engine's early-life baseline as input\n"
                    f"- **DATF-Net** – research model, evaluated offline (see Model Performance); not served by the API")
        render_section_header("Health-state methodology")
        st.markdown(f"{METHODOLOGY_TEXT}\n\n{PERSISTENCE_TEXT}\n\n"
                    "States: Insufficient History · Healthy / Baseline-consistent · Degrading · Boundary / Uncertain · "
                    "Near Failure. Evidence: `outputs/health_threshold_analysis/` (analysis, cross-validation and "
                    "cross-validated Near-Failure reports).")
        render_section_header("Technology")
        st.markdown("Python · PyTorch · FastAPI · Streamlit · NumPy · Plotly · httpx")
        st.caption("Scaling is implemented in NumPy (scikit-learn/SciPy and pandas are not used: their DLLs are "
                   "blocked by Windows Application Control on the development machine).")
    with b:
        render_section_header("DATF-Net architecture (research model)")
        steps = ["Sensor data (NASA C-MAPSS FD001)", "Preprocessing (standardisation, RUL labels)",
                 "Temporal window (30 cycles) + engine baseline (first 10 cycles)", "Temporal encoder (causal dilated TCN)",
                 "Degradation-aware representation (baseline deviation, slope, health index)",
                 "Feature fusion (attention pooling + gated fusion)", "RUL prediction"]
        md("<div class='flow'>" + "<div class='arrow'>↓</div>".join(f"<div class='step'>{s}</div>" for s in steps) + "</div>")
        st.caption("Serving path for live predictions: Streamlit → api_client (HTTP) → FastAPI → LSTM baseline.")


if selected_mode == "✈️ Mode 1: NASA C-MAPSS Demo":
    {"Overview": page_overview, "Engine Monitor": page_engine_monitor, "RUL Analysis": page_rul_analysis,
     "Simulation": page_simulation, "Model Performance": page_model_performance, "About": page_about}[page]()
elif selected_mode == "📁 Mode 2: External Dataset Upload":
    {"Dataset Upload & Validation": lambda: render_mode2_upload(api),
     "Model Safety Shield & Predict": lambda: render_mode2_predict(api)}[page_m2]()
else:
    {"Industry & Sensor Mapping": lambda: render_mode3_mapping(api),
     "Dataset Profiling & Heatmap": lambda: render_mode3_profiling(api),
     "Model Adaptation Training": lambda: render_mode3_training(api),
     "Model Registry & Comparison": lambda: render_mode3_registry(api),
     "Universal RUL Prediction": lambda: render_mode3_predict(api)}[page_m3]()
