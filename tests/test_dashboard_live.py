"""End-to-end: a real uvicorn backend process + the Streamlit dashboard script talking to it over HTTP."""
import os
import re
import socket
import subprocess
import sys
import time

import httpx
import pytest

from src import config as C
from src.data.load_data import dataset_available

pytestmark = pytest.mark.skipif(not (C.LSTM_CHECKPOINT.exists() and dataset_available()),
                                reason="trained model or dataset not available")
APP = str(C.ROOT / "dashboard" / "app.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def backend_url():
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--port", str(port)],
                            cwd=C.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(url + "/health", timeout=60).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.5)
        else:
            pytest.fail("backend did not start: " + proc.stderr.read().decode()[-2000:])
        yield url
    finally:
        proc.kill()


def run_app(url, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("RUL_API_URL", url)
    return AppTest.from_file(APP, default_timeout=180).run()


def kpis(at) -> dict:
    """Values of the dashboard's KPI cards (data-kpi / data-value attributes)."""
    return dict(re.findall(r"data-kpi='([^']*)' data-value='([^']*)'", "\n".join(m.value for m in at.markdown)))


def goto(at, page):
    at.radio(key="nav").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_real_mode_over_http(backend_url, monkeypatch):
    at = run_app(backend_url, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    k = kpis(at)  # Overview is the start page
    assert k["Engine"] == "Engine 1" and k["Remaining useful life"] == "111"
    assert k["Health state"] == "Healthy / Baseline-consistent"
    assert len(at.get("plotly_chart")) == 1  # RUL trajectory
    k = kpis(goto(at, "RUL Analysis"))
    assert k["Actual RUL (test label)"] == "112" and k["Predicted RUL (model)"] == "111.4"
    assert kpis(goto(at, "Model Performance"))["Test MAE (cycles)"] == "11.82"
    goto(at, "Engine Monitor")
    at.selectbox(key="engine_select").set_value(34).run()
    assert not at.exception and kpis(at)["Current RUL"] == "7"
    assert len(at.get("plotly_chart")) == 3  # sensor degradation index + sensor chart + channel comparison
    k = kpis(goto(at, "RUL Analysis"))  # selection persists across pages
    assert k["Predicted RUL (model)"] == "7.3" and any("Near Failure — latched" in m.value for m in at.markdown)


def test_simulated_mode_over_http(backend_url, monkeypatch):
    at = run_app(backend_url, monkeypatch)
    goto(at, "Engine Monitor")
    at.radio(key="data_mode_select").set_value("sim").run()
    assert not at.exception, [e.value for e in at.exception]
    k = kpis(goto(at, "Overview"))
    assert k["Engine"] == "Simulated engine" and "Actual RUL (test label)" not in k
    assert any("SIMULATED DATA" in m.value for m in at.markdown)
    k = kpis(goto(at, "RUL Analysis"))
    assert "Scenario RUL (simulated)" in k and "Actual RUL (test label)" not in k
    assert any(re.search(r"Latest cycle <b[^>]*>180</b>", m.value) for m in goto(at, "Engine Monitor").markdown)


def test_backend_down_shows_clear_message(monkeypatch):
    at = run_app(f"http://127.0.0.1:{free_port()}", monkeypatch)
    assert not at.exception
    assert any("Backend unavailable" in e.value for e in at.error)
    assert not kpis(at)  # nothing rendered from stale or fabricated data


# ---------------- simulation controls ----------------
def sim_state(at):
    caption = next(c.value for c in at.caption if c.value.startswith("Loaded sample"))
    k = kpis(at)
    return caption, k["Scenario stage (simulated)"], k["Scenario RUL (simulated)"], k["Simulated RUL (model prediction)"]


def set_sim_controls(at, stage, seed, cycles):
    at.radio(key="sim_stage_in").set_value(stage)
    at.number_input(key="sim_seed_in").set_value(seed)
    at.slider(key="sim_cycles_in").set_value(cycles)


def click_run_simulation(at):
    next(b for b in at.button if b.label == "Run simulation").click().run()
    assert not at.exception, [e.value for e in at.exception]


def expected_sim(backend_url, stage, seed, cycles):
    s = httpx.get(backend_url + "/simulate", params={"stage": stage, "seed": seed, "n_cycles": cycles}, timeout=60).json()
    p = httpx.post(backend_url + "/predict", json={"readings": s["readings"]}, timeout=60).json()
    return str(s["scenario_rul_at_last_cycle"]), f"{p['predicted_rul']:.0f}"


def test_sim_controls_apply_only_on_load_click(backend_url, monkeypatch):
    at = goto(run_app(backend_url, monkeypatch), "Simulation")
    before = sim_state(at)
    assert "Early-degradation scenario" in before[0] and "seed 0" in before[0] and "180 cycles" in before[0]
    assert before[1] == "normal" and (before[2], before[3]) == expected_sim(backend_url, "normal", 0, 180)

    # edit stage, seed and cycles WITHOUT clicking: loaded sample must not change
    set_sim_controls(at, "warning", 7, 90)
    at.run()
    assert not at.exception and sim_state(at) == before

    # an unrelated rerun (reference engine) must not change it either
    at.selectbox(key="engine_select_sim").set_value(34).run()
    assert sim_state(at) == before

    # click Run simulation: the new sample uses exactly the selected stage/seed/cycles
    set_sim_controls(at, "warning", 7, 90)
    click_run_simulation(at)
    after = sim_state(at)
    assert "Late-degradation scenario" in after[0] and "seed 7" in after[0] and "90 cycles" in after[0]
    assert after[1] == "warning" and (after[2], after[3]) == expected_sim(backend_url, "warning", 7, 90)

    # the loaded sample survives a trip to another page and back, and the form shows its settings
    goto(at, "About")
    goto(at, "Simulation")
    assert sim_state(at) == after
    assert at.number_input(key="sim_seed_in").value == 7 and at.radio(key="sim_stage_in").value == "warning"
    assert at.slider(key="sim_cycles_in").value == 90


def test_consecutive_loads_each_apply_selected_values(backend_url, monkeypatch):
    """Regression: without stable widget keys the second Load silently reused the previous values."""
    at = goto(run_app(backend_url, monkeypatch), "Simulation")
    for stage, seed, cycles in (("warning", 7, 90), ("critical", 3, 200), ("normal", 11, 150)):
        set_sim_controls(at, stage, seed, cycles)
        click_run_simulation(at)
        caption, scen_stage, scen_rul, sim_pred = sim_state(at)
        assert scen_stage == stage and f"seed {seed}" in caption and f"{cycles} cycles" in caption
        assert (scen_rul, sim_pred) == expected_sim(backend_url, stage, seed, cycles)
