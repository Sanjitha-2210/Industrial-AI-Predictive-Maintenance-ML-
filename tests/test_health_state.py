"""Data-driven health-state logic (src/health_state.py) and its API/dashboard integration.

Logic tests use deterministic synthetic inputs; integration tests use the production model (inference only).
"""
import re
from pathlib import Path

import numpy as np
import pytest

from src import config as C
from src.data.load_data import dataset_available
from src.health_state import (BOUNDARY, DEGRADING, HEALTH_STATES, HEALTHY, INSUFFICIENT, MODEL_ABOVE, MODEL_BOUNDARY,
                              MODEL_NEAR_FAILURE, NEAR_FAILURE, SENSOR_BASELINE_FORMING, SENSOR_CONSISTENT,
                              SENSOR_DEGRADING, HealthParams, HealthStateTracker, assess_trajectory, load_params,
                              sensor_index_series)

need_all = pytest.mark.skipif(not (C.LSTM_CHECKPOINT.exists() and dataset_available()),
                              reason="trained model or dataset not available")
P = HealthParams(sensor_columns=("a", "b"), weights=np.array([1.0, 0.0]))   # L1 0.641, m 20, R 45, gate 29
FAR, CLOSE, BELOW = 120.0, 44.0, 40.0   # predictions: well above / near / below the boundary


def run(preds, uncs=None, sis=None, n=None):
    n = n or len(preds)
    uncs = uncs if uncs is not None else [2.0] * n
    sis = sis if sis is not None else [0.0] * n
    return assess_trajectory(preds, uncs, sis, P)


# ---------------- frozen parameters ----------------
def test_frozen_parameters_match_approved_values():
    p = load_params()
    assert (p.sensor_boundary, p.persistence, p.near_failure_boundary, p.history_gate_cycle) == (0.641, 20, 45.0, 29)
    assert p.baseline_cycles == 10 and len(p.sensor_columns) == len(p.weights) == 15


# ---------------- 1-4: history gate and latch ----------------
def test_1_no_latch_before_cycle_29():
    out = run([BELOW] * 29)                          # prediction + uncertainty <= 45 from cycle 0
    assert all(s["health_state"] == INSUFFICIENT and not s["state_latched"] for s in out)
    assert all(s["model_evidence"] == MODEL_NEAR_FAILURE for s in out)   # evidence is reported, not latched


def test_2_cycle_29_latches_when_prediction_plus_uncertainty_le_45():
    out = run([FAR] * 29 + [43.0], uncs=[2.0] * 30)   # 43 + 2 = 45 <= 45
    assert out[29]["health_state"] == NEAR_FAILURE and out[29]["state_latched"] and out[29]["latch_cycle"] == 29
    assert out[28]["health_state"] == INSUFFICIENT and out[28]["latch_cycle"] is None


def test_3_prediction_plus_uncertainty_above_45_does_not_latch():
    out = run([FAR] * 29 + [43.5] * 5, uncs=[2.0] * 34)   # 43.5 + 2 = 45.5 > 45
    assert not any(s["state_latched"] for s in out)
    assert out[-1]["health_state"] == BOUNDARY and out[-1]["model_evidence"] == MODEL_BOUNDARY


def test_4_5_6_latch_persists_current_rul_visible_and_above_boundary_flag():
    # spec example (cycle 150 latch, cycle 160 RUL 51). The spec's "43 + 3 <= 45" is arithmetically 46 > 45, which
    # under the approved rule is Boundary / Uncertain, so the latch uses 42 + 3 = 45 <= 45.
    assert run([FAR] * 150 + [43.0], uncs=[3.0] * 151)[-1]["health_state"] == BOUNDARY
    preds = [FAR] * 150 + [42.0] + [48.0] * 9 + [51.0]
    out = run(preds, uncs=[3.0] * len(preds))
    assert out[150]["health_state"] == NEAR_FAILURE and out[150]["latch_cycle"] == 150
    last = out[160]
    assert last["health_state"] == NEAR_FAILURE and last["state_latched"] and last["latch_cycle"] == 150
    assert last["current_rul"] == 51.0 and last["uncertainty"] == 3.0          # current prediction still reported
    assert last["current_prediction_above_boundary"] is True
    assert last["model_evidence"] == MODEL_ABOVE                                # evidence reflects the current value
    assert out[150]["current_prediction_above_boundary"] is False


def test_7_boundary_uncertain_around_45():
    for p, u in ((45.0, 1.0), (46.5, 2.0), (43.5, 2.0), (47.0, 2.0)):
        s = run([FAR] * 29 + [p], uncs=[u] * 30)[-1]
        assert s["health_state"] == BOUNDARY and s["model_evidence"] == MODEL_BOUNDARY, (p, u)
    s = run([FAR] * 29 + [47.5], uncs=[2.0] * 30)[-1]           # 47.5 - 2 > 45: clearly above
    assert s["health_state"] == HEALTHY and s["model_evidence"] == MODEL_ABOVE


def test_boundary_does_not_replace_a_latched_near_failure():
    out = run([FAR] * 29 + [40.0, 45.5], uncs=[2.0] * 31)       # latch, then prediction inside the boundary band
    assert out[-1]["health_state"] == NEAR_FAILURE and out[-1]["model_evidence"] == MODEL_BOUNDARY


# ---------------- 8-11: sensor persistence ----------------
def test_8_sensor_index_at_or_below_boundary_never_degrades():
    out = run([FAR] * 100, sis=[0.641] * 100)
    assert all(s["sensor_state"] != SENSOR_DEGRADING for s in out) and out[-1]["health_state"] == HEALTHY


def test_9_fewer_than_20_consecutive_cycles_does_not_degrade():
    sis = [0.0] * 10 + [0.9] * 19 + [0.0] * 5 + [0.9] * 19 + [0.0] * 5
    out = run([FAR] * len(sis), sis=sis)
    assert all(not s["sensor_state"] == SENSOR_DEGRADING for s in out)
    assert max(s["sensor_consecutive_cycles_above"] for s in out) == 19


def test_10_twenty_consecutive_cycles_degrades_and_latches():
    sis = [0.0] * 10 + [0.9] * 20 + [0.0] * 10                   # run on cycles 10..29
    out = run([FAR] * len(sis), sis=sis)
    assert out[28]["sensor_state"] == SENSOR_CONSISTENT and out[29]["sensor_state"] == SENSOR_DEGRADING
    assert out[29]["degrading_since_cycle"] == 29 and out[29]["health_state"] == DEGRADING
    assert out[-1]["health_state"] == DEGRADING and out[-1]["sensor_index"] == 0.0   # latched after the index drops


def test_11_counter_resets_when_index_drops():
    sis = [0.0] * 10 + [0.9] * 15 + [0.5] + [0.9] * 15
    out = run([FAR] * len(sis), sis=sis)
    assert out[24]["sensor_consecutive_cycles_above"] == 15 and out[25]["sensor_consecutive_cycles_above"] == 0
    assert out[-1]["sensor_consecutive_cycles_above"] == 15 and out[-1]["sensor_state"] == SENSOR_CONSISTENT


def test_sensor_counter_does_not_count_baseline_cycles():
    out = run([FAR] * 40, sis=[0.9] * 40)                       # above from cycle 0; counting starts at cycle 10
    assert out[9]["sensor_state"] == SENSOR_BASELINE_FORMING and out[9]["sensor_consecutive_cycles_above"] == 0
    assert out[29]["sensor_state"] == SENSOR_DEGRADING and out[28]["sensor_state"] == SENSOR_CONSISTENT


def test_sensor_index_series_is_engine_relative_and_causal():
    X = np.column_stack([np.r_[np.full(10, 5.0), np.linspace(5, 7, 30)], np.zeros(40)])
    si = sensor_index_series(X, P)
    assert np.isnan(si[:9]).all() and abs(si[9]) < 1e-12              # baseline complete at cycle 9
    X2 = X.copy(); X2[:, 0] += 100.0                                   # engine-specific offset cancels out
    assert np.allclose(sensor_index_series(X2, P)[9:], si[9:])
    X3 = X.copy(); X3[30:, 0] += 50.0                                  # future readings do not change the past
    assert np.allclose(sensor_index_series(X3, P)[:30], si[:30], equal_nan=True)


# ---------------- 12-13: independence and reset ----------------
def test_12_engines_do_not_share_latch_state():
    a = HealthStateTracker(P); b = HealthStateTracker(P)
    for _ in range(29):
        a.update(FAR, 2.0, 0.0); b.update(FAR, 2.0, 0.0)
    assert a.update(40.0, 2.0, 0.0)["state_latched"]
    assert not b.update(FAR, 2.0, 0.0)["state_latched"]
    assert run([FAR] * 40)[-1]["state_latched"] is False               # a separate call is independent too


def test_13_new_trajectory_resets_latch():
    first = run([FAR] * 29 + [40.0] + [FAR] * 5)
    again = run([FAR] * 35)
    assert first[-1]["state_latched"] and not again[-1]["state_latched"]
    assert run([FAR] * 29 + [40.0] + [FAR] * 5) == first                 # deterministic / reproducible


# ---------------- disagreement ----------------
def test_disagreement_is_explicit():
    s = run([FAR] * 29 + [40.0], sis=[0.0] * 30)[-1]                    # model Near Failure, sensors at baseline
    assert s["disagreement"]["present"] and s["disagreement"]["kind"] == "conflict"
    s = run([FAR] * 29 + [45.0], sis=[0.0] * 30)[-1]                    # model boundary, sensors at baseline
    assert s["disagreement"]["present"] and s["disagreement"]["kind"] == "conflict"
    sis = [0.0] * 10 + [0.9] * 30
    s = run([FAR] * 39 + [40.0], sis=sis)[-1]                           # model Near Failure, sensors Degrading
    assert s["disagreement"]["present"] and s["disagreement"]["kind"] == "level_difference"
    s = run([FAR] * 39 + [45.0], sis=sis)[-1]                           # boundary + degrading: both shown
    assert not s["disagreement"]["present"] and s["model_evidence"] == MODEL_BOUNDARY and s["sensor_evidence"] == SENSOR_DEGRADING
    assert run([BELOW] * 10)[-1]["disagreement"]["kind"] == "not_evaluated"


# ---------------- 14-15: old logic absent ----------------
RUNTIME_FILES = ("src/health_state.py", "backend/main.py", "dashboard/app.py", "dashboard/components.py",
                 "dashboard/theme.py", "dashboard/api_client.py", "src/config.py")


def test_14_15_old_50_25_rules_absent_from_runtime():
    for f in RUNTIME_FILES:
        src = (Path(C.ROOT) / f).read_text(encoding="utf-8")
        for banned in ("WARNING_RUL", "CRITICAL_RUL", "warning_rul", "critical_rul", "<= 25", "<= 50", "≤ 25", "≤ 50",
                       "health_status(", '"Critical"', '"Warning"', '"Normal"'):
            assert banned not in src, (f, banned)
    assert set(HEALTH_STATES) == {INSUFFICIENT, HEALTHY, DEGRADING, BOUNDARY, NEAR_FAILURE}
    assert not hasattr(C, "WARNING_RUL") and not hasattr(C, "CRITICAL_RUL")
    assert not any("Critical" in s for s in HEALTH_STATES)


def test_state_machine_never_emits_old_labels():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 150
        out = run(list(rng.uniform(0, 130, n)), uncs=list(rng.uniform(0, 10, n)), sis=list(rng.uniform(-1, 3, n)))
        assert {s["health_state"] for s in out} <= set(HEALTH_STATES)


# ---------------- 16-17: API compatibility and unchanged predictions ----------------
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app
    return TestClient(app)


@need_all
def test_16_existing_api_fields_remain(client):
    r = client.post("/predict", json={"engine_id": 34}).json()
    for f in ("engine_id", "predicted_rul", "uncertainty", "uncertainty_method", "health_status", "cycles_used",
              "window_padded", "sensor_trends", "explanation", "true_rul", "thresholds_note"):
        assert f in r, f
    for f in ("current_rul", "health_state", "state_latched", "latch_cycle", "near_failure_boundary", "sensor_index",
              "sensor_boundary", "sensor_state", "history_status", "current_prediction_above_boundary",
              "model_evidence", "sensor_evidence", "disagreement"):
        assert f in r, f
    assert r["health_status"] == r["health_state"] == NEAR_FAILURE and r["near_failure_boundary"] == 45.0
    assert r["sensor_boundary"] == 0.641 and r["history_status"] == "sufficient"
    t = client.post("/trajectory", json={"engine_id": 34}).json()
    for f in ("engine_id", "cycles", "predicted_rul", "true_rul_at_last_cycle", "rul_clip"):
        assert f in t
    assert len(t["health_states"]) == len(t["cycles"]) == len(t["uncertainty"]) and t["latch_cycle"] == r["latch_cycle"]
    assert all(s == INSUFFICIENT for s in t["health_states"][:29]) and t["health_states"][29] != INSUFFICIENT
    h = client.get("/health").json()
    assert h["thresholds"]["near_failure_boundary"] == 45.0 and h["health_states"] == list(HEALTH_STATES)
    # 0-based history indices and data cycle numbers are both exposed and consistent
    assert r["latch_cycle_number"] == t["cycles"][r["latch_cycle"]] and r["cycle_number"] == t["cycles"][-1]
    assert r["degrading_since_cycle_number"] == t["cycles"][r["degrading_since_cycle"]]


@need_all
def test_17_model_predictions_unchanged(client):
    import csv
    with open(C.TEST_PREDICTIONS_PATH, newline="") as f:
        saved = {int(r["engine_id"]): r for r in csv.DictReader(f)}
    for eid in (1, 20, 34, 81, 100):
        r = client.post("/predict", json={"engine_id": eid}).json()
        assert abs(r["predicted_rul"] - float(saved[eid]["predicted_rul"])) < 1e-2
        assert r["current_rul"] == r["predicted_rul"]
    from src.data.load_data import load_test
    from src.data.table import engine
    from src.predict import Predictor
    p = Predictor()
    direct = p.predict_last(engine(load_test(), 34))
    r = client.post("/predict", json={"engine_id": 34}).json()
    assert r["predicted_rul"] == direct["predicted_rul"] and r["uncertainty"] == direct["uncertainty"]


@need_all
def test_api_states_follow_the_rules_on_a_real_trajectory(client):
    t = client.post("/trajectory", json={"engine_id": 20}).json()
    states, unc, pred = t["health_states"], t["uncertainty"], t["predicted_rul"]
    lc = t["latch_cycle"]
    assert lc is not None and lc >= 29 and all(s == NEAR_FAILURE for s in states[lc:])
    assert all(s != NEAR_FAILURE for s in states[:lc])


@need_all
def test_readings_request_gets_a_fresh_state(client):
    """Two readings-based requests from different engines must not share state."""
    s1 = client.get("/simulate", params={"stage": "critical", "seed": 1, "n_cycles": 150}).json()
    s2 = client.get("/simulate", params={"stage": "normal", "seed": 1, "n_cycles": 150}).json()
    r1 = client.post("/predict", json={"readings": s1["readings"]}).json()
    r2 = client.post("/predict", json={"readings": s2["readings"]}).json()
    assert r1["state_latched"] and not r2["state_latched"]
    assert client.post("/predict", json={"readings": s1["readings"][:25]}).json()["health_state"] == INSUFFICIENT
