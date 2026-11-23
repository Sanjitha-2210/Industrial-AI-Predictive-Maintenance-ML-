"""API tests. Model/dataset dependent tests are skipped (not faked) if artifacts are missing."""
import csv

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src import config as C
from src.data.load_data import dataset_available

client = TestClient(app)
need_all = pytest.mark.skipif(not (C.LSTM_CHECKPOINT.exists() and dataset_available()),
                              reason="trained model or dataset not available")


def saved_test_predictions():
    with open(C.TEST_PREDICTIONS_PATH, newline="") as f:
        return {int(r["engine_id"]): r for r in csv.DictReader(f)}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and "model_loaded" in r.json() and "thresholds" in r.json()


def test_predict_validation():
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"engine_id": 1, "readings": []}).status_code == 422
    assert client.post("/predict", json={"readings": [{"sensor_1": 1}]}).status_code == 422
    assert client.post("/trajectory", json={}).status_code == 422


def test_simulate_validation():
    assert client.get("/simulate?n_cycles=5").status_code == 422
    assert client.get("/simulate?n_cycles=5000").status_code == 422
    assert client.get("/simulate?seed=-1").status_code == 422
    assert client.get("/simulate?seed=abc").status_code == 422
    assert client.get("/simulate?stage=failing").status_code == 422


@need_all
def test_health_reports_model():
    h = client.get("/health").json()
    assert h["model_loaded"] and h["dataset_available"]
    assert h["model"]["seq_len"] == 30 and h["model"]["rul_clip"] == 125 and len(h["model"]["feature_cols"]) == 17


@need_all
def test_engines_and_predict():
    ids = client.get("/engines").json()["engine_ids"]
    assert len(ids) == 100
    assert client.get("/engines/99999").status_code == 404
    e = client.get(f"/engines/{ids[0]}").json()
    assert e["n_cycles"] > 0
    r = client.post("/predict", json={"engine_id": ids[0]})
    assert r.status_code == 200
    body = r.json()
    from src.health_state import HEALTH_STATES
    assert body["health_state"] in HEALTH_STATES and body["health_status"] == body["health_state"]
    assert body["predicted_rul"] >= 0 and body["current_rul"] == body["predicted_rul"]
    assert client.post("/predict", json={"engine_id": 99999}).status_code == 404


@need_all
def test_predict_from_readings():
    e = client.get("/engines/1?last_n=35").json()
    r = client.post("/predict", json={"readings": e["recent_readings"]})
    assert r.status_code == 200, r.text


@need_all
def test_predict_matches_baseline_for_all_test_engines():
    """Every API prediction equals the baseline written by src.evaluate (outputs/test_predictions.csv)."""
    for eid, row in saved_test_predictions().items():
        body = client.post("/predict", json={"engine_id": eid}).json()
        assert abs(body["predicted_rul"] - float(row["predicted_rul"])) < 1e-2, eid
        assert body["true_rul"] == int(float(row["true_rul"]))  # (the CSV's legacy band column is not compared)


@need_all
def test_trajectory_by_engine_and_by_readings():
    from src.data.load_data import load_test
    from src.data.table import engine
    from src.predict import Predictor
    ref = Predictor().predict_trajectory(engine(load_test(), 34))
    t = client.post("/trajectory", json={"engine_id": 34}).json()
    assert t["cycles"] == ref["cycle"].tolist() and t["true_rul_at_last_cycle"] == 7 and t["rul_clip"] == 125
    assert np.allclose(t["predicted_rul"], ref["predicted_rul"], atol=1e-4)
    # last trajectory point == /predict result (tolerance: float32 GPU results vary slightly with batch size)
    assert abs(t["predicted_rul"][-1] - client.post("/predict", json={"engine_id": 34}).json()["predicted_rul"]) < 1e-2
    rows = client.get("/engines/34?last_n=1000").json()["recent_readings"]
    t2 = client.post("/trajectory", json={"readings": rows}).json()
    assert t2["engine_id"] is None and np.allclose(t2["predicted_rul"], t["predicted_rul"], atol=1e-4)
    assert client.post("/trajectory", json={"engine_id": 99999}).status_code == 404


@need_all
def test_duplicate_cycles_rejected():
    rows = client.get("/engines/1?last_n=3").json()["recent_readings"]
    rows[1]["cycle"] = rows[0]["cycle"]
    r = client.post("/predict", json={"readings": rows})
    assert r.status_code == 422 and "unique" in r.json()["detail"]


@need_all
def test_simulate_then_predict_matches_in_process_model():
    from src.demo import get_profile, simulate_engine
    from src.predict import Predictor
    s = client.get("/simulate?seed=7&n_cycles=90&stage=warning").json()
    assert s["simulated"] is True and "SIMULATED" in s["label"] and len(s["readings"]) == 90
    assert "unit" not in s["readings"][0] and s["readings"][-1]["cycle"] == 90
    assert s["stage"] == "warning" and 33 <= s["scenario_rul_at_last_cycle"] <= 42 and len(s["scenario_rul"]) == 90
    assert client.get("/simulate?seed=7&n_cycles=90&stage=warning").json() == s  # deterministic for a seed
    p = Predictor()
    ref = p.predict_last(simulate_engine(90, 7, "warning", get_profile())[0], mc_samples=0)["predicted_rul"]
    got = client.post("/predict", json={"readings": s["readings"]}).json()
    assert got["engine_id"] is None and got["true_rul"] is None and abs(got["predicted_rul"] - ref) < 1e-4
    t = client.post("/trajectory", json={"readings": s["readings"]}).json()
    assert len(t["cycles"]) == 90 and t["true_rul_at_last_cycle"] is None


@need_all
def test_metrics_endpoints():
    m = client.get("/metrics").json()
    assert abs(m["test"]["mae"] - 11.817148451805116) < 1e-9 and abs(m["test"]["rmse"] - 16.069191319486503) < 1e-9
    h = client.get("/metrics/history").json()
    assert len(h["train_loss"]) == len(h["val_loss"]) == h["epochs_run"]
    tp = client.get("/metrics/test-predictions").json()
    assert tp["count"] == 100 and {"engine_id", "true_rul", "predicted_rul"} <= set(tp["rows"][0])
    assert "health_status" not in tp["rows"][0]  # legacy 50/25 band column is not served
