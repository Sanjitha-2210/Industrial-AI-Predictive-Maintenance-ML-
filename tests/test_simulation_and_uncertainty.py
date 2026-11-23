"""Stage-aware simulation (Issue 1) and reproducible MC-dropout uncertainty (Issue 2)."""
import csv
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src import config as C
from src.data.load_data import COLUMNS, dataset_available, load_test
from src.demo import STAGE_RUL_BANDS, STAGES, get_profile, scenario_rul_end, simulate_engine

need_data = pytest.mark.skipif(not dataset_available(), reason="FD001 dataset not available")
need_all = pytest.mark.skipif(not (C.LSTM_CHECKPOINT.exists() and dataset_available()),
                              reason="trained model or dataset not available")


@pytest.fixture(scope="module")
def predictor():
    from src.predict import Predictor
    return Predictor()


# ---------------- Issue 1: simulation ----------------
def test_scenario_bands_unchanged_ordered_and_not_health_thresholds():
    """Scenario bands keep their original values; they define demo scenarios, not health states."""
    assert STAGE_RUL_BANDS == {"normal": (80, 110), "warning": (33, 42), "critical": (5, 16)}
    assert STAGE_RUL_BANDS["critical"][1] < STAGE_RUL_BANDS["warning"][0] and STAGE_RUL_BANDS["warning"][1] < STAGE_RUL_BANDS["normal"][0]
    assert STAGE_RUL_BANDS["normal"][1] <= C.RUL_CLIP


def test_scenario_rul_is_reproducible_and_in_band():
    for stage in STAGES:
        vals = [scenario_rul_end(stage, s) for s in range(40)]
        assert vals == [scenario_rul_end(stage, s) for s in range(40)]
        lo, hi = STAGE_RUL_BANDS[stage]
        assert all(lo <= v <= hi for v in vals) and len(set(vals)) > 3


def test_scenario_rul_trajectory_is_consistent():
    t, info = simulate_engine(120, 3, "warning", None)
    r = info["scenario_rul"]
    assert list(t) == COLUMNS and len(r) == 120 and r[-1] == info["scenario_rul_at_last_cycle"]
    assert (np.diff(r) == -1).all() and (np.diff(t["cycle"]) == 1).all()


def test_invalid_stage_rejected():
    with pytest.raises(ValueError):
        simulate_engine(60, 0, "failing", None)


@need_data
def test_profile_tracks_real_degradation_and_constants():
    from src.data.load_data import load_train
    prof, train = get_profile(), load_train()
    assert set(prof.constants) == {c for c in COLUMNS[2:] if np.std(train[c]) <= 1e-9}
    # sensor_11 (HPC static pressure) rises towards failure in FD001: profile must reflect that
    j = prof.varying.index("sensor_11")
    assert prof.at(np.array([0]))[0, j] > prof.at(np.array([150]))[0, j] + 3 * prof.within_std[j]
    t, _ = simulate_engine(200, 1, "critical", prof)
    for c, v in prof.constants.items():
        assert (t[c] == v).all()  # constant channels keep their real value (not 0)
    assert np.all(prof.within_std > 0) and np.all(prof.between_std >= 0)


@need_data
def test_simulated_sensors_follow_the_stage():
    prof = get_profile()
    j = prof.varying.index("sensor_11")
    last = {s: np.mean([simulate_engine(180, k, s, prof)[0]["sensor_11"][-10:].mean() for k in range(10)])
            for s in STAGES}
    assert last["normal"] < last["warning"] < last["critical"]


@need_all
def test_model_predictions_agree_with_simulated_stages(predictor):
    """Deterministic regression guard (fixed seeds). Measured agreement: normal 100%, warning ~70%,
    critical ~94% of seeds; warning agreement is limited by model error (~12 cycles MAE) vs a 25-cycle band,
    the same as on real test engines whose true RUL is in the Warning range."""
    prof, preds = get_profile(), {}
    for stage in STAGES:
        rows = []
        for seed in range(30):
            t, info = simulate_engine(180, seed, stage, prof)
            rows.append((predictor.predict_last(t, mc_samples=0)["predicted_rul"], info["scenario_rul_at_last_cycle"]))
        preds[stage] = np.array(rows)
    # predictions fall in their scenario band +-15 cycles for most seeds; measured 0.83 / 0.90 / 1.00 (30 seeds)
    inside = {s: np.mean([STAGE_RUL_BANDS[s][0] - 15 <= p <= STAGE_RUL_BANDS[s][1] + 15 for p, _ in preds[s]]) for s in STAGES}
    assert inside["normal"] >= 0.7 and inside["warning"] >= 0.6 and inside["critical"] >= 0.85, inside
    med = {s: np.median(preds[s][:, 0]) for s in STAGES}
    assert med["critical"] < med["warning"] < med["normal"], med
    # mean |pred - scenario| measured at ~19-21 (normal), ~11-13 (warning), ~4-5 (critical) cycles: the model is
    # least accurate at high RUL, as on real data. Limits leave a small margin above those measurements.
    limits = {"normal": 25, "warning": 15, "critical": 8}
    for s in STAGES:
        assert np.abs(preds[s][:, 0] - preds[s][:, 1]).mean() < limits[s], s


@need_all
def test_simulate_endpoint_stages():
    from backend.main import app
    c = TestClient(app)
    for stage in STAGES:
        s = c.get(f"/simulate?stage={stage}&seed=4&n_cycles=100").json()
        lo, hi = STAGE_RUL_BANDS[stage]
        assert s["simulated"] and s["stage"] == stage and lo <= s["scenario_rul_at_last_cycle"] <= hi
        assert s["scenario_rul"][-1] == s["scenario_rul_at_last_cycle"] and "SIMULATED" in s["label"]
    assert c.get("/simulate?stage=warning&seed=4").json() != c.get("/simulate?stage=warning&seed=5").json()


# ---------------- Issue 2: uncertainty ----------------
@need_all
def test_uncertainty_is_reproducible_and_nonzero(predictor):
    from src.data.table import engine
    t = engine(load_test(), 34)
    a, b = predictor.predict_last(t), predictor.predict_last(t)
    assert a["uncertainty"] == b["uncertainty"] > 0 and a["predicted_rul"] == b["predicted_rul"]
    assert "fixed-seed" in a["uncertainty_method"]


@need_all
def test_uncertainty_reproducible_through_api():
    from backend.main import app
    c = TestClient(app)
    u = {c.post("/predict", json={"engine_id": 20}).json()["uncertainty"] for _ in range(3)}
    assert len(u) == 1 and next(iter(u)) > 0


@need_all
def test_uncertainty_independent_of_batching_and_rng(predictor):
    from src.data.preprocess import last_windows
    Xt, ids = last_windows(predictor.prep.transform(load_test()), predictor.prep.feature_cols, predictor.seq_len)
    torch.manual_seed(123)
    _, batch_std = predictor._predict_windows(Xt[:10], 30)
    torch.manual_seed(999)  # a different global RNG state must not matter
    single = [predictor._predict_windows(Xt[i:i + 1], 30)[1][0] for i in range(10)]
    assert np.array_equal(batch_std, np.array(single, dtype=np.float32))
    state = torch.get_rng_state()
    predictor._predict_windows(Xt[:3], 30)
    assert torch.equal(state, torch.get_rng_state())  # global RNG untouched
    assert not any(m.training for m in predictor.model.modules())  # shared model never left in train mode


@need_all
def test_uncertainty_is_a_real_spread_not_a_constant(predictor):
    """Different mask seeds give different (but similar) estimates; values vary across engines."""
    from src.data.preprocess import last_windows
    Xt, _ = last_windows(predictor.prep.transform(load_test()), predictor.prep.feature_cols, predictor.seq_len)
    x = torch.from_numpy(Xt[:20]).to(predictor.device)
    s0 = np.array([predictor._mc_dropout_std(x[i:i + 1], 30, seed=0) for i in range(20)])
    s1 = np.array([predictor._mc_dropout_std(x[i:i + 1], 30, seed=1) for i in range(20)])
    assert not np.array_equal(s0, s1) and np.corrcoef(s0, s1)[0, 1] > 0.8 and s0.std() > 0


@need_all
def test_point_predictions_unchanged_under_concurrent_requests():
    """Real-data baseline preserved, including when requests run concurrently (the old MC code toggled
    dropout on the shared model, which could leak into another request's point prediction)."""
    from backend.main import app
    c = TestClient(app)
    with open(C.TEST_PREDICTIONS_PATH, newline="") as f:
        saved = {int(r["engine_id"]): float(r["predicted_rul"]) for r in csv.DictReader(f)}
    ids = list(saved)[:40]
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda e: c.post("/predict", json={"engine_id": e}).json()["predicted_rul"], ids))
    assert max(abs(g - saved[e]) for g, e in zip(got, ids)) < 1e-2
