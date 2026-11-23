"""FastAPI backend - the only component that runs model inference.

Run:  python -m uvicorn backend.main:app --port 8000   (docs at /docs)
The Streamlit dashboard talks to this API over HTTP (see dashboard/api_client.py).
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from src import config as C
from src.data.load_data import OP_COLS, SENSOR_COLS, DatasetMissingError, dataset_available, load_test, load_test_rul
from src.data.table import engine as engine_rows, from_records, n_rows, tail, to_records, unit_ids
from src.demo import SIMULATED_LABEL, STAGE_RUL_BANDS, get_profile, simulate_engine
from src.health_state import HEALTH_STATES, assess_engine, load_params
from src.predict import ModelNotAvailableError, Predictor

app = FastAPI(title="Industrial RUL Prediction API",
              description="LSTM-based Remaining Useful Life prediction on NASA C-MAPSS FD001 with data-driven health "
                          "states (engine-relative sensor degradation index; fleet-level Near-Failure boundary of 45 "
                          "cycles, which is not an individual-engine failure threshold or safety limit).",
              version="1.0.0")

FEATURES = OP_COLS + SENSOR_COLS


@lru_cache(maxsize=1)
def _predictor() -> Predictor:
    return Predictor()


@lru_cache(maxsize=1)
def _test_data():
    return load_test(), load_test_rul()


def get_predictor() -> Predictor:
    try:
        return _predictor()
    except ModelNotAvailableError as e:
        raise HTTPException(503, detail=str(e))


def health_params():
    """Frozen health-state parameters, or None if the calibration file is missing (health fields are then null)."""
    try:
        return load_params()
    except FileNotFoundError:
        return None


@lru_cache(maxsize=256)
def _engine_assessment(engine_id: int):
    """Deterministic per-engine prediction + health-state trajectory for a test-set engine (fresh tracker each time)."""
    predictor = _predictor()
    test, _ = _test_data()
    df = engine_rows(test, engine_id)
    result = predictor.predict_last(df)
    params = health_params()
    states = assess_engine(predictor, df, params, last_prediction=result) if params is not None else None
    return result, states


def assessment_for(req, df, predictor):
    if req.engine_id is not None:
        return _engine_assessment(req.engine_id)
    result = predictor.predict_last(df)
    params = health_params()
    return result, (assess_engine(predictor, df, params, last_prediction=result) if params is not None else None)


HEALTH_FIELDS = ("current_rul", "health_state", "state_latched", "latch_cycle", "near_failure_boundary", "sensor_index",
                 "sensor_boundary", "sensor_state", "sensor_consecutive_cycles_above", "sensor_persistence_required",
                 "degrading_since_cycle", "history_status", "current_prediction_above_boundary", "model_evidence",
                 "sensor_evidence", "disagreement", "cycle_index", "cycle_number", "latch_cycle_number",
                 "degrading_since_cycle_number")


def get_test_data():
    try:
        return _test_data()
    except DatasetMissingError as e:
        raise HTTPException(503, detail=str(e))


class Reading(BaseModel):
    cycle: Optional[int] = Field(None, ge=1)
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float; sensor_2: float; sensor_3: float; sensor_4: float; sensor_5: float
    sensor_6: float; sensor_7: float; sensor_8: float; sensor_9: float; sensor_10: float
    sensor_11: float; sensor_12: float; sensor_13: float; sensor_14: float; sensor_15: float
    sensor_16: float; sensor_17: float; sensor_18: float; sensor_19: float; sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    engine_id: Optional[int] = Field(None, ge=1, description="Engine of the FD001 test set to predict")
    readings: Optional[list[Reading]] = Field(None, min_length=1, max_length=2000,
                                              description="Ordered sensor readings of one engine (oldest first)")

    @model_validator(mode="after")
    def one_source(self):
        if (self.engine_id is None) == (self.readings is None):
            raise ValueError("Provide exactly one of 'engine_id' or 'readings'.")
        return self


class TrajectoryResponse(BaseModel):
    engine_id: Optional[int]
    cycles: list[int]
    predicted_rul: list[float]
    true_rul_at_last_cycle: Optional[int] = None
    rul_clip: int
    note: str = "Early cycles use front-padded windows (fewer than seq_len cycles of history)."
    # data-driven health states per cycle (additive)
    uncertainty: Optional[list[float]] = None
    health_states: Optional[list[str]] = None
    state_latched: Optional[list[bool]] = None
    sensor_index: Optional[list[Optional[float]]] = None
    latch_cycle: Optional[int] = None
    degrading_since_cycle: Optional[int] = None
    near_failure_boundary: Optional[float] = None
    sensor_boundary: Optional[float] = None
    history_gate_cycle: Optional[int] = None


class PredictResponse(BaseModel):
    engine_id: Optional[int]
    predicted_rul: float
    uncertainty: float
    uncertainty_method: str
    health_status: str = Field(description="Data-driven health state (same value as health_state); kept for compatibility")
    cycles_used: int
    window_padded: bool
    sensor_trends: list[dict]
    explanation: str
    true_rul: Optional[int] = None
    thresholds_note: str = ("Health states are data-driven from FD001 training data. The 45-cycle Near-Failure boundary is "
                            "fleet-level, not an individual-engine failure threshold or safety limit.")
    # data-driven health state at the last supplied cycle (additive)
    current_rul: Optional[float] = None
    health_state: Optional[str] = None
    state_latched: Optional[bool] = None
    latch_cycle: Optional[int] = Field(None, description="0-based history index at which Near Failure latched")
    near_failure_boundary: Optional[float] = None
    sensor_index: Optional[float] = None
    sensor_boundary: Optional[float] = None
    sensor_state: Optional[str] = None
    sensor_consecutive_cycles_above: Optional[int] = None
    sensor_persistence_required: Optional[int] = None
    degrading_since_cycle: Optional[int] = None
    history_status: Optional[str] = None
    current_prediction_above_boundary: Optional[bool] = None
    model_evidence: Optional[str] = None
    sensor_evidence: Optional[str] = None
    disagreement: Optional[dict] = None
    cycle_index: Optional[int] = Field(None, description="0-based index in the supplied history (gate: index >= 29)")
    cycle_number: Optional[int] = Field(None, description="cycle number of the last supplied reading")
    latch_cycle_number: Optional[int] = Field(None, description="cycle number at which Near Failure latched")
    degrading_since_cycle_number: Optional[int] = Field(None, description="cycle number at which Degrading was declared")
    health_methodology: Optional[dict] = None


@app.get("/health")
def health():
    model, model_error = None, None
    try:
        p = _predictor()
        model = {**p.info, "type": "LSTM baseline", "rul_clip": p.prep.rul_clip, "feature_cols": p.prep.feature_cols}
    except ModelNotAvailableError as e:
        model_error = str(e)
    return {"status": "ok", "model_loaded": model is not None, "model": model, "model_error": model_error,
            "dataset": C.DATASET, "dataset_available": dataset_available(),
            "thresholds": health_params().public() if health_params() is not None else None,
            "health_states": list(HEALTH_STATES)}


@app.get("/engines")
def engines():
    test, rul = get_test_data()
    ids = unit_ids(test)
    return {"dataset": C.DATASET, "split": "test", "count": len(ids), "engine_ids": ids}


@app.get("/engines/{engine_id}")
def engine(engine_id: int, last_n: int = 50):
    test, rul = get_test_data()
    g = engine_rows(test, engine_id)
    if n_rows(g) == 0:
        raise HTTPException(404, detail=f"Engine {engine_id} not found in the {C.DATASET} test set.")
    if last_n < 1:
        raise HTTPException(422, detail="last_n must be >= 1")
    return {"engine_id": engine_id, "n_cycles": n_rows(g), "latest_cycle": int(g["cycle"].max()),
            "true_rul_at_last_cycle": rul[engine_id],
            "recent_readings": to_records(tail(g, last_n), [c for c in g if c != "unit"])}


def resolve_input(req: PredictRequest):
    """Sensor table + true RUL (test engines only) for a request given by engine_id or readings."""
    true_rul = None
    if req.engine_id is not None:
        test, rul = get_test_data()
        df = engine_rows(test, req.engine_id)
        if n_rows(df) == 0:
            raise HTTPException(404, detail=f"Engine {req.engine_id} not found in the {C.DATASET} test set.")
        true_rul = rul[req.engine_id]
    else:
        records = [r.model_dump() for r in req.readings]
        cycles = [r["cycle"] for r in records]
        if all(c is None for c in cycles):
            cycles = list(range(1, len(records) + 1))
        elif any(c is None for c in cycles):
            raise HTTPException(422, detail="Either give 'cycle' for every reading or for none.")
        df = from_records(records, FEATURES)
        df["cycle"] = np.array(cycles, dtype=np.int64)
        if len(set(cycles)) != len(cycles):
            raise HTTPException(422, detail="'cycle' values must be unique within one engine's readings.")
    return df, true_rul


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predicted RUL at the last supplied cycle, its uncertainty, explanation and the data-driven health state
    (evaluated over the whole supplied history, which is assumed to start at the engine's first cycle)."""
    predictor = get_predictor()
    df, true_rul = resolve_input(req)
    try:
        result, states = assessment_for(req, df, predictor)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    params = health_params()
    health = {k: states[-1][k] for k in HEALTH_FIELDS} if states else {}
    return PredictResponse(engine_id=req.engine_id, true_rul=true_rul,
                           health_status=health.get("health_state", "Unavailable"),
                           health_methodology=params.public() if params is not None else None,
                           **result, **health)


@app.post("/trajectory", response_model=TrajectoryResponse)
def trajectory(req: PredictRequest):
    """Predicted RUL at every supplied cycle (same body as /predict)."""
    predictor = get_predictor()
    df, true_rul = resolve_input(req)
    try:
        traj = predictor.predict_trajectory(df)
        _, states = assessment_for(req, df, predictor)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    params = health_params()
    extra = {}
    if states:
        extra = {"uncertainty": [s["uncertainty"] for s in states], "health_states": [s["health_state"] for s in states],
                 "state_latched": [s["state_latched"] for s in states], "sensor_index": [s["sensor_index"] for s in states],
                 "latch_cycle": states[-1]["latch_cycle"], "degrading_since_cycle": states[-1]["degrading_since_cycle"],
                 "near_failure_boundary": params.near_failure_boundary, "sensor_boundary": params.sensor_boundary,
                 "history_gate_cycle": params.history_gate_cycle}
    return TrajectoryResponse(engine_id=req.engine_id, cycles=traj["cycle"].tolist(),
                              predicted_rul=traj["predicted_rul"].tolist(), true_rul_at_last_cycle=true_rul,
                              rul_clip=predictor.prep.rul_clip, **extra)


@app.get("/simulate")
def simulate(seed: int = Query(0, ge=0, le=1_000_000), n_cycles: int = Query(180, ge=10, le=1000),
             stage: Literal["normal", "warning", "critical"] = Query(
                 "normal", description="Intended degradation stage at the last simulated cycle")):
    """SIMULATED sensor readings for demo mode (not NASA data; never used for metrics).

    Readings follow the mean FD001 training-set degradation profile at a scenario RUL chosen from the
    stage's band (reproducible per seed). Send `readings` to /predict or /trajectory to run the model."""
    t, info = simulate_engine(n_cycles, seed, stage, get_profile())
    return {"simulated": True, "label": SIMULATED_LABEL, "seed": seed, "n_cycles": n_cycles, "stage": stage,
            "stage_band": list(STAGE_RUL_BANDS[stage]),
            "scenario_rul_at_last_cycle": info["scenario_rul_at_last_cycle"],
            "scenario_rul": info["scenario_rul"].tolist(),
            "method": info["method"],
            "note": "Scenario RUL is defined by the simulation (not a measured ground truth) and is never "
                    "used for model metrics.",
            "readings": to_records(t, [c for c in t if c != "unit"])}


@app.get("/metrics")
def metrics():
    if not C.METRICS_PATH.exists():
        raise HTTPException(404, detail="No metrics yet. Run: python -m src.train  then  python -m src.evaluate")
    return json.loads(C.METRICS_PATH.read_text())


@app.get("/metrics/history")
def training_history():
    """Per-epoch training/validation loss recorded by src.train."""
    if not C.HISTORY_PATH.exists():
        raise HTTPException(404, detail="No training history yet. Run: python -m src.train")
    return json.loads(C.HISTORY_PATH.read_text())


@app.get("/metrics/test-predictions")
def test_predictions():
    """Per-engine test-set predictions recorded by src.evaluate (real FD001 data only)."""
    if not C.TEST_PREDICTIONS_PATH.exists():
        raise HTTPException(404, detail="No test predictions yet. Run: python -m src.evaluate")
    with open(C.TEST_PREDICTIONS_PATH, newline="") as f:
        rows = [{"engine_id": int(r["engine_id"]), "true_rul": int(float(r["true_rul"])),
                 "predicted_rul": float(r["predicted_rul"]), "uncertainty": float(r["uncertainty"])}
                for r in csv.DictReader(f)]
    return {"source": "outputs/test_predictions.csv", "count": len(rows), "rows": rows,
            "note": "The file's legacy health_status column (old 50/25 bands) is not served; health states are "
                    "data-driven (see /health)."}
