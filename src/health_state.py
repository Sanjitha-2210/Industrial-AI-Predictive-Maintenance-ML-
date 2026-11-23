"""Data-driven health states for FD001 (approved methodology; see outputs/health_threshold_analysis/*_report.md).

States: Insufficient History, Healthy / Baseline-consistent, Degrading, Boundary / Uncertain, Near Failure.

* History gate: cycles 0-28 (0-based index in the engine's observed history) -> "Insufficient History".
  Cycle 29 is the first cycle with a complete 30-cycle model window. The gate never alters predictions.
* Sensor evidence (model-independent, engine-relative): sensor index = frozen training-PCA weights . (x_t - b),
  b = mean of the engine's first 10 cycles, smoothed by a causal 10-cycle mean. Degrading when the index exceeds
  L1 = 0.641 for m = 20 consecutive cycles (counted from cycle 10; counter resets when the index drops to <= L1;
  latched once reached). m = 20 is an implementation policy choice from the validation trade-off, not a discovered
  physical boundary.
* Model evidence: the LSTM's predicted RUL p and its production MC-dropout uncertainty u, against the fleet-level
  FD001 acceleration boundary R_acc = 45 cycles (NOT an individual-engine failure threshold or safety limit).
  Near Failure latches at the first cycle >= 29 with p + u <= 45 and stays latched; the current prediction is always
  reported, and `current_prediction_above_boundary` is true whenever p > 45 after the latch.
  Boundary / Uncertain: |p - 45| <= u (and not p + u <= 45).
* Overall health_state (explicit, documented precedence): Insufficient History (cycle < 29) > Near Failure (latched)
  > Boundary / Uncertain > Degrading > Healthy / Baseline-consistent. Model and sensor evidence are always reported
  separately, and differences between them are exposed in `disagreement`, never resolved silently.

State is held per trajectory in a HealthStateTracker instance; assess_* functions create a fresh tracker per call,
so one engine's latch can never leak into another engine or a later request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import PROCESSED_DIR

PARAMS_PATH = PROCESSED_DIR / "health_state_params.json"

INSUFFICIENT = "Insufficient History"
HEALTHY = "Healthy / Baseline-consistent"
DEGRADING = "Degrading"
BOUNDARY = "Boundary / Uncertain"
NEAR_FAILURE = "Near Failure"
HEALTH_STATES = (INSUFFICIENT, HEALTHY, DEGRADING, BOUNDARY, NEAR_FAILURE)

# sensor evidence labels
SENSOR_BASELINE_FORMING = "Baseline forming"          # fewer than 10 cycles: baseline not yet established
SENSOR_CONSISTENT = "Baseline-consistent"
SENSOR_DEGRADING = "Degrading"
# model evidence labels (current prediction)
MODEL_ABOVE = "Above Near-Failure boundary"
MODEL_BOUNDARY = "Boundary / Uncertain"
MODEL_NEAR_FAILURE = "Near Failure"

METHODOLOGY_NOTE = (
    "Health states are derived from FD001 training-data analysis. Degrading is identified using an engine-relative "
    "sensor degradation index. Near Failure uses a fleet-level RUL acceleration boundary of 45 cycles. The 45-cycle "
    "boundary is not an individual-engine failure threshold or safety limit.")
PERSISTENCE_NOTE = "The 20-cycle persistence requirement is an implementation policy chosen from validation analysis."


@dataclass(frozen=True)
class HealthParams:
    sensor_columns: tuple
    weights: np.ndarray
    baseline_cycles: int = 10
    smoothing_window: int = 10
    sensor_boundary: float = 0.641
    persistence: int = 20
    near_failure_boundary: float = 45.0
    history_gate_cycle: int = 29

    @staticmethod
    def from_dict(d: dict) -> "HealthParams":
        return HealthParams(tuple(d["sensor_columns"]), np.asarray(d["weights"], dtype=float), d["baseline_cycles"],
                            d["smoothing_window"], d["sensor_boundary"], d["persistence"], d["near_failure_boundary"],
                            d["history_gate_cycle"])

    def public(self) -> dict:
        return {"sensor_boundary": self.sensor_boundary, "sensor_persistence_cycles": self.persistence,
                "near_failure_boundary": self.near_failure_boundary, "history_gate_cycle": self.history_gate_cycle,
                "baseline_cycles": self.baseline_cycles, "smoothing_window": self.smoothing_window,
                "methodology": METHODOLOGY_NOTE, "persistence_note": PERSISTENCE_NOTE}


@lru_cache(maxsize=1)
def load_params(path: str = str(PARAMS_PATH)) -> HealthParams:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Health-state parameters not found at {p}. Create them with: "
                                "python -m src.health_state_calibrate")
    return HealthParams.from_dict(json.loads(p.read_text(encoding="utf-8")))


# ---------------- sensor index ----------------
def sensor_index_series(sensor_matrix: np.ndarray, params: HealthParams) -> np.ndarray:
    """Smoothed engine-relative sensor degradation index for every cycle of one engine.

    sensor_matrix: (T, n_sensors) scaled sensor readings in params.sensor_columns order, oldest first.
    Returns NaN for cycles before the baseline is complete (index < baseline_cycles - 1).
    """
    X = np.asarray(sensor_matrix, dtype=float)
    T, k = len(X), params.baseline_cycles
    out = np.full(T, np.nan)
    if T < k:
        return out
    di = (X - X[:k].mean(axis=0)) @ params.weights
    c = np.cumsum(np.insert(di, 0, 0.0))
    t = np.arange(T)
    lo = np.maximum(0, t - params.smoothing_window + 1)
    smoothed = (c[t + 1] - c[lo]) / (t + 1 - lo)
    out[k - 1:] = smoothed[k - 1:]
    return out


# ---------------- per-trajectory state machine ----------------
class HealthStateTracker:
    """Deterministic state machine for ONE engine trajectory. Create a new instance per engine / session."""

    def __init__(self, params: HealthParams):
        self.p = params
        self.cycle = -1
        self.sensor_run = 0
        self.degrading = False
        self.degrading_cycle: int | None = None
        self.latched = False
        self.latch_cycle: int | None = None

    def update(self, predicted_rul: float, uncertainty: float, sensor_index: float | None) -> dict:
        p = self.p
        self.cycle += 1
        c = self.cycle
        R = p.near_failure_boundary
        pr, u = float(predicted_rul), max(float(uncertainty), 0.0)

        # ---- sensor evidence (counter from the first cycle after the baseline window) ----
        si = None if sensor_index is None or np.isnan(sensor_index) else float(sensor_index)
        above_now = si is not None and c >= p.baseline_cycles and si > p.sensor_boundary
        if not self.degrading:
            self.sensor_run = self.sensor_run + 1 if above_now else 0
            if self.sensor_run >= p.persistence:
                self.degrading, self.degrading_cycle = True, c
        if si is None or c < p.baseline_cycles:
            sensor_state = SENSOR_BASELINE_FORMING
        else:
            sensor_state = SENSOR_DEGRADING if self.degrading else SENSOR_CONSISTENT

        # ---- model evidence (current prediction) ----
        if pr + u <= R:
            model_now = MODEL_NEAR_FAILURE
        elif abs(pr - R) <= u:
            model_now = MODEL_BOUNDARY
        else:
            model_now = MODEL_ABOVE
        sufficient = c >= p.history_gate_cycle
        if sufficient and not self.latched and model_now == MODEL_NEAR_FAILURE:
            self.latched, self.latch_cycle = True, c

        # ---- overall state (documented precedence) ----
        if not sufficient:
            state = INSUFFICIENT
        elif self.latched:
            state = NEAR_FAILURE
        elif model_now == MODEL_BOUNDARY:
            state = BOUNDARY
        elif self.degrading:
            state = DEGRADING
        else:
            state = HEALTHY

        return {
            "cycle_index": c, "health_state": state,
            "history_status": "sufficient" if sufficient else "insufficient",
            "current_rul": pr, "uncertainty": u,
            "state_latched": self.latched, "latch_cycle": self.latch_cycle,
            "current_prediction_above_boundary": bool(self.latched and pr > R),
            "near_failure_boundary": R,
            "sensor_index": si, "sensor_boundary": p.sensor_boundary,
            "sensor_consecutive_cycles_above": self.sensor_run if not self.degrading else None,
            "sensor_persistence_required": p.persistence,
            "sensor_state": sensor_state, "degrading_since_cycle": self.degrading_cycle,
            "model_evidence": model_now, "sensor_evidence": sensor_state,
            "disagreement": _disagreement(sufficient, self.latched, model_now, sensor_state),
        }


def _disagreement(sufficient: bool, latched: bool, model_now: str, sensor_state: str) -> dict:
    """Explicit comparison of model and sensor evidence (never resolved silently)."""
    if not sufficient or sensor_state == SENSOR_BASELINE_FORMING:
        return {"present": False, "kind": "not_evaluated", "message": "Insufficient history to compare evidence."}
    model_level = MODEL_NEAR_FAILURE if latched else model_now
    if model_level == MODEL_NEAR_FAILURE and sensor_state == SENSOR_CONSISTENT:
        return {"present": True, "kind": "conflict",
                "message": "Model evidence indicates Near Failure, but the sensor index is still consistent with this "
                           "engine's early-life baseline."}
    if model_level == MODEL_BOUNDARY and sensor_state == SENSOR_CONSISTENT:
        return {"present": True, "kind": "conflict",
                "message": "The model prediction is near the Near-Failure boundary, but the sensor index is still "
                           "consistent with this engine's early-life baseline."}
    if model_level == MODEL_NEAR_FAILURE and sensor_state == SENSOR_DEGRADING:
        return {"present": True, "kind": "level_difference",
                "message": "Model evidence indicates Near Failure; sensor evidence indicates Degrading. The sensor "
                           "index has no near-failure level, so the two sources describe different stages."}
    return {"present": False, "kind": "consistent" if model_level == MODEL_ABOVE else "both_shown",
            "message": "Model and sensor evidence are shown side by side."}


def assess_trajectory(predicted_rul, uncertainty, sensor_index, params: HealthParams) -> list[dict]:
    """Per-cycle states for one engine; a fresh tracker is used, so every call starts unlatched."""
    tracker = HealthStateTracker(params)
    return [tracker.update(p, u, s) for p, u, s in zip(predicted_rul, uncertainty, sensor_index)]


def assess_engine(predictor, table, params: HealthParams, last_prediction: dict | None = None) -> list[dict]:
    """Health states for every cycle of one engine's observed history (oldest first) using the production model and
    its production MC-dropout uncertainty (Predictor._predict_windows). The history is assumed to start at the
    engine's first operating cycle (baseline = first 10 supplied cycles).
    `last_prediction` (from Predictor.predict_last) makes the final cycle use exactly the reported values."""
    from src.data.table import sort_by
    t = sort_by(table, "cycle")
    W, _ = predictor._windows(t, pad_all=True)
    from src import config as C
    mean, std = predictor._predict_windows(W, mc_samples=C.MC_DROPOUT_SAMPLES)
    if last_prediction is not None:
        mean[-1], std[-1] = last_prediction["predicted_rul"], last_prediction["uncertainty"]
    X = predictor.prep.features(t)
    cols = [predictor.prep.feature_cols.index(c) for c in params.sensor_columns]
    si = sensor_index_series(X[:, cols], params)
    return with_cycle_numbers(assess_trajectory(mean, std, si, params), t["cycle"])


def with_cycle_numbers(states: list[dict], cycle_numbers) -> list[dict]:
    """Add the data's cycle numbers next to the 0-based history indices (cycle_index, latch_cycle,
    degrading_since_cycle), so displays never confuse the two."""
    nums = [int(c) for c in cycle_numbers]
    for s in states:
        s["cycle_number"] = nums[s["cycle_index"]]
        s["latch_cycle_number"] = None if s["latch_cycle"] is None else nums[s["latch_cycle"]]
        s["degrading_since_cycle_number"] = (None if s["degrading_since_cycle"] is None
                                             else nums[s["degrading_since_cycle"]])
    return states
