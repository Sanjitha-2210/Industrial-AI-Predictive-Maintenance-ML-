"""Freeze the health-state parameters from the 80 TRAINING engines (no test data, no model, no retraining).

Usage:  python -m src.health_state_calibrate
Writes data/processed/health_state_params.json (a new file; the saved scaler/preprocessor is only read).

Values:
* sensor-index weights: first principal component of baseline deviations of the 15 scaled sensors over the 80
  training engines (recomputed here; identical to outputs/health_threshold_analysis/analysis_results.json);
* sensor boundary L1 = 0.641 (approved; the recomputed 3-sigma value is stored alongside for provenance);
* persistence m = 20 (implementation policy from cv_validation_report.md);
* Near-Failure boundary R_acc = 45 (fleet-level; analysis_report.md, cv_validation_report.md,
  near_failure_cv/near_failure_cv_report.md);
* history gate 29 (first cycle with a complete 30-cycle window).
"""
from __future__ import annotations

import json

import numpy as np

from src import config as C
from src.data.load_data import SENSOR_COLS, load_train
from src.data.preprocess import Preprocessor
from src.data.table import groups, matrix
from src.health_state import PARAMS_PATH

BASE, W = 10, 10
APPROVED = {"sensor_boundary": 0.641, "persistence": 20, "near_failure_boundary": 45.0, "history_gate_cycle": 29}


def calibrate() -> dict:
    prep = Preprocessor.load()
    raw = load_train()
    val = set(int(u) for u in prep.val_units)
    train_units = [u for u in sorted(int(u) for u in np.unique(raw["unit"])) if u not in val]
    sens = [c for c in prep.feature_cols if c in SENSOR_COLS]
    X = matrix(prep.transform(raw), sens)
    eng = {u: idx[np.argsort(raw["cycle"][idx])] for u, idx in groups(raw) if u in train_units}
    dev = {u: X[eng[u]] - X[eng[u]][:BASE].mean(0) for u in train_units}
    P = np.concatenate([dev[u] for u in train_units])
    ev, V = np.linalg.eigh(np.cov(P, rowvar=False))
    w = V[:, np.argmax(ev)]
    if np.corrcoef(P @ w, np.concatenate([np.arange(len(dev[u])) for u in train_units]))[0, 1] < 0:
        w = -w
    DI = {u: dev[u] @ w for u in train_units}

    def cmean(x):
        c = np.cumsum(np.insert(x, 0, 0.0)); t = np.arange(len(x)); lo = np.maximum(0, t - W + 1)
        return (c[t + 1] - c[lo]) / (t + 1 - lo)
    sigma_w = float(np.concatenate([cmean(DI[u])[W - 1:BASE + 20] - DI[u][:BASE].mean() for u in train_units]).std(ddof=1))
    return {
        "sensor_columns": sens, "weights": [float(v) for v in w], "baseline_cycles": BASE, "smoothing_window": W,
        **APPROVED,
        "provenance": {
            "training_engines": len(train_units), "validation_engines_excluded": sorted(val), "test_set": "not used",
            "scaler": "data/processed/preprocessor.joblib (read-only)",
            "sensor_boundary_recomputed_3sigma": 3 * sigma_w, "explained_variance_pc1": float(ev.max() / ev.sum()),
            "near_failure_boundary": "median RUL at the supported acceleration change point, 80 training engines "
                                     "(45.0); fold values 41.5-48.0; fleet-level only",
            "persistence": "implementation policy chosen from the validation trade-off; not a discovered boundary",
            "reports": ["outputs/health_threshold_analysis/analysis_report.md",
                        "outputs/health_threshold_analysis/cv_validation_report.md",
                        "outputs/health_threshold_analysis/near_failure_cv/near_failure_cv_report.md"]},
    }


def main():
    params = calibrate()
    ref = C.OUTPUT_DIR / "health_threshold_analysis" / "analysis_results.json"
    if ref.exists():
        ref_w = json.loads(ref.read_text())["degradation_index"]["weights"]
        diff = max(abs(params["weights"][i] - ref_w[c]) for i, c in enumerate(params["sensor_columns"]))
        assert diff < 1e-9, f"weights differ from the validated analysis by {diff}"
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"wrote {PARAMS_PATH}; recomputed L1 {params['provenance']['sensor_boundary_recomputed_3sigma']:.4f} "
          f"(approved {params['sensor_boundary']})")


if __name__ == "__main__":
    main()
