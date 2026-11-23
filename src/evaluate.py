"""Evaluate the saved LSTM on validation engines and the official FD001 test set.

Usage:  python -m src.evaluate
Writes outputs/metrics.json and outputs/test_predictions.csv from REAL model predictions.
"""
import csv
import json

import numpy as np

from src import config as C
from src.data.load_data import load_test, load_test_rul, load_train
from src.data.preprocess import add_rul, last_windows, make_windows
from src.predict import Predictor, health_status


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def nasa_score(true, pred):
    """Asymmetric PHM08 score (lower is better; late predictions penalised more)."""
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def main():
    p = Predictor()
    clip = p.prep.rul_clip

    # validation engines (held out from training and scaler fitting); windows with full history
    scaled = p.prep.transform(add_rul(load_train(), clip))
    Xv, yv, _ = make_windows(scaled, p.prep.feature_cols, p.seq_len, set(p.prep.val_units))
    pv, _ = p._predict_windows(Xv)

    # official test set: last window of each engine vs RUL_FD001.txt
    Xt, ids = last_windows(p.prep.transform(load_test()), p.prep.feature_cols, p.seq_len)
    pt, ut = p._predict_windows(Xt, mc_samples=30)
    test_rul = load_test_rul()
    true = np.array([test_rul[int(i)] for i in ids])
    true_clipped = np.minimum(true, clip)

    metrics = {
        "dataset": C.DATASET,
        "model": "LSTM baseline",
        "seq_len": p.seq_len,
        "rul_clip": clip,
        "checkpoint_epoch": p.info["epoch"],
        "validation": {"n_windows": int(len(yv)), "n_engines": len(p.prep.val_units),
                       "mae": mae(yv, pv), "rmse": rmse(yv, pv), "target": f"RUL clipped at {clip}"},
        "test": {"n_engines": int(len(ids)),
                 "mae": mae(true, pt), "rmse": rmse(true, pt), "nasa_score": nasa_score(true, pt),
                 "mae_vs_clipped_true": mae(true_clipped, pt), "rmse_vs_clipped_true": rmse(true_clipped, pt),
                 "target": "official RUL_FD001.txt (unclipped) at last recorded cycle"},
    }
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    with open(C.TEST_PREDICTIONS_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine_id", "true_rul", "predicted_rul", "uncertainty", "health_status"])
        for row in zip(ids.tolist(), true.tolist(), pt.tolist(), ut.tolist(), [health_status(v) for v in pt]):
            w.writerow(row)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
