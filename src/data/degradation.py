"""Degradation-aware data pipeline for DATF-Net experiments.

Re-uses the EXISTING fitted preprocessor (scaler, feature list, validation engines) read-only, so every model
sees exactly the same split, scaling, windows (seq_len 30, stride 1) and RUL clipping as the LSTM baseline.

Per window it provides:
  X          (N, T, F)  scaled features (identical to src.data.preprocess.make_windows)
  y          (N,)       clipped RUL at the window's last cycle (identical to make_windows)
  b          (N, F)     engine baseline: mean scaled features over the engine's first `k` OBSERVED cycles.
                        Always in the past of the window (window ends at cycle >= seq_len >= k).
  hi_target  (N, T)     health-index target = min(RUL_t, clip) / clip for every cycle t of the window
                        (1 = healthy, 0 = failure; non-increasing in t).
Test data is only produced by `load_test_split`, which the tuning stage never calls.
"""
from __future__ import annotations

import numpy as np

from src.data.load_data import load_test, load_test_rul, load_train
from src.data.preprocess import Preprocessor, add_rul, windows_for_engine
from src.data.table import Table, groups, matrix

BASELINE_CYCLES = 10


def engine_baseline(F: np.ndarray, k: int = BASELINE_CYCLES) -> np.ndarray:
    """Mean of the first min(k, len(F)) rows of one engine's cycle-ordered scaled feature matrix."""
    if len(F) == 0:
        raise ValueError("engine has no cycles")
    return F[: min(k, len(F))].mean(axis=0)


def _sorted_idx(t: Table, idx: np.ndarray) -> np.ndarray:
    return idx[np.argsort(t["cycle"][idx], kind="stable")]


def build_windows(scaled: Table, feature_cols, seq_len: int, clip: int, units, k: int = BASELINE_CYCLES) -> dict:
    """Sliding windows with baselines and per-cycle HI targets. `scaled` must contain a clipped "RUL" column."""
    F_all = matrix(scaled, feature_cols).astype(np.float32)
    out = {"X": [], "y": [], "b": [], "hi_target": [], "unit": [], "end_cycle": []}
    units = set(int(u) for u in units)
    for unit, idx in groups(scaled):
        if unit not in units:
            continue
        idx = _sorted_idx(scaled, idx)
        Fe, rul, cyc = F_all[idx], scaled["RUL"][idx].astype(np.float32), scaled["cycle"][idx]
        W, ends = windows_for_engine(Fe, seq_len)
        if (ends < k - 1).any():
            raise AssertionError("baseline cycles must precede the window end")
        win_idx = (ends - seq_len + 1)[:, None] + np.arange(seq_len)[None, :]
        out["X"].append(W)
        out["y"].append(rul[ends])
        out["b"].append(np.repeat(engine_baseline(Fe, k)[None, :], len(ends), axis=0))
        out["hi_target"].append(np.minimum(rul[win_idx], clip) / clip)
        out["unit"].append(np.full(len(ends), unit))
        out["end_cycle"].append(cyc[ends])
    return {key: np.concatenate(v).astype(np.float32 if key in ("X", "y", "b", "hi_target") else np.int64)
            for key, v in out.items()}


def load_train_val(prep: Preprocessor, k: int = BASELINE_CYCLES) -> tuple[dict, dict]:
    """Training and validation windows using the saved preprocessor's scaler and validation engines."""
    clip, seq_len = prep.rul_clip, prep.seq_len
    df = add_rul(load_train(), clip=clip)
    scaled = prep.transform(df)
    val_units = set(prep.val_units)
    train_units = sorted(set(int(u) for u in np.unique(df["unit"])) - val_units)
    return (build_windows(scaled, prep.feature_cols, seq_len, clip, train_units, k),
            build_windows(scaled, prep.feature_cols, seq_len, clip, sorted(val_units), k))


def load_test_split(prep: Preprocessor, k: int = BASELINE_CYCLES) -> dict:
    """Last window of every official test engine, its baseline, and the (unclipped) RUL_FD001 labels.
    Only for FINAL evaluation - never for training, early stopping or hyperparameter selection."""
    test = prep.transform(load_test())
    rul = load_test_rul()
    F_all = matrix(test, prep.feature_cols).astype(np.float32)
    X, b, ids = [], [], []
    for unit, idx in groups(test):
        Fe = F_all[_sorted_idx(test, idx)]
        W, _ = windows_for_engine(Fe, prep.seq_len, pad=True)
        X.append(W[-1])
        b.append(engine_baseline(Fe, k))
        ids.append(unit)
    ids = np.array(ids)
    return {"X": np.stack(X).astype(np.float32), "b": np.stack(b).astype(np.float32), "engine_id": ids,
            "true_rul": np.array([rul[int(i)] for i in ids], dtype=np.float32)}
