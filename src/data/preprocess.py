"""Preprocessing: RUL labels, engine-wise split, scaling, sliding windows.

The fitted `Preprocessor` (feature list + scaler + settings) is saved and reused at
inference so training and inference preprocessing are identical. Data is a `Table`
(dict of NumPy columns, see src/data/table.py); pandas is not used.
"""
from __future__ import annotations

import joblib
import numpy as np

from src.config import PREPROCESSOR_PATH, RUL_CLIP, SEED, SEQ_LEN, VAL_FRACTION
from src.data.load_data import OP_COLS, SENSOR_COLS
from src.data.table import Table, groups, matrix


def add_rul(t: Table, clip: int | None = RUL_CLIP) -> Table:
    """RUL = (last cycle of engine) - cycle, optionally clipped at `clip`."""
    t = dict(t)
    _, inv = np.unique(t["unit"], return_inverse=True)
    last = np.zeros(inv.max() + 1, dtype=t["cycle"].dtype)
    np.maximum.at(last, inv, t["cycle"])
    rul = last[inv] - t["cycle"]
    t["RUL"] = np.minimum(rul, clip) if clip else rul
    return t


def split_engines(units, val_fraction: float = VAL_FRACTION, seed: int = SEED):
    """Split engine ids (not rows) so no engine appears in both sets."""
    units = np.array(sorted(units))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * val_fraction)))
    return np.sort(units[n_val:]), np.sort(units[:n_val])


class StandardScaler:
    """NumPy z-score scaler (same maths as sklearn's StandardScaler; avoids the scipy dependency,
    whose DLLs are blocked by some Windows Application Control policies)."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)  # population std, like sklearn
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


class Preprocessor:
    """Drops constant channels, standardises features (fit on training engines only)."""

    def __init__(self, seq_len: int = SEQ_LEN, rul_clip: int = RUL_CLIP):
        self.seq_len = seq_len
        self.rul_clip = rul_clip
        self.feature_cols: list[str] = []
        self.scaler: StandardScaler | None = None
        self.val_units: list[int] = []
        self.degradation_slope: dict[str, float] = {}  # used only by the simulated demo

    def fit(self, train: Table, val_units=()) -> "Preprocessor":
        """`train` must contain training engines only (no validation engines)."""
        candidates = OP_COLS + SENSOR_COLS
        std = matrix(train, candidates).std(axis=0, ddof=1)
        self.feature_cols = [c for c, s in zip(candidates, std) if s > 1e-6]
        self.scaler = StandardScaler().fit(matrix(train, self.feature_cols))
        self.val_units = [int(u) for u in val_units]
        # mean scaled change between first and last 10% of each engine's life
        X = self.scaler.transform(matrix(train, self.feature_cols))
        slopes = []
        for _, idx in groups(train):
            g = X[idx]
            k = max(1, len(g) // 10)
            slopes.append(g[-k:].mean(axis=0) - g[:k].mean(axis=0))
        self.degradation_slope = dict(zip(self.feature_cols, np.mean(slopes, axis=0).astype(float)))
        return self

    def features(self, t: Table) -> np.ndarray:
        """Scaled (rows, n_features) matrix in `feature_cols` order."""
        missing = [c for c in self.feature_cols if c not in t]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return self.scaler.transform(matrix(t, self.feature_cols))

    def transform(self, t: Table) -> Table:
        """Copy of `t` with the feature columns replaced by their scaled values."""
        out = dict(t)
        X = self.features(t)
        for i, c in enumerate(self.feature_cols):
            out[c] = X[:, i]
        return out

    def save(self, path=PREPROCESSOR_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=PREPROCESSOR_PATH) -> "Preprocessor":
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {path}. Run: python -m src.train")
        return joblib.load(path)


def windows_for_engine(values: np.ndarray, seq_len: int, pad: bool = False):
    """Return (windows, end_indices) for one engine's scaled feature matrix (T, F).

    pad=False: only full windows. pad=True: every cycle gets a window, and the front
    is padded by repeating the first row (also used for engines shorter than seq_len).
    """
    T = len(values)
    if pad or T < seq_len:
        values = np.vstack([np.repeat(values[:1], seq_len - 1, axis=0), values])
        ends = np.arange(T)
        idx = ends[:, None] + np.arange(seq_len)[None, :]
    else:
        ends = np.arange(seq_len - 1, T)
        idx = (ends - seq_len + 1)[:, None] + np.arange(seq_len)[None, :]
    return values[idx].astype(np.float32), ends


def make_windows(scaled: Table, feature_cols, seq_len: int, units=None):
    """Sliding windows (stride 1) with the RUL of each window's last cycle."""
    X, y, u = [], [], []
    F = matrix(scaled, feature_cols)
    for unit, idx in groups(scaled):
        if units is not None and unit not in units:
            continue
        w, ends = windows_for_engine(F[idx], seq_len)
        X.append(w)
        y.append(scaled["RUL"][idx][ends])
        u.append(np.full(len(ends), unit))
    return np.concatenate(X), np.concatenate(y).astype(np.float32), np.concatenate(u)


def last_windows(scaled: Table, feature_cols, seq_len: int):
    """One window per engine ending at its final recorded cycle. Returns (X, unit_ids)."""
    X, ids = [], []
    F = matrix(scaled, feature_cols)
    for unit, idx in groups(scaled):
        w, _ = windows_for_engine(F[idx], seq_len, pad=True)
        X.append(w[-1])
        ids.append(unit)
    return np.stack(X), np.array(ids)
