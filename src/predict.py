"""Inference: loads the saved checkpoint + preprocessor and predicts RUL and its MC-dropout uncertainty.
Health states are NOT decided here; see src/health_state.py (data-driven methodology)."""
from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False

from src import config as C
from src.data.preprocess import Preprocessor, windows_for_engine
from src.data.table import Table, n_rows, sort_by
from src.models.lstm import LSTMRegressor


_LEGACY_BAND_UPPER, _LEGACY_BAND_LOWER = 50, 25


def health_status(rul: float) -> str:
    """LEGACY - retained only because the frozen evaluation script (src/evaluate.py) imports it to reproduce the
    historical `health_status` column of outputs/test_predictions.csv. It is NOT a health state: it is not used by
    the API, the dashboard or src/health_state.py, and its cut-offs are not FD001 findings."""
    if rul <= _LEGACY_BAND_LOWER:
        return "Critical"
    if rul <= _LEGACY_BAND_UPPER:
        return "Warning"
    return "Normal"


class ModelNotAvailableError(RuntimeError):
    pass


def _load_numpy_checkpoint(checkpoint_path: Path) -> dict:
    with zipfile.ZipFile(checkpoint_path) as z:
        prefix = 'lstm_best/' if 'lstm_best/data.pkl' in z.namelist() else ''
        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == '_rebuild_tensor_v2':
                    return lambda storage, storage_offset, size, stride, requires_grad, backward_hooks: storage[storage_offset:storage_offset+np.prod(size)].reshape(size)
                if 'torch' in module:
                    return lambda *args, **kwargs: (module, name, args)
                return super().find_class(module, name)
            def persistent_load(self, pid):
                key = str(pid[2])
                raw = z.read(f'{prefix}data/{key}')
                return np.frombuffer(raw, dtype=np.float32)
        with z.open(f'{prefix}data.pkl') as f:
            ckpt = Unpickler(f).load()
    return ckpt


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _run_numpy_lstm(sd: dict, x: np.ndarray, hidden: int, rul_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Runs 2-layer LSTM forward pass in pure NumPy.
    x shape: (B, T, F).
    Returns (mean_rul, last_hidden) where last_hidden has shape (B, hidden)."""
    B, T, _ = x.shape
    H = hidden

    h0 = np.zeros((B, H), dtype=np.float32)
    c0 = np.zeros((B, H), dtype=np.float32)
    h1 = np.zeros((B, H), dtype=np.float32)
    c1 = np.zeros((B, H), dtype=np.float32)

    w_ih0, w_hh0 = sd['lstm.weight_ih_l0'], sd['lstm.weight_hh_l0']
    bias0 = sd['lstm.bias_ih_l0'] + sd['lstm.bias_hh_l0']

    w_ih1, w_hh1 = sd['lstm.weight_ih_l1'], sd['lstm.weight_hh_l1']
    bias1 = sd['lstm.bias_ih_l1'] + sd['lstm.bias_hh_l1']

    for t in range(T):
        xt = x[:, t, :]
        # Layer 0
        gates0 = xt @ w_ih0.T + h0 @ w_hh0.T + bias0
        i0 = _sigmoid(gates0[:, :H])
        f0 = _sigmoid(gates0[:, H:2*H])
        g0 = np.tanh(gates0[:, 2*H:3*H])
        o0 = _sigmoid(gates0[:, 3*H:])
        c0 = f0 * c0 + i0 * g0
        h0 = o0 * np.tanh(c0)

        # Layer 1
        gates1 = h0 @ w_ih1.T + h1 @ w_hh1.T + bias1
        i1 = _sigmoid(gates1[:, :H])
        f1 = _sigmoid(gates1[:, H:2*H])
        g1 = np.tanh(gates1[:, 2*H:3*H])
        o1 = _sigmoid(gates1[:, 3*H:])
        c1 = f1 * c1 + i1 * g1
        h1 = o1 * np.tanh(c1)

    # Head without dropout
    z0 = np.maximum(h1 @ sd['head.0.weight'].T + sd['head.0.bias'], 0.0)
    z1 = np.maximum(z0 @ sd['head.3.weight'].T + sd['head.3.bias'], 0.0)
    out = z1 @ sd['head.6.weight'].T + sd['head.6.bias']
    mean = np.clip(out.squeeze(-1) * rul_scale, 0.0, None)
    return mean, h1


class Predictor:
    def __init__(self, checkpoint=C.LSTM_CHECKPOINT, preprocessor_path=C.PREPROCESSOR_PATH, device=None):
        if not checkpoint.exists() or not preprocessor_path.exists():
            raise ModelNotAvailableError(
                f"Trained model not found ({checkpoint} / {preprocessor_path}). "
                "Train it first with:  python -m src.train")

        self.prep = Preprocessor.load(preprocessor_path)
        self.use_torch = _TORCH_AVAILABLE

        if self.use_torch:
            try:
                self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
                ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
                if ckpt["feature_cols"] != self.prep.feature_cols:
                    raise ModelNotAvailableError("Checkpoint and preprocessor feature lists differ; retrain the model.")
                self.seq_len = ckpt["seq_len"]
                self.rul_scale = ckpt["rul_scale"]
                self.model = LSTMRegressor(ckpt["n_features"], hidden=ckpt["hidden"]).to(self.device)
                self.model.load_state_dict(ckpt["state_dict"])
                self.model.eval()
                self.info = {"epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"], "seq_len": self.seq_len,
                             "device": str(self.device), "features": len(self.prep.feature_cols)}
            except Exception:
                self.use_torch = False

        if not self.use_torch:
            ckpt = _load_numpy_checkpoint(checkpoint)
            if ckpt["feature_cols"] != self.prep.feature_cols:
                raise ModelNotAvailableError("Checkpoint and preprocessor feature lists differ; retrain the model.")
            self.seq_len = ckpt["seq_len"]
            self.rul_scale = ckpt["rul_scale"]
            self.hidden = ckpt["hidden"]
            self.state_dict = ckpt["state_dict"]
            self.device = "cpu (numpy-fallback)"
            self.info = {"epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"], "seq_len": self.seq_len,
                         "device": "cpu (numpy)", "features": len(self.prep.feature_cols)}

    # ---- core ----
    def _windows(self, t: Table, pad_all: bool):
        t = sort_by(t, "cycle") if "cycle" in t else t
        return windows_for_engine(self.prep.features(t), self.seq_len, pad=pad_all)

    def _predict_windows(self, W: np.ndarray, mc_samples: int = 0, batch: int = 1024):
        """Returns (mean_rul, std_rul). mean is the deterministic prediction; std is the MC-dropout spread."""
        if self.use_torch:
            with torch.no_grad():
                x = torch.from_numpy(W).to(self.device)
                means = [self.model(x[i:i + batch]).cpu().numpy() for i in range(0, len(x), batch)]
                mean = np.clip(np.concatenate(means) * self.rul_scale, 0, None)
                if mc_samples > 0:
                    std = np.array([self._mc_dropout_std_torch(x[i:i + 1], mc_samples) for i in range(len(x))], dtype=np.float32)
                else:
                    std = np.zeros(len(x), dtype=np.float32)
                return mean, std * self.rul_scale
        else:
            # NumPy forward pass
            means = []
            last_hs = []
            for i in range(0, len(W), batch):
                m, lh = _run_numpy_lstm(self.state_dict, W[i:i + batch], self.hidden, self.rul_scale)
                means.append(np.atleast_1d(m))
                last_hs.append(lh)
            mean = np.concatenate(means) if len(means) > 1 else means[0]
            if mc_samples > 0:
                all_lh = np.concatenate(last_hs, axis=0) if len(last_hs) > 1 else last_hs[0]
                std = np.array([self._mc_dropout_std_numpy(all_lh[i:i + 1], mc_samples) for i in range(len(W))], dtype=np.float32)
            else:
                std = np.zeros(len(W), dtype=np.float32)
            return mean, std * self.rul_scale

    def _mc_dropout_std_torch(self, x1, n: int, seed: int = C.MC_DROPOUT_SEED) -> float:
        h = self.model.lstm(x1)[0][:, -1].expand(n, -1)
        gen = torch.Generator(device=x1.device).manual_seed(seed)
        for layer in self.model.head:
            if isinstance(layer, torch.nn.Dropout):
                keep = 1.0 - layer.p
                h = h * torch.bernoulli(torch.full_like(h, keep), generator=gen) / keep
            else:
                h = layer(h)
        return float(h.squeeze(-1).std())

    def _mc_dropout_std_numpy(self, h1: np.ndarray, n: int, seed: int = C.MC_DROPOUT_SEED) -> float:
        sd = self.state_dict
        h_exp = np.tile(h1, (n, 1))
        rng = np.random.RandomState(seed)

        z0 = np.maximum(h_exp @ sd['head.0.weight'].T + sd['head.0.bias'], 0.0)
        mask0 = rng.binomial(1, 0.8, size=z0.shape) / 0.8
        z0_drop = z0 * mask0

        z1 = np.maximum(z0_drop @ sd['head.3.weight'].T + sd['head.3.bias'], 0.0)
        mask1 = rng.binomial(1, 0.8, size=z1.shape) / 0.8
        z1_drop = z1 * mask1

        out = z1_drop @ sd['head.6.weight'].T + sd['head.6.bias']
        return float(np.std(out.squeeze(-1), ddof=1))

    def predict_last(self, t: Table, mc_samples: int = C.MC_DROPOUT_SAMPLES) -> dict:
        """Predict RUL at the last cycle of `t` (raw sensor rows of ONE engine)."""
        n = n_rows(t)
        if n == 0:
            raise ValueError("No sensor readings supplied.")
        W, _ = self._windows(t, pad_all=True)
        mean, std = self._predict_windows(W[-1:], mc_samples)
        rul, unc = float(mean[0]), float(std[0])
        return {
            "predicted_rul": rul,
            "uncertainty": unc,
            "uncertainty_method": (f"MC-dropout std over {mc_samples} fixed-seed dropout masks on the FC head "
                                   "(reproducible; approximate, not calibrated)"),
            "cycles_used": int(n),
            "window_padded": bool(n < self.seq_len),
            "sensor_trends": self.sensor_trends(t),
            "explanation": self.explain(t, rul, unc),
        }

    def predict_trajectory(self, t: Table) -> dict[str, np.ndarray]:
        """Predicted RUL at every cycle of one engine (early cycles use front-padded windows).
        Returns {"cycle": array, "predicted_rul": array}."""
        t = sort_by(t, "cycle")
        W, _ = self._windows(t, pad_all=True)
        mean, _ = self._predict_windows(W)
        return {"cycle": np.asarray(t["cycle"]), "predicted_rul": mean}

    # ---- explanation ----
    def sensor_trends(self, t: Table, top_k: int = 5) -> list[dict]:
        """Sensors whose recent level deviates most from the engine's early baseline (in training std units)."""
        X = self.prep.features(sort_by(t, "cycle"))
        k = max(1, min(10, len(X) // 3))
        drift = X[-k:].mean(axis=0) - X[:k].mean(axis=0)
        order = np.argsort(-np.abs(drift), kind="stable")[:top_k]
        cols = self.prep.feature_cols
        return [{"sensor": cols[i], "drift_std_units": float(drift[i]),
                 "direction": "rising" if drift[i] > 0 else "falling"}
                for i in order if not cols[i].startswith("op_setting")]

    def explain(self, t: Table, rul, unc) -> str:
        n = n_rows(t)
        trends = self.sensor_trends(t, 3)
        txt = (f"The LSTM estimates about {rul:.0f} cycles of remaining life (±{unc:.1f} MC-dropout spread) "
               f"based on the last {min(n, self.seq_len)} cycles.")
        if trends and n >= 6:
            txt += " Channels drifting most from this engine's early baseline: " + ", ".join(
                f"{tr['sensor']} ({tr['direction']}, {tr['drift_std_units']:+.1f}σ)" for tr in trends) + \
                ". This shows correlated sensor movement, not a verified causal attribution."
        elif n < 6:
            txt += " Too few cycles supplied for a meaningful trend summary."
        return txt
