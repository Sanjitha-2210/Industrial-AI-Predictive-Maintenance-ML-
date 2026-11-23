"""Inference: loads the saved checkpoint + preprocessor and predicts RUL and its MC-dropout uncertainty.
Health states are NOT decided here; see src/health_state.py (data-driven methodology)."""
from __future__ import annotations

import numpy as np
import torch

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


class Predictor:
    def __init__(self, checkpoint=C.LSTM_CHECKPOINT, preprocessor_path=C.PREPROCESSOR_PATH, device=None):
        if not checkpoint.exists() or not preprocessor_path.exists():
            raise ModelNotAvailableError(
                f"Trained model not found ({checkpoint} / {preprocessor_path}). "
                "Train it first with:  python -m src.train")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.prep = Preprocessor.load(preprocessor_path)
        if ckpt["feature_cols"] != self.prep.feature_cols:
            raise ModelNotAvailableError("Checkpoint and preprocessor feature lists differ; retrain the model.")
        self.seq_len = ckpt["seq_len"]
        self.rul_scale = ckpt["rul_scale"]
        self.model = LSTMRegressor(ckpt["n_features"], hidden=ckpt["hidden"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.info = {"epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"], "seq_len": self.seq_len,
                     "device": str(self.device), "features": len(self.prep.feature_cols)}

    # ---- core ----
    def _windows(self, t: Table, pad_all: bool):
        t = sort_by(t, "cycle") if "cycle" in t else t
        return windows_for_engine(self.prep.features(t), self.seq_len, pad=pad_all)

    @torch.no_grad()
    def _predict_windows(self, W: np.ndarray, mc_samples: int = 0, batch: int = 1024):
        """Returns (mean_rul, std_rul). mean is the deterministic (dropout-off) prediction; std is the
        MC-dropout spread (0 if mc_samples == 0), see `_mc_dropout_std`."""
        x = torch.from_numpy(W).to(self.device)
        means = [self.model(x[i:i + batch]).cpu().numpy() for i in range(0, len(x), batch)]
        mean = np.clip(np.concatenate(means) * self.rul_scale, 0, None)
        if mc_samples > 0:
            std = np.array([self._mc_dropout_std(x[i:i + 1], mc_samples) for i in range(len(x))], dtype=np.float32)
        else:
            std = np.zeros(len(x), dtype=np.float32)
        return mean, std * self.rul_scale

    @torch.no_grad()
    def _mc_dropout_std(self, x1: torch.Tensor, n: int, seed: int = C.MC_DROPOUT_SEED) -> float:
        """MC-dropout spread for ONE window (normalised RUL units).

        The LSTM runs once in eval mode (its inter-layer dropout is inactive at inference, as before); the FC
        head is then evaluated `n` times with independent Bernoulli dropout masks (keep-prob 1-p, inverted
        scaling, same p as training). Masks come from a local generator seeded with `seed`, so identical
        inputs give identical results, the global RNG is untouched, and the shared model is never switched
        to train mode (safe under concurrent API requests). Returns the sample std (ddof=1).
        """
        h = self.model.lstm(x1)[0][:, -1].expand(n, -1)
        gen = torch.Generator(device=x1.device).manual_seed(seed)
        for layer in self.model.head:
            if isinstance(layer, torch.nn.Dropout):
                keep = 1.0 - layer.p
                h = h * torch.bernoulli(torch.full_like(h, keep), generator=gen) / keep
            else:
                h = layer(h)
        return float(h.squeeze(-1).std())

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
