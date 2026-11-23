"""Unified Multi-Industry Prediction Engine.

Dispatches inference across default NASA benchmark models or any custom adapted model
from the Model Registry. Enforces strict schema validation, handles sequence padding,
and produces standardized RUL predictions with uncertainty and audit metadata.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False


from src import config as C
from src.data.preprocess import StandardScaler, windows_for_engine
from src.data.table import Table, from_records, n_rows, sort_by
from src.models.lstm import LSTMRegressor
from src.predict import Predictor, _run_numpy_lstm
from src.registry.model_registry import ModelMetadata, get_model_registry

HISTORY_PATH = C.OUTPUT_DIR / "prediction_history.json"


class IncompatibleSchemaError(ValueError):
    pass


@dataclass
class UniversalPredictionResult:
    model_id: str
    model_name: str
    model_version: str
    industry: str
    machine_category: str
    machine_id: str | None
    latest_cycle: int
    predicted_rul: float
    uncertainty: float | None
    uncertainty_method: str
    cycles_used: int
    window_padded: bool
    sensor_trends: list[dict[str, Any]]
    explanation: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UniversalPredictionEngine:
    """Multi-model inference engine supporting NASA and adapted models."""

    def __init__(self):
        self.registry = get_model_registry()
        self._loaded_models: dict[str, Any] = {}
        self._history: list[dict[str, Any]] = self._load_history()

    def _load_history(self) -> list[dict[str, Any]]:
        if HISTORY_PATH.exists():
            try:
                return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))[-200:]
            except Exception:
                return []
        return []

    def _save_history(self):
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            HISTORY_PATH.write_text(json.dumps(self._history[-200:], indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(reversed(self._history))[:limit]

    def _load_adapted_model(self, meta: ModelMetadata):
        """Loads weights and preprocessor for an adapted model (pure NumPy or PyTorch)."""
        if meta.model_id in self._loaded_models:
            return self._loaded_models[meta.model_id]

        ckpt_file = C.ROOT / meta.checkpoint_path
        prep_file = C.ROOT / meta.preprocessor_path

        if not ckpt_file.exists() or not prep_file.exists():
            raise FileNotFoundError(f"Model artifacts missing for '{meta.model_id}'.")

        # Load checkpoint dictionary with joblib first (or torch fallback)
        ckpt = None
        try:
            ckpt = joblib.load(ckpt_file)
        except Exception:
            if _TORCH_AVAILABLE:
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
                except Exception:
                    pass
        if ckpt is None:
            raise RuntimeError(f"Could not load checkpoint for model '{meta.model_id}'.")

        prep_data = joblib.load(prep_file)

        model = None
        device = "cpu"
        if _TORCH_AVAILABLE and ckpt.get("engine") == "pytorch":
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = LSTMRegressor(ckpt["n_features"], hidden=ckpt.get("hidden", 64)).to(device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
            except Exception:
                model = None

        bundle = {
            "model": model,
            "state_dict": ckpt["state_dict"],
            "hidden": ckpt.get("hidden", 64),
            "scaler": prep_data["scaler"] if isinstance(prep_data, dict) else prep_data.scaler,
            "feature_cols": meta.feature_schema,
            "seq_len": meta.sequence_length,
            "rul_scale": float(meta.rul_clip),
            "device": device,
        }
        self._loaded_models[meta.model_id] = bundle
        return bundle

    def predict(
        self,
        table: Table,
        model_id: str = "nasa_cmapss_lstm_v1",
        machine_id: str | int | None = None,
        column_mapping: dict[str, str] | None = None,
        mc_samples: int = 30,
    ) -> UniversalPredictionResult:
        """Runs RUL prediction for a sensor table using the specified model."""
        meta = self.registry.get_model(model_id)

        # Apply column mapping if provided
        mapped_table = dict(table)
        if column_mapping:
            for src_col, tgt_col in column_mapping.items():
                if src_col in table:
                    mapped_table[tgt_col] = table[src_col]

        # Auto-coerce numeric feature columns stored as strings/objects to float
        for col in list(mapped_table.keys()):
            arr = mapped_table[col]
            if hasattr(arr, "dtype") and arr.dtype.kind in ("U", "S", "O"):
                try:
                    mapped_table[col] = arr.astype(np.float64)
                except (ValueError, TypeError):
                    pass


        # Verify feature schema compatibility
        missing = [f for f in meta.feature_schema if f not in mapped_table]
        if missing:
            raise IncompatibleSchemaError(
                f"Dataset is incompatible with model '{meta.name}'. "
                f"Missing required features: {missing}. "
                f"Verify sensor mapping or select a model trained on this feature schema."
            )

        # Handle NASA default model
        if model_id == "nasa_cmapss_lstm_v1":
            predictor = Predictor()
            # Predict last cycle
            pred = predictor.predict_last(mapped_table, mc_samples=mc_samples)
            latest_cyc = int(mapped_table["cycle"][-1]) if "cycle" in mapped_table and len(mapped_table["cycle"]) else n_rows(mapped_table)
            res = UniversalPredictionResult(
                model_id=meta.model_id,
                model_name=meta.name,
                model_version=meta.version,
                industry=meta.industry,
                machine_category=meta.machine_category,
                machine_id=str(machine_id) if machine_id is not None else None,
                latest_cycle=latest_cyc,
                predicted_rul=round(float(pred["predicted_rul"]), 2),
                uncertainty=round(float(pred["uncertainty"]), 2) if "uncertainty" in pred else None,
                uncertainty_method=pred.get("uncertainty_method", "MC-Dropout"),
                cycles_used=pred.get("cycles_used", n_rows(mapped_table)),
                window_padded=pred.get("window_padded", False),
                sensor_trends=pred.get("sensor_trends", []),
                explanation=pred.get("explanation", f"Predicted Remaining Useful Life: {pred['predicted_rul']:.1f} cycles."),
                timestamp=datetime.datetime.now().isoformat(),
            )
            self._history.append(res.to_dict())
            self._save_history()
            return res

        # Handle adapted models
        bundle = self._load_adapted_model(meta)
        model = bundle["model"]
        scaler = bundle["scaler"]
        feats = bundle["feature_cols"]
        seq_len = bundle["seq_len"]
        rul_scale = bundle["rul_scale"]
        dev = bundle["device"]

        # Sort by cycle if present
        sorted_tab = sort_by(mapped_table, "cycle") if "cycle" in mapped_table else mapped_table
        num_rows = n_rows(sorted_tab)
        if num_rows == 0:
            raise ValueError("No sensor readings supplied.")

        # Extract feature matrix
        raw_mat = np.column_stack([np.asarray(sorted_tab[f], dtype=float) for f in feats])
        scaled_mat = scaler.transform(raw_mat)

        windows, _ = windows_for_engine(scaled_mat, seq_len, pad=True)
        last_window = windows[-1:]  # (1, seq_len, n_features)

        # Deterministic prediction (supports pure NumPy and PyTorch backends)
        if bundle.get("model") is not None and _TORCH_AVAILABLE:
            x_tensor = torch.from_numpy(last_window).float().to(bundle["device"])
            with torch.no_grad():
                pred_norm = bundle["model"](x_tensor).cpu().numpy().item()
                pred_rul = max(0.0, float(pred_norm * rul_scale))
        else:
            mean_pred, _ = _run_numpy_lstm(
                bundle["state_dict"], last_window, hidden=bundle["hidden"], rul_scale=rul_scale
            )
            pred_rul = float(mean_pred[0])

        # MC-Dropout uncertainty
        unc = 0.0
        if mc_samples > 0:
            if bundle.get("model") is not None and _TORCH_AVAILABLE:
                gen = torch.Generator(device=x_tensor.device).manual_seed(C.MC_DROPOUT_SEED)
                h = bundle["model"].lstm(x_tensor)[0][:, -1].expand(mc_samples, -1)
                for layer in bundle["model"].head:
                    if isinstance(layer, torch.nn.Dropout):
                        keep = 1.0 - layer.p
                        h = h * torch.bernoulli(torch.full_like(h, keep), generator=gen) / keep
                    else:
                        h = layer(h)
                unc = float(h.squeeze(-1).std().item() * rul_scale)
            else:
                sd = bundle["state_dict"]
                _, last_h = _run_numpy_lstm(sd, last_window, hidden=bundle["hidden"], rul_scale=1.0)
                rng = np.random.default_rng(C.MC_DROPOUT_SEED)
                h = np.repeat(last_h, mc_samples, axis=0)  # (mc_samples, hidden)
                keep = 0.8
                m0 = (rng.random(h.shape) < keep).astype(np.float32) / keep
                z0 = np.maximum((h * m0) @ sd["head.0.weight"].T + sd["head.0.bias"], 0.0)
                m1 = (rng.random(z0.shape) < keep).astype(np.float32) / keep
                z1 = np.maximum((z0 * m1) @ sd["head.3.weight"].T + sd["head.3.bias"], 0.0)
                out = (z1 @ sd["head.6.weight"].T + sd["head.6.bias"]).squeeze(-1)
                unc = float(np.std(np.clip(out * rul_scale, 0.0, None)))


        # Trends on primary features
        trends = []
        for i, fname in enumerate(feats[:6]):
            series = raw_mat[:, i]
            delta = float(series[-1] - series[0]) if len(series) > 1 else 0.0
            trends.append({
                "feature": fname,
                "current": round(float(series[-1]), 3),
                "delta": round(delta, 3),
                "direction": "up" if delta > 0.01 else ("down" if delta < -0.01 else "stable"),
            })

        latest_cyc = int(sorted_tab["cycle"][-1]) if "cycle" in sorted_tab else num_rows
        padded = num_rows < seq_len
        explanation = (
            f"Model '{meta.name}' predicts {pred_rul:.1f} cycles of Remaining Useful Life "
            f"(uncertainty ±{unc:.1f} cycles). Based on {min(num_rows, seq_len)} observed cycles."
        )

        res = UniversalPredictionResult(
            model_id=meta.model_id,
            model_name=meta.name,
            model_version=meta.version,
            industry=meta.industry,
            machine_category=meta.machine_category,
            machine_id=str(machine_id) if machine_id is not None else None,
            latest_cycle=latest_cyc,
            predicted_rul=round(pred_rul, 2),
            uncertainty=round(unc, 2) if mc_samples > 0 else None,
            uncertainty_method=f"MC-dropout std over {mc_samples} masks" if mc_samples > 0 else "None",
            cycles_used=num_rows,
            window_padded=padded,
            sensor_trends=trends,
            explanation=explanation,
            timestamp=datetime.datetime.now().isoformat(),
        )
        self._history.append(res.to_dict())
        self._save_history()
        return res


_ENGINE_INSTANCE: UniversalPredictionEngine | None = None


def get_prediction_engine() -> UniversalPredictionEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = UniversalPredictionEngine()
    return _ENGINE_INSTANCE
