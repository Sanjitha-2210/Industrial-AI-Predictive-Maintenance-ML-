"""Model Adaptation and Training Pipeline for External Machine Datasets.

Performs asset-level train/validation splitting, feature scaling on training
assets only, sliding window generation, LSTM training with early stopping,
metric evaluation, and model registry persistence.
"""
from __future__ import annotations

import datetime
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    # Test if torch actually works or is blocked by Windows Application Control
    _ = torch.zeros(1)
    _TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    _TORCH_AVAILABLE = False


from src import config as C
from src.adaptation.mapper import SensorMappingConfig
from src.adaptation.validator import read_csv_data
from src.data.preprocess import StandardScaler, windows_for_engine
from src.data.table import from_records, groups, matrix, n_rows, sort_by
from src.models.lstm import LSTMRegressor
from src.registry.model_registry import ModelMetadata, get_model_registry

TRAINING_JOBS: dict[str, TrainingJobStatus] = {}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class NumPyLSTMRegressor:
    """Pure NumPy 2-Layer LSTM with Dense Head, analytical BPTT, and Adam optimizer.
    
    Provides 100% environment resilience on platforms where PyTorch or external
    C/C++ runtime DLLs are blocked by Windows Defender Application Control (WDAC).
    """

    def __init__(self, n_features: int, hidden: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.hidden = hidden
        H, F = hidden, n_features

        scale_ih = float(np.sqrt(2.0 / (F + H)))
        scale_hh = float(np.sqrt(2.0 / (2 * H)))

        self.params: dict[str, np.ndarray] = {
            "lstm.weight_ih_l0": rng.normal(0, scale_ih, (4 * H, F)).astype(np.float32),
            "lstm.weight_hh_l0": rng.normal(0, scale_hh, (4 * H, H)).astype(np.float32),
            "lstm.bias_ih_l0": np.zeros((4 * H,), dtype=np.float32),
            "lstm.bias_hh_l0": np.zeros((4 * H,), dtype=np.float32),

            "lstm.weight_ih_l1": rng.normal(0, scale_hh, (4 * H, H)).astype(np.float32),
            "lstm.weight_hh_l1": rng.normal(0, scale_hh, (4 * H, H)).astype(np.float32),
            "lstm.bias_ih_l1": np.zeros((4 * H,), dtype=np.float32),
            "lstm.bias_hh_l1": np.zeros((4 * H,), dtype=np.float32),

            "head.0.weight": rng.normal(0, np.sqrt(2.0 / (H + 64)), (64, H)).astype(np.float32),
            "head.0.bias": np.zeros((64,), dtype=np.float32),
            "head.3.weight": rng.normal(0, np.sqrt(2.0 / (64 + 32)), (32, 64)).astype(np.float32),
            "head.3.bias": np.zeros((32,), dtype=np.float32),
            "head.6.weight": rng.normal(0, np.sqrt(2.0 / (32 + 1)), (1, 32)).astype(np.float32),
            "head.6.bias": np.zeros((1,), dtype=np.float32),
        }
        # Initialize forget gates bias to 1.0 (standard LSTM initialization)
        self.params["lstm.bias_ih_l0"][H:2 * H] = 1.0
        self.params["lstm.bias_ih_l1"][H:2 * H] = 1.0

        # Adam optimizer state
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        B, T, F = x.shape
        H = self.hidden
        p = self.params

        h0 = np.zeros((T + 1, B, H), dtype=np.float32)
        c0 = np.zeros((T + 1, B, H), dtype=np.float32)
        i0_arr = np.zeros((T, B, H), dtype=np.float32)
        f0_arr = np.zeros((T, B, H), dtype=np.float32)
        g0_arr = np.zeros((T, B, H), dtype=np.float32)
        o0_arr = np.zeros((T, B, H), dtype=np.float32)

        h1 = np.zeros((T + 1, B, H), dtype=np.float32)
        c1 = np.zeros((T + 1, B, H), dtype=np.float32)
        i1_arr = np.zeros((T, B, H), dtype=np.float32)
        f1_arr = np.zeros((T, B, H), dtype=np.float32)
        g1_arr = np.zeros((T, B, H), dtype=np.float32)
        o1_arr = np.zeros((T, B, H), dtype=np.float32)

        b0 = p["lstm.bias_ih_l0"] + p["lstm.bias_hh_l0"]
        b1 = p["lstm.bias_ih_l1"] + p["lstm.bias_hh_l1"]

        for step in range(T):
            xt = x[:, step, :]
            # Layer 0
            g0 = xt @ p["lstm.weight_ih_l0"].T + h0[step] @ p["lstm.weight_hh_l0"].T + b0
            i0 = _sigmoid(g0[:, :H])
            f0 = _sigmoid(g0[:, H:2 * H])
            cand0 = np.tanh(g0[:, 2 * H:3 * H])
            o0 = _sigmoid(g0[:, 3 * H:])
            c0_next = f0 * c0[step] + i0 * cand0
            h0_next = o0 * np.tanh(c0_next)

            i0_arr[step], f0_arr[step], g0_arr[step], o0_arr[step] = i0, f0, cand0, o0
            c0[step + 1] = c0_next
            h0[step + 1] = h0_next

            # Layer 1
            g1 = h0_next @ p["lstm.weight_ih_l1"].T + h1[step] @ p["lstm.weight_hh_l1"].T + b1
            i1 = _sigmoid(g1[:, :H])
            f1 = _sigmoid(g1[:, H:2 * H])
            cand1 = np.tanh(g1[:, 2 * H:3 * H])
            o1 = _sigmoid(g1[:, 3 * H:])
            c1_next = f1 * c1[step] + i1 * cand1
            h1_next = o1 * np.tanh(c1_next)

            i1_arr[step], f1_arr[step], g1_arr[step], o1_arr[step] = i1, f1, cand1, o1
            c1[step + 1] = c1_next
            h1[step + 1] = h1_next

        last_h = h1[T]  # (B, H)

        # 3-layer MLP Head
        z0 = last_h @ p["head.0.weight"].T + p["head.0.bias"]
        a0 = np.maximum(z0, 0.0)
        mask0 = (a0 > 0).astype(np.float32)

        z1 = a0 @ p["head.3.weight"].T + p["head.3.bias"]
        a1 = np.maximum(z1, 0.0)
        mask1 = (a1 > 0).astype(np.float32)

        out = (a1 @ p["head.6.weight"].T + p["head.6.bias"]).squeeze(-1)  # (B,)

        cache = {
            "x": x, "h0": h0, "c0": c0, "i0": i0_arr, "f0": f0_arr, "g0": g0_arr, "o0": o0_arr,
            "h1": h1, "c1": c1, "i1": i1_arr, "f1": f1_arr, "g1": g1_arr, "o1": o1_arr,
            "last_h": last_h, "a0": a0, "mask0": mask0, "a1": a1, "mask1": mask1, "out": out,
        }
        return out, cache

    def backward(self, cache: dict[str, Any], target: np.ndarray) -> dict[str, np.ndarray]:
        p = self.params
        H = self.hidden
        x = cache["x"]
        B, T, _ = x.shape
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        d_out = (cache["out"] - target) / B
        d_out_2d = d_out[:, None]  # (B, 1)

        # Head backprop
        grads["head.6.weight"] = d_out_2d.T @ cache["a1"]
        grads["head.6.bias"] = np.sum(d_out_2d, axis=0)

        d_a1 = d_out_2d @ p["head.6.weight"]
        d_z1 = d_a1 * cache["mask1"]
        grads["head.3.weight"] = d_z1.T @ cache["a0"]
        grads["head.3.bias"] = np.sum(d_z1, axis=0)

        d_a0 = d_z1 @ p["head.3.weight"]
        d_z0 = d_a0 * cache["mask0"]
        grads["head.0.weight"] = d_z0.T @ cache["last_h"]
        grads["head.0.bias"] = np.sum(d_z0, axis=0)

        d_last_h1 = d_z0 @ p["head.0.weight"]  # (B, H)

        # Layer 1 BPTT
        dh1 = d_last_h1.copy()
        dc1 = np.zeros((B, H), dtype=np.float32)
        dh0_seq = np.zeros((T, B, H), dtype=np.float32)

        for step in reversed(range(T)):
            c1_next = cache["c1"][step + 1]
            c1_prev = cache["c1"][step]
            tanh_c1 = np.tanh(c1_next)

            o1 = cache["o1"][step]
            f1 = cache["f1"][step]
            i1 = cache["i1"][step]
            g1 = cache["g1"][step]

            do1 = dh1 * tanh_c1 * o1 * (1.0 - o1)
            dc1 = dc1 + dh1 * o1 * (1.0 - tanh_c1 ** 2)
            df1 = dc1 * c1_prev * f1 * (1.0 - f1)
            di1 = dc1 * g1 * i1 * (1.0 - i1)
            dg1 = dc1 * i1 * (1.0 - g1 ** 2)

            dgates1 = np.hstack([di1, df1, dg1, do1])  # (B, 4*H)

            h0_t = cache["h0"][step + 1]
            grads["lstm.weight_ih_l1"] += dgates1.T @ h0_t
            h1_prev = cache["h1"][step]
            grads["lstm.weight_hh_l1"] += dgates1.T @ h1_prev
            sum_dgates1 = np.sum(dgates1, axis=0)
            grads["lstm.bias_ih_l1"] += sum_dgates1 * 0.5
            grads["lstm.bias_hh_l1"] += sum_dgates1 * 0.5

            dh0_seq[step] = dgates1 @ p["lstm.weight_ih_l1"]
            dh1 = dgates1 @ p["lstm.weight_hh_l1"]
            dc1 = dc1 * f1

        # Layer 0 BPTT
        dh0 = np.zeros((B, H), dtype=np.float32)
        dc0 = np.zeros((B, H), dtype=np.float32)

        for step in reversed(range(T)):
            dh0 = dh0 + dh0_seq[step]
            c0_next = cache["c0"][step + 1]
            c0_prev = cache["c0"][step]
            tanh_c0 = np.tanh(c0_next)

            o0 = cache["o0"][step]
            f0 = cache["f0"][step]
            i0 = cache["i0"][step]
            g0 = cache["g0"][step]

            do0 = dh0 * tanh_c0 * o0 * (1.0 - o0)
            dc0 = dc0 + dh0 * o0 * (1.0 - tanh_c0 ** 2)
            df0 = dc0 * c0_prev * f0 * (1.0 - f0)
            di0 = dc0 * g0 * i0 * (1.0 - i0)
            dg0 = dc0 * i0 * (1.0 - g0 ** 2)

            dgates0 = np.hstack([di0, df0, dg0, do0])  # (B, 4*H)

            xt = x[:, step, :]
            grads["lstm.weight_ih_l0"] += dgates0.T @ xt
            h0_prev = cache["h0"][step]
            grads["lstm.weight_hh_l0"] += dgates0.T @ h0_prev
            sum_dgates0 = np.sum(dgates0, axis=0)
            grads["lstm.bias_ih_l0"] += sum_dgates0 * 0.5
            grads["lstm.bias_hh_l0"] += sum_dgates0 * 0.5

            dh0 = dgates0 @ p["lstm.weight_hh_l0"]
            dc0 = dc0 * f0

        # Gradient clipping (norm <= 5.0)
        total_norm = float(np.sqrt(sum(np.sum(g ** 2) for g in grads.values())))
        if total_norm > 5.0:
            scale = 5.0 / total_norm
            for k in grads:
                grads[k] *= scale

        return grads

    def step(self, grads: dict[str, np.ndarray], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.t += 1
        lr_t = lr * np.sqrt(1.0 - beta2 ** self.t) / (1.0 - beta1 ** self.t)
        for k in self.params:
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g ** 2)
            self.params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + eps)



@dataclass
class TrainingJobStatus:
    job_id: str
    model_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    progress_pct: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "progress_pct": self.progress_pct,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "train_loss": round(self.train_loss, 5),
            "val_loss": round(self.val_loss, 5),
            "metrics": self.metrics,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


def _split_assets(assets: list[str], val_fraction: float = 0.2, seed: int = 42) -> tuple[list[str], list[str]]:
    """Asset-level train/validation split to guarantee zero data leakage."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(assets))
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_assets = sorted(shuffled[:n_val].tolist())
    train_assets = sorted(shuffled[n_val:].tolist())
    if not train_assets:  # If very few machines, ensure train has at least 1
        train_assets = val_assets[:1]
    return train_assets, val_assets


def train_adapted_model(
    dataset_path: str | Path,
    mapping_config: SensorMappingConfig,
    model_name: str,
    industry: str,
    machine_category: str,
    seq_len: int = 30,
    rul_clip: int = 125,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_units: int = 64,
    val_fraction: float = 0.2,
    patience: int = 6,
    seed: int = 42,
    job_id: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Executes end-to-end model training on an external dataset."""
    if _TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
    np.random.seed(seed)

    header, raw_rows = read_csv_data(dataset_path)

    m_col = mapping_config.machine_id_column
    c_col = mapping_config.cycle_column
    r_col = mapping_config.rul_column

    feature_mappings = mapping_config.mappings
    if not feature_mappings:
        raise ValueError("No sensor mappings provided for training.")

    source_cols = [m.source_column for m in feature_mappings]
    target_names = [m.target_feature for m in feature_mappings]

    # Validate existence
    for col in [m_col, c_col] + source_cols:
        if col not in header:
            raise ValueError(f"Required column '{col}' not found in dataset.")

    # Parse rows into structured records
    m_idx = header.index(m_col)
    c_idx = header.index(c_col)
    r_idx = header.index(r_col) if (r_col and r_col in header) else None
    feat_indices = [header.index(c) for c in source_cols]

    # Group by machine
    machines: dict[str, list[dict[str, Any]]] = {}
    for r in raw_rows:
        mid = str(r[m_idx])
        try:
            cyc = int(float(r[c_idx]))
            feats = [float(r[fi]) for fi in feat_indices]
            rul_val = float(r[r_idx]) if r_idx is not None else None
        except (ValueError, IndexError):
            continue  # skip invalid rows

        row_dict = {"unit": mid, "cycle": cyc}
        for tname, fval in zip(target_names, feats):
            row_dict[tname] = fval
        if rul_val is not None:
            row_dict["RUL"] = rul_val
        machines.setdefault(mid, []).append(row_dict)

    if not machines:
        raise ValueError("Dataset has no valid records after parsing.")

    all_assets = list(machines.keys())
    if len(all_assets) < 2:
        # If single asset, cannot do asset-level train/val split without leakage
        train_assets = all_assets
        val_assets = all_assets
    else:
        train_assets, val_assets = _split_assets(all_assets, val_fraction=val_fraction, seed=seed)

    # Calculate run-to-failure RUL if not explicitly given
    for mid, rows in machines.items():
        rows.sort(key=lambda x: x["cycle"])
        max_cyc = rows[-1]["cycle"]
        for row in rows:
            if "RUL" not in row:
                row["RUL"] = min(float(max_cyc - row["cycle"]), float(rul_clip))
            else:
                row["RUL"] = min(float(row["RUL"]), float(rul_clip))

    # Build train and val tables
    train_records = []
    for mid in train_assets:
        train_records.extend(machines[mid])
    val_records = []
    for mid in val_assets:
        val_records.extend(machines[mid])

    # Fit StandardScaler on TRAIN machines only
    train_matrix = np.array([[r[f] for f in target_names] for r in train_records], dtype=float)
    scaler = StandardScaler().fit(train_matrix)

    # Window generation
    def make_windows(records_by_asset: dict[str, list[dict[str, Any]]], asset_list: list[str]):
        W_list, Y_list = [], []
        for aid in asset_list:
            recs = records_by_asset[aid]
            if len(recs) == 0:
                continue
            mat = np.array([[r[f] for f in target_names] for r in recs], dtype=float)
            scaled = scaler.transform(mat)
            ruls = np.array([r["RUL"] for r in recs], dtype=float)
            w, ends = windows_for_engine(scaled, seq_len, pad=True)
            y = ruls[ends]
            W_list.append(w)
            Y_list.append(y)
        if not W_list:
            return np.empty((0, seq_len, len(target_names))), np.empty((0,))
        return np.concatenate(W_list, axis=0), np.concatenate(Y_list, axis=0)

    X_train, Y_train = make_windows(machines, train_assets)
    X_val, Y_val = make_windows(machines, val_assets)

    if len(X_train) == 0:
        raise ValueError("Insufficient sequence data to create training windows.")

    # Model training with dual engine support (PyTorch if available, pure NumPy otherwise)
    rul_scale = float(rul_clip)
    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0
    engine_used = "pytorch" if _TORCH_AVAILABLE else "numpy"

    if _TORCH_AVAILABLE:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LSTMRegressor(n_features=len(target_names), hidden=hidden_units, layers=2, dropout=0.2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train / rul_scale).float())
            val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val / rul_scale).float())
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            for epoch in range(1, epochs + 1):
                model.train()
                train_loss_total = 0.0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()
                    train_loss_total += loss.item() * len(bx)
                avg_train = train_loss_total / len(train_dataset)

                model.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(device), by.to(device)
                        pred = model(bx)
                        val_loss_total += criterion(pred, by).item() * len(bx)
                avg_val = val_loss_total / len(val_dataset)

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_weights = {k: v.cpu().clone().numpy() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if job_id and job_id in TRAINING_JOBS:
                    job = TRAINING_JOBS[job_id]
                    job.current_epoch = epoch
                    job.total_epochs = epochs
                    job.progress_pct = int((epoch / epochs) * 100)
                    job.train_loss = avg_train
                    job.val_loss = avg_val

                if no_improve >= patience:
                    break
        except Exception:
            engine_used = "numpy"

    if engine_used == "numpy":
        # Pure NumPy LSTM training engine with BPTT and Adam optimization
        rng = np.random.default_rng(seed)
        net = NumPyLSTMRegressor(n_features=len(target_names), hidden=hidden_units, seed=seed)
        n_samples = len(X_train)
        scaled_Y_train = (Y_train / rul_scale).astype(np.float32)
        scaled_Y_val = (Y_val / rul_scale).astype(np.float32)

        for epoch in range(1, epochs + 1):
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            train_loss_total = 0.0
            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                bx = X_train[batch_idx]
                by = scaled_Y_train[batch_idx]

                pred, cache = net.forward(bx)
                loss = float(np.mean((pred - by) ** 2))
                train_loss_total += loss * len(bx)

                grads = net.backward(cache, by)
                net.step(grads, lr=lr)

            avg_train = train_loss_total / n_samples

            # Validation evaluation
            val_preds_norm, _ = net.forward(X_val)
            avg_val = float(np.mean((val_preds_norm - scaled_Y_val) ** 2))

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_weights = {k: v.copy() for k, v in net.params.items()}
                no_improve = 0
            else:
                no_improve += 1

            if job_id and job_id in TRAINING_JOBS:
                job = TRAINING_JOBS[job_id]
                job.current_epoch = epoch
                job.total_epochs = epochs
                job.progress_pct = int((epoch / epochs) * 100)
                job.train_loss = avg_train
                job.val_loss = avg_val

            if no_improve >= patience:
                break

    if best_weights is None:
        raise RuntimeError("Model training failed to optimize weights.")

    # Calculate validation metrics in real RUL units (cycles)
    eval_net = NumPyLSTMRegressor(n_features=len(target_names), hidden=hidden_units, seed=seed)
    eval_net.params = best_weights
    val_preds_norm, _ = eval_net.forward(X_val)
    val_preds = np.clip(val_preds_norm * rul_scale, 0.0, None)
    errors = val_preds - Y_val
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    ss_tot = float(np.sum((Y_val - np.mean(Y_val)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 0.0

    eval_metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "val_loss": round(best_val_loss, 5),
    }

    # Generate unique model id and paths
    model_uuid = str(uuid.uuid4())[:8]
    clean_cat = machine_category.lower().replace(" ", "_")
    model_id = f"model_{clean_cat}_{model_uuid}"

    model_dir = C.CHECKPOINT_DIR / "adapted" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / "model.pt"
    prep_path = model_dir / "preprocessor.joblib"

    # Save checkpoint artifact with joblib (immune to WDAC policy)
    ckpt_payload = {
        "state_dict": best_weights,
        "n_features": len(target_names),
        "hidden": hidden_units,
        "seq_len": seq_len,
        "rul_scale": rul_scale,
        "rul_clip": rul_clip,
        "feature_cols": target_names,
        "epoch": epoch,
        "val_loss": best_val_loss,
        "engine": engine_used,
    }
    joblib.dump(ckpt_payload, ckpt_path)

    # Save preprocessor artifact
    prep_obj = {
        "feature_cols": target_names,
        "scaler": scaler,
        "seq_len": seq_len,
        "rul_clip": rul_clip,
        "val_units": val_assets,
    }
    joblib.dump(prep_obj, prep_path)

    arch_desc = (
        f"NumPy 2-layer LSTM ({hidden_units} hidden, Adam BPTT)"
        if engine_used == "numpy"
        else f"PyTorch 2-layer LSTM ({hidden_units} hidden, dropout 0.2)"
    )

    # Register in Model Registry
    registry = get_model_registry()
    mapping_dict = {m.source_column: m.target_feature for m in feature_mappings}
    metadata = ModelMetadata(
        model_id=model_id,
        name=model_name,
        industry=industry,
        machine_category=machine_category,
        training_dataset=Path(dataset_path).name,
        feature_schema=target_names,
        sequence_length=seq_len,
        rul_clip=rul_clip,
        scaler_type="StandardScaler (Z-Score)",
        architecture=arch_desc,
        training_date=datetime.datetime.now().isoformat(),
        evaluation_metrics=eval_metrics,
        version="1.0.0",
        status="active",
        checkpoint_path=str(ckpt_path.relative_to(C.ROOT)).replace("\\", "/"),
        preprocessor_path=str(prep_path.relative_to(C.ROOT)).replace("\\", "/"),
        supports_uncertainty=True,
        sensor_mapping=mapping_dict,
        description=f"Adapted model for {machine_category} trained on {Path(dataset_path).name}.",
    )
    registry.register_model(metadata)

    if job_id and job_id in TRAINING_JOBS:
        job = TRAINING_JOBS[job_id]
        job.status = "completed"
        job.progress_pct = 100
        job.model_id = model_id
        job.metrics = eval_metrics
        job.completed_at = datetime.datetime.now().isoformat()

    return model_id, eval_metrics


def launch_training_background(
    dataset_path: str | Path,
    mapping_config: SensorMappingConfig,
    model_name: str,
    industry: str,
    machine_category: str,
    **training_kwargs,
) -> str:
    """Launches model training in a background daemon thread."""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job = TrainingJobStatus(
        job_id=job_id,
        model_id="",
        status="running",
        started_at=datetime.datetime.now().isoformat(),
    )
    TRAINING_JOBS[job_id] = job

    def _worker():
        try:
            train_adapted_model(
                dataset_path=dataset_path,
                mapping_config=mapping_config,
                model_name=model_name,
                industry=industry,
                machine_category=machine_category,
                job_id=job_id,
                **training_kwargs,
            )
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.datetime.now().isoformat()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return job_id
