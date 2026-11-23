"""Training / prediction for DATF-Net and fresh LSTM comparison runs (separate from src/train.py).

Training conditions are identical to src/train.py for every model: same windows and split (saved preprocessor,
read-only), targets RUL/rul_clip, MSE main loss, Adam lr 1e-3, batch 256, ReduceLROnPlateau(0.5, patience 4),
grad-norm clip 1.0, early stopping on validation MSE (patience 10, min improvement 1e-6), max 60 epochs,
seeding order torch.manual_seed -> np.random.seed -> DataLoaders -> model init.
DATF-Net adds auxiliary HI losses to the training objective; early stopping and LR scheduling use the main
validation MSE only, exactly as for the LSTM. The validation engines are never trained on; test data is not used here.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.datf_net import DATFConfig, DATFNet, count_parameters, hi_supervision_loss, monotonicity_loss
from src.models.lstm import LSTMRegressor
from src.models.lstm_control import BaselineInfoLSTM


@dataclass
class TrainSettings:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 10
    lambda_hi: float = 0.0
    lambda_mono: float = 0.0
    grad_clip: float = 1.0
    deterministic: bool = True

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class RunSpec:
    """model_type 'lstm' (baseline architecture, hidden 64), 'lstm_info' (same LSTM fed [x, x - b]: information-
    matched control) or 'datf' (with DATFConfig overrides)."""
    name: str
    model_type: str
    datf: dict = field(default_factory=dict)
    lambda_hi: float | None = None
    lambda_mono: float | None = None


def build_model(spec: RunSpec, n_features: int, seq_len: int) -> nn.Module:
    if spec.model_type == "lstm":
        return LSTMRegressor(n_features, hidden=64)
    if spec.model_type == "lstm_info":
        return BaselineInfoLSTM(n_features, hidden=64)
    if spec.model_type == "datf":
        return DATFNet(DATFConfig(n_features=n_features, seq_len=seq_len, **spec.datf))
    raise ValueError(spec.model_type)


def forward(model, xb, bb):
    if isinstance(model, DATFNet):
        return model(xb, bb)
    if getattr(model, "uses_baseline", False):
        return {"rul": model(xb, bb), "hi": None}
    return {"rul": model(xb), "hi": None}


def set_determinism(seed: int, deterministic: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def train_run(spec: RunSpec, train: dict, val: dict, seed: int, settings: TrainSettings, clip: float,
              ckpt_path: Path, device: torch.device, extra_meta: dict | None = None, log=print) -> dict:
    set_determinism(seed, settings.deterministic)
    scale = float(clip)
    mk = lambda d, shuffle: DataLoader(  # noqa: E731 - same construction as src/train.py (+ b, hi_target)
        TensorDataset(torch.from_numpy(d["X"]), torch.from_numpy(d["y"] / scale),
                      torch.from_numpy(d["b"]), torch.from_numpy(d["hi_target"])),
        batch_size=settings.batch_size, shuffle=shuffle)
    tr_loader, va_loader = mk(train, True), mk(val, False)
    model = build_model(spec, train["X"].shape[2], train["X"].shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=settings.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    mse = nn.MSELoss()
    use_aux = isinstance(model, DATFNet) and model.cfg.use_degradation

    hist = {"train_loss": [], "train_total_loss": [], "val_loss": [], "val_hi_loss": [], "val_rmse_cycles": [], "lr": []}
    best, bad, best_epoch = float("inf"), 0, 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(1, settings.epochs + 1):
        model.train()
        tot_main = tot_all = n = 0.0
        for xb, yb, bb, hb in tr_loader:
            xb, yb, bb, hb = xb.to(device), yb.to(device), bb.to(device), hb.to(device)
            out = forward(model, xb, bb)
            main = mse(out["rul"], yb)
            loss = main
            if use_aux:
                if settings.lambda_hi:
                    loss = loss + settings.lambda_hi * hi_supervision_loss(out["hi"], hb)
                if settings.lambda_mono:
                    loss = loss + settings.lambda_mono * monotonicity_loss(out["hi"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            opt.step()
            tot_main += main.item() * len(xb)
            tot_all += loss.item() * len(xb)
            n += len(xb)
        va, va_hi = evaluate_loss(model, va_loader, device, use_aux)
        sched.step(va)
        hist["train_loss"].append(tot_main / n)
        hist["train_total_loss"].append(tot_all / n)
        hist["val_loss"].append(va)
        hist["val_hi_loss"].append(va_hi)
        hist["val_rmse_cycles"].append(float(np.sqrt(va) * scale))
        hist["lr"].append(opt.param_groups[0]["lr"])
        log(f"  [{spec.name} seed {seed}] epoch {epoch:3d} train {tot_main / n:.5f} val {va:.5f} "
            f"(RMSE {np.sqrt(va) * scale:6.2f})")
        if va < best - 1e-6:
            best, bad, best_epoch = va, 0, epoch
            torch.save({"state_dict": model.state_dict(), "model_type": spec.model_type, "run_name": spec.name,
                        "datf_config": model.cfg.to_dict() if isinstance(model, DATFNet) else None,
                        "seed": seed, "settings": settings.to_dict(), "epoch": epoch, "val_loss": va,
                        "rul_clip": clip, **(extra_meta or {})}, ckpt_path)
        else:
            bad += 1
            if bad >= settings.patience:
                break
    hist.update(best_epoch=best_epoch, best_val_loss=best, epochs_run=len(hist["val_loss"]),
                train_seconds=round(time.time() - t0, 1), n_parameters=count_parameters(model), device=str(device))
    return hist


@torch.no_grad()
def evaluate_loss(model, loader, device, use_aux):
    model.eval()
    tot = tot_hi = n = 0.0
    for xb, yb, bb, hb in loader:
        xb, yb, bb, hb = xb.to(device), yb.to(device), bb.to(device), hb.to(device)
        out = forward(model, xb, bb)
        tot += nn.functional.mse_loss(out["rul"], yb).item() * len(xb)
        if use_aux:
            tot_hi += hi_supervision_loss(out["hi"], hb).item() * len(xb)
        n += len(xb)
    return tot / n, (tot_hi / n if use_aux else None)


def load_checkpoint(path: Path, n_features: int, seq_len: int, device) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    spec = RunSpec(ck["run_name"], ck["model_type"], datf={k: v for k, v in (ck["datf_config"] or {}).items()
                                                            if k not in ("n_features", "seq_len")})
    if spec.datf.get("tcn_dilations") is not None:
        spec.datf["tcn_dilations"] = tuple(spec.datf["tcn_dilations"])
    model = build_model(spec, n_features, seq_len).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def predict(model, X: np.ndarray, b: np.ndarray, clip: float, device, batch: int = 1024) -> tuple[np.ndarray, np.ndarray | None]:
    """RUL in cycles (clipped at >= 0, as the baseline Predictor does) and the HI sequence (or None)."""
    model.eval()
    ruls, his = [], []
    for i in range(0, len(X), batch):
        out = forward(model, torch.from_numpy(X[i:i + batch]).to(device), torch.from_numpy(b[i:i + batch]).to(device))
        ruls.append(out["rul"].cpu().numpy())
        if out["hi"] is not None:
            his.append(out["hi"].cpu().numpy())
    return np.clip(np.concatenate(ruls) * clip, 0, None), (np.concatenate(his) if his else None)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
