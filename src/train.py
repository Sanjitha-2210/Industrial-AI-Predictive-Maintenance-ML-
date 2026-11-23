"""Train the LSTM baseline.  Usage:  python -m src.train [--epochs 60 --seq-len 30 ...]"""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config as C
from src.data.load_data import load_train
from src.data.preprocess import Preprocessor, add_rul, make_windows, split_engines
from src.data.table import take, unit_ids
from src.models.lstm import LSTMRegressor


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=C.SEQ_LEN)
    ap.add_argument("--rul-clip", type=int, default=C.RUL_CLIP)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    df = add_rul(load_train(), clip=args.rul_clip)
    train_units, val_units = split_engines(unit_ids(df), C.VAL_FRACTION, args.seed)
    print(f"Engines: {len(train_units)} train / {len(val_units)} validation")

    # scaler is fitted on training engines only
    prep = Preprocessor(args.seq_len, args.rul_clip).fit(take(df, np.isin(df["unit"], train_units)), val_units)
    prep.save()
    scaled = prep.transform(df)
    Xtr, ytr, _ = make_windows(scaled, prep.feature_cols, args.seq_len, set(train_units))
    Xva, yva, _ = make_windows(scaled, prep.feature_cols, args.seq_len, set(val_units))
    print(f"Windows: train {Xtr.shape}, val {Xva.shape}; features: {len(prep.feature_cols)}")

    scale = float(args.rul_clip)  # targets are trained as RUL / rul_clip
    mk = lambda X, y, shuffle: DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y / scale)),
        batch_size=args.batch_size, shuffle=shuffle)
    tr_loader, va_loader = mk(Xtr, ytr, True), mk(Xva, yva, False)

    model = LSTMRegressor(len(prep.feature_cols), hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_rmse_cycles": []}
    best, bad, best_epoch = float("inf"), 0, 0
    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, tr_loader, device, criterion, optimizer)
        va = run_epoch(model, va_loader, device, criterion)
        scheduler.step(va)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        history["val_rmse_cycles"].append(float(np.sqrt(va) * scale))
        print(f"epoch {epoch:3d}  train {tr:.5f}  val {va:.5f}  val RMSE {np.sqrt(va) * scale:6.2f} cycles")
        if va < best - 1e-6:
            best, bad, best_epoch = va, 0, epoch
            torch.save({
                "state_dict": model.state_dict(),
                "n_features": len(prep.feature_cols),
                "hidden": args.hidden,
                "seq_len": args.seq_len,
                "rul_scale": scale,
                "rul_clip": args.rul_clip,
                "feature_cols": prep.feature_cols,
                "epoch": epoch,
                "val_loss": va,
            }, C.LSTM_CHECKPOINT)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    history.update(best_epoch=best_epoch, best_val_loss=best, epochs_run=len(history["train_loss"]),
                   train_seconds=round(time.time() - t0, 1), device=str(device), args=vars(args))
    C.HISTORY_PATH.write_text(json.dumps(history, indent=2))
    print(f"Saved checkpoint {C.LSTM_CHECKPOINT} (best epoch {best_epoch}) and history {C.HISTORY_PATH}")


if __name__ == "__main__":
    main()
