"""LSTM baseline: sensor sequence -> LSTM -> FC layers -> RUL (normalised by rul_scale)."""
from __future__ import annotations

try:
    import torch
    from torch import nn
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False
    class _DummyModule:
        pass
    nn = type("nn", (), {"Module": _DummyModule})


class LSTMRegressor(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        if _TORCH_AVAILABLE:
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
        self.n_features = n_features
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout

    def forward(self, x):  # x: (B, T, F)
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available in the current environment.")
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)

