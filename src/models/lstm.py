"""LSTM baseline: sensor sequence -> LSTM -> FC layers -> RUL (normalised by rul_scale)."""
import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)
