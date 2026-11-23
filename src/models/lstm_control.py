"""Information-matched LSTM control for the DATF-Net study (research model; not used by the API/dashboard).

Identical architecture to the LSTM baseline (src/models/lstm.py: 2-layer LSTM, hidden 64, FC 64-32-1, dropout 0.2),
but each cycle's input is [x_t, x_t - b]: the scaled window plus its deviation from the engine baseline b (mean of
the engine's first 10 observed cycles) - the same extra information DATF-Net receives. Anything DATF-Net derives
from b or from the window (e.g. its slope) is therefore available to this control, so a remaining performance gap
can be attributed to architecture rather than to information.
"""
import torch
from torch import nn

from src.models.lstm import LSTMRegressor


class BaselineInfoLSTM(nn.Module):
    uses_baseline = True

    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.n_features = n_features
        self.lstm_model = LSTMRegressor(2 * n_features, hidden=hidden)

    @staticmethod
    def build_input(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(B, T, F) window + (B, F) baseline -> (B, T, 2F) = [x_t, x_t - b]."""
        return torch.cat([x, x - b[:, None, :]], dim=-1)

    def forward(self, x, b):
        if b is None:
            raise ValueError("engine baseline b is required")
        return self.lstm_model(self.build_input(x, b))
