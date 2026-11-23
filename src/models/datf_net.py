"""DATF-Net - Degradation-Aware Temporal Fusion Network (research model; not used by the API/dashboard).

Inputs  x: (B, T=30, F=17) scaled sensor window;  b: (B, F) mean of the engine's first 10 observed cycles.
Outputs dict: "rul" (B,) normalised RUL (x rul_clip = cycles); "hi" (B, T) health index or None.

Health index (HI): HI_t is trained to equal min(RUL_t, rul_clip) / rul_clip, i.e. normalised remaining useful
life at cycle t of the window. Direction: 1 = healthy (>= rul_clip cycles left), 0 = failure; for a real
engine it is non-increasing over time.

Blocks: [A] causal dilated TCN (or 2-layer LSTM ablation) -> H (B,T,d)
        [B] degradation module: D_t = ReLU(W(x_t - b)), HI_t = sigmoid(w.D_t + c), window slope s -> z_d (B,d)
        [C] attention pooling over H with the query from z_d -> z_t (B,d); gated fusion of z_t and z_d
        [D] head: Linear d->32, ReLU, Dropout(0.2), Linear 32->1
Ablation switches are fields of DATFConfig. No claim of novelty or superiority is made without experiments.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DATFConfig:
    n_features: int = 17
    seq_len: int = 30
    d_model: int = 48
    d_deg: int = 32
    d_slope: int = 16
    head_hidden: int = 32
    tcn_dilations: tuple = (1, 2, 4)
    kernel_size: int = 3
    dropout_tcn: float = 0.1
    dropout_head: float = 0.2
    encoder: str = "tcn"              # "tcn" | "lstm"                      (ablation: encoder)
    use_degradation: bool = True      # False -> temporal encoder + last-step pooling only
    use_baseline: bool = True         # False -> degradation branch sees x_t instead of x_t - b
    use_slope: bool = True
    pooling: str = "attention"        # "attention" | "last"
    fusion: str = "gate"              # "gate" | "concat"

    def to_dict(self):
        return asdict(self)


class TCNBlock(nn.Module):
    """Two causal dilated convolutions (left padding) with a residual connection."""

    def __init__(self, cin: int, cout: int, dilation: int, k: int, dropout: float):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(cin, cout, k, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):  # (B, C, T)
        y = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.res(x))


def slope_weights(T: int) -> torch.Tensor:
    """Least-squares slope filter: s = sum_t w_t x_t with w_t = (t - mean) / sum (t - mean)^2."""
    t = torch.arange(T, dtype=torch.float32)
    c = t - t.mean()
    return c / (c ** 2).sum()


class DATFNet(nn.Module):
    def __init__(self, cfg: DATFConfig | None = None, **overrides):
        super().__init__()
        cfg = cfg or DATFConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.cfg = cfg
        F_, d = cfg.n_features, cfg.d_model

        # [A] temporal encoder
        if cfg.encoder == "tcn":
            chans = [F_] + [d] * len(cfg.tcn_dilations)
            self.encoder = nn.Sequential(*[TCNBlock(chans[i], chans[i + 1], dil, cfg.kernel_size, cfg.dropout_tcn)
                                           for i, dil in enumerate(cfg.tcn_dilations)])
        elif cfg.encoder == "lstm":
            self.encoder = nn.LSTM(F_, d, num_layers=2, batch_first=True, dropout=cfg.dropout_tcn)
        else:
            raise ValueError(f"unknown encoder {cfg.encoder}")

        # [B] degradation module
        if cfg.use_degradation:
            self.dev = nn.Linear(F_, cfg.d_deg)
            self.hi = nn.Linear(cfg.d_deg, 1)
            n_sum = cfg.d_deg + 1
            if cfg.use_slope:
                self.register_buffer("slope_w", slope_weights(cfg.seq_len), persistent=False)
                self.slope = nn.Linear(F_, cfg.d_slope)
                n_sum += cfg.d_slope
            self.deg_summary = nn.Linear(n_sum, d)
            # [C] pooling + fusion
            if cfg.pooling == "attention":
                self.q = nn.Linear(d, d, bias=False)
                self.k = nn.Linear(d, d, bias=False)
                self.v = nn.Linear(d, d, bias=False)
            elif cfg.pooling != "last":
                raise ValueError(f"unknown pooling {cfg.pooling}")
            if cfg.fusion not in ("gate", "concat"):
                raise ValueError(f"unknown fusion {cfg.fusion}")
            self.fuse = nn.Linear(2 * d, d)  # gate weights (gate) or projection (concat): same size

        # [D] head
        self.head = nn.Sequential(nn.Linear(d, cfg.head_hidden), nn.ReLU(), nn.Dropout(cfg.dropout_head),
                                  nn.Linear(cfg.head_hidden, 1))

    def encode(self, x):
        if self.cfg.encoder == "tcn":
            return self.encoder(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d)
        return self.encoder(x)[0]

    def forward(self, x, b=None):
        cfg = self.cfg
        H = self.encode(x)
        if not cfg.use_degradation:
            return {"rul": self.head(H[:, -1]).squeeze(-1), "hi": None}
        if cfg.use_baseline:
            if b is None:
                raise ValueError("engine baseline b is required when use_baseline=True")
            dev_in = x - b[:, None, :]
        else:
            dev_in = x
        D = F.relu(self.dev(dev_in))                        # (B, T, d_deg)
        hi = torch.sigmoid(self.hi(D)).squeeze(-1)          # (B, T)
        parts = [D[:, -1], hi[:, -1:]]
        if cfg.use_slope:
            s = torch.einsum("btf,t->bf", x, self.slope_w[: x.size(1)] if x.size(1) == cfg.seq_len
                             else slope_weights(x.size(1)).to(x.device))
            parts.append(F.relu(self.slope(s)))
        z_d = F.relu(self.deg_summary(torch.cat(parts, dim=-1)))   # (B, d)

        if cfg.pooling == "attention":
            q = self.q(z_d)[:, None, :]                              # (B, 1, d)
            att = torch.softmax((q * self.k(H)).sum(-1) / math.sqrt(H.size(-1)), dim=1)  # (B, T)
            z_t = (att[..., None] * self.v(H)).sum(1)                # (B, d)
        else:
            z_t = H[:, -1]
        cat = torch.cat([z_t, z_d], dim=-1)
        if cfg.fusion == "gate":
            g = torch.sigmoid(self.fuse(cat))
            z = g * z_t + (1 - g) * z_d
        else:
            z = F.relu(self.fuse(cat))
        return {"rul": self.head(z).squeeze(-1), "hi": hi}


def hi_supervision_loss(hi: torch.Tensor, hi_target: torch.Tensor) -> torch.Tensor:
    """MSE between predicted HI_t and normalised RUL targets min(RUL_t, clip)/clip over every window cycle."""
    return F.mse_loss(hi, hi_target)


def monotonicity_loss(hi: torch.Tensor) -> torch.Tensor:
    """Penalises increases of health over time (health must be non-increasing): mean ReLU(HI_{t+1} - HI_t)."""
    return F.relu(hi[:, 1:] - hi[:, :-1]).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
