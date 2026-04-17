# models.py — WNet and MNet neural network definitions

import torch
import torch.nn as nn
from config import W_NET_HIDDEN_LAYERS, M_NET_HIDDEN_LAYERS, HIDDEN_NEURONS


def _build_network(n_hidden: int, n_neurons: int) -> nn.Sequential:
    """Build a fully-connected network: 1 → [n_neurons]*n_hidden → 1."""
    layers = []
    in_features = 1
    for _ in range(n_hidden):
        linear = nn.Linear(in_features, n_neurons)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.SiLU())
        in_features = n_neurons
    out = nn.Linear(in_features, 1)
    nn.init.xavier_uniform_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class WNet(nn.Module):
    """Predicts deflection w(x).  Input x ∈ [0, 1]."""

    def __init__(self):
        super().__init__()
        self.net = _build_network(W_NET_HIDDEN_LAYERS, HIDDEN_NEURONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MNet(nn.Module):
    """Predicts bending moment M(x).  Input x ∈ [0, 1]."""

    def __init__(self):
        super().__init__()
        self.net = _build_network(M_NET_HIDDEN_LAYERS, HIDDEN_NEURONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DataOnlyNet(nn.Module):
    """Single network for the data-only baseline (8×64, SiLU)."""

    def __init__(self):
        super().__init__()
        self.net = _build_network(W_NET_HIDDEN_LAYERS, HIDDEN_NEURONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
