from __future__ import annotations

import torch.nn as nn


class FFTMLP(nn.Module):
    """MLP classifier for FFT-based descriptors."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
