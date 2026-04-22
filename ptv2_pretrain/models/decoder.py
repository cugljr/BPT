from __future__ import annotations

import torch
from torch import nn


class GlobalPointDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        output_points: int = 2048,
    ) -> None:
        super().__init__()
        self.output_points = output_points
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_points * 3),
        )

    def forward(self, global_token: torch.Tensor) -> torch.Tensor:
        batch_size = global_token.shape[0]
        points = self.net(global_token)
        return points.view(batch_size, self.output_points, 3)
