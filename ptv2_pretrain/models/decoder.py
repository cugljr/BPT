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


class TokenCrossAttentionPointDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_points: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_points = output_points
        self.point_queries = nn.Parameter(torch.randn(output_points, input_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "query_norm": nn.LayerNorm(input_dim),
                        "memory_norm": nn.LayerNorm(input_dim),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=input_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "ff_norm": nn.LayerNorm(input_dim),
                        "ff": nn.Sequential(
                            nn.Linear(input_dim, ff_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(ff_dim, input_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(input_dim)
        self.to_points = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 3),
        )

    def forward(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = cond_tokens.shape[0]
        queries = self.point_queries.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            q = layer["query_norm"](queries)
            memory = layer["memory_norm"](cond_tokens)
            attended, _ = layer["cross_attn"](q, memory, memory, need_weights=False)
            queries = queries + attended
            queries = queries + layer["ff"](layer["ff_norm"](queries))
        queries = self.norm(queries)
        return torch.tanh(self.to_points(queries))
