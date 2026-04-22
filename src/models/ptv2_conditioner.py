from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ptv2_pretrain.models.ptv2_encoder import PTV2EncoderWrapper


class PTV2Conditioner(nn.Module):
    def __init__(
        self,
        repo_path: str,
        encoder_ckpt_path: str,
        feature_dim: int = 512,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if not encoder_ckpt_path:
            raise ValueError("ptv2 encoder checkpoint path is required")

        self.point_encoder = PTV2EncoderWrapper.from_exported_checkpoint(
            checkpoint_path=encoder_ckpt_path,
            repo_path=repo_path,
        )
        cond_dim = self.point_encoder.encoder_channels
        self.cond_head_proj = nn.Linear(cond_dim, feature_dim)
        self.cond_proj = nn.Linear(cond_dim, feature_dim)

        if freeze:
            for param in self.point_encoder.parameters():
                param.requires_grad = False

    def forward(self, pc_xyz: torch.Tensor) -> torch.Tensor:
        point_feature = self.point_encoder(pc_xyz)
        pc_embed_head = self.cond_head_proj(point_feature[:, 0:1])
        pc_embed = self.cond_proj(point_feature[:, 1:])
        pc_embed = torch.cat([pc_embed_head, pc_embed], dim=1)
        assert not torch.any(torch.isnan(pc_embed)), "NAN values in point cloud embeddings"
        return pc_embed
