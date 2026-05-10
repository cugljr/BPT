from __future__ import annotations

from contextlib import nullcontext
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

        self.freeze = freeze
        self.point_encoder = PTV2EncoderWrapper.from_exported_checkpoint(
            checkpoint_path=encoder_ckpt_path,
            repo_path=repo_path,
        )
        cond_dim = self.point_encoder.encoder_channels
        self.cond_norm = nn.LayerNorm(cond_dim)
        self.cond_head_proj = nn.Linear(cond_dim, feature_dim)
        self.cond_proj = nn.Linear(cond_dim, feature_dim)

        if freeze:
            for param in self.point_encoder.parameters():
                param.requires_grad = False
            self.point_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.point_encoder.eval()
        return self

    def forward(self, pc_xyz: torch.Tensor) -> torch.Tensor:
        autocast_context = torch.cuda.amp.autocast(enabled=False) if pc_xyz.is_cuda else nullcontext()
        with autocast_context:
            pc_xyz = torch.nan_to_num(pc_xyz.float(), nan=0.0, posinf=1.0, neginf=-1.0)
            point_feature = self.point_encoder(pc_xyz)
            if not torch.isfinite(point_feature).all():
                raise FloatingPointError("PTV2 point encoder produced NaN or Inf")

            point_feature = torch.nan_to_num(
                point_feature.float(),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            point_feature = self.cond_norm(point_feature)
            pc_embed_head = self.cond_head_proj(point_feature[:, 0:1])
            pc_embed = self.cond_proj(point_feature[:, 1:])
            pc_embed = torch.cat([pc_embed_head, pc_embed], dim=1)
            if not torch.isfinite(pc_embed).all():
                raise FloatingPointError("PTV2 point cloud embeddings contain NaN or Inf")
        return pc_embed
