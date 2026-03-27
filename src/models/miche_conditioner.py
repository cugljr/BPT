import torch
from torch import nn
from miche.encode import load_model
from os.path import join
from typing import Optional


class PointConditioner(nn.Module):
    def __init__(
        self,
        miche_path: str = "",
        miche_ckpt_path: Optional[str] = None,
        miche_config_path: Optional[str] = None,
        cond_dim: int = 768,
        feature_dim: int = 512,
        freeze: bool = True,
    ):
        super().__init__()

        ckpt_path = miche_ckpt_path or join(miche_path, "shapevae-256.ckpt")
        config_path = miche_config_path or join(miche_path, "shapevae-256.yaml")
        self.point_encoder = load_model(ckpt_path=ckpt_path, config_path=config_path)
        self.uses_shapevae_encoder = hasattr(self.point_encoder, "sal")

        # Adapt the encoder output to downstream model
        self.cond_head_proj = nn.Linear(cond_dim, feature_dim)
        self.cond_proj = nn.Linear(cond_dim, feature_dim)

        if freeze:
            for param in self.point_encoder.parameters():
                param.requires_grad = False

    def _encode_shapevae(self, pc_norm: torch.Tensor) -> torch.Tensor:
        pc = pc_norm[..., 0:3]
        feats = pc_norm[..., 3:6]
        latents, _ = self.point_encoder.sal.encoder(pc, feats)

        # The original aligned encoder exposes a dedicated head token. For the
        # shape-only Tallinn encoder, synthesize one from the latent average so
        # BPT keeps the same 1 + N token conditioning structure.
        head = latents.mean(dim=1, keepdim=True)
        return torch.cat([head, latents], dim=1)

    def forward(self, pc_norm: torch.Tensor):
        if self.uses_shapevae_encoder:
            point_feature = self._encode_shapevae(pc_norm)
        else:
            point_feature = self.point_encoder.encode_latents(pc_norm)

        pc_embed_head = self.cond_head_proj(point_feature[:, 0:1])
        pc_embed = self.cond_proj(point_feature[:, 1:])
        pc_embed = torch.cat([pc_embed_head, pc_embed], dim=1)

        assert not torch.any(torch.isnan(pc_embed)), "NAN values in pc_norm embeddings"

        return pc_embed
