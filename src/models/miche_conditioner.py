import torch
from torch import nn
from miche.encode import load_model
from os.path import join


class PointConditioner(nn.Module):
    def __init__(
        self,
        miche_path: str = "",
        cond_dim: int = 768,
        feature_dim: int = 512,
        freeze: bool = True,
    ):
        super().__init__()

        ckpt_path = join(miche_path, "shapevae-256.ckpt")
        config_path = join(miche_path, "shapevae-256.yaml")
        self.point_encoder = load_model(ckpt_path=ckpt_path, config_path=config_path)

        # Adapt the encoder output to downstream model
        self.cond_head_proj = nn.Linear(cond_dim, feature_dim)
        self.cond_proj = nn.Linear(cond_dim, feature_dim)

        if freeze:
            for param in self.point_encoder.parameters():
                param.requires_grad = False

    def forward(self, pc_norm: torch.Tensor):
        point_feature = self.point_encoder.encode_latents(pc_norm)
        pc_embed_head = self.cond_head_proj(point_feature[:, 0:1])
        pc_embed = self.cond_proj(point_feature[:, 1:])
        pc_embed = torch.cat([pc_embed_head, pc_embed], dim=1)

        assert not torch.any(torch.isnan(pc_embed)), "NAN values in pc_norm embeddings"

        return pc_embed
