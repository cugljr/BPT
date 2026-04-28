from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from ptv2_pretrain.utils import ensure_ptv2_repo, split_offsets


def _import_ptv2_class(repo_path: str, variant: str):
    ensure_ptv2_repo(repo_path)
    if variant == "ptv2m2":
        from pcr.models.point_transformer2.point_transformer_v2m2_base import PointTransformerV2
        return PointTransformerV2
    if variant == "ptv2m1":
        from pcr.models.point_transformer2.point_transformer_v2m1_origin import PointTransformerV2
        return PointTransformerV2
    raise ValueError(f"Unsupported PointTransformerV2 variant: {variant}")


class PTV2EncoderWrapper(nn.Module):
    def __init__(
        self,
        repo_path: str,
        variant: str = "ptv2m2",
        in_channels: int = 3,
        num_cond_tokens: int = 256,
        patch_embed_depth: int = 1,
        patch_embed_channels: int = 48,
        patch_embed_groups: int = 6,
        patch_embed_neighbours: int = 8,
        enc_depths: tuple[int, ...] = (2, 2, 6, 2),
        enc_channels: tuple[int, ...] = (96, 192, 384, 512),
        enc_groups: tuple[int, ...] = (12, 24, 48, 64),
        enc_neighbours: tuple[int, ...] = (16, 16, 16, 16),
        grid_sizes: tuple[float, ...] = (0.06, 0.15, 0.375, 0.9375),
        enable_checkpoint: bool = False,
        use_coord_token_embed: bool = True,
        token_sample_mode: str = "fps",
    ) -> None:
        super().__init__()
        self.repo_path = str(ensure_ptv2_repo(repo_path))
        self.variant = variant
        self.in_channels = in_channels
        self.num_cond_tokens = num_cond_tokens
        self.encoder_channels = enc_channels[-1]
        self.use_coord_token_embed = use_coord_token_embed
        self.token_sample_mode = token_sample_mode
        if token_sample_mode not in {"fps", "linspace"}:
            raise ValueError(f"Unsupported token_sample_mode: {token_sample_mode}")
        self.model_config = dict(
            patch_embed_depth=patch_embed_depth,
            patch_embed_channels=patch_embed_channels,
            patch_embed_groups=patch_embed_groups,
            patch_embed_neighbours=patch_embed_neighbours,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_groups=enc_groups,
            enc_neighbours=enc_neighbours,
            grid_sizes=grid_sizes,
            enable_checkpoint=enable_checkpoint,
            use_coord_token_embed=use_coord_token_embed,
            token_sample_mode=token_sample_mode,
        )

        ptv2_cls = _import_ptv2_class(self.repo_path, variant)
        base_model = ptv2_cls(
            in_channels=in_channels,
            num_classes=0,
            dec_depths=(1, 1, 1, 1),
            dec_channels=(48, 96, 192, 384),
            dec_groups=(6, 12, 24, 48),
            dec_neighbours=(16, 16, 16, 16),
            unpool_backend="map",
            attn_qkv_bias=True,
            pe_multiplier=False,
            pe_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            **self.model_config,
        )

        self.patch_embed = base_model.patch_embed
        self.enc_stages = base_model.enc_stages
        self.coord_token_embed = nn.Sequential(
            nn.Linear(3, self.encoder_channels),
            nn.GELU(),
            nn.Linear(self.encoder_channels, self.encoder_channels),
        ) if use_coord_token_embed else nn.Identity()

    @staticmethod
    def pack_batch(pc_xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, point_count, _ = pc_xyz.shape
        coord = pc_xyz.reshape(batch_size * point_count, 3).contiguous()
        feat = coord.clone()
        offset = torch.arange(
            point_count,
            point_count * (batch_size + 1),
            point_count,
            device=pc_xyz.device,
            dtype=torch.long,
        )
        return {"coord": coord, "feat": feat, "offset": offset}

    def encode_backbone(self, pc_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_dict = self.pack_batch(pc_xyz)
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        points = [coord, feat, offset]
        points = self.patch_embed(points)
        for enc_stage in self.enc_stages:
            points, _ = enc_stage(points)
        return points

    @staticmethod
    def _linspace_indices(token_count: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        if token_count == num_tokens:
            return torch.arange(token_count, device=device, dtype=torch.long)
        if token_count > num_tokens:
            return torch.linspace(0, token_count - 1, steps=num_tokens, device=device).round().long()
        repeat = int((num_tokens + token_count - 1) // token_count)
        return torch.arange(token_count, device=device, dtype=torch.long).repeat(repeat)[:num_tokens]

    @staticmethod
    def _fps_indices(coord: torch.Tensor, num_tokens: int) -> torch.Tensor:
        token_count = coord.shape[0]
        if token_count <= num_tokens:
            repeat = int((num_tokens + token_count - 1) // token_count)
            return torch.arange(token_count, device=coord.device, dtype=torch.long).repeat(repeat)[:num_tokens]

        selected = torch.empty(num_tokens, device=coord.device, dtype=torch.long)
        centroid = coord.mean(dim=0, keepdim=True)
        current = ((coord - centroid) ** 2).sum(dim=-1).argmin()
        min_dist = torch.full((token_count,), float("inf"), device=coord.device, dtype=coord.dtype)
        for i in range(num_tokens):
            selected[i] = current
            dist = ((coord - coord[current]) ** 2).sum(dim=-1)
            min_dist = torch.minimum(min_dist, dist)
            current = min_dist.argmax()
        return selected

    def _sample_indices(self, coord: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if self.token_sample_mode == "fps":
            return self._fps_indices(coord, num_tokens)
        return self._linspace_indices(coord.shape[0], num_tokens, coord.device)

    def build_condition_tokens(self, coord: torch.Tensor, feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        coord_chunks = split_offsets(coord, offset)
        feat_chunks = split_offsets(feat, offset)
        cond_tokens = []
        for coord_chunk, feat_chunk in zip(coord_chunks, feat_chunks):
            feat_chunk = feat_chunk + self.coord_token_embed(coord_chunk)
            indices = self._sample_indices(coord_chunk, self.num_cond_tokens)
            sampled = feat_chunk.index_select(0, indices)
            head = feat_chunk.mean(dim=0, keepdim=True)
            cond_tokens.append(torch.cat([head, sampled], dim=0))
        return torch.stack(cond_tokens, dim=0)

    def forward(self, pc_xyz: torch.Tensor) -> torch.Tensor:
        coord, feat, offset = self.encode_backbone(pc_xyz)
        return self.build_condition_tokens(coord, feat, offset)

    def export_state(self) -> Dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "encoder_config": {
                "repo_path": self.repo_path,
                "variant": self.variant,
                "in_channels": self.in_channels,
                "num_cond_tokens": self.num_cond_tokens,
                **self.model_config,
            },
        }

    @classmethod
    def from_exported_checkpoint(cls, checkpoint_path: str, repo_path: str | None = None) -> "PTV2EncoderWrapper":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder_config = payload["encoder_config"]
        if repo_path is not None:
            encoder_config["repo_path"] = repo_path
        model = cls(**encoder_config)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model
