from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ptv2_pretrain.losses import chamfer_distance
from ptv2_pretrain.models.decoder import GlobalPointDecoder, TokenCrossAttentionPointDecoder
from ptv2_pretrain.models.ptv2_encoder import PTV2EncoderWrapper


class BuildingPointAutoencoder(nn.Module):
    def __init__(
        self,
        repo_path: str,
        variant: str = "ptv2m2",
        output_points: int = 2048,
        decoder_hidden_dim: int = 1024,
        decoder_type: str = "cross_attention",
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        decoder_dropout: float = 0.0,
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder = PTV2EncoderWrapper(repo_path=repo_path, variant=variant, **encoder_kwargs)
        self.decoder_type = decoder_type
        if decoder_type == "global_mlp":
            self.decoder = GlobalPointDecoder(
                input_dim=self.encoder.encoder_channels,
                hidden_dim=decoder_hidden_dim,
                output_points=output_points,
            )
        elif decoder_type == "cross_attention":
            self.decoder = TokenCrossAttentionPointDecoder(
                input_dim=self.encoder.encoder_channels,
                output_points=output_points,
                num_layers=decoder_layers,
                num_heads=decoder_heads,
                ff_dim=decoder_ff_dim,
                dropout=decoder_dropout,
            )
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    def forward(self, input_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        cond_tokens = self.encoder(input_points)
        if self.decoder_type == "global_mlp":
            pred_points = self.decoder(cond_tokens[:, 0])
        else:
            pred_points = self.decoder(cond_tokens)
        return {"pred_points": pred_points, "cond_tokens": cond_tokens}


class BuildingPointPretrainModule(LightningModule):
    def __init__(
        self,
        repo_path: str,
        variant: str = "ptv2m2",
        learning_rate: float = 1e-4,
        encoder_learning_rate: float | None = None,
        weight_decay: float = 0.05,
        eta_min: float = 1e-6,
        output_points: int = 2048,
        decoder_hidden_dim: int = 1024,
        decoder_type: str = "cross_attention",
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        decoder_dropout: float = 0.0,
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = BuildingPointAutoencoder(
            repo_path=repo_path,
            variant=variant,
            output_points=output_points,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_type=decoder_type,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            decoder_ff_dim=decoder_ff_dim,
            decoder_dropout=decoder_dropout,
            **encoder_kwargs,
        )

    def forward(self, input_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(input_points)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        outputs = self.model(batch["input_points"])
        loss = chamfer_distance(outputs["pred_points"], batch["target_points"])
        self.log(
            f"{stage}/chamfer",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["input_points"].shape[0],
        )
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        encoder_lr = self.hparams.encoder_learning_rate or self.hparams.learning_rate
        optimizer = AdamW(
            [
                {"params": self.model.encoder.parameters(), "lr": encoder_lr},
                {"params": self.model.decoder.parameters(), "lr": self.hparams.learning_rate},
            ],
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def export_encoder_checkpoint(self, export_path: str | Path) -> None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.encoder.export_state(), export_path)
