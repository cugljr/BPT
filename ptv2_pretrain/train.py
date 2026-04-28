from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptv2_pretrain.datasets import BuildingPointDataModule
from ptv2_pretrain.models import BuildingPointPretrainModule
from ptv2_pretrain.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a PointTransformerV2 encoder for BPT")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "ptv2_pretrain" / "configs" / "building_ptv2_ae.yaml"),
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg["seed"]))
    torch.set_float32_matmul_precision("high")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]
    logging_cfg = cfg["logging"]

    data_module = BuildingPointDataModule(
        data_root=data_cfg["data_root"],
        split_root=data_cfg["split_root"],
        input_points=int(data_cfg["input_points"]),
        target_points=int(data_cfg["target_points"]),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        point_dropout=float(data_cfg["point_dropout"]),
        jitter_sigma=float(data_cfg["jitter_sigma"]),
        jitter_clip=float(data_cfg["jitter_clip"]),
        scale_min=float(data_cfg["scale_min"]),
        scale_max=float(data_cfg["scale_max"]),
        rotation_prob=float(data_cfg.get("rotation_prob", 0.0)),
    )

    model = BuildingPointPretrainModule(
        repo_path=model_cfg["ptv2_repo_path"],
        variant=model_cfg["variant"],
        learning_rate=float(model_cfg["learning_rate"]),
        encoder_learning_rate=(
            float(model_cfg["encoder_learning_rate"])
            if model_cfg.get("encoder_learning_rate") is not None
            else None
        ),
        weight_decay=float(model_cfg["weight_decay"]),
        eta_min=float(model_cfg["eta_min"]),
        output_points=int(model_cfg["output_points"]),
        decoder_hidden_dim=int(model_cfg["decoder_hidden_dim"]),
        decoder_type=model_cfg.get("decoder_type", "cross_attention"),
        decoder_layers=int(model_cfg.get("decoder_layers", 2)),
        decoder_heads=int(model_cfg.get("decoder_heads", 8)),
        decoder_ff_dim=int(model_cfg.get("decoder_ff_dim", 2048)),
        decoder_dropout=float(model_cfg.get("decoder_dropout", 0.0)),
        in_channels=int(model_cfg["in_channels"]),
        num_cond_tokens=int(model_cfg["num_cond_tokens"]),
        patch_embed_depth=int(model_cfg["patch_embed_depth"]),
        patch_embed_channels=int(model_cfg["patch_embed_channels"]),
        patch_embed_groups=int(model_cfg["patch_embed_groups"]),
        patch_embed_neighbours=int(model_cfg["patch_embed_neighbours"]),
        enc_depths=tuple(model_cfg["enc_depths"]),
        enc_channels=tuple(model_cfg["enc_channels"]),
        enc_groups=tuple(model_cfg["enc_groups"]),
        enc_neighbours=tuple(model_cfg["enc_neighbours"]),
        grid_sizes=tuple(model_cfg["grid_sizes"]),
        enable_checkpoint=bool(model_cfg["enable_checkpoint"]),
        use_coord_token_embed=bool(model_cfg.get("use_coord_token_embed", True)),
        token_sample_mode=model_cfg.get("token_sample_mode", "fps"),
    )

    experiment_name = logging_cfg.get("experiment_name") or f"ptv2_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / experiment_name
    ckpt_dir = run_dir / "checkpoints"

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor="val/chamfer",
            mode="min",
            filename="best",
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = None
    if logging_cfg.get("use_wandb", False):
        logger = WandbLogger(
            project=logging_cfg.get("wandb_project", "BPT-PTV2-Pretrain"),
            name=experiment_name,
            id=experiment_name,
            save_dir=str(run_dir),
            log_model=False,
        )

    trainer = Trainer(
        max_epochs=int(trainer_cfg["max_epochs"]),
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "16-mixed"),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 10)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.0)),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(run_dir),
    )

    trainer.fit(model=model, datamodule=data_module)

    model.export_encoder_checkpoint(ckpt_dir / "encoder_last.pt")
    best_path = callbacks[0].best_model_path
    if best_path:
        best_model = BuildingPointPretrainModule.load_from_checkpoint(best_path)
        best_model.export_encoder_checkpoint(ckpt_dir / "encoder_best.pt")
        print(f"Exported best encoder to {ckpt_dir / 'encoder_best.pt'}")


if __name__ == "__main__":
    main()
