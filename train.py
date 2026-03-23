import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
import torch
import hydra
from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from randomname import get_name

seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")

def main() -> None:
    with hydra.initialize_config_module(version_base=None, config_module="src.config"):
        cfg = hydra.compose(config_name="bpt.yaml")
        model_config = hydra.utils.instantiate(cfg.model_config)

    data_module = model_config.data_module
    model = model_config.model
    epochs = model_config.epochs

    experiment_name = f"{model_config.exp_flag}_{get_name()}"

    gradient_clip_val = model_config.gradient_clip_val
    accumulate_grad_batches = model_config.accumulate_grad_batches

    learning_rate_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath=(Path("runs") / experiment_name / "checkpoints"),
        monitor="val/loss_ce",
        save_last=True,
        save_top_k=1,
        mode="min",
        verbose=False,
        auto_insert_metric_name=False,
        filename="best",
    )

    wandb_logger = WandbLogger(
        project="BPT",
        name=experiment_name,
        id=experiment_name,
        log_model=True,
    )

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[1],
        precision="16-mixed",
        log_every_n_steps=1,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[
            checkpoint_callback,
            learning_rate_callback,
        ],
        logger=[wandb_logger],
    )
    trainer.fit(model=model, datamodule=data_module)

if __name__ == "__main__":
    main()
