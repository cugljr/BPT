from typing import Dict, Any
from src.models.data_module import BPTDataModule
from src.models.mesh_transformer import MeshTransformer
import math
import torch


class BPTModelConfig:

    def __init__(
        self,
        exp_flag: str,
        dataset_dir: str,
        num_discrete: int,
        n_points: int,
        batch_size: int,
        block_size: int,
        offset_size: int,
        augment: bool,
        decoder_config: Dict[str, Any],
        accumulate_grad_batches: int,
        gradient_clip_val: float,
        miche_path: str,
        miche_ckpt_path: str,
        miche_config_path: str,
        miche_freeze: bool,
        max_seq_len: int,
        finetune: bool,
        base_ckpt: str,
        learning_rate: float,
        eta_min: float,
        epochs: int,
    ) -> None:

        self.exp_flag = exp_flag
        self.epochs = epochs
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.finetune = finetune
        self.base_ckpt = base_ckpt

        self.data_module = BPTDataModule(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            augment=augment,
            num_discrete=num_discrete,
            block_size=block_size,
            offset_size=offset_size,
            n_points=n_points,
            max_seq_len=max_seq_len,
        )

        self.dataset_length = len(self.data_module.train_dataset)

        self.total_steps = (
            math.floor(
                self.dataset_length / (batch_size * self.accumulate_grad_batches)
            )
            * self.epochs
        )
        # 预热步数为总步数的1%
        self.warmup_steps = math.floor(self.total_steps * 0.01)
        self.cosine_steps = self.total_steps - self.warmup_steps

        self.model = MeshTransformer(
            decoder_config=decoder_config,
            batch_size=batch_size,
            block_size=block_size,
            offset_size=offset_size,
            miche_path=miche_path,
            miche_ckpt_path=miche_ckpt_path,
            miche_config_path=miche_config_path,
            miche_freeze=miche_freeze,
            max_seq_len=max_seq_len,
            learning_rate=learning_rate,
            eta_min=eta_min,
            warmup_steps=self.warmup_steps,
            cosine_steps=self.cosine_steps,
        )

        if finetune:
            self.model.load_state_dict(
                torch.load(base_ckpt, weights_only=False)["state_dict"]
            )
