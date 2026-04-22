from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ptv2_pretrain.utils import read_split_file, sample_cache_name, sample_or_repeat_indices


class BuildingPointNPZDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split_file: str,
        input_points: int,
        target_points: int,
        augment: bool = False,
        point_dropout: float = 0.1,
        jitter_sigma: float = 0.01,
        jitter_clip: float = 0.05,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
    ) -> None:
        self.data_root = Path(data_root)
        self.sample_ids = read_split_file(split_file)
        self.input_points = input_points
        self.target_points = target_points
        self.augment = augment
        self.point_dropout = point_dropout
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_surface_points(self, sample_id: str) -> np.ndarray:
        npz_path = self.data_root / f"{sample_cache_name(sample_id)}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Preprocessed point cache not found for sample '{sample_id}'")
        payload = np.load(npz_path)
        return payload["surface_points"].astype(np.float32)

    def _augment_input(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.augment:
            return points

        if rng.random() < 0.9:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rot = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            points = points @ rot.T

        scale = rng.uniform(self.scale_min, self.scale_max)
        points = points * scale

        if self.point_dropout > 0 and rng.random() < 0.8:
            keep_mask = rng.random(points.shape[0]) > self.point_dropout
            if keep_mask.sum() >= max(32, points.shape[0] // 4):
                points = points[keep_mask]

        if self.jitter_sigma > 0:
            noise = rng.normal(0.0, self.jitter_sigma, size=points.shape).astype(np.float32)
            noise = np.clip(noise, -self.jitter_clip, self.jitter_clip)
            points = points + noise

        return points.astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[index]
        surface_points = self._load_surface_points(sample_id)
        rng = np.random.default_rng()

        target_idx = sample_or_repeat_indices(surface_points.shape[0], self.target_points, rng)
        target_points = surface_points[target_idx]

        input_source = surface_points.copy()
        input_source = self._augment_input(input_source, rng)
        input_idx = sample_or_repeat_indices(input_source.shape[0], self.input_points, rng)
        input_points = input_source[input_idx]

        return {
            "input_points": torch.from_numpy(input_points),
            "target_points": torch.from_numpy(target_points),
            "sample_id": sample_id,
        }


class BuildingPointDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        split_root: str,
        input_points: int,
        target_points: int,
        batch_size: int,
        num_workers: int = 8,
        point_dropout: float = 0.1,
        jitter_sigma: float = 0.01,
        jitter_clip: float = 0.05,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
    ) -> None:
        super().__init__()
        split_root = Path(split_root)
        self.train_dataset = BuildingPointNPZDataset(
            data_root=data_root,
            split_file=str(split_root / "train.txt"),
            input_points=input_points,
            target_points=target_points,
            augment=True,
            point_dropout=point_dropout,
            jitter_sigma=jitter_sigma,
            jitter_clip=jitter_clip,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        self.val_dataset = BuildingPointNPZDataset(
            data_root=data_root,
            split_file=str(split_root / "val.txt"),
            input_points=input_points,
            target_points=target_points,
            augment=False,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
