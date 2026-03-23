from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from src.utils.data_utils import *
from src.utils.serializaiton import BPT_serialize
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from os.path import join, splitext, basename
import random


class BPTDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        mode: str = "train",
        augment: bool = False,
        num_discrete: int = 128,
        block_size: int = 8,
        offset_size: int = 16,
        n_points: int = 4096,
        max_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.augment = augment
        self.num_discrete = num_discrete
        self.block_size = block_size
        self.offset_size = offset_size
        self.n_points = n_points
        self.max_seq_len = max_seq_len
        self.augment = mode == "train" and augment

        with open(join(dataset_dir, "split", f"{mode}.txt"), "r") as f:
            self.data_pathes = [
                join(dataset_dir, "model", f"{line.strip()}.obj")
                for line in f.readlines()
            ]

        if self.max_seq_len is not None:
            self.data_pathes = self._filter_by_seq_len(self.data_pathes)

        print(f"BPTDataset Number: {len(self.data_pathes)}")

    def _filter_by_seq_len(self, paths: List[str]) -> List[str]:
        """保留 BPT 序列长度不超过 max_seq_len 的样本（augment=False 下统计）。"""
        valid: List[str] = []
        total = len(paths)
        for i, p in enumerate(paths):
            if total > 2000 and (i + 1) % 2000 == 0:
                print(f"  seq_len filter [{self.mode}]: {i + 1}/{total} ...")
            try:
                vertices, faces = process_mesh(
                    p,
                    num_discrete=self.num_discrete,
                    augment=False,
                    transpose=True,
                )
                mesh = to_mesh(vertices, faces, transpose=True)
                codes = BPT_serialize(mesh, self.block_size, self.offset_size)
                if len(codes) <= self.max_seq_len:
                    valid.append(p)
            except Exception:
                continue
        if len(valid) == 0:
            raise RuntimeError(
                f"No samples left after max_seq_len={self.max_seq_len} filtering ({self.mode})"
            )
        print(
            f"BPTDataset [{self.mode}]: kept {len(valid)} / {total} (max_seq_len={self.max_seq_len})"
        )
        return valid

    def __len__(self) -> int:
        return len(self.data_pathes)

    def load_data(self, data_path):
        while True:
            try:
                vertices, faces = process_mesh(
                    data_path, 
                    num_discrete=self.num_discrete, 
                    augment=self.augment, 
                    transpose=True
                )
                mesh = to_mesh(vertices, faces, transpose=True)
                return mesh
            except Exception as e:
                with open(f"{self.mode}_corrupt", "a") as log:
                    log.write(f"Corrupted file: {data_path}, reason: {e}\n")
                data_path = random.choice(self.data_pathes)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        max_retries = 300
        for _ in range(max_retries):
            data_path = self.data_pathes[idx]
            filename = splitext(basename(data_path))[0]
            mesh = self.load_data(data_path)
            codes = BPT_serialize(mesh, self.block_size, self.offset_size)
            if self.max_seq_len is None or len(codes) <= self.max_seq_len:
                break
            idx = random.randint(0, len(self.data_pathes) - 1)
        else:
            raise RuntimeError(
                f"Could not sample mesh with seq_len <= {self.max_seq_len} after {max_retries} retries"
            )

        pc_norm = sample_pc(mesh, self.n_points, with_normal=True)
        codes = torch.tensor(codes, dtype=torch.int32)
        pc_norm = torch.tensor(pc_norm, dtype=torch.float32)

        data_dict = dict(
            codes=codes,
            pc_norm=pc_norm,
            filename=filename,
        )
        return data_dict


class BPTDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_dir: str,
        batch_size: int = 16,
        augment: bool = True,
        num_discrete: int = 128,
        block_size: int = 8,
        offset_size: int = 16,
        n_points: int = 4096,
        pad_id: int = -1,
        max_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.batch_size = batch_size
        self.pad_id = pad_id

        self.train_dataset = BPTDataset(
            dataset_dir=dataset_dir,
            mode="train",
            augment=augment,
            num_discrete=num_discrete,
            n_points=n_points,
            block_size=block_size,
            offset_size=offset_size,
            max_seq_len=max_seq_len,
        )
        self.val_dataset = BPTDataset(
            dataset_dir=dataset_dir,
            mode="val",
            augment=False,
            num_discrete=num_discrete,
            n_points=n_points,
            block_size=block_size,
            offset_size=offset_size,
            max_seq_len=max_seq_len,
        )
        self.test_dataset = BPTDataset(
            dataset_dir=dataset_dir,
            mode="test",
            augment=False,
            num_discrete=num_discrete,
            n_points=n_points,
            block_size=block_size,
            offset_size=offset_size,
            max_seq_len=max_seq_len,
        )

    def collate_model_batch(
        self, ds: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        pad_value_map = {
            "codes": self.pad_id,
            "pc_norm": 0,
        }
        collated = {}
        for key in ds[0].keys():
            values = [d[key] for d in ds]
            sample = values[0]
            if torch.is_tensor(sample):
                pad_val = pad_value_map.get(key, 0)
                collated[key] = pad_sequence(
                    values, batch_first=True, padding_value=pad_val
                )
            else:
                collated[key] = values
        return collated

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_model_batch,
            num_workers=16,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_model_batch,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_model_batch,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )


if __name__ == "__main__":

    dataset = BPTDataset(
        dataset_dir="/home/kemove/devdata1/zyx/BuildingGPT/dataset/BuildingPCC_NL_Silksong-128",
        mode="val",
    )
    dataset[0]
