from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


def ensure_ptv2_repo(repo_path: str | Path) -> Path:
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"PointTransformerV2 repo not found: {repo_path}")
    repo_path_str = str(repo_path)
    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)
    return repo_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0, dtype=np.float32)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 0:
        scale = 1.0
    normalized = centered / scale
    return normalized.astype(np.float32), center.astype(np.float32), scale


def sample_or_repeat_indices(num_points: int, out_points: int, rng: np.random.Generator) -> np.ndarray:
    if num_points >= out_points:
        return rng.choice(num_points, size=out_points, replace=False)
    repeat = int(np.ceil(out_points / num_points))
    tiled = np.tile(np.arange(num_points), repeat)[:out_points]
    rng.shuffle(tiled)
    return tiled


def split_offsets(tensor: torch.Tensor, offset: torch.Tensor) -> List[torch.Tensor]:
    chunks = []
    start = 0
    for end in offset.tolist():
        chunks.append(tensor[start:end])
        start = end
    return chunks


def resample_tokens(tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    token_count = tokens.shape[0]
    if token_count == num_tokens:
        return tokens
    if token_count > num_tokens:
        indices = torch.linspace(0, token_count - 1, steps=num_tokens, device=tokens.device)
        indices = indices.round().long()
        return tokens.index_select(0, indices)
    repeat = int(np.ceil(num_tokens / token_count))
    tiled = tokens.repeat(repeat, 1)[:num_tokens]
    return tiled


def read_split_file(split_file: str | Path) -> List[str]:
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def resolve_mesh_path(mesh_dir: str | Path, sample_id: str) -> Path:
    mesh_dir = Path(mesh_dir)
    candidates: Iterable[Path] = (
        mesh_dir / sample_id,
        mesh_dir / f"{sample_id}.obj",
        mesh_dir / f"{sample_id}.ply",
        mesh_dir / Path(sample_id).name,
        mesh_dir / f"{Path(sample_id).name}.obj",
        mesh_dir / f"{Path(sample_id).name}.ply",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve mesh path for sample '{sample_id}' in {mesh_dir}")


def sample_cache_name(sample_id: str) -> str:
    return sample_id.replace("\\", "__").replace("/", "__")
