from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptv2_pretrain.utils import normalize_points, read_split_file, resolve_mesh_path, sample_cache_name


def sample_surface_points(mesh_path: Path, num_points: int) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    surface_points, _ = trimesh.sample.sample_surface(mesh, num_points)
    surface_points = np.asarray(surface_points, dtype=np.float32)
    surface_points, center, scale = normalize_points(surface_points)
    return surface_points, center, scale


def process_sample(mesh_dir: str, output_dir: str, num_points: int, sample_id: str) -> str:
    mesh_path = resolve_mesh_path(mesh_dir, sample_id)
    surface_points, center, scale = sample_surface_points(mesh_path, num_points)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_cache_name(sample_id)}.npz"
    np.savez_compressed(
        output_path,
        surface_points=surface_points,
        center=center,
        scale=np.asarray(scale, dtype=np.float32),
        sample_id=np.asarray(sample_id),
    )
    return str(output_path)


def collect_sample_ids(split_root: str) -> list[str]:
    split_root = Path(split_root)
    sample_ids: list[str] = []
    for split_name in ("train.txt", "val.txt", "test.txt"):
        split_path = split_root / split_name
        if split_path.exists():
            sample_ids.extend(read_split_file(split_path))
    return sorted(set(sample_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute normalized xyz-only point clouds for PTv2 pretraining")
    parser.add_argument("--mesh-dir", type=str, required=True)
    parser.add_argument("--split-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = collect_sample_ids(args.split_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending: Iterable[str]
    if args.overwrite:
        pending = sample_ids
    else:
        pending = [sample_id for sample_id in sample_ids if not (output_dir / f"{Path(sample_id).name}.npz").exists()]

    worker = partial(
        process_sample,
        args.mesh_dir,
        str(output_dir),
        args.num_points,
    )

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for idx, output_path in enumerate(executor.map(worker, pending), start=1):
            print(f"[{idx}/{len(pending)}] saved {output_path}")


if __name__ == "__main__":
    main()
