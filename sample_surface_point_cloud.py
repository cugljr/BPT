import argparse
from pathlib import Path

import numpy as np
import trimesh

from src.utils.data_utils import pc_norm, sample_pc, write_pts


def get_args():
    parser = argparse.ArgumentParser(
        description="Sample an xyz-only point cloud from an OBJ mesh surface"
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        required=True,
        help="Path to the input OBJ mesh",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output PLY point cloud",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=4096,
        help="Number of points sampled from the mesh surface",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize mesh vertices before surface sampling",
    )
    return parser.parse_args()


def main():
    args = get_args()

    mesh = trimesh.load(args.mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if args.normalize:
        vertices = pc_norm(vertices)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    pc_xyz = sample_pc(mesh, args.n_points)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pts(pc_xyz, str(output_path))
    print(f"Saved {args.n_points} xyz points to {output_path}")


if __name__ == "__main__":
    main()
