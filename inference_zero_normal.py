import argparse
import time
from os.path import basename, join, splitext
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import torch
import trimesh
from tqdm import tqdm

from metrics.metrics_utils import compute_chamfer_distance
from src.models.mesh_transformer import MeshTransformer
from src.utils.data_utils import pc_norm, read_triangle_mesh, sample_pc, to_mesh
from src.utils.serializaiton import BPT_deserialize


def get_args():
    parser = argparse.ArgumentParser(
        description="BPT inference with xyz-only point clouds for OBJ inputs"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="btmgpt",
        help="Run name used to load the model checkpoint",
    )
    parser.add_argument(
        "--ckpt_type",
        type=str,
        choices=["best", "last"],
        default="last",
        help="Checkpoint type to load",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to an OBJ file for inference",
    )
    parser.add_argument(
        "--n_trial",
        type=int,
        default=5,
        help="Number of generation trials",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=4096,
        help="Number of sampled surface points",
    )
    parser.add_argument(
        "--decimation",
        action="store_true",
        help="Enable mesh decimation before point sampling",
    )
    parser.add_argument(
        "--decimation_target_nfaces",
        type=int,
        default=500,
        help="Target number of faces after decimation",
    )
    args = parser.parse_args()
    args.file_paths = [args.file_path]
    args.save_folder = join("results", "single_zero_normal", args.exp_name)
    return args


def load_model(exp_name, device, ckpt_type):
    ckpt_path = join("runs", exp_name, "checkpoints", f"{ckpt_type}.ckpt")
    model = MeshTransformer.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device).eval()
    print(f"Model contains {sum(p.numel() for p in model.parameters()) / 1e6} M params")
    return model


def load_mesh_pc_xyz(
    mesh_path, device, decimation, decimation_target_nfaces, n_points
):
    vertices, triangles = read_triangle_mesh(mesh_path)
    if decimation:
        n_triangles = min(decimation_target_nfaces, len(triangles))
        faces_pyvista = (
            np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
            .astype(np.int32)
            .flatten()
        )
        mesh = pv.PolyData(vertices, faces_pyvista)
        decimated_mesh = mesh.decimate_pro(
            1 - n_triangles / len(triangles),
            boundary_vertex_deletion=True,
        )
        vertices = np.array(decimated_mesh.points)
        triangles = np.array(decimated_mesh.faces).reshape(-1, 4)[:, 1:]

    vertices = pc_norm(vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    pc_sample = sample_pc(mesh, n_points).astype(np.float32)
    pc_sample_tensor = (
        torch.tensor(pc_sample).unsqueeze(0).to(dtype=torch.float32, device=device)
    )
    return pc_sample_tensor, mesh


def reorganize_mesh(codes, model):
    codes = codes[codes != model.pad_id].cpu().numpy()
    vertices = BPT_deserialize(codes, model.block_size, model.offset_size)
    n = vertices.shape[0]
    faces = torch.arange(1, n + 1).view(-1, 3).numpy()
    mesh = to_mesh(vertices, faces, transpose=False, post_process=True)
    return mesh


def inference_codes(model, pc_input, mesh_gt, n_points, n_trial=5):
    best_cd = float("inf")
    best_mesh = None
    for _ in range(n_trial):
        with torch.no_grad():
            codes = model.generate(pc_input, top_k=50, top_p=0.95, temperature=0.5)
        mesh_pred = reorganize_mesh(codes[0], model)
        mesh_pred_sample = sample_pc(mesh_pred, n_points)
        mesh_gt_sample = sample_pc(mesh_gt, n_points)
        mesh_pred_sample = torch.tensor(mesh_pred_sample)
        mesh_gt_sample = torch.tensor(mesh_gt_sample)
        cd = compute_chamfer_distance(mesh_pred_sample, mesh_gt_sample)
        if cd < best_cd:
            best_cd = cd
            best_mesh = mesh_pred
    return best_mesh


def save_outputs(save_folder, mesh_pred, pc_input=None, mesh_gt=None, face_color=None):
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    if face_color is not None:
        pred_face_colors = np.tile(face_color, (len(mesh_pred.faces), 1))
        mesh_pred.visual.face_colors = pred_face_colors

    mesh_pred.export(join(save_folder, "mesh_pred.obj"))

    if mesh_gt is not None:
        mesh_gt.export(join(save_folder, "mesh_gt.obj"))

    if pc_input is not None:
        point_cloud = trimesh.points.PointCloud(pc_input[:, :3])
        point_cloud.export(join(save_folder, "pc_input_xyz.ply"))


def infer_dataset(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(args.exp_name, device, args.ckpt_type)
    infer_csv_path = join(args.save_folder, f"inference_info_{time.time()}.csv")
    csv_result = []

    inference_length = len(args.file_paths)
    for file_path in tqdm(args.file_paths, desc=f"Inference for {inference_length}"):
        model_id, file_ext = splitext(basename(file_path))
        save_folder = join(args.save_folder, model_id)
        if file_ext.lower() != ".obj":
            print(f"[WARNING] Unsupported file format: {file_ext}. Skipping {file_path}.")
            continue

        pc_input, mesh_gt = load_mesh_pc_xyz(
            file_path,
            device,
            args.decimation,
            args.decimation_target_nfaces,
            args.n_points,
        )

        start_time = time.time()
        mesh_pred = inference_codes(
            model, pc_input, mesh_gt, args.n_points, args.n_trial
        )
        end_time = time.time()

        pc_input_np = pc_input.squeeze(0).cpu().numpy()

        csv_result.append(
            {
                "model_id": model_id,
                "count_pts": len(pc_input_np),
                "count_verts": len(mesh_pred.vertices),
                "count_faces": len(mesh_pred.faces),
                "inference_time": round((end_time - start_time) / args.n_trial, 2),
            }
        )
        face_color = np.array([120, 154, 192, 255], dtype=np.uint8)
        save_outputs(save_folder, mesh_pred, pc_input_np, mesh_gt, face_color)

    pd.DataFrame(csv_result).to_csv(infer_csv_path, index=False)
    print(f"Inference results saved to {args.save_folder}")


if __name__ == "__main__":
    args = get_args()
    infer_dataset(args)
