import numpy as np
import torch
from pathlib import Path
import trimesh
import pyvista as pv
import argparse
import time
from src.models.mesh_transformer import MeshTransformer
from os.path import join, dirname, basename, splitext, exists
from src.utils.data_utils import *
from src.utils.serializaiton import BPT_deserialize
from metrics.metrics_utils import compute_chamfer_distance
import pandas as pd
from tqdm import tqdm


# Define argument parser
def get_args():
    parser = argparse.ArgumentParser(description="BPT Inference")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="btmgpt",
        help="Path to the run name for load model checkpoint",
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
        default="",
        help="Path to the input txt paths or file path for inference",
    )
    parser.add_argument(
        "--n_trial",
        type=int,
        default=5,
        help="Number of trials for inference",
    )
    parser.add_argument(
        "--use_vertices",
        action="store_true",
        help="Use mesh vertices as base points",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=4096,
        help="Number of points to sample from the mesh",
    )
    parser.add_argument(
        "--decimation",
        action="store_true",
        help="Enable or disable mesh decimation",
    )
    parser.add_argument(
        "--decimation_target_nfaces",
        type=int,
        default=500,
        help="Target number of faces after decimation",
    )
    args = parser.parse_args()
    file_path = Path(args.file_path)
    file_ext = splitext(args.file_path)[1].lower()
    if file_ext == ".txt":
        dataset_name = basename(dirname(dirname(args.file_path)))
        input_dir = join(dirname(dirname(args.file_path)), "partial")
        with open(file_path, "r") as f:
            filenames = f.readlines()
        args.file_paths = [
            join(input_dir, f"{filename.strip()}.xyz") for filename in filenames
        ]
        args.save_folder = join("results", args.exp_name, dataset_name)
    else:
        args.file_paths = [args.file_path]
        filename = splitext(args.file_path)[0]
        args.save_folder = join("results", "single", args.exp_name)
    return args


def load_model(exp_name, device, ckpt_type):
    ckpt_path = join("runs", exp_name, "checkpoints", f"{ckpt_type}.ckpt")
    model = MeshTransformer.load_from_checkpoint(ckpt_path)
    model.to(device).eval()
    print(f"Model contains {sum(p.numel() for p in model.parameters()) / 1e6} M params")
    return model


def load_partial_pc(partial_path, n_points, device, use_vertices, gt_vertices=None):
    pc_partial = read_pts_common(partial_path)
    if use_vertices:
        gt_vertices, center, scale = pc_norm(gt_vertices, return_cs=True)
        pc_partial = pc_norm_with_center_and_scale(pc_partial, center, scale)
        pc_partial = add_base_points(
            pc_partial, mid_points=n_points, vertices=gt_vertices, use_vertices=True
        )
    else:
        pc_partial = pc_norm(pc_partial)
        pc_partial = add_base_points(pc_partial, n_points, None, use_vertices=False)
    pc_partial = sample_pts_to_fixed_num(pc_partial, n_points)
    pc_partial_tensor = (
        torch.tensor(pc_partial).unsqueeze(0).to(dtype=torch.float32, device=device)
    )
    return pc_partial_tensor


def load_mesh_pc(mesh_path, device, decimation, decimation_target_nfaces, n_points):
    vertices, triangles = read_triangle_mesh(mesh_path)
    # Mesh decimation
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
        # Remove leading '3' per triangle
        triangles = np.array(decimated_mesh.faces).reshape(-1, 4)[:, 1:]
    # Point cloud sampling
    vertices = pc_norm(vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    pc_sample = sample_pc(mesh, args.n_points, with_normal=True)
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
        mesh_pred_sample = sample_pc(mesh_pred, n_points, with_normal=False)
        mesh_gt_sample = sample_pc(mesh_gt, n_points, with_normal=False)
        mesh_pred_sample = torch.tensor(mesh_pred_sample)
        mesh_gt_sample = torch.tensor(mesh_gt_sample)
        cd = compute_chamfer_distance(mesh_pred_sample, mesh_gt_sample)
        # 记录最优
        if cd < best_cd:
            best_cd = cd
            best_mesh = mesh_pred
    return best_mesh


def save_outputs(
    save_folder,
    mesh_pred: trimesh.Trimesh,
    pc_input: torch.Tensor = None,
    mesh_gt: trimesh.Trimesh = None,
    face_color: np.array = None,
):
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    if face_color is not None:
        pred_face_colors = np.tile(face_color, (len(mesh_pred.faces), 1))
        mesh_pred.visual.face_colors = pred_face_colors

    # 保存重建网格
    mesh_save_path = join(save_folder, "mesh_pred.obj")
    mesh_pred.export(mesh_save_path)

    # 保存输入点云对应的网格
    if mesh_gt is not None:
        mesh_gt_save_path = join(save_folder, "mesh_gt.obj")
        mesh_gt.export(mesh_gt_save_path)

    # 保存输入点云
    if pc_input is not None:
        pc_input_save_path = join(save_folder, "pc_input.ply")
        if pc_input.shape[1] == 3:
            write_pts(pc_input, pc_input_save_path)
        else:
            write_pts(pc_input[:, :3], pc_input_save_path, normals=pc_input[:, 3:])


def infer_dataset(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(args.exp_name, device, args.ckpt_type)
    infer_csv_path = join(args.save_folder, f"inference_info_{time.time()}.csv")
    csv_result = []

    inference_length = len(args.file_paths)
    for file_path in tqdm(args.file_paths, desc=f"Inference for {inference_length}"):
        model_id, file_ext = splitext(basename(file_path))
        save_folder = join(args.save_folder, model_id)
        mesh_gt = None
        if file_ext == ".xyz":
            mesh_gt_path = file_path.replace("partial", "model").replace(".xyz", ".obj")
            mesh_gt = trimesh.load(mesh_gt_path)
            gt_vertices = np.array(mesh_gt.vertices)
            pc_input = load_partial_pc(
                file_path, args.n_points, device, args.use_vertices, gt_vertices
            )
        elif file_ext == ".obj":
            pc_input, mesh_gt = load_mesh_pc(
                file_path,
                device,
                args.decimation,
                args.decimation_target_nfaces,
                args.n_points,
            )
        else:
            print(
                f"[WARNING] Unsupported file format: {file_ext}. Skipping {file_path}."
            )
            continue

        start_time = time.time()
        mesh_pred = inference_codes(
            model, pc_input, mesh_gt, args.n_points, args.n_trial
        )
        end_time = time.time()

        pc_input_np = pc_input.squeeze(0).cpu().numpy()

        count_pts = len(pc_input_np)
        count_verts = len(mesh_pred.vertices)
        count_faces = len(mesh_pred.faces)
        inference_time = round((end_time - start_time) / args.n_trial, 2)

        csv_result.append(
            {
                "model_id": model_id,
                "count_pts": count_pts,
                "count_verts": count_verts,
                "count_faces": count_faces,
                "inference_time": inference_time,
            }
        )
        face_color = np.array([120, 154, 192, 255], dtype=np.uint8)
        save_outputs(save_folder, mesh_pred, pc_input_np, mesh_gt, face_color)

    df = pd.DataFrame(csv_result)
    df.to_csv(infer_csv_path, index=False)
    print(f"Inference results saved to {args.save_folder}")


if __name__ == "__main__":
    args = get_args()
    infer_dataset(args)
