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
        default="best",
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
        default=20,
        help="Number of trials for inference",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling value for autoregressive generation",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling value for autoregressive generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help="Sampling temperature for autoregressive generation",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=0,
        help="Generation length. 0 means use the checkpoint model max_seq_len.",
    )
    parser.add_argument(
        "--disable_grammar_mask",
        action="store_true",
        help="Disable BPT token grammar masking during generation.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use argmax decoding instead of multinomial sampling.",
    )
    parser.add_argument(
        "--topology_weight",
        type=float,
        default=0.001,
        help="Weight for vertex/face/short-edge count penalty when choosing the best trial.",
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
        "--vertex_sample_ratio",
        type=float,
        default=0.0,
        help="Ratio of mesh vertices mixed into OBJ point-cloud conditioning.",
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


def load_mesh_pc(
    mesh_path,
    device,
    decimation,
    decimation_target_nfaces,
    n_points,
    vertex_sample_ratio,
):
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
    pc_sample = sample_pc_with_vertices(mesh, n_points, vertex_sample_ratio)
    pc_sample_tensor = (
        torch.tensor(pc_sample).unsqueeze(0).to(dtype=torch.float32, device=device)
    )
    return pc_sample_tensor, mesh


def reorganize_mesh(codes, model):
    codes = codes[codes != model.pad_id].cpu().numpy()
    vertices = BPT_deserialize(codes, model.block_size, model.offset_size)
    n = vertices.shape[0]
    if n < 3 or n % 3 != 0:
        raise ValueError(f"Generated invalid vertex count: {n}")
    faces = torch.arange(1, n + 1).view(-1, 3).numpy()
    mesh = to_mesh(vertices, faces, transpose=False, post_process=True)
    return mesh


def has_sampleable_faces(mesh: trimesh.Trimesh) -> bool:
    return (
        mesh is not None
        and hasattr(mesh, "vertices")
        and hasattr(mesh, "faces")
        and len(mesh.vertices) >= 3
        and len(mesh.faces) > 0
    )


def mesh_edge_stats(mesh: trimesh.Trimesh, short_threshold=0.1) -> dict:
    edges = getattr(mesh, "edges_unique", None)
    if edges is None or len(edges) == 0:
        return {"edge_count": 0, "short_edge_count": 0, "edge_median": 0.0}

    vertices = np.asarray(mesh.vertices)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return {
        "edge_count": int(len(edge_lengths)),
        "short_edge_count": int((edge_lengths < short_threshold).sum()),
        "edge_median": float(np.median(edge_lengths)),
    }


def topology_penalty(mesh_pred: trimesh.Trimesh, mesh_gt: trimesh.Trimesh) -> float:
    pred_edges = mesh_edge_stats(mesh_pred)
    gt_edges = mesh_edge_stats(mesh_gt)

    def ratio_penalty(pred_value, gt_value):
        return abs(np.log((float(pred_value) + 1.0) / (float(gt_value) + 1.0)))

    return (
        ratio_penalty(len(mesh_pred.vertices), len(mesh_gt.vertices))
        + ratio_penalty(len(mesh_pred.faces), len(mesh_gt.faces))
        + ratio_penalty(pred_edges["short_edge_count"], gt_edges["short_edge_count"])
    )


def mesh_diagnostics(mesh_pred: trimesh.Trimesh, mesh_gt: trimesh.Trimesh) -> dict:
    pred_edges = mesh_edge_stats(mesh_pred)
    gt_edges = mesh_edge_stats(mesh_gt)
    return {
        "count_verts_gt": len(mesh_gt.vertices),
        "count_faces_gt": len(mesh_gt.faces),
        "count_edges": pred_edges["edge_count"],
        "count_edges_gt": gt_edges["edge_count"],
        "short_edges_0p1": pred_edges["short_edge_count"],
        "short_edges_0p1_gt": gt_edges["short_edge_count"],
        "edge_median": round(pred_edges["edge_median"], 6),
        "edge_median_gt": round(gt_edges["edge_median"], 6),
        "vert_ratio": round(len(mesh_pred.vertices) / max(len(mesh_gt.vertices), 1), 4),
        "face_ratio": round(len(mesh_pred.faces) / max(len(mesh_gt.faces), 1), 4),
    }


        # 记录最优
def inference_codes(
    model,
    pc_input,
    mesh_gt,
    n_points,
    n_trial=5,
    top_k=20,
    top_p=0.9,
    temperature=0.3,
    max_seq_len=0,
    grammar_mask=True,
    greedy=False,
    topology_weight=0.001,
):
    best_score = float("inf")
    best_mesh = None
    best_info = None
    failed_trials = 0
    resolved_max_seq_len = model.max_seq_len if max_seq_len <= 0 else max_seq_len

    for trial_idx in range(n_trial):
        with torch.no_grad():
            codes = model.generate(
                pc_input,
                max_seq_len=resolved_max_seq_len,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                grammar_mask=grammar_mask,
                greedy=greedy,
            )
        try:
            mesh_pred = reorganize_mesh(codes[0], model)
            if not has_sampleable_faces(mesh_pred):
                raise ValueError(
                    f"Generated mesh has {len(mesh_pred.vertices)} vertices and {len(mesh_pred.faces)} faces"
                )
            mesh_pred_sample = sample_pc(mesh_pred, n_points)
            mesh_gt_sample = sample_pc(mesh_gt, n_points)
        except Exception as exc:
            failed_trials += 1
            print(f"[WARNING] Skip invalid generation trial {trial_idx + 1}/{n_trial}: {exc}")
            continue

        cd = compute_chamfer_distance(mesh_pred_sample, mesh_gt_sample)
        topo_penalty = topology_penalty(mesh_pred, mesh_gt)
        selection_score = cd + topology_weight * topo_penalty
        generated_tokens = int((codes[0] != model.pad_id).sum().item())
        hit_max_seq_len = generated_tokens >= resolved_max_seq_len

        if selection_score < best_score:
            best_score = selection_score
            best_mesh = mesh_pred
            best_info = {
                "chamfer": round(float(cd), 8),
                "topology_penalty": round(float(topo_penalty), 6),
                "selection_score": round(float(selection_score), 8),
                "generated_tokens": generated_tokens,
                "hit_max_seq_len": hit_max_seq_len,
            }

    if best_mesh is None:
        raise RuntimeError(f"All {n_trial} generation trials produced invalid meshes")
    if failed_trials:
        print(f"[WARNING] Invalid generation trials: {failed_trials}/{n_trial}")

    best_info["failed_trials"] = failed_trials
    best_info.update(mesh_diagnostics(best_mesh, mesh_gt))
    return best_mesh, best_info


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
                args.vertex_sample_ratio,
            )
        else:
            print(
                f"[WARNING] Unsupported file format: {file_ext}. Skipping {file_path}."
            )
            continue

        start_time = time.time()
        try:
            mesh_pred, infer_info = inference_codes(
                model,
                pc_input,
                mesh_gt,
                args.n_points,
                args.n_trial,
                args.top_k,
                args.top_p,
                args.temperature,
                args.max_seq_len,
                not args.disable_grammar_mask,
                args.greedy,
                args.topology_weight,
            )
        except RuntimeError as exc:
            print(f"[WARNING] Failed to infer {file_path}: {exc}")
            continue
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
                **infer_info,
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
