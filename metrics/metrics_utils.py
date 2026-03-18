import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


def compute_chamfer_distance(pred_points, gt_points):
    """
    Compute the Chamfer Distance (CD) between two point clouds.
    Args:
         pred_points (np.ndarray): Predicted point cloud of shape (N, 3)
         gt_points (np.ndarray): Ground truth point cloud of shape (N, 3)
     Returns:
         float: Chamfer Distance scalar
    """
    # Build KD-trees
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)

    # Compute distances
    dist_pred_to_gt, _ = pred_tree.query(gt_points)
    dist_gt_to_pred, _ = gt_tree.query(pred_points)

    # Compute Chamfer distance
    chamfer_dist = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)

    return chamfer_dist


def compute_hausdorff_distance(pred_points, gt_points):
    """
    Compute the Hausdorff Distance (HD) between two point clouds.
    Hausdorff Distance is the max of:
        - the maximum of the minimum distances from pred_points to gt_points
        - the maximum of the minimum distances from gt_points to pred_points

    Args:
        pred_points (np.ndarray): Predicted point cloud, shape (N, 3)
        gt_points (np.ndarray): Ground truth point cloud, shape (M, 3)

    Returns:
        float: Hausdorff Distance
    """
    # Build KD-trees
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)

    # Compute min distance from each pred point to nearest gt point -> h(pred, gt)
    dist_pred_to_gt, _ = gt_tree.query(pred_points)  # shape: (N,)
    h_pred_to_gt = np.max(dist_pred_to_gt)  # max of min distances

    # Compute min distance from each gt point to nearest pred point -> h(gt, pred)
    dist_gt_to_pred, _ = pred_tree.query(gt_points)  # shape: (M,)
    h_gt_to_pred = np.max(dist_gt_to_pred)  # max of min distances

    # Hausdorff Distance is the max of the two
    hausdorff_dist = max(h_pred_to_gt, h_gt_to_pred)

    return hausdorff_dist


def compute_earth_mover_distance(
    pred_points: np.ndarray, gt_points: np.ndarray
) -> float:
    """
    Compute Earth Mover's Distance (EMD) between two point clouds using Hungarian algorithm.

    Args:
        pred_points (np.ndarray): Predicted point cloud, shape (N, 3)
        gt_points (np.ndarray): Ground truth point cloud, shape (N, 3)

    Returns:
        float: Earth Mover's Distance (mean per-point distance)
    """
    assert pred_points.shape == gt_points.shape, "Point clouds must have same shape"
    N = pred_points.shape[0]

    # Compute pairwise distance matrix
    dist_matrix = np.linalg.norm(
        pred_points[:, np.newaxis, :] - gt_points[np.newaxis, :, :], axis=-1
    )  # shape (N, N)

    # Solve the optimal assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Compute mean matching distance
    emd = np.mean(dist_matrix[row_ind, col_ind])
    return emd


def compute_fscore(pred_points, gt_points, threshold=0.01):
    """
    Compute the F-score between two point clouds.
    Args:
        pred_points (np.ndarray): Predicted point cloud of shape (N, 3)
        gt_points (np.ndarray): Ground truth point cloud of shape (N, 3)
        threshold (float): Distance threshold to consider a point as correctly predicted
    Returns:
        float: F-score (0-1, higher is better)
    """
    # Build KD-trees
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)

    # Compute distances
    dist_pred_to_gt, _ = pred_tree.query(gt_points)
    dist_gt_to_pred, _ = gt_tree.query(pred_points)

    # Compute precision and recall
    precision = np.mean(dist_gt_to_pred < threshold)
    recall = np.mean(dist_pred_to_gt < threshold)

    # Compute F-score
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return fscore


def compute_normal_consistency(pred_mesh, gt_mesh):
    """
    Compute Normal Consistency (NC) and |NC| between two meshes.
    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh
    Returns:
        NC (float): Normal Consistency (-1 to 1, higher is better)
        ABSNC (float): Absolute Normal Consistency (0 to 1, higher is better
    """
    # Get face centers (to measure nearest faces)
    pred_centers = pred_mesh.triangles_center
    gt_centers = gt_mesh.triangles_center

    # Get face normals
    pred_normals = pred_mesh.face_normals
    gt_normals = gt_mesh.face_normals

    # Build KDTree for nearest face search
    pred_tree = cKDTree(pred_centers)
    gt_tree = cKDTree(gt_centers)

    # For each gt face, find nearest pred face
    _, idx_pred = pred_tree.query(gt_centers)
    cos_gt_to_pred = np.sum(gt_normals * pred_normals[idx_pred], axis=1)

    # For each pred face, find nearest gt face
    _, idx_gt = gt_tree.query(pred_centers)
    cos_pred_to_gt = np.sum(pred_normals * gt_normals[idx_gt], axis=1)

    # Average both directions
    NC = 0.5 * (np.mean(cos_gt_to_pred) + np.mean(cos_pred_to_gt))
    ABSNC = 0.5 * (np.mean(np.abs(cos_gt_to_pred)) + np.mean(np.abs(cos_pred_to_gt)))

    return NC, ABSNC


def compute_mesh_volume_difference(pred_mesh, gt_mesh):
    """
    Compute the volume difference between two meshes.

    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh

    Returns:
        float: Absolute volume difference
    """
    # Ensure meshes are watertight
    pred_mesh.fill_holes()
    gt_mesh.fill_holes()

    # Compute volumes
    pred_volume = pred_mesh.volume
    gt_volume = gt_mesh.volume

    # Compute volume difference
    volume_diff = abs(pred_volume - gt_volume)

    return volume_diff


def compute_surface_area_difference(pred_mesh, gt_mesh):
    """
    Compute the surface area difference between two meshes.

    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh

    Returns:
        float: Absolute surface area difference
    """
    # Compute surface areas
    pred_area = pred_mesh.area
    gt_area = gt_mesh.area

    # Compute area difference
    area_diff = abs(pred_area - gt_area)

    return area_diff


def compute_all_metrics(
    pc_input, pred_mesh, gt_mesh, pred_points, gt_points, threshold=0.01
):
    """
    Compute all metrics between two meshes.

    Args:
        pc_input (np.ndarray): Point cloud input of shape (N, 3)
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh
        pred_points (np.ndarray): Predicted point cloud of shape (N, 3)
        gt_points (np.ndarray): Ground truth point cloud of shape (N, 3)
        threshold (float): Distance threshold for F-score

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "chamfer_distance(inp->rec)": compute_chamfer_distance(pc_input, pred_points),
        "chamfer_distance(rec->ref)": compute_chamfer_distance(pred_points, gt_points),
        "hausdorff_distance": compute_hausdorff_distance(pred_points, gt_points),
        "fscore": compute_fscore(pred_points, gt_points, threshold),
        "normal_consistency": compute_normal_consistency(pred_mesh, gt_mesh),
        # "earth_mover_distance": compute_earth_mover_distance(pred_points, gt_points),
        # "volume_difference": compute_mesh_volume_difference(pred_mesh, gt_mesh),
        # "surface_area_difference": compute_surface_area_difference(pred_mesh, gt_mesh),
    }
    return metrics
