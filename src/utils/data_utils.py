import sys
sys.path.append("/home/kemove/devdata1/zyx/BPT-Lighting")
import numpy as np
from torch import Tensor
import random
import trimesh
import open3d as o3d
from typing import Tuple
import networkx as nx
from scipy.spatial.transform import Rotation
from src.utils.mesh_utils import trimesh_fix_mesh


def discretize(t, continuous_range=(-1, 1), num_discrete=128):
    lo, hi = continuous_range
    assert hi > lo
    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5
    return t.round().astype(np.int32).clip(min=0, max=num_discrete - 1)


def undiscretize(t, continuous_range=(-1, 1), num_discrete=128):
    lo, hi = continuous_range
    assert hi > lo
    t = t.astype(np.float32)
    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo


def pc_norm(pc, return_cs=False):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / scale
    if return_cs:
        return pc, centroid, scale
    else:
        return pc


def pc_norm_with_center_and_scale(pc, center, scale):
    """pc: NxC, return NxC"""
    pc = (pc - center) / scale
    return pc


def pc_center(pc, return_centroid=False):
    """
    将点云平移到中心
    """
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    if return_centroid:
        return pc_centered, centroid
    else:
        return pc_centered


def pc_normalize(pc, return_scale=False):
    """
    将点云缩放到最大距离为1
    """
    distances = np.sqrt(np.sum(pc**2, axis=1))
    scale = np.max(distances)
    pc_normalized = pc / scale
    if return_scale:
        return pc_normalized, scale
    else:
        return pc_normalized


def read_pts_ply(file_path):
    mesh = trimesh.load(file_path)
    return np.array(mesh.vertices, dtype=np.float32)


def read_pts_xyz(file_path):
    pts = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            xyz = line.split(" ")
            pts.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return np.array(pts, dtype=np.float32)


def read_pts_common(file_path):
    ext = file_path.split(".")[-1]
    if ext == "ply":
        pts = read_pts_ply(file_path)
    elif ext == "xyz" or ext == "txt":
        pts = read_pts_xyz(file_path)
    else:
        raise NotImplementedError
    return pts


def write_pts(pc, save_path, colors=None, normals=None):
    point_cloud = trimesh.points.PointCloud(pc)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        point_cloud.vertex_normals = normals
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        point_cloud.visual.vertex_colors = colors
    point_cloud.export(save_path)


def read_triangle_mesh(mesh_file):
    """Load mesh file"""
    mesh = trimesh.load_mesh(mesh_file)
    vertices = mesh.vertices
    faces = mesh.faces
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def random_down_sample(pts, n_points):
    assert pts.shape[0] > n_points
    choice = np.random.permutation(pts.shape[0])
    pts = pts[choice[:n_points]]
    return pts


def random_up_sample(pts, n_points):
    assert pts.shape[0] < n_points

    while pts.shape[0] < n_points:
        pts = np.concatenate([pts, pts], axis=0)

    choice = np.random.permutation(pts.shape[0])
    pts = pts[choice[:n_points]]
    return pts


def sample_pts_to_fixed_num(pts, n_points):
    if pts.shape[0] == n_points:
        return pts
    elif pts.shape[0] > n_points:
        pts = random_down_sample(pts, n_points)
    elif pts.shape[0] < n_points:
        pts = random_up_sample(pts, n_points)
    return pts


def add_base_points(pc, mid_points=2048, vertices=None, use_vertices=False):
    """
    给 pc 添加底面点，比例根据点云大小动态变化
    Args:
        pc: (N,3) numpy array
        vertices: (M,3) numpy array, 网格顶点
        use_vertices: bool, 是否用 vertices 的最低点作为底面
    """
    n_points = len(pc)

    # 动态生成比例
    ratio = np.random.normal(0.5, 0.15)
    ratio = np.clip(ratio, 0.1, 1.0)
    ratio *= mid_points / max(n_points, mid_points)
    ratio = np.clip(ratio, 0.1, 1.0)

    n_base = max(10, int(n_points * ratio))

    # 选择基准高度
    if use_vertices and vertices is not None:
        z_base = np.min(vertices[:, 2])  # mesh 的最低点
    else:
        z_base = -np.max(pc[:, 2])  # partial 点云最高点取反

    # 随机选取部分点
    indices = np.random.choice(n_points, n_base, replace=False)
    base_points = pc[indices].copy()
    base_points[:, 2] = z_base

    return np.vstack([pc, base_points])


def to_mesh(vertices, faces, transpose=True, post_process=False):
    if transpose:
        vertices = vertices[:, [1, 2, 0]]

    if faces.min() == 1:
        faces = (np.array(faces) - 1).tolist()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if post_process:
        mesh = trimesh_fix_mesh(mesh)
        
    return mesh


def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))


def block_indices(vertices, block_size=8):
    return (vertices[:, [2, 1, 0]] // block_size).astype(np.int32)


def block_ids(vertices, block_size=8, num_blocks=16):
    b = block_indices(vertices, block_size)
    return (b[:, 0] * num_blocks**2 + b[:, 1] * num_blocks + b[:, 2]).astype(np.int32)


def quantize_process_mesh(
    vertices,
    faces,
    num_discrete=128,
    block_first_order=True,
    block_size=8,
    num_blocks=16,
):
    # 1. 量化,去重
    vertices = discretize(vertices, num_discrete=num_discrete)
    vertices, inverse_ind = np.unique(vertices, axis=0, return_inverse=True)

    # 2. 排序(块优先/Z-Y-X)
    if block_first_order:
        block_id_array = block_ids(vertices, block_size, num_blocks)
        sort_inds = np.lexsort(
            (vertices[:, 0], vertices[:, 1], vertices[:, 2], block_id_array)
        )
    else:
        sort_inds = np.lexsort(vertices.T)

    inv_sort = np.argsort(sort_inds)
    vertices = vertices[sort_inds]
    faces = [inv_sort[inverse_ind[f]] for f in faces]

    # 3. 处理平面循环
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            if c_length > 2:
                d = np.argmin(f)
                sub_faces.append([f[(d + i) % c_length] for i in range(c_length)])

    # 4. 按顶点序排序面片
    faces = sorted(sub_faces, key=lambda f: tuple(sorted(f)))

    faces = np.array(faces)
    # 5. 去除退化的平面
    collapsed_mask = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 0] == faces[:, 2])
        | (faces[:, 1] == faces[:, 2])
    )
    faces = faces[~collapsed_mask]

    # 6. 去除未使用的顶点
    used_vertices = np.unique(faces)
    is_sorted = np.all(np.diff(used_vertices) > 0)
    vertices = vertices[used_vertices]

    # 7. 索引映射
    faces = np.searchsorted(used_vertices, faces)
    return vertices, faces


def process_mesh(mesh_path, num_discrete=128, augment=False, transpose=False):
    """Process mesh vertices and faces."""
    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Transpose so that z-axis is vertical.
    if transpose:
        vertices = vertices[:, [2, 0, 1]]

    vertices = pc_center(vertices)

    if augment:
        vertices = augment_mesh(vertices)

    vertices = pc_normalize(vertices)

    vertices, faces = quantize_process_mesh(
        vertices, faces, num_discrete=num_discrete
    )

    vertices = undiscretize(vertices, num_discrete=num_discrete)
    return vertices, faces


def augment_mesh(vertices, scale_min=0.75, scale_max=1.25):
    # 1. 随机缩放
    scale = np.random.uniform(scale_min, scale_max, size=(1, 3))
    vertices *= scale

    # 2. 随机旋转
    rot_z = np.random.uniform(-np.pi, np.pi)
    rotation_options = [0.5 * np.pi, -0.5 * np.pi]
    rot_x = np.random.choice(rotation_options)
    rot_y = np.random.choice(rotation_options)
    case = np.random.choice([1, 2])

    xy_mat = (
        Rotation.from_rotvec([rot_x, 0, 0]).as_matrix()
        if case == 1
        else Rotation.from_rotvec([0, rot_y, 0]).as_matrix()
    )
    z_mat = Rotation.from_rotvec([0, 0, rot_z]).as_matrix()
    rotation_mat = z_mat @ xy_mat
    vertices = vertices @ rotation_mat.T

    # 3. 随机镜像
    M_ID = np.eye(3)
    M_X = np.diag([-1, 1, 1])
    M_Z = np.diag([1, 1, -1])
    M_XZ = M_Z @ M_X

    mirror_options = [M_XZ, M_X, M_Z, M_ID]
    mirror_option_idx = np.random.randint(0, len(mirror_options))
    mirror_mat = mirror_options[mirror_option_idx]
    vertices = vertices @ mirror_mat.T

    return vertices


def pc_mesh_normals(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """
    利用 mesh 为点云赋法线，返回拼接坐标+法线数组。

    参数：
        points: np.ndarray [N,3] - 输入点云
        mesh: trimesh.Trimesh 对象

    返回：
        np.ndarray [N,6] - 拼接坐标与法线的点云数组
    """
    # 1. 在 mesh 上查找最近点和对应面索引
    closest_points, distances, face_idx = mesh.nearest.on_surface(points)

    # 2. 获取对应面的法线
    normals = mesh.face_normals[face_idx]

    # 3. 翻转法线方向，保证一致（点在面法线反方向则翻转）
    vec = points - closest_points
    flip_mask = np.sum(vec * normals, axis=1) > 0  # 向量与法线同向则翻转
    normals[flip_mask] *= -1

    # 4. 拼接 [x, y, z, nx, ny, nz]
    pc_with_normals = np.hstack((points, normals))
    return pc_with_normals


def sample_pc(mesh, n_points, with_normal=False):
    if not with_normal:
        points, _ = mesh.sample(n_points, return_index=True)
        return points

    points, face_idx = mesh.sample(n_points * 10, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float32)
    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], n_points, replace=False)
    pc_normal = pc_normal[ind]
    return pc_normal


if __name__ == "__main__":
    from tqdm import tqdm
    import os
    folder = "/home/kemove/devdata1/zyx/BuildingGPT/dataset/Objaverse_500_Silksong-128/model"
    mesh_file = "/home/kemove/devdata1/zyx/bpt/gt.obj"
    trimesh.load(mesh_file, force="mesh", process=False).export("origin.obj")
    for i in tqdm(range(60)):
        for file in tqdm(os.listdir(folder)):
            file_path = os.path.join(folder, file)
            vertices, faces = process_mesh(
                mesh_file, num_discrete=128, augment=True, transpose=True
            )
    mesh = to_mesh(vertices, faces, transpose=True)
    pc = sample_pc(mesh, 4096, True)
    mesh.export("reconstructed.obj")
