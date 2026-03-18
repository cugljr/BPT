import pymeshlab
import trimesh
import open3d as o3d


def meshlab_fix_mesh(mesh_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(1))
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh(
        mesh_path,
        save_vertex_color=False,
        save_vertex_coord=False,
        save_face_color=False,
        save_wedge_texcoord=False,
    )
    ms.clear()


def trimesh_fix_mesh(mesh: trimesh.Trimesh):
    # Remove collapsed triangles and duplicates
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def o3d_fix_mesh(mesh_path, save_path=None, tol_ratio=1e-4):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # 根据整体范围动态设置 merge 容差
    bbox = mesh.get_axis_aligned_bounding_box()
    diag = bbox.get_max_bound() - bbox.get_min_bound()
    tol = tol_ratio * np.linalg.norm(diag)  # 相对尺度
    
    mesh = mesh.merge_close_vertices(tol)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh

def trimesh_fix_with_meshlab(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # 转为 numpy 格式
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # 新建 MeshSet
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))

    # 执行 MeshLab 修复操作
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(1))
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()

    # 取回修复后的 mesh
    m = ms.current_mesh()
    v = np.asarray(m.vertex_matrix())
    f = np.asarray(m.face_matrix())

    # 转回 trimesh
    fixed_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    ms.clear()

    return fixed_mesh
