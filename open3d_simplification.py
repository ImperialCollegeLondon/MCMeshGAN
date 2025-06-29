import os
import open3d as o3d

obj_dir_path = 'FollowupMesh_ORISize'
obj_target_path = 'FollowupMesh_DownSampled'
obj_lists= os.listdir(obj_dir_path)
i=0
for obj in obj_lists:
    if obj.endswith('.obj'):
        target_verts=10000
        obj_path = os.path.join(obj_dir_path, obj)
        pcd = o3d.io.read_triangle_mesh(obj_path)
        mesh_smp = pcd.simplify_quadric_decimation(
            target_number_of_triangles=target_verts)
        print(f'{obj} Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
        mesh_smp.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(obj_target_path, obj), mesh_smp)
