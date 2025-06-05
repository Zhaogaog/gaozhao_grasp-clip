import open3d as o3d
import numpy as np
print(o3d.__version__)
# print("Load a ply point cloud, print it, and render it")
# # sample_ply_data = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud('/home/gaozhao/下载/fragment.ply')
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])
# 使用sync函数的变体生成纯n乘3矩阵
x = np.linspace(-3, 3, 401)
mesh_x, mesh_y = np.meshgrid(x, x)
z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
z_norm = (z - z.min()) / (z.max() - z.min())
xyz = np.zeros((np.size(mesh_x), 3))
xyz[:, 0] = np.reshape(mesh_x, -1)
xyz[:, 1] = np.reshape(mesh_y, -1)
xyz[:, 2] = np.reshape(z_norm, -1)
print('xyz')
print(xyz)
# 将 xyz值传给Open3D.o3d.geometry.PointCloud并保存点云
pcd = o3d.geometry.PointCloud()
xyz = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/points.npy')
for i in range(5):

    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd],)