import numpy as np
import os
from graspnetAPI.graspnet_eval import GraspGroup
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
dump_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4_large_dataset_new_try/dump_epoch10_add_language_frozen_with_rgb'
root = '/media/ama/data0/gz/graspnet/graspnet_sim'
scene_id = 'scene_'+ str(401).zfill(4)
ann = 100
for obj in os.listdir(os.path.join(dump_dir, 'SORTED', scene_id, str(ann).zfill(4))):
    if 'T' in obj:
        continue

    gg = np.load(os.path.join(dump_dir, 'SORTED', scene_id, str(ann).zfill(4), obj))
    gg = GraspGroup().from_npy(os.path.join(dump_dir, 'SORTED', scene_id, str(ann).zfill(4), obj))
    gg = gg.nms()
    gg = gg.sort_by_score()
    if gg.__len__() > 30:
        gg = gg[:30]
    grasp_array = gg.grasp_group_array
    trans = grasp_array[:,13:16]
    rot = grasp_array[:,4:13]
    grasp_depth = grasp_array[:,3]
    T_world = []
    T_SIM = []
    for i in range(grasp_array.shape[0]):
        T = np.zeros((4,4),dtype=np.float64)
        T[:3,:3] = rot[i].reshape(3,3)
        q = R.from_matrix(T[:3,:3]).as_quat()

        matrix = R.from_quat(q).as_matrix()
        print('xuznahun',matrix)
        T[:3,3] = trans[i]
        T[3,3] = 1
        # print(rot[i],trans[i])
        print(T)
        T_0 = np.zeros((4, 4), dtype=np.float64)
        T_0 =  np.zeros((4, 4), dtype=np.float64)
        T_0[0,0] = 1
        T_0[1, 1] = 1
        T_0[2, 2] = 1
        T_0[3, 3] = 1
        T_0[0, 3] = grasp_depth[i]
        print(T_0)
        T_1 = np.dot(T,T_0)
        print(T_1)
        camera_pose = np.load(os.path.join(root, 'scenes', scene_id, 'camera_poses.npy'))[ann]
        T_2 = np.dot(camera_pose, T_1)
        T_3 = np.zeros((4,4))
        # T_3[:3,:3] = np.array([[0,0,-1],[0,-1,0],[1,0,0]])
        # T_3[:3, :3] = np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]])
        # T_3[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        #绕自身y轴旋转90
        T_3[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        # T_3[:3, :3] = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        T_3[3, 3] = 1
        # T_3[3, 3] = 1
        T_4 = np.dot(T_2,T_3)
        # q = Quaternion(matrix=T_2[:3,:3])
        # q = np.array([q.w, q.x, q.y, q.z])
        # q = R.from_matrix(T_2[:3,:3]).as_quat()
        # print(q,)
        grasp_array[i,3] = 0
        grasp_array[i, 4:13] = T_2[:3,:3].reshape((9))
        grasp_array[i, 13:16] = T_2[:3,3]
        q = R.from_matrix(T_2[:3, :3]).as_quat()
        t = T_2[:3,3]
        print(q, t)
        # matrix = R.from_quat(q).as_matrix()
        # print(matrix)
        # t q(x,y,z,w)
        T_world.append(np.hstack([t,q]))

        q = R.from_matrix(T_4[:3, :3]).as_quat()
        t = T_4[:3, 3]
        T_SIM.append(np.hstack([t, q]))
        print(q, t)
    gg.grasp_group_array = grasp_array
    T_world = np.array(T_world)
    print(T_world)
    np.savetxt(os.path.join(dump_dir, 'SORTED', scene_id, str(ann).zfill(4), obj.split('.')[0]+'_T_WORLD'), T_world)
    T_SIM = np.array(T_SIM)
    print(T_SIM)
    np.savetxt(os.path.join(dump_dir, 'SORTED', scene_id, str(ann).zfill(4), obj.split('.')[0]+'_T_SIM'), T_SIM)
# grippers = gg.to_open3d_geometry_list()
# cloud = o3d.geometry.PointCloud()
# cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
# o3d.visualization.draw_geometries([cloud, *grippers])