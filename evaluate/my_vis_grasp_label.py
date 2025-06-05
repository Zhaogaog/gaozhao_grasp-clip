import sys
sys.path.append('/home/gaozhao/miniconda3/envs/graspnet/lib/python3.8/site-packages')
import os
import time
import numpy as np
import open3d as o3d

from transforms3d.euler import euler2mat, quat2mat
from graspnetAPI.utils.utils import generate_scene_model, generate_scene_pointcloud, generate_views, get_model_grasps, plot_gripper_pro_max, transform_points
from graspnetAPI.utils.rotation import viewpoint_params_to_matrix, batch_viewpoint_params_to_matrix
def visObjGrasp(dataset_root, obj_idx, num_grasp=10, th=0.5, max_width=0.08, save_folder='save_fig', show=False):
    '''
    Author: chenxi-wang

    **Input:**

    - dataset_root: str, graspnet dataset root

    - obj_idx: int, index of object model

    - num_grasp: int, number of sampled grasps

    - th: float, threshold of friction coefficient

    - max_width: float, only visualize grasps with width<=max_width

    - save_folder: str, folder to save screen captures

    - show: bool, show visualization in open3d window if set to True
    '''
    # plyfile = os.path.join(dataset_root, 'models', '%03d' % obj_idx, 'nontextured.ply')
    plyfile = '/media/ama/data0/gz/OBJ/test_similar/scenes/scene_0301/green_apple.ply'
    model = o3d.io.read_point_cloud(plyfile)

    num_views, num_angles, num_depths = 300, 12, 4
    views = generate_views(num_views)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    # ctr = vis.get_view_control()
    # param = get_camera_parameters(camera='kinect')

    # cam_pos = np.load(os.path.join(dataset_root, 'scenes', 'scene_0000', 'kinect', 'cam0_wrt_table.npy'))
    # param.extrinsic = np.linalg.inv(cam_pos).tolist()

    # sampled_points, offsets, scores, _ = get_model_grasps('%s/grasp_label/%03d_labels.npz' % (dataset_root, obj_idx))
    sampled_points, offsets, scores, _ = get_model_grasps('%s/agveuv_labels.npz' % dataset_root)
    cnt = 0
    point_inds = np.arange(sampled_points.shape[0])
    np.random.shuffle(point_inds)
    grippers = []

    for point_ind in point_inds:
        target_point = sampled_points[point_ind]
        offset = offsets[point_ind]
        score = scores[point_ind]
        view_inds = np.arange(300)
        np.random.shuffle(view_inds)
        flag = False
        for v in view_inds:
            if flag: break
            view = views[v]
            angle_inds = np.arange(12)
            np.random.shuffle(angle_inds)
            for a in angle_inds:
                if flag: break
                depth_inds = np.arange(4)
                np.random.shuffle(depth_inds)
                for d in depth_inds:
                    if flag: break
                    angle, depth, width = offset[v, a, d]
                    if score[v, a, d] > th or score[v, a, d] < 0 or width > max_width:
                        continue
                    R = viewpoint_params_to_matrix(-view, angle)
                    t = target_point
                    gripper = plot_gripper_pro_max(t, R, width, depth, 1.1 - score[v, a, d])
                    grippers.append(gripper)
                    flag = True
        if flag:
            cnt += 1
        if cnt == num_grasp:
            break

    vis.add_geometry(model)
    for gripper in grippers:
        vis.add_geometry(gripper)
    # ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    # filename = os.path.join(save_folder, 'object_{}_grasp.png'.format(obj_idx))
    filename = os.path.join(save_folder, 'object_{}_grasp.png'.format('green_apple'))
    vis.capture_screen_image(filename, do_render=True)
    if show:
        o3d.visualization.draw_geometries([model, *grippers])
visObjGrasp(dataset_root='/media/ama/data0/gz/graspnet/graspness_unofficial/object-grasp-annotation/grasp_label', obj_idx=0, num_grasp=10, th=0.5, max_width=0.08, save_folder='/media/ama/data0/gz/graspnet/graspness_unofficial/object-grasp-annotation/grasp_label', show=True)