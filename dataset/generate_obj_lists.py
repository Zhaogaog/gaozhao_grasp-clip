import numpy as np
import os
from PIL import Image
import scipy.io as scio
import sys
import json
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from knn.knn_modules import knn
import torch
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, transform_points
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/home/gaozhao/graspnet', required=False)
parser.add_argument('--camera_type', default='kinect', help='Camera split [realsense/kinect]')

if __name__ == '__main__':
    cfgs = parser.parse_args()
    dataset_root = cfgs.dataset_root   # set dataset root
    camera_type = cfgs.camera_type   # kinect / realsense
    save_path_root = os.path.join(dataset_root, 'obj_lists', camera_type)
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    num_points = 150000
    obj_lists = []
    file = open(save_path_root + '/obj_lists.json', 'w')


    for scene_id in range(100):
        #一个场景一个碰撞标签。
        labels = np.load(
            os.path.join(dataset_root, 'collision_label', 'scene_' + str(scene_id).zfill(4), 'collision_labels.npz'))
        collision_dump = []

        for j in range(len(labels)):
            collision_dump.append(labels['arr_{}'.format(j)])

        for ann_id in range(256):
            # get scene point cloud
            depth = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                     camera_type, 'depth', str(ann_id).zfill(4) + '.png')))
            seg = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                   camera_type, 'label', str(ann_id).zfill(4) + '.png')))
            meta = scio.loadmat(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                             camera_type, 'meta', str(ann_id).zfill(4) + '.mat'))
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
            camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                                factor_depth)
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
            #返回bool 与原数组同维度
            # remove outlier and get objectness label
            depth_mask = (depth > 0)
            camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'camera_poses.npy'))
            camera_pose = camera_poses[ann_id]
            align_mat = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                             camera_type, 'cam0_wrt_table.npy'))
            #trans表示到世界坐标系（从当前帧到第一帧，再到世界）
            trans = np.dot(align_mat, camera_pose)
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
            cloud_masked = cloud[mask]
            #cloud_masked维度 是(H*W)*3,展开了。
            #比seg>0扩大了范围。必要吗？？
            seg_masked = seg[mask]
            objectness_label = seg[mask]

            # sample points
            if len(cloud_masked) >= num_points:
                idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), num_points - len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            # num_points*3
            cloud_sampled = cloud_masked[idxs]
            # num_points(一维)
            seg_sampled = seg_masked[idxs]
            # num_points*1
            # graspness_sampled = graspness[idxs]
            # # num_points(一维)
            # objectness_label = seg_sampled.copy()
            # # 包含-1到0到87的类别标签
            # objectness_label = objectness_label - 1
            # i从0开始 物体位姿和抓取的一些定义？


            # get scene object and grasp info
            #annotations表示每个场景每张图片对应的物体位姿
            scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                  camera_type, 'annotations', '%04d.xml' % ann_id))
            pose_vectors = scene_reader.getposevectorlist()
            obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)

            for i in obj_list:
                if (seg_sampled == i+1).sum() < 50:
                    print('obj_id : {}, scene_id : {},ann_id :{}'.format(i, scene_id, ann_id))
                    continue
                json_i = json.dumps({'obj_id' : i, 'scene_id' : scene_id, 'ann_id' : ann_id})
                file.write(json_i + '\n')
    file.close()
