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
parser.add_argument('--dataset_root', default='/media/ama/data0/gz/graspnet/graspnet', required=False)
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
    file = open(save_path_root + '/obj_lists_test_seen.json', 'w')


    for scene_id in range(100,130):
        for ann_id in range(256):
            camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'camera_poses.npy'))
            camera_pose = camera_poses[ann_id]

            scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                  camera_type, 'annotations', '%04d.xml' % ann_id))
            pose_vectors = scene_reader.getposevectorlist()
            obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)
            for i in obj_list:
                json_i = json.dumps({'obj_id' : i, 'scene_id' : scene_id, 'ann_id' : ann_id})
                file.write(json_i + '\n')
    file.close()
