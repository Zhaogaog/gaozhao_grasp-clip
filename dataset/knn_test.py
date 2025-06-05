import numpy as np
import os
from PIL import Image
import scipy.io as scio
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from knn.knn_modules import knn
import torch
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, transform_points
import argparse

# a = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]]])
# b = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3,3,3], [-1,-1,-1]]])
a = np.array([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
b = np.array([[[0, 1, 2,3,-1], [0, 1, 2,3,-1],[0, 1, 2,3,-1]]])
print(a.shape,b.shape)
a= torch.from_numpy(a).cuda()
b= torch.from_numpy(b).cuda()
nn_inds = knn(a, b, k=1)

nn_inds = knn(a, b, k=1).squeeze() - 1
print(nn_inds)