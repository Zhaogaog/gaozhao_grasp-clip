""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import numpy as np
import scipy.io as scio
from PIL import Image
import json
import random

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import MinkowskiEngine as ME
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask


class GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        self.imgpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        if 'test' in self.split :
            self.obj_lists_path = os.path.join(root, 'obj_lists', camera, 'obj_lists'+ '_' + self.split + '.json')
            self.obj_lists = []
            self.scene_lists = []
            self.ann_lists = []
            self.obj_ids = []
            with open(self.obj_lists_path, 'r') as f:
                for line in f.readlines():
                    id = 'scene_{}'.format(str(json.loads(line)['scene_id']).zfill(4))
                    if id not in self.sceneIds:
                        break
                    self.scene_lists.append('scene_{}'.format(str(json.loads(line)['scene_id']).zfill(4)))
                    self.ann_lists.append(str(json.loads(line)['ann_id']).zfill(4))
                    self.obj_lists.append(json.loads(line))
                    self.obj_ids.append(str(json.loads(line)['obj_id']).zfill(2))
                f.close()
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.imgpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                #去掉字符串两边的空格
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        if 'test' in self.split :
            return self.scene_lists
        else:
            return self.scenename
    def ann_list(self):
        # if 'test' in self.split :
        #     return self.scene_lists
        # else:
        return self.ann_lists
    def obj_id_list(self):
        # if 'test' in self.split :
        #     return self.scene_lists
        # else:
        return self.obj_ids
    def __len__(self):
        if 'test' in self.split :
            return len(self.obj_lists)
        else:
            return len(self.depthpath)
    #对涉及到点坐标的点云和物体位姿的物体随机进行变换。为了实点云和图像点的对应 不采用 augment_data
    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        pack_obj_id = self.obj_lists[index]['obj_id']#0-87
        if self.split == 'test':
            index = (self.obj_lists[index]['scene_id']- 100 )*256+self.obj_lists[index]['ann_id']
        elif self.split == 'test_seen':
            index = (self.obj_lists[index]['scene_id'] - 100) * 256 + self.obj_lists[index]['ann_id']
        elif self.split == 'test_similar':
            index = (self.obj_lists[index]['scene_id'] - 130) * 256 + self.obj_lists[index]['ann_id']
        elif self.split == 'test_novel':
            index = (self.obj_lists[index]['scene_id'] - 160) * 256 + self.obj_lists[index]['ann_id']
        # scene_id = torch.div(index, 256, rounding_mode='trunc')
        # ann_id = index%256
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])

        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)

        # obj_id_list = []
        # # i从0开始 物体位姿和抓取的一些定义？
        # for i, obj_idx in enumerate(obj_idxs):
        #     obj_id_list.append(obj_idx)
        # pack_obj_id = random.choice(obj_id_list)

        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]

        if return_raw_cloud:
            return cloud_masked
        # sample points random
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    'pack_obj_id': np.array([pack_obj_id])}
        return ret_dict

    def get_data_label(self, index):
        # pack_obj_id = self.obj_lists[index]['obj_id']#0-87
        # index = self.obj_lists[index]['scene_id']*256+self.obj_lists[index]['ann_id']
        # scene_id = index//256
        # ann_id = index%256
        img = np.array(Image.open(self.imgpath[index]))
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        scene = self.scenename[index]
        try:
            #meta和label对应，从1-88
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)


        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points #过滤点的方式 mask作用起到什么 桌面点 或者是 数据中的标签产生的影响
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        seg_masked = seg[mask]
        img_masked = img[mask]
        #从mask到idxs
        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        #num_points*3
        cloud_sampled = cloud_masked[idxs]
        #num_points(一维)
        seg_sampled = seg_masked[idxs]#0-1-88
        img_sampled = img_masked[idxs]
        # num_points*1
        graspness_sampled = graspness[idxs]
        # num_points(一维)
        objectness_label = seg_sampled.copy()

        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        obj_id_list = []
        #i从0开始 物体位姿和抓取的一些定义？
        #加载抓取标签 抓取点 碰撞 抓取分数 宽度
        for i, obj_idx in enumerate(obj_idxs):
            #限制了seg_sampled 点数要大于50.
            #在这里将有可能被遮挡了的物体剔除掉。
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            obj_id_list.append(obj_idx)
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)
            #从前者中抽取
            #< 300 抽取len（points）
            #>1200 ,抽取len(points) / 4
            #300到1200之间，抽取300
            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
        pack_obj_id = random.choice(obj_id_list)
        # 包含-1到0到87的类别标签
        objectness_label[objectness_label!=pack_obj_id]= 0
        objectness_label[objectness_label == pack_obj_id] = 1
        objectness_label_num = np.sum(objectness_label)
        # print('样本标签中的合格数： ', objectness_label_num)
        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    'graspness_label': graspness_sampled.astype(np.float32),
                    'objectness_label': objectness_label.astype(np.int64),
                    'object_poses_list': object_poses_list,
                    'grasp_points_list': grasp_points_list,
                    'grasp_widths_list': grasp_widths_list,
                    'grasp_scores_list': grasp_scores_list,
                    'pack_obj_id': np.array([pack_obj_id-1])}
        return ret_dict

#load简化后的grasp_labels
def load_grasp_labels(root):
    #obj_names从1到88
    obj_names = list(range(1, 89))
    grasp_labels = {}
    for obj_name in tqdm(obj_names, desc='Loading grasping labels...'):
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))

    return grasp_labels


def minkowski_collate_fn(list_data):
    #list_data是get_item返回的集合的列表 4个元素，每个是字典 含有9个属性？
    #将不同点云的坐标（列表 每个元素是一个点云的坐标列表），特征输入，得到整数化 离散化的输出(tensor) 其中coordinates_batch是n*4,列的第一维表示来自哪个点云
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    # 将在内存中不连续存储转的数组化为连续存储的数组，处理更快。
    coordinates_batch = np.ascontiguousarray(coordinates_batch, dtype=np.int32)
    #体素化，离散化，输出两个tensor,与sparse_collate输出的区别在？quantize2original好像是索引？
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    #三个tensor 其中coor为int32
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }
    #关键是要把obj_id转化成合适的格式到dataloader的输出。
    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    # res[list_data[0][-1][]] = collate_fn_(list_data)
    # print(res['pack_obj_id'])
    res = collate_fn_(list_data)
    return res
