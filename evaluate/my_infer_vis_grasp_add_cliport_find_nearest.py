import os
import sys

import cv2
import numpy as np
import argparse
from PIL import Image
import time
from tqdm import tqdm
import scipy.io as scio
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import open3d as o3d
import random
from graspnetAPI.graspnet_eval import GraspGroup
import json
# import time
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append('/media/ama/data0/gz/graspnet/Grounded-Segment-Anything')
from my_infer_vis_grasp import data_process as my_data_process
# from grounded_sam_demo import groungdino_and_sam
# from models.GRASPNET.graspnet_add_sam import GraspNet, pred_decode
from models.GRASPNET.graspnet_add_cliport import GraspNet, pred_decode
# from models._clip import  load_clip_no_resize
from dataset.graspnet_dataset import minkowski_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, my_create_point_cloud_from_depth_image, my_get_workspace_mask, create_point_cloud_from_depth_image, get_workspace_mask
from thop import profile
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/ama/data0/gz/graspnet/graspnet_sim')
parser.add_argument('--checkpoint_path', default='/media/ama/data0/gz/graspnet/graspness_unofficial/WEIGHT/np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar')
# parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_resnet50_frozen_no_resize/dump_epoch10_add_language_frozen_with_rgb/SORTED')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_1/dump_epoch40_add_language_frozen_with_rgb/SORTED')

parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
# parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=-1,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=True)
parser.add_argument('--scene', type=str, default='0010')
parser.add_argument('--index', type=str, default='0085')
parser.add_argument('--obj', type=str, default='04')

cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
# if not os.path.exists(cfgs.dump_dir):
#     os.mkdir(cfgs.dump_dir)
# _, preprocess = load_clip_no_resize("RN50", device='cpu')

def compute_point_distance(A, B):
    '''
    **Input:**
    - A: (N, 3)

    - B: (M, 3)

    **Output:**
    - dists: (N, M)
    '''
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A - B, axis=-1)
    return dists


def compute_closest_points(A, B):
    '''
    **Input:**

    - A: (N, 3)

    - B: (M, 3)

    **Output:**

    - indices: (N,) closest point index in B for each point in A
    '''
    dists = compute_point_distance(A, B)
    indices = np.argmin(dists, axis=-1)
    return indices


def compute_flops(data_input):
    batch_data = minkowski_collate_fn([data_input])

    tic = time.time()

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    # Forward pass

    # end_points = net(batch_data)
    # f1 = FlopCountAnalysis(net, batch_data)

    # print('fvcore',f1.total()/1e9)
    # print('fvcore', parameter_count_table(net))
    # get graspness flops
    flops, params = profile(net, (batch_data,))
    print('profile_flops: ', flops / 1e9, 'params: ', params / 1e9)
    print('para:\n')
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    trainable_pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total - ', pytorch_total_params / 1e9)
    print('Trainable - ', trainable_pytorch_total_params / 1e9)
    return

def data_process(scene,index, obj_idxs=None, save_res=False):
    # print(scene_id, index)
    res ={}
    root = cfgs.dataset_root
    # camera_type = cfgs.camera
    image_path = os.path.join(root, 'scenes', scene_id, 'rgb', index + '.png')
    depth = np.load(os.path.join(root, 'scenes', scene_id, 'depth', index + '.npy'))
    seg = cv2.imread(os.path.join(root, 'scenes', scene_id, 'label', index + '.png'), -1)
    # image_numpy = preprocess(Image.open(os.path.join(root, 'scenes', scene_id, 'rgb', index + '.png')).convert('RGB')).numpy()
    # meta = scio.loadmat(os.path.join(root, 'scenes', scene_id, 'meta', index + '.mat'))
    try:
        intrinsic = np.load(os.path.join(root,'scenes', scene_id, 'camK.npy'))
        factor_depth = 1.0
    except Exception as e:
        print(repr(e))
        # print(scene)





    # if int(num_pred)
    # depth_mask = (depth > 0)


    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    # cloud = my_create_point_cloud_from_depth_image(depth, intrinsic)
    #由深度图像生成点云。

    # get valid points #过滤点的方式 mask作用起到什么 桌面点 或者是 数据中的标签产生的影响
    depth_mask = (depth > 0)

    camera_poses = np.load(os.path.join(root, 'scenes', scene_id, 'camera_poses.npy'))
    # align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
    trans = camera_poses[int(index)]




    # workspace_mask = my_get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)

    # mask = depth_mask
    mask = (depth_mask & workspace_mask )

    cloud_masked = cloud[mask]
    # if cfgs.vis:
    #     mask = (depth_mask & workspace_mask )
    #     return {'point_clouds': cloud[mask].astype(np.float32)}, {}, {}
    seg_masked = seg[mask]

    # sample points random
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    # seg_sampled = seg_masked[idxs]
    # objectness_label = seg_sampled.copy()


    # i从0开始 物体位姿和抓取的一些定义？
    # 加载抓取标签 抓取点 碰撞 抓取分数 宽度


    # 包含-1到0到87的类别标签
    # objectness_label[objectness_label != pack_obj_id+1] = 0
    # objectness_label[objectness_label == pack_obj_id+1] = 1

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                # 'objectness_label': objectness_label.astype(np.int64),
                # 'pack_obj_id': np.array([pack_obj_id]),
                # 'mask_remove_outlier': mask.astype(np.int64),
                # 'mask_sampled': idxs.astype(np.int64),
                # 'img': image_numpy,
                }
    return ret_dict



def inference(data_input,save_real=False,save_res=False):

    batch_data = minkowski_collate_fn([data_input])



    # tic = time.time()

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    # Forward pass
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds= pred_decode(end_points)
    if end_points['batch_good_list'][0] == 1:

        preds = grasp_preds[0].detach().cpu().numpy()
        # cur_res = res[0]


        gg = GraspGroup(preds)
        # collision detection
        if cfgs.collision_thresh > 0:
            cloud = data_input['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            gg = gg[~collision_mask]
        if save_res:
            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, 'SORTED', scene_id)
            save_path = os.path.join(save_dir,str(index).zfill(4)+ '.npy')
            # save_path = os.path.join(save_dir,)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_scene_name(category):


    root_dir = '/media/ama/data0/gz/graspnet/graspnet_sim/scenes'
    scene_list = {}
    for scene in os.listdir(root_dir):
        for ann in os.listdir(os.path.join(root_dir, scene, 'label')):
            if 'png' in ann:
                continue
            with open(os.path.join(root_dir, scene, 'label', ann), 'r') as f:
                data = f.read()
                label = eval(data)
            if category in label.keys():
                scene_list[scene]=ann[0:4]
                break
    return scene_list


if __name__ == '__main__':
    # my_groundingdino_and_sam = groungdino_and_sam()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # split = 'test_seen'
    # split = 'test_similar'
    split = 'test_similar'
    # split = 'test_train'
    with open('/media/ama/data0/gz/graspnet/graspnet_sim/obj_lists/all_category_' + split + '_dict.json', 'r') as f:
        obj_classes_dict = json.loads(f.read())
    obj_classes_dict = {v: k for k, v in obj_classes_dict.items()}
    model_type = 'cliport'
    epoch = 10
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    cfgs.dump_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs/' + model_type + '/' + 'dump_epoch' + str(
        epoch).zfill(2)
    # real_dump_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs/' + model_type + '/' + 'real_dump_epoch' + str(
    #     epoch).zfill(2)
    # cfgs.checkpoint_path = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs/' + model_type + '/' + 'minkuresunet_add_language_frozen_epoch_with_rgb' + str(
    #     epoch).zfill(2) + '.tar'
    if not os.path.exists(cfgs.dump_dir):
        os.mkdir(cfgs.dump_dir)
    net.to(device)
    # Load checkpoint
    checkpoint1 = torch.load(
        '/media/ama/data0/gz/graspnet/graspness_unofficial/WEIGHT/np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar')
    net.load_state_dict(checkpoint1['model_state_dict'], strict=True)
    # checkpoint = torch.load(cfgs.checkpoint_path)
    # net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_epoch = checkpoint1['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    net.eval()


    category = 'grey_folder'
    # scene_list = get_scene_name(category)
    scene_list = {'scene_0301':'0012'}
    if split == 'test_train':
        scene_list = get_scene_name(category)
    print(scene_list)
    time_list = []
    # list_1 = [i for i in range(300,310)] + [i for i in range(400,427)]
    list_1 =[i for i in range(426, 427)]
    print(list_1)
    for scene in tqdm(list_1, desc='scene inference'):
    # for scene in tqdm(list(scene_list.keys()), desc='scene inference'):
        cfgs.scene = str(scene).zfill(4)
        scene_id = 'scene_' + cfgs.scene
        # scene_id = scene
        cfgs.vis = True
        cfgs.infer = False
        show_one = False
        Compute_flops = True

        # index_list = []
        # index_list.append(scene_list[scene])
        # for index in tqdm(index_list, desc='ann inference'):
        for index in tqdm(range(255), desc='ann inference'):
            obj_idxs = None
            # obj_idxs = [10]
            while cfgs.infer:
                # index = cfgs.index
                if cfgs.infer:
                    time_begin = time.time()
                    data_dict = data_process(scene_id,str(index).zfill(4),obj_idxs=obj_idxs,save_res=False)
                    inference(data_dict,save_res=True)
                    time_end = time.time()
                    time_list.append(time_end-time_begin)
                    print('per_time:',time_end-time_begin)
                    break
                    # print(scene_id, index,obj_id,'done')
                    # if len(obj_idxs) ==0 :
                    #     break
            if cfgs.vis:
                cfgs.obj = '24'
                data_dict= data_process(scene_id, str(index).zfill(4), obj_idxs=obj_idxs)


                # compute_flops(data_dict)
                # if Compute_flops:
                #     sys.exit()
                # data_dict, _, _ = my_data_process(scene_id, str(index).zfill(4), obj_idxs=obj_idxs)
                pc = data_dict['point_clouds']
                xyz_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs/cliport/checkpoint_10000/' + scene_id + '/' +str(index).zfill(4)
                if not os.path.exists(xyz_dir):
                    continue
                obj = str(index).zfill(4) + '.npy'
                # print(obj.split('.')[0])
                if show_one:
                    if cfgs.obj not in obj:
                        continue
                # gg = np.load()
                # real_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4_large_dataset_new_try/real_dump_epoch10_add_language_frozen_with_rgb'
                if not os.path.exists(os.path.join(cfgs.dump_dir, 'SORTED', scene_id, obj)):
                    continue
                gg = GraspGroup().from_npy(os.path.join(cfgs.dump_dir, 'SORTED', scene_id, obj))

                # gg = GraspGroup().from_npy(os.path.join(real_dir, 'SORTED', scene_id, obj))
                # gg = gg.nms()
                # gg = gg.sort_by_score()
                # if gg.__len__() > 30:
                #     gg = gg[:30]
                # grasp_array = gg.grasp_group_array



                for xyz_path in sorted(os.listdir(xyz_dir)):
                    # obj_name = obj_classes_dict[int(xyz_path)]
                    # print(obj_name)
                    xyz = np.loadtxt(os.path.join(xyz_dir,xyz_path))
                    camera_pose = np.load(os.path.join(cfgs.dataset_root, 'scenes', scene_id, 'camera_poses.npy'))[int(index)]
                    xyz_1 = np.vstack((xyz.reshape([3,1]),np.ones([1,1])))
                    # print(xyz_1)
                    xyz_camera = np.linalg.inv(camera_pose).dot(xyz_1)[:3,:].reshape([1,3])
                    indice_camera = compute_closest_points(xyz_camera, gg.translations)

                    # gg_translations = gg.translations
                    # ones = np.ones((gg_translations.shape[0], 1), dtype=gg_translations.dtype)
                    # gg_translations_nml = np.hstack((gg_translations, ones))
                    # gg_translations_world = (np.dot(camera_pose, gg_translations_nml.T).T)[:, 0:3]
                    # indice_world = compute_closest_points(xyz.reshape([1,3]), gg_translations_world)
                    # if (indice_camera==indice_world).all():
                    #     pass
                    # else:
                    #     print(indice_camera,indice_world)

                    gg_camera = gg[indice_camera]
                    grasp_array = gg_camera.grasp_group_array
                    grasp_array[:,13:16] = xyz_camera
                    gg_camera.grasp_group_array = grasp_array
                    save_dir = os.path.join('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/' + model_type + '/dump_epoch_10_with_cliport', 'SORTED', scene_id, str(index).zfill(4))
                    os.makedirs(save_dir,exist_ok=True)
                    save_path = os.path.join(save_dir, str(xyz_path)+'.npy')
                    gg_camera.save_npy(save_path)
                    # gg_world = gg[indice_world]

                    # print(xyz_camera)
                    # print(gg_.translations)




                    # if gg.__len__() > 30:
                #     gg = gg[:30]
                # gg = gg[0:1]
                #     grippers_camera = gg_camera.to_open3d_geometry_list()
                #     # grippers_world = gg_world.to_open3d_geometry_list()
                #     cloud = o3d.geometry.PointCloud()
                #     cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
                #     o3d.visualization.draw_geometries([cloud, *grippers_camera])
                    # o3d.visualization.draw_geometries([cloud, *grippers_camera, *grippers_world])
                    # o3d.visualization.draw_geometries([cloud, ])
                    # o3d.visualization.draw_geometries([*grippers])



                # cloud = o3d.geometry.PointCloud()
                # cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
                # # o3d.visualization.draw_geometries([cloud, *grippers])
                # o3d.visualization.draw_geometries([cloud, ])
                # # Example code for execution
                # g = gg[0]
                # translation = g.translation
                # rotation = g.rotation_matrix

                # pose = translation_rotation_2_matrix(translation,rotation) #transform into 4x4 matrix, should be easy
                # # Transform the grasp pose from camera frame to robot coordinate, implement according to your robot configuration
                # tcp_pose = Camera_To_Robot(pose)


                # tcp_ready_pose = copy.deepcopy(tcp_pose)
                # tcp_ready_pose[:3, 3] = tcp_ready_pose[:3, 3] - 0.1 * tcp_ready_pose[:3, 2] # The ready pose is backward along the actual grasp pose by 10cm to avoid collision

                # tcp_away_pose = copy.deepcopy(tcp_pose)

                # # to avoid the gripper rotate around the z_{tcp} axis in the clock-wise direction.
                # tcp_away_pose[3,:3] = np.array([0,0,-1], dtype=np.float64)

                # # to avoid the object collide with the scene.
                # tcp_away_pose[2,3] += 0.1

                # # We rely on python-urx to send the tcp pose the ur5 arm, the package is available at https://github.com/SintefManufacturing/python-urx
                # urx.movels([tcp_ready_pose, tcp_pose], acc = acc, vel = vel, radius = 0.05)

                # # CLOSE_GRIPPER(), implement according to your robot configuration
                # urx.movels([tcp_away_pose, self.throw_pose()], acc = 1.2 * acc, vel = 1.2 * vel, radius = 0.05, wait=False)

    print('inference_ave_time:',np.mean(time_list))