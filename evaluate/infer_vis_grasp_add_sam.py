import os
import sys
import numpy as np
import argparse
from PIL import Image
import time
import scipy.io as scio
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append('/media/ama/data0/gz/graspnet/Grounded-Segment-Anything')
from Grounded-Segment-Anything.grounded_sam_demo import groungdino_and_sam
from models.GRASPNET.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/ama/data0/gz/graspnet/graspnet')
parser.add_argument('--checkpoint_path', default='/media/ama/data0/gz/graspnet/graspness_unofficial/WEIGHT/np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar')
# parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_resnet50_frozen_no_resize/dump_epoch10_add_language_frozen_with_rgb/SORTED')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_clip_rn50_frozen_no_resize/dump_epoch10_add_language_frozen_with_rgb/SORTED')

parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=-1,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=True)
parser.add_argument('--scene', type=str, default='0130')
parser.add_argument('--index', type=str, default='0014')
parser.add_argument('--obj', type=str, default='00')

cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


def data_process():

    # my_groundingdino_and_sam.Get_grounding_output(image_path=image_path, text_prompt=text_prompt)

    mask = my_groundingdino_and_sam.mask
    root = cfgs.dataset_root
    camera_type = cfgs.camera
    rgb = np.array(Image.open(os.path.join(root, 'scenes', scene_id, 'rgb', index + '.png')))
    depth = np.array(Image.open(os.path.join(root, 'scenes', scene_id, 'depth', index + '.png')))
    seg = np.array(Image.open(os.path.join(root, 'scenes', scene_id, 'label', index + '.png')))
    text_prompt =
    my_groundingdino_and_sam.Get_grounding_output(image_path=rgb, text_prompt=text_prompt)
    my_groundingdino_and_sam.Get_predictions()
    # meta = scio.loadmat(os.path.join(root, 'scenes', scene_id, camera_type, 'meta', index + '.mat'))
    try:
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
    except Exception as e:
        print(repr(e))
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    camera_poses = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[int(index)])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)

    cloud_masked = cloud[mask]

    # sample points random
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                }
    return ret_dict


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference(data_input):
    batch_data = minkowski_collate_fn([data_input])
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    net.eval()
    tic = time.time()

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
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()

    # Filtering grasp poses for real-world execution. 
    # The first mask preserves the grasp poses that are within a 30-degree angle with the vertical pose and have a width of less than 9cm.
    # mask = (preds[:,9] > 0.9) & (preds[:,1] < 0.09)
    # The second mask preserves the grasp poses within the workspace of the robot.
    # workspace_mask = (preds[:,12] > -0.20) & (preds[:,12] < 0.21) & (preds[:,13] > -0.06) & (preds[:,13] < 0.18) & (preds[:,14] > 0.63) 
    # preds = preds[mask & workspace_mask]

    # if len(preds) == 0:
    #         print('No grasp detected after masking')
    #         return

    gg = GraspGroup(preds)
    # collision detection
    if cfgs.collision_thresh > 0:
        cloud = data_input['point_clouds']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]

    # save grasps
    save_dir = os.path.join(cfgs.dump_dir, scene_id, cfgs.camera)
    save_path = os.path.join(save_dir, cfgs.index + '.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gg.save_npy(save_path)

    toc = time.time()
    print('inference time: %fs' % (toc - tic))


if __name__ == '__main__':

    scene_id = 'scene_' + cfgs.scene
    index = cfgs.index
    my_groundingdino_and_sam = groungdino_and_sam()
    data_dict = data_process()
    show_one = True

    if cfgs.infer:
        inference(data_dict)
    if cfgs.vis:
        cfgs.obj = '67'
        pc = data_dict['point_clouds']
        for obj in os.listdir(os.path.join(cfgs.dump_dir, scene_id, cfgs.camera, cfgs.index)):
            if show_one:
                if cfgs.obj not in obj:
                    continue
            gg = np.load(os.path.join(cfgs.dump_dir, scene_id, cfgs.camera, cfgs.index, obj))
            gg = GraspGroup(gg)
            gg = gg.nms()
            gg = gg.sort_by_score()
            if gg.__len__() > 30:
                gg = gg[:30]
            grippers = gg.to_open3d_geometry_list()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
            o3d.visualization.draw_geometries([cloud, *grippers])

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

