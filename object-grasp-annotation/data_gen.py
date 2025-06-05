import os
import time
import open3d as o3d
import numpy as np
from multiprocessing import Process, Manager

from graspnetAPI.utils.rotation import matrix_to_dexnet_params, viewpoint_params_to_matrix
from graspnetAPI.utils.utils import generate_views
from obj_params import configs as cfgs 

from dexnet.grasping import ParallelJawPtGrasp3D, GraspableObject3D, GraspQualityConfigFactory
from dexnet.grasping.quality import PointGraspMetrics3D
from autolab_core import YamlConfig
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile

# some params
V = 300    # number of views
A = 12      # number of gripper rotation angles
H = 0.02     # height of gripper
depth_base = 0.02
collision_thresh = 0.01
save_path = 'grasp_label'
# model_dir = './models'
model_dir = '/media/ama/data0/gz/OBJ/Similar'
phi = (np.sqrt(5) - 1) / 2
views = generate_views(V, phi, R=1)
angles = np.array([x for x in range(A)]) * np.pi / A
depth_list = [0.01, 0.02, 0.03, 0.04]
width_list = [0.01 * x for x in range(1, 16, 1)]

if not os.path.exists(save_path):
    os.mkdir(save_path)

def sample_points(points, max_num_sample):
    num_points = points.shape[0]
    if num_points <= max_num_sample:
        return points
    inds = np.random.choice(num_points, max_num_sample, replace=False)
    points_sampled = points[inds]
    return points_sampled

def evaluate_grasp(grasp, obj, fc_list, force_closure_quality_config, contacts=None):
    tmp, is_force_closure = False, False
    quality = -1
    for ind_, value_fc in enumerate(fc_list):
        value_fc = round(value_fc, 2)
        tmp = is_force_closure
        is_force_closure = PointGraspMetrics3D.grasp_quality(grasp, obj, force_closure_quality_config[value_fc], contacts=contacts, vis=False)
        if tmp and not is_force_closure:
            quality = round(fc_list[ind_ - 1], 2)
            break
        elif is_force_closure and value_fc == fc_list[-1]:
            quality = value_fc
            break
        elif value_fc == fc_list[0] and not is_force_closure:
            break
    return quality

def do_job(obj_name, pool_size=8):
    # load models
    sample_voxel_size, model_voxel_size, model_num_sample = cfgs[obj_name]
    cloud = o3d.io.read_point_cloud(os.path.join(model_dir, obj_name, 'nontextured.ply'))
    cloud_sampled = cloud.voxel_down_sample(sample_voxel_size)
    cloud = cloud.voxel_down_sample(model_voxel_size)
    points = sample_points(np.array(cloud.points), model_num_sample).astype(np.float32)
    points_sampled = np.array(cloud_sampled.points, dtype=np.float32)

    yaml_config = YamlConfig("config.yaml")
    of = ObjFile('{}/{}/textured.obj'.format(model_dir, obj_name))
    sf = SdfFile('{}/{}/textured.sdf'.format(model_dir, obj_name))
    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)
    print('models loaded')

    # score configurations
    force_closure_quality_config = {}
    fc_list = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        yaml_config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(yaml_config['metrics']['force_closure'])

    score = Manager().dict()
    offset = Manager().dict()
    collision = Manager().dict()
    params = (obj, points, points_sampled, fc_list, force_closure_quality_config)

    pool = []
    process_cnt = 0
    work_list = [x for x in range(len(points_sampled))]
    for _ in range(pool_size):
        point_ind = work_list.pop(0)
        pool.append(Process(target=worker, args=(obj_name, point_ind, params, offset, collision, score)))
    [p.start() for p in pool]
    # refill
    while len(work_list) > 0:
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                point_ind = work_list.pop(0)
                p = Process(target=worker, args=(obj_name, point_ind, params, offset, collision, score))
                p.start()
                pool.append(p)
                process_cnt += 1
                print('{}/{}'.format(process_cnt, len(points_sampled)))
                break
    while len(pool) > 0:
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                process_cnt += 1
                print('{}/{}'.format(process_cnt, len(points_sampled)))
                break
    
    saved_score = [None for _ in range(len(points_sampled))]
    saved_offset = [None for _ in range(len(points_sampled))]
    saved_collision = [None for _ in range(len(points_sampled))]
    for i in range(len(points_sampled)):
        saved_score[i] = score[i]
        saved_offset[i] = offset[i]
        saved_collision[i] = collision[i]
    saved_score = np.array(saved_score)
    saved_offset = np.array(saved_offset)
    saved_collision = np.array(saved_collision)
    np.savez_compressed('{}/{}_labels.npz'.format(save_path, obj_name),
             points=points_sampled,
             offsets=saved_offset,
             collision=saved_collision,
             scores=saved_score)


def worker(obj_name, point_ind, params, offset, collision, label):
    obj, points, points_sampled, fc_list, force_closure_quality_config = params
    point_sampled = points_sampled[point_ind]
    curr_label = -1 * np.ones([V, A, len(depth_list)], dtype=np.float32)
    curr_offset = np.zeros([V, A, len(depth_list), 3], dtype=np.float32) # angle, depth, width
    curr_collision = np.ones([V, A, len(depth_list)], dtype=bool) # has collision: 1, no collision:0
    tic = time.time()
    for j, view in enumerate(views):
        for k, angle in enumerate(angles):
            curr_offset[j, k, :, 0] = angle
            # transform model to gripper frame
            R = viewpoint_params_to_matrix(-view, angle)
            points_centered = points - point_sampled.reshape([1, -1])
            points_rotated = np.dot(R.T, points_centered.T).T
            # check depth
            mask1 = (points_rotated[:,2] > (-H / 2.0))
            mask2 = (points_rotated[:,2] < (H / 2.0))
            target = points_rotated[mask1 & mask2]
            for d, depth in enumerate(depth_list):
                curr_offset[j, k, d, 1] = depth
                points_cropped = target[target[:, 0] < depth].copy()
                # compute width
                points_in_gripper = None
                for w in width_list:
                    mask3 = (points_cropped[:,1] > (-w / 2.0))
                    mask4 = (points_cropped[:,1] < (w / 2.0))
                    mask = (mask3 & mask4)
                    inner_points = points_cropped[mask]
                    outer_points = points_cropped[~mask]
                    # check depth
                    if np.any(inner_points[:, 0] < -depth_base):
                        continue
                    # check inner space
                    if len(inner_points) < 10:
                        continue
                    # check outer collision
                    ymin = inner_points[:, 1].min()
                    ymax = inner_points[:, 1].max()
                    half_width = max(abs(ymin), abs(ymax))
                    outer_mask1 = (outer_points[:, 1] >= -(half_width + collision_thresh))
                    outer_mask2 = (outer_points[:, 1] <= (half_width + collision_thresh))
                    if np.any(outer_mask1 & outer_mask2):
                        continue
                    points_in_gripper = inner_points
                    break
                if points_in_gripper is None:
                    continue

                ymin = points_in_gripper[:, 1].min()
                ymax = points_in_gripper[:, 1].max()
                width = 2 * max(abs(ymin), abs(ymax))
                curr_offset[j, k, d, 2] = width
                curr_collision[j, k, d] = False

                # get score
                center = np.array([depth, 0, 0]).reshape([-1, 1])
                center = np.dot(R, center).reshape([3])
                center = center + point_sampled
                binormal, approach_angle = matrix_to_dexnet_params(R)
                grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(center, binormal, width, approach_angle), depth)
                score = evaluate_grasp(grasp, obj, fc_list, force_closure_quality_config, contacts=None)
                if score < 0:
                    continue
                curr_label[j, k, d] = score

    label[point_ind] = curr_label
    offset[point_ind] = curr_offset
    collision[point_ind] = curr_collision
    toc = time.time()
    print('{}: point {} time'.format(obj_name, point_ind), toc - tic)

if __name__ == '__main__':
    obj_list = sorted(os.listdir(model_dir))
    for obj_name in obj_list:
        p = Process(target=do_job, args=(obj_name, 16))
        p.start()
        p.join()
