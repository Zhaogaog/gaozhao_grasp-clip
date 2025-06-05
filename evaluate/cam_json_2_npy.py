import numpy as np
import json
for scene_id in range(1,5):
    with open('/media/ama/data0/gz/graspnet/graspnet_real/scenes/scene_'+str(scene_id).zfill(4)+'/intrinsics.json','r') as f:
        INTR = json.loads(f.read())
    cam_K_path = '/media/ama/data0/gz/graspnet/graspnet_real/scenes/scene_'+str(scene_id).zfill(4)+'/camK.npy'
    intrinsics_matrix = np.zeros((3, 3))
    intrinsics_matrix[0, 0] = INTR['fx']
    intrinsics_matrix[1, 1] = INTR['fy']
    intrinsics_matrix[0, 2] = INTR['ppx']
    intrinsics_matrix[1, 2] = INTR['ppy']
    intrinsics_matrix[2, 2] = 1
    np.save(cam_K_path, intrinsics_matrix)