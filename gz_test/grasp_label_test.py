import numpy as np
import os
import cv2
dataset_root = '/home/gaozhao/graspnet'
file = np.load(os.path.join(dataset_root, 'collision_label', 'scene_0000', 'collision_labels.npz'))
print(file['arr_0'].shape[0])
file = np.load(os.path.join(dataset_root, 'grasp_label', '014_labels.npz'))
print(file['points'].shape[0])
file = cv2.imread(os.path.join(dataset_root, 'scenes', 'scene_0000', 'kinect', 'label', '0000.png'),)
h,w,_= file.shape
idx = set()
for i in range(h):
    for j in range(w):
        idx.add(file[i,j,1]-1)
        idx.add(file[i, j, 0]-1)
        idx.add(file[i, j, 2]-1)
print(idx)

