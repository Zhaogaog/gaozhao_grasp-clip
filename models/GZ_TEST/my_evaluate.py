import os
import json
import numpy as np
root_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs'
model_type = 'log_kn_with_rgb_resnet50_frozen_no_resize'
model_type = 'log_kn_with_rgb_clip_rn50_frozen_no_resize'

# epoch_chosen = 'real_dump_epoch10_add_language_frozen_with_rgb'
# data_path = os.path.join(root_dir, model_type, epoch_chosen)
split = 'test_seen'
class my_evaluate():
    def __init__(self, root_dir, model_type, epoch_chosen, obj_top_k, split, ANN_TOP_K):
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
        elif split == 'test_train':
            self.sceneIds = list(range(0, 5))
        elif split == 'test_evaluate':
            self.sceneIds = list(range(100, 105))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        self.TOP_K = ANN_TOP_K
        self.data_path = os.path.join(root_dir, model_type, epoch_chosen, obj_top_k)

    def evaluate(self) :
        ann_grasp_acc = []
        for x in self.sceneIds :
            scene_path = os.path.join(self.data_path, x, 'kinect')
            if not os.path.exists(scene_path) :
                continue
            for ann in os.listdir(scene_path) :
                grasp_acc_list = []
                with open(os.path.join(scene_path, ann),'r') as f:
                    for line in f:
                        grasp_acc = json.loads(line)['grasp_acc']
                        grasp_acc_list.append(grasp_acc)
                if len(grasp_acc_list) > self.TOP_K:
                    grasp_acc_list.sort(reverse=True)
                ann_grasp_acc.append(np.mean(grasp_acc_list[0:self.TOP_K]))

        return np.mean(ann_grasp_acc)
obj_top_k = 'TOP_K_5'
print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=6'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 6).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=7'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 7).evaluate())

obj_top_k = 'TOP_K_3'
print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
#
# obj_top_k = 'TOP_K_1'
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())

