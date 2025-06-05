import os
import json
import numpy as np
root_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs'
model_type = 'log_kn_with_rgb_resnet50_frozen_no_resize'
obj_top_k = 'TOP_K_5'
epoch_chosen = 'dump_epoch10_add_language_frozen_with_rgb'
model_type = 'log_kn_with_rgb_clip_rn50_frozen_no_resize'
# data_path = os.path.join(root_dir, model_type, epoch_chosen)
split = 'test_similar'
class my_evaluate():
    def __init__(self, root_dir, model_type, epoch_chosen, obj_top_k, split):
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
        # self.TOP_K = ANN_TOP_K

        self.real_data_path = os.path.join(root_dir, model_type, 'real_'+epoch_chosen, obj_top_k)
        self.pre_data_path = os.path.join(root_dir, model_type, epoch_chosen, obj_top_k)
    def evaluate_loose(self) :
        # ann_grasp_acc = []
        res_list = []
        res_name_scene = {}
        for x in self.sceneIds :
            scene_path = os.path.join(self.pre_data_path, x, 'kinect')
            if not os.path.exists(scene_path) :
                continue
            for ann in os.listdir(scene_path) :
                grasp_acc_dict = {}
                real_grasp_acc_dict = {}
                with open(os.path.join(scene_path, ann),'r') as f:
                    for line in f:
                        # res_dict = json.loads(line)
                        if json.loads(line)['acc'] != 0:
                            res_list.append(json.loads(line))
                            if json.loads(line)['scene'] not in res_name_scene.keys():
                                res_name_scene[json.loads(line)['scene']] = json.loads(line)['pre'][0]
                            else:
                                if json.loads(line)['pre'][0] not in res_name_scene[json.loads(line)['scene']]:
                                    res_name_scene[json.loads(line)['scene']] = res_name_scene[json.loads(line)['scene']] +'_' + json.loads(line)['pre'][0]
                        # grasp_acc = json.loads(line)['grasp_acc']

        with open(os.path.join(self.pre_data_path, split + '_acc_no_zero.txt'), 'w') as f:
            for res_dict in res_list :
                f.write(json.dumps(res_dict))
                f.write('\n')
        f.close()

        print(res_name_scene)
        # print(res_list)
my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split).evaluate_loose()