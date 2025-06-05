import os
import json
import numpy as np
root_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs'
model_type = 'log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4_large_dataset_new_try'
obj_top_k = 'TOP_K_5'
epoch_chosen = 'dump_epoch10_add_language_frozen_with_rgb'
# model_type = 'log_kn_with_rgb_clip_rn50_frozen_no_resize'
# data_path = os.path.join(root_dir, model_type, epoch_chosen)
split = 'test_seen'
class my_evaluate():
    def __init__(self, root_dir, model_type, epoch_chosen, obj_top_k, split):
        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(400, 427))
        elif split == 'test_similar':
            self.sceneIds = list(range(300, 310))
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

    def evaluate_strict(self) :
        # ann_grasp_acc = []
        obj_ratio_list = []
        abs_obj_list = []
        abs_real_obj_list = []
        # ann_ratio_list = []
        # abs_ann_list = []
        # abs_real_ann_list = []
        # scene_ave_0= []
        # abs_
        scene_ave_1= []
        for x in self.sceneIds :
            # scene_obj_ratio_list = []
            # scene_ann_ratio_list = []
            real_scene_path = os.path.join(self.real_data_path, x)
            if not os.path.exists(real_scene_path) :
                continue
            for ann in sorted(os.listdir(real_scene_path)) :
                grasp_acc_dict = {}
                real_grasp_acc_dict = {}
                with open(os.path.join(real_scene_path, ann),'r') as f:
                    for line in f:
                        # grasp_acc = json.loads(line)['grasp_acc']
                        if json.loads(line)['label'][0] != 'NONE':
                            real_grasp_acc_dict[json.loads(line)['pre'][0][-4:]]= json.loads(line)['grasp_acc']
                pre_ann_path = os.path.join(self.pre_data_path,x,ann)
                if os.path.exists(pre_ann_path):
                    with open(pre_ann_path, 'r') as f:
                        for line in f:
                            # grasp_acc = json.loads(line)['grasp_acc']
                            if json.loads(line)['label'][0] != 'NONE':
                                grasp_acc_dict[json.loads(line)['pre'][0][-4:]] = json.loads(line)['grasp_acc']
                # ann_ratio_list = []
                temp_list = []
                for key in real_grasp_acc_dict.keys() :
                    abs_real_obj_list.append(real_grasp_acc_dict[key])
                    if key in grasp_acc_dict.keys():
                        abs_obj_list.append(grasp_acc_dict[key])
                    else:
                        abs_obj_list.append(0)
                        # obj_ratio_list.append(0)
                    if real_grasp_acc_dict[key] ==.0:
                        continue
                    if key in grasp_acc_dict.keys():
                        A=grasp_acc_dict[key]
                        B=real_grasp_acc_dict[key]
                        # if A/B >1:
                        #     print(A/B, ann, key)
                        obj_ratio_list.append(grasp_acc_dict[key]/real_grasp_acc_dict[key])
                        # scene_obj_ratio_list.append(grasp_acc_dict[key]/real_grasp_acc_dict[key])
                        # temp_list.append(grasp_acc_dict[key]/real_grasp_acc_dict[key])
                    else:
                        obj_ratio_list.append(0)
                        # scene_obj_ratio_list.append(0)
                        # temp_list.append(0)
                # if temp_list != []:
                #     ann_ratio_list.append(np.mean(temp_list))
                    # scene_ann_ratio_list.append(np.mean(temp_list))
            # scene_ave_0.append(np.mean(scene_ann_ratio_list))
            # scene_ave_1.append(np.mean(scene_obj_ratio_list))
            # print(x,'every_scene_obj,scene_ann',np.mean(scene_obj_ratio_list), np.mean(scene_ann_ratio_list))
        # print('scene_ave_scene_obj, scene_ann', np.mean(scene_ave_1), np.mean(scene_ave_0))
        # return np.mean(obj_ratio_list), np.mean(ann_ratio_list)
        return np.mean(abs_real_obj_list), np.mean(abs_obj_list), np.mean(obj_ratio_list),
    def evaluate_loose(self) :
        # ann_grasp_acc = []
        obj_ratio_list = []
        ann_ratio_list = []
        abs_obj_list = []
        abs_ann_list = []
        for x in self.sceneIds :
            scene_path = os.path.join(self.pre_data_path, x)
            if not os.path.exists(scene_path) :
                continue
            for ann in os.listdir(scene_path) :
                grasp_acc_dict = {}
                real_grasp_acc_dict = {}
                with open(os.path.join(scene_path, ann),'r') as f:
                    for line in f:
                        if json.loads(line)['label'][0] != 'NONE':
                        # grasp_acc = json.loads(line)['grasp_acc']
                            grasp_acc_dict[json.loads(line)['pre'][0][-4:]]= json.loads(line)['grasp_acc']
                real_ann_path = os.path.join(self.real_data_path,x, ann)
                if os.path.exists(real_ann_path):
                    with open(real_ann_path, 'r') as f:
                        for line in f:
                            if json.loads(line)['label'][0] != 'NONE':
                            # grasp_acc = json.loads(line)['grasp_acc']
                                real_grasp_acc_dict[json.loads(line)['pre'][0][-4:]] = json.loads(line)['grasp_acc']
                # ann_ratio_list = []
                temp_list = []
                for key in grasp_acc_dict.keys() :
                    if key in real_grasp_acc_dict.keys():
                        if real_grasp_acc_dict[key] == 0:
                            continue
                        obj_ratio_list.append(grasp_acc_dict[key]/real_grasp_acc_dict[key])
                        temp_list.append(grasp_acc_dict[key]/real_grasp_acc_dict[key])
                if temp_list != [] :
                    ann_ratio_list.append(np.mean(temp_list))
        return np.mean(ann_ratio_list), np.mean(obj_ratio_list)
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  '.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split).evaluate_strict())
split = 'test_similar'
obj_top_k = 'TOP_K_1'
evaluation = my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split)
print('{} {} {}, abs_real_obj,abs_obj,obj_ratio  '.format(model_type, epoch_chosen, obj_top_k), evaluation.evaluate_strict())
# model_type = 'sam_graspnet'
# epoch_chosen = 'dump_epoch10'
model_type = 'cliport'
epoch_chosen = 'dump_epoch_10_with_cliport'
# obj_top_k = 'TOP_K_1'
evaluation.pre_data_path = os.path.join(root_dir, model_type, epoch_chosen, obj_top_k)
print('{} {} {}, abs_real_obj,abs_obj,obj_ratio  '.format(model_type, epoch_chosen,obj_top_k), evaluation.evaluate_strict())
# print('dump_epoch10_add_language_frozen_with_rgb {}  '.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split).evaluate_loose())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 6).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=6'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 7).evaluate())

# obj_top_k = 'TOP_K_3'
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
#
# obj_top_k = 'TOP_K_1'
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())

