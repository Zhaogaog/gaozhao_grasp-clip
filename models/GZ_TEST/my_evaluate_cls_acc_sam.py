import os
import json
import numpy as np
root_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs'
# model_type = 'log_kn_with_rgb_resnet50_frozen_no_resize'
# obj_top_k = 'TOP_K_5'

# model_type = 'log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4'
# data_path = os.path.join(root_dir, model_type, epoch_chosen)
split = 'test_similar'
class my_evaluate():
    def __init__(self, root_dir, model_type, epoch_chosen, split):
        if split == 'train':
            self.sceneIds = list(range(142))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(400, 427))
        elif split == 'test_similar':
            # self.sceneIds = list(range(27, 39))
            # self.sceneIds = list(range(258,259))
            self.sceneIds = list(range(300, 310))
            # self.sceneIds = list(range(200, 212))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        elif split == 'test_train':
            self.sceneIds = list(range(0, 5))
        elif split == 'test_evaluate':
            self.sceneIds = list(range(400, 403))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        # self.TOP_K = ANN_TOP_K

        self.real_data_path = os.path.join(root_dir, model_type, 'real_'+epoch_chosen, )
        self.pre_data_path = os.path.join(root_dir, model_type, epoch_chosen, 'CLS', split )

    def evaluate_strict(self) :
        # ann_grasp_acc = []
        obj_ratio_list = []
        ann_ratio_list = []
        for x in self.sceneIds :
            real_scene_path = os.path.join(self.real_data_path, x)
            if not os.path.exists(real_scene_path) :
                continue
            for ann in os.listdir(real_scene_path) :
                grasp_acc_dict = {}
                real_grasp_acc_dict = {}
                with open(os.path.join(real_scene_path, ann),'r') as f:
                    for line in f:
                        # grasp_acc = json.loads(line)['grasp_acc']
                        real_grasp_acc_dict[json.loads(line)['pre'][0][-4:]]= json.loads(line)['grasp_acc']
                pre_ann_path = os.path.join(self.pre_data_path,x, ann)
                if os.path.exists(pre_ann_path):
                    with open(pre_ann_path, 'r') as f:
                        for line in f:
                            # grasp_acc = json.loads(line)['grasp_acc']
                            grasp_acc_dict[json.loads(line)['pre'][0][-4:]] = json.loads(line)['grasp_acc']
                # ann_ratio_list = []
                temp_list = []
                for key in real_grasp_acc_dict.keys() :
                    if real_grasp_acc_dict[key] ==0:
                        continue
                    if key in grasp_acc_dict.keys():
                        obj_ratio_list.append(grasp_acc_dict[key]/real_grasp_acc_dict[key])
                        temp_list.append(grasp_acc_dict[key]/real_grasp_acc_dict[key])
                    else:
                        obj_ratio_list.append(0)
                        temp_list.append(0)
                if temp_list != []:
                    ann_ratio_list.append(np.mean(temp_list))

        return np.mean(ann_ratio_list), np.mean(obj_ratio_list)
    def evaluate_loose(self) :
        # ann_grasp_acc = []
        # obj_ratio_list = []
        # ann_ratio_list = []
        acc = []
        prec = []
        recall = []
        scene_acc = []
        scene_prec = []
        scene_recall = []
        for x in self.sceneIds :
            cur_acc = []
            cur_prec = []
            cur_recall = []
            scene_path = os.path.join(self.pre_data_path, x,)
            if not os.path.exists(scene_path) :
                continue
            for ann in sorted(os.listdir(scene_path)) :

                for obj in os.listdir(os.path.join(scene_path,ann)) :
                    # grasp_acc_dict = {}
                    # real_grasp_acc_dict = {}
                    #
                    if 'res' in obj:
                        continue
                    with open(os.path.join(scene_path, ann, obj),'r') as f:
                        for line in f:
                            if json.loads(line)['num_label'] == 0:
                                continue
                            # grasp_acc = json.loads(line)['grasp_acc']
                            if not np.isnan(json.loads(line)['acc']):
                                acc.append(json.loads(line)['acc'])
                                cur_acc.append(json.loads(line)['acc'])
                            if not np.isnan(json.loads(line)['prec']):
                                prec.append(json.loads(line)['prec'])
                                cur_prec.append(json.loads(line)['prec'])
                            else:
                                prec.append(0)
                                cur_prec.append(0)
                            if not np.isnan(json.loads(line)['recall']):
                                recall.append(json.loads(line)['recall'])
                                cur_recall.append(json.loads(line)['recall'])
                # print(x, ann, 'done')
            print(x, np.mean(cur_acc), np.mean(cur_prec), np.mean(cur_recall))
            scene_acc.append(np.mean(cur_acc))
            scene_prec.append(np.mean(cur_prec))
            scene_recall.append(np.mean(cur_recall))
        print('scene_ave', np.mean(scene_acc), np.mean(scene_prec), np.mean(scene_recall))
        # np.savetxt(os.path.join(self.pre_data_path, 'acc.txt'),np.array([np.mean(acc)]))
        # np.savetxt(os.path.join(self.pre_data_path, 'prec.txt'), np.array([np.mean(prec)]))
        # np.savetxt(os.path.join(self.pre_data_path, 'recall.txt'), np.array([np.mean(recall)]))
        return np.mean(acc), np.mean(prec), np.mean(recall)
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print('real_dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=5'.format(obj_top_k), my_evaluate(root_dir, model_type, 'real_dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 5).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=1'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 1).evaluate())
# print('dump_epoch10_add_language_frozen_with_rgb {}  ANN_TOPK=3'.format(obj_top_k), my_evaluate(root_dir, model_type, 'dump_epoch10_add_language_frozen_with_rgb', obj_top_k, split, 3).evaluate())
# print(f'{epoch_chosen} ', my_evaluate(root_dir, model_type, epoch_chosen,  split).evaluate_strict())

# model_type = 'log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4_large_dataset_with_L2loss_1e-05'
# model_type = 'log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4'
# data_path = os.path.join(root_dir, model_type, epoch_chosen)
if __name__ == '__main__':
    epoch_chosen = 'dump_epoch10'
    model_type = 'sam_graspnet'
    split = 'test_similar'
    evaluation = my_evaluate(root_dir, model_type, epoch_chosen, split)
    print(f'{model_type} {epoch_chosen} {split}', evaluation.evaluate_loose())
# model_type = 'sam_graspnet'
# epoch_chosen = 'dump_epoch10'
# evaluation.pre_data_path = os.path.join(root_dir, model_type, epoch_chosen, 'SAM', split )
# print(f'{model_type} {epoch_chosen} {split}', evaluation.evaluate_loose())

# epoch_chosen = 'dump_epoch06_add_language_frozen_with_rgb'
# split = 'test_train'
# print(f'{epoch_chosen} ', my_evaluate(root_dir, model_type, epoch_chosen, split).evaluate_loose())


# epoch_chosen = 'dump_epoch15_add_language_frozen_with_rgb'
# print(f'{epoch_chosen} ', my_evaluate(root_dir, model_type, epoch_chosen, split).evaluate_loose())
#
# epoch_chosen = 'dump_epoch20_add_language_frozen_with_rgb'
# print(f'{epoch_chosen} ', my_evaluate(root_dir, model_type, epoch_chosen, split).evaluate_loose())
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

