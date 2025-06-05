import json
import os
import numpy as np
sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in range(500,512)]
root_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs'
# model_type = 'log_kn_with_rgb_resnet50_frozen_no_resize'
# obj_top_k = 'TOP_K_5'
epoch_chosen = 'dump_epoch10_add_language_frozen_with_rgb'
model_type = 'log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4_large_dataset_new_try'
# model_type = 'log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4'
# data_path = os.path.join(root_dir, model_type, epoch_chosen)
split = 'test_similar'
if split == 'test_novel':
    sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in range(500, 512)]
if split == 'test_similar':
    sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in range(310, 311)]
pre_data_path = os.path.join(root_dir, model_type, epoch_chosen, 'CLS', split )
with open('/media/ama/data0/gz/graspnet/graspnet_sim/obj_lists/all_category_' + split + '_dict.json', 'r') as f:
    obj_classes_dict = json.loads(f.read())
obj_classes_dict = {v: k for k, v in obj_classes_dict.items()}
for x in sceneIds:
    obj_dict_ave_cls = {}
    # cur_acc = []
    # cur_prec = []
    # cur_recall = []
    scene_path = os.path.join(pre_data_path, x, )

    if not os.path.exists(scene_path):
        continue
    for ann in sorted(os.listdir(scene_path)):
        for obj in os.listdir(os.path.join(scene_path, ann)):
            # grasp_acc_dict = {}
            # real_grasp_acc_dict = {}
            #
            if 'res' in obj:
                continue
            with open(os.path.join(scene_path, ann, obj), 'r') as f:
                for line in f:
                    if json.loads(line)['num_label'] == 0:
                        continue
                    # grasp_acc = json.loads(line)['grasp_acc']
                    if obj_classes_dict[int(obj.split('.')[0])] not in obj_dict_ave_cls.keys():
                        if not np.isnan(json.loads(line)['prec']):
                            obj_dict_ave_cls[obj_classes_dict[int(obj.split('.')[0])]] = [json.loads(line)['prec']]
                        else:
                            obj_dict_ave_cls[obj_classes_dict[int(obj.split('.')[0])]] = [0]
                    else:
                        if not np.isnan(json.loads(line)['prec']):
                            obj_dict_ave_cls[obj_classes_dict[int(obj.split('.')[0])]].append(json.loads(line)['prec'])
                        else:
                            obj_dict_ave_cls[obj_classes_dict[int(obj.split('.')[0])]].append(0)


    # print(obj_dict_ave_cls)
    for key in obj_dict_ave_cls.keys():
        obj_dict_ave_cls[key] = np.mean(obj_dict_ave_cls[key])
        print(key, obj_dict_ave_cls[key])
        # if obj_dict_ave_cls[key]> 0.5:
        #     with open(
        #             '/media/ama/data0/gz/graspnet/graspnet_sim/obj_lists/all_category_' + 'test_novel_good' + '_dict.json',
        #             'a') as f:

                # f.write(json.dumps({key:obj_dict_ave_cls[key]}))
                # f.write('\n')
    # with open('/media/ama/data0/gz/graspnet/graspnet_sim/obj_lists/all_category_' + 'test_novel_good' + '_dict.json', 'a') as f:

    # print(obj_dict_ave_cls)