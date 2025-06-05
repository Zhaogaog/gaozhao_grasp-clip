import os
import json
import numpy as np
root_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs'
# model_type = 'log_kn_with_rgb_resnet50_frozen_no_resize'
# obj_top_k = 'TOP_K_5'

# model_type = 'log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4'
# data_path = os.path.join(root_dir, model_type, epoch_chosen)
split = 'test_similar'
prec = []
recall = []
root_dir = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4_large_dataset_new_try/real_scene_dump_epoch10_add_language_frozen_with_rgb/CLS/test_similar/'
for scene_id in range(1,5):
    res_path= os.path.join(root_dir, 'scene_{}'.format(str(scene_id).zfill(4)),'0000','res.txt')
    with open(res_path,'r') as f:
        for line in f:
            # grasp_acc = json.loads(line)['grasp_acc']
            pre = json.loads(line)['prec']
            if not np.isnan(pre) :
                # if pre>0.1:
                prec.append(pre)
                recall.append(json.loads(line)['recall'])

print(prec,recall)
print(np.mean(prec),np.mean(recall))
