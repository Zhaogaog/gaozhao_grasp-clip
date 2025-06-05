import os
import numpy as np
import torch
import json
dataset_root = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen'
from Clip import _clip_text_encoder
encoder = _clip_text_encoder()
path = os.path.join(dataset_root, 'cls_res_kinect_test_seen.txt')
print(path)
with open(path, 'r') as f:
    a= f.readlines()
    print(f.readlines())
    # for line in f:
    #     scene = list(line)
    #     print(line)
    #     for ann  in scene:
    #         for obj in ann:
    #             print(obj)
    #             break
    #         break
    #     break

# f.close()
# id_np = np.array(id_list).reshape(len(id_list),1)
# encoder.text = torch.tensor(id_np)
# encoder.get_classes()
# sen_list = encoder.sentences_list
# for i in range(len(sen_list)):
#     sen_list[i] = sen_list[i][12:]
# save_path = os.path.join(dataset_root, 'scenes', 'scene_' + str(scene).zfill(4), 'object_name_list.txt' )
# with open(save_path, 'w') as f:
#     for i in range(len(sen_list)):
#         f.write(sen_list[i])
#         f.write('\n')
# f.close()









