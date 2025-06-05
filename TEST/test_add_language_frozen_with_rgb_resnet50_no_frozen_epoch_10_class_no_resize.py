import os
import sys
import numpy as np
import argparse
import time
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval_add_language_frozen import GraspGroup, GraspNetEval
# from models.Clip import _clip_text_encoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from utils.collision_detector import ModelFreeCollisionDetector
from models.GRASPNET.graspnet_add_language_frozen_with_rgb_resnet50_test import GraspNet, pred_decode
from dataset.graspnet_dataset_add_language_frozen_with_rgb_no_resize import GraspNetDataset, minkowski_collate_fn
from models.Clip import my_clip

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/ama/data0/gz/graspnet/graspnet', required=False)
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default='/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_resnet50_no_frozen_no_resize/minkuresunet_add_language_frozen_epoch_with_rgb10.tar', required=False)
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_resnet50_no_frozen_no_resize/dump_epoch10_add_language_frozen_with_rgb', required=False)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=True)
parser.add_argument('--eval', action='store_true', default=True)
cfgs = parser.parse_args()
cfgs.collision_thresh = -1
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference(split):
    test_dataset = GraspNetDataset(cfgs.dataset_root, split=split, camera=cfgs.camera, num_points=cfgs.num_point,
                                   voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False)
    print('Test dataset length: ', len(test_dataset))
    scene_list = test_dataset.scene_list()
    ann_list = test_dataset.ann_list()
    obj_list = test_dataset.obj_id_list()
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=8, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
    print('Test dataloader length: ', len(test_dataloader))
    # Init the model
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False, img_encoder_frozen=False)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint1 = torch.load('/media/ama/data0/gz/graspnet/graspness_unofficial/WEIGHT/np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar')
    net.load_state_dict(checkpoint1['model_state_dict'], strict=False)
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    batch_interval = 100
    net.eval()
    tic = time.time()
    # tqdm(TRAIN_DATALOADER, desc='Train')
    # for batch_idx, batch_data_label in enumerate(tqdm(TRAIN_DATALOADER, desc='Train')):
    for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc='inference')):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            t1 = time.time()
            end_points = net(batch_data)
            # print('train_time:',time.time()-t1)
            t1 = time.time()
            grasp_preds, res = pred_decode(end_points)
            # print('pred_time:', time.time() - t1)
        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            cur_res = res[i]
            gg = GraspGroup(preds)
            # collision detection
            if cfgs.collision_thresh > 0:
                cloud = test_dataset.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, 'SORTED', scene_list[data_idx], cfgs.camera, ann_list[data_idx])
            save_path = os.path.join(save_dir, obj_list[data_idx] + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

            save_dir_res = os.path.join(cfgs.dump_dir, 'CLS', split, scene_list[data_idx], cfgs.camera, ann_list[data_idx])
            if not os.path.exists(save_dir_res):
                os.makedirs(save_dir_res)
            # save_path = os.path.join(save_dir, obj_list[data_idx] + '.npy')
            with open(os.path.join(save_dir_res, obj_list[data_idx]+'.txt'), 'a') as f:
                f.write(json.dumps(cur_res))
                f.write('\n')
            f.close()
            cur_res['obj'] = obj_list[data_idx]
            with open(os.path.join(save_dir_res, 'res.txt'), 'a') as f:
                f.write(json.dumps(cur_res))
                f.write('\n')
            f.close()

            # gg.save_npy(save_path)

        if (batch_idx + 1) % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx + 1, (toc - tic) / batch_interval))
            tic = time.time()


def evaluate(dump_dir, split):
    return_grasp_acc = True
    return_cls_acc= True
    use_origin = False
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    # split = 'test_seen'
    t1 = time.time()
    if split =='test_seen':

        res_list, cls_acc_list, cls_res_list, ap, cls_ap= ge.eval_seen(dump_folder=dump_dir, return_grasp_acc=return_grasp_acc, return_cls_acc=return_cls_acc, use_origin=use_origin, proc=6)
    elif split =='test_similar':
        res_list, cls_acc_list, cls_res_list, ap, cls_ap = ge.eval_similar(dump_folder=dump_dir,
                                                                        return_grasp_acc=return_grasp_acc,
                                                                        return_cls_acc=return_cls_acc,
                                                                        use_origin=use_origin, proc=6)
    elif split =='test_novel':
        res_list, cls_acc_list, cls_res_list, ap, cls_ap = ge.eval_novel(dump_folder=dump_dir,
                                                                        return_grasp_acc=return_grasp_acc,
                                                                        return_cls_acc=return_cls_acc,
                                                                        use_origin=use_origin, proc=6)
    elif split =='test':
        res_list, cls_acc_list, cls_res_list, ap, cls_ap = ge.eval_all(dump_folder=dump_dir,
                                                                        return_grasp_acc=return_grasp_acc,
                                                                        return_cls_acc=return_cls_acc,
                                                                        use_origin=use_origin, proc=6)
    elif split =='test_train':
        res_list, cls_acc_list, cls_res_list, ap, cls_ap = ge.eval_train(dump_folder=dump_dir,
                                                                        return_grasp_acc=return_grasp_acc,
                                                                        return_cls_acc=return_cls_acc,
                                                                        use_origin=use_origin, proc=5)
    elif split =='test_evaluate':
        res_list, cls_acc_list, cls_res_list, ap, cls_ap = ge.eval_evaluate(dump_folder=dump_dir,
                                                                        return_grasp_acc=return_grasp_acc,
                                                                        return_cls_acc=return_cls_acc,
                                                                        use_origin=use_origin, proc=5)
    # cls_acc_list, cls_res_list, cls_ap = ge.eval_seen(dump_folder=dump_dir,
    #                                                                 return_grasp_acc=return_grasp_acc,
    #                                                                 return_cls_acc=return_cls_acc,
    #                                                                 use_origin=use_origin, proc=6)
    save_dir_res = os.path.join(cfgs.dump_dir, 'res_{}_{}.txt'.format(cfgs.camera, split))
    with open(save_dir_res, 'w') as f:
        for list_ in res_list:
            f.write(str(list_))
            f.write('\n')
    f.close()
    save_dir_ap = os.path.join(cfgs.dump_dir, 'ap_{}_{}.txt'.format(cfgs.camera, split))
    np.savetxt(save_dir_ap, np.array([ap]))
    # print('test_seen_time:',time.time()-t1)

    save_dir_res = os.path.join(cfgs.dump_dir, 'cls_acc_{}_{}.txt'.format(cfgs.camera, split))
    with open(save_dir_res, 'w') as f:
        for list_ in cls_acc_list:
            f.write(str(list_))
            f.write('\n')
    f.close()
    save_dir_ap = os.path.join(cfgs.dump_dir, 'cls_ap_{}_{}.txt'.format(cfgs.camera, split))
    np.savetxt(save_dir_ap, np.array([cls_ap]))
    # # print('test_seen_time:', time.time() - t1)

    # save_dir_res = os.path.join(cfgs.dump_dir, 'cls_res_{}_{}.txt'.format(cfgs.camera, split))

    # encoder = my_clip()
    # for scene in cls_res_list:
    #     # if scene
    #     for ann in scene:
    #         if len(ann) == 0:
    #             continue
    #         id_list = []
    #         # print(ann)
    #         path = os.path.join(cfgs.dataset_root, 'scenes', 'scene_' + str(ann[0]['scene']).zfill(4),
    #                             'object_id_list.txt')
    #         with open(path, 'r') as f:
    #             for line in f:
    #                 id_list.append(int(line))
    #         # save_dir_root = os.path.join(cfgs.dump_dir, split)
    #         save_dir = os.path.join(cfgs.dump_dir, split, 'scene_' + str(ann[0]['scene']).zfill(4), cfgs.camera)
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         with open(os.path.join(save_dir, str(ann[0]['ann']).zfill(4)),'w') as f:
    #             for obj in ann:
    #                 # save_dir_root = os.path.join(cfgs.dump_dir, split)
    #                 # save_dir = os.path.join(cfgs.dump_dir, split, 'scene_' + str(obj['scene']).zfill(4), cfgs.camera, str(obj['ann']).zfill(4))
    #
    #                 label = np.array(id_list)[obj['cls_res'][0]]
    #                 pre = np.array(id_list)[obj['cls_res'][1]]
    #
    #                 id_np = label.reshape(label.shape[0],1)
    #                 encoder.text = torch.tensor(id_np)
    #                 encoder.get_classes()
    #                 sen_list = encoder.sentences_list
    #                 for i in range(len(sen_list)):
    #                     sen_list[i] = sen_list[i][12:]
    #                     sen_list[i] = sen_list[i] + '_' + str(id_np[i][0]).zfill(4)
    #                 sen_label = sen_list
    #
    #                 id_np = pre.reshape(pre.shape[0], 1)
    #                 encoder.text = torch.tensor(id_np)
    #                 encoder.get_classes()
    #                 sen_list = encoder.sentences_list
    #                 for i in range(len(sen_list)):
    #                     sen_list[i] = sen_list[i][12:]
    #                     sen_list[i] = sen_list[i] + '_' + str(id_np[i][0]).zfill(4)
    #                 sen_pre = sen_list
    #
    #                 dict_cls_res  = {}
    #                 dict_cls_res['scene'] = obj['scene']
    #                 dict_cls_res['ann'] = obj['ann']
    #                 dict_cls_res['acc'] = obj['acc']
    #                 dict_cls_res['num'] = len(sen_list)
    #                 dict_cls_res['pre'] = sen_pre
    #                 dict_cls_res['label'] = sen_label
    #
    #                 # print(dict_cls_res)
    #                 f.write(json.dumps(dict_cls_res))
    #                 f.write('\n')
    #         f.close()
                    #     save_path = os.path.join(dataset_root, 'scenes', 'scene_' + str(scene).zfill(4),
                    #                              'object_name_list.txt')
                    #     with open(save_path, 'w') as f:
                    #         for i in range(len(sen_list)):
                    #             f.write(sen_list[i])
                    #             f.write('\n')
                    #     f.close()
                    #
                    # save_path = os.path.join(save_dir, obj_list[data_idx] + '.npy')
                    # if not os.path.exists(save_dir):
                    #     os.makedirs(save_dir)
                    # gg.save_npy(save_path)
                    # save_dir
                    # f.write(json.dumps(obj))
                    # f.write('\n')
    # save_dir_res = os.path.join(cfgs.dump_dir, 'res_{}_{}.txt'.format(cfgs.camera, split))
    # with open(save_dir_res, 'w') as f:
    #     for list_ in res:
    #         f.write(str(list_))
    #         f.write('\n')
    # f.close()
    # save_dir_ap = os.path.join(cfgs.dump_dir, 'ap_{}_{}.npy'.format(cfgs.camera, split))
    # np.save(save_dir_ap, ap)
    # print('test_seen_time:',time.time()-t1)

    # split = 'test_similar'
    # res, ap = ge.eval_similar(dump_folder=dump_dir, proc=6)
    # save_dir_res = os.path.join(cfgs.dump_dir, 'res_{}_{}.txt'.format(cfgs.camera, split))
    # with open(save_dir_res, 'w') as f:
    #     for list_ in res:
    #         f.write(str(list_))
    #         f.write('\n')
    # f.close()
    # save_dir_ap = os.path.join(cfgs.dump_dir, 'ap_{}_{}.npy'.format(cfgs.camera, split))
    # np.save(save_dir_ap, ap)
    #
    # split = 'test_novel'
    # res, ap = ge.eval_novel(dump_folder=dump_dir, proc=6)
    # save_dir_res = os.path.join(cfgs.dump_dir, 'res_{}_{}.txt'.format(cfgs.camera, split))
    # with open(save_dir_res, 'w') as f:
    #     for list_ in res:
    #         f.write(str(list_))
    #         f.write('\n')
    # f.close()
    # save_dir_ap = os.path.join(cfgs.dump_dir, 'ap_{}_{}.npy'.format(cfgs.camera, split))
    # np.save(save_dir_ap, ap)


if __name__ == '__main__':
    # inference('test_evaluate')
    evaluate(os.path.join(cfgs.dump_dir,'SORTED'), 'test_evaluate')
    # inference('test_train')
    # evaluate(os.path.join(cfgs.dump_dir,'SORTED'), 'test_train')
    # inference('test_similar')
    # inference('test_seen')
    # inference('test_similar')
    # inference('test_novel')

    # inference('test_novel')
    # evaluate(cfgs.dump_dir, 'test_seen')


    # if cfgs.infer:
    # inference('test_similar')
    # inference('test_novel')
    # if cfgs.eval:

    # evaluate(cfgs.dump_dir, 'test_similar')
    # evaluate(cfgs.dump_dir, 'test_novel')
