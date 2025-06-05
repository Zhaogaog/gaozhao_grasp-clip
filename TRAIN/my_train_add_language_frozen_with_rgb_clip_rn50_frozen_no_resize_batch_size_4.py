import os
import sys
import numpy as np
from datetime import datetime
import argparse
import time
os.environ['CUDA_VISIBLE_DEVICES']='0'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.GRASPNET.my_graspnet_add_language_frozen_with_rgb_clip_rn50 import GraspNet
from models.LOSS.my_loss_add_language_frozen import get_loss
from dataset.my_graspnet_dataset_add_language_frozen_with_rgb_no_resize import GraspNetDataset, minkowski_collate_fn

import MinkowskiEngine as ME

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--dataset_root', default='/media/ama/data0/gz/graspnet/graspnet_sim', required=False)
# parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None)
parser.add_argument('--model_name', type=str, default='minkuresunet')
parser.add_argument('--log_dir', default='/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4_large_dataset_new_try')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size to process point clouds ')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--resume', action='store_true', default=False, help='Whether to resume from checkpoint')
cfgs = parser.parse_args()
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
# cfgs.checkpoint_path = '/media/ama/data0/gz/graspnet/graspness_unofficial/np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar'
# cfgs.checkpoint_path = '/media/ama/data0/gz/graspnet/graspness_unofficial/np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar'
cfgs.checkpoint_path = '/media/ama/data0/gz/graspnet/graspness_unofficial/WEIGHT/np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar'
cfgs.checkpoint_path = '/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_my_clip_rn50_frozen_no_resize_batch_size_4_large_dataset_new_try/minkuresunet_add_language_frozen_epoch_with_rgb05.tar'
cfgs.resume = True
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train_add_language_frozen_with_rgb.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


# grasp_labels = load_grasp_labels(cfgs.dataset_root)
#dataset内部函数的一些功能
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, split='train',
                                    num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                    remove_outlier=True, augment=False, load_label=True)
print('train dataset length: ', len(TRAIN_DATASET))
# shuffle=True打乱顺序,collate_fn填充。
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=16, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print('train dataloader length: ', len(TRAIN_DATALOADER))
torch.cuda.device_count()
net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=True)

# device = torch.device("cpu")
net.to(device)
# Load the Adam optimizer
# optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):

    # net_state_dict = net.state_dict()
    for (name, param) in net.backbone.named_parameters():
        # if name in checkpoint['model_state_dict']:
        param.requires_grad = False
    for (name, param) in net.graspable.named_parameters():
        # if name in checkpoint['model_state_dict']:
        param.requires_grad = False
    for (name, param) in net.clip.clip_rn50.named_parameters():
        param.requires_grad = False

    # for (name, param) in net.named_parameters():
    #     # if name in checkpoint['model_state_dict']:
    #     param.to(device)
    name_list = []
    # start_epoch = checkpoint['epoch']
    for (name, param) in net.named_parameters():
        if param.requires_grad == True:
            name_list.append(name)
    # print(name_list)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfgs.learning_rate)
    checkpoint = torch.load(CHECKPOINT_PATH)
    if 'minkuresunet' in CHECKPOINT_PATH:
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
    #冻结部分bn层
    # for module in net.backbone.children():
    #     module.train(False)
    # for module in net.graspable.children():
    #     module.train(False)
    # for module in net.rotation.children():
    #     module.train(False)
    # for module in net.crop.children():
    #     module.train(False)
    # for module in net.swad.children():
    #     module.train(False)
    # for module in net.rotation.children():
    #     module.train(False)
    # for m in net.backbone.modules():
    #     if
# torch.distributed.init_process_group('nccl',init_method="tcp://127.0.0.1:23456",world_size=1,rank=0)
# net  = torch.nn.parallel.DistributedDataParallel(net, device_ids=[2])
# net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(net)
# parameters = net.parameters()

# if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
#     checkpoint = torch.load(CHECKPOINT_PATH)
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # optimizer_state_dict_loaded = checkpoint['optimizer_state_dict']
    #
    # optimizer_state_dict = optimizer.state_dict()
    # optimizer.load_state_dict(optimizer_state_dict_loaded)
    # start_epoch = checkpoint['epoch']
    # log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train_add_language_frozen_with_rgb'))
# net.to(device)



def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (0.95 ** epoch)
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    # global net
    net.eval()
    net.lang_select_with_rgb.train()
    net.clip.train()
    net.clip.clip_rn50.eval()
    # net.lang_select.train()#暂时注释
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 20
    recall_count = 0
    t_1 =time.time()
    # net = nn.DataParallel(net)
    for batch_idx, batch_data_label in enumerate(tqdm(TRAIN_DATALOADER, desc='Train')):
        t_2 = time.time()
        # print('total',t_2-t_1)
        t_1 = time.time()
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)
        t1 = time.time()
        end_points = net(batch_data_label)
        # print('one_batch_time:',time.time()-t1)
        time_start_1 = time.time()
        loss, end_points = get_loss(end_points, net)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        time_end_1 = time.time()
        # print("梯度时间：" + str(time_end_1 - time_start_1) + "秒")

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if np.isnan(end_points[key].item()):
                    stat_dict[key] += 0
                else:
                    stat_dict[key] += end_points[key].item()
                    if 'recall' in key:
                        recall_count += 1
                        # print(recall_count)

        if (batch_idx + 1) % batch_interval == 0:
            time_start_1 = time.time()
            log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                if 'recall' in key:
                    TRAIN_WRITER.add_scalar(key, stat_dict[key] / recall_count,
                                            (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                    recall_count = 0
                # time_start_1 = time.time()
                else:

                    TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                # time_end_1 = time.time()
                # print("运行时间：" + str(time_end_1 - time_start_1) + "秒")

                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))

                stat_dict[key] = 0
                time_end_1 = time.time()
                # print("写入时间：" + str(time_end_1 - time_start_1) + "秒")
        # if (batch_idx + 1) % 5000 == 0:
        #     save_dict = {'epoch': EPOCH_CNT + 1, 'optimizer_state_dict': optimizer.state_dict(),
        #                  'model_state_dict': net.state_dict()}
        #     # time_start_1 = time.time()
        #     torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_add_language_frozen_epoch_with_rgb' + str(EPOCH_CNT + 1).zfill(2) + '_' + str(batch_idx + 1) + '.tar'))



def train(start_epoch):
    global EPOCH_CNT
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        torch.cuda.empty_cache()

        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': net.state_dict()}
        time_start_1 = time.time()
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_add_language_frozen_epoch_with_rgb' + str(epoch + 1).zfill(2) + '.tar'))
        time_end_1 = time.time()
        # print("运行时间：" + str(time_end_1 - time_start_1) + "秒")

if __name__ == '__main__':
    train(start_epoch)