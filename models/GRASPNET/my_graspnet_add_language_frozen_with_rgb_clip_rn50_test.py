""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from models.backbone_resunet14 import MinkUNet14D
from models.modules import ApproachNet, GraspableNet, CloudCrop, SWADNet, Lang_selectNet_with_rgb
from loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from models.Clip import my_clip


class GraspNet(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True, img_encoder_frozen=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.num_view = NUM_VIEW
        self.img_encoder_frozen =img_encoder_frozen
        self.clip = my_clip(img_encoder_frozen=self.img_encoder_frozen)
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        #抓取点网络
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)

        #确定最佳接近方向
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.lang_select_with_rgb = Lang_selectNet_with_rgb(seed_feature_dim=self.seed_feature_dim + self.seed_feature_dim)
        self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        #
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        t1 = time.time()
        #文本编码器
        pack_obj_id_batch = end_points['pack_obj_id']
        img = end_points['img']

        img_feat, text_feat = self.clip(img, pack_obj_id_batch)
        mask_remove_outlier_batch = end_points['mask_remove_outlier']
        img_feat = img_feat.transpose(1, 2).transpose(2, 3)
        mask_sampled_batch = end_points['mask_sampled']
        img_feat_list = []
        t1 = time.time()
        for i in range(img_feat.shape[0]):
            mask_remove_outlier = (mask_remove_outlier_batch[i] == 1)
            img_feat_remove_outlier = img_feat[i][mask_remove_outlier]
            mask_sampled = mask_sampled_batch[i]
            img_feat_sampled = img_feat_remove_outlier[mask_sampled]
            img_feat_list.append(img_feat_sampled)
        img_feat = torch.stack(img_feat_list, dim=0)

        # text_feat, _, _ = _clip_text_encoder(pack_obj_id_batch).encode_sentence()
        # print('language_select_time:', time.time() - t1)
        #点云编码器
        #（4，15000，3）点的坐标
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
        B, point_num, _ = seed_xyz.shape  # batch _size
        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)
        #（B,512，15000)
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        objectness_score = end_points['objectness_score']
        #(4，2，15000)
        graspness_score = end_points['graspness_score'].squeeze(1)
        # (4，15000)
        objectness_pred = torch.argmax(objectness_score, 1)
        #(4，15000), 将2压缩为1 （第二维最大值的索引）0大表示无关的物体，1表示有关的物体
        objectness_mask = (objectness_pred == 1)
        # (4，15000) True 或者 False
        objectness_mask_num = torch.sum(objectness_mask.float())

        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        # (4，15000) True 或者 False
        graspable_mask = objectness_mask & graspness_mask

        seed_features_graspable = []
        seed_xyz_graspable = []
        seed_img_features_graspable = []
        seed_objectness_graspable = []
        graspable_num_batch = 0.
        batch_good_list = torch.zeros_like(end_points['pack_obj_id'])
        # print('language_select_time:', time.time() - t1)
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_objectness_label = end_points['objectness_label'][i][cur_mask] #Ns
            cur_objectness_label = cur_objectness_label.unsqueeze(0).unsqueeze(-1) #1,Ns,1
            #挑选出来可以抓的点的512特征，和三维坐标，Ns调试时可认为是9721
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_img_feat = img_feat[i][cur_mask]
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3
            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            #1024
            if cur_seed_xyz.shape[1] != 0:
                # print('训练合格的点数为：', cur_seed_xyz.shape[1])
                batch_good_list[i,0] = 1
                fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
                cur_objectness_label_flipped = cur_objectness_label.transpose(1, 2).contiguous()#1,1,Ns
                cur_objectness_label = gather_operation(cur_objectness_label_flipped.float(), fps_idxs).transpose(1, 2).squeeze(0).squeeze(-1).contiguous() #Ns,1
                cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
                cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
                # （1024，3）
                cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
                cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns
                cur_img_feat_flipped = cur_img_feat.unsqueeze(0).transpose(1, 2).contiguous()
                cur_img_feat = gather_operation(cur_img_feat_flipped, fps_idxs).squeeze(0).contiguous()  # feat_dim*Ns
                #（512，1024）
                seed_objectness_graspable.append(cur_objectness_label)
                seed_features_graspable.append(cur_feat)
                seed_xyz_graspable.append(cur_seed_xyz)
                seed_img_features_graspable.append(cur_img_feat)
            else:
                continue
        # print('language_select_time:', time.time() - t1)
        end_points['batch_good_list']= batch_good_list
        # end_points['graspable_count_stage1'] = graspable_num_batch / B  # 平均每个可以被抓起来的点数
        # if  torch.sum(batch_good_list) != 0:
        seed_objectness_graspable = torch.stack(seed_objectness_graspable,0)
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3 1024
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns 1024
        seed_img_features_graspable = torch.stack(seed_img_features_graspable, 0)
        #当进一步把点数减少后，如果剩余的点数不足1024怎么办。减少为128
        end_points['objectness_graspable_label'] = seed_objectness_graspable
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B #平均每个可以被抓起来的点数


        end_points = self.lang_select_with_rgb(seed_features_graspable, seed_img_features_graspable, text_feat, end_points)

        # print('language_select_time:', time.time() - t1)
        end_points, res_feat = self.rotation(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            end_points = process_grasp_labels(end_points)#获取1024个点对应的标签
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        group_features = self.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        end_points = self.swad(group_features, end_points)

        return end_points


def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    res = []
    for i in range(batch_size):
        obj_id = end_points['pack_obj_id'][i].item()
        lang_select_objectness = end_points['lang_select_objectness'][i]
        objectness_graspable_label = end_points['objectness_graspable_label'][i]
        # (2，1024)
        lang_select_objectness_pred = torch.argmax(lang_select_objectness, 0)
        # (1024,), 将2压缩为1 （第二维最大值的索引）0大表示无关的物体，1表示有关的物体
        # lang_select_objectness_mask = (lang_select_objectness_pred == 1)

        acc = (lang_select_objectness_pred == objectness_graspable_label.long()).float().mean().item()
        prec = (lang_select_objectness_pred == objectness_graspable_label.long())[
            lang_select_objectness_pred == 1].float().mean().item()
        recall = (lang_select_objectness_pred == objectness_graspable_label.long())[
            objectness_graspable_label == 1].float().mean().item()
        num_label = torch.sum(objectness_graspable_label).item()
        num_pred = torch.sum(lang_select_objectness_pred).item()
        lang_select_objectness_ratio = lang_select_objectness[1,:]/(lang_select_objectness[0,:]+1e-7)
        index = torch.argsort(-lang_select_objectness[1,:]/(lang_select_objectness[0,:]+1e-7))
        lang_select_objectness_sorted = lang_select_objectness_ratio[index]
        lang_select_objectness_mask = (lang_select_objectness_sorted > 1)
        lang_select_objectness_mask = index[lang_select_objectness_mask]
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = torch.div(grasp_score_inds, NUM_DEPTH, rounding_mode='trunc') * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = obj_id * torch.ones_like(grasp_score)
        # obj_ids = obj_id
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1)[lang_select_objectness_mask])
        res.append({'num_label':num_label, 'num_pred': num_pred, 'acc':acc, 'prec':prec, 'recall':recall})
    return grasp_preds, res

def real_pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    res = []
    for i in range(batch_size):
        obj_id = end_points['pack_obj_id'][i].item()
        lang_select_objectness = end_points['lang_select_objectness'][i]
        objectness_graspable_label = end_points['objectness_graspable_label'][i]

        objectness_graspable_label_mask = (objectness_graspable_label==1)
        # (2，1024)
        lang_select_objectness_pred = torch.argmax(lang_select_objectness, 0)
        # (1024,), 将2压缩为1 （第二维最大值的索引）0大表示无关的物体，1表示有关的物体

        acc = (lang_select_objectness_pred == objectness_graspable_label.long()).float().mean().item()
        prec = (lang_select_objectness_pred == objectness_graspable_label.long())[
            lang_select_objectness_pred == 1].float().mean().item()
        recall = (lang_select_objectness_pred == objectness_graspable_label.long())[
            objectness_graspable_label == 1].float().mean().item()
        num_label = torch.sum(objectness_graspable_label).item()
        num_pred = torch.sum(lang_select_objectness_pred).item()
        lang_select_objectness_ratio = lang_select_objectness[1,:]/(lang_select_objectness[0,:]+1e-7)
        index = torch.argsort(-lang_select_objectness[1,:]/(lang_select_objectness[0,:]+1e-7))
        lang_select_objectness_sorted = lang_select_objectness_ratio[index]
        lang_select_objectness_mask = (lang_select_objectness_sorted > 1)
        lang_select_objectness_mask = index[lang_select_objectness_mask]
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = torch.div(grasp_score_inds, NUM_DEPTH, rounding_mode='trunc') * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = obj_id * torch.ones_like(grasp_score)
        # obj_ids = obj_id
        # grasp_preds.append(
        #     torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1)[lang_select_objectness_mask])
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1)[objectness_graspable_label_mask])
        res.append({'num_label':num_label, 'num_pred': num_pred, 'acc':acc, 'prec':prec, 'recall':recall})
    return grasp_preds, res