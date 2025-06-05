import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)(4，3, 15000)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points

class Lang_selectNet_with_rgb(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        # self.linear = nn.Linear(self.in_dim)
        self.text_enc_dim = 1024
        mlps = [self.in_dim, 512, self.in_dim]

        self.conv1 = nn.Conv1d(512,512,1)
        self.conv2 = nn.Conv1d(16, 512, 1)
        self.mlps1 = pt_utils.SharedMLP(mlps, bn=True)
        self.mlps2 = pt_utils.SharedMLP(mlps, bn=True)
        self.mlps3 = pt_utils.SharedMLP(mlps, bn=True)
        self.mlps4 = pt_utils.SharedMLP(mlps, bn=True)
        self.mlps5 = pt_utils.SharedMLP([self.in_dim, 512, 2], bn=True)

        self.lang_proj1 = nn.Linear(self.text_enc_dim, self.in_dim)
        self.lang_proj2 = nn.Linear(self.text_enc_dim, self.in_dim)
        self.lang_proj3 = nn.Linear(self.text_enc_dim, self.in_dim)
        self.lang_proj4 = nn.Linear(self.text_enc_dim, self.in_dim)

        self.bn_x = nn.BatchNorm1d(512)
        self.bn_emb = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(self.in_dim)
        self.bn3 = nn.BatchNorm1d(self.in_dim)
        self.bn4 = nn.BatchNorm1d(self.in_dim)
        self.bn5 = nn.BatchNorm1d(self.in_dim)
        self.bn6 = nn.BatchNorm1d(self.in_dim)
        self.bn7 = nn.BatchNorm1d(self.in_dim)
        self.bn8 = nn.BatchNorm1d(self.in_dim)

        # self.bn = nn.BatchNorm1d(2*self.in_dim)


    def forward(self, x, emb, l, end_points):
        l = l.float()
        x = F.relu(self.conv1(x), inplace=True)
        x = self.bn_x(x)
        emb = F.relu(self.conv2(emb), inplace=True)
        emb = self.bn_emb(emb)
        x = torch.cat([x, emb], dim=1)

        # l1 = self.lang_proj1(l)
        # l1 = l1.unsqueeze(-1).expand(l1.shape[0], l1.shape[1], x.shape[2])
        # x_norm = self.bn1(x)
        # x_mul = x * l1
        # # x = self.bn2(x_mul.unsqueeze(-1)).squeeze(-1)
        # x = self.bn2(x_mul)
        # x = x + x_norm
        # x = self.mlps1(x)

        # l2 = self.lang_proj2(l)
        # l2 = l2.unsqueeze(-1).expand(l2.shape[0], l2.shape[1], x.shape[2])
        # x_norm = self.bn3(x)
        # x_mul = x * l2
        # x = self.bn4(x_mul)
        # x = x + x_norm
        # x = self.mlps2(x)

        # l3 = self.lang_proj3(l)
        # l3 = l3.unsqueeze(-1).expand(l3.shape[0], l3.shape[1], x.shape[2])
        # x_norm = self.bn5(x)
        # x_mul = x * l3
        # x = self.bn6(x_mul)
        # x = x + x_norm
        # x = self.mlps3(x)

        # l4 = self.lang_proj4(l)
        # l4 = l4.unsqueeze(-1).expand(l4.shape[0], l4.shape[1], x.shape[2])
        # x_norm = self.bn7(x)
        # x_mul = x * l4
        # x = self.bn8(x_mul)
        # x = x + x_norm
        # x = self.mlps4(x)

        x = self.mlps5(x)
        end_points['lang_select_objectness'] = x

        # self.mlps = pt_utils.SharedMLP(mlps, bn=True)
        # graspable_scor(e = self.conv_graspable(seed_features)  # (B, 3, num_seed)(4，3, 15000)
        # end_points['objectness_score'] = graspable_score[:, :2]
        # end_points['graspness_score'] = graspable_score[:, 2]
        return end_points

class Lang_selectNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        # self.linear = nn.Linear(self.in_dim)
        self.text_enc_dim = 1024
        mlps = [self.in_dim, 512, self.in_dim]

        self.mlps1 = pt_utils.SharedMLP(mlps, bn=True)
        self.mlps2 = pt_utils.SharedMLP(mlps, bn=True)
        self.mlps3 = pt_utils.SharedMLP(mlps, bn=True)
        self.mlps4 = pt_utils.SharedMLP(mlps, bn=True)
        self.mlps5 = pt_utils.SharedMLP([self.in_dim, 512, 2], bn=True)

        self.lang_proj1 = nn.Linear(self.text_enc_dim, self.in_dim)
        self.lang_proj2 = nn.Linear(self.text_enc_dim, self.in_dim)
        self.lang_proj3 = nn.Linear(self.text_enc_dim, self.in_dim)
        self.lang_proj4 = nn.Linear(self.text_enc_dim, self.in_dim)

        self.bn1 = nn.BatchNorm2d(self.in_dim)
        self.bn2 = nn.BatchNorm2d(self.in_dim)
        self.bn3 = nn.BatchNorm2d(self.in_dim)
        self.bn4 = nn.BatchNorm2d(self.in_dim)
        self.bn5 = nn.BatchNorm2d(self.in_dim)
        self.bn6 = nn.BatchNorm2d(self.in_dim)
        self.bn7 = nn.BatchNorm2d(self.in_dim)
        self.bn8 = nn.BatchNorm2d(self.in_dim)

        # self.bn = nn.BatchNorm1d(2*self.in_dim)


    def forward(self, x, l, end_points):
        # l = l.float()

        l1= self.lang_proj1(l)
        l1 = l1.unsqueeze(-1).expand(l1.shape[0], l1.shape[1], x.shape[2])
        x_norm = self.bn1(x.unsqueeze(-1)).squeeze(-1)
        x_mul = x * l1
        x = self.bn2(x_mul.unsqueeze(-1)).squeeze(-1)
        x = x + x_norm
        x= self.mlps1(x.unsqueeze(-1)).squeeze(-1)

        l2 = self.lang_proj2(l)
        l2 = l2.unsqueeze(-1).expand(l2.shape[0], l2.shape[1], x.shape[2])
        x_norm = self.bn3(x.unsqueeze(-1)).squeeze(-1)
        x_mul = x * l2
        x = self.bn4(x_mul.unsqueeze(-1)).squeeze(-1)
        x = x + x_norm
        x = self.mlps2(x.unsqueeze(-1)).squeeze(-1)

        l3 = self.lang_proj3(l)
        l3 = l3.unsqueeze(-1).expand(l3.shape[0], l3.shape[1], x.shape[2])
        x_norm = self.bn5(x.unsqueeze(-1)).squeeze(-1)
        x_mul = x * l3
        x = self.bn6(x_mul.unsqueeze(-1)).squeeze(-1)
        x = x + x_norm
        x = self.mlps3(x.unsqueeze(-1)).squeeze(-1)

        l4 = self.lang_proj4(l)
        l4 = l4.unsqueeze(-1).expand(l4.shape[0], l4.shape[1], x.shape[2])
        x_norm = self.bn7(x.unsqueeze(-1)).squeeze(-1)
        x_mul = x * l4
        x = self.bn8(x_mul.unsqueeze(-1)).squeeze(-1)
        x = x + x_norm
        x = self.mlps4(x.unsqueeze(-1)).squeeze(-1)

        x = self.mlps5(x.unsqueeze(-1)).squeeze(-1)
        end_points['lang_select_objectness'] = x

        # self.mlps = pt_utils.SharedMLP(mlps, bn=True)
        # graspable_scor(e = self.conv_graspable(seed_features)  # (B, 3, num_seed)(4，3, 15000)
        # end_points['objectness_score'] = graspable_score[:, :2]
        # end_points['graspness_score'] = graspable_score[:, 2]
        return end_points

class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        # self.is_training = False
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        #选取最佳的方向
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous() # (B, num_seed, num_view)
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            #类似于repeat  重复
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            top_view_inds = []
            for i in range(B):
                #在每一行取出不为0的元素，按照权重理解
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)
            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            #从接近方向得到的旋转矩阵作用是
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP_2d(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # (BATCH_SIZE,B*3 + feat_dim*M*K,M,K)
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num angle * num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points
