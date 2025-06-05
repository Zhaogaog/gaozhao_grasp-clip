import torch.nn as nn
import torch


def get_loss(end_points, model, alpha = 0):
    if len(end_points['batch_good_list'])==0:
        print(end_points['pack_obj_id'])
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(len(end_points['batch_good_list']))
    # print(len(end_points['batch_good_list']))
    if torch.sum(end_points['batch_good_list']).item() == 0:
        print('batch不合格')
        return 0, end_points
    objectness_graspable_loss, end_points = compute_objectness_graspable_loss(end_points)
    # objectness_loss, end_points = compute_objectness_loss(end_points)

    # graspness_loss, end_points = compute_graspness_loss(end_points)
    # view_loss, end_points = compute_view_graspness_loss(end_points)
    # score_loss, end_points = compute_score_loss(end_points)
    # width_loss, end_points = compute_width_loss(end_points)
    # loss = objectness_loss + 10 * graspness_loss + 100 * view_loss + 15 * score_loss + 10 * width_loss

    l2_loss = L2Loss(model, alpha)
    end_points['loss/l2_loss'] = l2_loss

    loss = objectness_graspable_loss + l2_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points

def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            if 'bias' not in name:
                l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(param, 2)))
    return l2_loss

def compute_objectness_graspable_loss(end_points):

    # criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_graspable_score = end_points['lang_select_objectness']
    # index_1 = torch.nonzero(end_points['batch_good_list'])[:,0]
    # if index_1.shape[0]!= 4:
    #     print(index_1)
    # index = torch.tensor([0, 1, 3], device='cuda:0')
    # if index_1.shape[0]!= 4:
    #     print(index)
    #     print(end_points['objectness_graspable_label'].shape)
    objectness_graspable_label = end_points['objectness_graspable_label']
    # if index_1.shape[0]!= 4:
    #     print(end_points['objectness_graspable_label'].shape)
    # # print(end_points['objectness_graspable_label'].shape)
    # objectness_graspable_score = end_points['lang_select_objectness'][index,:]
    objectness_label_sum  = torch.sum(objectness_graspable_label)
    # print('样本中合格的点数为：', objectness_label_sum)
    # print(end_points['pack_obj_id'])
    weight = torch.tensor([0.75,0.25], device=objectness_graspable_score.device)
    focal_loss = Focal_Loss(weight = weight)

    #分类问题，二分类，0表示不是要抓取的物体，1表示要抓取的物体。
    objectness_graspable_label_one_hot = torch.zeros_like(objectness_graspable_score, dtype=objectness_graspable_label.dtype)
    # objectness_label_one_hot[:,:,objectness_label[0,:]]=1
    # objectness_label_one_hot[:, objectness_label[:,:], :] = 1
    index = objectness_graspable_label.unsqueeze(1).expand(objectness_graspable_score.shape[0], objectness_graspable_score.shape[1],objectness_graspable_score.shape[2]).long()
    src = torch.ones_like(objectness_graspable_label).unsqueeze(1).expand(objectness_graspable_score.shape[0], objectness_graspable_score.shape[1],objectness_graspable_score.shape[2])
    objectness_graspable_label_one_hot.scatter_(1, index, src)
    # src = torch.ones_like(objectness_label)
    # objectness_label_one_hot.scatter_(1, objectness_label, src)
    # objectness_label_one_hot[:]
    objectness_label_one_hot_sum = torch.sum(objectness_graspable_label_one_hot[:,1,:])


    # loss = criterion(objectness_graspable_score, objectness_graspable_label.long())
    # loss_1 = criterion(objectness_graspable_score, objectness_graspable_label_one_hot.float())
    loss_2 = focal_loss(torch.softmax(objectness_graspable_score, 1), objectness_graspable_label_one_hot.float(),end_points)
    end_points['loss/objectness_graspable_loss'] = loss_2
    # print('loss,  focal_loss', loss.item(),  loss_2.item())
    objectness_pred = torch.argmax(objectness_graspable_score, 1)
    end_points['objectness_graspable_acc'] = (objectness_pred == objectness_graspable_label.long()).float().mean()
    end_points['objectness_graspable_prec'] = (objectness_pred == objectness_graspable_label.long())[
        objectness_pred == 1].float().mean()
    end_points['objectness_graspable_recall'] = (objectness_pred == objectness_graspable_label.long())[
        objectness_graspable_label == 1].float().mean()
    # print('总数：', objectness_label_one_hot_sum.item())
    objectness_pred_sum = torch.sum(objectness_pred)
    # print('预测合格的点数为：', objectness_pred_sum)
    return loss_2, end_points
def compute_objectness_loss(end_points):

    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    objectness_label_sum  = torch.sum(objectness_label)
    # print('样本中合格的点数为：', objectness_label_sum)
    focal_loss = Focal_Loss(weight = torch.tensor([0.75,0.25], device=objectness_score.device))
    #分类问题，二分类，0表示不是要抓取的物体，1表示要抓取的物体。
    objectness_label_one_hot = torch.zeros_like(objectness_score, dtype=objectness_label.dtype)
    # objectness_label_one_hot[:,:,objectness_label[0,:]]=1
    # objectness_label_one_hot[:, objectness_label[:,:], :] = 1
    src = torch.ones_like(objectness_label).unsqueeze(1).expand(objectness_score.shape[0], objectness_score.shape[1], objectness_score.shape[2])
    index = objectness_label.unsqueeze(1).expand(objectness_score.shape[0], objectness_score.shape[1], objectness_score.shape[2])
    objectness_label_one_hot.scatter_(1, index, src)
    # src = torch.ones_like(objectness_label)
    # objectness_label_one_hot.scatter_(1, objectness_label, src)
    # objectness_label_one_hot[:]
    objectness_label_one_hot_sum = torch.sum(objectness_label_one_hot[:,1,:])


    loss = criterion(objectness_score, objectness_label)
    loss_1 = criterion(objectness_score, objectness_label_one_hot.float())
    loss_2 = focal_loss(torch.softmax(objectness_score, 1), objectness_label_one_hot.float(),end_points)
    end_points['loss/stage1_objectness_loss'] = loss_2
    print('loss,  loss2', loss.item(),  loss_2.item())
    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
        objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
        objectness_label == 1].float().mean()

    return loss_2, end_points


def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['graspness_label'].squeeze(-1)
    loss_mask = end_points['objectness_label'].bool()
    loss = criterion(graspness_score, graspness_label)#只指引是要抓物体的抓取分数
    loss = loss[loss_mask]
    loss = loss.mean()
    
    graspness_score_c = graspness_score.detach().clone()[loss_mask]
    graspness_label_c = graspness_label.detach().clone()[loss_mask]
    graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)#限制 将超出区间的转化为区间限值
    graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
    rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()#裁剪只取整数部分
    end_points['stage1_graspness_acc_rank_error'] = rank_error

    end_points['loss/stage1_graspness_loss'] = loss
    return loss, end_points


def compute_view_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)
    end_points['loss/stage2_view_loss'] = loss
    return loss, end_points


def compute_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)

    end_points['loss/stage3_score_loss'] = loss
    return loss, end_points


def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points


class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels, end_points):
        """
        preds:softmax输出结果
        labels:真实值
        """
        self.weight = self.weight.unsqueeze(0).unsqueeze(-1).expand(preds.shape[0],preds.shape[1],preds.shape[2])
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)
        graspable_point_num = end_points['graspable_count_stage1']
        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)
        target_sum = torch.sum(target[:,1,:])
        ce = -1 * torch.log(y_pred + eps) * target
        ce_negative = torch.sum(ce[:,0,:])
        ce_positive= torch.sum(ce[:, 1, :])
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss_negative = torch.sum(floss[:,0,:])
        floss_positive = torch.sum(floss[:, 1, :])
        floss_weight = torch.mul(floss, self.weight)
        floss_weight_negative = torch.sum(floss_weight[:, 0, :])
        floss_weight_positive= torch.sum(floss_weight[:, 1, :])
        floss_weight_sum = torch.sum(floss_weight)
        floss_weight = torch.sum(floss, dim=1)
        # floss_weight = torch.sum(floss)
        return torch.mean(floss_weight)
        # return floss_weight/target_sum