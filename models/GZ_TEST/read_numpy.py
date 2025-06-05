import numpy as np
# import tensor
import torch

# a = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/09.npy')
# b = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/11.npy')
# c = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/20.npy')
# d = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/29.npy')
# e = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/30.npy')
# f = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/41.npy')
# g = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/48.npy')
# h = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/52.npy')
# i = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/58.npy')
# j = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/scene_0100/kinect/0004/62.npy')
# # print(a)
# ap = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn/dump_epoch15_add_language_frozen/ap_kinect_test_seen.npy')
# # print(ap)
# res= np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb_resnet18/dump_epoch10_add_language_frozen_with_rgb/scene_0130/kinect/0000/25.npy')
# print(res)
# ap = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb/dump_epoch15_add_language_frozen_with_rgb/ap_kinect_test_seen.npy')
# cls_ap = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb/dump_epoch15_add_language_frozen_with_rgb/cls_ap_kinect_test_seen.npy')
# ap = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb/dump_epoch15_add_language_frozen_with_rgb/ap_kinect_test_seen.npy')
# cls_ap = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/logs/log_kn_with_rgb/dump_epoch15_add_language_frozen_with_rgb/cls_ap_kinect_test_seen.npy')
# print(ap,cls_ap)
# ap = np.load('/media/ama/data0/gz/graspnet/graspness_unofficial/object-grasp-annotation/grasp_label/agveuv_labels.npz')
# print(np.mean(ap))/media/ama/data0/gz/graspnet/graspness_unofficial/object-grasp-annotation/grasp_label/agveuv_labels.npz
a = torch.tensor([0,1,2])
b = torch.sum(a)
print(b.item()==3)