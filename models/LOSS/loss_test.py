import torch
import torch.nn as nn
import numpy as np
# loss_func = nn.CrossEntropyLoss(reduction='none')
# pre = torch.tensor([[[1.0, 0],[0, 1],[0, 0]], [[1.0, 1.0],[1,1],[1, 1]]], dtype=torch.float)#2*3*2
# tgt = torch.tensor([[1, 0] ,[0, 0]])#2*2
# print(pre)
# print(tgt)
# a = loss_func(pre, tgt)
# print(a)
#
# c = np.array([[[1,2], [3,4]], [[1,2], [3,4]]])
# d= np.asarray([[0,1], [3,4]])
#
# # e= c[:,d]
# # print(e)
# f = np.array([0,1])
# g = c[:,f,d]
# c[:,f,d] = 1
# print(g)
# print(c)
h = torch.tensor([1,2])
# h = torch.tensor([1,2]).
j = h.squeeze(0)
k = h.unsqueeze(-1)
i = k.unsqueeze(0).expand(2,2,100)
print(j)
print(k)
print(i)
# j = i.expand(4,2,100)