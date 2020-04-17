import  torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
# init_tensor = torch.rand(4, 8)*2 - 1
# print(init_tensor)
# def pairwise_distances(x, y):
#     '''
#     Input: x is a Nxd matrix
#            y is a Mxd matirx
#     Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
#     i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
#
#     Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
#     Computationally more expensive? Maybe, Not sure.
#     adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
#     '''
#     x_norm = (x ** 2).sum(1).view(-1, 1)   #sum(1)将一个矩阵的每一行向量相加
#     y_norm = (y ** 2).sum(1).view(1, -1)
#     y_t = torch.transpose(y, 0, 1)  #交换一个tensor的两个维度
#
#     # a^2 + b^2 - 2ab
#     dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)    #torch.mm 矩阵a和b矩阵相乘
#     # dist[dist != dist] = 0 # replace nan values with 0
#     return dist
#
# a= torch.rand(1, 8)*2 - 1
# pairwise_distances(a, init_tensor)
embeddings=np.arange(8).reshape(4,2)
print(embeddings)

embeddings = Variable(torch.from_numpy(embeddings).float(), requires_grad=False)
embeddings=embeddings+1
print(embeddings.requires_grad)
init_tensor = torch.rand(4,2)*2 - 1
print(init_tensor)
init_tensor=init_tensor+1
print(init_tensor.requires_grad)