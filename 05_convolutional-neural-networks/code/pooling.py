import torch
from torch import nn
from d2l import torch as d2l

# 以卷积的特例实现一个二维平均池化层

K = torch.ones((1, 2))

def conv_avgpool2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum() /(h * w)
    return Y

# 以卷积的特例实现一个二维最大池化层
def conv_maxpool2d(X, K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).max()
    return Y