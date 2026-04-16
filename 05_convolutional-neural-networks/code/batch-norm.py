import torch
from torch import nn
from d2l import torch as d2l

torch.manual_seed(0)

import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # gamma和beta是可学习的参数，moving_mean和moving_var分别是移动平均的均值和方差
    # eps是为了数值稳定而添加到方差中的一个小常数，momentum是更新移动平均的动量
    
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式，因为在评估模式下，我们不需要计算梯度，自动求导机制关闭。
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4) 
        # assert 语句用于检查条件是否为真，如果条件不满足，则会抛出一个AssertionError异常。在这里，我们检查输入X的形状是否是二维或四维的，因为批量归一化通常应用于全连接层（二维）或卷积层（四维）。如果输入的形状不符合要求，程序将会抛出异常并停止执行。
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

'''
def batch_norm1(X, gamma, beta, moving_mean, moving_var, eps, momentum): 
    
    if not torch.is_grad_enabled():
        # 评估模式使用传入的移动平均所得的均值和方差。用is_grad_enabled函数来判断当前模式是训练模式还是评估模式，因为在评估模式下，我们不需要计算梯度，自动求导机制关闭。
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 训练模式使用当前批量的均值和方差
        assert len(X.shape) in (2, 4) 
        

        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3))
            var = ((X - mean.reshape((1, -1, 1, 1))) ** 2).mean(dim=(0, 2, 3))
        X_hat = (X - mean.reshape((1, -1, 1, 1))) / torch.sqrt(var.reshape((1, -1, 1, 1)) + eps) 
        # reshape((1, -1, 1, 1))的作用是将均值和方差的形状调整为与输入X的形状兼容，以便进行广播操作。
        # -1表示自动推断维度的大小，通常用于保持其他维度不变的情况下调整特定维度的大小。
        # 在这里，我们将均值和方差调整为形状为(1, num_features, 1, 1)，其中num_features是输入X的通道数，这样在后续的归一化计算中就可以正确地进行广播操作。

        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma.reshape((1, -1, 1, 1)) * X_hat + beta.reshape((1, -1, 1, 1))
    return Y, moving_mean, moving_var
'''

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
gamma = torch.tensor([1.0, 1.0])
beta = torch.tensor([0.0, 0.0])
moving_mean = torch.tensor([0.0, 0.0])
moving_var = torch.tensor([1.0, 1.0])
eps = 1e-5
momentum = 0.9
Y, moving_mean, moving_var = batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum)
print(Y)

# 为什么输出是tensor([[[[-1.0000,  0.0000],[ 1.0000,  2.0000]], [[-2.0000, -1.0000],[ 0.0000,  1.0000]]]])

