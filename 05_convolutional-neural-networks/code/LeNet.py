import torch
from torch import nn
from d2l import torch as d2l

# 用深度学习框架实现LeNet
LeNet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
# 我们将输入X设置为一个形状为（1，1，28，28）的四维张量
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# 我们查看一下每一层的输出形状。其中批量大小和通道数都为1。
# 对于每一层，我们打印出它的类名和输出的形状。
for layer in LeNet:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
# 可以看到，卷积层和池化层改变了输出的高度和宽度，而全连接层改变了输出的通道数。

batch_size = 256
