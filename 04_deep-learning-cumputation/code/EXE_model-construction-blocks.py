'''
import torch
from torch import nn
from torch.nn import functional as F

class block_unit(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear = nn.Linear(10,256)
        self.relu = nn.ReLU()

    def forward(self,X):
        return self.ReLU(self.linear(X))

def mutiply_blocks(block_type,nums,*args,**kwargs):
    layers = []
    for _ in range(nums):
        layers.append = block_type(*args,**kwargs)
    return nn.Sequential(*layers)

net = mutiply_blocks(block_unit, 3, 10, 20)

X = torch.rand(4,10)
Y = net(X)
'''

import torch
from torch import nn

class Block(nn.Module):
    """一个简单的块：线性层 + ReLU 激活（可根据需要修改）"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.relu(self.linear(X))

def multiply_blocks(block_type, num_instances, *args, **kwargs):
    """
    生成同一个块的多个实例，并串联成一个 Sequential 网络。

    参数:
        block_type: 块的类（例如 Block）
        num_instances: 实例数量
        *args, **kwargs: 传递给 block_type 构造函数的参数
    """
    layers = []
    for _ in range(num_instances):
        # 每次实例化一个新的块
        layers.append(block_type(*args, **kwargs))
    return nn.Sequential(*layers)

# 示例：创建 3 个 Block 实例，每个输入输出 20 维，然后串联
net = multiply_blocks(Block, 3, 20, 20)

# 测试
X = torch.randn(4, 20)
out = net(X)
print(out.shape)  # torch.Size([4, 20])