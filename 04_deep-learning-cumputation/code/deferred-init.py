import torch
from torch import nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(256)   # 不指定输入大小
        self.relu = nn.ReLU()
        self.out = nn.LazyLinear(10)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.out(x)
        return x
    
net = MyNet()
print(net) # 输出网络结构，此时参数尚未初始化
X = torch.rand(2, 20)  # 输入数据的特征维度为20
Y = net(X)  # 进行前向传播，触发参数初始化
print(Y)