import torch
from torch import nn
from torch.nn import functional as F

class sequential_block(nn.Module):
    def __init__(self, *args, **kwargs): 

        super().__init__(*args, **kwargs)
        # 这里我们创建两个并行的网络net1和net2，它们具有相同的结构，但参数不同。
        self.net1 = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
        self.net2 = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
        
    def forward(self, X):
        return self.net1(X) + self.net2(X)

net = sequential_block()
X = torch.rand(2, 20)
print(net(X)) 



