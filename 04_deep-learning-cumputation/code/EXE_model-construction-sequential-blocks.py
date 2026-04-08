import torch
from torch import nn
from torch.nn import functional as F

class sequential_block(nn.Module):
    def __init__(self, *args, **kwargs): 
        # *args表示函数接受任意数量的位置参数，并将它们存储在一个元组中。
        # **kwargs表示函数接受任意数量的关键字参数，并将它们存储在一个字典中。
        # 位置参数是指在函数调用时按照位置传递的参数。
        # 关键字参数是指在函数调用时使用参数名称进行传递的参数。
        super().__init__(*args, **kwargs)
        # 这里我们创建两个并行的网络net1和net2，它们具有相同的结构，但参数不同。
        self.net1 = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
        self.net2 = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 10))
        
    def forward(self, X):
        return self.net2(self.net1(X))
        # 在前向传播中，我们首先将输入X传递给net1，得到一个输出，然后将这个输出传递给net2，得到最终的输出.

net = sequential_block()
X = torch.rand(2, 20)
print(net(X))



