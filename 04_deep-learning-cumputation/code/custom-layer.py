import torch
import torch.nn.functional as F
from torch import nn

class layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    
X = torch.tensor([[0.0, 1.0, 2.0]])
net = layer()
# print(net(X))

X = torch.rand(2,4)
print(X.mean())
print(net(X).mean())

net = nn.Sequential(nn.Linear(4,128), layer())
# print(net(X).mean())

class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.zeros(units,))

    def forward(self, X):
        linear = torch.matmul(X,self.weight) + self.bias 
        # 在这里使用self.weight和self.weight.data有什么区别？
        # 在 PyTorch 中，self.weight 是一个 nn.Parameter，它会被自动注册为模型的可学习参数，并参与反向传播和优化。而 self.weight.data 是其底层的 Tensor 数据，不会被 autograd 跟踪。
        # 因此，使用 self.weight 可以确保在训练过程中正确地更新参数，而使用 self.weight.data 可能会导致参数更新不正确，因为它不会被 autograd 跟踪。
        return F.relu(linear)
    
net = MyLinear(4,8)
print(net.weight)