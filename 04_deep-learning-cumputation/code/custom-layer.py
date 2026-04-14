import torch
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
print(net(X).mean())