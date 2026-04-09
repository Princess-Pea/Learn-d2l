import torch
from torch import nn
from torch.nn import functional as F

# 在 forward 方法中多次调用同一个 nn.Linear 实例 self.shared 实现参数共享
class shared_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,256)
        self.shared = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256,1)

    def forward(self,X):
        out = F.relu(self.fc1(X))
        out = F.relu(self.shared(out))
        out = F.relu(self.shared(out))
        out = F.relu(self.fc_out(out))
        return out

net = shared_mlp()

X = torch.rand(size = (256,10))
Y = X.sum(dim=1,keepdim=True)

lr = 0.01
epochs = 500
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(epochs) :
    loss = criterion(net(X),Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')

print(net(X)-Y)
