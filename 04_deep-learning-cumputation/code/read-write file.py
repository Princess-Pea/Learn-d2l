import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(10)

torch.save(x, "x.pt")

z = torch.load("x.pt")
# print(z)

mydict = {"name": "李四", "age": 20, "score":x}
torch.save(mydict, "mydict.pt")
mydict2 = torch.load("mydict.pt")
# print(mydict2)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), "mlp.params") # state_dict()返回一个字典，包含模型所有参数

clone = MLP()
Z = clone(X) # clone的参数是随机初始化的
print(Z==Y) # 所以输出不一样
clone.load_state_dict(torch.load("mlp.params")) # load_state_dict()将加载的参数复制到模型
Z = clone(X) 
print(Z==Y) # 现在输出一样了，因为clone的参数已经和net一样了