import torch
from torch import nn
from d2l import torch as d2l
import os
import torchvision
from torchvision import transforms
from torch.utils import data
from my_utils import load_data_patch, train_epoch, evaluate_accuracy

batch_size = 256
data_path = "/home/syalis/d2l_Code/d2l-pytorch/pytorch/data"

class MyFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(MyFashionMNIST, self).__init__(root, train=train, transform=transform, 
                                            target_transform=target_transform, download=False)
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'FashionMNIST', 'raw')
    def _check_exists(self):
        return True

trans = transforms.ToTensor()
train_set = MyFashionMNIST(root=data_path, train=True, transform=trans)
test_set = MyFashionMNIST(root=data_path, train=False, transform=trans)

train_iter = data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = data.DataLoader(test_set, batch_size, shuffle=False)
print("数据加载成功！")

num_inputs = 784
num_outputs = 10
num_hiddens = 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01) 
# nn.Parameter()会将一个普通的张量转换成一个参数，并且会自动将requires_grad设置为True
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
# a 是一个和 X 形状相同的全零张量
# torch.max(X, a) 会逐元素比较 X 和 a 的值，并返回一个新的张量，其中每个元素都是 X 和 a 中较大的那个值
# 换而言之，这个函数会将输入张量 X 中的负值替换为零，而正值保持不变，从而实现了 ReLU 激活函数的效果

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)

loss = nn.CrossEntropyLoss(reduction='none') # reduction='none'表示不对损失进行任何缩放或求和，而是返回每个样本的损失值。这对于某些应用场景非常有用，例如在训练过程中需要对每个样本的损失进行单独处理时。而我们的隐藏层使用了非线性激活函数，它恰要求每个样本的损失值都必须被保留，以便我们可以对它们进行单独处理，例如在训练过程中进行样本权重调整或进行其他类型的损失分析。

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
    test_acc = evaluate_accuracy(net, test_iter)
    print(f'epoch {epoch + 1}: loss {train_loss:.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

