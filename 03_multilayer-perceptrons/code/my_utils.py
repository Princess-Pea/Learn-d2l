import torch
import os
import torchvision
from torchvision import transforms
from torch.utils import data

# --- 1. 数据加载补丁 ---
class MyFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None):
        super(MyFashionMNIST, self).__init__(root, train=train, transform=transform, download=False)
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'FashionMNIST', 'raw')
    def _check_exists(self):
        return True

def load_data_patch(batch_size, root_path):
    trans = transforms.ToTensor()
    train_set = MyFashionMNIST(root=root_path, train=True, transform=trans)
    test_set = MyFashionMNIST(root=root_path, train=False, transform=trans)
    return (data.DataLoader(train_set, batch_size, shuffle=True),
            data.DataLoader(test_set, batch_size, shuffle=False))

# --- 2. 评估与准确率 ---
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += accuracy(net(X), y)
            n += y.shape[0]
    return acc_sum / n

# --- 3. 训练逻辑 ---
def train_epoch(net, train_iter, loss, updater):
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        # 兼容标量和向量 loss
        (l.mean() if l.ndim > 0 else l).backward()
        updater.step()
        total_loss += l.sum().detach().item()
        total_acc += accuracy(y_hat, y)
        n += y.shape[0]
    return total_loss / n, total_acc / n