import torch
from d2l import torch as d2l
from IPython import display 
# 导入IPython.display模块以使用display函数，该函数可以显示图像、文本等内容。

# 加载Fashion-MNIST数据集，并将其分成训练集和测试集.
batch_size = 256 
# 使用指定的批量大小。


import os
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

data_path = "/home/syalis/d2l_Code/d2l-pytorch/pytorch/data"

# 定义补丁类
class MyFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        # 强制设置 download=False
        super(MyFashionMNIST, self).__init__(root, train=train, transform=transform, 
                                            target_transform=target_transform, download=False)

    @property
    def raw_folder(self) -> str:
        # 强制指向已有的 FashionMNIST 文件夹，而不是类名对应的文件夹
        return os.path.join(self.root, 'FashionMNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'FashionMNIST', 'processed')

    def _check_exists(self):
        # 告诉它文件肯定在
        return True

# 实例化
trans = transforms.ToTensor()
batch_size = 256

train_set = MyFashionMNIST(root=data_path, train=True, transform=trans)
test_set = MyFashionMNIST(root=data_path, train=False, transform=trans)

train_iter = data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = data.DataLoader(test_set, batch_size, shuffle=False)

print("数据加载成功！")

# 这里的train_iter和test_iter是数据迭代器，可以用于在训练和测试过程中按批次获取数据。
# 通过调用d2l.load_data_fashion_mnist函数，我们可以轻松地加载Fashion-MNIST数据集，并准备好用于训练和评估模型的数据迭代

# 定义输入和输出的维度。
num_inputs = 28 * 28
num_outputs = 10
# 对于Fashion-MNIST数据集，每个图像是28x28像素的灰度图，因此输入维度为28*28=784。
# 输出维度为10，因为有10个类别（0-9）对应不同的服装类型。

# 定义模型参数，并使用正态分布初始化权重，偏置初始化为零。
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
# W是权重矩阵，大小为(784, 10)，每个元素从均值为0、标准差为0.01的正态分布中随机生成。
# b是偏置向量，大小为(10,)，初始化为零。

# 定义softmax函数，将模型的输出转换为概率分布。
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True) # 按第1维求和，保持维度不变。
    # 如何保持维度不变？把求和后的结果保留为一个列向量（即每行一个元素），这样就可以与原始的X_exp进行下面的广播运算：
    return X_exp / partition 

# 定义模型函数，计算输入数据的线性变换并应用softmax函数。
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
# reshape(-1, W.shape[0])将输入X调整为二维矩阵，其中-1表示自动推断行数，W.shape[0]表示列数（即输入特征的数量）。这样可以确保输入数据与权重矩阵W的维度匹配，从而进行矩阵乘法运算。
# matual()函数执行矩阵乘法，计算输入数据与权重矩阵的线性变换，然后加上偏置b，最后通过softmax函数将结果转换为概率分布。

# 定义交叉熵损失函数，计算模型预测与真实标签之间的差异。
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
# y_hat是模型的预测概率分布，y是实际的类别标签。cross_entropy函数通过取出正确类别的预测概率并计算其对数，然后取负值来计算交叉熵损失。

# 定义优化器，使用随机梯度下降（SGD）算法更新模型参数。
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
# updater函数使用d2l.sgd函数来更新模型参数W和b，传入学习率lr和当前批次的样本数量batch_size。

# 定义计算模型在数据集上的准确率的函数。
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) # 取出每行最大值的索引作为预测类别
    cmp = y_hat.type(y.dtype) == y # 将y_hat转换为与y相同的数据类型，然后进行元素级比较，得到一个布尔张量，表示每个预测是否正确。
    # 最后，使用sum()函数统计预测正确的数量，并将结果转换为float类型返回。
    return float(cmp.type(y.dtype).sum())

# 训练模型的函数，包含训练循环、损失计算和参数更新。
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式，这样某些特定于训练的层（如dropout和batch normalization）会以正确的方式工作。
    # net.train()
    # 初始化总损失和正确预测的数量，以及样本总数。
    total_loss, total_acc, n = 0.0, 0.0, 0
    # 遍历训练数据集中的每个批次。
    for X, y in train_iter:
        # 计算模型的预测输出。
        y_hat = net(X)
        # 计算当前批次的损失。
        l = loss(y_hat, y)
        # 如果updater是一个torch.optim.Optimizer对象，则使用它来更新模型参数。
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad() # 清除之前的梯度
            l.mean().backward() # 反向传播计算梯度
            updater.step() # 更新参数
        else:
            l.sum().backward() # 反向传播计算梯度
            updater(X.shape[0]) # 更新参数，传入当前批次的样本数量
        # 累加当前批次的损失和正确预测的数量，并更新样本总数。
        # total_loss += float(l.sum()) 换成下面的 detach().item()，以避免潜在的内存泄漏问题。
        total_loss += l.sum().detach().item()
        total_acc += float(accuracy(y_hat, y))
        n += y.shape[0]
    # 返回平均损失和平均准确率。
    return total_loss / n, total_acc / n

epochs = 10
lr = 0.1
for i in range(epochs):
    train_loss, train_acc = train_epoch_ch3(net, train_iter, cross_entropy, updater)
    print(f'epoch {i + 1}, loss {train_loss:.4f}, train acc {train_acc:.3f}')

# 评估模型在测试集上的准确率。
def evaluate_accuracy(net, data_iter):  #@save
    """评估模型在指定数据集上的准确率"""
    # net.eval()  # 将模型设置为评估模式，这样某些特定于训练的层（如dropout和batch normalization）会以正确的方式工作。
    acc_sum, n = 0.0, 0
    with torch.no_grad(): # 在评估过程中不需要计算梯度，因此使用torch.no_grad()上下文管理器来禁用梯度计算，以节省内存和计算资源。
        for X, y in data_iter:
            acc_sum += accuracy(net(X), y) # 累加正确预测的数量
            n += y.shape[0] # 累加样本总数
    return acc_sum / n # 返回平均准确率

test_acc = evaluate_accuracy(net, test_iter)
print(f'test acc {test_acc:.3f}')

# 预测函数，使用训练好的模型对输入数据进行预测，并返回预测的类别标签。
def predict(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
# 该函数首先从测试数据迭代器中获取一个批次的输入数据和对应的标签。然后，它使用训练好的模型对输入数据进行预测，得到预测的类别标签。接下来，它将真实标签和预测标签组合成标题，并使用d2l.show_images函数显示前n个图像及其对应的标题。每个标题包含真实标签和预测标签，以便于比较模型的预测结果与实际情况。
predict(net, test_iter)