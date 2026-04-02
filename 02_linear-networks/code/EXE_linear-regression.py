import math
import time
import numpy as np
import torch
from d2l import torch as d2l

'''
# 我们用随机数生成器来创建一个随机向量，并将其作为我们线性回归模型的输入。
# 我们还创建了一个随机标量 b 作为模型的参数，并设置 requires_grad=True 来启用自动求导功能，以便我们可以计算损失函数相对于 b 的梯度。
# 目标是通过迭代优化 b 来最小化损失函数，从而使模型更好地拟合数据。
# 另外，由于随机数生成于[0, 1)区间，我们的模型实际上是在拟合平均值为 0.5 的数据，因此我们可以推断最终的 b 接近 0.5。

# 生成长为 1000 的随机向量
x = torch.rand(1000, dtype=torch.float32)
print(x)

b = torch.rand(1, dtype=torch.float32, requires_grad=True)
print(b)

l=0.1
for i in range(100):

    # 每次迭代都要重新计算 loss，因为 b 已经更新了
    loss = torch.sum((x - b) ** 2) / len(x)
    print(f'第 {i} 次迭代，loss: {loss.item()}')
    loss.backward()

    # 不要直接 b = ...，会破坏计算图
    with torch.no_grad():  # 临时关闭梯度计算
        b -= l * b.grad

    # 每次迭代后都要清零梯度，否则会累积
    b.grad.zero_()

print(b)
'''

# 线性回归的向量化实现

# 生成数据集
def syn_data(w,b,n): # 传入预设的真实权重、偏差，并指定样本数量  #@save
    
    # 生成特征，语法是 torch.normal(mean, std, size)，n 是样本数量，len(w) 是特征维度
    x = torch.normal(0,1,(n,len(w))) 
    # 生成一个 n 行 len(w) 列的矩阵，每个元素都是从均值为 0、标准差为 1 的正态分布中随机采样的数值

    # 生成标签，语法是 torch.mv(matrix, vector)，其中 matrix 是权重矩阵，vector 是特征向量
    y = torch.mv(x, w) + b
    # torch.mv(x, w)是矩阵 x 和向量 w 的乘积，得到一个长度为 n 的向量，然后加上偏差 b，得到标签 y

    # 生成噪声，语法是 torch.normal(mean, std, size)，这里我们添加一个标准差为 0.01 的高斯噪声
    y += torch.normal(0, 0.01, y.shape)

    # 最后，我们返回特征 x 和标签 y。注意将标签 y 的形状调整为列向量，以便后续的计算和模型训练。
    return x, y.reshape((-1,1))  

# 设定真实的权重和偏差
true_w = torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32)
true_b = torch.tensor([1.42], dtype=torch.float32)

# 生成数据
X,y=syn_data(true_w,true_b,1000)

# 查看第一行特征和标签
print(X[0],y[0])

# 定义数据迭代器
def data_iter(batch_size, features, labels):  #@save
    num_examples = len(features)
    indices = list(range(num_examples))
    # 生成一个包含所有样本索引的列表

    np.random.shuffle(indices)
    # 将索引随机打乱，以确保每次迭代的数据顺序不同

    # 从 0 开始，以 batch_size 为步长，遍历整个数据集
    for i in range(0, num_examples, batch_size): 
        batch_indices = torch.tensor(
            # 获取当前批次的索引，使用切片操作从打乱后的索引列表中获取当前批次的索引，min 函数确保最后一个批次不会超出数据集的范围
            indices[i: min(i + batch_size, num_examples)]) 

        yield features[batch_indices], labels[batch_indices] 
        # 使用 yield 生成当前批次的特征和标签，供模型训练使用
        # 其中 yield 关键字使函数成为一个生成器，每次调用时返回当前批次的数据，结束调用时暂停，并在下一次调用时继续执行函数的剩余部分。

# 初始化模型参数
w = torch.normal(0, 0.01, size=(10,1), requires_grad=True) 
# 生成一个 10 行 1 列的权重矩阵，每个元素都是从均值为 0、标准差为 0.01 的正态分布中随机采样的数值

b = torch.zeros(1, requires_grad=True)
# 生成一个长度为 1 的偏差向量，初始值为 0

# 定义模型
def linreg(X,w,b): #@save
    return torch.mm(X,w)+b
    # torch.mm(X,w)是矩阵 X 和权重 w 的乘积，得到一个长度为 n 的向量，然后加上偏差 b，得到模型的预测值

# 定义损失函数
def squared_loss(y_hat,y): #@save
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    # 计算预测值 y_hat 和真实标签 y 之间的均方误差损失，除以 2 是为了在后续计算梯度时更方便

# 定义优化算法
def sgd(params, lr, batch_size): #@save
    with torch.no_grad(): # 临时关闭梯度计算
        for param in params:
            param -= lr * param.grad / batch_size
            # 更新参数，使用学习率 lr 和当前批次的大小 batch_size 来调整梯度的步长
            # 由于损失函数是对整个批次的平均损失，所以我们需要除以 batch_size 来得到每个样本的平均梯度，从而使更新更稳定

            param.grad.zero_()
            # 每次更新后清零梯度，以避免累积梯度影响下一次更新

# 训练模型
lr = 0.03
num_epochs = 1000
batch_size = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,X,y):
        l = loss(net(X,w,b),y) 
        # 计算当前批次的损失，net(X,w,b) 是模型的预测值，y 是真实标签

        l.sum().backward() 
        # 计算损失函数相对于模型参数 w 和 b 的梯度，l.sum() 是因为我们需要对整个批次的损失求和来计算梯度

        sgd([w,b], lr, batch_size) 
        # 使用随机梯度下降算法更新模型参数 w 和 b，传入当前的学习率 lr 和批次大小 batch_size

    with torch.no_grad(): 
        train_l = loss(net(X,w,b),y)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        # 每个 epoch 结束后，计算整个训练集上的损失，并打印当前 epoch 的编号和平均损失值

print('真实的权重:', true_w)
print('估计的权重:', w.reshape(true_w.shape))
print('真实的偏差:', true_b)    
print('估计的偏差:', b)
