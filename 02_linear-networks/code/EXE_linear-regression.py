import math
import time
import numpy as np
import torch
from d2l import torch as d2l

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




