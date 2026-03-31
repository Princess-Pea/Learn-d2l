import torch

# 已知函数f(x) = 3x^2 + 4x + 5，求f在x=2处的导数
x = torch.tensor(2.0, requires_grad=True) # 只有张量具有requires_grad属性
def f(x):
    return 3 * x * x + 4 * x + 5
y = f(x)
y.backward()
x.grad
print(x.grad, x.grad == 3 * 2 * x + 4)

# 在循环中执行反向传播而不清零梯度，会累积梯度
x.grad.zero_() # 将x的梯度清零
x = torch.rand(4, dtype=torch.float32, requires_grad=True)
y = x.sum()
for _ in range(4):
    y.backward()
    print(x.grad)

# 对比不使用detach和使用detach的区别
u = torch.rand(1, dtype=torch.float32, requires_grad=True)
v = torch.rand(1, dtype=torch.float32, requires_grad=True)
f = u * v
f.backward()
print(u.grad, v.grad)

u.grad.zero_()
v.grad.zero_()
w = u.detach()
g = w * v
g.backward()
print(u.grad, v.grad)

# 用detach分离计算图后，u的梯度消失，因为h不再依赖于u;
# 而v的梯度仍然存在，因为h仍然依赖于v.
u.grad.zero_()
v.grad.zero_()
u = torch.rand(1, dtype=torch.float32, requires_grad=True)
v = torch.rand(1, dtype=torch.float32, requires_grad=True)
f = u * v
g = f.detach()
h = g * v
h.backward()
print(u.grad, v.grad)
