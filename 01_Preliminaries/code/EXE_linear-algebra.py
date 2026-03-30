import torch
# 证明矩阵的转置的转置还是原矩阵
for _ in range(3):
    m=torch.randint(1,10,()).item()
    n=torch.randint(1,10,()).item()
    A = torch.randint(0,11,(m, n), dtype=torch.float32)
    print(A, ' \n', A.T.T == A)

# 证明矩阵的转置的和等于矩阵的和的转置
m = torch.randint(1,10,()).item()
n = torch.randint(1,10,()).item()
A = torch.randint(1,11,(m, n), dtype=torch.float32)
B = torch.randint(1,11,(m, n), dtype=torch.float32)
print(A.T+B.T==(A+B).T)

# 证明矩阵与其转置的和是对称矩阵
A = torch.randint(1,11,(m, m), dtype=torch.float32)
print(A+A.T == (A+A.T).T)

# 张量的维度即轴的数量
# 调用len()函数返回张量的维度总是0轴的长度，即张量的第一个维度的大小
for _ in range(5):
    m = torch.randint(1,10,()).item()
    n = torch.randint(1,10,()).item()
    X = torch.randint(0, 10, (m, n, m+n))
    print(len(X)==m)

A = torch.arange(1,21).reshape(5, 4)
print(A/A.sum(axis=0)) 
# 不需要保持维度，因为A.sum(axis=0)的形状是(4,)，与A的形状(5, 4)广播兼容
print(A/A.sum(axis=1, keepdims=True))
# 需要保持维度，因为A.sum(axis=1, keepdims=True)的形状是(5, 1)，与A的形状(5, 4)广播兼容. 
# 如果不保持维度，A.sum(axis=1)的形状是(5,)，与A的形状(5, 4)不兼容，无法进行广播运算.

A = torch.arange(1,25).reshape(2,3,4)
print(A.sum(axis=0).shape, A.sum(axis=1).shape, A.sum(axis=2).shape)
# 这里A是一个三维张量，包含2个3行4列的矩阵。
# A.sum(axis=0)计算的是沿着轴0（行）进行求和，结果是一个形状为(3, 4)的二维张量；
# A.sum(axis=1)计算的是沿着轴1（列）进行求和，结果是一个形状为(2, 4)的二维张量；
# A.sum(axis=2)计算的是沿着轴2（深）进行求和，结果是一个形状为(2, 3)的二维张量。

print(A.float().norm())
# 这里A是一个三维张量，包含2个3行4列的矩阵。A.float().norm()计算的是A中所有元素的平方和的平方根，即A的Frobenius范数。由于A包含24个元素，所以结果是一个标量值，表示A中所有元素的整体大小。