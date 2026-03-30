import torch
X = torch.randint(0, 10, (1, 3, 1))
Y = torch.randint(0, 10, (2, 1, 5))
# 每个维度的长度要么相等，要么其中一个为1
print ((X+Y))