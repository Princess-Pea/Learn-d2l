import torch
from torch import nn
from d2l import torch as d2l
# 导入工具箱
from my_utils import load_data_patch, train_epoch, evaluate_accuracy

# 1. 定义模型架构（高级 API）
net = nn.Sequential(
    nn.Flatten(),        # 将 28x28 的图像展平为 784 维向量
    nn.Linear(784, 256), # 隐藏层：784 -> 256
    nn.ReLU(),           # 激活函数
    nn.Linear(256, 10)   # 输出层：256 -> 10
)

# 2. 初始化参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 3. 设置超参数、损失函数和优化器
batch_size, lr, num_epochs = 256, 0.1, 10
data_path = "/home/syalis/d2l_Code/d2l-pytorch/pytorch/data"

# 替换数据加载部分
train_iter, test_iter = load_data_patch(batch_size, data_path)

loss = nn.CrossEntropyLoss(reduction='none')
# 使用高级优化器，传入 net.parameters()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 4. 执行训练循环
print("开始高级 API 训练...")
for epoch in range(num_epochs):
    # 使用工具箱中的训练逻辑
    train_metrics = train_epoch(net, train_iter, loss, trainer)
    test_acc = evaluate_accuracy(net, test_iter)
    
    print(f'epoch {epoch + 1}: loss {train_metrics[0]:.4f}, '
          f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')

print("训练完成！")