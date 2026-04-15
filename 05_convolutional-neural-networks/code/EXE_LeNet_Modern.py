import torch
from torch import nn
from d2l import torch as d2l
from my_utils import load_data_patch, train_epoch_gpu, evaluate_accuracy_gpu

# 现代化版本的 LeNet
LeNet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), 
    nn.ReLU(),  # 1. Sigmoid -> ReLU
    nn.MaxPool2d(kernel_size=2, stride=2), # 2. AvgPool -> MaxPool
    
    nn.Conv2d(6, 16, kernel_size=5), 
    nn.ReLU(),  # 1. Sigmoid -> ReLU
    nn.MaxPool2d(kernel_size=2, stride=2), # 2. AvgPool -> MaxPool
    
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), 
    nn.ReLU(),  # 1. Sigmoid -> ReLU
    nn.Linear(120, 84), 
    nn.ReLU(),  # 1. Sigmoid -> ReLU
    nn.Linear(84, 10)
)

# 2. 数据加载
batch_size = 256
data_path = "/home/syalis/d2l_Code/d2l-pytorch/pytorch/chapter_multilayer-perceptrons/data"
train_iter, test_iter = load_data_patch(batch_size, data_path)

# 3. 训练配置
# 选择设备：如果有 GPU 就用 GPU，否则用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用设备: {device}")

# 将模型搬运到选定的设备上
LeNet.to(device)

# 初始化权重 (打破对称性)
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight) # 使用适合 Sigmoid 的 Xavier 初始化

LeNet.apply(init_weights)

# ⚠️ 注意：改用 ReLU 后，学习率要调小！
lr, num_epochs = 0.1, 10
trainer = torch.optim.SGD(LeNet.parameters(), lr=lr)
loss = nn.CrossEntropyLoss() # 默认 reduction='mean'

# 4. 训练循环
print("开始在 Fashion-MNIST 上训练 LeNet...")
for epoch in range(num_epochs):
    train_metrics = train_epoch_gpu(LeNet, train_iter, loss, trainer, device)
    test_acc = evaluate_accuracy_gpu(LeNet, test_iter, device)
    print(f'epoch {epoch + 1}: loss {train_metrics[0]:.4f}, '
          f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')

print("训练完成！")