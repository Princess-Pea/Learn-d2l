import torch
from torch import nn
from d2l import torch as d2l
# 导入本地预设工具箱
from my_utils import load_data_patch, train_epoch, evaluate_accuracy

# --- 1. 配置路径与加载数据 ---
batch_size = 256
data_path = "/home/syalis/d2l_Code/d2l-pytorch/pytorch/chapter_multilayer-perceptrons/data"
train_iter, test_iter = load_data_patch(batch_size, data_path)
print("数据加载成功！")

# --- 2. Dropout 层实现 ---
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    # 使用 torch.rand 生成 0-1 均匀分布，大于 dropout 的保留
    mask = (torch.rand(X.shape) > dropout).float()
    # 缩放剩余元素，保持期望值不变
    return mask * X / (1.0 - dropout)

# --- 3. 定义模型架构 ---
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # self.training 是 nn.Module 的内置属性
        # 当调用 net.train() 时为 True，调用 net.eval() 时为 False
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        
        return self.lin3(H2)

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# --- 4. 初始化权重 ---
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# --- 5. 训练 ---
num_epochs, lr = 10, 0.5
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

print("开始训练 (使用 Dropout)...")
for epoch in range(num_epochs):
    # train_epoch 内部会执行 net.train()
    train_metrics = train_epoch(net, train_iter, loss, trainer)
    # evaluate_accuracy 内部会执行 net.eval()
    test_acc = evaluate_accuracy(net, test_iter)
    
    print(f'epoch {epoch + 1}: loss {train_metrics[0]:.4f}, '
          f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')

print("训练完成！")