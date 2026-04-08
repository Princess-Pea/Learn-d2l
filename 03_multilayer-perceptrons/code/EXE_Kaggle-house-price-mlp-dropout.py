import hashlib
import os
import tarfile
import zipfile
import requests

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# print(train_data.shape)
# print(test_data.shape)

# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 将训练数据和测试数据连接在一起，删除掉ID列
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
# 缺失值是preliminaries中提到的在数据集中没有记录的值，通常表示为NaN（Not a Number）。
# 在处理数据时，缺失值可能会导致问题，因为许多算法无法处理NaN值。因此，在进行数据预处理时，我们需要对缺失值进行处理，例如填充、删除或使用其他方法来处理它们。在这个代码中，我们将缺失值设置为0，因为在标准化数据之后，所有均值消失了，所以我们可以将缺失值视为0。
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
# 如果不使用“dummy_na=True”，则缺失值将被忽略，并且不会为其创建指示符特征。这可能会导致模型无法正确处理缺失值，从而影响模型的性能。因此，在处理数据时，建议使用“dummy_na=True”来确保模型能够正确处理缺失值。
all_features = pd.get_dummies(all_features, dummy_na=True)
# all_features.shape

n_train = train_data.shape[0]

# 确保所有特征都是数值类型（如果还有 object 列，则再次进行 one-hot）
if all_features.select_dtypes(include=['object']).shape[1] > 0:
    all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype(np.float32)   # 强制转换为 float32

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 下面我们使用均方误差作为损失函数
loss = nn.MSELoss()
in_features = train_features.shape[1]

# 使用其他模型（这里采用多层感知机 + 暂退）
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5), # 在第一个全连接层的激活函数后添加一个Dropout层，丢弃率为0.5
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    return net

# 下面的函数用于在训练过程中评估模型的表现。由于房价的数量级可能很大，我们对预测值和真实值取对数来稳定它们。
def log_rmse(net, features, labels):
    net.eval()                     # 关闭 Dropout，避免在评估模型时随机丢弃神经元导致结果不稳定
    with torch.no_grad():
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    net.train()                    # 恢复训练模式（下一次训练时重新开启 Dropout）
    return rmse.item()

# 下面的函数实现了训练过程。我们使用Adam优化算法来更新模型的参数。
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels)) 
        # 训练集上的log rmse，每个epoch都记录一次
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels)) 
    return train_ls, test_ls

# 下面的函数实现了K折交叉验证。
# 我们将训练数据分成K份，每次使用其中的一份作为验证集，其他K-1份作为训练集。我们重复这个过程K次，最终得到平均的训练和验证误差。
def get_k_fold_data(k, i, X, y):
    assert k > 1 # 断言k必须大于1，否则无法进行K折交叉验证
    fold_size = X.shape[0] // k # 计算每一折的大小
    X_train, y_train = None, None # 初始化训练集和标签
    for j in range(k): # 遍历每一折
        idx = slice(j * fold_size, (j + 1) * fold_size) # 计算当前折的索引范围
        X_part, y_part = X[idx, :], y[idx] # 获取当前折的数据和标签
        if j == i: # 如果当前折是第i折，则将其作为验证集
            X_valid, y_valid = X_part, y_part 
        elif X_train is None: # 如果训练集还没有被初始化，则将当前折的数据和标签作为训练集
            X_train, y_train = X_part, y_part
        else: # 否则，将当前折的数据和标签连接到训练集中
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

# 下面的函数实现了K折交叉验证的训练过程。
# 我们使用前面定义的get_k_fold_data函数来获取每一折的训练和验证数据，然后使用train函数来训练模型并评估其在训练集和验证集上的表现。我们记录每一折的训练和验证误差，并最终返回平均的训练和验证误差。
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0 # 初始化训练误差和验证误差的总和
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train) # 获取第i折的训练和验证数据
        net = get_net() # 初始化模型
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, 
                                   weight_decay, batch_size) # 训练模型并评估其在训练集和验证集上的表现
        train_l_sum += train_ls[-1] # 累加第i折的训练误差
        valid_l_sum += valid_ls[-1] # 累加第i折的验证误差
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log') # 只绘制第一折的训练和验证误差曲线
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}') # 打印第i折的训练和验证误差
    return train_l_sum / k, valid_l_sum / k

# 修改超参数，因为多层感知机比线性回归更复杂，所以我们需要更多的迭代次数和更大的权重衰减来防止过拟合。
k, num_epochs, lr, weight_decay, batch_size = 5, 200, 0.01, 0.001, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

# 训练模型并在测试集上进行预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    net.eval() # 关闭 Dropout，避免在评估模型时随机丢弃神经元导致结果不稳定
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

# 训练模型并在测试集上进行预测
train_and_pred(train_features, test_features, train_labels, test_data,
            num_epochs, lr, weight_decay, batch_size)
# 运行上面的代码，会得到一个名为submission.csv的文件，其中包含了测试集的预测结果。
# 将这个文件上传到Kaggle的比赛页面，可查看模型在排行榜上的表现如何。