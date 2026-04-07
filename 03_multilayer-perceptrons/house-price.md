# 解决 PyTorch 中 `TypeError: can't convert np.ndarray of type numpy.object_` 的问题

## 问题背景

在 Kaggle 房价预测的数据预处理过程中，使用以下代码将特征转换为 PyTorch 张量时：

```python
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
```

出现了如下错误：

```TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, ...```

## 原因分析:
`all_features` 是一个 pandas DataFrame，其中包含了类别特征（`dtype='object'`），例如街道类型、区域划分等字符串列。

虽然代码中已经使用了 `pd.get_dummies(all_features, dummy_na=True)` 对类别特征进行 one‑hot 编码，但某些情况下仍可能残留 `object` 类型的列（例如数值列被误读为字符串，或者存在全空的类别列）。

`all_features.values` 返回的 `numpy` 数组包含 `object` 类型元素，而 `torch.tensor` 不支持该类型，因此抛出异常。

解决方案
在转换为张量之前，强制确保所有特征列都是数值类型，并将整个 DataFrame 转换为 `float32`。

修改后的代码：
```
# 原有 one‑hot 编码
all_features = pd.get_dummies(all_features, dummy_na=True)

# 新增：如果仍有 object 列，再次进行 one‑hot 编码
if all_features.select_dtypes(include=['object']).shape[1] > 0:
    all_features = pd.get_dummies(all_features, dummy_na=True)

# 强制将所有列转换为 float32
all_features = all_features.astype(np.float32)

# 正常分割训练集和测试集特征
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features  = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels   = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
```