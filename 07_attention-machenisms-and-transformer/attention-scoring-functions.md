# 加性注意力（Additive Attention）

## 1. 定义与公式

加性注意力用于查询 $\mathbf{q} \in \mathbb{R}^q$ 和键 $\mathbf{k} \in \mathbb{R}^k$ 长度不同的情况。评分函数为：

$$
a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k}) \in \mathbb{R}
$$

可学习参数：
- $\mathbf{W}_q \in \mathbb{R}^{h \times q}$：将查询投影到隐藏维度 $h$
- $\mathbf{W}_k \in \mathbb{R}^{h \times k}$：将键投影到相同隐藏维度 $h$
- $\mathbf{w}_v \in \mathbb{R}^{h}$：将隐藏层输出映射为标量得分

超参数 $h$ 为隐藏层大小。

## 2. 计算流程

1. **线性变换**：$\mathbf{W}_q \mathbf{q}$ 和 $\mathbf{W}_k \mathbf{k}$ 将两者投影到同一 $h$ 维空间
2. **相加与激活**：$\tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k})$，引入非线性
3. **输出标量**：$\mathbf{w}_v^\top$ 点积得到注意力得分

整体结构等价于**单隐藏层 MLP**（无偏置项）。

## 3. 为什么叫“加性”？

核心操作为**线性变换后的向量相加**（$\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k}$），区别于点积注意力直接计算 $\mathbf{q}^\top \mathbf{k}$ 或 $\mathbf{q}^\top \mathbf{W} \mathbf{k}$。

## 4. 与点积注意力对比

| 特性 | 加性注意力 | 点积注意力 |
|------|-----------|-----------|
| 查询与键长度 | 允许不同（$q \ne k$） | 必须相同（$q = k$） |
| 计算复杂度 | 较高（矩阵乘 + MLP） | 较低（一次矩阵乘） |
| 表达能力 | 更强（非线性） | 线性相似度 |
| 典型应用 | 多模态、维度不匹配 | Transformer 自注意力 |

## 5. 与核回归的联系

相比 Nadaraya–Watson 核回归的固定评分函数（如高斯核），加性注意力引入**可学习参数** $(\mathbf{W}_q, \mathbf{W}_k, \mathbf{w}_v)$，是从数据中学习相似度度量的通用形式，属于**带参数注意力汇聚**的一种实现。

## 6. 小结

- 加性注意力通过单隐层 MLP 将查询和键映射为标量得分。
- 不要求查询与键长度相等，适用性更广。
- 现代深度学习中多被点积注意力取代，但在跨模态或特征维度不一致时仍有价值。


# 缩放点积注意力

## 评分函数

对于单个查询 $\mathbf{q}$ 和键 $\mathbf{k}$（长度均为 $d$）：

$$
a(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d}}
$$

## 为什么缩放？

- 点积的方差为 $d$，除以 $\sqrt{d}$ 后方差变为 $1$，稳定梯度。
- 避免 softmax 进入饱和区（输出过于 one‑hot）。

## 批量计算（矩阵形式）

查询 $\mathbf{Q} \in \mathbb{R}^{n \times d}$，键 $\mathbf{K} \in \mathbb{R}^{m \times d}$，值 $\mathbf{V} \in \mathbb{R}^{m \times v}$：

$$
\mathrm{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}} \right) \mathbf{V} \in \mathbb{R}^{n \times v}
$$

- $\mathbf{Q}\mathbf{K}^\top$ 计算所有查询‑键对的相似度。
- 每行 softmax 得到注意力权重。
- 乘以 $\mathbf{V}$ 得到加权和。

## 优点

- 矩阵乘法高度优化，计算效率高。
- 相比加性注意力更简单、更快。

## 常见变体

- **掩码缩放点积注意力**：在 softmax 前对非法位置加 $-\infty$，用于自回归生成。