# 多头注意力 (Multi-Head Attention)

## 1. 动机

单头注意力只能从单一角度计算相似度，难以同时捕捉不同类型的关系（如短距离依赖、长距离依赖等）。多头注意力通过多组可学习的线性投影，将查询、键、值映射到不同的子空间，并行计算注意力，最后融合输出，从而增强模型表达能力。

## 2. 单个头的计算

给定查询 $\mathbf{q} \in \mathbb{R}^{d_q}$、键 $\mathbf{k} \in \mathbb{R}^{d_k}$、值 $\mathbf{v} \in \mathbb{R}^{d_v}$，第 $i$ 个头的输出为：

$$
\mathbf{h}_i = f\big( \mathbf{W}_i^{(q)} \mathbf{q},\; \mathbf{W}_i^{(k)} \mathbf{k},\; \mathbf{W}_i^{(v)} \mathbf{v} \big) \in \mathbb{R}^{p_v}
$$

- $\mathbf{W}_i^{(q)} \in \mathbb{R}^{p_q \times d_q}$、$\mathbf{W}_i^{(k)} \in \mathbb{R}^{p_k \times d_k}$、$\mathbf{W}_i^{(v)} \in \mathbb{R}^{p_v \times d_v}$ 是可学习的投影矩阵。
- $f$ 是注意力汇聚函数（如缩放点积注意力或加性注意力）。

实践中常取 $p_q = p_k = p_v = d / h$，其中 $d$ 为模型维度，$h$ 为头数。

## 3. 多头输出

将所有 $h$ 个头的输出沿特征维拼接，再经过一个输出投影矩阵 $\mathbf{W}_o \in \mathbb{R}^{p_o \times h p_v}$：

$$
\text{MultiHead}(\mathbf{q}, \mathbf{k}, \mathbf{v}) = \mathbf{W}_o \begin{bmatrix} \mathbf{h}_1 \\ \mathbf{h}_2 \\ \vdots \\ \mathbf{h}_h \end{bmatrix}
$$

通常 $p_o = d_q$ 或等于模型维度。

## 4. 批量矩阵形式（缩放点积注意力）

对于查询矩阵 $\mathbf{Q} \in \mathbb{R}^{n \times d_q}$、键矩阵 $\mathbf{K} \in \mathbb{R}^{m \times d_k}$、值矩阵 $\mathbf{V} \in \mathbb{R}^{m \times d_v}$：

1. 对每个头 $i$，计算线性投影：$\mathbf{Q}_i' = \mathbf{Q} \mathbf{W}_i^{(q)\top}$，$\mathbf{K}_i' = \mathbf{K} \mathbf{W}_i^{(k)\top}$，$\mathbf{V}_i' = \mathbf{V} \mathbf{W}_i^{(v)\top}$。
2. 计算缩放点积注意力：
   $$
   \text{head}_i = \mathrm{softmax}\left( \frac{\mathbf{Q}_i' \mathbf{K}_i'^\top}{\sqrt{p_k}} \right) \mathbf{V}_i'
   $$
3. 拼接所有 $\text{head}_i$ 并做线性变换。

## 5. 优点

- **多视角**：每个头关注不同的特征子空间或位置。
- **并行**：所有头可同时计算，效率高。
- **适度参数量**：仅增加投影矩阵和输出矩阵，无额外复杂结构。

## 6. 应用

多头注意力是 Transformer、BERT、GPT 等模型的核心组件，广泛用于自然语言处理、计算机视觉等任务。