# 循环神经网络的梯度分析（BPTT）

## 1. 简化模型

定义时间步 $t$ 的：
- 隐状态 $h_t$
- 输入 $x_t$
- 输出 $o_t$
- 隐藏层权重 $w_h$，输出层权重 $w_o$

变换：
$$
h_t = f(x_t, h_{t-1}, w_h), \quad o_t = g(h_t, w_o)
$$

目标函数（$T$ 个时间步的平均损失）：
$$
L = \frac{1}{T} \sum_{t=1}^T l(y_t, o_t)
$$

## 2. 梯度计算的难点

需要计算 $\frac{\partial L}{\partial w_h}$。由链式法则：
$$
\frac{\partial L}{\partial w_h} = \frac{1}{T} \sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \cdot \frac{\partial g(h_t, w_o)}{\partial h_t} \cdot \frac{\partial h_t}{\partial w_h}
$$

其中 $\frac{\partial h_t}{\partial w_h}$ 是核心困难，因为它递归依赖于 $h_{t-1}$ 和 $w_h$。

## 3. 递归展开

由 $h_t = f(x_t, h_{t-1}, w_h)$ 得：
$$
\frac{\partial h_t}{\partial w_h} = \underbrace{\frac{\partial f(x_t, h_{t-1}, w_h)}{\partial w_h}}_{b_t} + \underbrace{\frac{\partial f(x_t, h_{t-1}, w_h)}{\partial h_{t-1}}}_{c_t} \cdot \frac{\partial h_{t-1}}{\partial w_h}
$$

这是一个线性递归：$a_t = b_t + c_t a_{t-1}$，其中 $a_t = \frac{\partial h_t}{\partial w_h}$。

## 4. 求解递归

对于 $t \ge 1$，解为：
$$
a_t = b_t + \sum_{i=1}^{t-1} \left( \prod_{j=i+1}^{t} c_j \right) b_i
$$

代入原符号：
$$
\frac{\partial h_t}{\partial w_h} = \frac{\partial f(x_t, h_{t-1}, w_h)}{\partial w_h} + \sum_{i=1}^{t-1} \left( \prod_{j=i+1}^{t} \frac{\partial f(x_j, h_{j-1}, w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_i, h_{i-1}, w_h)}{\partial w_h}
$$

## 5. 梯度消失与爆炸

- 连乘积 $\prod_{j=i+1}^{t} c_j$ 中的每个 $c_j$ 是一个雅可比矩阵（或标量）。
- 若 $\|c_j\| > 1$，乘积随长度指数增长 → **梯度爆炸**
- 若 $\|c_j\| < 1$，乘积随长度指数衰减 → **梯度消失**

这解释了传统 RNN 难以学习长距离依赖的根本原因。

## 6. 解决方案

LSTM、GRU 等通过门控机制调节梯度流动，使得连乘积中的因子可被控制接近 1，从而缓解梯度消失/爆炸。

> 总结：BPTT 将梯度递归展开为带连乘积的求和形式，揭示了长期依赖中的指数级不稳定问题，为改进循环架构提供了理论基础。