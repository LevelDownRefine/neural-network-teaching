# Hebbian 与 Delta 本质

两者共用前向计算、链式法则和普通 SGD 更新（k、v 视为常量）：

$$
z=W_{\mathrm{old}}k,\qquad
 g=\nabla_z L,\qquad
\nabla_W L=gk^\top,\qquad
\boxed{W_{\mathrm{new}}=W_{\mathrm{old}}-\eta gk^\top.}
$$

区别只在 loss 及其对输出 z 的梯度：

| 规则 | loss | $g=\nabla_z L$ | 代入共同更新公式 | 生物学解释 |
| --- | --- | --- | --- | --- |
| Hebbian | $L_H=-v^\top z$ | $-v$ | $W_{\mathrm{new}}=W_{\mathrm{old}}+\eta vk^\top$ | 突触前、后共同活动增强关联；这里把 k、v 类比为两端活动。 |
| Delta | $L_D=\frac12\sum_j(z_j-v_j)^2$ | $z-v$ | $W_{\mathrm{new}}=W_{\mathrm{old}}+\eta(v-z)k^\top$ | DeltaNet 的记忆编辑解释：减去旧关联 $\eta zk^\top$，再写入新关联 $\eta vk^\top$。 |

逐对更新，loss 对输出维度求和，SGD 不使用动量或权重衰减。

Delta 一栏采用 [DeltaNet（Schlag 等，2021）](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf) 的计算解释，不代表已证实的生物抑制机制。

## 点积 Attention 与 Hebbian 记忆的联系

忽略 softmax 和缩放，取 Hebbian 学习率 $\eta=1$，记忆矩阵从零开始：

$$
\underbrace{z=\sum_i v_i(k_i^\top q)}_{\text{点积 Attention}}
\quad\Longleftrightarrow\quad
z=\left(\sum_i v_i k_i^\top\right)q
\quad\Longleftrightarrow\quad
\underbrace{z=Mq,\quad M=\sum_i v_i k_i^\top}_{\text{Hebbian 记忆}}.
$$

两者结果相同：前者查询时匹配 key 再组合 value；后者写入时累加外积，查询时直接乘记忆矩阵。
