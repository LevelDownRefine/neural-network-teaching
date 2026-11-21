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

## 从查询与内容的匹配看读出

统一写成：

$$
z=\sum_i W_i v_i.
$$

这里 $W_i$ 是第 i 条内容的标量读出权重，不是上文的记忆矩阵 W。

| 方法 | 权重 $W_i$ |
| --- | --- |
| 相似度检索（Top-1） | $W_i=\delta_{i,i^{\ast}},\quad i^{\ast}=\arg\max_j\frac{q^\top k_j}{\lVert q\rVert\lVert k_j\rVert}$ |
| Softmax Attention | $W_i=\frac{\exp(q^\top k_i/\sqrt d)}{\sum_j\exp(q^\top k_j/\sqrt d)}$ |
| Hebbian | $W_i=\eta q^\top k_i$ |

$\delta_{i,i^{\ast}}$ 在 $i=i^{\ast}$ 时为 1，否则为 0；相似度检索假设向量非零，并列时选取一项。Hebbian 一行假设记忆矩阵从零开始，以固定学习率累加所列配对。
