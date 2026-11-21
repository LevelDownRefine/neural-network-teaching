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

将所有 key、原始 value 分别按列组成 K、V，α 为全部记忆的权重向量：

$$
\boxed{z=V\alpha.}
$$

| 方法 | $\alpha$ |
| --- | --- |
| 相似度检索（Top-1） | $\alpha=\delta_{\arg\max(\widehat K^\top\widehat q)}$ |
| Softmax Attention | $\alpha=\mathrm{softmax}(K^\top q/\sqrt d)$ |
| Hebbian | $\alpha=\eta K^\top q$ |
| Delta | $\alpha=\eta(I+\eta U)^{-1}K^\top q,\quad U=\mathrm{triu}(K^\top K,1)$ |

$\widehat K$ 的各列及 $\widehat q$ 均单位归一化；$\delta_r$ 表示仅第 r 个分量为 1、其余为 0 的向量，并列最大值选取一项。U 取严格上三角部分，K、V 按写入顺序排列。Hebbian、Delta 均假设记忆矩阵从零开始、学习率固定；Delta 一行是逐次更新的等价展开，不要求实际求逆。
