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

## 与常见查询方式的区别

相似度检索和标准 Attention 保留各条 K、V，查询时匹配 key，再选取或组合 value。我们的 Hebbian / Delta 将配对写入记忆矩阵 W，查询时直接计算 Wq，不再读取原始 K、V，也不显式计算各条记忆的权重 α。

以下 K、V 分别由 key、value 按列组成：

| 方法 | 存储内容 | 查询计算 |
| --- | --- | --- |
| 相似度检索（Top-1） | 各条 K、V | $\alpha=\mathrm{hardmax}(\widehat K^\top\widehat q),\quad z=V\alpha$ |
| Softmax Attention | 各条 K、V | $\alpha=\mathrm{softmax}(K^\top q/\sqrt d),\quad z=V\alpha$ |
| Hebbian / Delta | 共同写入的记忆矩阵 W | $z=Wq$ |

带帽表示单位归一化；hardmax 将最大分数所在位置置为 1，其余置为 0，并列时选取一项。检索中的 α 只是选择结果的数学表示，实现时可直接返回选中的 value。

**联系：** 从零开始、固定学习率写入时，Hebbian 可展开为：

$$
W=\eta VK^\top,\qquad z=Wq=V(\eta K^\top q).
$$

因此它等价于用点积系数组合 value，但实际实现将配对预先累加到 W，并不保存或显式计算 α。Delta 同样直接读出 Wq，只是用残差更新 W，不能套用 Hebbian 的上述展开式。

**区别：** 显式 K、V 的存储随条目数增长；固定维度的 W 大小不变，但不同配对共享权重，可能相互干扰。Hebbian 与 Delta 在这里的查询方式相同，区别仍在写入规则。
