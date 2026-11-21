# 从 loss 到 Hebbian 与 Delta 更新

两者共用前向计算、链式法则和普通 SGD 更新（k、v 视为常量）：

$$
z=W_{\mathrm{old}}k,\qquad
 g=\nabla_z L,\qquad
\nabla_W L=gk^\top,\qquad
\boxed{W_{\mathrm{new}}=W_{\mathrm{old}}-\eta gk^\top.}
$$

区别只在 loss 及其对输出 z 的梯度：

| 规则 | loss | $g=\nabla_z L$ | 代入共同更新公式 |
| --- | --- | --- | --- |
| Hebbian | $L_H=-v^\top z$ | $-v$ | $W_{\mathrm{new}}=W_{\mathrm{old}}+\eta vk^\top$ |
| Delta | $L_D=\frac12\sum_j(z_j-v_j)^2$ | $z-v$ | $W_{\mathrm{new}}=W_{\mathrm{old}}+\eta(v-z)k^\top$ |

逐对更新，loss 对输出维度求和，SGD 不使用动量或权重衰减。
