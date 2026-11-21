# 从 loss 到 Hebbian 与 Delta 更新

训练时 $z=Wk$。对 W 反向传播，普通 SGD 更新为：

$$
W_{\mathrm{new}}=W_{\mathrm{old}}-\eta\nabla_W L.
$$

## Hebbian

$$
L_H=-v^\top z=-v^\top Wk,
\qquad
\nabla_W L_H=-vk^\top.
$$

因此：

$$
\boxed{W_{\mathrm{new}}=W_{\mathrm{old}}+\eta vk^\top.}
$$

## Delta

$$
L_D=\frac12\|z-v\|^2,
\qquad
\nabla_W L_D=(z-v)k^\top.
$$

因此：

$$
\boxed{W_{\mathrm{new}}=W_{\mathrm{old}}+\eta(v-z)k^\top,\qquad z=W_{\mathrm{old}}k.}
$$

以上对 k、v 视为常量，逐对更新，loss 对输出维度求和，SGD 不使用动量或权重衰减。
