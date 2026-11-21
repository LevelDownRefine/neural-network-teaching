"""Parameter-free scaled dot-product attention over caller-owned keys/values."""

from torch.nn import functional as F

from .model_base import ModelBase


class ModelAttention(ModelBase):
    def __init__(self, d=64, m=32):
        super().__init__(d, m)

    def forward(self, **kwargs):
        q, k, v = kwargs["q"], kwargs["k"], kwargs["v"]
        self._validate(q, self.d)
        self._validate(k, self.d)
        self._validate(v, self.m)
        if k.ndim != 2 or v.ndim != 2 or len(k) != len(v) or len(k) == 0:
            raise ValueError(
                "keys and values must be nonempty matrices of equal length"
            )
        z = F.scaled_dot_product_attention(
            q.reshape(1, -1, self.d), k.unsqueeze(0), v.unsqueeze(0), dropout_p=0.0
        )[0]
        return z[0] if q.ndim == 1 else z
