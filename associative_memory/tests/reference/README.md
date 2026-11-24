# Reference provenance

Unmodified `fla/ops/gated_delta_rule/naive.py` from:
https://github.com/fla-org/flash-linear-attention/tree/f5a9a8751139b3625999e055a83cc0895f6e1674

MIT license included. Tests extract and execute only
`naive_recurrent_gated_delta_rule`, avoiding optional GPU imports.
Mapping: g=0, beta=eta, scale=1, FLA state=W.T.
FLA computes in float32; comparisons therefore use float32 tolerances.
This validates the shared delta recurrence, not the full language model.
