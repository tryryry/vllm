from typing import Tuple

from torch import Tensor


# ============================================================
# 题 29：Triton kernel —— fused add + RMSNorm（需要 GPU + triton）
#   triton_add_rmsnorm(x, residual, weight, eps) -> (out, new_residual)
#   语义：new_residual = x + residual; out = rms_norm(new_residual, weight, eps)
#         （这就是 SGLang 里 layernorm 的 residual 融合形态）
#   要求：
#     1) 自己写 @triton.jit kernel，一个 program 处理一行（D <= 8192，用 BLOCK 覆盖整行 + mask）
#     2) 行内归约用 tl.sum，注意用 fp32 累加（输入可能是 bf16）
#     3) 同时把 new_residual 写回（真实实现是 in-place 写 residual）
#   判分：与 torch 参考实现 bf16 下 atol=2e-2 对齐。
# ============================================================


def triton_add_rmsnorm(
    x: Tensor, residual: Tensor, weight: Tensor, eps: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    raise NotImplementedError
