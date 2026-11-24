import torch
from typing import Callable, Tuple

from torch import Tensor


# ============================================================
# 题 24：RMSNorm + 量化融合，并用 torch.compile 编译
# 24a) rmsnorm_quant(x, weight, eps) -> (q: int8 [T,D], scale: float32 [T])
#      语义 = 先 rms_norm 再做题 6 的 per-token int8 量化（这两步在真实推理里会被融合）
# 24b) get_compiled_rmsnorm_quant() 返回 torch.compile 后的可调用对象
#      要求 mode="max-autotune-no-cudagraphs" 或默认 mode 均可，但必须真的编译
#      （判分只检查：编译版与 eager 版输出一致，且对象不是原函数本身）
# 额外练习（不判分）：用 TORCH_LOGS="graph_breaks" 跑一遍，确认没有 graph break；
#      再用题 30 的 benchmark 对比两者耗时。
# ============================================================


def rmsnorm_quant(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    x_sq = x * x
    x_mean = x_sq.mean(dim=-1, keepdim=True) # [T, 1]
    x_eps = x_mean + eps # [T, 1]
    x_rsqrt = torch.rsqrt(x_eps) # [T, 1]
    x_rms = x * x_rsqrt * weight  # [T, D]
    
    x_max = x_rms.abs().amax(dim=-1, keepdim=True) # [T, 1]
    x_scale = x_max / 127
    return (x / x_scale).round().clamp(-127, 127).to(torch.int8), x_scale.squeeze(-1)
    


def get_compiled_rmsnorm_quant() -> Callable[..., Tuple[Tensor, Tensor]]:
     return torch.compile(rmsnorm_quant)
