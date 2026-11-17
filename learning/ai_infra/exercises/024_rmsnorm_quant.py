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
    raise NotImplementedError


def get_compiled_rmsnorm_quant() -> Callable[..., Tuple[Tensor, Tensor]]:
    raise NotImplementedError
