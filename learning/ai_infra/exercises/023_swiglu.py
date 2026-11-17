import torch
from torch import Tensor


# ============================================================
# 题 23：自定义 autograd —— SwiGLU（手写 backward）
#   前向: out = silu(gate) * up，silu(x) = x * sigmoid(x)
#   要求：
#     1) 实现 SwiGLUFunction(torch.autograd.Function)，forward 里 save_for_backward，
#        backward 手工推导（不许在 backward 里再跑一遍 autograd）
#     2) swiglu(gate, up) 调用它
#     3) 判分会用 torch.autograd.gradcheck（float64）+ 与自动求导版对比
#   推导提示：d(silu)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
# ============================================================


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:  # type: ignore[override]
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        raise NotImplementedError


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    raise NotImplementedError
