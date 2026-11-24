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
        out = (gate * torch.sigmoid(gate)) * up
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        gate, up = ctx.saved_tensors
        gate_grad = torch.sigmoid(gate) * (1 + gate * (1 - torch.sigmoid(gate))) * up * grad_out
        up_grad = (gate * torch.sigmoid(gate)) * grad_out
        return gate_grad, up_grad

def swiglu (gate: Tensor, up: Tensor) -> Tensor:
    return SwiGLUFunction.apply(gate, up)
