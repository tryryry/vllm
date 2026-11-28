# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl
from torch import Tensor

# ============================================================
# 题 29：Triton kernel —— fused add + RMSNorm（需要 GPU + triton）
#   triton_add_rmsnorm(x, residual, weight, eps) -> (out, new_residual)
#   语义：new_residual = x + residual; out = rms_norm(new_residual, weight, eps)
#         （这就是 SGLang 里 layernorm 的 residual 融合形态）
#   要求：
#     1) 自己写 @triton.jit kernel。一个 program 处理一行，D <= 8192。
#        用 BLOCK 覆盖整行，用 mask 屏蔽越界位置。
#     2) 行内归约用 tl.sum，注意用 fp32 累加（输入可能是 bf16）
#     3) 同时把 new_residual 写回（真实实现是 in-place 写 residual）
#   判分：与 torch 参考实现 bf16 下 atol=2e-2 对齐。
# ============================================================


@triton.jit
def _add_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    new_residual_ptr,
    out_ptr,
    weight_ptr,
    D,
    BLOCK: tl.constexpr,
    eps: float = 1e-6,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * D + cols
    mask = cols < D

    x = tl.load(x_ptr + offsets, mask, other=0)
    residual = tl.load(residual_ptr + offsets, mask, other=0)
    weight = tl.load(weight_ptr + cols, mask, other=0)
    tl.store(new_residual_ptr + offsets, x + residual, mask)
    new_residual = x + residual
    new_residual = new_residual.to(tl.float32)
    mean_sq = tl.sum(new_residual * new_residual, axis=0) / D
    scale = tl.rsqrt(mean_sq + eps)
    out = tl.mul(new_residual, scale)
    out = tl.mul(out, weight)
    tl.store(out_ptr + offsets, out, mask)


def triton_add_rmsnorm(
    x: Tensor, residual: Tensor, weight: Tensor, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    T, D = x.shape
    block = triton.next_power_of_2(D)
    new_residual = torch.empty_like(x)
    out = torch.empty_like(x)
    _add_rmsnorm_kernel[(T,)](x, residual, new_residual, out, weight, D, block, eps)
    return out, new_residual


# 实测结论
# BF16 CUDA 自测 3/3 通过，new_residual 与 PyTorch 逐元素一致。
# out 对照 PyTorch RMSNorm，要求 atol=2e-2、rtol=0。
# (2, 6) 行偏移和尾部 mask，最大绝对误差 0。
# (2, 1) 单列、零残差和自定义 eps，最大绝对误差 0。
# (2, 8192) 最大宽度、非零存储偏移的连续输入，最大绝对误差 0。


# ------------------------------------------------------------
# 题 29 的自测入口（运行时加 --test）
# ------------------------------------------------------------
def _rmsnorm_self_test() -> None:
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise SystemExit("第 29 题的自测需要 CUDA")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(29)

    # 两行不同、权重逐列变化，D=6 还会触发尾部 mask。
    x_masked = torch.tensor(
        [[1, -2, 3, -4, 5, -6], [6, -5, 4, -3, 2, -1]],
        dtype=dtype,
        device=device,
    )
    residual_masked = torch.tensor(
        [[-0.5, 1, 0.25, -0.5, 1, 0.5], [1, 0, -0.5, 0.25, -1, -0.25]],
        dtype=dtype,
        device=device,
    )
    weight_masked = torch.tensor(
        [0.5, 0.75, 1, 1.25, 1.5, 0.625], dtype=dtype, device=device
    )

    # 单列包含相消到零的行和受 eps 影响的小值行。
    x_single = torch.tensor([[1], [0.00390625]], dtype=dtype, device=device)
    residual_single = torch.tensor([[-1], [0.00390625]], dtype=dtype, device=device)
    weight_single = torch.tensor([0.75], dtype=dtype, device=device)

    # 最大宽度，x 使用连续但存储偏移非零的 view。
    x_wide = torch.randn(2 * 8192 + 1, dtype=dtype, device=device, generator=generator)
    x_wide = x_wide[1:].view(2, 8192)
    residual_wide = torch.randn(
        x_wide.shape, dtype=dtype, device=device, generator=generator
    )
    weight_wide = 0.5 + torch.rand(
        8192, dtype=dtype, device=device, generator=generator
    )

    cases = (
        ("row offsets and tail mask", x_masked, residual_masked, weight_masked, 1e-6),
        ("single column and eps", x_single, residual_single, weight_single, 1e-4),
        ("max width and offset", x_wide, residual_wide, weight_wide, 1e-6),
    )
    for label, x, residual, weight, eps in cases:
        expected_residual = x + residual
        expected_out = F.rms_norm(expected_residual, (x.shape[-1],), weight, eps)
        out, new_residual = triton_add_rmsnorm(x, residual, weight, eps)
        torch.testing.assert_close(new_residual, expected_residual, atol=0, rtol=0)
        torch.testing.assert_close(out, expected_out, atol=2e-2, rtol=0)
        max_error = (out.float() - expected_out.float()).abs().max().item()
        print(f"{label}: PASS {tuple(x.shape)}, max abs error={max_error:g}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        _rmsnorm_self_test()
    else:
        print("用法: 029_triton_add_rmsnorm.py --test")
