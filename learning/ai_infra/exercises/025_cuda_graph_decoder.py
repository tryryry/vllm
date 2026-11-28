# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import Tensor

# ============================================================
# 题 25：CUDA Graph 捕获一个 decode step（需要 GPU）
# 要求实现 CudaGraphDecoder：
#   __init__(self, weight: Tensor, max_bs: int)
#       weight: [D, D] (cuda)；预分配静态输入 [max_bs, D] 和静态输出缓冲
#   capture(self)
#       用 torch.cuda.graph 捕获 "out = silu(x @ weight) @ weight" 这个 step；
#       捕获前记得 warmup（在 side stream 上跑几次），否则容易失败
#   replay(self, x: Tensor) -> Tensor
#       x: [bs, D]，bs <= max_bs。把 x 拷进静态输入的前 bs 行，replay，
#       返回静态输出前 bs 行的 clone()
# 判分：replay 结果与 eager 计算一致；连续用不同输入 replay 结果要跟着变；
#      且 replay 期间新增显存分配为 0（因此必须写进预分配缓冲、返回前再 clone）。
# ============================================================


class CudaGraphDecoder:
    def __init__(self, weight: Tensor, max_bs: int):
        self.weight = weight
        self.max_bs = max_bs

        D = weight.shape[0]
        self.static_input = weight.new_zeros((max_bs, D))
        self.static_output = weight.new_empty((max_bs, D))

        self.g = torch.cuda.CUDAGraph()

    def capture(self) -> None:
        current_stream = torch.cuda.current_stream(self.weight.device)
        side_stream = torch.cuda.Stream(self.weight.device)

        side_stream.wait_stream(current_stream)

        with torch.cuda.stream(side_stream):
            for _ in range(3):
                out = self.static_input @ self.weight
                out = torch.nn.functional.silu(out)
                out = out @ self.weight

        current_stream.wait_stream(side_stream)

        with torch.cuda.graph(self.g):
            self.static_output.copy_(
                torch.nn.functional.silu(self.static_input @ self.weight) @ self.weight
            )

    def replay(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        self.static_input[:bs].copy_(x)

        self.g.replay()

        out = self.static_output[:bs].clone()

        return out
