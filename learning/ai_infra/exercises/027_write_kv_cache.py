# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import Tensor

# ============================================================
# 题 27：KV cache 原地写入（零新增分配，需要 GPU）
#   k_pool/v_pool: [num_slots, H, D]（预分配好的池子）
#   loc: [T] int64，本次要写入的 slot 下标（可能不连续、不递增）
#   k/v: [T, H, D]
#   要求：原地写入，函数返回 None；
#        判分会在调用前后比较 torch.cuda.memory_allocated()，差值必须为 0，
#        所以不能产生任何中间张量（注意：k[:, None] 之类的 view 没问题，
#        但 k.float()、torch.cat 这种会分配）
#   dtype 提示：k 可能是 bf16 而池子是 bf16，保持一致即可。
# ============================================================


def write_kv_cache_(
    k_pool: Tensor, v_pool: Tensor, loc: Tensor, k: Tensor, v: Tensor
) -> None:
    k_pool[loc] = k
    v_pool[loc] = v

    return
