from typing import Tuple

import torch
from torch import Tensor


# ============================================================
# 题 15：连续批处理 prefill 的元信息
#   prefix_lens: [B] int64，命中前缀缓存的长度（这部分 KV 已存在，不再计算）
#   extend_lens: [B] int64，本次需要前向的 token 数
#   返回 (positions, extend_start_loc)
#     positions: [sum(extend_lens)] int64，
#                第 b 条序列贡献 prefix_lens[b] .. prefix_lens[b]+extend_lens[b]-1
#     extend_start_loc: [B] int64，每条序列在拼接后张量里的起始下标（即 extend_lens 的
#                       exclusive 前缀和）
# 要求：无 python 循环。提示 repeat_interleave + arange 减去 offset。
# ============================================================
# prefix_lens: [2 0] extend_lens: [3 2]


def build_position_ids(
    prefix_lens: Tensor, extend_lens: Tensor
) -> Tuple[Tensor, Tensor]:
    B = prefix_lens.shape[0]
    T = torch.sum(extend_lens)
    extend_start_loc = (
        torch.cumsum(extend_lens, dim=0) - extend_lens
    )  # [3 5] - [3 2] = [0 3]

    offset_id = torch.arange(B).repeat_interleave(
        extend_lens
    )  # [sum(extend_lens)] [0 0 0 1 1]
    offset = torch.arange(T) - extend_lens[offset_id]  # [0 1 2 0 1]
    pos = (
        prefix_lens.repeat_interleave(extend_lens) + offset
    )  # [2 2 2 0 0] + [0 1 2 0 1] = [2 3 4 0 1]

    return pos, extend_start_loc
