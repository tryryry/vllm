import torch
from torch import Tensor


# ============================================================
# 题 18：分段归约
#   values: [T, D]，seg_ids: [T] int64（取值 0..num_segs-1，无序，可能有空段）
#   segment_sum -> [num_segs, D]，空段为 0
#   segment_logsumexp -> [num_segs, D]，空段为 -inf
# 要求：
#   segment_sum 用 index_add_ / scatter_add_（禁止循环）
#   segment_logsumexp 要数值稳定：先按段求 max（提示 scatter_reduce_ 的 "amax"），
#   再减去 max 求 exp 和，最后 log 加回；不能直接 exp 大数
#   segment_sum:        a + b
#   segment_logsumexp:  log(exp(a - max) + exp(b - max))
# ============================================================


def segment_sum(values: Tensor, seg_ids: Tensor, num_segs: int) -> Tensor:
    T, D = values.shape
    out = torch.zeros((num_segs, D))
    out.index_add_(0, seg_ids, values)
    return out


def segment_logsumexp(values: Tensor, seg_ids: Tensor, num_segs: int) -> Tensor:
    T, D = values.shape
    seg_max = torch.full((num_segs, D), float("-inf"))  # [num_segs, D]
    index = seg_ids.unsqueeze(1).expand(-1, D)
    seg_max.scatter_reduce_(0, index, values, reduce="amax", include_self=False)

    # 关键在这一步，使用 seg_ids 得到原vaues的对应位置的最大值，用于后续减法
    row_max = seg_max[seg_ids]  # [T, D]

    value_minus_max = values - row_max  # [T, D]
    value_exp = torch.exp(value_minus_max)

    out = torch.zeros((num_segs, D))
    out.index_add_(0, seg_ids, value_exp)
    # 最后需要加回 seg_max
    return torch.log(out) + seg_max
