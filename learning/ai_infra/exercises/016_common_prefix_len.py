import torch
from torch import Tensor


# ============================================================
# 题 16：前缀缓存匹配长度（RadixCache 里最内层的操作）
#   cached_ids: [M] int64，new_ids: [N] int64
#   返回 int：两者最长公共前缀长度
# 要求：不允许 python 循环 / .tolist()；用张量比较 + cumprod 或 argmin 之类做到 O(1) 个 kernel。
# 边界：任一为空返回 0；完全相同返回 min(M, N)。
# ============================================================


def common_prefix_len(cached_ids: Tensor, new_ids: Tensor) -> int:
    L = min(cached_ids.shape[0], new_ids.shape[0])
    if torch.equal(cached_ids[:L], new_ids[:L]):
        return L
    compare = cached_ids[:L] == new_ids[:L]
    cum_compare = torch.cumprod(compare, dim=0)
    arg_min = torch.argmin(cum_compare, dim=0)
    return arg_min.item()
