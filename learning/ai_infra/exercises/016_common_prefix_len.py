from collections.abc import Callable

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

# ============================================================
# 题 17：MoE 的 dispatch / combine
#   x: [T, D]
#   topk_ids: [T, K] int64，每个 token 选中的 expert
#   topk_weights: [T, K] float，对应权重（已归一化）
#   expert_fn: Callable[[int, Tensor], Tensor]，输入 (expert_id, tokens[[n, D]]) 返回 [n, D]
#   num_experts: int
#   语义：out[t] = sum_k topk_weights[t,k] * expert_fn(topk_ids[t,k], x[t])
# 要求：
#   1) 每个 expert 只允许调用 expert_fn 一次（用 argsort 把同 expert 的 token 聚到一起）
#   2) 合并阶段用 index_add_ 之类完成逆置换加权求和，不允许 python 循环遍历 token
#   3) 允许在 num_experts 上循环
#   4) 某 expert 没有 token 时不要调用它（避免空 batch kernel）
#   返回 [T, D]
# ============================================================

def moe_dispatch_combine(
    x: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    expert_fn: Callable[[int, Tensor], Tensor],
    num_experts: int,
) -> Tensor:
    # T, D = x.shape
    # K = topk_ids.shape[1]
    # out = torch.zeros((T, D), dtype=x.dtype, device=x.device)
    # for i in range(T):
    #     for j in range(K):
    #         expert_out = expert_fn(topk_ids[i, j].item(), x[i:i+1]) # [1, D]
    #         out[i] +=  topk_weights[i, j] * expert_out[0] # [D]
    # return out
    T, D = x.shape
    K = topk_ids.shape[1]
    out = torch.zeros((T, D), dtype=x.dtype, device=x.device)
    flat_ids = topk_ids.reshape(-1)
    flat_weight = topk_weights.reshape(-1)

    order = torch.argsort(flat_ids) # [T * K]
    sorted_ids = flat_ids[order]  # 对应哪一个专家 [T * K]
    sorted_weight = flat_weight[order] # 对应哪一个权重 [T * K]
    token_ids = order // K # 对应x的哪一行 [T * K]

    for i in range(num_experts):
        # 取专家
        mask = sorted_ids == i # [T * K]
        if not mask.any():
            continue
        # 取输入
        expert_token_ids = token_ids[mask] # [n]
        expert_x = x[expert_token_ids] # [n, D]
        # 取权重
        expert_weight = sorted_weight[mask] # [n]
        # 专家运算
        expert_out = expert_fn(i, expert_x) # [n, D]
        # 乘以专家在当前x的每一行的权重
        expert_out_weight = expert_weight.unsqueeze(1) * expert_out # [n, D]
        out.index_add_(0, expert_token_ids, expert_out_weight)
    return out


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
    seg_max = torch.full((num_segs, D), float("-inf")) # [num_segs, D]
    index = seg_ids.unsqueeze(1).expand(-1, D)
    seg_max.scatter_reduce_(0, index, values, reduce="amax", include_self=False)

    row_max = seg_max[seg_ids] # [T, D]
    value_minus_max =  values - row_max # [T, D]
    value_exp = torch.exp(value_minus_max)

    out = torch.zeros((num_segs, D))
    out.index_add_(0, seg_ids, value_exp)
    # 最后需要加回 seg_max
    return torch.log(out) + seg_max
