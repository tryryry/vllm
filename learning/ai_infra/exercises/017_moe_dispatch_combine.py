from collections.abc import Callable

import torch
from torch import Tensor


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
