from typing import Tuple

import torch
from torch import Tensor


# ============================================================
# 题 22：logprobs 输出
# 禁止使用 F.log_softmax / torch.log_softmax（手写，且要数值稳定）
#   logits: [B, V]
#   token_ids: [B] int64，本步实际选出的 token
#   top_k: int
#   返回 (chosen_logprobs: [B], topk_logprobs: [B, top_k], topk_ids: [B, top_k])
#   topk 按 logprob 降序
# 判分：与 F.log_softmax + topk 对齐，误差 < 1e-5；logits 会包含 1e4 量级的值。
# ============================================================


def token_logprobs(
    logits: Tensor, token_ids: Tensor, top_k: int
) -> Tuple[Tensor, Tensor, Tensor]:
    log_maxs, _ = torch.max(logits, dim=-1)  # [B]
    logits_minus_max = logits - log_maxs.unsqueeze(-1)  # [B, V]
    logits_log = torch.log(torch.sum(torch.exp(logits_minus_max), dim=-1))  # [B]
    logits_logprob = logits_minus_max - logits_log.unsqueeze(-1)  # [B, V]

    chosen_logprobs = logits_logprob.gather(
        dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)

    logits_logpro_sorted, logits_logpro_sorted_ids = logits_logprob.sort(
        dim=-1, descending=True, stable=True
    )
    topk_logprobs = logits_logpro_sorted[:, :top_k]
    topk_ids = logits_logpro_sorted_ids[:, :top_k]

    return chosen_logprobs, topk_logprobs, topk_ids
