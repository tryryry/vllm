import torch
from torch import Tensor


# ============================================================
# 题 20：惩罚项与 min_p
# 20a) apply_penalties(logits, output_ids, presence_penalty, frequency_penalty,
#                      repetition_penalty, pad_id=-1):
#   logits: [B, V]
#   output_ids: [B, L] int64，已生成的 token（等于 pad_id 的位置要忽略）
#   presence_penalty/frequency_penalty/repetition_penalty: [B]
#   语义（与 OpenAI / HF 一致）：
#     count[b, t] = t 在 output_ids[b] 中出现次数
#     先做 repetition_penalty（乘性，只对 count>0 的位置）：
#        logits > 0 时 logits / rp，否则 logits * rp
#     再做加性：logits -= presence * (count > 0) + frequency * count
#   返回 [B, V]
#   要求：用 scatter_add_ 统计次数，不允许在 B 或 L 上循环。
#
# 20b) min_p_filter(probs, min_p):
#   probs: [B, V]（每行和为 1），min_p: [B]
#   语义：阈值 = min_p[b] * max(probs[b])，小于阈值的置 0，然后重新归一化
#   返回 [B, V]
# ============================================================


def apply_penalties(
    logits: Tensor,
    output_ids: Tensor,
    presence_penalty: Tensor,
    frequency_penalty: Tensor,
    repetition_penalty: Tensor,
    pad_id: int = -1,
) -> Tensor:
    B, V = logits.shape
    valid = output_ids != pad_id
    output_ids_zeros = torch.where(valid, output_ids, 0)
    count = torch.zeros_like(logits)
    # count[b, output_ids_zeros[b, j]] += src[b, j]
    count.scatter_add_(
        dim=1, index=output_ids_zeros, src=valid.to(dtype=count.dtype)
    )  # [B, V]

    mask = count > 0
    logits_repetition_penalty = torch.where(
        mask,
        torch.where(
            logits > 0,
            logits / repetition_penalty.unsqueeze(-1),
            logits * repetition_penalty.unsqueeze(-1),
        ),
        logits,
    )
    logits_presence_frequency_penalty = (
        logits_repetition_penalty
        - presence_penalty.unsqueeze(-1) * (mask)
        - frequency_penalty.unsqueeze(-1) * count
    )
    return logits_presence_frequency_penalty


def min_p_filter(probs: Tensor, min_p: Tensor) -> Tensor:
    probs_max, _ = probs.max(dim=1)  # [B]
    min_p_threshold = min_p * probs_max  # [B]
    out = torch.where(probs < min_p_threshold.unsqueeze(-1), 0, probs)  # [B, V]
    return out / torch.sum(out, dim=1, keepdim=True)
