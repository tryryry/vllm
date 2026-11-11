from typing import Tuple

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
    raise NotImplementedError
