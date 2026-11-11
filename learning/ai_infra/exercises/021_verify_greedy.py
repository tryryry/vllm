from typing import Tuple

from torch import Tensor


# ============================================================
# 题 21：投机解码的 greedy 验证（对应 SGLang 的 verify 阶段）
#   draft_tokens: [B, K] int64，draft model 提出的 K 个 token
#   target_logits: [B, K+1, V]，target model 在"接受了前 i 个 draft token"的前提下
#                  对第 i 个位置的输出 logits
#   语义：
#     从 i=0 开始，若 draft_tokens[b, i] == argmax(target_logits[b, i]) 则接受，继续；
#     第一次不等就停止。accept_len[b] = 接受的 draft token 数（0..K）
#     最终输出 = 已接受的 draft token + 1 个来自 target 的修正/bonus token，
#     即 argmax(target_logits[b, accept_len[b]])
#   返回 (accept_len: [B] int64, out_tokens: [B, K+1] int64)
#     out_tokens 前 accept_len+1 个是有效 token，其余位置填 -1
#   要求：不允许在 B 上循环（K 上可以循环，也可以整体向量化；推荐先写向量化版本，
#        提示 cumprod 找第一个 0）
# ============================================================


def verify_greedy(draft_tokens: Tensor, target_logits: Tensor) -> Tuple[Tensor, Tensor]:
    raise NotImplementedError
