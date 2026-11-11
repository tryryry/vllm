import torch
from typing import Optional, Tuple
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
    seg_max = torch.full((num_segs, D), float("-inf")) # [num_segs, D]
    index = seg_ids.unsqueeze(1).expand(-1, D)
    seg_max.scatter_reduce_(0, index, values, reduce="amax", include_self=False)

    # 关键在这一步，使用 seg_ids 得到原vaues的对应位置的最大值，用于后续减法
    row_max = seg_max[seg_ids] # [T, D]

    value_minus_max =  values - row_max # [T, D]
    value_exp = torch.exp(value_minus_max)

    out = torch.zeros((num_segs, D))
    out.index_add_(0, seg_ids, value_exp)
    # 最后需要加回 seg_max
    return torch.log(out) + seg_max

"""第 4 章：采样与 logits 处理（题 19-22）

判分：python check.py 4
"""

# ============================================================
# 题 19：temperature + top_k + top_p 过滤
#   logits: [B, V] float32
#   temperature: [B]（>0；等于 1 时不变）
#   top_k: [B] int64（<=0 或 >=V 表示不限制）
#   top_p: [B] float（1.0 表示不限制）
#   返回 probs: [B, V]，被过滤掉的位置为 0，每行和为 1
# 语义顺序：
#   1) logits / temperature -> softmax 得到 probs
#   2) top_k：只保留每行最大的 k 个（并列时按下标小的优先）
#   3) top_p：在剩余分布上降序累加，保留累积概率首次 >= top_p 的那个 token 及其之前的；
#      即使第一个 token 概率就 > top_p，它也必须保留（否则会得到空集）
#   4) 重新归一化
# 要求：批内向量化，不允许在 B 上循环。
# 提示：sort + cumsum + scatter 回原下标。
# ============================================================


def sample_filter(logits: Tensor, temperature: Tensor, top_k: Tensor, top_p: Tensor) -> Tensor:
    raise NotImplementedError


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
    raise NotImplementedError


def min_p_filter(probs: Tensor, min_p: Tensor) -> Tensor:
    raise NotImplementedError


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


def token_logprobs(logits: Tensor, token_ids: Tensor, top_k: int) -> Tuple[Tensor, Tensor, Tensor]:
    raise NotImplementedError
