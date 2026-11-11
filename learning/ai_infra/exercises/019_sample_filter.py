from torch import Tensor


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


def sample_filter(
    logits: Tensor, temperature: Tensor, top_k: Tensor, top_p: Tensor
) -> Tensor:
    raise NotImplementedError
