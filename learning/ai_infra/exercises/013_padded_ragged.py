import torch
from torch import Tensor


# ============================================================
# 题 13：padded <-> ragged 互转（cu_seqlens 是 FlashAttention 的输入约定）
#   pad_to_ragged(padded, seq_lens):
#       padded: [B, S_max, D]，seq_lens: [B] int64 2 3
#       返回 (flat, cu_seqlens)
#       flat: [sum(seq_lens), D]，按 batch 顺序把有效 token 拼起来
#       cu_seqlens: [B+1] int32，前缀和，cu_seqlens[0] == 0
#   ragged_to_pad(flat, cu_seqlens, pad_value=0.0):
#       逆操作，返回 [B, S_max, D]，S_max = max(seq 长度)，padding 位填 pad_value
# 要求：都不允许 python 循环（提示：arange 广播出 mask，再布尔索引 / index_put）
# ============================================================
def pad_to_ragged(padded, seq_lens):
    B, S_max, D = padded.shape
    mask = torch.arange(S_max)[None, :] < seq_lens[:, None]

    # mask = torch.zeros(B, S_max, dtype=torch.bool)
    # for b in range(B):
    #     for s in range(S_max):
    #         mask[b, s] = s < seq_lens[b]
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)

    return padded[mask], cu_seqlens


def ragged_to_pad(flat: Tensor, cu_seqlens: Tensor, pad_value=0.0):
    B = cu_seqlens.shape[0] - 1
    D = flat.shape[1]

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    S_max = torch.max(seq_lens)
    mask = torch.arange(S_max)[None, :] < seq_lens[:, None]

    out = torch.full((B, S_max, D), pad_value)
    out[mask] = flat
    return out
