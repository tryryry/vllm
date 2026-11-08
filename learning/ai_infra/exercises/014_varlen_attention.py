import torch
from torch import Tensor


# ============================================================
# 题 14：变长（块对角）causal attention
#   q_flat/k_flat/v_flat: [T, H, D]，T = sum(seq_lens)，多条序列首尾拼接
#   cu_seqlens: [B+1]
#   语义：每个 token 只能注意到"同一条序列内且不晚于自己"的 token
#   返回 [T, H, D]
# 要求：不允许在 B 上写 python 循环；构造 [T, T] 的块对角 causal mask 一次算完。
# 提示：先用 cu_seqlens 生成每个 token 的 seq_id 和序列内 position，
#      mask = (seq_id_i == seq_id_j) & (pos_j <= pos_i)
# ============================================================
# flatten


def varlen_attention(
    q_flat: Tensor, k_flat: Tensor, v_flat: Tensor, cu_seqlens: Tensor
) -> Tensor:
    T, H, D = q_flat.shape
    B = cu_seqlens.shape[0] - 1
    score = torch.matmul(
        q_flat.transpose(0, 1), k_flat.transpose(0, 1).transpose(-1, -2)
    ) / (D**0.5)
    # mask
    seq_len = cu_seqlens[1:] - cu_seqlens[:-1]
    # [0 0  1 1 B-1]
    seq_id = torch.arange(B).repeat_interleave(seq_len)  # [T]
    # [0 1 T-1] - [ 0 ]
    seq_pos = torch.arange(T) - cu_seqlens[seq_id]  # [T]
    # for i in range(T):
    #     for j in range(T):
    #         mask[i][j] = (seq_id[i] == seq_id[j]) and (seq_pos[i] >= seq_pos[j])
    same_seq = seq_id[:, None] == seq_id[None, :]  # [T, T]
    causal_seq = seq_pos[:, None] >= seq_pos[None, :]  # [T, T]
    mask = same_seq & causal_seq
    score = score.masked_fill(~mask, float("-inf"))

    att = torch.softmax(score, dim=-1)
    out = torch.matmul(att, v_flat.transpose(0, 1))
    return out.transpose(0, 1)
