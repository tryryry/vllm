from typing import List, Tuple

from torch import Tensor


# ============================================================
# 题 26：H2D 拷贝与计算重叠（需要 GPU，测量题）
#   overlap_copy_compute(host_chunks, weight, use_pinned, overlap) -> (out, elapsed_ms)
#     host_chunks: List[cpu Tensor [n, D]]（若 use_pinned 则你要负责 pin_memory）
#     weight: cuda [D, D]
#     语义：把每个 chunk 拷到 GPU 再做 chunk @ weight，最后 cat 成 [sum(n), D] 返回
#     overlap=True 时：用一个 copy stream + 一个 compute stream + event 让第 i 块的拷贝
#                     与第 i-1 块的计算重叠（记得 record_stream 防止 buffer 被提前回收）
#     overlap=False 时：串行 non_blocking=False
#     elapsed_ms 用 torch.cuda.Event 测（不要用 time.time 而不同步）
# 判分：两种模式结果一致；并打印 4 组耗时（pinned x overlap 的组合），
#      自己观察 pinned+overlap 是否最快，把结论写在注释里。
# ============================================================


def overlap_copy_compute(
    host_chunks: List[Tensor],
    weight: Tensor,
    use_pinned: bool = True,
    overlap: bool = True,
) -> Tuple[Tensor, float]:
    raise NotImplementedError
