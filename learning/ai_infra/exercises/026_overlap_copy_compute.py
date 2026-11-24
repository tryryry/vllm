import torch
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
    if use_pinned:
        host_chunks = [chunk.pin_memory() for chunk in host_chunks]
    
    copy_stream = torch.cuda.Stream(device=weight.device)
    compute_stream = torch.cuda.Stream(device=weight.device)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    current_stream = torch.cuda.current_stream(device=weight.device)
    start.record(current_stream)
    
    res = []
    for chunk in host_chunks:
        if overlap:
            copy_stream.wait_event(start)
            with torch.cuda.stream(copy_stream):
                chunk_gpu = chunk.to(weight.device, non_blocking=True)
                copy_done = torch.cuda.Event()
                copy_done.record()
            
            compute_stream.wait_event(copy_done) # compute_stream上后续的GPU操作需要等copy_done完成，python可以继续执行
            with torch.cuda.stream(compute_stream):
                chunk_computer = chunk_gpu @ weight
                chunk_gpu.record_stream(compute_stream)
                res.append(chunk_computer)
        else:
            chunk_gpu = chunk.to(weight.device, non_blocking=False)
            res.append(chunk_gpu @ weight)
    res_stream = compute_stream if overlap else current_stream
    
    with torch.cuda.stream(res_stream):
        out = torch.cat(res, dim=0)
        end.record()
    
    end.synchronize() # 当前cpu需要等待 res_stream 完成才能执行下一行python代码
    elapsed_ms = start.elapsed_time(end)
    return out, elapsed_ms