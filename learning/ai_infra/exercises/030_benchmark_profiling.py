# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.profiler import ProfilerActivity, profile

# ============================================================
# 题 30：正确的计时与 profiling
# 30a) benchmark(fn, warmup=10, iters=50) -> Dict[str, float]
#      返回 {"mean_ms", "p50_ms", "p90_ms"}
#      要求：先 warmup，再用 torch.cuda.Event 逐次计时。
#           记得同步，CPU 张量时用 time.perf_counter。
# 30b) top_kernels(fn, k=3) -> List[Tuple[str, float]]
#      用 torch.profiler.profile(activities=[CPU, CUDA]) 跑一次 fn，
#      返回按 self CUDA time 降序的前 k 个 (kernel_name, self_cuda_time_us)
# 判分：benchmark 返回的 key 齐全且 mean_ms > 0。
#      top_kernels 返回非空结果，且包含矩乘相关 kernel。
# ============================================================


def benchmark(
    fn: Callable[[], object], warmup: int = 10, iters: int = 50
) -> dict[str, float]:
    if iters == 0:
        return {"mean_ms": 0, "p50_ms": 0, "p90_ms": 0}
    for _ in range(warmup - 1):
        fn()
    res = fn()
    if isinstance(res, torch.Tensor):
        device = res.device
        use_cuda = res.is_cuda
    else:
        device = torch.device("cpu")
        use_cuda = False

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed_ms = []
    if use_cuda:
        torch.cuda.synchronize(device)
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            end.synchronize()
            elapsed_ms.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            elapsed_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = sum(elapsed_ms) / len(elapsed_ms)
    elapsed_ms.sort()
    p50_ms = statistics.median(elapsed_ms)
    p90_ms = elapsed_ms[math.ceil(iters * 0.9) - 1]

    return {"mean_ms": mean_ms, "p50_ms": p50_ms, "p90_ms": p90_ms}


def top_kernels(fn: Callable[[], object], k: int = 3) -> list[tuple[str, float]]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(path))
        trace = json.loads(path.read_text(encoding="utf-8"))

    out = []
    for event in trace["traceEvents"]:
        if event.get("ph") == "X" and event.get("cat") == "kernel":
            out.append((event["name"], event["dur"]))
    out.sort(key=lambda item: item[1], reverse=True)
    return out[:k]
