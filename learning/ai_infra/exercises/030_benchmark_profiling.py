from typing import Callable, Dict, List, Tuple


# ============================================================
# 题 30：正确的计时与 profiling
# 30a) benchmark(fn, warmup=10, iters=50) -> Dict[str, float]
#      返回 {"mean_ms", "p50_ms", "p90_ms"}
#      要求：先 warmup；用 torch.cuda.Event 逐次计时（每次 record 前后不要漏 synchronize）；
#           CPU 张量时退化为 time.perf_counter
# 30b) top_kernels(fn, k=3) -> List[Tuple[str, float]]
#      用 torch.profiler.profile(activities=[CPU, CUDA]) 跑一次 fn，
#      返回按 self CUDA time 降序的前 k 个 (kernel_name, self_cuda_time_us)
# 判分：benchmark 返回的 key 齐全且 mean_ms > 0；top_kernels 返回非空且含矩乘相关 kernel。
# ============================================================


def benchmark(fn: Callable[[], object], warmup: int = 10, iters: int = 50) -> Dict[str, float]:
    raise NotImplementedError


def top_kernels(fn: Callable[[], object], k: int = 3) -> List[Tuple[str, float]]:
    raise NotImplementedError
