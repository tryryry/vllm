# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
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
    host_chunks: list[Tensor],
    weight: Tensor,
    use_pinned: bool = True,
    overlap: bool = True,
) -> tuple[Tensor, float]:
    D = weight.shape[0]
    if not host_chunks:
        return weight.new_empty((0, D)), 0.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(device=weight.device)

    start.record(current_stream)

    if use_pinned:
        # 把普通 CPU tensor 复制到页锁定内存，再返回使用这块内存的 tensor。
        host_chunks = [chunk.pin_memory() for chunk in host_chunks]

    copy_stream = torch.cuda.Stream(device=weight.device)
    compute_stream = torch.cuda.Stream(device=weight.device)

    res = []
    for chunk in host_chunks:
        if overlap:
            copy_stream.wait_event(start)
            with torch.cuda.stream(copy_stream):
                chunk_gpu = chunk.to(weight.device, non_blocking=True)
                copy_done = torch.cuda.Event()
                copy_done.record()

            # wait_event 管的是数据什么时候可读
            compute_stream.wait_event(
                copy_done
            )  # compute_stream上后续的GPU操作需要等copy_done完成，python可以继续执行
            with torch.cuda.stream(compute_stream):
                chunk_computer = chunk_gpu @ weight
                # record_stream管的是显存什么时候可以复用
                chunk_gpu.record_stream(compute_stream)
                res.append(chunk_computer)
        else:
            chunk_gpu = chunk.to(weight.device, non_blocking=False)
            res.append(chunk_gpu @ weight)
    res_stream = compute_stream if overlap else current_stream

    with torch.cuda.stream(res_stream):
        out = torch.cat(res, dim=0)
        end.record()

    end.synchronize()  # 当前cpu需要等待 res_stream 完成才能执行下一行python代码
    elapsed_ms = start.elapsed_time(end)
    return out, elapsed_ms


# 实测结论
# 正确性测试覆盖空列表、单个零行 chunk、含零行的混合长度输入。
# 每种输入都跑了 pinned 和 overlap 的四种组合，12 项验证全部通过。
# 空列表返回与 weight 同设备、同 dtype 的 (0, D) Tensor，耗时为 0.0。
# 非空输入的四种组合都与参考结果一致。
#
# 另用 8 个 float32 chunk（共 84 MiB，D=2048）测量。
# 下表是预热后每组运行 9 次的 CUDA Event 耗时中位数，单位为 ms。
#                  overlap=False  overlap=True
# use_pinned=False          6.084        12.128
# use_pinned=True         228.489       189.757
# 这组测量里普通内存串行最快。start 在 pin_memory() 前提交，CPU 锁页
# 造成的 GPU 等待可能计入 elapsed_ms。用提前锁页的输入复测，串行和重叠
# 分别为 4.090 ms 和 6.058 ms。
# 单次 profiler 采样确认了普通内存和提前锁页的输入都能发生拷贝与计算重叠。
# 是否缩短总耗时取决于输入、计时范围和实际调度。


# ------------------------------------------------------------
# 题 26 的自测入口（运行时加 --test）
# ------------------------------------------------------------
_SELF_TEST_MODES = ((False, False), (False, True), (True, False), (True, True))


def _check_correctness() -> None:
    from unittest.mock import patch

    weight_cpu = torch.randn(8, 8, dtype=torch.float16)
    chunks = [torch.randn(rows, 8, dtype=weight_cpu.dtype) for rows in (2, 0, 3)]
    expected = (torch.cat(chunks).float() @ weight_cpu.float()).half()

    with torch.cuda.stream(torch.cuda.Stream()):
        weight = weight_cpu.to("cuda", non_blocking=True)
        for pinned, overlap in _SELF_TEST_MODES:
            with patch.object(
                torch.cuda,
                "Event",
                side_effect=AssertionError("empty list used an event"),
            ):
                out, elapsed = overlap_copy_compute([], weight, pinned, overlap)
            assert isinstance(out, torch.Tensor)
            assert out.shape == (0, 8)
            assert out.device == weight.device and out.dtype == weight.dtype
            assert isinstance(elapsed, float) and elapsed == 0.0

            for inputs, reference in (([chunks[1]], expected[:0]), (chunks, expected)):
                out, elapsed = overlap_copy_compute(inputs, weight, pinned, overlap)
                assert out.device == weight.device and out.dtype == weight.dtype
                assert isinstance(elapsed, float) and elapsed > 0.0
                torch.testing.assert_close(out.cpu(), reference)

            print(f"Correctness: use_pinned={pinned}, overlap={overlap}: 3 passed")


def _measure(weight, chunks, pinned_chunks, repeats) -> None:
    from statistics import median

    cases = [
        (f"use_pinned={pinned}, overlap={overlap}", chunks, pinned, overlap)
        for pinned, overlap in _SELF_TEST_MODES
    ] + [
        (f"pre-pinned input, overlap={overlap}", pinned_chunks, True, overlap)
        for overlap in (False, True)
    ]
    samples = {label: [] for label, *_ in cases}
    for _, inputs, pinned, overlap in cases:
        overlap_copy_compute(inputs, weight, pinned, overlap)

    for index in range(repeats):
        order = cases if index % 2 == 0 else reversed(cases)
        for label, inputs, pinned, overlap in order:
            elapsed = overlap_copy_compute(inputs, weight, pinned, overlap)[1]
            samples[label].append(elapsed)

    print(f"\nCUDA Event timings for 84 MiB, {repeats} runs after warmup")
    for label, times in samples.items():
        print(f"{label}: median {median(times):.3f} ms")


def _profile_overlap(weight, chunks, pinned_chunks) -> None:
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from torch.profiler import ProfilerActivity, profile

    cases = (
        ("pageable serial", chunks, False, False),
        ("pageable overlap", chunks, False, True),
        ("pre-pinned overlap", pinned_chunks, True, True),
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        for label, inputs, pinned, overlap in cases:
            overlap_copy_compute(inputs, weight, pinned, overlap)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as trace:
                overlap_copy_compute(inputs, weight, pinned, overlap)
            trace.export_chrome_trace(str(path))
            spans = [
                event
                for event in json.loads(path.read_text())["traceEvents"]
                if event.get("ph") == "X" and event.get("dur", 0) > 0
            ]
            matmuls = {
                event["args"]["External id"]: []
                for event in spans
                if event.get("cat") == "cpu_op" and event.get("name") == "aten::mm"
            }
            copies = [
                event
                for event in spans
                if event.get("cat") == "gpu_memcpy" and "htod" in event["name"].lower()
            ]
            for event in spans:
                op_id = event.get("args", {}).get("External id")
                if event.get("cat") == "kernel" and op_id in matmuls:
                    matmuls[op_id].append(event)
            if (
                len(copies) != len(inputs)
                or len(matmuls) != len(inputs)
                or not all(matmuls.values())
            ):
                raise RuntimeError(
                    f"{label}: could not identify one H2D copy and one matmul "
                    "with CUDA kernel activity per chunk"
                )

            copies.sort(key=lambda event: event["ts"])
            ordered_matmuls = sorted(
                matmuls.values(),
                key=lambda kernels: min(event["ts"] for event in kernels),
            )
            count = sum(
                any(
                    max(copy["ts"], kernel["ts"])
                    < min(copy["ts"] + copy["dur"], kernel["ts"] + kernel["dur"])
                    for kernel in kernels
                )
                for copy, kernels in zip(copies[1:], ordered_matmuls[:-1], strict=True)
            )
            print(f"{label}: {count}/{len(inputs) - 1} adjacent pairs overlap")


def _overlap_self_test() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check exercise 26 on CUDA")
    parser.add_argument("--test", action="store_true", help="run the self-test")
    parser.add_argument("--repeats", type=int, default=9, help="timed runs per mode")
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--correctness-only", action="store_true")
    options.add_argument("--profile", action="store_true", help="inspect a CUDA trace")
    args = parser.parse_args()
    if not args.test:
        parser.print_usage()
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA device is required")

    torch.manual_seed(26)
    _check_correctness()
    if args.correctness_only:
        return

    dim = 2048
    chunks = [torch.randn(rows, dim) for rows in (1024, 1536, 2048, 768) * 2]
    weight = torch.randn(dim, dim, device="cuda")
    pinned_chunks = [chunk.pin_memory() for chunk in chunks]
    _measure(weight, chunks, pinned_chunks, args.repeats)
    if args.profile:
        _profile_overlap(weight, chunks, pinned_chunks)


if __name__ == "__main__":
    _overlap_self_test()
