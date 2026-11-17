from typing import Optional

import torch
from torch import Tensor


# ============================================================
# 题 28：张量并行的两种线性层（需要 2 卡，torchrun 单独跑）
#   假设完整权重 W: [out_features, in_features]（torch.nn.Linear 的布局），x: [T, in]
#   tp_col_linear：按 out_features 切分（column parallel）
#       weight_shard: [out/W_world, in]，每个 rank 算自己那块，
#       gather_output=True 时用 all_gather 拼回 [T, out]；False 时直接返回本地 [T, out/ws]
#   tp_row_linear：按 in_features 切分（row parallel）
#       weight_shard: [out, in/ws]，输入 x 已经是切好的 [T, in/ws]，
#       本地矩乘后 all_reduce(SUM) 得到 [T, out]；bias 只能加一次（rank 0 加，或 reduce 后加）
#   判分（在 __main__ 里）：与单卡 full weight 的 F.linear 结果对齐。
# 运行：torchrun --nproc_per_node=2 028_tp_linear.py --tp
# ============================================================


def tp_col_linear(
    x: Tensor, weight_shard: Tensor, bias_shard: Optional[Tensor], gather_output: bool = True
) -> Tensor:
    raise NotImplementedError


def tp_row_linear(x_shard: Tensor, weight_shard: Tensor, bias: Optional[Tensor]) -> Tensor:
    raise NotImplementedError


# ------------------------------------------------------------
# 题 28 的自测入口（实现后运行 torchrun --nproc_per_node=2 028_tp_linear.py --tp）
# ------------------------------------------------------------
def _tp_self_test() -> None:
    import sys

    import torch.distributed as dist
    import torch.nn.functional as F

    col_fn, row_fn = tp_col_linear, tp_row_linear
    if "--ref" in sys.argv:  # 用参考实现自检本测试
        import _solutions as sol

        col_fn, row_fn = sol.tp_col_linear, sol.tp_row_linear

    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(0)  # 各 rank 用同一份完整权重再自行切片

    T, in_f, out_f = 8, 64, 96
    x = torch.randn(T, in_f, device="cuda")
    w = torch.randn(out_f, in_f, device="cuda")
    b = torch.randn(out_f, device="cuda")
    ref = F.linear(x, w, b)

    col_out = col_fn(
        x, w.chunk(ws, dim=0)[rank].contiguous(), b.chunk(ws, dim=0)[rank].contiguous(), True
    )
    row_out = row_fn(
        x.chunk(ws, dim=1)[rank].contiguous(), w.chunk(ws, dim=1)[rank].contiguous(), b
    )
    if rank == 0:
        print("col parallel max err:", (col_out - ref).abs().max().item())
        print("row parallel max err:", (row_out - ref).abs().max().item())
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys

    if "--tp" in sys.argv:
        _tp_self_test()
    else:
        print("用法: torchrun --nproc_per_node=2 028_tp_linear.py --tp")
