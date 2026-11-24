from torch import Tensor


# ============================================================
# 题 25：CUDA Graph 捕获一个 decode step（需要 GPU）
# 要求实现 CudaGraphDecoder：
#   __init__(self, weight: Tensor, max_bs: int)
#       weight: [D, D] (cuda)；预分配静态输入 [max_bs, D] 和静态输出缓冲
#   capture(self)
#       用 torch.cuda.graph 捕获 "out = silu(x @ weight) @ weight" 这个 step；
#       捕获前记得 warmup（在 side stream 上跑几次），否则容易失败
#   replay(self, x: Tensor) -> Tensor
#       x: [bs, D]，bs <= max_bs。把 x 拷进静态输入的前 bs 行，replay，
#       返回静态输出前 bs 行的 clone()
# 判分：replay 结果与 eager 计算一致；连续用不同输入 replay 结果要跟着变；
#      且 replay 期间新增显存分配为 0（因此必须写进预分配缓冲、返回前再 clone）。
# ============================================================


class CudaGraphDecoder:
    def __init__(self, weight: Tensor, max_bs: int):
       self.weight = weight
       self.max_bs = max_bs
       
       D = weight.shape[0]
       self.statis_input = weight.new_zeros((max_bs, D))
       self.static_output = weight.new_empty((max_bs, D))

    def capture(self) -> None:
        raise NotImplementedError

    def replay(self, x: Tensor) -> Tensor:
        raise NotImplementedError
