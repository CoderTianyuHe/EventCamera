from typing import Any
import torch
from torch import Tensor


class Heaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, u_t: torch.Tensor) -> torch.Tensor:
        o_t = (u_t >= 0).to(u_t)
        ctx.save_for_backward(u_t)
        return o_t
    

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        u_t, = ctx.saved_tensors
        grad_input = grad_output * (torch.abs(u_t) <= 0.5).to(u_t)
        return grad_input

# LIF模型，主要难点在于递归算法
# 
class LIF(torch.nn.Module):
    def __init__(self, tau_m: float, u_rest: float, u_th: float) -> None:
        super().__init__()
        self.tau_m = tau_m
        self.u_rest = u_rest
        self.u_th = u_th
        self.u_t_1 = self.u_rest
        self.spiking = Heaviside()
    

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        o_seq = []
        for t in range(ir.shape[0]):
            ir_t = ir[t]
            if isinstance(self.u_t_1, float):
                self.u_t_1 = torch.full_like(ir_t, self.u_rest)
            h = -(self.u_t_1 - self.u_rest)
            du = (1 / self.tau_m) * (h + ir_t)
            u_t = du + self.u_t_1
            o_t = self.spiking.apply(u_t - self.u_th)
            self.u_t_1 = u_t * (1 - o_t) + self.u_rest * o_t
            o_seq.append(o_t)
        o = torch.stack(o_seq)
        self.u_t_1 = self.u_rest
        return o

# 为了模拟一个突触
class SNNLinear(torch.nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        t = input.shape[0]
        b = input.shape[1]
        l = list(input.shape[2:])
        reshape_shape = [t * b] + l
        output = super().forward(input.reshape(reshape_shape))
        reshape_shape = [t, b] + list(output.shape[1:])
        output = output.reshape(reshape_shape)
        return output


if __name__ == "__main__":
    batch_size = 4
    time_steps = 16

    model = torch.nn.Sequential(
        SNNLinear(2 * 128 * 128, 10),  
        LIF(2.0, 0.0, 1.0) # 类似于激活函数
    )

    x = torch.Tensor(batch_size, time_steps, 2, 128, 128)

    x = x.permute(1, 0, 2, 3, 4) # [T, B, ...]
    x = x.flatten(2)
    y = model(x) # [T, B, ...]
    y = y.permute(1, 0, 2) # [B, T, ...]

    print(y.shape)