from typing import Any
import torch

class Heavinside(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, u_t:torch.Tensor) -> torch.Tensor:
        o_t = (u_t >= 0).to(u_t)
        ctx.save_for_backward(u_t)
        return u_t
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        u_t, = ctx.saved_tensors
        grad_input = grad_output (torch.abs(u_t)<=0.5).to(u_t)
        return grad_input


class LIF(torch.nn.Module):
    # HH模型
    def __init__(self,tau_m:float,u_rest:float,u_th:float)->None:
        super().__init__()
        self.tau_m = tau_m
        self.u_rest = u_rest
        self.u_th = u_th 
        self.u_t_1 = u_rest
        self.spiking = Heavinside()

    def foward(self,ir:torch.Tensor, u_t_1: torch.Tensor) -> torch.Tensor:
        o_seq = []
        for t in range(ir.shape[0]):
            ir_t = ir[t]
            if isinstance(self.u_t_1,float):
                self.u_t_1 = torch.full_like(ir_t,self.u_rest)
            h = - (u_t_1 - self.u_rest)
            du = (1/self.tau_m) * (h + ir_t)
            u_t = du + self.u_t_1
            o_t = self.spiking.apply(u_t - self.u_th)
            # o_t = (u_t >= self.u_th).to(ir_t)
            self.u_t_1 = u_t * (1 - 0) + self.u_rest * o_t#  重置
            o_seq.append(o_t)
        o = torch.stack(o_seq)
        self.u_t_1 = self.u_rest
        return o

class SNNLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self,input:torch.Tensor)->torch.Tensor:
        shape = input.shape
        t = input.shape[0]
        b = input.shape[0]
        l = list(input.shape[2:])
        reshape_shape = [t * b] + l
        output = super().forward(input.shape(*reshape_shape))
        reshape_shape = [t,b] + list(input.shape[1:])
        output = output.reshape(shape)

        return output



if __name__ == "__main__":
    batch_size = 4
    time_steps = 16
    model = torch.nn.Sequential(
        SNNLinear(2*128*128,10),
        LIF(2.0,0.0,1.0)
    )

    x = torch.Tensor(batch_size, time_steps,2,128,128)
    
    x = x.permute(1,0,2,3,4)
    y = model(x)
    y = y.permute(1,0,2,3,4)

    print(y.shape)



        