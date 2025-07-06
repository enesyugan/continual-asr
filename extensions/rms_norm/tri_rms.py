import torch
from tri_rmsnorm import (
    _rms_norm_fwd_fused,
    _rms_norm_bwd_dx_fused,
)


class RMSNormFunctionCustomKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.x_shape = x.shape
        hidden_size = weight.numel()
        xmat = x.view((-1, hidden_size))
        M, N = xmat.shape
        bias = torch.zeros_like(weight)
        y = torch.empty_like(x)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        _rms_norm_fwd_fused[(M,)](xmat, y, weight, bias, rstd, x.stride(0), N, eps, BLOCK_SIZE=1024)
        ctx.save_for_backward(xmat, weight, bias, rstd)
        ctx.eps = eps
        ctx.N = N
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()  # this happens!
        x, weight, bias, rstd = ctx.saved_tensors
        dymat = dy.view(x.shape)
        eps = ctx.eps
        N = ctx.N
        M = x.shape[0]
        dx = torch.empty_like(x)
        _dw = torch.empty_like(weight)
        _db = torch.empty_like(bias)
        locks = torch.zeros(2 * 32, dtype=torch.int32, device=x.device)

        # print(dx.size(), dymat.size(), _dw.size(), _db.size(), x.size(), weight.size(), bias.size(), rstd.size())
        _rms_norm_bwd_dx_fused[(M,)](
            dx,
            dymat,
            _dw,
            _db,
            x,
            weight,
            bias,
            rstd,
            locks,
            x.stride(0),
            N,
            eps,
            GROUP_SIZE_M=32,
            BLOCK_SIZE_N=1024,
        )

        dx = dx.view(ctx.x_shape)

        return dx, _dw, None


def _tritton_rms_norm(x, weight, epsilon):

    return RMSNormFunctionCustomKernel.apply(x, weight, epsilon)