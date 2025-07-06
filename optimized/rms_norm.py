import torch

try:
    import fast_rms_norm_cuda
except (ModuleNotFoundError, ImportError) as e:
    fast_rms_norm_cuda = None

if fast_rms_norm_cuda is not None:
    print("[INFO] Fast & Efficient RMS norm implementation detected.")

from optimized.utils import _cast_if_autocast_enabled
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class FastRMSNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, epsilon):
        ctx.x_shape = x.shape

        x = x.contiguous()
        gamma = gamma.contiguous()
        hidden_size = gamma.numel()

        xmat = x.view((-1, hidden_size))
        ymat, rss = fast_rms_norm_cuda.rms_fwd(xmat, gamma, epsilon)

        ctx.save_for_backward(xmat, gamma, rss)

        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()  # this happens!
        xmat, gamma, rss = ctx.saved_tensors
        dymat = dy.view(xmat.shape)
        dxmat, dgamma, _ = fast_rms_norm_cuda.rms_bwd(dymat, xmat, rss, gamma)
        dx = dxmat.view(ctx.x_shape)

        return dx, dgamma, None


def _fast_rms_norm(x, weight, epsilon):
    args = _cast_if_autocast_enabled(x, weight, epsilon)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        return FastRMSNormFN.apply(*args)


class EfficientQwen2RMSNorm(Qwen2RMSNorm):


    def forward(self, x):
        return _fast_rms_norm(x, self.weight, self.variance_epsilon)

    def extra_repr(self):
        # TODO add dropout probability
        s = F"Fast RMS Norm w/ Hidden sizes: {self.weight.size(0)}"
        return s
