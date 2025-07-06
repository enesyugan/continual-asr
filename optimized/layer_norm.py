import torch
from optimized.utils import _cast_if_autocast_enabled

try:
    import fast_layer_norm_cuda
except (ModuleNotFoundError, ImportError) as e:
    fast_layer_norm_cuda = None

if fast_layer_norm_cuda is not None:
    print("[INFO] Fast & Efficient layer norm implementation detected.")


#### LAYER NORM IMPLEMENTATION ##############
class FastLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon, memory_efficient=False):
        ctx.x_shape = x.shape
        ctx.memory_efficient = memory_efficient

        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        hidden_size = gamma.numel()

        xmat = x.view((-1, hidden_size))
        ymat, mu, rsigma = fast_layer_norm_cuda.ln_fwd(xmat, gamma, beta, epsilon)
        if ctx.memory_efficient:
            ctx.save_for_backward(ymat, gamma, None, rsigma, beta)
        else:
            ctx.save_for_backward(xmat, gamma, mu, rsigma, None)

        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()  # this happens!
        x_or_y_mat, gamma, mu, rsigma, beta = ctx.saved_tensors
        dymat = dy.view(x_or_y_mat.shape)
        dxmat, dgamma, dbeta, _, _ = fast_layer_norm_cuda.ln_bwd(dymat, x_or_y_mat, mu, rsigma, gamma, beta,
                                                                 ctx.memory_efficient)
        dx = dxmat.view(ctx.x_shape)
        return dx, dgamma, dbeta, None, None


def fast_layer_norm_affine(input, weight, bias, normalized_shape, eps=1e-5, memory_efficient=False):
    args = _cast_if_autocast_enabled(input, weight, bias, eps, memory_efficient)
    with torch.amp.autocast('cuda', enabled=False):
        return FastLayerNormFN.apply(*args)


class MemoryEfficientLayerNorm(torch.nn.LayerNorm):
    """
    See LayerNorm for details.

    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set the memory efficient = True by default.
        # by benchmarking there is no speed difference and we can save ~2 GB VRAM in WhisperEncoder
        self.memory_efficient = True

    def forward(self, input, fast=True):
        if input.is_cuda and fast and fast_layer_norm_cuda is not None \
                and input.size(-1) in [768, 1024, 1280, 1536, 2048, 3072, 4096] and self.bias is not None:
            # Note: this layer norm only supports a number of dimension
            return fast_layer_norm_affine(input, self.weight, self.bias,
                                          self.normalized_shape, self.eps, self.memory_efficient)

        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
