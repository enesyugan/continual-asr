import torch
import torch.nn as nn
import unittest
from time import time
import numpy as np
from xformers.ops import swiglu, unbind, SwiGLUFusedOp
import tqdm
from typing import Dict, Optional, Sequence, Tuple, Union
from optimized.utils import _cast_if_autocast_enabled

try:
    import swiglu_mlp_cuda
except (ModuleNotFoundError, ImportError) as e:
    swiglu_mlp_cuda = None

if swiglu_mlp_cuda is not None:
    print("[INFO] Fused MLPSwiGLU implementation detected.")


#### XFORMERS' version (better numerical precision)
class SwiGLU_mlp(nn.Module):
    """
    A Module that encapsulates the call to :attr:`xformers.ops.swiglu_mlp`,
    and holds the weights for the 3 linear layers
    """

    def __init__(
            self,
            hidden_size, intermediate_size,
    ) -> None:
        """

        Args:
            hidden_size:
            intermediate_size:
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.op = SwiGLUFusedOp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes :attr:`swiglu_mlp` with the module's weights

        Args:
            x (torch.Tensor): A Tensor of shape ``[..., in_features]``

        Returns:
            torch.Tensor: A Tensor of shape ``[..., out_features]``
        """
        return swiglu(x, *self._ordered_params(), op=self.op)

    def _ordered_params(self):
        """Used for testing - returns ordered arguments for operators"""

        w1, w2, w3 = self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight
        b1, b2, b3 = None, None, None
        # return [
        #     w1,
        #     w2,
        #     w3,
        # ]
        return (
            w1,
            b1,
            w2,
            b2,
            w3,
            b3
        )


#### FUSED MLP SWIGLU IMPLEMENTATION ##############
class MLPSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, w3):
        ctx.x_shape = x.shape
        ctx.need_gradWeight = w1.requires_grad

        x = x.contiguous()
        input_size = x.size(-1)
        output_size = w3.size(0)

        y_shape = list(ctx.x_shape)
        y_shape[-1] = output_size
        y_shape = torch.Size(y_shape)

        x = x.view(-1, input_size)
        x1, x2, x5 = swiglu_mlp_cuda.forward(x, w1, w2, w3)

        ctx.save_for_backward(x, w1, w2, w3, x1, x2)

        return x5.view(y_shape)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()  # this happens!
        x, w1, w2, w3, x1, x2 = ctx.saved_tensors

        output_size = w3.size(0)

        dy = dy.view(-1, output_size)
        if ctx.need_gradWeight:
            dx, dw1, dw2, dw3 = swiglu_mlp_cuda.backward_full(dy, x, w1, w2, w3, x1, x2)

        else:
            # dx = swiglu_mlp_cuda.backward_gradInput(dy, x, w1, w2, w3, x1, x2)
            # dw1, dw2, dw3 = None, None, None

            dx4 = torch.mm(dy, w3)
            # std::vector < at::Tensor > swiglu_grads = swiglu_bw_fused(x1, x2, dx4);
            # dx1 = swiglu_grads[0];
            # dx2 = swiglu_grads[1];
            # x4 = swiglu_grads[2];
            dx1, dx2, x4 = swiglu_mlp_cuda.swiglu_bw_fused(x1, x2, dx4)

            # dx = swiglu_mlp_cuda.bi_gemm_sum(dx1, dx2, w1, w2)

            #
            dx = torch.mm(dx1, w1)
            dx.addmm_(dx2, w2, beta=1, alpha=1)

            dw1, dw2, dw3 = None, None, None

        dx = dx.view(ctx.x_shape)

        return dx, dw1, dw2, dw3


def swiglu_mlp_function(x, w1, w2, w3):
    args = _cast_if_autocast_enabled(x, w1, w2, w3)
    with torch.amp.autocast('cuda', enabled=False):
        return MLPSwiGLUFunction.apply(*args)


from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MLPSwiGLU(nn.Module):
    """
    A Module that encapsulates the call to :attr:`xformers.ops.swiglu_mlp`,
    and holds the weights for the 3 linear layers
    """

    def __init__(
            self,
            hidden_size, intermediate_size,
    ) -> None:
        """

        Args:
            hidden_size:
            intermediate_size:
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes :attr:`swiglu_mlp` with the module's weights

        Args:
            x (torch.Tensor): A Tensor of shape ``[..., in_features]``

        Returns:
            torch.Tensor: A Tensor of shape ``[..., out_features]``
        """
        return swiglu_mlp_function(x, *self._ordered_params())

    def _ordered_params(self):
        """Used for testing - returns ordered arguments for operators"""

        w1, w2, w3 = self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight
        b1, b2, b3 = None, None, None
        return [
            w1,
            w2,
            w3,
        ]


def copy_weight(src, tgt):
    intermediate_size = src.intermediate_size

    with torch.no_grad():
        # tgt.w12.weight.data[:intermediate_size].copy_(src.gate_proj.weight)
        # tgt.w12.weight.data[intermediate_size:].copy_(src.up_proj.weight)

        tgt.gate_proj.weight.data.copy_(src.gate_proj.weight)
        tgt.up_proj.weight.data.copy_(src.up_proj.weight)

        tgt.down_proj.weight.data.copy_(src.down_proj.weight)


def test_grad(src, tgt, atol=1e-5, rtol=1e-5):
    intermediate_size = src.intermediate_size

    w1_grad = tgt.gate_proj.weight.grad.detach().cpu().float().numpy()
    w2_grad = tgt.up_proj.weight.grad.detach().cpu().float().numpy()
    w3_grad = tgt.down_proj.weight.grad.detach().cpu().float().numpy()

    np.testing.assert_allclose(
        w1_grad,
        src.gate_proj.weight.grad.detach().cpu().float().numpy(),
        atol=atol, rtol=rtol)

    np.testing.assert_allclose(
        w2_grad,
        src.up_proj.weight.grad.detach().cpu().float().numpy(),
        atol=atol, rtol=rtol)

    np.testing.assert_allclose(
        w3_grad,
        src.down_proj.weight.grad.detach().cpu().float().numpy(),
        atol=atol, rtol=rtol)


def memory_benchmark(module, input):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = module(input)  # Forward only
    mem = torch.cuda.max_memory_allocated()
    return mem / 1024 ** 2  # MiB


def memory_benchmark_with_backward(module, input):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = module(input)
    grad = torch.randn_like(out)
    out.backward(grad)
    mem = torch.cuda.max_memory_allocated()
    return mem / 1024 ** 2  # MiB


if __name__ == '__main__':
    class TestSwiGLU(unittest.TestCase):
        def passed_test_numeric(self):
            print("Test numeric 3D .... in float32")
            # for dropout in [0.0, 0.2, 0.5, 0.7]:
            bsz = 4
            seq_len = 16
            hidden_sizes = [
                768,
                1024,
                1280,
                1536,
                2048,
                2304,
                3072,
                3584,
                3840,
                4096,
                # 5120,
                # 6144,
                # 8192,
                # 10240,
                # 12288,
                # 12800,
                # 15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in hidden_sizes:
                ref_model = Qwen2MLP(hidden, hidden * 4)
                new_model = MLPSwiGLU(hidden, hidden * 4)

                ref_model.to(torch.float32).cuda()
                new_model.to(torch.float32).cuda()
                print(ref_model, new_model, "\n")
                copy_weight(ref_model, new_model)

                test_input = torch.empty(seq_len, bsz, hidden, device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()

                ref_out = ref_model(ref_input)
                test_out = new_model(test_input)

                np.testing.assert_allclose(
                    ref_out.detach().cpu().numpy(),
                    test_out.detach().cpu().numpy(),
                    atol=1e-5, rtol=1e-4)

                test_out.mean().mul(10.).backward()
                ref_out.mean().mul(10.).backward()
                np.testing.assert_allclose(
                    test_input.grad.detach().cpu().numpy(),
                    ref_input.grad.detach().cpu().numpy(),
                    atol=1e-7, rtol=1e-5)

                test_grad(ref_model, new_model, atol=1e-7, rtol=1e-5)

        def _test_numeric_bfloat16(self):
            print("Test numeric 3D .... in bfloat16")
            bsz = 4
            seq_len = 16
            hidden_sizes = [
                768,
                1024,
                1280,
                1536,
                2048,
                2304,
                3072,
                3840,
                4096,
                # 5120,
                # 6144,
                # 8192,
                # 10240,
                # 12288,
                # 12800,
                # 15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in hidden_sizes:
                ref_model = Qwen2MLP(hidden, hidden * 4)
                new_model = MLPSwiGLU(hidden, hidden * 4)

                ref_model.to(torch.bfloat16).cuda()
                new_model.to(torch.bfloat16).cuda()
                print(ref_model, new_model, "\n")
                copy_weight(ref_model, new_model)

                test_input = torch.empty(seq_len, bsz, hidden,
                                         dtype=torch.bfloat16, device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()

                ref_out = ref_model(ref_input)
                test_out = new_model(test_input)

                np.testing.assert_allclose(
                    ref_out.detach().cpu().float().numpy(),
                    test_out.detach().cpu().float().numpy(),
                    atol=1e-1, rtol=5e-2)

                test_out.mean().mul(10.).backward()
                ref_out.mean().mul(10.).backward()
                np.testing.assert_allclose(
                    test_input.grad.detach().cpu().float().numpy(),
                    ref_input.grad.detach().cpu().float().numpy(),
                    atol=1e-1, rtol=5e-2)

                test_grad(ref_model, new_model, atol=1e-1, rtol=5e-2)

        def _test_numeric_bfloat16_gradInput(self):
            print("Test numeric gradInput (no weight backprop) 3D .... in bfloat16")
            bsz = 4
            seq_len = 16
            hidden_sizes = [
                768,
                1024,
                1280,
                1536,
                2048,
                2304,
                3072,
                3840,
                4096,
                # 5120,
                # 6144,
                # 8192,
                # 10240,
                # 12288,
                # 12800,
                # 15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in hidden_sizes:
                ref_model = Qwen2MLP(hidden, hidden * 4)
                new_model = MLPSwiGLU(hidden, hidden * 4)

                for param in ref_model.parameters():
                    param.requires_grad = False

                for param in new_model.parameters():
                    param.requires_grad = False

                ref_model.to(torch.bfloat16).cuda()
                new_model.to(torch.bfloat16).cuda()
                print(ref_model, new_model, "\n")
                copy_weight(ref_model, new_model)

                test_input = torch.empty(seq_len, bsz, hidden,
                                         dtype=torch.bfloat16, device="cuda").uniform_(-1., 1.).requires_grad_()
                ref_input = test_input.clone().detach().requires_grad_()

                ref_out = ref_model(ref_input)
                test_out = new_model(test_input)

                np.testing.assert_allclose(
                    ref_out.detach().cpu().float().numpy(),
                    test_out.detach().cpu().float().numpy(),
                    atol=1e-1, rtol=5e-2)

                test_out.mean().mul(10.).backward()
                ref_out.mean().mul(10.).backward()
                np.testing.assert_allclose(
                    test_input.grad.detach().cpu().float().numpy(),
                    ref_input.grad.detach().cpu().float().numpy(),
                    atol=1e-1, rtol=5e-2)

                # test_grad(ref_model, new_model, atol=1e-1, rtol=5e-2)

        def performance_half(self):
            num_iters = 32
            bsz = 4
            seq_len = 1024
            print("Testing performance ...")
            hidden_sizes = [
                # 768,
                1024,
                1280,
                1536,
                2048,
                2304,
                3072,
                3840,
                4096,
                # 5120,
                # 6144,
                # 8192,
                # 10240,
                # 12288,
                # 12800,
                # 15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in hidden_sizes:
                ref_model = Qwen2MLP(hidden, hidden * 4)
                new_model = MLPSwiGLU(hidden, hidden * 4)

                ref_model.to(torch.float16).cuda()
                new_model.to(torch.float16).cuda()
                print("\n", ref_model, new_model, "\n")
                copy_weight(ref_model, new_model)

                test_input = torch.empty(seq_len * bsz, hidden,
                                         device="cuda", dtype=torch.float16).uniform_(-1., 1.).requires_grad_()

                ref_input = test_input.clone().detach().requires_grad_()

                # Warm up GPU
                for _ in range(num_iters):
                    ref_out = ref_model(ref_input)
                    ref_loss = ref_out.mean()
                    ref_model.zero_grad()
                    ref_loss.backward()
                    test_out = new_model(test_input)
                    test_loss = test_out.mean()
                    new_model.zero_grad()
                    test_loss.backward()

                total_fwd_time = 0
                total_bwd_time = 0
                torch.cuda.profiler.start()
                torch.cuda.synchronize()
                for _ in range(num_iters):
                    start_time = time()
                    ref_out = ref_model(ref_input)
                    torch.cuda.synchronize()
                    total_fwd_time += (time() - start_time)

                    ref_loss = ref_out.mean()
                    ref_model.zero_grad()

                    start_time = time()
                    ref_loss.backward()
                    torch.cuda.synchronize()
                    total_bwd_time += (time() - start_time)

                print(F"\nPytorch MLP fwd time {total_fwd_time * 1000. / num_iters:.4f} ms")
                print(F"Pytorch MLP bwd time {total_bwd_time * 1000. / num_iters:.4f} ms")

                total_fwd_time = 0
                total_bwd_time = 0
                torch.cuda.synchronize()
                for _ in range(num_iters):
                    start_time = time()
                    test_out = new_model(test_input)
                    torch.cuda.synchronize()
                    total_fwd_time += (time() - start_time)

                    test_loss = test_out.mean()
                    new_model.zero_grad()

                    start_time = time()
                    test_loss.backward()
                    torch.cuda.synchronize()
                    total_bwd_time += (time() - start_time)

                print(F"\nXformers MLP SiLU fwd time {total_fwd_time * 1000. / num_iters:.4f} ms")
                print(F"Xformers MLP SiLU bwd time {total_bwd_time * 1000. / num_iters:.4f} ms")

        def _test_performance_bhalf(self):
            num_iters = 32
            bsz = 4
            seq_len = 1024
            print("Testing performance ...")
            hidden_sizes = [
                # 768,
                1024,
                1280,
                1536,
                2048,
                2304,
                3072,
                3840,
                4096,
                # 5120,
                # 6144,
                # 8192,
                # 10240,
                # 12288,
                # 12800,
                # 15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in hidden_sizes:
                ref_model = Qwen2MLP(hidden, hidden * 4)
                new_model = MLPSwiGLU(hidden, hidden * 4)

                ref_model.to(torch.bfloat16).cuda()
                new_model.to(torch.bfloat16).cuda()
                print("\n", ref_model, new_model, "\n")
                copy_weight(ref_model, new_model)

                test_input = torch.empty(seq_len * bsz, hidden,
                                         device="cuda", dtype=torch.bfloat16).uniform_(-1., 1.).requires_grad_()

                ref_input = test_input.clone().detach().requires_grad_()

                # Warm up GPU
                for _ in range(num_iters):
                    ref_out = ref_model(ref_input)
                    ref_loss = ref_out.mean()
                    ref_model.zero_grad()
                    ref_loss.backward()
                    test_out = new_model(test_input)
                    test_loss = test_out.mean()
                    new_model.zero_grad()
                    test_loss.backward()

                total_fwd_time = 0
                total_bwd_time = 0
                torch.cuda.profiler.start()
                torch.cuda.synchronize()
                for _ in range(num_iters):
                    start_time = time()
                    ref_out = ref_model(ref_input)
                    torch.cuda.synchronize()
                    total_fwd_time += (time() - start_time)

                    ref_loss = ref_out.mean()
                    ref_model.zero_grad()

                    start_time = time()
                    ref_loss.backward()
                    torch.cuda.synchronize()
                    total_bwd_time += (time() - start_time)

                print(F"\nPytorch MLP fwd time {total_fwd_time * 1000. / num_iters:.4f} ms")
                print(F"Pytorch MLP bwd time {total_bwd_time * 1000. / num_iters:.4f} ms")

                total_fwd_time = 0
                total_bwd_time = 0
                torch.cuda.synchronize()
                for _ in range(num_iters):
                    start_time = time()
                    test_out = new_model(test_input)
                    torch.cuda.synchronize()
                    total_fwd_time += (time() - start_time)

                    test_loss = test_out.mean()
                    new_model.zero_grad()

                    start_time = time()
                    test_loss.backward()
                    torch.cuda.synchronize()
                    total_bwd_time += (time() - start_time)

                print(F"\nXformers MLP SiLU fwd time {total_fwd_time * 1000. / num_iters:.4f} ms")
                print(F"Xformers MLP SiLU bwd time {total_bwd_time * 1000. / num_iters:.4f} ms")

        def test_performance_bhalf_freeze_grad(self):
            num_iters = 32
            bsz = 4
            seq_len = 1024
            print("Testing performance ...")
            hidden_sizes = [
                # 768,
                1024,
                1280,
                1536,
                2048,
                2304,
                3072,
                3840,
                4096,
                # 5120,
                # 6144,
                # 8192,
                # 10240,
                # 12288,
                # 12800,
                # 15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in hidden_sizes:
                ref_model = Qwen2MLP(hidden, hidden * 4)
                new_model = MLPSwiGLU(hidden, hidden * 4)

                ref_model.to(torch.bfloat16).cuda()
                new_model.to(torch.bfloat16).cuda()
                print("\n", ref_model, new_model, "\n")
                copy_weight(ref_model, new_model)

                for param in ref_model.parameters():
                    param.requires_grad = False

                for param in new_model.parameters():
                    param.requires_grad = False

                test_input = torch.empty(seq_len * bsz, hidden,
                                         device="cuda", dtype=torch.bfloat16).uniform_(-1., 1.).requires_grad_()

                ref_input = test_input.clone().detach().requires_grad_()

                # Warm up GPU
                for _ in range(num_iters):
                    ref_out = ref_model(ref_input)
                    ref_loss = ref_out.mean()
                    ref_model.zero_grad()
                    ref_loss.backward()
                    test_out = new_model(test_input)
                    test_loss = test_out.mean()
                    new_model.zero_grad()
                    test_loss.backward()

                total_fwd_time = 0
                total_bwd_time = 0
                torch.cuda.profiler.start()
                torch.cuda.synchronize()
                for _ in range(num_iters):
                    start_time = time()
                    ref_out = ref_model(ref_input)
                    torch.cuda.synchronize()
                    total_fwd_time += (time() - start_time)

                    ref_loss = ref_out.mean()
                    ref_model.zero_grad()

                    start_time = time()
                    ref_loss.backward()
                    torch.cuda.synchronize()
                    total_bwd_time += (time() - start_time)

                print(F"\nPytorch MLP fwd time {total_fwd_time * 1000. / num_iters:.4f} ms")
                print(F"Pytorch MLP bwd time gradInput {total_bwd_time * 1000. / num_iters:.4f} ms")

                total_fwd_time = 0
                total_bwd_time = 0
                torch.cuda.synchronize()
                for _ in range(num_iters):
                    start_time = time()
                    test_out = new_model(test_input)
                    torch.cuda.synchronize()
                    total_fwd_time += (time() - start_time)

                    test_loss = test_out.mean()
                    new_model.zero_grad()

                    start_time = time()
                    test_loss.backward()
                    torch.cuda.synchronize()
                    total_bwd_time += (time() - start_time)

                print(F"\nXformers MLP SiLU fwd time {total_fwd_time * 1000. / num_iters:.4f} ms")
                print(F"Xformers MLP SiLU bwd time gradInput {total_bwd_time * 1000. / num_iters:.4f} ms")

        def passed_test_mem(self):
            num_iters = 4
            bsz = 16
            seq_len = 4096
            print("Testing memory ...")
            hidden_sizes = [
                # 768,
                1024,
                # 1280,
                # 1536,
                2048,
                # 2304,
                # 3072,
                # 3840,
                4096,
                # 5120,
                # 6144,
                # 8192,
                # 10240,
                # 12288,
                # 12800,
                # 15360,
                # 16384,
                # 18432,
                # 20480,
                # 24576,
                # 25600,
                # 30720,
                # 32768,
                # 40960,
                # 49152,
                # 65536,
            ]
            for hidden in hidden_sizes:
                ref_model = Qwen2MLP(hidden, hidden * 4)
                new_model = MLPSwiGLU(hidden, hidden * 4)

                ref_model.to(torch.bfloat16).cuda()
                new_model.to(torch.bfloat16).cuda()
                print("\n", ref_model, new_model, "\n")
                copy_weight(ref_model, new_model)

                test_input = torch.empty(seq_len * bsz, hidden,
                                         device="cuda", dtype=torch.bfloat16).uniform_(-1., 1.).requires_grad_()

                ref_input = test_input.clone().detach().requires_grad_()

                # Warm up GPU
                # print("Warming up ...")
                # for _ in range(num_iters):
                #     ref_out = ref_model(ref_input)
                #     ref_loss = ref_out.mean()
                #     ref_model.zero_grad()
                #     ref_loss.backward()
                #     test_out = new_model(test_input)
                #     test_loss = test_out.mean()
                #     new_model.zero_grad()
                #     test_loss.backward()

                total_mem = 0
                # for _ in range(num_iters):
                for _ in tqdm.tqdm(range(num_iters), desc="Measuring memory"):
                    mem_qwen = memory_benchmark_with_backward(new_model, ref_input)
                    total_mem += mem_qwen

                mem_qwen = total_mem / num_iters
                print(f"Xformers 2MLP peak memory: {mem_qwen:.2f} MiB")

                total_mem = 0
                # for _ in range(num_iters):
                for _ in tqdm.tqdm(range(num_iters), desc="Measuring memory"):
                    mem_qwen = memory_benchmark_with_backward(ref_model, ref_input)
                    total_mem += mem_qwen

                mem_qwen = total_mem / num_iters
                print(f"Qwen2MLP peak memory: {mem_qwen:.2f} MiB")


    unittest.main()
