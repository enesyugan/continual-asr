/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/native/cuda/Loops.cuh>
#include <vector>

#include <cuda_runtime.h>
#include <cmath>

// Helper for __half and bfloat16 math if needed
template<typename T>
__device__ inline float to_float(T x) { return static_cast<float>(x); }
template<>
__device__ inline float to_float(__half x) { return __half2float(x); }
#ifdef __CUDA_BF16_TYPES_EXIST__

__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
#endif// If you support bf16, do a similar overload for __nv_bfloat16

namespace swiglu {

template <typename scalar_t>
__global__ void swiglu_bw_fused_kernel(
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x2,
    const scalar_t* __restrict__ dx4,
    scalar_t* __restrict__ dx1,  // output [B, H]
    scalar_t* __restrict__ dx2,  // output [B, H]
    scalar_t* __restrict__ x4,   // output [B, H]
    int64_t B,
    int64_t H)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H;

    using acc_t = float; // or typename at::accumulate_type<scalar_t, true>::type;

     for (; i < N; i += blockDim.x * gridDim.x) {
        scalar_t x1_ = x1[i];
        scalar_t x2_ = x2[i];
        scalar_t dx4_ = dx4[i];

        // Use float for math
        acc_t x1f = to_float(x1_);
        acc_t x2f = to_float(x2_);
        acc_t dx4f = to_float(dx4_);

        acc_t sigm = acc_t(1) / (acc_t(1) + expf(-x1f)); // sigmoid
        acc_t x3_ = sigm * x1f;  // silu (x)
        acc_t dx3_ = dx4f * x2f;
        acc_t dx2_ = dx4f * x3_;
        acc_t dx1_ = dx3_ * sigm * (acc_t(1) + x1f * (acc_t(1) - sigm));
        acc_t x4_ = x3_ * x2f;

        // Write outputs
        dx1[i] = static_cast<scalar_t>(dx1_);
        dx2[i] = static_cast<scalar_t>(dx2_);
        x4[i]  = static_cast<scalar_t>(x4_);

    }
}

// template <typename scalar_t>
// void swiglu_bw_fused_cuda(
//     const scalar_t* x1,
//     const scalar_t* x2,
//     const scalar_t* dx4,
//     scalar_t* dx1,
//     scalar_t* dx2,
//     scalar_t* x4,
//     int64_t B,
//     int64_t H)
// {
//     int N = B * H;
//     int threads = 256;
//     int blocks = (N + threads - 1) / threads;
//     swiglu_bw_fused_kernel<scalar_t><<<blocks, threads>>>(
//         x1, x2, dx4, dx1, dx2, x4, B, H
//     );
//
// }

// // For double
// template void swiglu_bw_fused_cuda<double>(
//     const double*, const double*, const double*, double*, double*, double*, int64_t, int64_t);
//
// // For float
// template void swiglu_bw_fused_cuda<float>(
//     const float*, const float*, const float*, float*, float*, float*, int64_t, int64_t);
//
//
// // For half
// template void swiglu_bw_fused_cuda<at::Half>(
//     const at::Half*, const at::Half*, const at::Half*, at::Half*, at::Half*, at::Half*, int64_t, int64_t);
//
// // For bfloat16 (add this if you want BFloat16 support!)
// template void swiglu_bw_fused_cuda<at::BFloat16>(
//     const at::BFloat16*, const at::BFloat16*, const at::BFloat16*,
//     at::BFloat16*, at::BFloat16*, at::BFloat16*, int64_t, int64_t);


std::vector<at::Tensor> swiglu_bw_fused_cuda(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& dx4) {

  int64_t B = x2.size(0);
  int64_t H = x2.size(1);
  at::Tensor dx1 = at::empty({B, H}, x2.options());
  at::Tensor dx2 = at::empty({B, H}, x2.options());
  at::Tensor x4 = at::empty({B, H}, x2.options());

  auto iter = at::TensorIteratorConfig()
                  .add_output(dx1)
                  .add_output(dx2)
                  .add_output(x4)
                  .add_input(x1)
                  .add_input(x2)
                  .add_input(dx4)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(false)
                  .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x2.scalar_type(),
      "silu_bw_fused",
      ([&] {
        using acc_t = typename at::AccumulateType<scalar_t, true>::type;
        at::native::gpu_kernel_multiple_outputs(
            iter,
            [=] GPU_LAMBDA(scalar_t x1_, scalar_t x2_, scalar_t dx4_)
                -> thrust::tuple<scalar_t, scalar_t, scalar_t> {
              acc_t sigm = acc_t(1) / (acc_t(1) + std::exp(-acc_t(x1_)));
              acc_t x3_ = sigm * x1_;
              acc_t dx3_ = acc_t(dx4_) * acc_t(x2_);
              acc_t dx2_ = acc_t(dx4_) * acc_t(x3_);
              acc_t dx1_ =
                  (dx3_ * sigm * (acc_t(1) + acc_t(x1_) * (acc_t(1) - sigm)));
              acc_t x4_ = x3_ * x2_;
              return thrust::tuple<scalar_t, scalar_t, scalar_t>{
                  dx1_, dx2_, x4_};
            });
      }));
//   return std::make_tuple(dx1dx2, x4);

    return {dx1, dx2, x4};
}

} // namespace swiglu