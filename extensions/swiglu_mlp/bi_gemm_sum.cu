/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "bi_gemm/device/bi_gemm.h"
#include "epilogues.h"

namespace swiglu {

template <typename scalar_t, bool kStoreD = false>
at::Tensor bi_gemm_sum_cutlass(
    const at::Tensor& x0,
    const at::Tensor& x1,
    const at::Tensor& w0,
    const at::Tensor& w1) {
  TORCH_CHECK(x0.dim() == 2);
  TORCH_CHECK(x1.dim() == 2);
  TORCH_CHECK(w0.dim() == 2);
  TORCH_CHECK(w1.dim() == 2);

  TORCH_CHECK(x0.stride(-1) == 1);
  TORCH_CHECK(x1.stride(-1) == 1);
  TORCH_CHECK(w0.stride(-1) == 1);
  TORCH_CHECK(w1.stride(-1) == 1);

  at::cuda::CUDAGuard device_guard(x0.device());
//   at::cuda::CUDAGuard device_guard(x1.device());

  int64_t B = x0.size(0);
  int64_t I = x0.size(1);

  // note: here we w0 is not transposed
  int64_t H = w0.size(1);

  at::Tensor d0 = at::empty({B, H}, x0.options());
  at::Tensor d1 = at::empty({B, H}, x0.options());
  at::Tensor d2 = at::empty({B, H}, x0.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // templati-ze the cutlass kernel
  cutlass::gemm::GemmCoord problem_size(B, H, I);

  constexpr int kStages = 3;
  constexpr bool kSplitKSerial = false;

  using ElementOutput = scalar_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using EpilogueOutputOp01 = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using EpilogueOutputOp2 = EpilogueSum<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementOutput,
      ElementCompute>;

  const ElementCompute alpha0 = ElementCompute(1);
  const ElementCompute beta0 =
      ElementCompute(0);
  const ElementCompute alpha1 = ElementCompute(1);
  const ElementCompute beta1 =
      ElementCompute(0);

  // Good for A100
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Optionally, we might not need intermediate GEMM outputs
  constexpr bool kStoreD0 = kStoreD;
  constexpr bool kStoreD1 = kStoreD;
  using ArchTag = cutlass::arch::Sm80;

  // since no transposing is needed here, we use RowMajor for
  using BiGemm = cutlass::gemm::device::BiGemm<
      scalar_t,
      cutlass::layout::RowMajor,  // LayoutA0
      cutlass::layout::RowMajor,  // LayoutA1
      scalar_t,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp01,
      EpilogueOutputOp01,
      EpilogueOutputOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      kStages,  // 3 stages
      kStoreD0,
      kStoreD1,
      kSplitKSerial>;
  {
    cudaDeviceProp* p = at::cuda::getDeviceProperties(x0.device().index());
    TORCH_CHECK(
        p->major * 10 + p->minor >= ArchTag::kMinComputeCapability,
        "Only A100+ GPUs are supported");
  }

  int split_k_slices = BiGemm::kSplitKSerial ? 2 : 1;
  using RefA0 = typename cutlass::
      TensorRef<typename BiGemm::ElementA, typename BiGemm::LayoutA0>;
  using RefB0 = typename cutlass::
      TensorRef<typename BiGemm::ElementB, typename BiGemm::LayoutB0>;
  using RefA1 = typename cutlass::
      TensorRef<typename BiGemm::ElementA, typename BiGemm::LayoutA1>;
  using RefB1 = typename cutlass::
      TensorRef<typename BiGemm::ElementB, typename BiGemm::LayoutB1>;
  using RefC = typename cutlass::
      TensorRef<typename BiGemm::ElementC, typename BiGemm::LayoutC>;

  RefC ref_b0 = RefC{nullptr, 0};
  RefC ref_b1 = RefC{nullptr, 0};

  RefC ref_d0 = RefC{nullptr, 0};
  RefC ref_d1 = RefC{nullptr, 0};

  if (kStoreD)  {
    ref_d0 = RefC{
          (scalar_t*)d0.data_ptr(),
          typename BiGemm::LayoutC::Stride(d0.stride(0))};
    ref_d1 = RefC{
          (scalar_t*)d1.data_ptr(),
          typename BiGemm::LayoutC::Stride(d1.stride(0))};
  };

  typename BiGemm::Arguments arguments{
      cutlass::gemm::BiGemmMode::kGemm,
      problem_size,
      RefA0{
          (scalar_t*)x0.data_ptr(),
          typename BiGemm::LayoutA0::Stride(x0.stride(0))},
      RefB0{
          (scalar_t*)w0.data_ptr(),
          typename BiGemm::LayoutB0::Stride(w0.stride(0))},
      ref_b0,
      ref_d0,
      RefA1{
          (scalar_t*)x1.data_ptr(),
          typename BiGemm::LayoutA1::Stride(x1.stride(0))},
      RefB1{
          (scalar_t*)w1.data_ptr(),
          typename BiGemm::LayoutB1::Stride(w1.stride(0))},
      ref_b1,
      ref_d1,
      RefC{
          (scalar_t*)d2.data_ptr(),
          typename BiGemm::LayoutC::Stride(d2.stride(0))},
      typename BiGemm::EpilogueOutputOp0::Params{alpha0, beta0},
      typename BiGemm::EpilogueOutputOp1::Params{alpha1, beta1},
      typename BiGemm::EpilogueOutputOp2::Params{},
      split_k_slices};

  BiGemm bi_gemm;
  at::Tensor workspace = at::empty(
      {int64_t(bi_gemm.get_workspace_size(arguments))},
      x0.options().dtype(at::ScalarType::Byte));
  cutlass::Status status = bi_gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "`bi_gemm_sum` does not support this input: ",
      cutlass::cutlassGetStatusString(status));

  status = bi_gemm.initialize(arguments, (uint8_t*)workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel initialize failed");
  status = bi_gemm(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel run failed");

//   return std::make_tuple(d0, d1, d2);
  // this function is specifically used in backward pass
  return d2;
}

///////////////////////////////////////////////////////////////////////////////////

at::Tensor bi_gemm_sum_(
    const at::Tensor& x0,
    const at::Tensor& x1,
    const at::Tensor& w0,
    const at::Tensor& w1) {
  // TODO: Check all params. This would take a lot of lines of code...
  TORCH_CHECK(x0.dim() == 2);
  TORCH_CHECK(x1.dim() == 2);
  TORCH_CHECK(w0.dim() == 2);
  TORCH_CHECK(w1.dim() == 2);

  #define FWD_PARAMS x0, x1, w0, w1
  if (x0.scalar_type() == at::ScalarType::Half) {
    return bi_gemm_sum_cutlass<cutlass::half_t>(
        FWD_PARAMS);
  } else {
    TORCH_CHECK(
        x0.scalar_type() == at::ScalarType::BFloat16, "Only supports bf16/f16");
    return bi_gemm_sum_cutlass<cutlass::bfloat16_t>(
        FWD_PARAMS);
  }
}

}