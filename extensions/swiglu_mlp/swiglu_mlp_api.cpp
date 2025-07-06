#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <vector>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"

namespace swiglu {

    bool shapesMatch(at::Tensor x, std::vector<int64_t> expectedShape) {
      if (x.dim() != int64_t(expectedShape.size())) {
        return false;
      }
      for (size_t i = 0; i < expectedShape.size(); ++i) {
        if (expectedShape[i] != -1 && x.size(i) != expectedShape[i]) {
          return false;
        }
      }
      return true;
    }

    std::string shapeToStr(c10::IntArrayRef shape) {
      std::stringstream oss;
      oss << "[" << shape[0];
      for (size_t i = 1; i < shape.size(); ++i) {
        oss << ", " << shape[i];
      }
      oss << "]";
      return oss.str();
    }

#define TORCH_INTERNAL_ASSERT_SHAPE(X, ...) \
  TORCH_INTERNAL_ASSERT(                    \
      shapesMatch(X, {__VA_ARGS__}),        \
      "%s: shape is %s but expected %s",    \
      #X,                                   \
      shapeToStr(X.sizes()).c_str(),        \
      shapeToStr({__VA_ARGS__}).c_str());

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_lhs_silu_and_mul_(
    const at::Tensor& x,
    const at::Tensor& w0,
    const at::Tensor& w1);

    std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_lhs_silu_and_mul(
        const at::Tensor& x,
        const at::Tensor& w0,
        const at::Tensor& w1) {
      // TODO: Check all params. This would take a lot of lines of code...
      TORCH_CHECK(x.dim() == 2);
      TORCH_CHECK(w0.dim() == 2);
      TORCH_CHECK(w1.dim() == 2);

    #define DUAL_GEMM_PARAMS x, w0, w1
      return dual_gemm_lhs_silu_and_mul_(
            DUAL_GEMM_PARAMS);
    }

    ////////////////////////////////////////////////////////////////////////////////////////

    at::Tensor bi_gemm_sum_(
    const at::Tensor& x0,
    const at::Tensor& x1,
    const at::Tensor& w0,
    const at::Tensor& w1);

    at::Tensor bi_gemm_sum(
        const at::Tensor& x0,
        const at::Tensor& x1,
        const at::Tensor& w0,
        const at::Tensor& w1) {
      // TODO: Check all params. This would take a lot of lines of code...
      TORCH_CHECK(x0.dim() == 2);
      TORCH_CHECK(x1.dim() == 2);
      TORCH_CHECK(w0.dim() == 2);
      TORCH_CHECK(w1.dim() == 2);

    #define BI_GEMM_PARAMS x0, x1, w0, w1
      return bi_gemm_sum_(
            BI_GEMM_PARAMS);
    }

    ////////////////////////////////////////////////////////////////////////////////////////

//  template <typename scalar_t>
//  void swiglu_bw_fused_cuda(
//    const scalar_t* x1,
//    const scalar_t* x2,
//    const scalar_t* dx4,
//    scalar_t* dx1,
//    scalar_t* dx2,
//    scalar_t* x4,
//    int64_t B,
//    int64_t H);

  std::vector<at::Tensor> swiglu_bw_fused_cuda(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& dx4);

  std::vector<at::Tensor> swiglu_bw_fused(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& dx4) {
  // TODO: Check all params. This would take a lot of lines of code...
  TORCH_CHECK(x2.dim() == 2);
  TORCH_CHECK(dx4.dim() == 2);
  TORCH_CHECK(x2.sym_size(0) == dx4.sym_size(0));
  TORCH_CHECK(x2.sym_size(1) == dx4.sym_size(1));

  at::SymInt B = x2.sym_size(0);
  at::SymInt H = x2.sym_size(1);
//  at::Tensor dx1dx2 = at::empty_symint({B, 2, H}, x2.options());
//  at::Tensor  x4 = at::empty_symint({B, H}, x2.options());
//  at::Tensor dx1 = at::empty_symint({B, H}, x2.options());
//  at::Tensor dx2 = at::empty_symint({B, H}, x2.options());

  // Regular mode logic: perform actual computations
//  at::Tensor dx1 = dx1dx2.select(1, 0);
//  at::Tensor dx2 = dx1dx2.select(1, 1);
  return swiglu_bw_fused_cuda(x1, x1, dx4);

//  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,
//                                  at::ScalarType::BFloat16,
//                                  x1.scalar_type(),
//                                  "swiglu_bw_fused_cuda", ([&] {
//    swiglu_bw_fused_cuda<scalar_t>(
//        x1.data_ptr<scalar_t>(),
//        x2.data_ptr<scalar_t>(),
//        dx4.data_ptr<scalar_t>(),
//        dx1.data_ptr<scalar_t>(),
//        dx2.data_ptr<scalar_t>(),
//        x4.data_ptr<scalar_t>(),
//        x1.size(0),
//        x1.size(1)
//    );
//  }));

//  return {dx1, dx2, x4};
}


// forward function
std::vector<at::Tensor> forward(
  const at::Tensor& x,
  const at::Tensor& w1,
  const at::Tensor& w2,
  const at::Tensor w3) {

    at::Tensor x1, x2, x4;
    std::tie(x1, x2, x4) = dual_gemm_lhs_silu_and_mul(x, w1, w2);

    auto x5 = torch::mm(x4, w3.t());

    return {x1, x2, x5};
}

    // Backward function
std::vector<at::Tensor> backward_full(at::Tensor& dx5,
                                      const at::Tensor& x,
                                      const at::Tensor& w1,
                                      const at::Tensor& w2,
                                      const at::Tensor& w3,
                                      at::Tensor& x1,
                                      at::Tensor& x2) {

    int64_t B = x.size(0);
    int64_t H = x2.size(1);
    int64_t I = x.size(1);
    int64_t O = dx5.size(1);

    TORCH_INTERNAL_ASSERT_SHAPE(x1, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(x2, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(dx5, B, O);
    TORCH_INTERNAL_ASSERT_SHAPE(w1, H, I);
    TORCH_INTERNAL_ASSERT_SHAPE(w1, H, I);
    TORCH_INTERNAL_ASSERT_SHAPE(w3, O, H);

    // Compute BW
    at::Tensor dx1, dx2, x4;
    TORCH_INTERNAL_ASSERT(dx5.size(1) == w3.size(0));
    auto dx4 = torch::mm(dx5, w3);
    std::vector<at::Tensor> swiglu_grads = swiglu_bw_fused(x1, x2, dx4);
    dx1 = swiglu_grads[0];
    dx2 = swiglu_grads[1];
    x4 = swiglu_grads[2];
    TORCH_INTERNAL_ASSERT_SHAPE(dx2, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(dx2, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(x4, B, H);
    x1.reset();
    x2.reset();
    dx4.reset();

    at::Tensor dw3;

    dw3 = torch::mm(dx5.transpose(-2, -1), x4);
    x4.reset();
    dx5.reset();

    at::Tensor dx = bi_gemm_sum(dx1, dx2, w1, w2);

    // backward of linear1 + linear2 - packed
    at::Tensor dw1, dw2;
    dw1 = torch::mm(dx1.transpose(-2, -1), x);
    dw2 = torch::mm(dx2.transpose(-2, -1), x);

    return {dx, dw1, dw2, dw3};
}

at::Tensor backward_gradInput(at::Tensor& dx5,
                                      const at::Tensor& x,
                                      const at::Tensor& w1,
                                      const at::Tensor& w2,
                                      const at::Tensor& w3,
                                      at::Tensor& x1,
                                      at::Tensor& x2) {

    int64_t B = x.size(0);
    int64_t H = x2.size(1);
    int64_t I = x.size(1);
    int64_t O = dx5.size(1);

    TORCH_INTERNAL_ASSERT_SHAPE(x1, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(x2, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(dx5, B, O);
    TORCH_INTERNAL_ASSERT_SHAPE(w1, H, I);
    TORCH_INTERNAL_ASSERT_SHAPE(w1, H, I);
    TORCH_INTERNAL_ASSERT_SHAPE(w3, O, H);

    // Compute BW
    at::Tensor dx1, dx2, x4;
    TORCH_INTERNAL_ASSERT(dx5.size(1) == w3.size(0));
    auto dx4 = torch::mm(dx5, w3);
    std::vector<at::Tensor> swiglu_grads = swiglu_bw_fused(x1, x2, dx4);
    dx1 = swiglu_grads[0];
    dx2 = swiglu_grads[1];
    x4 = swiglu_grads[2];
    TORCH_INTERNAL_ASSERT_SHAPE(dx1, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(dx2, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(x4, B, H);
    x1.reset();
    x2.reset();
    dx4.reset();

    x4.reset();
    dx5.reset();

    at::Tensor dx = bi_gemm_sum(dx1, dx2, w1, w2);

    dx1.reset();
    dx2.reset();

    return dx;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dual_gemm_lhs_silu_and_mul", &dual_gemm_lhs_silu_and_mul, "Dual GEMM for SwiGLU");
  m.def("bi_gemm_sum", &bi_gemm_sum, "Bi GEMM for SwiGLU backward ");
  m.def("swiglu_bw_fused", &swiglu_bw_fused, "Dual GEMM for SwiGLU");
  m.def("forward", &forward, "SwiGLU MLP forward function");
  m.def("backward_full", &backward_full, "SwiGLU MLP backward function");
  m.def("backward_gradInput", &backward_gradInput, "SwiGLU MLP backward function for gradInput only");
}

}