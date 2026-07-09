/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include "check.h"
#include "mha_fwd.h"
#include "mha_bwd.h"

namespace {

namespace tffi = tvm::ffi;

// tvm-ffi entry points. Parameter order is args, rets, attrs — it must match the
// arg_spec used at registration time in flash_hlo.py. TensorView converts
// implicitly to the ffi::TensorArg shim the handlers are written against.

void Fa2Fwd(
    tffi::TensorView q, tffi::TensorView k, tffi::TensorView v,
    tffi::TensorView o, tffi::TensorView lse, tffi::TensorView oaccum, tffi::TensorView lseaccum,
    double softmax_scale, bool is_causal,
    int64_t window_size_left, int64_t window_size_right) {
  mha_fwd_impl(q, k, v, o, lse, oaccum, lseaccum,
               softmax_scale, is_causal, window_size_left, window_size_right);
}

void Fa2Bwd(
    tffi::TensorView dout, tffi::TensorView q, tffi::TensorView k, tffi::TensorView v,
    tffi::TensorView o, tffi::TensorView lse,
    tffi::TensorView dq, tffi::TensorView dk, tffi::TensorView dv,
    tffi::TensorView softmax_d, tffi::TensorView dq_accum,
    double softmax_scale, bool is_causal,
    int64_t window_size_left, int64_t window_size_right, bool deterministic) {
  mha_bwd_impl(dout, q, k, v, o, lse, dq, dk, dv, softmax_d, dq_accum,
               softmax_scale, is_causal, window_size_left, window_size_right, deterministic);
}

void Fa2VarlenFwd(
    tffi::TensorView q, tffi::TensorView k, tffi::TensorView v,
    tffi::TensorView cu_seqlens_q, tffi::TensorView cu_seqlens_k, tffi::TensorView seqused_k,
    tffi::TensorView out, tffi::TensorView lse, tffi::TensorView oaccum, tffi::TensorView lseaccum,
    int64_t max_seqlen_q, int64_t max_seqlen_k, double softmax_scale,
    bool zero_tensors, bool is_causal,
    int64_t window_size_left, int64_t window_size_right) {
  mha_varlen_fwd_impl(q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_k,
                      out, lse, oaccum, lseaccum,
                      max_seqlen_q, max_seqlen_k, softmax_scale,
                      zero_tensors, is_causal, window_size_left, window_size_right);
}

void Fa2VarlenBwd(
    tffi::TensorView dout, tffi::TensorView q, tffi::TensorView k, tffi::TensorView v,
    tffi::TensorView o, tffi::TensorView lse,
    tffi::TensorView cu_seqlens_q, tffi::TensorView cu_seqlens_k,
    tffi::TensorView dq, tffi::TensorView dk, tffi::TensorView dv,
    tffi::TensorView softmax_d, tffi::TensorView dq_accum,
    int64_t max_seqlen_q, int64_t max_seqlen_k, double softmax_scale,
    bool zero_tensors, bool is_causal,
    int64_t window_size_left, int64_t window_size_right, bool deterministic) {
  mha_varlen_bwd_impl(dout, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k,
                      dq, dk, dv, softmax_d, dq_accum,
                      max_seqlen_q, max_seqlen_k, softmax_scale,
                      zero_tensors, is_causal, window_size_left, window_size_right, deterministic);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fa2_fwd, Fa2Fwd);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fa2_bwd, Fa2Bwd);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fa2_varlen_fwd, Fa2VarlenFwd);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fa2_varlen_bwd, Fa2VarlenBwd);
