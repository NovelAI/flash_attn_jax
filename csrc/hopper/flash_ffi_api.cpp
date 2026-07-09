/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include "flash_ffi_common.h"
#include "mha_fwd_ffi.h"

namespace {

namespace tffi = tvm::ffi;

// tvm-ffi entry point. Parameter order is rets, args, attrs — it must match the
// arg_spec used at registration time in flash_hlo.py. TensorView converts
// implicitly to the ffi::TensorArg shim the handler is written against.
void Fa3Fwd(
    // rets
    tffi::TensorView out, tffi::TensorView softmax_lse,
    tffi::TensorView out_accum, tffi::TensorView softmax_lse_accum,
    tffi::TensorView scheduler_metadata,
    // args
    tffi::TensorView q, tffi::TensorView k, tffi::TensorView v,
    tffi::TensorView k_new, tffi::TensorView v_new, tffi::TensorView q_v,
    tffi::TensorView cu_seqlens_q, tffi::TensorView cu_seqlens_k, tffi::TensorView cu_seqlens_k_new,
    tffi::TensorView page_table, tffi::TensorView kv_batch_idx, tffi::TensorView leftpad_k,
    tffi::TensorView rotary_cos, tffi::TensorView rotary_sin, tffi::TensorView seqlens_rotary,
    tffi::TensorView q_descale, tffi::TensorView k_descale, tffi::TensorView v_descale,
    // attrs
    int64_t max_seqlen_q, int64_t max_seqlen_k, double softmax_scale,
    bool is_causal, int64_t window_size_left, int64_t window_size_right,
    int64_t attention_chunk, double softcap, bool is_rotary_interleaved,
    int64_t num_splits, bool pack_gqa, int64_t sm_margin) {
  mha_fwd_ffi_impl(
      out, softmax_lse, out_accum, softmax_lse_accum, scheduler_metadata,
      q, k, v, k_new, v_new, q_v,
      cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new,
      page_table, kv_batch_idx, leftpad_k,
      rotary_cos, rotary_sin, seqlens_rotary,
      q_descale, k_descale, v_descale,
      max_seqlen_q, max_seqlen_k, softmax_scale,
      is_causal, window_size_left, window_size_right,
      attention_chunk, softcap, is_rotary_interleaved,
      num_splits, pack_gqa, sm_margin);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fa3_fwd, Fa3Fwd);
// Future work: add bwd
