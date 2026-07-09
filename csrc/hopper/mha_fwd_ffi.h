/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "flash_ffi_common.h"

// Multi-head attention forward pass, tvm-ffi calling convention.
// Parameter order matches the jax-tvm-ffi arg_spec: outputs first, then inputs,
// then scalar attributes. Missing optional tensors are 0-d tensors.
void mha_fwd_ffi_impl(
    // Outputs (pre-allocated by XLA):
    ffi::TensorArg out,           // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    ffi::TensorArg softmax_lse,   // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
    ffi::TensorArg out_accum,
    ffi::TensorArg softmax_lse_accum,
    ffi::TensorArg scheduler_metadata,

    // Inputs:
    ffi::TensorArg q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    ffi::TensorArg k,   // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
    ffi::TensorArg v,   // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
    ffi::TensorArg k_new,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
    ffi::TensorArg v_new,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
    ffi::TensorArg q_v,    // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    ffi::TensorArg cu_seqlens_q,      // b+1, int32
    ffi::TensorArg cu_seqlens_k,      // b+1, int32
    ffi::TensorArg cu_seqlens_k_new,  // b+1, int32
    ffi::TensorArg page_table,        // (b_k, max_num_pages_per_seq), int32
    ffi::TensorArg kv_batch_idx,      // b, int32. indices to index into the KV cache
    ffi::TensorArg leftpad_k,         // b, int32
    ffi::TensorArg rotary_cos,        // seqlen_ro x (rotary_dim / 2)
    ffi::TensorArg rotary_sin,        // seqlen_ro x (rotary_dim / 2)
    ffi::TensorArg seqlens_rotary,    // b, int32
    ffi::TensorArg q_descale,         // (b, h_k), not (b, h), f32
    ffi::TensorArg k_descale,         // (b, h_k), f32
    ffi::TensorArg v_descale,         // (b, h_k), f32

    // Attributes:
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t attention_chunk,
    double softcap,
    bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    int64_t num_splits,
    bool pack_gqa,
    int64_t sm_margin
    );
