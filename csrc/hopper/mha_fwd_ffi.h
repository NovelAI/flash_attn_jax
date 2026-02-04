/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Forward declaration of mha_fwd_ffi_impl
// Multi-head attention forward pass FFI implementation
ffi::Error mha_fwd_ffi_impl(
    cudaStream_t stream, int32_t device_ordinal,
    ffi::AnyBuffer q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    ffi::AnyBuffer k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
    ffi::AnyBuffer v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.

    // Optional arrays:
    ffi::AnyBuffer k_new,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
    ffi::AnyBuffer v_new,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
    ffi::AnyBuffer q_v,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    ffi::Buffer<ffi::S32> cu_seqlens_q,  // b+1
    ffi::Buffer<ffi::S32> cu_seqlens_k,  // b+1
    ffi::Buffer<ffi::S32> cu_seqlens_k_new,  // b+1
    // no seqused.
    // std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
    // std::optional<at::Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
    ffi::Buffer<ffi::S32> page_table, // (b_k, max_num_pages_per_seq)
    ffi::Buffer<ffi::S32> kv_batch_idx, // b. indices to index into the KV cache
    ffi::Buffer<ffi::S32> leftpad_k, // b
    ffi::AnyBuffer rotary_cos, // seqlen_ro x (rotary_dim / 2)
    ffi::AnyBuffer rotary_sin, // seqlen_ro x (rotary_dim / 2)
    ffi::Buffer<ffi::S32> seqlens_rotary, // b
    ffi::Buffer<ffi::F32> q_descale,  // (b, h_k), not (b, h)
    ffi::Buffer<ffi::F32> k_descale,  // (b, h_k)
    ffi::Buffer<ffi::F32> v_descale,  // (b, h_k)

    // Return arrays: out, softmax_lse, out_accum, softmax_lse_accum
    ffi::Result<ffi::AnyBuffer> out, // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    ffi::ResultBuffer<ffi::F32> softmax_lse, // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
    ffi::ResultBuffer<ffi::F32> out_accum,
    ffi::ResultBuffer<ffi::F32> softmax_lse_accum,
    ffi::ResultBuffer<ffi::S32> scheduler_metadata,  // (b + 1)

    std::optional<int64_t> max_seqlen_q_,
    std::optional<int64_t> max_seqlen_k_,
    std::optional<double> softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t attention_chunk,
    double softcap,
    bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    int64_t num_splits,
    std::optional<bool> pack_gqa_,
    int64_t sm_margin
    );