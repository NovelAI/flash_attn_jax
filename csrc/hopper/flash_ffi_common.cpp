/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <stddef.h>
#include <cute/layout.hpp>
#include <cuda_runtime_api.h>
#include <cmath>

#include "hopper/gpu/flash.h"
#include "flash_ffi_common.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error set_params_fprop_ffi(
    Flash_fwd_params &params,
    ffi::DataType element_type,
    // sizes
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t h,
    const size_t h_k,
    const size_t d,
    // device pointers
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *out_ptr,
    void *softmax_lse_ptr,
    void *oaccum_ptr,
    void *lseaccum_ptr,
    float softmax_scale,
    float softcap,
    int window_size_left,
    int window_size_right,
    int num_splits) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = element_type == ffi::DataType::BF16;
    params.is_fp32 = false;
    params.is_e4m3 = false;

    // Set the pointers
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = out_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;
    params.oaccum_ptr = oaccum_ptr;
    params.softmax_lseaccum_ptr = lseaccum_ptr;

    // Calculate strides using CUTLASS cute library
    // All strides are in elements, not bytes.
    // Layout: (batch, seqlen, num_heads, head_dim)
    auto qs = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));
    auto ks = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));
    auto vs = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));
    auto os = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));

    params.q_batch_stride = cute::get<0>(qs);
    params.k_batch_stride = cute::get<0>(ks);
    params.v_batch_stride = cute::get<0>(vs);
    params.o_batch_stride = cute::get<0>(os);

    params.q_row_stride = cute::get<1>(qs);    // stride along seqlen
    params.k_row_stride = cute::get<1>(ks);
    params.v_row_stride = cute::get<1>(vs);
    params.o_row_stride = cute::get<1>(os);

    params.q_head_stride = cute::get<2>(qs);   // stride along num_heads
    params.k_head_stride = cute::get<2>(ks);
    params.v_head_stride = cute::get<2>(vs);
    params.o_head_stride = cute::get<2>(os);
    params.v_dim_stride = 1;  // head_dim is contiguous

    // Set the dimensions
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;
    params.dv = d;  // For now, V head_dim == Q/K head_dim

    // For regular (non-varlen) mode
    params.total_q = b * seqlen_q;
    params.total_k = b * seqlen_k;
    params.b_k = b;  // batch_size_k (same as b for non-paged KV)

    // Round dimensions for kernel dispatch and memory alignment
    auto round_up_headdim = [](int head_size) -> int {
        if (head_size <= 64) { return 64; }
        if (head_size <= 96) { return 96; }
        if (head_size <= 128) { return 128; }
        if (head_size <= 192) { return 192; }
        if (head_size <= 256) { return 256; }
        return 256;
    };
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

    params.d_rounded = round_up_headdim(d);
    params.dv_rounded = (params.dv == d) ? params.d_rounded : round_up_headdim(params.dv);
    params.seqlen_q_rounded = round_multiple(seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Set the scaling factors
    params.scale_softmax = softmax_scale;
    // params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.softcap = softcap;

    // Window attention settings
    // Causal is the special case where window_size_right == 0 and window_size_left < 0
    // Note: attention_chunk is not yet supported in FFI, defaulting to 0
    int attention_chunk = 0;
    params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) && !params.is_causal;

    // Adjust window sizes (matching PyTorch logic)
    if (window_size_left < 0) {
        window_size_left = seqlen_k - 1;
    }
    if (window_size_right < 0) {
        window_size_right = seqlen_q - 1;
    }
    if (attention_chunk > 0) {
        window_size_left = std::min(window_size_left, attention_chunk - 1);
        window_size_right = std::min(window_size_right, attention_chunk - 1);
    }

    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.attention_chunk = attention_chunk;

    // Split-KV for long sequences
    params.num_splits = num_splits;

    // Initialize other fields to 0 or defaults
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.cu_seqlens_knew = nullptr;
    params.seqused_q = nullptr;
    params.seqused_k = nullptr;
    params.leftpad_k = nullptr;

    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.qv_ptr = nullptr;
    params.seqlen_knew = 0;
    params.total_knew = 0;

    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.rotary_dim = 0;
    params.seqlens_rotary = nullptr;

    params.kv_batch_idx = nullptr;
    params.page_table = nullptr;
    params.page_size = 1;
    params.num_pages = 0;
    params.pagedkv_tma = false;

    params.is_rotary_interleaved = false;
    params.pack_gqa = false;

    params.tile_count_semaphore = nullptr;
    params.num_m_blocks_ptr = nullptr;
    params.num_splits_dynamic_ptr = nullptr;
    params.varlen_batch_idx_ptr = nullptr;
    params.num_nheads_in_l2_ptr = nullptr;
    params.skip_scheduler_metadata_computation = false;
    params.varlen_sort_batches = false;
    params.head_swizzle = false;
    params.prepare_varlen_pdl = false;

    // Accumulator strides for split-KV
    // Layout: (num_splits, batch, num_heads, seqlen_q, head_dim) for oaccum
    // Layout: (num_splits, batch, num_heads, seqlen_q) for lseaccum
    if (num_splits > 1) {
        auto oaccum_strides = cute::compact_row_major(cute::make_shape(num_splits, b, h, seqlen_q, d));
        params.oaccum_split_stride = cute::get<0>(oaccum_strides);
        params.oaccum_batch_stride = cute::get<1>(oaccum_strides);
        params.oaccum_head_stride = cute::get<2>(oaccum_strides);
        params.oaccum_row_stride = cute::get<3>(oaccum_strides);

        auto lseaccum_strides = cute::compact_row_major(cute::make_shape(num_splits, b, h, seqlen_q));
        params.lseaccum_split_stride = cute::get<0>(lseaccum_strides);
        params.lseaccum_batch_stride = cute::get<1>(lseaccum_strides);
        params.lseaccum_head_stride = cute::get<2>(lseaccum_strides);
    } else {
        params.oaccum_split_stride = 0;
        params.oaccum_batch_stride = 0;
        params.oaccum_row_stride = 0;
        params.oaccum_head_stride = 0;

        params.lseaccum_split_stride = 0;
        params.lseaccum_batch_stride = 0;
        params.lseaccum_head_stride = 0;
    }

    return ffi::Error();  // Success
}
