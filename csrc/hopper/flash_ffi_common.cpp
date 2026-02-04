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

ffi::Error set_params_fprop_ffi(Flash_fwd_params &params,
                        cudaStream_t stream,
                        int device_ordinal,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const ffi::AnyBuffer& q,
                      const ffi::AnyBuffer& k,
                      const ffi::AnyBuffer& v,
                      ffi::AnyBuffer& out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      int attention_chunk,
                      const float softcap,
                      const int sm_margin) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.element_type() == ffi::DataType::BF16;
    params.is_e4m3 = q.element_type() == ffi::DataType::F8E4M3FN;

    // Set the pointers and strides.
    params.q_ptr = q.untyped_data();
    params.k_ptr = k.untyped_data();
    params.v_ptr = v.untyped_data();

    if (q.dimensions().size() != 4 || k.dimensions().size() != 4 ||
        v.dimensions().size() != 4 || out.dimensions().size() != 4) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "All input and output buffers must be rank 4 tensors");
    }
    auto q_shape = cute::make_shape(q.dimensions()[0], q.dimensions()[1],
                                   q.dimensions()[2], q.dimensions()[3]);
    auto q_strides = cute::compact_row_major(q_shape);
    auto k_shape = cute::make_shape(k.dimensions()[0], k.dimensions()[1],
                                   k.dimensions()[2], k.dimensions()[3]);
    auto k_strides = cute::compact_row_major(k_shape);
    auto v_shape = cute::make_shape(v.dimensions()[0], v.dimensions()[1],
                                   v.dimensions()[2], v.dimensions()[3]);
    auto v_strides = cute::compact_row_major(v_shape);
    auto out_shape = cute::make_shape(out.dimensions()[0], out.dimensions()[1],
                                     out.dimensions()[2], out.dimensions()[3]);
    auto out_strides = cute::compact_row_major(out_shape);
    // All stride are in elements, not bytes.
    params.q_row_stride = cute::get<1>(q_strides);
    params.k_row_stride = cute::get<1>(k_strides);
    params.v_row_stride = cute::get<1>(v_strides);
    params.q_head_stride = cute::get<2>(q_strides);
    params.k_head_stride = cute::get<2>(k_strides);
    params.v_head_stride = cute::get<2>(v_strides);
    params.v_dim_stride = cute::get<3>(v_strides);
    params.o_ptr = out.untyped_data();
    params.o_row_stride = cute::get<1>(out_strides);
    params.o_head_stride = cute::get<2>(out_strides);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = cute::get<0>(q_strides);
        params.o_batch_stride = cute::get<0>(out_strides);
    }
    if (cu_seqlens_k_d == nullptr) {
        params.k_batch_stride = cute::get<0>(k_strides);
        params.v_batch_stride = cute::get<0>(v_strides);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_q = static_cast<int *>(seqused_q);
    params.seqused_k = static_cast<int *>(seqused_k);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    // Set this to probability of keeping an element to simplify things.
    // params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    // params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    // params.rp_dropout = 1.f / params.p_dropout;
    // TORCH_CHECK(p_dropout < 1.f);
    // #ifdef FLASHATTENTION_DISABLE_DROPOUT
    //     TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    // #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) && !params.is_causal;

    // TODO: check this
    if (window_size_left < 0) { window_size_left = seqlen_k - 1; }
    if (window_size_right < 0) { window_size_right = seqlen_q - 1; }
    if (attention_chunk > 0) {
        window_size_left = std::min(window_size_left, attention_chunk - 1);
        window_size_right = std::min(window_size_right, attention_chunk - 1);
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.attention_chunk = attention_chunk;

    int major, minor;
    FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal));
    FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal));
    params.arch = major * 10 + minor;
    FFI_CUDA_CHECK(cudaDeviceGetAttribute(&params.num_sm, cudaDevAttrMultiProcessorCount, device_ordinal));
    // params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
    // params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
    #endif
}
