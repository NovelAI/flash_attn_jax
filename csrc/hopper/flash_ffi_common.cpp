/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <stddef.h>
#include <vector>
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

    // Helper to compute row-major strides for a buffer
    // Returns strides in elements (not bytes) as a vector
    auto compute_strides = [](const ffi::AnyBuffer& buf) -> std::vector<int64_t> {
        int ndim = buf.dimensions().size();
        std::vector<int64_t> strides(ndim);
        int64_t s = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            strides[i] = s;
            s *= buf.dimensions()[i];
        }
        return strides;
    };

    // Support both 3D (varlen) and 4D (non-varlen) tensors
    // For 3D: shape is [total, heads, dim]
    // For 4D: shape is [batch, seq, heads, dim]
    // Use negative indexing like PyTorch: -3 = row, -2 = head, -1 = dim
    int q_ndim = q.dimensions().size();
    int k_ndim = k.dimensions().size();
    int v_ndim = v.dimensions().size();
    int out_ndim = out.dimensions().size();

    if ((q_ndim != 3 && q_ndim != 4) || (k_ndim != 3 && k_ndim != 4) ||
        (v_ndim != 3 && v_ndim != 4) || (out_ndim != 3 && out_ndim != 4)) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "All input and output buffers must be rank 3 or 4 tensors");
    }

    auto q_strides = compute_strides(q);
    auto k_strides = compute_strides(k);
    auto v_strides = compute_strides(v);
    auto out_strides = compute_strides(out);

    // All strides are in elements, not bytes.
    // Use negative indexing: -3 = row (seq), -2 = head, -1 = dim
    params.q_row_stride = q_strides[q_ndim - 3];
    params.k_row_stride = k_strides[k_ndim - 3];
    params.v_row_stride = v_strides[v_ndim - 3];
    params.q_head_stride = q_strides[q_ndim - 2];
    params.k_head_stride = k_strides[k_ndim - 2];
    params.v_head_stride = v_strides[v_ndim - 2];
    params.v_dim_stride = v_strides[v_ndim - 1];
    params.o_ptr = out.untyped_data();
    params.o_row_stride = out_strides[out_ndim - 3];
    params.o_head_stride = out_strides[out_ndim - 2];

    // Batch strides only make sense for non-varlen (4D) tensors
    if (cu_seqlens_q_d == nullptr && q_ndim == 4) {
        params.q_batch_stride = q_strides[0];
        params.o_batch_stride = out_strides[0];
    }
    if (cu_seqlens_k_d == nullptr && k_ndim == 4) {
        params.k_batch_stride = k_strides[0];
        params.v_batch_stride = v_strides[0];
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

    // sm90 (wgmma) kernels run only on compute capability 9.0; all other supported
    // archs (Ampere/Ada/consumer Blackwell) use the sm80-style kernels.
    FFI_CHECK(params.arch == 80 || params.arch == 86 || params.arch == 89
              || params.arch == 90 || params.arch == 120 || params.arch == 121)
        << "FA3 does not support compute capability " << major << "." << minor;
    #ifdef FLASHATTENTION_DISABLE_SM8x
    FFI_CHECK(params.arch == 90)
        << "This FA3 build only includes SM90 kernels; rebuild with 80/86/89/120 in FLASH_ATTN_CUDA_ARCHS to support compute capability " << major << "." << minor;
    #endif
    #ifdef FLASHATTENTION_DISABLE_SM90
    FFI_CHECK(params.arch != 90)
        << "This FA3 build does not include SM90 kernels; rebuild with 90a in FLASH_ATTN_CUDA_ARCHS";
    #endif

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
    #endif

    return ffi::Error();  // Success
}
