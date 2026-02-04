/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <stddef.h>

#include "cuda_check.h"
#include "flash_ffi_common.h"
#include "hopper/gpu/flash.h"
#include "hopper/gpu/static_switch.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"

#include "mha_fwd_ffi.h"

namespace ffi = xla::ffi;

// Forward declarations of kernel instantiation templates
// These are explicitly instantiated in gpu/instantiations/*.cu files
template <int Arch, typename Element, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA,
          bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

template <typename Element, int kHeadDim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

// Forward declaration for combine kernel
template <typename Element, typename ElementAccum, int kHeadDim>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);

ffi::Error mha_fwd_ffi_impl(cudaStream_t stream, int32_t device, ffi::AnyBuffer q, ffi::AnyBuffer k,
                            ffi::AnyBuffer v, ffi::Result<ffi::AnyBuffer> o,
                            ffi::ResultBuffer<ffi::F32> lse, ffi::ResultBuffer<ffi::F32> oaccum,
                            ffi::ResultBuffer<ffi::F32> lseaccum, float softmax_scale,
                            bool is_causal, int64_t window_size_left, int64_t window_size_right,
                            float softcap, int64_t num_splits) {

  // Get device compute capability
  int major, minor;
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

  bool is_sm8x = major == 8 && minor >= 0;
  bool is_sm90 = major == 9 && minor == 0;
  FFI_CHECK(is_sm90 || is_sm8x)
      << ffi::ErrorCode::kUnimplemented
      << "FlashAttention only supports Ampere (SM80+) and Hopper (SM90) GPUs";

  // Check dtype
  ffi::DataType dtype = q.element_type();
  FFI_CHECK(dtype == k.element_type() && dtype == v.element_type() && dtype == o->element_type())
      << ffi::ErrorCode::kInvalidArgument
      << "query, key, value, and output must have the same dtype";
  FFI_CHECK(dtype == ffi::DataType::F16 || dtype == ffi::DataType::BF16)
      << ffi::ErrorCode::kInvalidArgument
      << "FlashAttention only supports fp16 and bf16 data types";

  if (dtype == ffi::DataType::BF16) {
    FFI_CHECK(is_sm90 || is_sm8x) << ffi::ErrorCode::kInvalidArgument
                                  << "bfloat16 is only supported on Ampere GPUs or newer";
  }

  // Extract tensor shapes
  const int batch_size = q.dimensions()[0];
  int seqlen_q = q.dimensions()[1];
  int num_heads = q.dimensions()[2];
  const int head_size = q.dimensions()[3];
  const int seqlen_k = k.dimensions()[1];
  const int num_heads_k = k.dimensions()[2];

  // Validate shapes and parameters
  FFI_CHECK(batch_size > 0) << ffi::ErrorCode::kInvalidArgument << "batch size must be positive";
  FFI_CHECK(head_size > 0) << ffi::ErrorCode::kInvalidArgument << "head_size must be positive";
  FFI_CHECK(head_size <= 256)
      << ffi::ErrorCode::kInvalidArgument
      << "FlashAttention forward only supports head dimension at most 256, got " << head_size;
  FFI_CHECK(head_size % 8 == 0) << ffi::ErrorCode::kInvalidArgument
                                << "head_size must be divisible by 8, got " << head_size;
  FFI_CHECK(num_heads % num_heads_k == 0)
      << ffi::ErrorCode::kInvalidArgument
      << "Number of heads in key/value must divide number of heads in query; "
      << "got num_heads=" << num_heads << ", num_heads_k=" << num_heads_k;

  // Check output tensor
  FFI_CHECK(o->dimensions()[0] == batch_size && o->dimensions()[1] == seqlen_q &&
            o->dimensions()[2] == num_heads && o->dimensions()[3] == head_size)
      << ffi::ErrorCode::kInvalidArgument << "Output tensor shape mismatch";

  // Preprocessing window sizes (matching PyTorch logic before set_params_fprop)
  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_q - 1) {
    window_size_right = -1;
  }

  // Special case: if seqlen_q == 1, causal attention is equivalent to non-causal
  // (unless we have special head_size requirements for paged KV)
  if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1) {
    is_causal = false;
  }

  if (is_causal) {
    window_size_right = 0;
  }

  // Setup forward parameters
  Flash_fwd_params params;
  FFI_RET_CHECK(set_params_fprop_ffi(
      params, dtype, batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, head_size,
      q.untyped_data(), k.untyped_data(), v.untyped_data(), o->untyped_data(), lse->untyped_data(),
      oaccum->untyped_data(), lseaccum->untyped_data(), softmax_scale, softcap,
      static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      static_cast<int>(num_splits)));

  // Set device architecture information for kernel dispatch
  params.arch = major * 10 + minor;  // 80, 86, 89, or 90

  // Get number of SMs for split-KV heuristic
  int num_sms = 0;
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  params.num_sm = num_sms;

  // Zero out output accumulators if we're using split-KV
  if (num_splits > 1) {
    FFI_CUDA_CHECK(cudaMemsetAsync(oaccum->untyped_data(), 0, oaccum->size_bytes(), stream));
    FFI_CUDA_CHECK(cudaMemsetAsync(lseaccum->untyped_data(), 0, lseaccum->size_bytes(), stream));
  }

  // Dispatch to appropriate kernel based on architecture, head_dim, and dtype
  // We use static switches similar to the PyTorch bindings
  try {
    ARCH_SWITCH(params.arch, Arch, [&] {
      SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
        PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
          PACKGQA_SWITCH(params.pack_gqa, PackGQA, [&] {
            SOFTCAP_SWITCH(params.softcap > 0.0f, Has_softcap, [&] {
              constexpr bool PackGQA_forced = PackGQA || Split;
              if (!params.is_e4m3) {
                if (params.is_bf16) {
                  if (params.d <= 64) {
                    return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  } else if (params.d <= 96) {
                    return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  } else if (params.d <= 128) {
                    return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  } else if (params.d <= 192) {
                    return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  } else if (params.d <= 256) {
                    return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  }
                } else {
                  if (params.d <= 64) {
                    return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  } else if (params.d <= 96) {
                    return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  } else if (params.d <= 128) {
                    return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  } else if (params.d <= 192) {
                    return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  } else if (params.d <= 256) {
                    return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA,
                                        Has_softcap, PackGQA_forced>(params, stream);
                  }
                }
              }
            });
          });
        });
      });
    });
  } catch (const CudaException &e) {
    return ffi::Error(ffi::ErrorCode::kInternal, e.what());
  }

  FFI_CUDA_CHECK(cudaGetLastError());
  FFI_CUDA_CHECK(cudaStreamSynchronize(stream)) << "after run_mha_fwd_";

  // If using split-KV, combine the partial results
  if (num_splits > 1) {
    // Set output dtype for combine kernel
    if (dtype == ffi::DataType::BF16) {
      params.is_bf16 = true;
    }

    // Dispatch combine kernel based on dtype and head dimension
    try {
      if (params.is_bf16) {
        if (params.dv <= 64) {
          run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, true /*enable_pdl*/);
        } else {
          run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, true /*enable_pdl*/);
        }
      } else {
        if (params.dv <= 64) {
          run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream, true /*enable_pdl*/);
        } else {
          run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, true /*enable_pdl*/);
        }
      }
    } catch (const CudaException &e) {
      return ffi::Error(ffi::ErrorCode::kInternal, e.what());
    }

    FFI_CUDA_CHECK(cudaGetLastError());
  }

  FFI_CUDA_CHECK(cudaStreamSynchronize(stream)) << "after combine kernel";

  return ffi::Error(); // Success
}
