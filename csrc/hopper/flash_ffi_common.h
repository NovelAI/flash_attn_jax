/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <string>
#include <sstream>

#include "xla/ffi/api/ffi.h"
#include "hopper/gpu/flash.h"

namespace ffi = xla::ffi;

// Error checking helper class for FFI
class CheckHelper {
public:
  explicit CheckHelper(std::string expr) : expr_(expr) {}

  template <typename T> inline CheckHelper &operator<<(const T &value) {
    stream_ << value;
    return *this;
  }

  inline CheckHelper &operator<<(ffi::ErrorCode errc) {
    errc_ = errc;
    return *this;
  }

  inline operator ffi::Error() {
    std::ostringstream full_message;
    full_message << "Check failed: " << expr_;
    std::string additional = stream_.str();
    if (!additional.empty()) {
      full_message << "; " << additional;
    }
    return ffi::Error(errc_, full_message.str());
  }

private:
  ffi::ErrorCode errc_ = ffi::ErrorCode::kUnknown;
  std::string expr_;
  std::ostringstream stream_;
};

// FFI check macros - adapted from Flash Attention 2
#define FFI_CHECK(expr)                                                                            \
  static_assert(!std::is_same_v<decltype(expr), cudaError_t>,                                      \
                "Use FFI_CUDA_CHECK for CUDA error codes, not FFI_CHECK.");                        \
  if (!(expr))                                                                                     \
  return CheckHelper(#expr)

#define FFI_CUDA_CHECK(expr)                                                                       \
  static_assert(std::is_same_v<decltype(expr), cudaError_t>,                                       \
                "Expect cudaError_t for FFI_CUDA_CHECK.");                                         \
  if (cudaError_t _cuda_check = (expr); _cuda_check != cudaSuccess)                                \
  return CheckHelper(std::string(#expr)) << " CUDA Error: " << cudaGetErrorString(_cuda_check)

#define FFI_RET_CHECK(expr)                                                                        \
  if (auto _error = (expr); !_error.success())                                                     \
  return _error

// Forward declarations

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
                      const float softcap=0.f,
                      const int sm_margin=0);
