/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>
#include <string>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>

// Thin compatibility layer over tvm::ffi::TensorView exposing the buffer API the
// handlers were originally written against (XLA FFI style), so the handler bodies
// stay close to upstream. Missing optional tensors are passed as 0-d tensors and
// detected via dimensions().size() == 0.
namespace flash_ffi {

enum class DataType { INVALID, F16, BF16, F32, S32, F8E4M3FN, OTHER };
inline constexpr DataType F16 = DataType::F16;
inline constexpr DataType BF16 = DataType::BF16;
inline constexpr DataType F32 = DataType::F32;
inline constexpr DataType S32 = DataType::S32;
inline constexpr DataType F8E4M3FN = DataType::F8E4M3FN;

inline DataType from_dltype(DLDataType t) {
    if (t.lanes != 1) return DataType::OTHER;
    if (t.code == kDLFloat && t.bits == 16) return DataType::F16;
    if (t.code == kDLBfloat && t.bits == 16) return DataType::BF16;
    if (t.code == kDLFloat && t.bits == 32) return DataType::F32;
    if (t.code == kDLInt && t.bits == 32) return DataType::S32;
    if (t.code == kDLFloat8_e4m3fn && t.bits == 8) return DataType::F8E4M3FN;
    return DataType::OTHER;
}

class TensorArg {
public:
    TensorArg(tvm::ffi::TensorView t) : t_(t) {}  // NOLINT(*)

    tvm::ffi::ShapeView dimensions() const { return t_.shape(); }
    DataType element_type() const { return from_dltype(t_.dtype()); }
    void* untyped_data() const { return t_.data_ptr(); }
    int* typed_data() const { return static_cast<int*>(t_.data_ptr()); }
    int64_t element_count() const { return t_.numel(); }
    size_t size_bytes() const {
        return (static_cast<size_t>(t_.numel()) * t_.dtype().bits * t_.dtype().lanes + 7) / 8;
    }
    DLDevice device() const { return t_.device(); }

private:
    tvm::ffi::TensorView t_;
};

}  // namespace flash_ffi

namespace ffi = flash_ffi;

// Check macros: throw tvm::ffi::Error, which the jax-tvm-ffi bridge surfaces as a
// JaxRuntimeError with the streamed message intact.
#define FFI_CHECK(expr)                                                                            \
  static_assert(!std::is_same_v<decltype(expr), cudaError_t>,                                      \
                "Use FFI_CUDA_CHECK for CUDA error codes, not FFI_CHECK.");                        \
  if (TVM_FFI_PREDICT_FALSE(!(expr)))                                                              \
  TVM_FFI_THROW(ValueError) << "Check failed: (" #expr ") "

#define FFI_CUDA_CHECK(expr)                                                                       \
  static_assert(std::is_same_v<decltype(expr), cudaError_t>,                                       \
                "Expect cudaError_t for FFI_CUDA_CHECK.");                                         \
  if (cudaError_t _cuda_check = (expr); _cuda_check != cudaSuccess)                                \
  TVM_FFI_THROW(RuntimeError) << #expr << " CUDA Error: " << cudaGetErrorString(_cuda_check)

inline cudaStream_t flash_ffi_get_stream(DLDevice device) {
    return static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

// True when FLASH_ATTN_JAX_DEBUG=1 is set in the environment.
// Defined once per shared library (flash_common.cpp / flash_ffi_common.cpp).
bool flash_debug();
