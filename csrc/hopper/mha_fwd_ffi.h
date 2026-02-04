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
    cudaStream_t stream,
    int32_t device,
    ffi::AnyBuffer q,
    ffi::AnyBuffer k,
    ffi::AnyBuffer v,
    ffi::Result<ffi::AnyBuffer> o,
    ffi::ResultBuffer<ffi::F32> lse,
    ffi::ResultBuffer<ffi::F32> oaccum,
    ffi::ResultBuffer<ffi::F32> lseaccum,
    float softmax_scale,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    float softcap,
    int64_t num_splits);
