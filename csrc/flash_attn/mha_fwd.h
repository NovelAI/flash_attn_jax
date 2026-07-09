#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <stddef.h>

#include "check.h"

// tvm-ffi calling convention: args, rets, attrs (matches the arg_spec in flash_hlo.py).
// Missing optional tensors are 0-d tensors.
void mha_fwd_impl(
    ffi::TensorArg q,
    ffi::TensorArg k,
    ffi::TensorArg v,
    ffi::TensorArg o,
    ffi::TensorArg lse,
    ffi::TensorArg oaccum,
    ffi::TensorArg lseaccum,
    double softmax_scale,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right);

void
mha_varlen_fwd_impl(
    ffi::TensorArg q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    ffi::TensorArg k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::TensorArg v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::TensorArg cu_seqlens_q,  // b+1
    ffi::TensorArg cu_seqlens_k,  // b+1
    ffi::TensorArg seqused_k, // b (0-d if absent). If given, only this many elements of each batch element's keys are used.
    ffi::TensorArg out, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::TensorArg lse, // batch_size x num_heads x max_seqlen_q
    ffi::TensorArg oaccum,
    ffi::TensorArg lseaccum,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right);
