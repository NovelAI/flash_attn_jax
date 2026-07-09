#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <stddef.h>

#include "check.h"

// tvm-ffi calling convention: args, rets, attrs (matches the arg_spec in flash_hlo.py).
void mha_bwd_impl(
                  ffi::TensorArg dout, ffi::TensorArg q, ffi::TensorArg k,
                  ffi::TensorArg v, ffi::TensorArg o,
                  ffi::TensorArg lse, ffi::TensorArg dq,
                  ffi::TensorArg dk, ffi::TensorArg dv,
                  ffi::TensorArg softmax_d,  // batch_size x num_heads x seqlen_q_rounded
                  ffi::TensorArg dq_accum,   // batch_size x seqlen_q_rounded x num_heads x head_size_rounded
                  double softmax_scale, bool is_causal,
                  int64_t window_size_left, int64_t window_size_right, bool deterministic);

void
mha_varlen_bwd_impl(
    ffi::TensorArg dout,  // total_q x num_heads, x head_size
    ffi::TensorArg q,     // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    ffi::TensorArg k,     // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::TensorArg v,     // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::TensorArg o,     // total_q x num_heads x head_size,
    ffi::TensorArg lse,   // b x h x s   softmax logsumexp
    ffi::TensorArg cu_seqlens_q,  // b+1
    ffi::TensorArg cu_seqlens_k,  // b+1
    ffi::TensorArg dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    ffi::TensorArg dk,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::TensorArg dv,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::TensorArg softmax_d,  // batch_size x num_heads x seqlen_q_rounded
    ffi::TensorArg dq_accum,   // (total_q + 128 * batch_size) x num_heads x head_size_rounded
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,          // max sequence length to choose the kernel
    double softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    bool deterministic);
