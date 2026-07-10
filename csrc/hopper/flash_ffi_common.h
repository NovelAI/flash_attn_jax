/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>
#include <string>

#include "flash_ffi_shim.h"
#include "hopper/gpu/flash.h"

// Forward declarations

void set_params_fprop_ffi(Flash_fwd_params &params,
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
                      const ffi::TensorArg& q,
                      const ffi::TensorArg& k,
                      const ffi::TensorArg& v,
                      ffi::TensorArg& out,
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
