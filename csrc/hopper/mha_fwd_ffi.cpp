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
#include "hopper/gpu/tile_size.h"
#include "hopper/gpu/heuristics.h"
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

inline int round_up_headdim(int head_size) {
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    if (head_size <= 64) { return 64; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    if (head_size <= 96) { return 96; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    if (head_size <= 128) { return 128; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    if (head_size <= 192) { return 192; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    if (head_size <= 256) { return 256; }
    #endif
    return 256;
}

inline int round_up_headdimv(int head_size) {
    if (head_size <= 64) { return 64; }
    if (head_size <= 96) { return 96; }
    if (head_size <= 128) { return 128; }
    if (head_size <= 192) { return 192; }
    if (head_size <= 256) { return 256; }
    return 512;
}

inline int get_max_headdim() {
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    return 256;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    return 192;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    return 128;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    return 96;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    return 64;
    #endif
    return 0;
}

#define PREPARE_VARLEN_MAX_BATCHES_1CTA 992

template<typename Actual, typename... Dims>
inline bool check_shape_impl(const Actual& actual_dims, Dims... expected) {
    const std::array<int64_t, sizeof...(Dims)> expected_dims = {static_cast<int64_t>(expected)...};
    return actual_dims.size() == sizeof...(Dims) && 
           std::equal(actual_dims.begin(), actual_dims.end(), expected_dims.begin());
}

template<typename... Dims>
std::string render_varargs_shape(Dims... dims) {
    std::string result = "[";
    ((result += std::to_string(dims) + ", "), ...);
    if (!result.empty()) {
        result.pop_back(); // remove last space
        result.pop_back(); // remove last comma
    }
    result += "]";
    return result;
}

template<typename Dims>
std::string render_dims_shape(const Dims& dims) {
    std::string result = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        result += std::to_string(dims[i]);
        if (i + 1 < dims.size()) {
            result += ", ";
        }
    }
    result += "]";
    return result;
}

#define CHECK_SHAPE(buf, ...)                                                                         \
  FFI_CHECK(check_shape_impl((buf).dimensions(), __VA_ARGS__))                                       \
      << #buf << " must have shape (" #__VA_ARGS__ ")"                                             \
      << " (actual shape: " << render_dims_shape((buf).dimensions()) << "; expected shape: (" << render_varargs_shape(__VA_ARGS__) << "))"

template <typename Buffer>
std::vector<int64_t> get_strides(const Buffer& buf) {
    int ndim = buf.dimensions().size();
    std::vector<int64_t> strides(ndim);
    int64_t s = 1;
    for (int i = ndim-1; i >= 0; --i) {
        strides[i] = s;
        s *= buf.dimensions()[i];
    }
    return strides;
}

inline bool get_pagedkv_tma(Flash_fwd_params const& params) {
    if (params.arch < 90 || !params.page_table || params.leftpad_k || params.knew_ptr) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, false /*paged_kv_non_TMA*/, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    int const kBlockN = std::get<1>(kBlockMN_kernel_args_sm90);
    // Heuristic: when seqlen_q <= kBlockM, we're not compute bound, and somehow using TMA is slower,
    // at least for MLA.
    return params.page_size % kBlockN == 0 && params.seqlen_q * (params.h / params.h_k) > kBlockM;
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    return false;
    #else
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
    #endif
}

inline int get_num_splits(Flash_fwd_params const& params) {
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    return 1;
    #else
    // Always enable PackGQA for Split
    // params.page_table must already be set
    // This needs to match the kernel configs
    bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    // Always enable PackGQA for Split
    // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on num_splits.
    // We assume the case where there's 1 long sequence and the rest are short, i.e. pretending
    // that batch = 1.
    int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
    return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
    #endif
}

template <int Arch, int Split, bool PagedKVNonTMA, bool PackGQA, bool Has_softcap>
void run_mha_fwd_constexpr(Flash_fwd_params &params, cudaStream_t stream) {
    if (!params.is_e4m3) {
        if (params.is_bf16) {
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
                if constexpr (Arch == 90) {
                    if (params.dv > 256) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    } else if (params.dv > 64) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
                if constexpr (Arch == 90) {
                    if (params.dv <= 128) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
        } else {
            #ifndef FLASHATTENTION_DISABLE_FP16
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
                if constexpr (Arch == 90) {
                    if (params.dv > 256) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    } else if (params.dv > 64) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
                if constexpr (Arch == 90) {
                    if (params.dv <= 128) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #else
            throw std::runtime_error("This flash attention build does not support FP16.");
            #endif
        }
    } else {
        #ifndef FLASHATTENTION_DISABLE_FP8
        #ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.d <= 64) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM96
        if (params.d <= 96) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.d <= 128) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.d <= 192) {
            #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
            if constexpr (Arch == 90) {
                if (params.dv <= 128) {
                    return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                }
            }
            #endif
            return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
        }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.d <= 256) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #else
        throw std::runtime_error("This flash attention build does not support FP8.");
        #endif
    }
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // HEADDIM_SWITCH(params.d, [&] {
    //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
    // });
    if(!(params.num_splits >= 1)) { throw std::runtime_error("num_splits must be at least 1"); };
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
                    });
                });
            });
        });
    });
}

void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl=false) {
    #ifndef FLASHATTENTION_DISABLE_SPLIT
    // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    // so that kBlockM is smaller and we have more parallelism.
    if (params.is_fp32) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<float, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<float, float, 128>(params, stream, enable_pdl);
        }
    } else if (params.is_bf16) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, enable_pdl);
        }
    } else {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, enable_pdl);
        }
    }
    #else
    throw std::runtime_error("This flash attention build does not support combine kernels.");
    #endif
}

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
ffi::Error
mha_fwd_ffi_impl(
        cudaStream_t stream, int32_t device_ordinal,
        ffi::AnyBuffer q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        ffi::AnyBuffer k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        ffi::AnyBuffer v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.

        // Optional arrays:
        ffi::AnyBuffer k_new,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        ffi::AnyBuffer v_new,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
        ffi::AnyBuffer q_v,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
        ffi::Buffer<ffi::S32> cu_seqlens_q,  // b+1
        ffi::Buffer<ffi::S32> cu_seqlens_k,  // b+1
        ffi::Buffer<ffi::S32> cu_seqlens_k_new,  // b+1
        // no seqused.
        // std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        // std::optional<at::Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        ffi::Buffer<ffi::S32> page_table, // (b_k, max_num_pages_per_seq)
        ffi::Buffer<ffi::S32> kv_batch_idx, // b. indices to index into the KV cache
        ffi::Buffer<ffi::S32> leftpad_k, // b
        ffi::AnyBuffer rotary_cos, // seqlen_ro x (rotary_dim / 2)
        ffi::AnyBuffer rotary_sin, // seqlen_ro x (rotary_dim / 2)
        ffi::Buffer<ffi::S32> seqlens_rotary, // b
        ffi::Buffer<ffi::F32> q_descale,  // (b, h_k), not (b, h)
        ffi::Buffer<ffi::F32> k_descale,  // (b, h_k)
        ffi::Buffer<ffi::F32> v_descale,  // (b, h_k)

        // Return arrays: out, softmax_lse, out_accum, softmax_lse_accum
        ffi::Result<ffi::AnyBuffer> out, // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        ffi::ResultBuffer<ffi::F32> softmax_lse, // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
        ffi::ResultBuffer<ffi::F32> out_accum,
        ffi::ResultBuffer<ffi::F32> softmax_lse_accum,
        ffi::ResultBuffer<ffi::S32> scheduler_metadata,  // (b + 1)


        std::optional<int64_t> max_seqlen_q_,
        std::optional<int64_t> max_seqlen_k_,
        std::optional<double> softmax_scale_,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        int64_t attention_chunk,
        double softcap,
        bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
        int64_t num_splits,
        std::optional<bool> pack_gqa_,
        int64_t sm_margin
        ) {

    int major, minor;
    FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal));
    FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal));
    int arch = major * 10 + minor;
    bool is_sm8x = major >= 8;
    FFI_CHECK(is_sm8x) << "FlashAttention only supports Ampere GPUs or newer.";

    auto q_type = q.element_type();
    FFI_CHECK(q_type == ffi::F16 || q_type == ffi::BF16 || q_type == ffi::F8E4M3FN)
                << "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type";
    if (major < 9) {
        FFI_CHECK(q_type == ffi::F16 || q_type == ffi::BF16)
                    << "FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type";
    }
    FFI_CHECK(k.element_type() == q_type) << "query and key must have the same dtype";
    FFI_CHECK(v.element_type() == q_type) << "query and value must have the same dtype";

    // jax tensor are always contiguous
    // TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    // We represent missing optional tensors as a zero-dimensional tensor, since that will never happen.
    const bool paged_KV = page_table.dimensions().size() > 0;
    if (paged_KV) {
        FFI_CHECK(page_table.element_type() == ffi::S32) << "page_table must have dtype torch.int32";
        // TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
    }

    bool const is_varlen_q = cu_seqlens_q.dimensions().size() > 0;
    if (is_varlen_q) {
        FFI_CHECK(cu_seqlens_q.element_type() == ffi::S32) << "cu_seqlens_q must have dtype torch.int32";
        FFI_CHECK(max_seqlen_q_.has_value()) << "max_seqlen_q must be provided if cu_seqlens_q is provided";
    }
    bool const is_varlen_k = cu_seqlens_k.dimensions().size() > 0;
    if (is_varlen_k) {
        FFI_CHECK(cu_seqlens_k.element_type() == ffi::S32) << "cu_seqlens_k must have dtype torch.int32";
        FFI_CHECK(max_seqlen_k_.has_value()) << "max_seqlen_k must be provided if cu_seqlens_k is provided";
        FFI_CHECK(!paged_KV) << "If cu_seqlens_k is passed in, then page table is not supported";
        FFI_CHECK(!(kv_batch_idx.dimensions().size() > 0)) << "If cu_seqlens_k is passed in, then page table is not supported";
    }

    auto const sizes = q.dimensions();
    const int batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.dimensions()[0] - 1;
    int seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_.value();
    int total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
    int num_heads = q.dimensions()[q.dimensions().size()-2];
    int const head_size = q.dimensions()[q.dimensions().size()-1];
    int const head_size_v = v.dimensions()[v.dimensions().size()-1];
    int const max_num_pages_per_seq = !paged_KV ? 0 : page_table.dimensions()[1];
    int const num_pages = !paged_KV ? 0 : k.dimensions()[0];
    int const page_size = !paged_KV ? 1 : k.dimensions()[1];
    int const seqlen_k = !is_varlen_k ? (!paged_KV ? k.dimensions()[1] : max_num_pages_per_seq * page_size) : max_seqlen_k_.value();
    int const total_k = !is_varlen_k ? batch_size * k.dimensions()[1] : k.dimensions()[0];
    int const num_heads_k = k.dimensions()[k.dimensions().size()-2];
    int const batch_size_k = !paged_KV ? (!is_varlen_k ? k.dimensions()[0] : cu_seqlens_k.dimensions()[0] - 1) : page_table.dimensions()[0];
    double softmax_scale = 1.0 / sqrt(double(head_size));
    if (softmax_scale_.has_value()) {
        softmax_scale = softmax_scale_.value();
    }
    if (!(kv_batch_idx.dimensions().size() > 0)) {
        FFI_CHECK(batch_size == batch_size_k) << "batch_size must be equal to batch_size_k";
    }
    int const max_headdim = get_max_headdim();
    FFI_CHECK(head_size <= max_headdim) << "FlashAttention forward only supports head dimension at most " + std::to_string(max_headdim);
    FFI_CHECK(num_heads % num_heads_k == 0) << "Number of heads in key/value must divide number of heads in query";
    if (head_size_v != head_size) {
        FFI_CHECK((head_size > 128 && head_size <= 192 && head_size_v > 96 && head_size_v <= 128) ||
                   (head_size <= 64 && head_size_v <= 512)) << 
                   "If V headdim is different from Q/K dim, we only support Q/K headdim in (128, 192] and V headdim in (96, 128], "
                   "or (Q/K <= 64 and V <= 512).";
        FFI_CHECK(major == 9) << "Only Hopper supports different V headdim";
        if (head_size_v > 256) {
            FFI_CHECK(q_type == ffi::F16 || q_type == ffi::BF16) << "HeaddimV > 256 requires fp16 and bf16 data type";
        }
    }

    // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and is_causal to set kBlockM
    // TODO: check this
    if (window_size_left >= seqlen_k - 1) { window_size_left = -1; }
    if (window_size_right >= seqlen_q - 1) { window_size_right = -1; }
    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1 && attention_chunk == 0) {
        // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
        if ((head_size <= 64 || head_size > 128) || !paged_KV) {
            is_causal = false;
        }
    }
    if (is_causal) { window_size_right = 0; }

    if (!is_varlen_q) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    } else {
        CHECK_SHAPE(q, total_q, num_heads, head_size);
        CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    }
    if (!paged_KV) {
        if (!is_varlen_k) {
            CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
            CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size_v);
        } else {
            CHECK_SHAPE(k, total_k, num_heads_k, head_size);
            CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
            CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
        }
    } else {
        CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
        CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
    }

    // if (seqused_q.has_value()){
    //     auto seqused_q = seqused_q.value();
    //     FFI_CHECK(seqused_q.dtype() == torch::kInt32) << "seqused_q must have dtype int32";
    //     CHECK_DEVICE(seqused_q); CHECK_CONTIGUOUS(seqused_q);
    //     CHECK_SHAPE(seqused_q, batch_size);
    // }
    // if (seqused_k.has_value()) {
    //     auto seqused_k = seqused_k.value();
    //     FFI_CHECK(seqused_k.dtype() == torch::kInt32) << "seqused_k must have dtype int32";
    //     CHECK_DEVICE(seqused_k); CHECK_CONTIGUOUS(seqused_k);
    //     CHECK_SHAPE(seqused_k, batch_size);
    // }

    if (leftpad_k.dimensions().size() > 0) {
        FFI_CHECK(leftpad_k.element_type() == ffi::S32) << "leftpad_k must have dtype int32";
        CHECK_SHAPE(leftpad_k, batch_size);
    }

    // This is what we will template on
    bool const is_varlen = is_varlen_q || is_varlen_k || (leftpad_k.dimensions().size() > 0);
    #ifdef FLASHATTENTION_DISABLE_VARLEN
        FFI_CHECK(!is_varlen) << "This flash attention build does not support varlen.";
    #endif

    int const alignment = q_type == ffi::F8E4M3FN ? 16 : 8;
    FFI_CHECK(head_size % alignment == 0) << "head_size should be a multiple of " + std::to_string(alignment);
    FFI_CHECK(head_size_v % alignment == 0) << "head_size_v should be a multiple of " + std::to_string(alignment);

    // auto opts = q.options();
    auto out_type = q_type == ffi::F8E4M3FN ? ffi::BF16 : q_type;
    FFI_CHECK(out->dimensions().size() > 0);

    FFI_CHECK(out->element_type() == out_type) << "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16";
    if (!is_varlen_q) {
        CHECK_SHAPE(*out, batch_size, seqlen_q, num_heads, head_size_v);
    } else {
        CHECK_SHAPE(*out, total_q, num_heads, head_size_v);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int const head_size_rounded = round_up_headdim(head_size);
    int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdimv(head_size_v);
    int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
    int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // at::Tensor softmax_lse;
    if (!is_varlen_q) {
        CHECK_SHAPE(*softmax_lse, batch_size, num_heads, seqlen_q);
    } else {
        CHECK_SHAPE(*softmax_lse, num_heads, total_q);
    }

    Flash_fwd_params params;
    set_params_fprop_ffi(params, stream, device_ordinal,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, *out,
                     !is_varlen_q ? nullptr : cu_seqlens_q.untyped_data(),
                     !is_varlen_k ? nullptr : cu_seqlens_k.untyped_data(),
                     /*seqused_q=*/ nullptr,
                     /*seqused_k=*/ nullptr,
                     softmax_lse->untyped_data(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     attention_chunk,
                     softcap,
                     sm_margin);
    params.total_q = total_q;
    params.total_k = total_k;
    params.b_k = batch_size_k;
    params.dv = head_size_v;
    params.dv_rounded = head_size_v_rounded;
    if (leftpad_k.dimensions().size() > 0) {  // This needs to be set before get_pagedkv_tma
        params.leftpad_k = static_cast<int *>(leftpad_k.untyped_data());
    }
    if (paged_KV) {
        auto page_table_strides = get_strides(page_table);
        params.page_table = page_table.typed_data();
        params.page_table_batch_stride = page_table_strides[0];
    }
    params.page_size = page_size;
    params.num_pages = num_pages;

    if (k_new.dimensions().size() > 0) {  // This needs to be set before get_pagedkv_tma
        FFI_CHECK(v_new.dimensions().size() > 0) << "If k_new is supplied, v_new must also be passed in";
        FFI_CHECK(cu_seqlens_k.dimensions().size() > 0) << "If k_new is supplied, seqlens_k must also be passed in";
        FFI_CHECK(seqlen_q <= seqlen_k) << "If k_new is supplied, it must have seqlen <= the seqlen of the KV cache";
        bool const is_varlen_k_new = cu_seqlens_k_new.dimensions().size() > 0;
        if (is_varlen_k_new) {
            FFI_CHECK(cu_seqlens_k_new.element_type() == ffi::S32) << "cu_seqlens_k_new must have dtype torch.int32";
        }
        FFI_CHECK(k_new.element_type() == q_type) << "k_new must have the same dtype as query";
        FFI_CHECK(v_new.element_type() == q_type) << "v_new must have the same dtype as query";
        auto k_new_strides = get_strides(k_new);
        auto v_new_strides = get_strides(v_new);
        // FFI_CHECK(k_new_strides.back() == 1) << "k_new tensor must have contiguous last dimension";
        // FFI_CHECK(v_new_strides.back() == 1) << "v_new tensor must have contiguous last dimension";
        // We don't need max_seqlen_k_new, so seqlen_k_new can be whatever when is_varlen_k_new
        int seqlen_k_new = !is_varlen_k_new ? k_new.dimensions()[1] : 0;
        int total_k_new = !is_varlen_k_new ? batch_size * k_new.dimensions()[1]: k_new.dimensions()[0];
        if (!is_varlen_k_new) {
            CHECK_SHAPE(k_new, batch_size, seqlen_k_new, num_heads_k, head_size);
            CHECK_SHAPE(v_new, batch_size, seqlen_k_new, num_heads_k, head_size_v);
        } else {
            CHECK_SHAPE(k_new, total_k_new, num_heads_k, head_size);
            CHECK_SHAPE(v_new, total_k_new, num_heads_k, head_size_v);
            CHECK_SHAPE(cu_seqlens_k_new, batch_size + 1);
        }
        params.seqlen_knew = seqlen_k_new;
        params.total_knew = total_k_new;
        params.knew_ptr = k_new.untyped_data();
        params.vnew_ptr = v_new.untyped_data();
        // All stride are in elements, not bytes.
        params.knew_row_stride = k_new_strides[k_new_strides.size() - 3];
        params.vnew_row_stride = v_new_strides[v_new_strides.size() - 3];
        params.knew_head_stride = k_new_strides[k_new_strides.size() - 2];
        params.vnew_head_stride = v_new_strides[v_new_strides.size() - 2];
        if (!is_varlen_k_new) {
            params.knew_batch_stride = k_new_strides[0];
            params.vnew_batch_stride = v_new_strides[0];
        }
        if (is_varlen_k_new) {
            params.cu_seqlens_knew = cu_seqlens_k_new.typed_data();
        }
    }
    
    bool const use_prepare_varlen = is_varlen;
    params.prepare_varlen_pdl = use_prepare_varlen && params.b <= PREPARE_VARLEN_MAX_BATCHES_1CTA;
    // Temporarily set num_splits_dynamic_ptr to 1 since get_num_splits checks it
    params.num_splits_dynamic_ptr = !use_prepare_varlen ? nullptr : reinterpret_cast<int*>(1);

    params.pagedkv_tma = get_pagedkv_tma(params);
    params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
    // Always enable PackGQA for Split, and get_pack_gqa requires params.num_splits to decide
    params.pack_gqa = pack_gqa_.has_value() ? pack_gqa_.value() : get_pack_gqa(params);

    // This needs to be set after get_num_splits
    ffi::Buffer<ffi::S32>& tile_count_semaphore = *scheduler_metadata;  // Contains the semaphore and optionally num_splits_dynamic
    // We don't use the persistent scheduler if Split and not Varlen
    bool const scheduler_needs_semaphore = params.arch >= 90
        ? (((params.is_causal || params.is_local) && (params.num_splits == 1)) || is_varlen)
        : ((params.is_causal && !is_varlen) || (is_varlen && params.num_splits > 1));
    params.varlen_sort_batches = !params.is_local; // Use this value for Sort in scheduler template
    params.head_swizzle = params.is_causal || params.is_local; // Use this value for LPT in scheduler template
    if (scheduler_needs_semaphore || use_prepare_varlen) {
        int b_rounded = round_multiple(params.b, 4); // for 16 byte alignment of pointers
        int num_prepare_batch_vectors = use_prepare_varlen ? 2 : 0;
        if(params.varlen_sort_batches) { num_prepare_batch_vectors += 1; }
        if(params.head_swizzle) { num_prepare_batch_vectors += 1; }
        int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
        int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
        int metadata_size = int(scheduler_needs_semaphore) + tile_count_semaphore_offset;
        // printf("Num prepare batch vectors = %d, metadata_size = %d.\n", num_prepare_batch_vectors, metadata_size);

        // Always compute metadata in the fwd kernel for simplicity.
        params.skip_scheduler_metadata_computation = false; //scheduler_metadata.dimensions().size() > 0;
        FFI_CHECK(scheduler_metadata->dimensions().size() > 0) << "scheduler_metadata must be provided when scheduler_needs_semaphore or use_prepare_varlen is true";
        // if (scheduler_metadata.dimensions().size() > 0) {
        CHECK_SHAPE(*scheduler_metadata, metadata_size);
        FFI_CHECK(scheduler_metadata->element_type() == ffi::S32) << "scheduler_metadata must have dtype int32";
        // }
        // if (scheduler_needs_semaphore && !use_prepare_varlen) {
        // manually zero it
        cudaMemsetAsync(
            tile_count_semaphore.untyped_data(),
            0,
            tile_count_semaphore.size_bytes(),
            stream);
        // }
        // {num_splits_dynamic, num_m_blocks, varlen_batch_idx, num_nheads_in_l2}
        params.num_splits_dynamic_ptr = use_prepare_varlen ? tile_count_semaphore.typed_data() : nullptr;
        params.num_m_blocks_ptr =  use_prepare_varlen ? tile_count_semaphore.typed_data() + b_rounded : nullptr;
        params.varlen_batch_idx_ptr =  use_prepare_varlen && params.varlen_sort_batches ? tile_count_semaphore.typed_data() + b_rounded * 2 : nullptr;
        // params.num_n_blocks_ptr  = use_prepare_varlen && params.head_swizzle ? tile_count_semaphore.typed_data() + head_swizzle_offset : nullptr;
        params.num_nheads_in_l2_ptr = use_prepare_varlen && params.head_swizzle ? tile_count_semaphore.typed_data() + head_swizzle_offset : nullptr;
        params.tile_count_semaphore = scheduler_needs_semaphore ? tile_count_semaphore.typed_data() + tile_count_semaphore_offset : nullptr;
        params.tile_count_semaphore_offset = tile_count_semaphore_offset; // might need to zero out semaphore later
    }

    if (q_v.dimensions().size() > 0) {
        throw std::runtime_error("q_v not yet implemented");
        // FFI_CHECK(head_size <= 64) << "q_v is only supported for head_size <= 64";
        // FFI_CHECK(head_size_v >= 256) << "q_v is only supported for hdim_v >= 256.";
        // FFI_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16) << "q_v is only supported for fp16 and bf16 data type";
        // FFI_CHECK(params.arch == 90) << "q_v is only supported for Hopper GPUs";
        // at::Tensor q_v = q_v.value();
        // FFI_CHECK(q_v.dtype() == q_type) << "q_v must have the same dtype as query";
        // CHECK_DEVICE(q_v);
        // FFI_CHECK(q_v.stride(-1) == 1) << "q_v tensor must have contiguous last dimension";
        // if (!is_varlen_q) {
        //     CHECK_SHAPE(q_v, batch_size, seqlen_q, num_heads, head_size_v);
        // } else {
        //     CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
        // }
        // params.qv_ptr = q_v.data_ptr();
        // // All stride are in elements, not bytes.
        // params.qv_row_stride = q_v.stride(-3);
        // params.qv_head_stride = q_v.stride(-2);
        // if (!is_varlen_q) {
        //     params.qv_batch_stride = q_v.stride(0);
        // }
    }

    if (rotary_cos.dimensions().size() > 0) {
        throw std::runtime_error("rotary not yet implemented");
        // FFI_CHECK(k_new.dimensions().size() > 0) << "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided";
        // auto rotary_cos = rotary_cos.value();
        // CHECK_DEVICE(rotary_cos); CHECK_CONTIGUOUS(rotary_cos);
        // params.rotary_dim = rotary_cos.size(1) * 2;
        // FFI_CHECK(params.rotary_dim <= head_size) << "rotary_dim must be <= headdim";
        // FFI_CHECK(params.rotary_dim % 16 == 0) << "Only rotary dimensions divisible by 16 are currently supported";
        // const int seqlen_ro = rotary_cos.size(0);
        // if (paged_KV) {
        //     FFI_CHECK(seqlen_ro >= seqlen_k) << "cos/sin seqlen must be at least the seqlen of KV cache";
        // }
        // CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        // FFI_CHECK(rotary_cos.scalar_type() == q_type) << "rotary_cos must have the same dtype as query";

        // FFI_CHECK(rotary_sin.dimensions().size() > 0) << "If rotary cos is provided, rotary sin must also be provided";
        // auto rotary_sin = rotary_sin.value();
        // CHECK_DEVICE(rotary_sin); CHECK_CONTIGUOUS(rotary_sin);
        // CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        // FFI_CHECK(rotary_sin.scalar_type() == q_type) << "rotary_cos must have the same dtype as query";
        // params.rotary_cos_ptr = rotary_cos.data_ptr();
        // params.rotary_sin_ptr = rotary_sin.data_ptr();
        // params.is_rotary_interleaved = is_rotary_interleaved;
        // if (seqlens_rotary.dimensions().size() > 0) {
        //     at::Tensor seqlens_rotary = seqlens_rotary.value();
        //     CHECK_DEVICE(seqlens_rotary); CHECK_CONTIGUOUS(seqlens_rotary);
        //     FFI_CHECK(seqlens_rotary.dtype() == torch::kInt32) << "seqlens_rotary must have dtype torch.int32";
        //     CHECK_SHAPE(seqlens_rotary, batch_size);
        //     params.seqlens_rotary = seqlens_rotary.data_ptr<int>();
        // }
    } else {
        params.rotary_dim = 0;
    }

    if (kv_batch_idx.dimensions().size() > 0) {
        FFI_CHECK(kv_batch_idx.element_type() == ffi::S32) << "kv_batch_idx must have dtype int32";
        params.kv_batch_idx = kv_batch_idx.typed_data();
    }

    // at::Tensor out_accum, softmax_lse_accum;
    auto outaccum_type = ffi::S32;
    if (params.num_splits > 1) {
        FFI_CHECK(params.num_splits <= 256) << "num_splits > 256 not supported";
        auto out_accum_strides = get_strides(*out_accum);
        auto softmax_lse_accum_strides = get_strides(*softmax_lse_accum);
        if (!is_varlen_q) {
            CHECK_SHAPE(*out_accum, params.num_splits, batch_size, num_heads, seqlen_q, head_size_v);
            CHECK_SHAPE(*softmax_lse_accum, params.num_splits, batch_size, num_heads, seqlen_q);
            // out_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q, head_size_v}, opts.dtype(outaccum_type));
            // softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            params.oaccum_batch_stride = out_accum_strides[1];
            params.lseaccum_batch_stride = softmax_lse_accum_strides[1];
        } else {
            CHECK_SHAPE(*out_accum, params.num_splits, num_heads, total_q, head_size_v);
            CHECK_SHAPE(*softmax_lse_accum, params.num_splits, num_heads, total_q);
            // out_accum = torch::empty({params.num_splits, num_heads, total_q, head_size_v}, opts.dtype(outaccum_type));
            // softmax_lse_accum = torch::empty({params.num_splits, num_heads, total_q}, opts.dtype(at::kFloat));
        }
        params.is_fp32 = false;
        params.oaccum_ptr = out_accum->untyped_data();
        params.softmax_lseaccum_ptr = softmax_lse_accum->untyped_data();
        params.oaccum_split_stride = out_accum_strides[0];
        params.oaccum_row_stride = out_accum_strides[out_accum_strides.size()-2];
        params.oaccum_head_stride = out_accum_strides[out_accum_strides.size()-3];
        params.lseaccum_split_stride = softmax_lse_accum_strides[0];
        params.lseaccum_head_stride = softmax_lse_accum_strides[softmax_lse_accum_strides.size()-2];
    }

    if (q_type == ffi::F8E4M3FN) {
        if (q_descale.dimensions().size() > 0) {
            CHECK_SHAPE(q_descale, batch_size, num_heads_k);
            auto q_descale_strides = get_strides(q_descale);
            params.q_descale_ptr = q_descale.typed_data();
            params.q_descale_batch_stride = q_descale_strides[0];
            params.q_descale_head_stride = q_descale_strides[1];
        } else {
            params.q_descale_ptr = nullptr;
        }
        if (k_descale.dimensions().size() > 0) {
            CHECK_SHAPE(k_descale, batch_size, num_heads_k);
            auto k_descale_strides = get_strides(k_descale);
            params.k_descale_ptr = k_descale.typed_data();
            params.k_descale_batch_stride = k_descale_strides[0];
            params.k_descale_head_stride = k_descale_strides[1];
        } else {
            params.k_descale_ptr = nullptr;
        }
        if (v_descale.dimensions().size() > 0) {
            CHECK_SHAPE(v_descale, batch_size, num_heads_k);
            auto v_descale_strides = get_strides(v_descale);
            params.v_descale_ptr = v_descale.typed_data();
            params.v_descale_batch_stride = v_descale_strides[0];
            params.v_descale_head_stride = v_descale_strides[1];
        } else {
            params.v_descale_ptr = nullptr;
        }
    }

    #ifdef FLASHATTENTION_DISABLE_LOCAL
    FFI_CHECK(!params.is_local) << "This flash attention build does not support local attention.";
    #endif
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    FFI_CHECK(params.softcap == 0.0) << "This flash attention build does not support tanh softcapping.";
    #endif
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    FFI_CHECK(params.num_splits == 1) << "This flash attention build does not support splits.";
    #endif
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    FFI_CHECK(!params.pack_gqa || params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) << "This flash attention build does not support pack_gqa.";
    #endif
    #ifdef FLASHATTENTION_DISABLE_PAGEDKV
    FFI_CHECK(!(params.page_table && !params.pagedkv_tma)) << "This flash attention build does not support paged KV.";
    #endif
    #ifdef FLASHATTENTION_DISABLE_APPENDKV
    FFI_CHECK(!(k_new_.dimensions().size() > 0)) << "This flash attention build does not support appending KV.";
    #endif

    if (total_q > 0 && (total_k + params.total_knew) > 0 && num_heads_k > 0) {
        run_mha_fwd(params, stream);
        if (params.num_splits > 1) {
            if (out_type == ffi::BF16) {
                // Since we want output in BF16. Otherwise fwd_combine will output to FP16
                params.is_bf16 = true;
            }
            // Unless there's seqused_q, for the purpose of attn_combine, we can just treat it as batch=1
            // and seqlen = total_q, and don't need to dispatch to Varlen there.
            // However, with dynamic split, each row needs to know which batch it belongs to
            // to read the number of splits, so we just use the varlen version of combine kernel.
            // if (is_varlen_q && !seqused_q_.has_value()) {
            // if (is_varlen_q) {
            //     params.b = 1;
            //     params.seqlen_q = total_q;
            // }
            // This will zero out the semaphore if needed
            run_mha_fwd_combine(params, stream, true /*enable_pdl*/);
        }
    } else if (total_q > 0 && num_heads_k > 0) {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        // This may seem cursed, but that's because it is. 0xFEFEFEFE is approximately -1.69e+38, which is close enough to -inf.
        cudaMemsetAsync(out->untyped_data(), 0, out->size_bytes(), stream);
        cudaMemsetAsync(out->untyped_data(), 0xFE, out->size_bytes(), stream);
    }

    // return {out, softmax_lse};
    return ffi::Error();
}

// ffi::Error mha_fwd_ffi_impl(cudaStream_t stream, int32_t device, ffi::AnyBuffer q, ffi::AnyBuffer k,
//                             ffi::AnyBuffer v, ffi::Result<ffi::AnyBuffer> o,
//                             ffi::ResultBuffer<ffi::F32> lse, ffi::ResultBuffer<ffi::F32> oaccum,
//                             ffi::ResultBuffer<ffi::F32> lseaccum, float softmax_scale,
//                             bool is_causal, int64_t window_size_left, int64_t window_size_right,
//                             float softcap, int64_t num_splits) {

//   // Get device compute capability
//   int major, minor;
//   FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
//   FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

//   bool is_sm8x = major == 8 && minor >= 0;
//   bool is_sm90 = major == 9 && minor == 0;
//   FFI_CHECK(is_sm90 || is_sm8x)
//       << ffi::ErrorCode::kUnimplemented
//       << "FlashAttention only supports Ampere (SM80+) and Hopper (SM90) GPUs";

//   // Check dtype
//   ffi::DataType dtype = q.element_type();
//   FFI_CHECK(dtype == k.element_type() && dtype == v.element_type() && dtype == o->element_type())
//       << ffi::ErrorCode::kInvalidArgument
//       << "query, key, value, and output must have the same dtype";
//   FFI_CHECK(dtype == ffi::DataType::F16 || dtype == ffi::DataType::BF16)
//       << ffi::ErrorCode::kInvalidArgument
//       << "FlashAttention only supports fp16 and bf16 data types";

//   if (dtype == ffi::DataType::BF16) {
//     FFI_CHECK(is_sm90 || is_sm8x) << ffi::ErrorCode::kInvalidArgument
//                                   << "bfloat16 is only supported on Ampere GPUs or newer";
//   }

//   // Extract tensor shapes
//   const int batch_size = q.dimensions()[0];
//   int seqlen_q = q.dimensions()[1];
//   int num_heads = q.dimensions()[2];
//   const int head_size = q.dimensions()[3];
//   const int seqlen_k = k.dimensions()[1];
//   const int num_heads_k = k.dimensions()[2];

//   // Validate shapes and parameters
//   FFI_CHECK(batch_size > 0) << ffi::ErrorCode::kInvalidArgument << "batch size must be positive";
//   FFI_CHECK(head_size > 0) << ffi::ErrorCode::kInvalidArgument << "head_size must be positive";
//   FFI_CHECK(head_size <= 256)
//       << ffi::ErrorCode::kInvalidArgument
//       << "FlashAttention forward only supports head dimension at most 256, got " << head_size;
//   FFI_CHECK(head_size % 8 == 0) << ffi::ErrorCode::kInvalidArgument
//                                 << "head_size must be divisible by 8, got " << head_size;
//   FFI_CHECK(num_heads % num_heads_k == 0)
//       << ffi::ErrorCode::kInvalidArgument
//       << "Number of heads in key/value must divide number of heads in query; "
//       << "got num_heads=" << num_heads << ", num_heads_k=" << num_heads_k;

//   // Check output tensor
//   FFI_CHECK(o->dimensions()[0] == batch_size && o->dimensions()[1] == seqlen_q &&
//             o->dimensions()[2] == num_heads && o->dimensions()[3] == head_size)
//       << ffi::ErrorCode::kInvalidArgument << "Output tensor shape mismatch";

//   // Preprocessing window sizes (matching PyTorch logic before set_params_fprop)
//   if (window_size_left >= seqlen_k - 1) {
//     window_size_left = -1;
//   }
//   if (window_size_right >= seqlen_q - 1) {
//     window_size_right = -1;
//   }

//   // Special case: if seqlen_q == 1, causal attention is equivalent to non-causal
//   // (unless we have special head_size requirements for paged KV)
//   if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1) {
//     is_causal = false;
//   }

//   if (is_causal) {
//     window_size_right = 0;
//   }

//   // Setup forward parameters
//   Flash_fwd_params params;
//   FFI_RET_CHECK(set_params_fprop_ffi(
//       params, dtype, batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, head_size,
//       q.untyped_data(), k.untyped_data(), v.untyped_data(), o->untyped_data(), lse->untyped_data(),
//       oaccum->untyped_data(), lseaccum->untyped_data(), softmax_scale, softcap,
//       static_cast<int>(window_size_left), static_cast<int>(window_size_right),
//       static_cast<int>(num_splits)));

//   // Set device architecture information for kernel dispatch
//   params.arch = major * 10 + minor;  // 80, 86, 89, or 90

//   // Get number of SMs for split-KV heuristic
//   int num_sms = 0;
//   FFI_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
//   params.num_sm = num_sms;

//   // Zero out output accumulators if we're using split-KV
//   if (num_splits > 1) {
//     FFI_CUDA_CHECK(cudaMemsetAsync(oaccum->untyped_data(), 0, oaccum->size_bytes(), stream));
//     FFI_CUDA_CHECK(cudaMemsetAsync(lseaccum->untyped_data(), 0, lseaccum->size_bytes(), stream));
//   }

//   // Dispatch to appropriate kernel based on architecture, head_dim, and dtype
//   // We use static switches similar to the PyTorch bindings
//   try {
//     ARCH_SWITCH(params.arch, Arch, [&] {
//       SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
//         PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
//           PACKGQA_SWITCH(params.pack_gqa, PackGQA, [&] {
//             SOFTCAP_SWITCH(params.softcap > 0.0f, Has_softcap, [&] {
//               constexpr bool PackGQA_forced = PackGQA || Split;
//               if (!params.is_e4m3) {
//                 if (params.is_bf16) {
//                   if (params.d <= 64) {
//                     return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   } else if (params.d <= 96) {
//                     return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   } else if (params.d <= 128) {
//                     return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   } else if (params.d <= 192) {
//                     return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   } else if (params.d <= 256) {
//                     return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   }
//                 } else {
//                   if (params.d <= 64) {
//                     return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   } else if (params.d <= 96) {
//                     return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   } else if (params.d <= 128) {
//                     return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   } else if (params.d <= 192) {
//                     return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   } else if (params.d <= 256) {
//                     return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA,
//                                         Has_softcap, PackGQA_forced>(params, stream);
//                   }
//                 }
//               }
//             });
//           });
//         });
//       });
//     });
//   } catch (const CudaException &e) {
//     return ffi::Error(ffi::ErrorCode::kInternal, e.what());
//   }

//   FFI_CUDA_CHECK(cudaGetLastError());
//   FFI_CUDA_CHECK(cudaStreamSynchronize(stream)) << "after run_mha_fwd_";

//   // If using split-KV, combine the partial results
//   if (num_splits > 1) {
//     // Set output dtype for combine kernel
//     if (dtype == ffi::DataType::BF16) {
//       params.is_bf16 = true;
//     }

//     // Dispatch combine kernel based on dtype and head dimension
//     try {
//       if (params.is_bf16) {
//         if (params.dv <= 64) {
//           run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, true /*enable_pdl*/);
//         } else {
//           run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, true /*enable_pdl*/);
//         }
//       } else {
//         if (params.dv <= 64) {
//           run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream, true /*enable_pdl*/);
//         } else {
//           run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, true /*enable_pdl*/);
//         }
//       }
//     } catch (const CudaException &e) {
//       return ffi::Error(ffi::ErrorCode::kInternal, e.what());
//     }

//     FFI_CUDA_CHECK(cudaGetLastError());
//   }

//   FFI_CUDA_CHECK(cudaStreamSynchronize(stream)) << "after combine kernel";

//   return ffi::Error(); // Success
// }
