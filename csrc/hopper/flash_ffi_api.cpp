/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>

#include "flash_ffi_common.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "mha_fwd_ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace {

// Helper function to encapsulate FFI handler into a Python capsule
template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

// Define FFI handler for forward pass
XLA_FFI_DEFINE_HANDLER(
    hopper_mha_fwd, mha_fwd_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::AnyBuffer>()  // q
        .Arg<ffi::AnyBuffer>()  // k
        .Arg<ffi::AnyBuffer>()  // v
        .Ret<ffi::AnyBuffer>()  // out
        .Ret<ffi::Buffer<ffi::F32>>()  // softmax_lse
        .Ret<ffi::Buffer<ffi::F32>>()  // out_accum
        .Ret<ffi::Buffer<ffi::F32>>()  // softmax_lse_accum
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Attr<int64_t>("window_size_left")
        .Attr<int64_t>("window_size_right")
        .Attr<float>("softcap")
        .Attr<int64_t>("num_splits")
);

// FFI registrations dictionary
nb::dict FFIRegistrations() {
  nb::dict dict;
  dict["hopper_flash_mha_fwd"] = EncapsulateFfiCall(hopper_mha_fwd);
  // Future work: add varlen_fwd, bwd, varlen_bwd
  return dict;
}

}  // namespace

// Nanobind module definition
NB_MODULE(flash_hopper_ffi, m) {
    m.doc() = "FlashAttention for Hopper (XLA FFI frontend)";
    m.def("get_ffi_registrations", &FFIRegistrations,
          "Get FFI registrations for FlashAttention Hopper kernels");
}
