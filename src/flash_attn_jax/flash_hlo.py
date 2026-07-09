from functools import partial, wraps
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters.mlir import ir

from jax.extend.core import Primitive

from einops import rearrange
import einops
import math

import os

import tvm_ffi
import jax_tvm_ffi

import flash_attn_jax_lib

# Both backends are plain tvm-ffi shared libraries (stable C ABI, no Python ABI
# dependency). FA3 is only built when FLASH_ATTN_CUDA_ARCHS includes an
# FA3-capable arch (90a and/or 80/86/89/120/121); FA2 is always built.
_lib_dir = list(flash_attn_jax_lib.__path__)[0]

flash_api = tvm_ffi.load_module(os.path.join(_lib_dir, "flash_api.so"))

_hopper_so = os.path.join(_lib_dir, "flash_hopper_ffi.so")
if os.path.exists(_hopper_so):
    flash_hopper_ffi = tvm_ffi.load_module(_hopper_so)
else:
    flash_hopper_ffi = None

def has_fa3() -> bool:
    return flash_hopper_ffi is not None

# Scalar attributes of each call, in the C++ parameter order. FA2 entry points take
# (args, rets, attrs); the FA3 one takes (rets, args, attrs).
_FA2_FWD_ATTRS = ["softmax_scale", "is_causal", "window_size_left", "window_size_right"]
_FA2_BWD_ATTRS = _FA2_FWD_ATTRS + ["deterministic"]
_FA2_VARLEN_FWD_ATTRS = [
    "max_seqlen_q", "max_seqlen_k", "softmax_scale", "zero_tensors",
    "is_causal", "window_size_left", "window_size_right",
]
_FA2_VARLEN_BWD_ATTRS = _FA2_VARLEN_FWD_ATTRS + ["deterministic"]
_FA3_FWD_ATTRS = [
    "max_seqlen_q", "max_seqlen_k", "softmax_scale", "is_causal",
    "window_size_left", "window_size_right", "attention_chunk", "softcap",
    "is_rotary_interleaved", "num_splits", "pack_gqa", "sm_margin",
]

def _attrs(names):
    return [f"attrs.{name}" for name in names]

def register_custom_calls():
    # Register FA2 (flash_api) through the jax-tvm-ffi bridge
    for target, func, attrs in [
        ("flash_mha_fwd", flash_api.fa2_fwd, _FA2_FWD_ATTRS),
        ("flash_mha_bwd", flash_api.fa2_bwd, _FA2_BWD_ATTRS),
        ("flash_mha_varlen_fwd", flash_api.fa2_varlen_fwd, _FA2_VARLEN_FWD_ATTRS),
        ("flash_mha_varlen_bwd", flash_api.fa2_varlen_bwd, _FA2_VARLEN_BWD_ATTRS),
    ]:
        jax_tvm_ffi.register_ffi_target(
            target, func,
            arg_spec=["args", "rets"] + _attrs(attrs),
            platform="gpu",
            allow_cuda_graph=True,
        )

    # Register FA3 (flash_hopper_ffi)
    if flash_hopper_ffi is not None:
        jax_tvm_ffi.register_ffi_target(
            "hopper_flash_mha_fwd",
            flash_hopper_ffi.fa3_fwd,
            arg_spec=["rets", "args"] + _attrs(_FA3_FWD_ATTRS),
            platform="gpu",
            allow_cuda_graph=True,
        )
