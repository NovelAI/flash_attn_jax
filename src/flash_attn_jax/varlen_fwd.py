from functools import partial, wraps
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.extend.core import Primitive
import jax._src.dispatch

from einops import rearrange
import einops
import math

from flash_attn_jax.util import num_splits_heuristic, round_multiple, get_sm_count
from flash_attn_jax.fa3_util import (
    get_num_splits_fa3,
    calculate_scheduler_metadata_size_varlen,
)

# ==== Register primitives ====

# type: [sq, hq, d] [sk, hk, d] [sk, hk, d] [b+1] [b+1] -> [sq, hq, d] ([b, hq, max_seqlen_q] | [hq, sq])
# someday we will switch to [n, sq, hq, d] [n, sk, hk, d] [n, sk, hk, d] [n, b+1] [n, b+1] -> [n, sq, hq, d] [n, b, hq, max_seqlen_q]
# so that sharding works better
_flash_mha_varlen_fwd_p = Primitive("flash_mha_varlen_fwd")
_flash_mha_varlen_fwd_p.multiple_results = True
_flash_mha_varlen_fwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_varlen_fwd_p))
jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_varlen_fwd_p)

# ==== Frontend ====


def flash_mha_varlen_fwd(
    q,
    k,
    v,
    seqlens_q,
    seqlens_k,
    *,
    max_seqlen_q: int = -1,
    max_seqlen_k: int = -1,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    backend: str, # required
):
    assert backend in ["fa2", "fa3"], "backend must be either 'fa2' or 'fa3'"
    if max_seqlen_q == -1:
        max_seqlen_q = q.shape[0]
    if max_seqlen_k == -1:
        max_seqlen_k = k.shape[0]
    assert seqlens_q.shape == seqlens_k.shape, (
        "seqlens_q and seqlens_k must have the same shape."
    )
    kwargs = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        backend=backend,
    )
    return tuple(_flash_mha_varlen_fwd_p.bind(q, k, v, seqlens_q, seqlens_k, **kwargs))


# ==== HLO lowering ====


def _flash_mha_varlen_fwd_hlo_lowering_fa2(
    q,
    k,
    v,
    seqlens_q,
    seqlens_k,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [totalq, h, d] = q.shape
    b = seqlens_q.shape[0] - 1
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    assert b >= 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    if d <= 64:
        block_n = 256
    elif d <= 128:
        block_n = 128
    else:
        block_n = 64
    num_n_blocks = max(1, (max_seqlen_k + block_n - 1) // block_n)
    num_m_blocks = max(1, (max_seqlen_q + 64 - 1) // 64)
    sm_count = get_sm_count()
    num_splits = num_splits_heuristic(b * h * num_m_blocks, sm_count * 2, num_n_blocks, 128)
    lseaccum_shape = (num_splits, b, h, max_seqlen_q)
    oaccum_shape = (num_splits, b, max_seqlen_q, h, round_multiple(d, 32))

    dpad = (8 - (d % 8)) % 8
    if dpad > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, dpad)), mode="constant", constant_values=0)
        k = jnp.pad(k, ((0, 0), (0, 0), (0, dpad)), mode="constant", constant_values=0)
        v = jnp.pad(v, ((0, 0), (0, 0), (0, dpad)), mode="constant", constant_values=0)

    out_shape = [totalq, h, d + dpad]
    lse_shape = [b, h, max_seqlen_q]

    out_types = [
        jax.ShapeDtypeStruct(out_shape, q_dtype),
        jax.ShapeDtypeStruct(lse_shape, jnp.float32),
        jax.ShapeDtypeStruct(oaccum_shape, jnp.float32),
        jax.ShapeDtypeStruct(lseaccum_shape, jnp.float32),
    ]

    out, lse = jax.ffi.ffi_call(
        "flash_mha_varlen_fwd",
        result_shape_dtypes=out_types,
        has_side_effect=False,
        input_layouts=[None] * 5,  # default row major
        output_layouts=[None] * 4,
    )(
        q,
        k,
        v,
        seqlens_q,
        seqlens_k,
        max_seqlen_q=mlir.i32_attr(max_seqlen_q),
        max_seqlen_k=mlir.i32_attr(max_seqlen_k),
        softmax_scale=softmax_scale,
        zero_tensors=False,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )[:2]

    if dpad > 0:
        out = out[:, :, :d]

    return out, lse

def _flash_mha_varlen_fwd_hlo_lowering_fa3(
    q,
    k,
    v,
    seqlens_q,
    seqlens_k,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
):
    """FA3 (Hopper) lowering for varlen attention."""
    dtype = q.dtype
    [total_q, h, d] = q.shape
    [total_k, hk, _] = k.shape
    b = seqlens_q.shape[0] - 1  # batch_size from cumulative seqlens

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    # Match C++ window size adjustments (mha_fwd_ffi.cpp:499-508)
    # These affect is_causal/is_local which determine metadata_size
    if window_size_left >= max_seqlen_k - 1:
        window_size_left = -1
    if window_size_right >= max_seqlen_q - 1:
        window_size_right = -1
    # The C++ seqlen_q==1 branch has a paged-KV exception; this path never
    # passes a page_table, so it collapses to unconditional
    if max_seqlen_q == 1 and window_size_left == -1 and window_size_right == -1:
        is_causal = False
    if is_causal:
        window_size_right = 0

    # Final flags derived from adjusted windows (flash_ffi_common.cpp:153-154)
    is_causal = window_size_left < 0 and window_size_right == 0
    is_local = (window_size_left >= 0 or window_size_right >= 0) and not is_causal

    # Calculate num_splits using varlen-specific heuristics
    # For varlen with dynamic split, assume worst case: batch=1 long sequence
    num_splits = get_num_splits_fa3(
        batch_size=1,  # worst case for varlen
        seqlen_q=max_seqlen_q,
        seqlen_k=max_seqlen_k,
        num_heads=h,
        num_heads_k=hk,
        head_dim=d,
        head_dim_v=d,
        is_causal=is_causal,
        is_local=is_local,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        dtype=dtype,
        num_sm=114,
        max_splits=128,
    )

    # Calculate scheduler_metadata size for varlen
    metadata_size = calculate_scheduler_metadata_size_varlen(
        batch_size=b,
        num_splits=num_splits,
        is_causal=is_causal,
        is_local=is_local,
        arch=90,
    )

    # print('[varlen] Computing num_splits and metadata_size for configuration:')
    # print(f'[varlen] batch_size={1}, seqlen_q={max_seqlen_q}, seqlen_k={max_seqlen_k}, num_heads={h}, num_heads_k={hk}, head_dim={d}, is_causal={is_causal}, is_local={is_local}, window_size_left={window_size_left}, window_size_right={window_size_right}, dtype={dtype}, num_sm=114, max_splits=128')
    # print(f'[varlen] batch_size={b}, num_splits={num_splits}, is_causal={is_causal}, is_local={is_local}, arch=90')
    # print('[varlen] computed num_splits:', num_splits, 'metadata_size:', metadata_size)

    # Padding for head dimension alignment
    dpad = (8 - d % 8) % 8
    if dpad > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, dpad)), 'constant')
        k = jnp.pad(k, ((0, 0), (0, 0), (0, dpad)), 'constant')
        v = jnp.pad(v, ((0, 0), (0, 0), (0, dpad)), 'constant')
    d_padded = d + dpad

    # Varlen output shapes (different from non-varlen!)
    o_shape = (total_q, h, d_padded)
    lse_shape = (h, total_q)  # 2D for varlen
    oaccum_shape = (num_splits, h, total_q, d_padded)
    lseaccum_shape = (num_splits, h, total_q)
    scheduler_metadata_shape = (metadata_size,)

    out_types = [
        jax.ShapeDtypeStruct(o_shape, dtype),
        jax.ShapeDtypeStruct(lse_shape, jnp.float32),
        jax.ShapeDtypeStruct(oaccum_shape, jnp.float32),
        jax.ShapeDtypeStruct(lseaccum_shape, jnp.float32),
        jax.ShapeDtypeStruct(scheduler_metadata_shape, jnp.int32),
    ]

    # Empty tensors for inapplicable optional arrays (0-dimensional)
    empty_any = jnp.zeros((), dtype=dtype)
    empty_i32 = jnp.zeros((), dtype=jnp.int32)
    empty_f32 = jnp.zeros((), dtype=jnp.float32)

    # 18 inputs, 5 outputs
    o, lse, _, _, _ = jax.ffi.ffi_call(
        "hopper_flash_mha_fwd",
        result_shape_dtypes=out_types,
        has_side_effect=False,
        input_layouts=[None] * 18,
        output_layouts=[None] * 5,
    )(
        # Required inputs (3D for varlen)
        q,
        k,
        v,
        # Optional AnyBuffer inputs (empty)
        empty_any,  # k_new
        empty_any,  # v_new
        empty_any,  # q_v
        # cu_seqlens - THE KEY DIFFERENCE for varlen
        seqlens_q,  # cu_seqlens_q
        seqlens_k,  # cu_seqlens_k
        empty_i32,  # cu_seqlens_k_new
        empty_i32,  # page_table
        empty_i32,  # kv_batch_idx
        empty_i32,  # leftpad_k
        # Optional AnyBuffer inputs (empty)
        empty_any,  # rotary_cos
        empty_any,  # rotary_sin
        # Optional S32 buffer inputs (empty)
        empty_i32,  # seqlens_rotary
        # Optional F32 buffer inputs (empty) - descale arrays
        empty_f32,  # q_descale
        empty_f32,  # k_descale
        empty_f32,  # v_descale
        # Scalar attributes
        max_seqlen_q=np.int64(max_seqlen_q),
        max_seqlen_k=np.int64(max_seqlen_k),
        softmax_scale=np.float64(softmax_scale),
        is_causal=is_causal,
        window_size_left=np.int64(window_size_left),
        window_size_right=np.int64(window_size_right),
        attention_chunk=np.int64(0),
        softcap=np.float64(0.0),
        is_rotary_interleaved=False,
        num_splits=np.int64(num_splits),
        pack_gqa=False,
        sm_margin=np.int64(0),
    )

    if dpad > 0:
        o = o[:, :, :d]

    return o, lse


def _flash_mha_varlen_fwd_hlo_lowering(*args, backend: str, **kwargs):
    if backend == "fa2":
        return _flash_mha_varlen_fwd_hlo_lowering_fa2(*args, **kwargs)
    else:
        return _flash_mha_varlen_fwd_hlo_lowering_fa3(*args, **kwargs)

def _flash_mha_varlen_fwd_hlo_lowering_mlir(ctx, *args, **keywords):
    return mlir.lower_fun(_flash_mha_varlen_fwd_hlo_lowering, multiple_results=True)(
        ctx, *args, **keywords
    )


mlir.register_lowering(
    _flash_mha_varlen_fwd_p,
    _flash_mha_varlen_fwd_hlo_lowering_mlir,  # type: ignore
    platform="gpu",
)

# ==== Abstract Evaluation ====


def _flash_mha_varlen_fwd_abstract(
    q,
    k,
    v,
    seqlens_q,
    seqlens_k,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float],
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    backend: str,
):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [totalq, h, d] = q.shape
    b = seqlens_q.shape[0] - 1
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    assert b >= 1

    out_shape = [totalq, h, d]
    if backend == 'fa2':
        lse_shape = [b, h, max_seqlen_q]
    else:
        lse_shape = [h, totalq]

    return (ShapedArray(out_shape, q_dtype), ShapedArray(lse_shape, jnp.float32))


_flash_mha_varlen_fwd_p.def_abstract_eval(_flash_mha_varlen_fwd_abstract)

# ==== VMap rules ====


def _flash_mha_varlen_fwd_batch(
    vector_arg_values,
    batch_axes,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    backend: str,
    **kwargs,
):
    # move mapping axes to the front
    vector_arg_values, batch_axes = zip(
        *[
            (jnp.moveaxis(x, b, 0), 0) if b is not None else (x, b)
            for x, b in zip(vector_arg_values, batch_axes)
        ]
    )
    q, k, v, seqlens_q, seqlens_k = vector_arg_values
    assert all(isinstance(b, int) or b is None for b in batch_axes)
    if batch_axes == (0, 0, 0, 0, 0):
        b, sq, hq, dq = q.shape
        b, sk, hk, dk = k.shape
        assert dq == dk
        assert k.shape == v.shape
        assert seqlens_q.shape == seqlens_k.shape
        b, n_plus_1 = seqlens_q.shape
        new_q = q.reshape((b * sq, hq, dq))
        new_k = k.reshape((b * sk, hk, dk))
        new_v = v.reshape((b * sk, hk, dk))
        new_seqlens_q = (seqlens_q + (jnp.arange(b)[:, None] * sq)).reshape(
            (b * n_plus_1,)
        )
        new_seqlens_k = (seqlens_k + (jnp.arange(b)[:, None] * sk)).reshape(
            (b * n_plus_1,)
        )
        out, lse = flash_mha_varlen_fwd(
            new_q,
            new_k,
            new_v,
            new_seqlens_q,
            new_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            backend=backend,
            **kwargs,
        )
        # out: (b*sq) hq dq
        new_out = out.reshape((b, sq, hq, dq))
        if backend == 'fa2':
            # lse: (b*n_plus_1-1) hq max_seqlen_q
            new_lse = jnp.pad(lse, ((0, 1), (0, 0), (0, 0))).reshape(
                (b, n_plus_1, hq, max_seqlen_q)
            )[:, :-1]
        elif backend == 'fa3':
            # lse: hq b*sq
            new_lse = einops.rearrange(lse, "hq (b sq) -> b hq sq", b=b,sq=sq)
        return (new_out, new_lse), (0, 0)
    elif batch_axes == (None, 0, 0, None, 0):
        # broadcasting q over k,v with different seqlens_k
        x = k.shape[0]
        q = einops.repeat(q, "... -> x ...", x=x)
        seqlens_q = einops.repeat(seqlens_q, "... -> x ...", x=x)
        out, lse = jax.vmap(
            partial(
                flash_mha_varlen_fwd,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                backend=backend,
                **kwargs,
            )
        )(q, k, v, seqlens_q, seqlens_k)
        return (out, lse), (0, 0)
    elif batch_axes == (0, 0, 0, None, 0):
        # broadcast seqlens_q
        x = q.shape[0]
        seqlens_q = einops.repeat(seqlens_q, "... -> x ...", x=x)
        out, lse = jax.vmap(
            partial(
                flash_mha_varlen_fwd,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                backend=backend,
                **kwargs,
            )
        )(q, k, v, seqlens_q, seqlens_k)
        return (out, lse), (0, 0)
    elif batch_axes == (0, 0, 0, None, None):
        # broadcast seqlens_q & seqlens_k over batches of q/k/v is the same as doing more heads
        x = q.shape[0]
        new_q = rearrange(q, "x sq hq dq -> sq (x hq) dq")
        new_k = rearrange(k, "x sk hk dk -> sk (x hk) dk")
        new_v = rearrange(v, "x sk hk dk -> sk (x hk) dk")
        out, lse = flash_mha_varlen_fwd(
            new_q,
            new_k,
            new_v,
            seqlens_q,
            seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            backend=backend,
            **kwargs,
        )
        new_out = rearrange(out, "sq (x hq) dq -> x sq hq dq", x=x)
        if backend=='fa2':
            new_lse = rearrange(lse, "n (x hq) mseq -> x n hq mseq", x=x)
        elif backend=='fa3':
            new_lse = rearrange(lse, "(x hq) sq -> x hq sq", x=x)
        return (new_out, new_lse), (0, 0)
    else:
        raise NotImplementedError(
            f"flash_mha_varlen_fwd: unsupported vmap: {batch_axes}"
        )


batching.primitive_batchers[_flash_mha_varlen_fwd_p] = _flash_mha_varlen_fwd_batch
