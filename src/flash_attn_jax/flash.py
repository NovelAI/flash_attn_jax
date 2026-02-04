from jax.extend.mlir.dialects.func import NoneType
from dataclasses import dataclass, asdict, replace
from functools import partial, wraps

import numpy as np
import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.extend.core import Primitive

from einops import rearrange
import einops
import math

from .flash_fwd import flash_mha_fwd
from .flash_bwd import flash_mha_bwd



# ==== VJP Rule ====

def custom_vjp_class(cls, nondiff_argnums=()):
    f = jax.custom_vjp(cls.base, nondiff_argnums=nondiff_argnums)
    f.defvjp(cls.fwd, cls.bwd)
    return f

@jax.tree_util.register_static
@dataclass(frozen=True)
class FlashConfig:
    softmax_scale: float | None = None
    is_causal: bool = False
    window_size_left: int = -1
    window_size_right: int = -1
    backend: str = "fa2"
    softcap: float | None = 0.0

# Don't need nondiff_argnums if we use a static pytree.
@jax.custom_vjp
def _flash_mha_vjp(q,k,v,config):
    return flash_mha_fwd(q,k,v, **asdict(config))[0]
def _flash_mha_vjp_fwd(q,k,v,config):
    out, lse = flash_mha_fwd(q,k,v, **asdict(config))
    return out, (q,k,v,out,lse,config)
def _flash_mha_vjp_bwd(pack, dout):
    (q,k,v,out,lse,fwd_config) = pack
    # Force FA2 for backward pass since FA3 doesn't have backward yet
    bwd_config = replace(fwd_config, backend='fa2')
    dq, dk, dv = flash_mha_bwd(dout, q, k, v, out, lse, **asdict(bwd_config))
    return (dq,dk,dv, None)
_flash_mha_vjp.defvjp(_flash_mha_vjp_fwd, _flash_mha_vjp_bwd)

# ==== Frontend ====

def flash_mha(q, k, v, softmax_scale=None, is_causal=False, window_size=(-1,-1), backend="fa2", softcap=0.0):
    """Flash attention.

    Args:
        q: Query tensor [batch, seqlen_q, num_heads_q, head_dim]
        k: Key tensor [batch, seqlen_k, num_heads_k, head_dim]
        v: Value tensor [batch, seqlen_k, num_heads_k, head_dim]
        softmax_scale: Softmax scaling factor (default: 1/sqrt(head_dim)). Must be a python float if provided.
        is_causal: Whether to apply causal masking
        window_size: Tuple of (left, right) window sizes for sliding window attention
        backend: Backend implementation - "fa2" (default) or "fa3" (Hopper-optimized)
        softcap: Softcap value for attention scores (FA3 only, ignored by FA2)

    Returns:
        Attention output tensor with same shape as q

    Notes:
        - FA3 backend is optimized for Hopper (SM90) GPUs but also supports Ampere (SM80+)
        - FA3 currently only supports forward pass. Gradients automatically use FA2 backward.
        - Softcap parameter is only used with FA3 backend
    """
    [nq, sq, hq, dq] = q.shape
    [nk, sk, hk, dk] = k.shape
    [nv, sv, hv, dv] = v.shape
    assert nq == nk == nv
    assert hk == hv
    assert nq % nk == 0 # Can be larger than nk if GQA
    assert dq == dk == dv # Don't support head size mismatch
    assert sk == sv
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in [jnp.bfloat16, jnp.float16]

    window_size_left, window_size_right = window_size
    config = FlashConfig(
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        backend=backend,
        softcap=softcap,
    )
    return _flash_mha_vjp(q, k, v, config)