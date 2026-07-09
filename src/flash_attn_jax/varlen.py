import os
from dataclasses import asdict, dataclass
from typing import Optional

import jax
import jax._src.dispatch

from .varlen_bwd import flash_mha_varlen_bwd
from .varlen_fwd import flash_mha_varlen_fwd
from .flash_hlo import has_fa3


@jax.tree_util.register_static
@dataclass
class FlashConfig:
    max_seqlen_q: int
    max_seqlen_k: int
    softmax_scale: Optional[float]
    is_causal: bool
    window_size_left: int
    window_size_right: int
    backend: str


@jax.custom_vjp
def _flash_mha_varlen_vjp(q: jax.Array, k: jax.Array, v: jax.Array, seqlens_q: jax.Array, seqlens_k: jax.Array, config: FlashConfig):
    return flash_mha_varlen_fwd(q,k,v, seqlens_q, seqlens_k, **asdict(config))[0]
def _flash_mha_varlen_vjp_fwd(q,k,v,seqlens_q, seqlens_k, config):
    out, lse = flash_mha_varlen_fwd(q,k,v, seqlens_q, seqlens_k, **asdict(config))
    return out, (q,k,v,seqlens_q, seqlens_k, out,lse, config)
def _flash_mha_varlen_vjp_bwd(pack, dout):
    (q,k,v,seqlens_q, seqlens_k, out,lse, config) = pack
    dq, dk, dv = flash_mha_varlen_bwd(dout, q, k, v, out, lse, seqlens_q, seqlens_k, **asdict(config))
    return (dq,dk,dv,None,None,None)
_flash_mha_varlen_vjp.defvjp(_flash_mha_varlen_vjp_fwd, _flash_mha_varlen_vjp_bwd)

def flash_mha_varlen(q, k, v, seqlens_q, seqlens_k=None, *,
                     max_seqlen_q: int = -1, max_seqlen_k: int = -1,
                     softmax_scale: Optional[float] = None, is_causal: bool = False,
                     window_size: tuple = (-1, -1), backend: Optional[str] = None):
    if backend is None:
        backend = os.environ.get("FLASH_ATTN_JAX_BACKEND", "fa2")
    if backend == "fa3" and not has_fa3():
        raise RuntimeError("FA3 backend requested but flash_hopper_ffi was not built; "
                           "rebuild with an FA3-capable arch (90a and/or 80/86/89/120/121) in FLASH_ATTN_CUDA_ARCHS")
    if seqlens_k is None:
        seqlens_k = seqlens_q
    config = FlashConfig(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        backend=backend,
    )
    return _flash_mha_varlen_vjp(q, k, v, seqlens_q, seqlens_k, config)