import sys, glob, os

from functools import partial
import pytest
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import math
import einops
from hypothesis import given, settings, strategies as st

from flash_attn_jax import flash_mha
from flash_attn_jax.varlen import flash_mha_varlen
from .ref_mha import ref_mha

def test_import():
    import flash_attn_jax_lib.flash_hopper_ffi
    print(dir(flash_attn_jax_lib.flash_hopper_ffi))

@settings(deadline=None)
@given(d=st.integers(min_value=1, max_value=128).filter(lambda d: d % 8 == 0),  # hopper requires d to be a multiple of 8
       h=st.integers(min_value=1, max_value=8),
       seqlen_q=st.integers(min_value=1, max_value=16384),
       seqlen_k=st.integers(min_value=1, max_value=16384),
       n=st.integers(min_value=1, max_value=2),
       is_causal=st.booleans(),
       dtype=st.sampled_from([jnp.float16, jnp.bfloat16]))
def test_fwd(n, seqlen_q, seqlen_k, h, d, dtype, is_causal):
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen_q, h, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen_k, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen_k, h, d], dtype=dtype)
    jax_out = flash_mha(q, k, v, is_causal=is_causal, backend='fa3')
    jax_out.block_until_ready()

@settings(deadline=None)
@given(d=st.integers(min_value=1, max_value=128).filter(lambda d: d % 8 == 0),
       h=st.integers(min_value=1, max_value=8),
       seqlen=st.integers(min_value=1, max_value=16384),
       b=st.integers(min_value=1, max_value=128),
       is_causal=st.booleans(),
       seed=st.integers(min_value=0, max_value=10000),
       dtype=st.sampled_from([jnp.float16, jnp.bfloat16]))
def test_varlen_fwd(b, seqlen, h, d, dtype, is_causal, seed):
    total_seqlen = seqlen
    # generate random splits that sum to total_seqlen
    q_splits = np.random.RandomState(seed).randint(0, total_seqlen, size=b).tolist()
    k_splits = np.random.RandomState(seed+1).randint(0, total_seqlen, size=b).tolist()
    q_splits.sort()
    k_splits.sort()
    q_fenceposts = jnp.asarray([0] + q_splits + [seqlen], dtype=jnp.int32)
    k_fenceposts = jnp.asarray([0] + k_splits + [seqlen], dtype=jnp.int32)
    max_seqlen = seqlen
    q = jax.random.normal(jax.random.PRNGKey(0), [total_seqlen, h, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [total_seqlen, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [total_seqlen, h, d], dtype=dtype)
    jax_out = flash_mha_varlen(q, k, v, seqlens_q=q_fenceposts, seqlens_k=k_fenceposts,
                               max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                               is_causal=is_causal, backend='fa3')
    jax_out.block_until_ready()

NUM_THREADS = 8
NUM_STEPS = 1000

@jax.jit
def flash_mha_jit(q,k,v):
    return flash_mha(q, k, v, is_causal=False, backend='fa3')

def test_thread_safety_fwd():
    """Test that flash_mha can be called from multiple threads simultaneously."""
    d, h, seqlen_q, seqlen_k, n = 64, 4, 512, 512, 1
    dtype = jnp.bfloat16
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen_q, h, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen_k, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen_k, h, d], dtype=dtype)

    # Get reference output from single-threaded call
    ref_out = flash_mha(q, k, v, is_causal=False, backend='fa3')
    ref_out.block_until_ready()

    import threading
    errors = []
    outputs = [None] * NUM_THREADS

    def worker(idx):
        try:
            out = None
            for _ in range(NUM_STEPS):
                out = flash_mha_jit(q, k, v)
            out.block_until_ready()
            outputs[idx] = out
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread(s) raised exceptions: {errors}"
    for i, out in enumerate(outputs):
        assert out is not None, f"Thread {i} produced no output"
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref_out),
                                      err_msg=f"Thread {i} output differs from reference")


def test_thread_safety_varlen_fwd():
    """Test that flash_mha_varlen can be called from multiple threads simultaneously."""
    d, h, total_seqlen, b = 64, 4, 1024, 4
    dtype = jnp.bfloat16

    # Even splits for simplicity
    seqlen_per_batch = total_seqlen // b
    fenceposts = jnp.asarray([i * seqlen_per_batch for i in range(b + 1)], dtype=jnp.int32)
    max_seqlen = seqlen_per_batch

    q = jax.random.normal(jax.random.PRNGKey(0), [total_seqlen, h, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [total_seqlen, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [total_seqlen, h, d], dtype=dtype)

    # Get reference output from single-threaded call
    ref_out = flash_mha_varlen(q, k, v, seqlens_q=fenceposts, seqlens_k=fenceposts,
                                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                                is_causal=False, backend='fa3')
    ref_out.block_until_ready()

    import threading
    errors = []
    outputs = [None] * NUM_THREADS

    def worker(idx):
        try:
            out = flash_mha_varlen(q, k, v, seqlens_q=fenceposts, seqlens_k=fenceposts,
                                    max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                                    is_causal=False, backend='fa3')
            out.block_until_ready()
            outputs[idx] = out
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread(s) raised exceptions: {errors}"
    for i, out in enumerate(outputs):
        assert out is not None, f"Thread {i} produced no output"
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref_out),
                                      err_msg=f"Thread {i} output differs from reference")
