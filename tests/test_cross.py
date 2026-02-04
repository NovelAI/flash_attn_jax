import pytest

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
from typing import Sequence
from hypothesis import given, settings
from jax.tree_util import tree_map

from flash_attn_jax.test_util import array_summary
from flash_attn_jax import flash_mha

from .ref_mha import ref_mha


def pretty(tensor):
    shape = tensor.shape
    mx = jnp.max(tensor)
    mn = jnp.min(tensor)
    mean = jnp.mean(tensor)
    std = jnp.std(tensor)
    return f'[{shape}: {mn:.3g} | {mean:.3g}±{std:.3g} | {mx:.3g}]'

def check(ref_out, jax_out, out):
    def check1(ref_out, jax_out, out):
        assert jnp.max(jnp.abs(out - ref_out)).item() <= 3 * jnp.max(jnp.abs(jax_out - ref_out)).item(), (pretty(jnp.abs(out - ref_out)), 'vs', pretty(jnp.abs(jax_out - ref_out)))
    tree_map(check1, ref_out, jax_out, out)

def assert_allclose_quantiles(actual, desired, tol: Sequence[tuple[float,float,float]]):
    actual = actual.astype(jnp.float32)
    desired = desired.astype(jnp.float32)
    diff = jnp.abs(actual - desired)
    for (q, rtol, atol) in tol:
        diffq = jnp.quantile(diff, q/100.0, axis=0)
        with open('/tmp/flash_attn_jax_test_debug.txt', 'a') as f:
            f.write(f'Shape {actual.shape} Quantile {q} diff: min={diffq.min()} mean={diffq.mean()} max={diffq.max()}\n')
        desired_size = jnp.mean(jnp.abs(desired), axis=0)
        rdiffq = diffq / jnp.clip(desired_size, a_min=1e-8)
        close = (rdiffq <= rtol) | (diffq <= atol)
        mismatch = ~close
        if jnp.sum(mismatch) > 0:
            raise AssertionError(f'Quantile {q} check failed for rtol={rtol}, atol={atol}. Mismatch summary:\n{array_summary.summarize_bool_array(mismatch)}')

# @pytest.mark.parametrize("d", [59, 32])
# @pytest.mark.parametrize("h", [4])
# @pytest.mark.parametrize("seqlen_q", [32, 97, 128])
# @pytest.mark.parametrize("seqlen_k", [32, 63])
# @pytest.mark.parametrize("n", [1])
# @pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
# @pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@settings(deadline=None)
@given(d=st.integers(min_value=1, max_value=64),
         h=st.integers(min_value=1, max_value=8),
         seqlen_q=st.integers(min_value=1, max_value=128),
         seqlen_k=st.integers(min_value=1, max_value=128),
         n=st.integers(min_value=1, max_value=2),
         m=st.integers(min_value=1, max_value=2),
         is_causal=st.booleans())
def test_cross_fwd(n, seqlen_q, seqlen_k, h, d, m, dtype, is_causal, backend):
    n_samples = 10
    q = jax.random.normal(jax.random.PRNGKey(0), [n_samples, n, seqlen_q, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n_samples, n, seqlen_k, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n_samples, n, seqlen_k, h, d], dtype=jnp.float32)
    ref_out = jax.vmap(lambda q,k,v: ref_mha(q,k,v, is_causal=is_causal))(q,k,v)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    def cmp_flash(_,item):
        (q,k,v) = item
        out = flash_mha(q,k,v, is_causal=is_causal, backend=backend)
        return None, out
    _, out = jax.lax.scan(cmp_flash, None, (q,k,v))
    if dtype == jnp.float16:
        tol = [(50, 1e-3, 1e-3), (99, 5e-3, 5e-3), (100, 5e-2, 5e-2)]
    elif dtype == jnp.bfloat16:
        tol = [(50, 1e-2, 1e-2), (99, 5e-2, 5e-2), (100, 8e-2, 8e-2)]
    assert_allclose_quantiles(out, ref_out, tol=tol)

# @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
# @pytest.mark.parametrize("d", [59, 32])
# @pytest.mark.parametrize("h", [4])
# @pytest.mark.parametrize("seqlen_q", [97, 128])
# @pytest.mark.parametrize("seqlen_k", [32, 63])
# @pytest.mark.parametrize("n", [1])
# @pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
@settings(deadline=None)
@given(d=st.integers(min_value=1, max_value=64),
         h=st.integers(min_value=1, max_value=8),
         seqlen_q=st.integers(min_value=16, max_value=128),
         seqlen_k=st.integers(min_value=16, max_value=128),
         n=st.integers(min_value=1, max_value=2),
         m=st.integers(min_value=1, max_value=2),
         is_causal=st.booleans(),
         dtype=st.sampled_from([jnp.float16, jnp.bfloat16]))
def test_cross_bwd(n, seqlen_q, seqlen_k, h, d, m, dtype, is_causal: bool):
    @jax.grad
    def ref(qkv):
        return ref_mha(*qkv).sum()
    @jax.grad
    def flash(qkv):
        return flash_mha(*qkv).sum()
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen_q, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen_k, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen_k, h, d], dtype=jnp.float32)
    ref_out = ref((q,k,v))
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    jax_out = ref((q,k,v))
    out = flash((q,k,v))
    check(ref_out, jax_out, out)
