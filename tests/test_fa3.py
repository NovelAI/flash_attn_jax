import sys, glob, os

from functools import partial
import pytest
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import math
import einops

from flash_attn_jax import flash_mha
from .ref_mha import ref_mha

def test_import():
    import flash_attn_jax_lib.flash_hopper_ffi
    print(dir(flash_attn_jax_lib.flash_hopper_ffi))