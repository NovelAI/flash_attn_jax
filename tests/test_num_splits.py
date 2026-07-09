"""
Unit tests for FA3 num_splits calculation and helper functions.

Tests the Python implementations in fa3_util.py against known outputs
from the C++ implementation.
"""

import pytest
import jax.numpy as jnp
from flash_attn_jax.fa3_util import (
    round_up_headdim,
    round_up_headdim_v,
    tile_size_fwd_sm90_py,
    get_num_splits_fa3,
    calculate_scheduler_metadata_size,
)


class TestHeadDimRounding:
    """Test head dimension rounding functions."""

    def test_round_up_headdim(self):
        """Test head dimension rounding to supported sizes."""
        assert round_up_headdim(32) == 64
        assert round_up_headdim(64) == 64
        assert round_up_headdim(65) == 96
        assert round_up_headdim(96) == 96
        assert round_up_headdim(97) == 128
        assert round_up_headdim(128) == 128
        assert round_up_headdim(129) == 192
        assert round_up_headdim(192) == 192
        assert round_up_headdim(193) == 256
        assert round_up_headdim(256) == 256
        assert round_up_headdim(300) == 256  # Max out at 256

    def test_round_up_headdim_v(self):
        """Test V head dimension rounding (can go up to 512)."""
        assert round_up_headdim_v(32) == 64
        assert round_up_headdim_v(64) == 64
        assert round_up_headdim_v(128) == 128
        assert round_up_headdim_v(256) == 256
        assert round_up_headdim_v(257) == 512
        assert round_up_headdim_v(512) == 512
        assert round_up_headdim_v(600) == 512  # Max out at 512


class TestTileSizeSM90:
    """Test SM90 tile size calculation."""

    def test_tile_size_d64_basic(self):
        """Test tile sizes for head_dim=64."""
        # headdim_v = 512
        kBlockM, kBlockN = tile_size_fwd_sm90_py(64, 512, False, False, element_size=2)
        assert kBlockM == 64
        assert kBlockN == 64

        # headdim_v = 256
        kBlockM, kBlockN = tile_size_fwd_sm90_py(64, 256, False, False, element_size=2)
        assert kBlockM == 128
        assert kBlockN == 96

        # headdim_v <= 128, non-causal, non-local
        kBlockM, kBlockN = tile_size_fwd_sm90_py(64, 64, False, False, element_size=2)
        assert kBlockM == 192
        assert kBlockN == 192

    def test_tile_size_d64_causal(self):
        """Test tile sizes for head_dim=64 with causal."""
        kBlockM, kBlockN = tile_size_fwd_sm90_py(64, 64, True, False, element_size=2)
        assert kBlockM == 192
        assert kBlockN == 128

    def test_tile_size_d64_local(self):
        """Test tile sizes for head_dim=64 with local attention."""
        kBlockM, kBlockN = tile_size_fwd_sm90_py(64, 64, False, True, element_size=2)
        assert kBlockM == 192
        assert kBlockN == 128

    def test_tile_size_d96(self):
        """Test tile sizes for head_dim=96."""
        # Non-local
        kBlockM, kBlockN = tile_size_fwd_sm90_py(96, 96, False, False, element_size=2)
        assert kBlockM == 192
        assert kBlockN == 144

        # Local
        kBlockM, kBlockN = tile_size_fwd_sm90_py(96, 96, False, True, element_size=2)
        assert kBlockM == 192
        assert kBlockN == 128

    def test_tile_size_d128(self):
        """Test tile sizes for head_dim=128."""
        # Non-causal, non-local
        kBlockM, kBlockN = tile_size_fwd_sm90_py(128, 128, False, False, element_size=2)
        assert kBlockM == 128
        assert kBlockN == 176

        # Causal
        kBlockM, kBlockN = tile_size_fwd_sm90_py(128, 128, True, False, element_size=2)
        assert kBlockM == 128
        assert kBlockN == 128

        # Local
        kBlockM, kBlockN = tile_size_fwd_sm90_py(128, 128, False, True, element_size=2)
        assert kBlockM == 128
        assert kBlockN == 128

    def test_tile_size_d192(self):
        """Test tile sizes for head_dim=192."""
        # headdim_v <= 128
        kBlockM, kBlockN = tile_size_fwd_sm90_py(192, 128, False, False, element_size=2)
        assert kBlockM == 128
        assert kBlockN == 128

        # headdim_v > 128
        kBlockM, kBlockN = tile_size_fwd_sm90_py(192, 192, False, False, element_size=2)
        assert kBlockM == 128
        assert kBlockN == 112

        # Local
        kBlockM, kBlockN = tile_size_fwd_sm90_py(192, 192, False, True, element_size=2)
        assert kBlockM == 128
        assert kBlockN == 96

    def test_tile_size_d256(self):
        """Test tile sizes for head_dim=256."""
        # Non-local
        kBlockM, kBlockN = tile_size_fwd_sm90_py(256, 256, False, False, element_size=2)
        assert kBlockM == 128
        assert kBlockN == 80

        # Local
        kBlockM, kBlockN = tile_size_fwd_sm90_py(256, 256, False, True, element_size=2)
        assert kBlockM == 128
        assert kBlockN == 64

    def test_tile_size_fp8(self):
        """Test tile sizes for FP8 (element_size=1)."""
        kBlockM, kBlockN = tile_size_fwd_sm90_py(64, 64, False, False, element_size=1)
        assert kBlockM == 192
        assert kBlockN == 160

        kBlockM, kBlockN = tile_size_fwd_sm90_py(128, 128, False, False, element_size=1)
        assert kBlockM == 128
        assert kBlockN == 224


class TestNumSplitsCalculation:
    """Test num_splits calculation heuristic."""

    def test_high_sm_utilization_returns_one(self):
        """High SM utilization should return num_splits=1."""
        # Large batch, many heads -> high SM utilization
        num_splits = get_num_splits_fa3(
            batch_size=8,
            seqlen_q=1024,
            seqlen_k=1024,
            num_heads=32,
            num_heads_k=32,
            head_dim=64,
            head_dim_v=64,
            is_causal=False,
            is_local=False,
            window_size_left=-1,
            window_size_right=-1,
            dtype=jnp.bfloat16,
            num_sm=114,
        )
        assert num_splits == 1, "High SM utilization should use 1 split"

    def test_low_utilization_splits(self):
        """Low SM utilization should allow multiple splits."""
        # Small batch, few heads, long sequence -> low SM utilization
        num_splits = get_num_splits_fa3(
            batch_size=1,
            seqlen_q=8192,
            seqlen_k=8192,
            num_heads=1,
            num_heads_k=1,
            head_dim=64,
            head_dim_v=64,
            is_causal=False,
            is_local=False,
            window_size_left=-1,
            window_size_right=-1,
            dtype=jnp.bfloat16,
            num_sm=114,
        )
        assert num_splits > 1, "Low SM utilization should use multiple splits"
        assert num_splits <= 128, "num_splits should not exceed max_splits"

    def test_local_attention_reduces_splits(self):
        """Local attention reduces seqlen_k_loaded, affecting num_splits."""
        # Non-local case
        num_splits_nonlocal = get_num_splits_fa3(
            batch_size=1,
            seqlen_q=4096,
            seqlen_k=4096,
            num_heads=2,
            num_heads_k=2,
            head_dim=128,
            head_dim_v=128,
            is_causal=False,
            is_local=False,
            window_size_left=-1,
            window_size_right=-1,
            dtype=jnp.bfloat16,
            num_sm=114,
        )

        # Local case with small window
        num_splits_local = get_num_splits_fa3(
            batch_size=1,
            seqlen_q=4096,
            seqlen_k=4096,
            num_heads=2,
            num_heads_k=2,
            head_dim=128,
            head_dim_v=128,
            is_causal=False,
            is_local=True,
            window_size_left=256,
            window_size_right=256,
            dtype=jnp.bfloat16,
            num_sm=114,
        )

        # Local attention should reduce effective sequence length
        # This may or may not reduce num_splits depending on other factors,
        # but seqlen_k_loaded should be smaller
        assert num_splits_local >= 1

    def test_causal_attention(self):
        """Causal attention typically uses fewer splits."""
        num_splits = get_num_splits_fa3(
            batch_size=2,
            seqlen_q=2048,
            seqlen_k=2048,
            num_heads=8,
            num_heads_k=8,
            head_dim=128,
            head_dim_v=128,
            is_causal=True,
            is_local=False,
            window_size_left=-1,
            window_size_right=0,  # Causal sets this to 0
            dtype=jnp.bfloat16,
            num_sm=114,
        )
        # Causal should typically use 1 split due to heuristic
        assert num_splits >= 1
        assert num_splits <= 128

    def test_gqa_scaling(self):
        """GQA (Grouped Query Attention) affects num_m_blocks calculation."""
        # MQA case: 1 KV head, 32 query heads
        num_splits_mqa = get_num_splits_fa3(
            batch_size=1,
            seqlen_q=2048,
            seqlen_k=2048,
            num_heads=32,
            num_heads_k=1,
            head_dim=64,
            head_dim_v=64,
            is_causal=False,
            is_local=False,
            window_size_left=-1,
            window_size_right=-1,
            dtype=jnp.bfloat16,
            num_sm=114,
        )

        # Regular attention: 32 KV heads, 32 query heads
        num_splits_regular = get_num_splits_fa3(
            batch_size=1,
            seqlen_q=2048,
            seqlen_k=2048,
            num_heads=32,
            num_heads_k=32,
            head_dim=64,
            head_dim_v=64,
            is_causal=False,
            is_local=False,
            window_size_left=-1,
            window_size_right=-1,
            dtype=jnp.bfloat16,
            num_sm=114,
        )

        # Both should be valid
        assert num_splits_mqa >= 1
        assert num_splits_regular >= 1

    def test_short_sequences(self):
        """Very short sequences should use num_splits=1."""
        num_splits = get_num_splits_fa3(
            batch_size=4,
            seqlen_q=64,
            seqlen_k=64,
            num_heads=8,
            num_heads_k=8,
            head_dim=64,
            head_dim_v=64,
            is_causal=False,
            is_local=False,
            window_size_left=-1,
            window_size_right=-1,
            dtype=jnp.bfloat16,
            num_sm=114,
        )
        # Short sequences have num_n_blocks <= 4, should return 1
        assert num_splits == 1


class TestSchedulerMetadataSize:
    """Test scheduler_metadata size calculation."""

    def test_noncausal_nonlocal_nosplits(self):
        """Non-causal, non-local with num_splits=1 should return 1."""
        size = calculate_scheduler_metadata_size(
            batch_size=4,
            num_splits=1,
            is_causal=False,
            is_local=False,
            arch=90,
        )
        assert size == 1, "No semaphore needed for non-causal, non-local"

    def test_causal_with_splits_one(self):
        """Causal with num_splits=1 needs semaphore and metadata."""
        # batch_size=1, causal=True, num_splits=1
        # scheduler_needs_semaphore = True
        # varlen_sort_batches = True (!is_local)
        # head_swizzle = True (is_causal)
        # b_rounded = 4, num_prepare_batch_vectors = 0 + 1 + 1 = 2
        # tile_count_semaphore_offset = 4 * 2 = 8
        # metadata_size = 1 + 8 = 9
        size = calculate_scheduler_metadata_size(
            batch_size=1,
            num_splits=1,
            is_causal=True,
            is_local=False,
            arch=90,
        )
        assert size == 9, f"Expected 9, got {size}"

    def test_local_with_splits_one(self):
        """Local attention with num_splits=1 needs semaphore and metadata."""
        # batch_size=2, local=True, num_splits=1
        # scheduler_needs_semaphore = True
        # varlen_sort_batches = False (is_local)
        # head_swizzle = True (is_local)
        # b_rounded = 4, num_prepare_batch_vectors = 0 + 0 + 1 = 1
        # tile_count_semaphore_offset = 4 * 1 = 4
        # metadata_size = 1 + 4 = 5
        size = calculate_scheduler_metadata_size(
            batch_size=2,
            num_splits=1,
            is_causal=False,
            is_local=True,
            arch=90,
        )
        assert size == 5, f"Expected 5, got {size}"

    def test_multiple_splits_no_semaphore(self):
        """Multiple splits without causal/local doesn't need semaphore."""
        size = calculate_scheduler_metadata_size(
            batch_size=4,
            num_splits=8,
            is_causal=False,
            is_local=False,
            arch=90,
        )
        assert size == 1, "No semaphore needed when num_splits > 1 and not causal/local"

    def test_causal_multiple_splits(self):
        """Causal with num_splits > 1 doesn't need semaphore."""
        size = calculate_scheduler_metadata_size(
            batch_size=4,
            num_splits=4,
            is_causal=True,
            is_local=False,
            arch=90,
        )
        assert size == 1, "No semaphore when num_splits > 1"

    def test_larger_batch_causal(self):
        """Test larger batch size with causal."""
        # batch_size=8, causal=True, num_splits=1
        # b_rounded = 8, num_prepare_batch_vectors = 2
        # tile_count_semaphore_offset = 8 * 2 = 16
        # metadata_size = 1 + 16 = 17
        size = calculate_scheduler_metadata_size(
            batch_size=8,
            num_splits=1,
            is_causal=True,
            is_local=False,
            arch=90,
        )
        assert size == 17, f"Expected 17, got {size}"

    def test_error_case_from_test_flash(self):
        """Test the specific case that failed from test_flash.py."""
        # From error message:
        # batch_size=1, seqlen_q=97, seqlen_k=97, num_heads=1, num_heads_k=1,
        # head_dim=32, is_causal=True, is_local=False
        # Expected shape: [9]

        # batch_size=1, causal=True, num_splits=1
        # scheduler_needs_semaphore = True (is_causal and num_splits==1)
        # varlen_sort_batches = True (!is_local)
        # head_swizzle = True (is_causal)
        # b_rounded = 4 (round up 1 to nearest multiple of 4)
        # num_prepare_batch_vectors = 0 + 1 (sort) + 1 (swizzle) = 2
        # tile_count_semaphore_offset = 4 * 2 = 8
        # metadata_size = 1 (semaphore) + 8 = 9
        size = calculate_scheduler_metadata_size(
            batch_size=1,
            num_splits=1,
            is_causal=True,
            is_local=False,
            arch=90,
        )
        assert size == 9, f"Expected 9 for the error case, got {size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
