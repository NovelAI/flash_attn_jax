"""
FA3 (Hopper) specific utility functions for num_splits calculation and tile sizing.

This module contains Python ports of C++ functions from:
- csrc/hopper/gpu/tile_size.h
- csrc/hopper/gpu/heuristics.h
- csrc/hopper/mha_fwd_ffi.cpp

These functions are used to calculate optimal num_splits and scheduler metadata sizes
for Flash Attention 3 on Hopper architecture (SM90) GPUs.
"""

import math
import jax.numpy as jnp


def round_up_headdim(head_size: int) -> int:
    """
    Round head dimension to next supported size for Flash Attention 3.

    Supported sizes: 64, 96, 128, 192, 256

    Args:
        head_size: Actual head dimension

    Returns:
        Rounded head dimension
    """
    if head_size <= 64:
        return 64
    if head_size <= 96:
        return 96
    if head_size <= 128:
        return 128
    if head_size <= 192:
        return 192
    if head_size <= 256:
        return 256
    return 256


def round_up_headdim_v(head_size_v: int) -> int:
    """
    Round V head dimension to next supported size.

    V head dimension can go up to 512 on Hopper for special cases.
    Supported sizes: 64, 96, 128, 192, 256, 512

    Args:
        head_size_v: Actual V head dimension

    Returns:
        Rounded V head dimension
    """
    if head_size_v <= 64:
        return 64
    if head_size_v <= 96:
        return 96
    if head_size_v <= 128:
        return 128
    if head_size_v <= 192:
        return 192
    if head_size_v <= 256:
        return 256
    return 512


def tile_size_fwd_sm90_py(
    headdim: int,
    headdim_v: int,
    is_causal: bool,
    is_local: bool,
    element_size: int = 2,
    v_colmajor: bool = False,
    paged_kv_non_tma: bool = False,
    softcap: bool = False,
) -> tuple[int, int]:
    """
    Calculate SM90 (Hopper) tile sizes for Flash Attention 3 forward pass.

    This is a Python port of tile_size_fwd_sm90 from csrc/hopper/gpu/tile_size.h.
    Returns the optimal (kBlockM, kBlockN) tile sizes for the given configuration.

    Args:
        headdim: Rounded head dimension (64, 96, 128, 192, or 256)
        headdim_v: Rounded V head dimension (64, 96, 128, 192, 256, or 512)
        is_causal: Whether causal masking is enabled
        is_local: Whether local/window attention is enabled
        element_size: 2 for fp16/bf16, 1 for fp8_e4m3
        v_colmajor: Whether V tensor is column-major (default False)
        paged_kv_non_tma: Whether using paged KV without TMA (default False)
        softcap: Whether softcap is enabled (default False)

    Returns:
        (kBlockM, kBlockN): Tile sizes for M and N dimensions
    """
    if element_size == 2:  # fp16/bf16
        if headdim <= 64:
            if headdim_v == 512:
                return (64, 64)
            elif headdim_v == 256:
                return (128, 96)
            else:
                # headdim_v <= 128
                use_blockN_128 = is_causal or is_local or paged_kv_non_tma
                return (192, 128 if use_blockN_128 else 192)
        elif headdim <= 96:
            return (192, 128 if (is_local or paged_kv_non_tma) else 144)
        elif headdim <= 128:
            use_blockN_128 = is_causal or is_local or paged_kv_non_tma
            return (128, 128 if use_blockN_128 else 176)
        elif headdim <= 192:
            if paged_kv_non_tma or is_local:
                kBlockN = 96
            elif headdim_v <= 128:
                kBlockN = 128
            else:
                kBlockN = 112
            return (128, kBlockN)
        else:  # headdim <= 256
            return (128, 64 if is_local else 80)
    else:  # element_size == 1 (fp8)
        if headdim <= 64:
            return (192, 160)
        elif headdim <= 96:
            return (192, 128)
        elif headdim <= 128:
            if paged_kv_non_tma:
                kBlockN = 160
            elif v_colmajor or (softcap and is_local):
                kBlockN = 192
            else:
                kBlockN = 224
            return (128, kBlockN)
        elif headdim <= 192:
            kBlockN = 128 if ((paged_kv_non_tma or softcap) and is_local) else 160
            return (128, kBlockN)
        else:  # headdim <= 256
            return (128, 64 if is_local else 128)


def tile_size_fwd_sm8x_py(
    reduced_smem: bool,
    headdim: int,
    headdim_v: int,
    is_causal: bool,
    is_local: bool,
    element_size: int = 2,
    paged_kv: bool = False,
    varlen_and_split: bool = False,
    softcap: bool = False,
    append_kv: bool = False,
) -> tuple[int, int]:
    """
    Calculate sm80-family tile sizes for Flash Attention 3 forward pass.

    Python port of tile_size_fwd_sm8x from csrc/hopper/gpu/tile_size.h (kBlockM/kBlockN
    only). These kernels cover Ampere/Ada and consumer Blackwell.

    Args:
        reduced_smem: True on archs with ~100KB smem (86/89/120/121), matching the
            sm86_or_89 flag in C++ (see reduced_smem in mha_fwd_ffi.cpp get_num_splits)
        headdim: Rounded head dimension (64, 96, 128, 192, or 256)
        headdim_v: Rounded V head dimension
        is_causal: Whether causal masking is enabled
        is_local: Whether local/window attention is enabled
        element_size: 2 for fp16/bf16, 1 for fp8_e4m3
        paged_kv: Whether using paged KV
        varlen_and_split: Whether varlen with split-KV
        softcap: Whether softcap is enabled (unused for kBlockM/kBlockN)
        append_kv: Whether appending new KV (k_new)

    Returns:
        (kBlockM, kBlockN): Tile sizes for M and N dimensions
    """
    if element_size == 2:  # fp16/bf16
        if headdim <= 64:
            return (128, 80 if varlen_and_split else (96 if is_local else 112))
        elif headdim <= 96:
            return (128, 48 if (varlen_and_split or is_local) else 64)
        elif headdim <= 128:
            use_8_warps = reduced_smem or varlen_and_split
            if use_8_warps:
                if varlen_and_split:
                    kBlockN = 96 if is_local else 112
                else:
                    kBlockN = 96 if is_local else 128
            else:
                kBlockN = 48 if is_local else 64
            return (128, kBlockN)
        elif headdim <= 192:
            kBlockN_64 = append_kv or is_local or varlen_and_split or paged_kv
            return (128, 64 if kBlockN_64 else 96)
        else:  # headdim <= 256
            if reduced_smem:
                if append_kv:
                    kBlockN = 32
                else:
                    kBlockN = 48 if (varlen_and_split or is_local) else 64
            else:
                if append_kv:
                    kBlockN = 48
                else:
                    kBlockN = 64 if (varlen_and_split or is_local) else 96
            return (128, kBlockN)
    else:  # element_size == 1 (fp8) - placeholder in C++ too
        return (128, 64)


def is_reduced_smem_arch(arch: int) -> bool:
    """Archs with ~100KB smem per SM (Ada, consumer Blackwell) that need reduced sm8x tiles.

    Must match the reduced_smem condition in get_num_splits (mha_fwd_ffi.cpp) and
    ARCH_SWITCH (static_switch.h)."""
    return arch == 86 or arch == 89 or arch >= 120


def tile_size_fwd_py(
    arch: int,
    headdim: int,
    headdim_v: int,
    is_causal: bool,
    is_local: bool,
    element_size: int = 2,
    paged_kv_non_tma: bool = False,
    softcap: bool = False,
    varlen_and_split: bool = False,
) -> tuple[int, int]:
    """Arch-dispatched (kBlockM, kBlockN), mirroring get_num_splits in mha_fwd_ffi.cpp."""
    if arch == 90:
        return tile_size_fwd_sm90_py(
            headdim, headdim_v, is_causal, is_local, element_size,
            paged_kv_non_tma=paged_kv_non_tma, softcap=softcap,
        )
    else:
        return tile_size_fwd_sm8x_py(
            is_reduced_smem_arch(arch), headdim, headdim_v, is_causal, is_local,
            element_size, paged_kv=paged_kv_non_tma, varlen_and_split=varlen_and_split,
            softcap=softcap,
        )


def num_splits_heuristic_extended(
    total_mblocks: int,
    num_sm: int,
    num_n_blocks: int,
    num_m_blocks: int,
    size_one_kv_head: int,
    is_causal_or_local: bool,
    max_splits: int,
) -> int:
    """
    Extended num_splits heuristic for Flash Attention 3.

    This is a Python port of num_splits_heuristic from csrc/hopper/gpu/heuristics.h
    with additional L2 cache considerations for very long sequences.

    The heuristic finds the number of splits that maximizes SM occupancy while
    minimizing HBM reads/writes. It returns the smallest num_splits that achieves
    at least 85% of the best efficiency.

    Args:
        total_mblocks: Total number of M blocks (batch * num_heads_k * num_m_blocks)
        num_sm: Number of streaming multiprocessors (114 for H100)
        num_n_blocks: Number of blocks in K/V dimension
        num_m_blocks: Number of blocks in Q dimension
        size_one_kv_head: Size of one KV head in bytes
        is_causal_or_local: Whether using causal or local attention
        max_splits: Maximum allowed splits (typically 128)

    Returns:
        Optimal number of splits (1 if no splitting beneficial)
    """
    # If we have enough work to almost fill the SMs, use 1 split
    # Exception: super long sequences where KV doesn't fit in L2 cache
    if total_mblocks >= 0.8 * num_sm:
        # L2 cache size on H100 is 50MB
        size_l2 = 50 * 1024 * 1024
        # Only split if:
        # 1. KV head size > L2 cache
        # 2. Enough queries to go over KV at least twice
        # 3. Not causal or local (those don't benefit from splitting)
        if (size_one_kv_head > size_l2 and
            num_m_blocks >= num_sm * 2 and
            not is_causal_or_local):
            return min((size_one_kv_head + size_l2 - 1) // size_l2, max_splits)
        else:
            return 1

    # If num_n_blocks is too small, don't split
    # Example: hdim=128, seqlen_k=512 -> num_n_blocks ~3-4
    if num_n_blocks <= 4:
        return 1

    # Find optimal num_splits by maximizing SM efficiency
    max_splits = min(max_splits, num_sm, num_n_blocks)

    max_efficiency = 0.0
    efficiencies = []

    for num_splits in range(1, max_splits + 1):
        # Calculate wave efficiency
        n_waves = (total_mblocks * num_splits) / num_sm
        eff = n_waves / math.ceil(n_waves)
        efficiencies.append(eff)
        max_efficiency = max(max_efficiency, eff)

    # Return smallest num_splits achieving 85% of max efficiency
    for num_splits in range(1, max_splits + 1):
        if efficiencies[num_splits - 1] >= 0.85 * max_efficiency:
            return num_splits

    return 1


def get_num_splits_fa3(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    head_dim: int,
    head_dim_v: int,
    is_causal: bool,
    is_local: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: jnp.dtype,
    num_sm: int = 114,
    max_splits: int = 128,
    arch: int = 90,
    varlen: bool = False,
) -> int:
    """
    Calculate optimal num_splits for FA3.

    This is a Python port of get_num_splits from csrc/hopper/mha_fwd_ffi.cpp.
    It determines whether to use split-KV attention based on SM utilization,
    sequence lengths, and memory constraints.

    Args:
        batch_size: Number of sequences in batch
        seqlen_q: Query sequence length
        seqlen_k: Key/Value sequence length
        num_heads: Number of query heads
        num_heads_k: Number of key/value heads (for GQA/MQA)
        head_dim: Head dimension (actual, not rounded)
        head_dim_v: V head dimension (actual, not rounded)
        is_causal: Whether using causal masking
        is_local: Whether using local/window attention
        window_size_left: Left window size (-1 for infinite)
        window_size_right: Right window size (-1 for infinite)
        dtype: Data type (fp16/bf16/fp8_e4m3fn)
        num_sm: Number of SMs on the device (use util.get_sm_count())
        max_splits: Maximum allowed splits (default 128)
        arch: Device compute capability as major*10+minor (use util.get_compute_capability())
        varlen: Whether this is the varlen path (affects sm8x tile sizes)

    Returns:
        num_splits: Number of splits to use (1 if no splitting beneficial)
    """
    # Round head dimensions to supported sizes
    d_rounded = round_up_headdim(head_dim)
    dv_rounded = round_up_headdim_v(head_dim_v)

    # Determine element size (bytes per element)
    element_size = 1 if dtype == jnp.float8_e4m3fn else 2

    # Get tile sizes for this configuration and arch
    # Note: For JAX implementation:
    # - paged_kv_non_tma=False (no paging support yet)
    # - softcap=False (TODO: add softcap support if needed)
    kBlockM, kBlockN = tile_size_fwd_py(
        arch=arch,
        headdim=d_rounded,
        headdim_v=dv_rounded,
        is_causal=is_causal,
        is_local=is_local,
        element_size=element_size,
        paged_kv_non_tma=False,
        softcap=False,
        varlen_and_split=varlen,
    )

    # Calculate effective sequence length loaded for local attention
    # For local attention, we only load a window of K/V
    if is_local:
        seqlen_k_loaded = max(
            0,
            min(seqlen_k, window_size_right + window_size_left + 1 + kBlockM)
        )
    else:
        seqlen_k_loaded = seqlen_k

    # Calculate number of blocks in each dimension
    num_n_blocks = (seqlen_k_loaded + kBlockN - 1) // kBlockN

    # For PackGQA (Grouped Query Attention), seqlen_q is effectively
    # multiplied by the number of query heads per KV head
    qhead_per_khead = num_heads // num_heads_k
    seqlen_q_packgqa = seqlen_q * qhead_per_khead
    num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) // kBlockM

    # Calculate size of one KV head in bytes
    # Each position has (head_dim + head_dim_v) elements
    size_one_kv_head = seqlen_k * (head_dim + head_dim_v) * element_size

    # Total M blocks across batch and KV heads
    # For non-varlen: batch_size * num_heads_k * num_m_blocks
    total_mblocks = batch_size * num_heads_k * num_m_blocks

    # Use extended heuristic to determine optimal num_splits
    return num_splits_heuristic_extended(
        total_mblocks=total_mblocks,
        num_sm=num_sm,
        num_n_blocks=num_n_blocks,
        num_m_blocks=num_m_blocks,
        size_one_kv_head=size_one_kv_head,
        is_causal_or_local=is_causal or is_local,
        max_splits=max_splits,
    )


def calculate_scheduler_metadata_size(
    batch_size: int,
    num_splits: int,
    is_causal: bool,
    is_local: bool,
    arch: int = 90,
) -> int:
    """
    Calculate scheduler_metadata buffer size for FA3.

    Based on logic from csrc/hopper/mha_fwd_ffi.cpp:664-676.

    Args:
        batch_size: Number of sequences in batch
        num_splits: Number of splits being used
        is_causal: Whether using causal masking
        is_local: Whether using local/window attention
        arch: GPU architecture (90 for Hopper, default)

    Returns:
        metadata_size: Size of int32 buffer needed
    """
    # For non-varlen case (current JAX implementation)
    use_prepare_varlen = False
    is_varlen = False

    # Determine if scheduler needs semaphore
    # From mha_fwd_ffi.cpp:664-666
    if arch == 90:
        scheduler_needs_semaphore = ((is_causal or is_local) and (num_splits == 1)) or is_varlen
    else:
        scheduler_needs_semaphore = (is_causal and not is_varlen) or (is_varlen and num_splits > 1)

    # If we need metadata allocation
    # From mha_fwd_ffi.cpp:669-676
    if scheduler_needs_semaphore or use_prepare_varlen:
        # Round batch size to multiple of 4 for alignment
        b_rounded = ((batch_size + 3) // 4) * 4

        # Calculate number of prepare batch vectors
        # From mha_fwd_ffi.cpp:671-673
        num_prepare_batch_vectors = 2 if use_prepare_varlen else 0

        # varlen_sort_batches = !is_local (line 667)
        varlen_sort_batches = not is_local
        if varlen_sort_batches:
            num_prepare_batch_vectors += 1

        # head_swizzle = is_causal || is_local (line 668)
        head_swizzle = is_causal or is_local
        if head_swizzle:
            num_prepare_batch_vectors += 1

        # Calculate final size
        # From mha_fwd_ffi.cpp:675-676
        tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors
        metadata_size = int(scheduler_needs_semaphore) + tile_count_semaphore_offset

        return metadata_size
    else:
        # No metadata needed (but we still allocate size 1 minimum)
        return 1


def calculate_scheduler_metadata_size_varlen(
    batch_size: int,
    num_splits: int,
    is_causal: bool,
    is_local: bool,
    arch: int = 90,
) -> int:
    """
    Calculate scheduler_metadata buffer size for FA3 varlen case.

    For varlen, use_prepare_varlen=True and is_varlen=True.
    Based on logic from csrc/hopper/mha_fwd_ffi.cpp:664-676.

    Args:
        batch_size: Number of sequences in batch
        num_splits: Number of splits being used
        is_causal: Whether using causal masking
        is_local: Whether using local/window attention
        arch: GPU architecture (90 for Hopper, default)

    Returns:
        metadata_size: Size of int32 buffer needed
    """
    is_varlen = True

    # Determine if scheduler needs semaphore (line 664-666)
    if arch == 90:
        scheduler_needs_semaphore = ((is_causal or is_local) and (num_splits == 1)) or is_varlen
    else:
        scheduler_needs_semaphore = (is_causal and not is_varlen) or (is_varlen and num_splits > 1)

    # Round batch size to multiple of 4 for alignment
    b_rounded = ((batch_size + 3) // 4) * 4

    # Calculate number of prepare batch vectors (line 671-673)
    # For varlen, use_prepare_varlen=True, so we start with 2
    num_prepare_batch_vectors = 2

    # varlen_sort_batches = !is_local (line 667)
    varlen_sort_batches = not is_local
    if varlen_sort_batches:
        num_prepare_batch_vectors += 1

    # head_swizzle = is_causal || is_local (line 668)
    head_swizzle = is_causal or is_local
    if head_swizzle:
        num_prepare_batch_vectors += 1

    # Calculate final size (line 675-676)
    tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors
    metadata_size = int(scheduler_needs_semaphore) + tile_count_semaphore_offset

    return metadata_size
