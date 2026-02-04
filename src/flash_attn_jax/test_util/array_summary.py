"""
Utilities for summarizing the structure of multidimensional boolean arrays.

Helps answer: Is there significant correlation between the boolean values
and any subset of the axis coordinates?
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class AxisSummary:
    """Summary statistics for one axis of the boolean array."""
    axis: int
    size: int
    marginal_rates: np.ndarray  # True rate at each index position
    rate_min: float
    rate_max: float
    rate_std: float
    correlation: float  # Point-biserial correlation with coordinate
    eta_squared: float  # Variance explained by this axis
    chi_squared: float  # Chi-square statistic
    chi_squared_pvalue: float  # p-value for independence test


def _sparkline(
    values: np.ndarray, 
    width: int = 20,
    blocks: str = "▁▂▃▄▅▆▇█"
) -> str:
    """
    Generate a sparkline string from an array of values.
    
    Parameters
    ----------
    values : np.ndarray
        1D array of numeric values
    width : int
        Target width of sparkline (values will be binned if len > width)
    blocks : str
        Unicode block characters from lowest to highest
        
    Returns
    -------
    str
        Sparkline string representation
    """
    values = np.asarray(values).flatten()
    
    if len(values) == 0:
        return ""
    
    # Bin values if array is longer than target width
    if len(values) > width:
        # Resample to target width by averaging bins
        bins = np.array_split(values, width)
        values = np.array([b.mean() for b in bins])
    
    # Normalize to [0, 1]
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        # Constant values - use middle block
        return blocks[len(blocks) // 2] * len(values)
    
    normalized = (values - vmin) / (vmax - vmin)
    
    # Map to block characters
    n_blocks = len(blocks)
    indices = np.clip((normalized * (n_blocks - 1)).astype(int), 0, n_blocks - 1)
    
    return "".join(blocks[i] for i in indices)


def _is_axis_significant(ax: 'AxisSummary', alpha: float = 0.05, min_eta: float = 0.01) -> bool:
    """Check if an axis has a significant effect."""
    return ax.chi_squared_pvalue < alpha and ax.eta_squared >= min_eta


@dataclass  
class BoolArraySummary:
    """Complete summary of a multidimensional boolean array's structure."""
    shape: tuple
    ndim: int
    total_elements: int
    true_count: int
    density: float  # Overall proportion of True values
    axes: list[AxisSummary]  # Per-axis summaries
    
    def __repr__(self):
        lines = [
            f"BoolArraySummary(shape={self.shape})",
            f"  Overall density: {self.density:.4f} ({self.true_count:,}/{self.total_elements:,})",
            f"  Per-axis breakdown:"
        ]
        for ax in self.axes:
            sig = "***" if ax.chi_squared_pvalue < 0.001 else (
                  "**" if ax.chi_squared_pvalue < 0.01 else (
                  "*" if ax.chi_squared_pvalue < 0.05 else ""))
            
            line = (
                f"    Axis {ax.axis} (size={ax.size}): "
                f"rate=[{ax.rate_min:.3f}, {ax.rate_max:.3f}], "
                f"η²={ax.eta_squared:.4f}, "
                f"r={ax.correlation:+.3f}, "
                f"p={ax.chi_squared_pvalue:.2e} {sig}"
            )
            
            # Add sparkline for significant axes
            if _is_axis_significant(ax):
                spark = _sparkline(ax.marginal_rates)
                line += f"\n           {spark}"
            
            lines.append(line)
        return "\n".join(lines)


def summarize_bool_array(
    arr: np.ndarray,
    compute_pvalues: bool = True
) -> BoolArraySummary:
    """
    Summarize the structure of a multidimensional boolean array.
    
    Analyzes whether the boolean values are correlated with position
    along each axis, using multiple complementary metrics.
    
    Parameters
    ----------
    arr : np.ndarray
        Boolean array with 1-5 dimensions.
    compute_pvalues : bool, default True
        Whether to compute chi-squared p-values (requires scipy).
        
    Returns
    -------
    BoolArraySummary
        Structured summary including:
        - Overall density (proportion of True values)
        - Per-axis marginal rates (True rate at each index)
        - Coordinate correlation (linear trend along axis)
        - Eta-squared (variance explained by axis)
        - Chi-squared test (independence test)
    
    Examples
    --------
    >>> arr = np.random.rand(100, 50, 30) < 0.3  # Random ~30% density
    >>> summary = summarize_bool_array(arr)
    >>> print(summary)
    
    >>> # Array with strong axis-0 correlation
    >>> probs = np.linspace(0.1, 0.9, 100)[:, None, None]
    >>> arr = np.random.rand(100, 50, 30) < probs
    >>> summary = summarize_bool_array(arr)
    >>> print(summary)  # Will show high correlation on axis 0
    """
    arr = np.asarray(arr, dtype=bool)
    
    if arr.ndim < 1 or arr.ndim > 5:
        raise ValueError(f"Array must have 1-5 dimensions, got {arr.ndim}")
    
    if arr.size == 0:
        raise ValueError("Array must not be empty")
    
    # Overall statistics
    total = arr.size
    true_count = arr.sum()
    density = true_count / total
    
    # Analyze each axis
    axis_summaries = []
    for axis in range(arr.ndim):
        axis_summary = _analyze_axis(arr, axis, density, compute_pvalues)
        axis_summaries.append(axis_summary)
    
    return BoolArraySummary(
        shape=arr.shape,
        ndim=arr.ndim,
        total_elements=total,
        true_count=int(true_count),
        density=density,
        axes=axis_summaries
    )


def _analyze_axis(
    arr: np.ndarray, 
    axis: int, 
    global_density: float,
    compute_pvalues: bool
) -> AxisSummary:
    """Compute all summary statistics for a single axis."""
    
    # Marginal rates: mean along all OTHER axes
    other_axes = tuple(i for i in range(arr.ndim) if i != axis)
    if other_axes:
        marginal_rates = arr.mean(axis=other_axes)
    else:
        marginal_rates = arr.astype(float)  # 1D case
    
    # Coordinate correlation (point-biserial)
    correlation = _coordinate_correlation(arr, axis)
    
    # Eta-squared (variance explained)
    eta_sq = _eta_squared(arr, axis, marginal_rates, global_density)
    
    # Chi-squared test for independence
    chi_sq, p_value = _chi_squared_test(arr, axis, compute_pvalues)
    
    return AxisSummary(
        axis=axis,
        size=arr.shape[axis],
        marginal_rates=marginal_rates,
        rate_min=float(marginal_rates.min()),
        rate_max=float(marginal_rates.max()),
        rate_std=float(marginal_rates.std()),
        correlation=correlation,
        eta_squared=eta_sq,
        chi_squared=chi_sq,
        chi_squared_pvalue=p_value
    )


def _coordinate_correlation(arr: np.ndarray, axis: int) -> float:
    """
    Compute point-biserial correlation between boolean values 
    and coordinate position along the specified axis.
    
    This detects LINEAR trends (e.g., True rate increases with index).
    """
    # Create coordinate array for this axis
    shape = [1] * arr.ndim
    shape[axis] = arr.shape[axis]
    coords = np.arange(arr.shape[axis]).reshape(shape)
    coords = np.broadcast_to(coords, arr.shape)
    
    # Flatten and compute correlation
    flat_coords = coords.ravel()
    flat_arr = arr.ravel().astype(float)
    
    # Handle edge cases
    if flat_arr.std() == 0 or flat_coords.std() == 0:
        return 0.0
    
    # Pearson correlation (equivalent to point-biserial for binary)
    corr = np.corrcoef(flat_coords, flat_arr)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _eta_squared(
    arr: np.ndarray, 
    axis: int, 
    marginal_rates: np.ndarray,
    global_density: float
) -> float:
    """
    Compute eta-squared: proportion of variance explained by the axis.
    
    This is an ANOVA-style metric. Values range from 0 (no effect) 
    to 1 (axis completely determines the value).
    
    η² = SS_between / SS_total
    """
    # Total variance
    total_var = global_density * (1 - global_density)
    if total_var == 0:
        return 0.0
    
    # Number of elements per group (per index along this axis)
    n_per_group = arr.size // arr.shape[axis]
    
    # Between-group sum of squares
    ss_between = n_per_group * np.sum((marginal_rates - global_density) ** 2)
    
    # Total sum of squares
    ss_total = arr.size * total_var
    
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    return float(np.clip(eta_sq, 0, 1))


def _chi_squared_test(
    arr: np.ndarray, 
    axis: int,
    compute_pvalue: bool
) -> tuple[float, float]:
    """
    Chi-squared test for independence between boolean value and axis position.
    
    Tests H0: The boolean value is independent of position along this axis.
    """
    # Compute observed counts at each index
    other_axes = tuple(i for i in range(arr.ndim) if i != axis)
    if other_axes:
        true_counts = arr.sum(axis=other_axes)
    else:
        true_counts = arr.astype(int)
    
    n_per_group = arr.size // arr.shape[axis]
    false_counts = n_per_group - true_counts
    
    # Expected counts under independence
    total_true = arr.sum()
    total_false = arr.size - total_true
    expected_true = total_true / arr.shape[axis]
    expected_false = total_false / arr.shape[axis]
    
    if expected_true == 0 or expected_false == 0:
        return 0.0, 1.0
    
    # Chi-squared statistic
    chi_sq = (
        np.sum((true_counts - expected_true) ** 2 / expected_true) +
        np.sum((false_counts - expected_false) ** 2 / expected_false)
    )
    
    # Compute p-value if requested
    if compute_pvalue:
        try:
            from scipy import stats
            df = arr.shape[axis] - 1
            p_value = 1 - stats.chi2.cdf(chi_sq, df) if df > 0 else 1.0
        except ImportError:
            warnings.warn("scipy not available, p-values set to NaN")
            p_value = float('nan')
    else:
        p_value = float('nan')
    
    return float(chi_sq), float(p_value)


def find_significant_axes(
    summary: BoolArraySummary,
    alpha: float = 0.05,
    min_eta_squared: float = 0.01
) -> list[int]:
    """
    Identify axes with statistically significant and meaningful effects.
    
    Parameters
    ----------
    summary : BoolArraySummary
        Output from summarize_bool_array()
    alpha : float
        Significance level for chi-squared test
    min_eta_squared : float
        Minimum effect size (variance explained) to consider meaningful
        
    Returns
    -------
    list[int]
        Indices of axes with significant correlations
    """
    significant = []
    for ax in summary.axes:
        if ax.chi_squared_pvalue < alpha and ax.eta_squared >= min_eta_squared:
            significant.append(ax.axis)
    return significant


def print_detailed_report(summary: BoolArraySummary) -> None:
    """Print a detailed human-readable report."""
    print("=" * 60)
    print("BOOLEAN ARRAY STRUCTURE SUMMARY")
    print("=" * 60)
    print(f"\nShape: {summary.shape}")
    print(f"Total elements: {summary.total_elements:,}")
    print(f"True count: {summary.true_count:,}")
    print(f"Overall density: {summary.density:.4f} ({summary.density*100:.2f}%)")
    
    print("\n" + "-" * 60)
    print("PER-AXIS ANALYSIS")
    print("-" * 60)
    
    for ax in summary.axes:
        print(f"\n📊 Axis {ax.axis} (size={ax.size})")
        print(f"   Marginal rates: min={ax.rate_min:.4f}, max={ax.rate_max:.4f}, std={ax.rate_std:.4f}")
        print(f"   Rate range: {ax.rate_max - ax.rate_min:.4f} ({(ax.rate_max/max(ax.rate_min, 1e-10)):.2f}x variation)")
        print(f"   Correlation with coordinate: r = {ax.correlation:+.4f}")
        print(f"   Variance explained: η² = {ax.eta_squared:.4f} ({ax.eta_squared*100:.2f}%)")
        print(f"   Chi-squared test: χ² = {ax.chi_squared:.2f}, p = {ax.chi_squared_pvalue:.2e}")
        
        # Interpretation
        if ax.chi_squared_pvalue < 0.001 and ax.eta_squared > 0.05:
            print(f"   ⚠️  STRONG EFFECT: Boolean values strongly depend on axis {ax.axis} position")
            print(f"   Sparkline: {_sparkline(ax.marginal_rates, width=40)}")
        elif ax.chi_squared_pvalue < 0.05 and ax.eta_squared > 0.01:
            print(f"   ⚡ Moderate effect detected")
            print(f"   Sparkline: {_sparkline(ax.marginal_rates, width=40)}")
        else:
            print(f"   ✓ No significant effect (values appear independent of this axis)")


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("\n" + "=" * 60)
    print("DEMO 1: Random array (no structure)")
    print("=" * 60)
    random_arr = np.random.rand(100, 50, 30) < 0.3
    summary = summarize_bool_array(random_arr)
    print(summary)
    
    print("\n" + "=" * 60)
    print("DEMO 2: Linear gradient along axis 0")
    print("=" * 60)
    probs = np.linspace(0.1, 0.9, 100)[:, None, None]
    gradient_arr = np.random.rand(100, 50, 30) < probs
    summary = summarize_bool_array(gradient_arr)
    print(summary)
    
    print("\n" + "=" * 60)
    print("DEMO 3: Step function along axis 1")
    print("=" * 60)
    probs = np.where(np.arange(50) < 25, 0.2, 0.8)[None, :, None]
    step_arr = np.random.rand(100, 50, 30) < probs
    summary = summarize_bool_array(step_arr)
    print(summary)
    
    print("\n" + "=" * 60)
    print("DEMO 4: Multiple axes with effects")
    print("=" * 60)
    # Axis 0: linear gradient
    # Axis 1: step function
    # Axis 2: no effect
    prob_0 = np.linspace(0.2, 0.6, 80)[:, None, None]
    prob_1 = np.where(np.arange(60) < 30, 0.0, 0.3)[None, :, None]
    combined_prob = np.clip(prob_0 + prob_1, 0, 1)
    multi_arr = np.random.rand(80, 60, 40) < combined_prob
    summary = summarize_bool_array(multi_arr)
    print_detailed_report(summary)
    
    print("\n" + "=" * 60)
    print("DEMO 5: 5D array")
    print("=" * 60)
    probs_5d = np.linspace(0.1, 0.5, 20)[:, None, None, None, None]
    arr_5d = np.random.rand(20, 15, 10, 8, 5) < probs_5d
    summary = summarize_bool_array(arr_5d)
    print(summary)
    
    significant = find_significant_axes(summary)
    print(f"\nSignificant axes: {significant}")