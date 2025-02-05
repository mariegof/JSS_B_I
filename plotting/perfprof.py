"""
Performance profile implementation based on the Dolan-Moré paper.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def create_performance_profile(
    data: np.ndarray,
    method_names: List[str],
    title: str = "Performance Profile",
    max_ratio: Optional[float] = None,
    styles: Optional[List[str]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a performance profile following the Dolan-Moré methodology.
    
    Args:
        data: Matrix of shape (n_instances, n_methods) containing performance measures
        method_names: Names of the methods for the legend
        title: Plot title
        max_ratio: Maximum ratio to show on x-axis (default: auto-computed)
        styles: Line styles for each method
        
    Returns:
        Figure and Axes objects
    """
    # Input validation
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    n_instances, n_methods = data.shape
    if n_methods != len(method_names):
        raise ValueError("Number of methods doesn't match data columns")
    if np.any(data <= 0):
        raise ValueError("Performance measures must be positive")
        
    # Handle NaN values (failed solves)
    data = np.where(np.isnan(data), np.inf, data)
    
    # Calculate best performance for each instance
    best_per_instance = np.min(data, axis=1)
    if np.any(best_per_instance <= 0):
        raise ValueError("Each instance must have at least one valid result")
        
    # Calculate performance ratios
    ratios = data / best_per_instance[:, np.newaxis]
    
    # Determine maximum ratio to plot
    if max_ratio is None:
        # Use 95th percentile of finite ratios or default to 3.0
        finite_ratios = ratios[np.isfinite(ratios)]
        max_ratio = min(3.0, np.percentile(finite_ratios, 95)) if len(finite_ratios) > 0 else 3.0
    
    # Generate evaluation points
    tau_values = np.linspace(1.0, max_ratio, 41)  # 41 points for smooth curves
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set default styles if none provided
    if styles is None:
        styles = [f'-{marker}' for marker in 'os^vD']  # Line with markers
        
    # Plot profile for each method
    for i, (method, style) in enumerate(zip(method_names, styles)):
        method_ratios = ratios[:, i]
        
        # Calculate fraction of problems solved within each ratio τ
        fractions = []
        for tau in tau_values:
            solved = method_ratios <= tau
            fraction = np.sum(solved) / n_instances
            fractions.append(fraction)
            
        # Create step plot
        ax.step(
            tau_values,
            fractions,
            style,
            where='post',
            label=method,
            markersize=6,
            markevery=5
        )
    
    # Customize plot appearance
    ax.set_xlabel('Performance ratio (τ)')
    ax.set_ylabel('Fraction of problems solved')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(1.0, max_ratio)
    ax.set_ylim(0.0, 1.05)  # Slight padding above 1.0
    
    # Add legend
    ax.legend(
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0),
        ncol=1,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    return fig, ax