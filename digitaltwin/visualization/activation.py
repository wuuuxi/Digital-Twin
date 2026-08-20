"""Visualization API for activation surfaces.

The current plotting implementation remains shared with the former heatmap
module during migration; new callers should use this domain name.
"""

from .heatmap import (
    compare_activation_maps,
    draw_heatmap_2d,
    draw_load_sensitivity_heatmap_2d,
    plot_activation_3d,
    plot_compare_activation_3d,
    plot_compare_heatmap_2d,
    plot_compare_load_sensitivity_2d,
    plot_load_slices_comparison,
)

__all__ = [
    "plot_activation_3d", "compare_activation_maps", "draw_heatmap_2d",
    "draw_load_sensitivity_heatmap_2d", "plot_compare_activation_3d",
    "plot_compare_heatmap_2d", "plot_compare_load_sensitivity_2d",
    "plot_load_slices_comparison",
]
