"""Activation fitting workflow compatibility façade.

The numerical implementation is still shared with the former heatmap module
while callers migrate to the activation domain name.
"""

from digitaltwin.analysis.heatmap.heatmap_generator import (
    HeatmapGenerator,
    collect_cutted_data,
    estimate_load_from_df,
)

collect_segments = collect_cutted_data

__all__ = [
    "HeatmapGenerator", "collect_segments", "collect_cutted_data",
    "estimate_load_from_df",
]
