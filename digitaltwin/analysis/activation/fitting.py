"""High-level activation fitting functions.

The implementation delegates to the existing numerical routines for now so
the first architecture migration does not change fitting behaviour.
"""

from digitaltwin.analysis.heatmap.rbf_fitting import (
    fit_activation_map,
    fit_activation_map_3d,
)

__all__ = ["fit_activation_map", "fit_activation_map_3d"]
