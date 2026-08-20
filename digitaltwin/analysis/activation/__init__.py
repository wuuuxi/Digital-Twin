"""Muscle activation surface analysis.

This is the new domain name for the former ``analysis.heatmap`` package.
The compatibility package remains available while callers migrate, but new
code should import from ``digitaltwin.analysis.activation``.
"""

from .data import collect_segments, estimate_load_from_df
from .evaluation import compute_rmse_by_load, compute_rmse_percentage
from .fitting import fit_activation_map, fit_activation_map_3d
from .io import (
    heatmap_param_dir,
    load_heatmap_params_by_mode,
    load_pspline_params,
    load_rbf_params,
)
from .pspline import fit_monotone_pspline_2d, predict_monotone_pspline
from .rbf import (
    load_rbf_model,
    predict_at,
    rbf_fit,
    rbf_function,
    rbf_predict,
    save_rbf_params,
)
__all__ = [
    "collect_segments",
    "estimate_load_from_df",
    "fit_activation_map",
    "fit_activation_map_3d",
    "fit_monotone_pspline_2d",
    "predict_monotone_pspline",
    "rbf_function",
    "rbf_fit",
    "rbf_predict",
    "predict_at",
    "save_rbf_params",
    "load_rbf_model",
    "compute_rmse_percentage",
    "compute_rmse_by_load",
    "heatmap_param_dir",
    "load_rbf_params",
    "load_pspline_params",
    "load_heatmap_params_by_mode",
    "HeatmapGenerator",
]


def __getattr__(name):
    """Load the orchestration façade only when it is explicitly requested.

    Numerical activation analysis stays independent from visualization and
    optional plotting imports; the generator is a convenience boundary for
    examples that need the complete fitting-and-plotting workflow.
    """

    if name == "HeatmapGenerator":
        from .generator import HeatmapGenerator
        globals()[name] = HeatmapGenerator
        return HeatmapGenerator
    raise AttributeError(name)
