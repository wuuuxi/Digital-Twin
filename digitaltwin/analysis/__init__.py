"""Reusable analysis primitives.

The package initializer intentionally does not import pipelines or optional
optimization backends. This keeps data/activation analysis usable without
OpenSim, Pyomo/IPOPT, or the realtime stack.
"""

from .alignment import DataAligner, filter_movement_types
from .curves import CurveAnalyzer
from .activation import (
    collect_segments,
    compute_rmse_by_load,
    compute_rmse_percentage,
    estimate_load_from_df,
    fit_activation_map,
    fit_activation_map_3d,
    fit_monotone_pspline_2d,
    predict_at,
    predict_monotone_pspline,
)
from .tabular import (
    build_left_joint_coordinate_map,
    get_segment_from_results,
    interpolate_column_to_segment,
    print_summary_table,
    read_opensim_table,
    summarize_inverse_dynamics_moments,
)

__all__ = [
    "DataAligner", "filter_movement_types", "CurveAnalyzer",
    "collect_segments", "estimate_load_from_df",
    "fit_activation_map", "fit_activation_map_3d",
    "fit_monotone_pspline_2d", "predict_monotone_pspline", "predict_at",
    "compute_rmse_percentage", "compute_rmse_by_load",
    "read_opensim_table", "get_segment_from_results",
    "interpolate_column_to_segment", "build_left_joint_coordinate_map",
    "summarize_inverse_dynamics_moments", "print_summary_table",
]


def __getattr__(name):
    """Lazily expose legacy orchestration and optimization names."""

    if name in {
        "run_standard_data_pipeline",
        "load_or_create_cutted_pipeline_results",
        "get_action_windows",
    }:
        from digitaltwin.pipelines import standard_analysis
        return getattr(standard_analysis, name)
    if name == "run_symmetry_check":
        from digitaltwin.pipelines.symmetry_check import run_symmetry_check
        return run_symmetry_check
    if name in {
        "variable_load_optimization", "variable_load_optimization_max",
        "one_muscle_variable_load", "generate_variable_load",
        "load_planned_vload", "compute_rmse_at_actual_points",
        "compute_groups_rmse_for_muscle", "format_rmse_for_legend",
    }:
        from digitaltwin.analysis import vload
        return getattr(vload, name)
    raise AttributeError(name)
