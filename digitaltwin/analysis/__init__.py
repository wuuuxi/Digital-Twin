from .alignment import DataAligner, filter_movement_types
from .curve_analysis import CurveAnalyzer

# 子包重导出（heatmap 拟合 + 变负载）
from .heatmap import (
    rbf_function, rbf_fit, rbf_predict, predict_at,
    fit_activation_map, fit_activation_map_3d,
    save_rbf_params, load_rbf_params,
    compute_rmse_percentage, compute_rmse_by_load,
    fit_monotone_pspline_2d, predict_monotone_pspline,
    load_pspline_params, load_heatmap_params_by_mode,
)
from .vload import (
    variable_load_optimization, variable_load_optimization_max,
    one_muscle_variable_load, generate_variable_load,
    load_planned_vload,
    compute_rmse_at_actual_points,
    compute_groups_rmse_for_muscle,
    format_rmse_for_legend,
)

__all__ = [
    'DataAligner', 'filter_movement_types', 'CurveAnalyzer',
    # heatmap fitting
    'rbf_function', 'rbf_fit', 'rbf_predict', 'predict_at',
    'fit_activation_map', 'fit_activation_map_3d',
    'save_rbf_params', 'load_rbf_params',
    'compute_rmse_percentage', 'compute_rmse_by_load',
    'fit_monotone_pspline_2d', 'predict_monotone_pspline',
    'load_pspline_params', 'load_heatmap_params_by_mode',
    # variable load
    'variable_load_optimization', 'variable_load_optimization_max',
    'one_muscle_variable_load', 'generate_variable_load',
    'load_planned_vload',
    'compute_rmse_at_actual_points',
    'compute_groups_rmse_for_muscle',
    'format_rmse_for_legend',
]