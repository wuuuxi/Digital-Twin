"""Heatmap fitting submodule: RBF + monotone P-spline + parameter I/O.

模块组成：
  - rbf_fitting.py        : 普通 RBF 拟合
  - monotone_pspline.py   : 2D 单调 P-spline 拟合
  - heatmap_io.py         : 拟合结果 pkl 的加载（基于 Subject）
  - heatmap_generator.py  : 热力图生成编排（合并定负载 / 估算负载两条路径）
"""
from . import rbf_fitting, monotone_pspline, heatmap_io, heatmap_generator
from .heatmap_generator import (
    HeatmapGenerator, collect_cutted_data, estimate_load_from_df,
)
from .rbf_fitting import (
    rbf_function, rbf_fit, rbf_predict, predict_at,
    fit_activation_map, fit_activation_map_3d,
    save_rbf_params, load_rbf_params,
    compute_rmse_percentage, compute_rmse_by_load,
)
from .monotone_pspline import (
    fit_monotone_pspline_2d, predict_monotone_pspline,
)
from .heatmap_io import (
    heatmap_param_dir,
    load_pspline_params,
    load_heatmap_params_by_mode,
)

__all__ = [
    'rbf_fitting', 'monotone_pspline', 'heatmap_io',
    # rbf_fitting
    'rbf_function', 'rbf_fit', 'rbf_predict', 'predict_at',
    'fit_activation_map', 'fit_activation_map_3d',
    'save_rbf_params', 'load_rbf_params',
    'compute_rmse_percentage', 'compute_rmse_by_load',
    # monotone_pspline
    'fit_monotone_pspline_2d', 'predict_monotone_pspline',
    # heatmap_io
    'heatmap_param_dir',
    'load_pspline_params',
    'load_heatmap_params_by_mode',
    # heatmap_generator
    'HeatmapGenerator',
    'collect_cutted_data',
    'estimate_load_from_df',
]