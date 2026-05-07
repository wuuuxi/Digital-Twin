"""Variable-load visualization submodule.

模块组成：
  - variable_load_plot.py    : 优化结果热力图 / 激活 / 负载曲线
  - vload_comparison_plot.py : 不同负载下 Robot 运动学 + EMG 激活柱状图
  - vload_result_plot.py     : 实测 vs 预测对比 + per-muscle RMSE 柱状图
"""
from . import variable_load_plot, vload_comparison_plot, vload_result_plot
from .variable_load_plot import (
    plot_variable_load_result,
    plot_variable_load_result_multi_muscles,
    plot_danger_area,
)
from .vload_comparison_plot import (
    plot_robot_kinematics_bar,
    plot_emg_activation_bar,
)
from .vload_result_plot import (
    plot_vload_overlay,
    plot_vload_per_muscle_compare,
    print_vload_rmse_summary,
    print_groups_rmse,
    DEFAULT_BAR_COLORS,
)

__all__ = [
    'variable_load_plot', 'vload_comparison_plot', 'vload_result_plot',
    # variable_load_plot
    'plot_variable_load_result',
    'plot_variable_load_result_multi_muscles',
    'plot_danger_area',
    # vload_comparison_plot
    'plot_robot_kinematics_bar',
    'plot_emg_activation_bar',
    # vload_result_plot
    'plot_vload_overlay',
    'plot_vload_per_muscle_compare',
    'print_vload_rmse_summary',
    'print_groups_rmse',
    'DEFAULT_BAR_COLORS',
]