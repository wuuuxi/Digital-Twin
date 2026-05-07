"""Variable-load (vload) submodule: optimization, planning data I/O, RMSE metrics.

模块组成：
  - variable_load.py : Pyomo 变负载优化（一组肌肉）
  - vload_planning.py: 加载规划好的 vload csv
  - vload_metrics.py : 实测 vs 预测的 RMSE 计算
"""
from . import variable_load, vload_planning, vload_metrics
from .variable_load import (
    variable_load_optimization, variable_load_optimization_max,
    one_muscle_variable_load, generate_variable_load,
)
from .vload_planning import load_planned_vload
from .vload_metrics import (
    compute_rmse_at_actual_points,
    compute_groups_rmse_for_muscle,
    format_rmse_for_legend,
)

__all__ = [
    'variable_load', 'vload_planning', 'vload_metrics',
    # variable_load
    'variable_load_optimization', 'variable_load_optimization_max',
    'one_muscle_variable_load', 'generate_variable_load',
    # vload_planning
    'load_planned_vload',
    # vload_metrics
    'compute_rmse_at_actual_points',
    'compute_groups_rmse_for_muscle',
    'format_rmse_for_legend',
]