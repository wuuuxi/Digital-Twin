"""Variable-load (vload) submodule: optimization, planning data I/O, RMSE metrics.

模块组成：
  - variable_load.py : Pyomo 变负载优化（一组肌肉）
  - vload_planning.py: 加载规划好的 vload csv
  - vload_metrics.py : 实测 vs 预测的 RMSE 计算
"""
from importlib import import_module

_EXPORTS = {
    "variable_load": (".variable_load", None),
    "vload_planning": (".vload_planning", None),
    "vload_metrics": (".vload_metrics", None),
    "variable_load_optimization": (".variable_load", "variable_load_optimization"),
    "variable_load_optimization_max": (".variable_load", "variable_load_optimization_max"),
    "one_muscle_variable_load": (".variable_load", "one_muscle_variable_load"),
    "generate_variable_load": (".variable_load", "generate_variable_load"),
    "load_planned_vload": (".vload_planning", "load_planned_vload"),
    "compute_rmse_at_actual_points": (".vload_metrics", "compute_rmse_at_actual_points"),
    "compute_groups_rmse_for_muscle": (".vload_metrics", "compute_groups_rmse_for_muscle"),
    "format_rmse_for_legend": (".vload_metrics", "format_rmse_for_legend"),
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if exc.name == "pyomo":
            raise ImportError(
                "Variable-load optimization requires the optimization extra "
                "(Pyomo/IPOPT); data analysis and vload evaluation remain "
                "available without it."
            ) from None
        raise
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
