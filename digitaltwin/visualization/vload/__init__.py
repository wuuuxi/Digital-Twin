"""Variable-load visualization submodule.

模块组成：
  - variable_load_plot.py    : 优化结果热力图 / 激活 / 负载曲线
  - vload_comparison_plot.py : 不同负载下 Robot 运动学 + EMG 激活柱状图
  - vload_result_plot.py     : 实测 vs 预测对比 + per-muscle RMSE 柱状图
"""
from importlib import import_module

_EXPORTS = {
    "plot_variable_load_result": (".variable_load_plot", "plot_variable_load_result"),
    "plot_variable_load_result_multi_muscles": (".variable_load_plot", "plot_variable_load_result_multi_muscles"),
    "plot_danger_area": (".variable_load_plot", "plot_danger_area"),
    "plot_robot_kinematics_bar": (".vload_comparison_plot", "plot_robot_kinematics_bar"),
    "plot_emg_activation_bar": (".vload_comparison_plot", "plot_emg_activation_bar"),
    "plot_vload_overlay": (".vload_result_plot", "plot_vload_overlay"),
    "plot_vload_per_muscle_compare": (".vload_result_plot", "plot_vload_per_muscle_compare"),
    "print_vload_rmse_summary": (".vload_result_plot", "print_vload_rmse_summary"),
    "print_groups_rmse": (".vload_result_plot", "print_groups_rmse"),
    "DEFAULT_BAR_COLORS": (".vload_result_plot", "DEFAULT_BAR_COLORS"),
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
