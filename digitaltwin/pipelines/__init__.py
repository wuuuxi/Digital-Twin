"""
流水线编排层。

依赖方向：digitaltwin 业务包 → pipelines → analysis（analysis 不反向依赖）。

  - multi_load: MultiLoadPipeline（固定负载）
  - vload: VLoadPipeline（变负载）
  - standard_analysis: 标准切片流水线 + 动作窗口（带缓存）
  - symmetry_check: 左右对称性检查
"""
from .multi_load import MultiLoadPipeline
from .vload import VLoadPipeline
from .standard_analysis import (
    run_standard_data_pipeline,
    load_or_create_cutted_pipeline_results,
    get_action_windows,
)
from .symmetry_check import (
    run_symmetry_check,
    SymmetryCheckOptions,
    collect_side_data,
    Verdicts,
)

__all__ = [
    'MultiLoadPipeline',
    'VLoadPipeline',
    'run_standard_data_pipeline',
    'load_or_create_cutted_pipeline_results',
    'get_action_windows',
    'run_symmetry_check',
    'SymmetryCheckOptions',
    'collect_side_data',
    'Verdicts',
]