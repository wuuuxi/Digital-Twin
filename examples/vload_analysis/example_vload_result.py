"""
变负载结果对比示例

对比 variable_load_file 中的变负载实际结果（高度、EMG）与预期结果
（vload_file 中的 Height / Load / Activation 规划值）。

可选：再叠加 heatmap 拟合的曲面对该工况下激活的预测值
（在每个规划点 (Height, Load) 上用 RBF / P-spline 曲面预测）。
除了画出各预测曲线与实测散点的对比之外，还会在每个实测数据点上
计算三种预测方式（Expected / RBF / P-spline）与实测 EMG 的 RMSE。

选项：
  - MOVEMENT_TYPES: 只用上升 ['upward']、只用下降 ['downward']、或两者
    ['upward', 'downward']；None 表示不过滤。
  - HEATMAP_MODE: 'both' | 'rbf' | 'pspline' | 'none'，控制叠加哪种
    heatmap 预测曲线。

用法：
    python example_vload_result.py
"""
import numpy as np
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.analysis.heatmap.heatmap_io import load_heatmap_params_by_mode
from digitaltwin.analysis.vload.vload_planning import load_planned_vload
from digitaltwin.visualization.vload.vload_result_plot import (
    plot_vload_overlay, print_vload_rmse_summary,
)


# ---- 选项 ----
MOVEMENT_TYPES = ['upward']     # ['upward'] / ['downward'] / 两者 / None
HEATMAP_MODE = 'both'           # 'both' / 'rbf' / 'pspline' / 'none'


def main():
    subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # 加载变负载实测数据（aligned + cutted）
    vload_results = pipeline.run_vload()
    if not vload_results:
        print('未加载到变负载实际数据，终止。')
        return

    rmse_summary = {}

    for label, params in subject.vload_data.items():
        if label not in vload_results:
            print(f'\n>>> 跳过 {label}：未成功处理')
            continue

        vload_file = params.get('vload_file')
        target_muscle = params.get('target_muscle')
        target_activation = params.get('target_activation')

        print(f'\n>>> 对比 {label} '
              f'(muscle={target_muscle}, goal={target_activation})')

        planned_df = load_planned_vload(subject, vload_file)
        if planned_df is None:
            print('  跳过：未找到 vload_file 规划数据')
            continue

        heatmap_overlays = load_heatmap_params_by_mode(
            subject, target_muscle, HEATMAP_MODE)

        _, rmse_dict = plot_vload_overlay(
            label=label,
            vload_result=vload_results[label],
            planned_df=planned_df,
            heatmap_overlays=heatmap_overlays,
            target_muscle=target_muscle,
            target_activation=target_activation,
            movement_types=MOVEMENT_TYPES,
        )

        for k in ('expected', 'rbf', 'pspline'):
            if k in rmse_dict:
                rmse, n = rmse_dict[k]
                if np.isfinite(rmse):
                    print(f'  RMSE [{k:>9}] = {rmse:.4f}  (n={n})')
        rmse_summary[label] = rmse_dict

    print_vload_rmse_summary(rmse_summary)
    plt.show()


if __name__ == '__main__':
    main()