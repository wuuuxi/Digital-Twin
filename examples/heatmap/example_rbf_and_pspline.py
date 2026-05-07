"""
RBF vs P-spline 对比示例

在 example_vload_result.py 的基础上同时对比 RBF 与 P-spline 两种预测。

每个变负载实验（= 一块目标肌肉）一张 1×2 图：
  左：N 个变负载子图（N = subject.vload_data 组数），每个子图画
       该期间该肌肉的实测 EMG 与 RBF / P-spline 两种预测的对比。
       target 那一组额外画 Expected 与 Goal。
  右：该肌肉在 5 组固定负载 + 3 组变负载下，实测 EMG 与
       RBF / P-spline 两种预测的 RMSE 柱状图。

说明：
  - 运行本脚本前需先跑过 example_heatmap.py 生成
    {muscle}_rbf_params.pkl 与 {muscle}_pspline_params.pkl。

用法：
    python example_rbf_and_pspline.py
"""
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.analysis.heatmap.heatmap_io import load_heatmap_params_by_mode
from digitaltwin.analysis.vload.vload_planning import load_planned_vload
from digitaltwin.visualization.vload.vload_result_plot import (
    plot_vload_per_muscle_compare, print_groups_rmse,
)


# ---- 选项 ----
MOVEMENT_TYPES = ['upward']


def main():
    subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # 加载固定负载 + 变负载数据
    pipeline.run(include_xsens=False)
    vload_results = pipeline.run_vload()
    if not vload_results:
        print('未加载到变负载实际数据，终止。')
        return

    for label, params in subject.vload_data.items():
        if label not in vload_results:
            print(f'\n>>> 跳过 {label}：未成功处理')
            continue

        vload_file = params.get('vload_file')
        target_muscle = params.get('target_muscle')
        target_activation = params.get('target_activation')
        if target_muscle is None:
            print(f'\n>>> 跳过 {label}：未指定 target_muscle')
            continue

        print(f'\n>>> 处理 {label} '
              f'(muscle={target_muscle}, goal={target_activation})')

        planned_df = load_planned_vload(subject, vload_file)
        if planned_df is None:
            print('  跳过：未找到 vload_file 规划数据')
            continue

        heatmap_overlays = load_heatmap_params_by_mode(
            subject, target_muscle, 'both')
        if not heatmap_overlays:
            print('  跳过：未找到 RBF / P-spline 参数')
            continue

        _, _, groups = plot_vload_per_muscle_compare(
            target_label=label,
            vload_result=vload_results[label],
            planned_df_target=planned_df,
            heatmap_overlays=heatmap_overlays,
            target_muscle=target_muscle,
            target_activation=target_activation,
            pipeline=pipeline,
            subject=subject,
            vload_results=vload_results,
            movement_types=MOVEMENT_TYPES,
        )

        print_groups_rmse(target_muscle, groups)

    plt.show()


if __name__ == '__main__':
    main()