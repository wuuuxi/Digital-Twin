"""
变负载结果对比（包含估算负载热力图预测）

在 example_vload_result 的基础上，额外叠加一条预测曲线：
  加载 heatmap_estimated_load 参数，
  对变负载实测数据中逻样本估算实际负载 (交互力 / 加速度 + g)，
  再用该估算负载预测肌肉激活。

左图窗：  实测 EMG 散点 + Expected (C0) + Goal (C1)
              + Heatmap RBF (C3) + Heatmap P-spline (C2)
              + Est.Load Heatmap (C4/紫)
右图：  规划负载曲线 (C0) + 实测估算负载散点 (C4/紫)

前提：
  1. 已运行 example_heatmap_estimated_load.py 并保存参数至
     result_folder/heatmap_estimated_load/params/
  2. 已有变负载配置（subject.vload_data）与实测数据

用法：
    python example_vload_result_est_load.py
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.analysis.heatmap.heatmap_io import load_heatmap_params_by_mode
from digitaltwin.analysis.vload.vload_planning import load_planned_vload
from digitaltwin.visualization.vload.vload_result_plot import (
    plot_vload_overlay_est_load,
    print_vload_rmse_summary,
)


# ---- 选项 ----
MOVEMENT_TYPES = ['upward']   # ['upward'] / ['downward'] / ['upward','downward'] / None
HEATMAP_MODE   = 'both'       # 'both' / 'rbf' / 'pspline' / 'none'


def load_est_load_params(subject, muscle):
    """从 result_folder/heatmap_estimated_load/params/ 加载 P-spline 参数。"""
    params_dir = os.path.join(
        subject.result_folder, 'heatmap_estimated_load', 'params')
    path = os.path.join(params_dir, f'{muscle}_est_pspline_params.pkl')
    if not os.path.exists(path):
        print(f'  [est_load] 未找到参数文件: {path}')
        return None
    with open(path, 'rb') as f:
        params = pickle.load(f)
    print(f'  [est_load] 加载 P-spline 参数: {path}')
    return params


def main():
    subject = Subject('../config/20250514_squat_RCT009.json')
    # subject = Subject('../config/20250512_squat_Yuchen.json')
    # subject = Subject('../config/20250409_squat_NCMP001_mvc.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # 加载变负载实测数据
    vload_results = pipeline.run_vload()
    if not vload_results:
        print('未加载到变负载实际数据，终止。')
        return

    rmse_summary = {}

    for label, params in subject.vload_data.items():
        if label not in vload_results:
            print(f'\n>>> 跳过 {label}：未成功处理')
            continue

        vload_file        = params.get('vload_file')
        target_muscle     = params.get('target_muscle')
        target_activation = params.get('target_activation')

        print(f'\n>>> 对比 {label} '
              f'(muscle={target_muscle}, goal={target_activation})')

        planned_df = load_planned_vload(subject, vload_file)
        if planned_df is None:
            print('  跳过：未找到 vload_file 规划数据')
            continue

        # 标准 heatmap 预测器（规划负载）
        heatmap_overlays = load_heatmap_params_by_mode(
            subject, target_muscle, HEATMAP_MODE)

        # 估算负载热力图参数（逆样本估算负载）
        est_load_params = load_est_load_params(subject, target_muscle)

        _, rmse_dict = plot_vload_overlay_est_load(
            label=label,
            vload_result=vload_results[label],
            planned_df=planned_df,
            heatmap_overlays=heatmap_overlays,
            est_load_params=est_load_params,
            target_muscle=target_muscle,
            target_activation=target_activation,
            movement_types=MOVEMENT_TYPES,
        )

        for k in ('expected', 'rbf', 'pspline', 'est_load'):
            if k in rmse_dict:
                rmse, n = rmse_dict[k]
                if np.isfinite(rmse):
                    print(f'  RMSE [{k:>12}] = {rmse:.4f}  (n={n})')
        rmse_summary[label] = rmse_dict

    print_vload_rmse_summary(
        rmse_summary,
        keys=('expected', 'rbf', 'pspline', 'est_load'),
        header_map={
            'expected':  'Expected',
            'rbf':       'Heatmap (RBF)',
            'pspline':   'Heatmap (P-sp)',
            'est_load':  'EstLoad (P-sp)',
        },
    )
    plt.show()


if __name__ == '__main__':
    main()