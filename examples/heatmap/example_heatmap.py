"""
肌肉激活热力图：P-spline (默认) + RBF 基线 + 1×2 对比图 + RMSE 报告

输出 (默认 save_dir = result_folder/heatmap/)：
  P-spline (主曲面)：heatmap/{musc}_3D.png, _2D.png, _load_sensitivity_2D.png
  RBF (基线)：       heatmap/rbf/{musc}_3D.png, _2D.png, _load_sensitivity_2D.png
  1×2 对比图：       heatmap/{musc}_compare_3D.png,
                     heatmap/{musc}_compare_2D.png,
                     heatmap/{musc}_compare_load_sensitivity_2D.png
  pkl 参数：         heatmap/params/{musc}_rbf_params.pkl,
                     heatmap/params/{musc}_pspline_params.pkl

用法：
    python example_heatmap.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.visualization.heatmap import plot_load_slices_comparison


def print_mean_activations(pipeline, target_muscles, movement_types):
    """按 modeling_file 分别打印指定 movement_types 下各目标肌肉的平均激活。

    Parameters
    ----------
    pipeline : MultiLoadPipeline
        已调过 run() 或 generate_heatmaps() 的 pipeline（需 results 非空）。
    target_muscles : list of str
        要统计的肌肉名称，对应 cutted_data 中的 'emg_\{musc\}' 列。
    movement_types : list of str
        运动阶段过滤列表，如 ['upward'] 或 ['upward', 'downward']。
    """
    import numpy as np
    import pandas as pd

    if not pipeline.results:
        print('[mean-activation] pipeline.results 为空，请先 run() 或 generate_heatmaps()。')
        return

    header = '  '.join([f'{m:>10s}' for m in target_muscles])
    print()
    print(f'== 各 modeling_file 在 movement_types={movement_types} 下的平均肌肉激活 ==')
    print(f'{"load":>8s}  {header}')

    for load_weight, result in pipeline.results.items():
        cd = result.get('cutted_data')
        if cd is None or (hasattr(cd, '__len__') and len(cd) == 0):
            print(f'{str(load_weight):>8s}  (无切片数据)')
            continue
        if isinstance(cd, list):
            cd = pd.concat(cd, ignore_index=True)

        if movement_types is not None and 'movement_type' in cd.columns:
            cd = cd[cd['movement_type'].isin(movement_types)]

        if len(cd) == 0:
            print(f'{str(load_weight):>8s}  (过滤后为空)')
            continue

        means = []
        for musc in target_muscles:
            col = f'emg_{musc}'
            if col not in cd.columns:
                means.append('     N/A  ')
            else:
                v = float(np.nanmean(cd[col].values))
                means.append(f'{v:>10.4f}')
        print(f'{str(load_weight):>8s}  ' + '  '.join(means))
    print()


def main():
    # subject = Subject('../config/20250409_squat_NCMP001_mvc.json')
    subject = Subject('../config/20260513_squat_FTS09_mvc.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    target_muscles = ['LGL', 'LSOL', 'LFibLon', 'LVL', 'LRF', "LGlutMax"]
    # target_muscles = ['GL', 'FibLon', 'VL']
    # target_muscles = ['LGL', 'LFibLon', 'LVL']
    movement_types = ['upward']

    # 默认主曲面 = P-spline；同时跑 RBF 基线，并自动生成 1×2 对比图
    params = pipeline.generate_heatmaps(
        muscles=target_muscles,
        fit_3d=False,
        movement_types=movement_types,
        # movement_types=['upward', 'downward'],   # 上升 + 下降阶段

        # ---- P-spline 参数 ----
        pspline_n_basis_h=20,   # 高度方向 B-spline basis 数（越多越灵活）
        pspline_n_basis_l=10,   # 负载方向 B-spline basis 数
        pspline_degree=3,       # B-spline 阶数（3=三次）
        pspline_lambda_h=0.1,   # 高度方向二阶差分平滑权重
        pspline_lambda_l=1.0,   # 负载方向二阶差分平滑权重
        pspline_solver='auto',  # auto：优先 cvxpy QP，缺失时回退 L-BFGS-B
        pspline_max_iter=2000,
    )

    # ---- 每块肌肉一张：每个负载下的原始数据散点 + RBF / P-spline 拟合曲线 ----
    cutted = pipeline._collect_cutted_data(movement_types=movement_types)
    if cutted is not None and subject.height_range is not None:
        h_min, h_max = subject.height_range
        cutted = cutted[(cutted['pos_l'] >= h_min) &
                        (cutted['pos_l'] <= h_max)]

    if cutted is not None:
        load_col = 'load' if 'load' in cutted.columns else 'load_value'
        for musc in target_muscles:
            if musc not in params:
                continue
            plot_load_slices_comparison(
                cutted,
                params_orig=params.get(f'{musc}_rbf'),
                params_mono=params[musc],
                muscle=musc,
                pos_col='pos_l',
                load_col=load_col,
                result_folder=None,
            )

    # 打印每个 modeling_file 下、指定 movement_types 内各目标肌肉的平均激活
    print_mean_activations(pipeline, target_muscles, movement_types)

    plt.show()


if __name__ == '__main__':
    main()