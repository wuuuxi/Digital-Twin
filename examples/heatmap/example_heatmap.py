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


def main():
    subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    target_muscles = ['GL', 'SOL', 'FibLon', 'VL', 'RF', "GlutMax"]
    # target_muscles = ['GL', 'FibLon', 'VL']
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

    plt.show()


if __name__ == '__main__':
    main()