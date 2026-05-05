"""
肌肉激活热力图：RBF拟合 + 3D曲面 + 2D热力图 + RMSE报告

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

    # target_muscles = ['GL', 'SOL', 'FibLon', 'VL', 'RF', "GlutMax"]
    target_muscles = ['GL', 'FibLon', 'VL']
    movement_types = ['upward']

    # 自动执行 pipeline.run() 并生成热力图
    params = pipeline.generate_heatmaps(
        muscles=target_muscles,  # 可指定肌肉子集
        fit_3d=False, # 可选 3D 拟合
        movement_types=movement_types,  # 仅上升阶段（默认值）
        # movement_types=['upward', 'downward'],   # 上升 + 下降阶段
        monotonic_load=True,  # 额外生成单调修正后的曲面

        # 二选一：
        # 'smooth_projection' = RBF 拟合 + 平滑单调投影后处理
        # 'monotone_pspline'  = 直接用 2D 张量积 B-spline，负载方向单调约束
        # monotonic_method='smooth_projection',
        monotonic_method='monotone_pspline',

        # ---- smooth_projection 专属参数 ----
        projection_lambda_data=1.0,  # 越大越贴近原始 RBF
        projection_lambda_height=0.01,  # 高度方向平滑，越小越保留高度细节
        projection_lambda_load=5.0,  # 负载方向平滑，越大敏感度图越连续
        projection_lambda_cross=0.1,  # 高度-负载交叉平滑
        projection_min_slope=0.0,  # 0=非递减；可试 0.001 减少大片 0 敏感度
        projection_solver='auto',  # auto: 优先 cvxpy，缺失时回退 scipy 罚函数

        # ---- monotone_pspline 专属参数 ----
        pspline_n_basis_h=20,  # 高度方向 B-spline basis 个数（越多越灵活）
        pspline_n_basis_l=10,  # 负载方向 B-spline basis 个数
        pspline_degree=3,  # B-spline 阶数（3=三次）
        pspline_lambda_h=0.1,  # 高度方向二阶差分平滑权重
        pspline_lambda_l=1.0,  # 负载方向二阶差分平滑权重
        pspline_solver='auto',  # auto: 优先 cvxpy QP，缺失时回退 L-BFGS-B
        pspline_max_iter=2000,
    )

    # ====== 每块肌肉一张：每个负载下的原始数据散点 + 两条 RBF 拟合曲线 ======
    cutted = pipeline._collect_cutted_data(movement_types=movement_types)
    # 与拟合时一致：限制到 height_range
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
                params_orig=params[musc],
                params_mono=params.get(f'{musc}_monotonic'),
                muscle=musc,
                pos_col='pos_l',
                load_col=load_col,
                result_folder=None,
            )

    plt.show()


if __name__ == '__main__':
    main()