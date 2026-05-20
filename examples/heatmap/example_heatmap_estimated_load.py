"""
使用逆样本估算负载（替代 JSON 固定负载）生成肌肉激活热力图。

估算公式：
    estimated_load (kg) = (force_l + force_r) / ((acc_l + acc_r) / 2 + g)
    g = 9.81 m/s²

输出（保存至 result_folder/heatmap_estimated_load/）：
  每块肌肉生成：
    - 1×3 对比图：原始散点 + RBF 拟合 + P-spline 拟合
    - 2D 热力图对比图

用法：
    python example_heatmap_estimated_load.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline


def main():
    subject = Subject('../config/20250514_squat_RCT009.json')
    # subject = Subject('../config/20260513_squat_FTS09_mvc.json')
    # subject = Subject('../config/20250409_squat_NCMP001_mvc.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    pipeline.run(include_xsens=False)

    # target_muscles = ["LTA", "LGL", "LFibLon", "LVL", "LRF", "LVM", "LAddl", "LBF", "LGlutMax", "LGlutMed"]
    target_muscles = ['GL', 'FibLon', 'VL', 'RF']

    pipeline.generate_heatmaps_with_estimated_load(
        muscles=target_muscles,
        movement_types=['upward'],
        pspline_n_basis_h=20,
        pspline_n_basis_l=10,
        pspline_lambda_h=0.1,
        pspline_lambda_l=1.0,
    )
    plt.show()


if __name__ == '__main__':
    main()