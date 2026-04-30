"""
肌肉激活热力图：RBF拟合 + 3D曲面 + 2D热力图 + RMSE报告

用法：
    python example_heatmap.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline


def main():
    subject = Subject('../config/20250409_squat_NCMP001_mvc.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    target_muscles = ['GL', 'SOL', 'FibLon', 'VL', 'RF', "GlutMax"]

    # 自动执行 pipeline.run() 并生成热力图
    params = pipeline.generate_heatmaps(
        muscles=target_muscles,  # 可指定肌肉子集
        fit_3d=False, # 可选 3D 拟合
        movement_types=['upward'],  # 仅上升阶段（默认值）
        # movement_types=['upward', 'downward'],   # 上升 + 下降阶段

    )
    plt.show()


if __name__ == '__main__':
    main()