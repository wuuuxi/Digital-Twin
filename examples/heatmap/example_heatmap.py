"""
肌肉激活热力图：RBF拟合 + 3D曲面 + 2D热力图 + RMSE报告

用法：
    python example_heatmap.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline


def main():
    subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # 自动执行 pipeline.run() 并生成热力图
    params = pipeline.generate_heatmaps(
        # muscles=['biceps', 'triceps'],  # 可指定肌肉子集
        # fit_3d=True,                       # 可选 3D 拟合
    )
    plt.show()


if __name__ == '__main__':
    main()