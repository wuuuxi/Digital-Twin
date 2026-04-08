"""
变负载优化（需先生成热力图 RBF 参数）

用法：
    python example_variable_load.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline


def main():
    subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # Step 1: 确保 RBF 参数已生成
    pipeline.generate_heatmaps()

    # Step 2: 运行变负载优化
    pipeline.run_variable_load_optimization(variable_mode=1)
    plt.show()


if __name__ == '__main__':
    main()