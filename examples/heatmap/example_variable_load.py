"""
变负载优化（默认使用 P-spline 曲面）

generate_heatmaps 会同时产出 RBF 与 monotone P-spline 两套参数。
本示例默认走 P-spline 路径：从 {musc}_pspline_params.pkl 加载曲面，
转为截断幂次基，在 Pyomo 中作 C² 光滑的符号求值直接在
 ipopt 内部使用。如需回退到 RBF 路径，将 use_pspline=False 传入
 run_variable_load_optimization 即可。

用法：
    python example_variable_load.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline


def main():
    subject = Subject('../config/20250409_squat_NCMP001_mvc.json')
    # subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # # Step 1: 产出 RBF 与 P-spline 两套热力图参数
    # target_muscles = ['GL', 'FibLon', 'VL']
    # pipeline.generate_heatmaps(target_muscles)

    # Step 2: 变负载优化（默认 use_pspline=True）
    pipeline.run_variable_load_optimization(variable_mode=1, use_pspline=True)
    plt.show()


if __name__ == '__main__':
    main()