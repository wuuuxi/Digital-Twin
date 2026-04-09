"""
调试阶段：数据处理 + 对齐 / 运动切片 / 3D散点可视化 / 平均曲线绘制 / 肌肉分析 / 位置误差分析

用法：
    python example_data_analysis.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline


def main():
    subject = Subject('../config/20251009_BenchPress_Yuetian.json')
    # subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    results = pipeline.run(include_xsens=False)

    # 调试可视化
    pipeline.visualize_alignment()            # 对齐可视化
    pipeline.visualize_movement_segments()    # 运动切片
    pipeline.visualize_test_3d_scatter()      # 3D散点图

    # # 曲线与肌肉分析
    # pipeline.plot()                                               # 平均曲线
    # pipeline.visualize_muscle_analysis()                          # 肌肉分析
    # pipeline.visualize_analyze_kinematic_emg_errors_by_position() # 位置误差
    # pipeline.analyze_muscle_kinematic_errors_individual()         # 单肌肉误差

    plt.show()


if __name__ == '__main__':
    main()