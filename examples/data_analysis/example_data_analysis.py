"""
调试阶段：数据处理 + 对齐 / 运动切片 / 3D散点可视化 / 平均曲线绘制 / 肌肉分析 / 位置误差分析

用法：
    python example_data_analysis.py

说明：
    target_muscles 使用短名（不含 emg_ 前缀），如 ['VL', 'FibLon']。
    程序会自动匹配实际列名（支持 LFibLon / RFibLon 等变体命名）。
    - 只用单块肌肉的函数（运动切片、3D散点）：取列表第一个
    - 其余分析/可视化函数：使用全部肌肉
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline


def main():
    subject = Subject('../config/20260513_squat_FTS09.json')
    # subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    results = pipeline.run(include_xsens=False)

    target_muscles = ['LVL']

    # 将短名转为 emg_ 前缀形式
    emg_muscles = [f'emg_{m}' for m in target_muscles]
    emg_single = emg_muscles[:1]

    # 调试可视化
    pipeline.visualize_alignment(target_cols=emg_muscles)        # 对齐可视化
    pipeline.visualize_movement_segments(target_muscles=emg_single)  # 运动切片
    pipeline.visualize_test_3d_scatter(target_muscles=emg_single)    # 3D散点图

    # # 曲线与肌肉分析
    # pipeline.plot()                                                           # 平均曲线
    # pipeline.visualize_muscle_analysis(target_muscles=emg_muscles)            # 肌肉分析
    # pipeline.visualize_analyze_kinematic_emg_errors_by_position(             # 位置误差
    #     target_muscles=emg_muscles)
    # pipeline.analyze_muscle_kinematic_errors_individual(                      # 单肌肉误差
    #     target_muscles=emg_muscles)

    plt.show()


if __name__ == '__main__':
    main()