"""
Xsens 运动捕捉 + Robot + EMG 联合分析示例（仅固定负载）

图 1：对齐可视化     图 2：运动切片     图 3：位置散点
图 4：多关节角左右散点  图 5：关节角均值柱状图 + 左右差异

用法：
    python example_load_xsens.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.visualization.xsens_plot import (
    _get_motion_defaults, build_data_groups,
    plot_alignment, plot_movement_segments, plot_position_scatter,
    plot_joint_scatter_lr, plot_joint_bar_lr,
)


def main():
    subject = Subject('../config/20250409_squat_NCMP001_xsens.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    results = pipeline.run(include_xsens=True)

    defaults = _get_motion_defaults(subject.target_motion)
    xsens_joint = defaults['xsens_joint']
    target_emg = defaults['target_emg']
    joint_bases = defaults['joint_bases']
    muscle_col = target_emg[0]

    groups = build_data_groups(results)  # 仅固定负载

    plot_alignment(groups, target_emg, xsens_joint)
    plot_movement_segments(groups, muscle_col, xsens_joint)
    plot_position_scatter(groups, muscle_col, xsens_joint)
    plot_joint_scatter_lr(results, joint_bases)
    plot_joint_bar_lr(results, joint_bases)

    plt.show()


if __name__ == '__main__':
    main()