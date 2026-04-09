"""
EMG RMS 分析示例——固定负载 + 变负载对比

图 1：RMS vs Time（运动切片风格）——固定负载 + 变负载
图 2：RMS vs Position 散点图——固定负载 + 变负载
图 3：3行×N列 Position-Velocity / EMG / RMS 网格图
图 4：RMS 均值柱状图（固定负载 + 变负载）

用法：
    python example_emg_rms_vload.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.visualization.emg_feature_plot import (
    plot_feature_vs_time_combined,
    plot_feature_vs_position_combined,
    plot_pos_vel_emg_feature_grid_combined,
    plot_feature_bar_combined,
)

FEATURE = 'rms'
LABEL = 'RMS (mV)'


def main():
    subject = Subject('../config/20250512_squat_Yuchen.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    pipeline.run(include_xsens=False)
    vload_results = pipeline.run_vload()

    muscles = subject.musc_label[:6]

    plot_feature_vs_time_combined(pipeline.results, vload_results, muscles,
                                  feature=FEATURE, feature_label=LABEL)
    plot_feature_vs_position_combined(pipeline.results, vload_results, muscles,
                                     feature=FEATURE, feature_label=LABEL)
    plot_pos_vel_emg_feature_grid_combined(pipeline.results, vload_results, ['VL', 'RF'],
                                           subject=subject, feature=FEATURE, feature_label=LABEL)
    plot_feature_bar_combined(pipeline.results, vload_results, muscles,
                             feature=FEATURE, feature_label=LABEL)

    plt.show()


if __name__ == '__main__':
    main()