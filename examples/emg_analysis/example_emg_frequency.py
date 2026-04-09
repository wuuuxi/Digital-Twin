"""
EMG 中值频率（MDF）分析示例

图 1：各肌肉的 MDF 随时间变化（灰色底层 + 彩色切片）
图 2：对齐切片后 MDF 与位置的关系散点图
图 3：3行×N列 Position-Velocity / EMG / MDF 网格
图 4：各肌肉 MDF 均值 ± 标准差柱状图

用法：
    python example_emg_frequency.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.visualization.emg_feature_plot import (
    plot_feature_vs_time,
    plot_feature_vs_position,
    plot_pos_vel_emg_feature_grid,
    plot_feature_bar_by_load,
)

FEATURE = 'mdf'
LABEL = 'MDF (Hz)'


def main():
    subject = Subject('../config/20250512_squat_Yuchen.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True
    pipeline.run(include_xsens=False)

    muscles = subject.musc_label[:6]

    plot_feature_vs_time(pipeline.results, muscles, feature=FEATURE, feature_label=LABEL)
    plot_feature_vs_position(pipeline.results, muscles, feature=FEATURE, feature_label=LABEL)
    plot_pos_vel_emg_feature_grid(pipeline.results, ['VL', 'RF'], subject=subject,
                                  feature=FEATURE, feature_label=LABEL)
    plot_feature_bar_by_load(pipeline.results, muscles, feature=FEATURE, feature_label=LABEL)

    plt.show()


if __name__ == '__main__':
    main()