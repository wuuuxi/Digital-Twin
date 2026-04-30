"""
变负载综合对比分析示例（无 Xsens 数据版）

与 example_vload_comparison_xsens.py 类似，但适用于没有 Xsens 数据的情况。
跳过所有关节角相关分析（图 1-4），仅保留 Robot 运动学和 EMG 相关分析。

  图 1: Robot 位置/速度/加速度均值柱状图
  图 2: MDF 均值柱状图（固定 + 变负载）
  图 3: RMS 均值柱状图（固定 + 变负载）
  图 4: 肌肉激活均值柱状图（固定 + 变负载）

用法：
    python example_vload_comparison.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.visualization.vload_comparison_plot import (
    plot_robot_kinematics_bar,
    plot_emg_activation_bar,
)
from digitaltwin.visualization.emg_feature_plot import (
    plot_feature_bar_combined,
)


def main():
    # --- 配置（不含 Xsens 的 config） ---
    # subject = Subject('../config/20250409_squat_NCMP001.json')
    subject = Subject('../config/20250512_squat_Yuchen.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # --- 数据加载（不加载 Xsens） ---
    results = pipeline.run(include_xsens=False)
    vload_results = pipeline.run_vload()

    muscles = subject.musc_label[:6]
    # muscles = ['TA', 'GL', 'SOL', 'FibLon', 'VL', 'RF', "VM", "Addl", "BF", "ST", "GlutMax", "GlutMed"]
    # target_muscles = ['GL', 'SOL', 'FibLon', 'VL', 'RF', "GlutMax"]

    # ==================================================================
    #  图 1: Robot 位置/速度/加速度均值柱状图
    # ==================================================================
    plot_robot_kinematics_bar(results, vload_results)

    # ==================================================================
    #  图 2: MDF 均值柱状图（固定 + 变负载）
    # ==================================================================
    plot_feature_bar_combined(results, vload_results, muscles,
                             feature='mdf', feature_label='MDF (Hz)')

    # ==================================================================
    #  图 3: RMS 均值柱状图（固定 + 变负载）
    # ==================================================================
    plot_feature_bar_combined(results, vload_results, muscles,
                             feature='rms', feature_label='RMS (mV)')

    # ==================================================================
    #  图 4: 肌肉激活均值柱状图（固定 + 变负载）
    # ==================================================================
    plot_emg_activation_bar(results, muscles, vload_results)

    plt.show()


if __name__ == '__main__':
    main()