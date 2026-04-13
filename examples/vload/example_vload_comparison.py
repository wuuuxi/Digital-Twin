"""
变负载综合对比分析示例

汇集所有变负载 vs 固定负载的对比图：
  图 1: 关节角 vs Position 散点图（左右对比）
  图 2: 关节角均值柱状图 + |R-L| 差异
  图 3: 关节角速度 vs Position 散点图（左右对比）
  图 4: 关节角速度均值柱状图 + |R-L| 差异
  图 5: Robot 位置/速度/加速度均值柱状图
  图 6: MDF 均值柱状图（固定 + 变负载）
  图 7: RMS 均值柱状图（固定 + 变负载）
  图 8: 肌肉激活均值柱状图（固定 + 变负载）

用法：
    python example_vload_comparison.py
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.visualization.xsens_plot import (
    _get_motion_defaults,
    plot_joint_scatter_lr,
    plot_joint_bar_lr,
    plot_joint_vel_scatter_lr,
    plot_joint_vel_bar_lr,
)
from digitaltwin.visualization.vload_comparison_plot import (
    plot_robot_kinematics_bar,
    plot_emg_activation_bar,
)
from digitaltwin.visualization.emg_feature_plot import (
    plot_feature_bar_combined,
)


def main():
    # --- 配置 ---
    subject = Subject('../config/20250409_squat_NCMP001_xsens.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # --- 数据加载 ---
    results = pipeline.run(include_xsens=True)
    vload_results = pipeline.run_vload()

    # --- 运动类型默认参数 ---
    defaults = _get_motion_defaults(subject.target_motion)
    joint_bases = defaults['joint_bases']
    target_emg = defaults['target_emg']
    muscles = subject.musc_label[:6]

    # ==================================================================
    #  图 1: 关节角 vs Position 散点图（左右对比）
    # ==================================================================
    plot_joint_scatter_lr(results, joint_bases, vload_results)

    # ==================================================================
    #  图 2: 关节角均值柱状图 + |R-L| 差异
    # ==================================================================
    plot_joint_bar_lr(results, joint_bases, vload_results)

    # ==================================================================
    #  图 3: 关节角速度 vs Position 散点图（左右对比）
    # ==================================================================
    plot_joint_vel_scatter_lr(results, joint_bases, vload_results)

    # ==================================================================
    #  图 4: 关节角速度均值柱状图 + |R-L| 差异
    # ==================================================================
    plot_joint_vel_bar_lr(results, joint_bases, vload_results)

    # ==================================================================
    #  图 5: Robot 位置/速度/加速度均值柱状图
    # ==================================================================
    plot_robot_kinematics_bar(results, vload_results)

    # ==================================================================
    #  图 6: MDF 均值柱状图（固定 + 变负载）
    # ==================================================================
    plot_feature_bar_combined(results, vload_results, muscles,
                             feature='mdf', feature_label='MDF (Hz)')

    # ==================================================================
    #  图 7: RMS 均值柱状图（固定 + 变负载）
    # ==================================================================
    plot_feature_bar_combined(results, vload_results, muscles,
                             feature='rms', feature_label='RMS (mV)')

    # ==================================================================
    #  图 8: 肌肉激活均值柱状图（固定 + 变负载）
    # ==================================================================
    plot_emg_activation_bar(results, muscles, vload_results)

    plt.show()


if __name__ == '__main__':
    main()