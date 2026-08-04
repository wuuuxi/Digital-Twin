"""
Xsens 运动捕捉 + Robot + EMG 联合分析示例（仅固定负载）

职责：验证 Xsens 与机器人 / EMG 的同步性，并以图表形式展示原始数据的
简单分析结果。这里只看【原始 Xsens】，不涉及 .mot 转换的正确性——
.mot 的校验（含“原始左右对称性是否与转换后一致”）属于
example_validate_mot.py 的 C8，不在本脚本。

图 1：对齐可视化     图 2：运动切片     图 3：位置散点
图 4：多关节角左右散点  图 5：关节角均值柱状图 + 左右差异

图 1/2/3 画 JOINT_TO_PLOT 指定的关节，左右两侧同图对比：
  红实线 = 右侧，蓝虚线 = 左侧，且强制共用同一 Y 轴与归一化系数，
  否则自动缩放会把幅值差异抹平。

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


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'

# 图 1/2/3 画哪个关节角。None = 用该运动类型的默认值
# （深蹲 knee_angle_r，卧推 elbow_flex_r）。
# 深蹲可选：hip_flexion / hip_adduction / hip_rotation /
#           knee_angle / ankle_angle / subtalar_angle / mtp_angle
# 卧推可选：arm_flex / arm_add / arm_rot / elbow_flex
JOINT_TO_PLOT = 'ankle_angle'

# 图 4/5 画哪些关节。None = 用该运动类型的默认列表。
JOINT_BASES_TO_PLOT = None
# JOINT_BASES_TO_PLOT = 'hip_flexion'


def main():
    subject = Subject(CONFIG_FILE)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    results = pipeline.run(include_xsens=True)

    defaults = _get_motion_defaults(subject.target_motion)
    target_emg = defaults['target_emg']
    muscle_col = target_emg[0]
    joint_bases = JOINT_BASES_TO_PLOT or defaults['joint_bases']

    xsens_joint = JOINT_TO_PLOT or defaults['xsens_joint']
    if not xsens_joint.endswith(('_r', '_l')):
        # 只是给个基准侧；plot_* 内部会自动把左右两侧都画出来
        xsens_joint = f'{xsens_joint}_r'
    print(f'图 1/2/3 关节: {xsens_joint[:-2]}（左右同时绘制）')

    groups = build_data_groups(results)          # 仅固定负载

    plot_alignment(groups, target_emg, xsens_joint)
    plot_movement_segments(groups, muscle_col, xsens_joint)
    plot_position_scatter(groups, muscle_col, xsens_joint)
    plot_joint_scatter_lr(results, joint_bases)
    plot_joint_bar_lr(results, joint_bases)

    plt.show()


if __name__ == '__main__':
    main()