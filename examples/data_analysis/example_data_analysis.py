"""
调试阶段：数据处理 + 对齐 / 运动切片 / 3D散点可视化 / 平均曲线绘制 / 肌肉分析 / 位置误差分析

用法：
    python example_data_analysis.py

说明：
    target_muscles 使用短名（不含 emg_ 前缀），如 ['VL', 'FibLon']。
    程序会自动匹配实际列名（支持 LFibLon / RFibLon 等变体命名）。
    - 只用单块肌肉的函数（运动切片、3D散点）：取列表第一个
    - 其余分析/可视化函数：使用全部肌肉
    - 可通过 INCLUDE_INSOLE_GRF 可选加入鞋垫 GRF 传感器数据：
      grf_l / grf_r 会从 modeling_file.data[*].insole_file_l / insole_file_r
      读取，并插值到 aligned_data / cutted_data 的 time 轴。
"""
import matplotlib.pyplot as plt
from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.data.insole import check_insole_offsets

# ============================================================
#  ★ 在此修改目标肌肉（短名，无需 emg_ 前缀）
# ============================================================
target_muscles = ['VL', 'FibLon', 'RF']

# ============================================================
#  ★ 可选：加入其他传感器信息
# ============================================================
# True 时会读取 modeling_file.data[*].insole_file_l / insole_file_r，
# 并在 aligned_data / cutted_data 中加入 grf_l / grf_r 两列。
INCLUDE_INSOLE_GRF = True

# 鞋垫时间轴：只认 json 采集组里的 insole_time_offset（由
# example_insole_sync_offset.py 互相关标定得到）。本脚本会在跑流水线
# 前先用 check_insole_offsets 把缺标定的组一次性报出来。

EXTRA_SENSOR_COLS = ['grf_l', 'grf_r']


# 用带 _xsens 的那份 config：只有它含全部 9 组（IM-1 / IK-0.3 / IK-0.15
# 与 6 个定负载组）。之前指向 20260513_squat_FTS09.json，那份里没有
# 等长 / 等速组，所以本脚本从来看不到它们。
CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'


def main():
    # subject = Subject('config/20251009_BenchPress_Yuetian.json')
    subject = Subject(CONFIG_FILE)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    if INCLUDE_INSOLE_GRF:
        check_insole_offsets(subject)

    results = pipeline.run(
        include_xsens=False,
        include_insole=INCLUDE_INSOLE_GRF)

    # 将短名转为 emg_ 前缀形式，交由 _resolve_muscle_cols 做数据列匹配
    emg_muscles = [f'emg_{m}' for m in target_muscles]
    emg_single  = emg_muscles[:1]   # 仅需单块肌肉的函数使用

    # 调试可视化
    alignment_cols = emg_muscles + (EXTRA_SENSOR_COLS if INCLUDE_INSOLE_GRF else [])
    pipeline.visualize_alignment(target_cols=alignment_cols)     # 对齐可视化
    pipeline.visualize_movement_segments(                            # 运动切片
        target_muscles=emg_single,
        extra_sensor_cols=(EXTRA_SENSOR_COLS if INCLUDE_INSOLE_GRF else []))
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