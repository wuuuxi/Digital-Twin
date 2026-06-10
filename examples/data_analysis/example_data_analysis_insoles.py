"""
example_data_analysis_insoles.py

专门检查鞋垫 GRF 与机器人 / EMG 的对齐情况。

与 example_data_analysis.py 类似，本脚本只画：
  1. 对齐可视化图 data_alignment.png
  2. 运动切片图 movement_segmentation.png

必包含额外传感器列：
  - grf_l
  - grf_r

鞋垫时间处理已经整理到 digitaltwin.data.insole_processor.InsoleProcessor
和 MultiLoadPipeline 中，作为鞋垫数据的默认处理方式：

  - 默认 USE_INSOLE_INFO_TIMESTAMP = True；
  - 读取鞋垫文件同目录下的 info.csv；
  - 从 I2 或 measurement_date 字段读取测量起始时间；
  - 读取 robot_file 第一帧时间作为对齐零点；
  - 忽略日期和小时，只比较“分钟:秒.毫秒”；
  - 将鞋垫 time 修正为相对 robot 第一帧的时间；
  - 如需退回鞋垫文件原始相对时间，将 USE_INSOLE_INFO_TIMESTAMP 改为 False。
"""
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../config/20260513_squat_FTS09.json'

# 运动切片图中的 EMG 行。只用第一块肌肉作为代表。
TARGET_MUSCLES = ['VL', 'FibLon', 'RF']

# 本脚本固定包含鞋垫 GRF
INCLUDE_INSOLE_GRF = True
EXTRA_SENSOR_COLS = ['grf_l', 'grf_r']

# 鞋垫时间戳处理：默认 True。
# True  = 使用 info.csv measurement_date + robot_file 第一帧时间修正鞋垫时间；
# False = 退回鞋垫文件原始相对时间。
USE_INSOLE_INFO_TIMESTAMP = True


def main():
    subject = Subject(CONFIG_FILE)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    results = pipeline.run(
        include_xsens=False,
        include_insole=INCLUDE_INSOLE_GRF,
        use_insole_info_timestamp=USE_INSOLE_INFO_TIMESTAMP,
    )

    if not results:
        print('没有可用处理结果。')
        return

    # 将短名转为 emg_ 前缀形式，交由 _resolve_muscle_cols 做实际列名匹配
    emg_muscles = [f'emg_{m}' for m in TARGET_MUSCLES]
    emg_single = emg_muscles[:1]

    # 1) 对齐可视化：必包含 grf_l / grf_r
    alignment_cols = emg_muscles + EXTRA_SENSOR_COLS
    pipeline.visualize_alignment(target_cols=alignment_cols)

    # 2) 运动切片图：grf_l / grf_r 作为额外独立行
    pipeline.visualize_movement_segments(
        target_muscles=emg_single,
        extra_sensor_cols=EXTRA_SENSOR_COLS,
    )

    plt.show()


if __name__ == '__main__':
    main()