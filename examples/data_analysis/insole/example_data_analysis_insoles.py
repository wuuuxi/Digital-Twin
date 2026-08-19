"""
example_data_analysis_insoles.py

专门检查鞋垫 GRF 与机器人 / EMG 的对齐情况。

与 example_data_analysis.py 类似，本脚本只画：
  1. 对齐可视化图 data_alignment.png
  2. 运动切片图 movement_segmentation.png

必包含额外传感器列：
  - grf_l
  - grf_r

鞋垫时间对齐
----------
本脚本不再使用 info.csv 的 measurement_date 做对齐。那条路径只有分钟级
可信度（鞋垫软件落盘延迟、机器人时间戳定义、两台电脑时钟漂移全都
残留在里面），而一个下蹲相只有 1-2 s，秒级残差就足以让两条曲线
看上去毫不相干。

现在鞋垫时间轴只由 config 里每个采集组的字段决定：

    "insole_time_offset": 1.234        # corrected = raw + offset

它由 example_insole_sync_offset.py 在深蹲段上做互相关标定得到，精度到
亚采样点。若某组没有这个字段，本脚本会在跑之前先 beauty_print 告警，
并以鞋垫文件原始相对时间继续——此时 grf 与 force 的时相不可信。

先跑：
    python example_insole_sync_offset.py
再跑本脚本。
"""
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.utils.logger import beauty_print


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../../config/20250409_squat_NCMP001_xsens.json'

# 运动切片图中的 EMG 行。只用第一块肌肉作为代表。
TARGET_MUSCLES = ['VL', 'FibLon', 'RF']

# 本脚本固定包含鞋垫 GRF
INCLUDE_INSOLE_GRF = True
EXTRA_SENSOR_COLS = ['grf_l', 'grf_r']


def check_insole_offsets(subject):
    """检查每个采集组是否已经标定过 insole_time_offset。

    放在 run() 之前先跑一遍，而不是等到逐组处理时才报：标定缺失是
    【配置问题】，应该在看到任何图之前就知道哪几组不能信。

    Returns
    -------
    list[str] -- 尚未标定的组名
    """
    missing = []
    for key, info in (subject.modeling_data or {}).items():
        has_offset = any(
            info.get(k) is not None
            for k in ('insole_time_offset',
                      'insole_time_offset_l',
                      'insole_time_offset_r'))
        if not has_offset:
            missing.append(str(key))

    if missing:
        beauty_print(
            '以下采集组在 config 中没有 insole_time_offset：{}\n'
            '  鞋垫与机器人很可能没有对齐。这几组将按鞋垫文件原始相对时间绘制，\n'
            '  grf_l / grf_r 与 force_l / force_r 的时相不可信，不要据此下结论。\n'
            '  请先跑 example_insole_sync_offset.py 完成互相关标定。'.format(
                ', '.join(missing)),
            type="warning")
    else:
        print('[Insole] 全部 {} 组均已标定 insole_time_offset。'.format(
            len(subject.modeling_data or {})))

    return missing


def main():
    subject = Subject(CONFIG_FILE)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # 对齐完全依赖 config 里的 insole_time_offset，先查一遍
    check_insole_offsets(subject)

    results = pipeline.run(
        include_xsens=False,
        include_insole=INCLUDE_INSOLE_GRF,
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