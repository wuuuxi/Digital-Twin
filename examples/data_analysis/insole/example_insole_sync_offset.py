"""
example_insole_sync_offset.py

标定鞋垫与机器人的时间差，写回 json，并画三联诊断图。

约定：
    corrected_insole_time = raw_insole_time + insole_time_offset

跑完之后，json 的每个采集组里只会多出一个字段：
    "insole_time_offset": 1.234

标定会复用机器人位置/速度切片得到的动作窗口，并要求候选 offset 覆盖
足够比例的动作样本，避免周期信号只重合一两个动作时选中错误相关峰。

标定质量（corr / 拟合时长 / overlap / 段数 / 是否写回）只在终端的 print_report
表里看，不写进 json。corr 低于 MIN_CORR 的组不会被写回，所以
配置里出现 insole_time_offset 就意味着它通过了阀值。

之后所有用到鞋垫的分析都会自动读取 insole_time_offset，不需要再标。
只有重新导出了鞋垫或机器人数据时才需要重跑。

主体实现在 digitaltwin/data/insole/，本文件只负责改参数和调用。
"""
import os
import sys

# 仓库根目录：examples/data_analysis/insole/ 往上三层
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT)

from digitaltwin.data.insole import calibrate_insole_offsets, print_report
from digitaltwin.subject import Subject


# ----------------------------- 参数 -----------------------------
# 用绝对路径，不依赖当前工作目录：相对路径是相对于 cwd 而不是
# 脚本所在目录的，在 IDE 里从项目根目录跑就会找不到文件。
CONFIG_NAME = '20260513_squat_FTS09_xsens.json'
CONFIG_FILE = os.path.join(ROOT, 'examples', 'config', CONFIG_NAME)

LOAD_KEYS = None      # None = 全部采集组；也可以写 ['20', '35']
WRITE_JSON = True     # 是否把 offset 写回配置文件
MIN_CORR = 0.5        # 低于此相关系数只报告、不写回
MAX_LAG = 30.0        # 滞后搜索范围 ±(s)；峰值落在边界时加大它
CORR_THR = 0.5        # 左右一致性门槛，决定哪些段算深蹲
MIN_OVERLAP_FRAC = 0.4  # 候选 offset 至少覆盖的鞋垫动作样本比例
PLOT = True           # 是否画诊断图
# ----------------------------------------------------------------


def main():
    subject = Subject(CONFIG_FILE)

    report = calibrate_insole_offsets(
        subject,
        load_keys=LOAD_KEYS,
        write_json=WRITE_JSON,
        min_corr=MIN_CORR,
        max_lag=MAX_LAG,
        corr_thr=CORR_THR,
        min_overlap_frac=MIN_OVERLAP_FRAC,
        plot=PLOT,
    )

    print_report(report)


if __name__ == '__main__':
    main()
