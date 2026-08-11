"""
digitaltwin/data/insole/

鞋垫数据处理包。包括：

    io.py           文件读取、压力图解析、重采样、路径解析
    timebase.py     insole_time_offset 的唯一入口
    sync.py         互相关时间标定、深蹲段识别
    orientation.py  足趾端 / 内外侧方向诊断
    calibration.py  批量标定、写回 json、预检查
    processor.py    InsoleProcessor 门面（向后兼容的静态方法集）

时间轴只有一条规则：

    corrected_insole_time = raw_insole_time + insole_time_offset

offset 由 example_insole_sync_offset.py 标定后写在 json 的采集组里。没有它
就用原始相对时间并 beauty_print 告警，不再有任何基于 info.csv 时间戳的
猜测性对齐。
"""
from .processor import InsoleProcessor
from .timebase import (OFFSET_KEY, apply_time_offset, has_offset,
                       resolve_time_offset)
from .sync import estimate_time_offset, squat_phase_mask
from .orientation import diagnose_orientation, diagnose_side_orientation
from .calibration import (calibrate_insole_offsets, check_insole_offsets,
                          print_report)

__all__ = [
    'InsoleProcessor',
    'OFFSET_KEY',
    'apply_time_offset',
    'has_offset',
    'resolve_time_offset',
    'estimate_time_offset',
    'squat_phase_mask',
    'diagnose_orientation',
    'diagnose_side_orientation',
    'calibrate_insole_offsets',
    'check_insole_offsets',
    'print_report',
]