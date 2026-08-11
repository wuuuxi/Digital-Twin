"""
digitaltwin/data/insole/processor.py

InsoleProcessor 门面类。

实现已经拆到 io / timebase / sync / orientation 四个模块里，这里只把它们
重新导出成静态方法，让已有的 InsoleProcessor.load(...) 写法不用改。
新代码建议直接用子模块函数，语义更清楚：

    from digitaltwin.data.insole import io as insole_io
    from digitaltwin.data.insole.sync import estimate_time_offset
"""
from . import io as insole_io
from . import orientation
from . import sync


class InsoleProcessor:
    """鞋垫数据处理的统一入口（薄封装，不含逻辑）。"""

    # --- 常量 ---
    MAP_HEADER_ROWS = insole_io.MAP_HEADER_ROWS
    MAP_MIN_FORCE_N = insole_io.MAP_MIN_FORCE_N

    SYNC_DT = sync.SYNC_DT
    SYNC_MAX_LAG = sync.SYNC_MAX_LAG
    SYNC_CORR_WIN = sync.SYNC_CORR_WIN
    SYNC_CORR_THR = sync.SYNC_CORR_THR
    SYNC_MIN_SEG = sync.SYNC_MIN_SEG
    SYNC_FORCE_FRAC = sync.SYNC_FORCE_FRAC
    SYNC_DETREND_WIN = sync.SYNC_DETREND_WIN
    SYNC_MIN_CORR = sync.SYNC_MIN_CORR

    ORIENT_CONTACT_FRAC = orientation.ORIENT_CONTACT_FRAC
    ORIENT_FOOTPRINT_FRAC = orientation.ORIENT_FOOTPRINT_FRAC
    ORIENT_FOOTPRINT_ABS = orientation.ORIENT_FOOTPRINT_ABS
    ORIENT_END_FRAC = orientation.ORIENT_END_FRAC
    ORIENT_MIDLINE_GUARD_CM = orientation.ORIENT_MIDLINE_GUARD_CM

    # --- IO ---
    load = staticmethod(insole_io.load)
    load_pressure_map = staticmethod(insole_io.load_pressure_map)
    read_pressure_map_header = staticmethod(insole_io.read_pressure_map_header)
    resolve_insole_path = staticmethod(insole_io.resolve_insole_path)
    resample = staticmethod(insole_io.resample)
    resample_nan_safe = staticmethod(insole_io.resample_nan_safe)

    # --- 同步 ---
    estimate_time_offset = staticmethod(sync.estimate_time_offset)
    squat_phase_mask = staticmethod(sync.squat_phase_mask)

    # --- 方向诊断 ---
    mean_contact_pressure = staticmethod(orientation.mean_contact_pressure)
    peak_contact_pressure = staticmethod(orientation.peak_contact_pressure)
    diagnose_side_orientation = staticmethod(orientation.diagnose_side_orientation)
    diagnose_orientation = staticmethod(orientation.diagnose_orientation)