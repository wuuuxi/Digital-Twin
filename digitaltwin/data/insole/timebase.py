"""
digitaltwin/data/insole/timebase.py

鞋垫时间轴的唯一入口。

约定
----
    corrected_insole_time = raw_insole_time + insole_time_offset

offset 由 sync.estimate_time_offset 用互相关标定，写在 json 采集组里：

    "insole_time_offset": 1.234
    "insole_time_offset_l": 1.230   # 可选，左右分别标定时用
    "insole_time_offset_r": 1.238

为什么只剩这一条路径
--------------------
旧版还有一条 info.csv (measurement_date) + robot 第一帧时间戳的路径。
它只有分钟级可信度：鞋垫软件落盘延迟、机器人时间戳定义、两台电脑的时钟
漂移全都残留在里面，实测残差到秒级。而一个下蹲相只有 1-2 s，秒级残差
就足以让两条曲线看上去毫不相干。更糟的是它【总会给出一个数】，于是
“没标定”被伪装成“已对齐”，错位反而更难被发现。所以现在的规则是：
有 offset 就用 offset，没有就用原始相对时间并显式告警。
"""
import numpy as np

from digitaltwin.utils.logger import beauty_print


OFFSET_KEY = 'insole_time_offset'
SIDE_OFFSET_KEYS = {'l': 'insole_time_offset_l',
                    'r': 'insole_time_offset_r'}


def _as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def resolve_time_offset(file_info, side=None, load_key='', verbose=True,
                        warn=True):
    """从采集组配置中取出该侧应该用的时间偏移 (s)。

    优先级：insole_time_offset_<side> > insole_time_offset > None。

    Parameters
    ----------
    file_info : dict -- modeling_file.data[load_key]
    side : {'l', 'r', None}
    load_key : str -- 仅用于告警文案
    warn : bool -- 缺失时是否 beauty_print 告警

    Returns
    -------
    float or None -- None 表示这组没标定，调用方应使用原始相对时间
    """
    info = file_info or {}

    if side:
        key = SIDE_OFFSET_KEYS.get(str(side).lower())
        if key:
            value = _as_float(info.get(key))
            if value is not None:
                return value

    value = _as_float(info.get(OFFSET_KEY))
    if value is not None:
        return value

    if warn:
        beauty_print(
            '组 {}：配置中没有 {}，鞋垫将使用文件自带的原始相对时间，'
            '与机器人时钟【未对齐】，画出来会整体错位。\n'
            '请先运行 examples/data_analysis/insole/'
            'example_insole_sync_offset.py 做互相关标定。'.format(
                load_key or '?', OFFSET_KEY),
            type='warning')
    return None


def has_offset(file_info):
    """该采集组是否已经标定过（任意一个 offset 字段可解析即算）。"""
    info = file_info or {}
    keys = [OFFSET_KEY] + list(SIDE_OFFSET_KEYS.values())
    return any(_as_float(info.get(k)) is not None for k in keys)


def apply_time_offset(time, time_offset, verbose=True, label=''):
    """把 offset 加到原始时间轴上。offset 为 None 时原样返回。"""
    time = np.asarray(time, dtype=float)
    offset = _as_float(time_offset)

    if time_offset is not None and offset is None:
        beauty_print(
            '  [Insole] insole_time_offset 无法解析: {}，'
            '本次使用原始相对时间。'.format(time_offset),
            type='warning')
        return time

    if offset is None:
        return time

    if verbose:
        print('  [Insole] 应用 insole_time_offset={:+.3f}s  {}'.format(
            offset, label))
    return time + offset