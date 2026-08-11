"""
digitaltwin/data/insole/io.py

鞋垫原始文件读取与重采样。

支持两类文件：

1. 汇总力文件 `Medilogic Insoles-XT Force.csv`
   CSV，跳过前 3 行，第 1 列 time (s)，第 2 列 value (N，正方向向上)。
   由 load() 读取。

2. 逐点压力图 `... Medilogic Insoles G2. Foot.csv`
   signal_matrix 格式，由 load_pressure_map() 读取。
     第 1 行: 元数据字段名
     第 2 行: 元数据值 (frequency / count / units / cell_count_x /
              cell_count_y / cell_size_mm_x / cell_size_mm_y)
     第 3 行: 空行
     第 4 行: 列头 time,x1..xN
     之后每一帧 = cell_count_y 行 x cell_count_x 列压强。

时间轴约定
----------
本模块【只】负责把 json 里标定好的 insole_time_offset 加到鞋垫的原始
相对时间上（见 timebase.apply_time_offset）。没有 offset 时就保持原始
相对时间，不做任何猜测性对齐。info.csv / measurement_date 那条路径已经
整个删除：它只有分钟级可信度，会把“没标定”伪装成“已对齐”。
"""
import csv
import io as _io
import os

import numpy as np

from digitaltwin.utils.logger import beauty_print
from .timebase import apply_time_offset


# 鞋垫软件可能输出 UTF-8 (with/without BOM)、Latin-1 或 CP1252，
# Windows 中文环境还可能是 GBK
ENCODINGS = ('utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'gbk')

MAP_HEADER_ROWS = 4      # 元数据名 / 元数据值 / 空行 / 列头
MAP_MIN_FORCE_N = 20.0   # 求 COP 的最小总力，低于此值视为悬空


def _log(msg, verbose=True):
    if verbose:
        print(msg)


def read_lines_any_encoding(file_path, verbose=True):
    """按常见编码逐一尝试读取文本行，返回 (lines, encoding)。"""
    for enc in ENCODINGS:
        try:
            with open(file_path, 'r', encoding=enc) as fh:
                return fh.readlines(), enc
        except (UnicodeDecodeError, LookupError):
            continue
    _log('  [Insole] 无法以任何已知编码读取: ' + str(file_path), verbose)
    return None, None


def resolve_insole_path(subject, insole_file):
    """把 modeling_file.data 里写的鞋垫相对路径解析成实际路径。

    查找顺序：绝对路径 -> folder/insole_folder/file -> folder/file。
    找不到返回 None（由调用方决定告警还是跳过）。
    """
    if not insole_file:
        return None
    if os.path.isabs(insole_file):
        return insole_file if os.path.exists(insole_file) else None

    modeling = subject.config.get('modeling_file', {})
    insole_folder = modeling.get('insole_folder', 'Sorted')
    if os.path.isabs(insole_folder):
        candidates = [
            os.path.join(insole_folder, insole_file),
            os.path.join(subject.folder, insole_file),
        ]
    else:
        candidates = [
            os.path.join(subject.folder, insole_folder, insole_file),
            os.path.join(subject.folder, insole_file),
        ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ------------------------------------------------------------------
#  汇总力文件
# ------------------------------------------------------------------

def load(file_path, verbose=True, time_offset=None):
    """读取单个鞋垫汇总力 CSV。

    Parameters
    ----------
    file_path : str
    verbose : bool
    time_offset : float, optional
        json 中标定好的 insole_time_offset (s)。给出时返回
        time + time_offset；为 None 时返回鞋垫文件的原始相对时间。

    Returns
    -------
    time  : np.ndarray or None
    force : np.ndarray or None  -- +Y 向上的地面支撑力 (N)
    """
    if not os.path.exists(file_path):
        _log('  [Insole] 文件不存在: ' + str(file_path), verbose)
        return None, None

    raw_lines, used_enc = read_lines_any_encoding(file_path, verbose=verbose)
    if raw_lines is None:
        return None, None

    try:
        # 跳过前 3 行表头
        content = ''.join(raw_lines[3:])
        data = np.genfromtxt(_io.StringIO(content), delimiter=',',
                             usecols=(0, 1), invalid_raise=False)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) == 0:
            _log('  [Insole] 文件无有效数据: ' + str(file_path), verbose)
            return None, None

        time = apply_time_offset(data[:, 0], time_offset, verbose=verbose,
                                 label=os.path.basename(file_path))
        force = data[:, 1]
        _log('  [Insole] 已加载 ({}): {}  ({} frames)'.format(
            used_enc, os.path.basename(file_path), len(data)), verbose)
        return time, force
    except Exception as e:
        _log('  [Insole] 读取失败: {}'.format(e), verbose)
        return None, None


# ------------------------------------------------------------------
#  逐点压力图
# ------------------------------------------------------------------

def read_pressure_map_header(file_path, verbose=True):
    """读取逐点压力图的元数据行。

    Returns
    -------
    dict or None
        name / frequency / count / units / begin_time /
        n_cols / n_rows / cell_dx_cm / cell_dy_cm /
        cell_area_cm2 / width_cm / length_cm
    """
    lines, _ = read_lines_any_encoding(file_path, verbose=verbose)
    if lines is None or len(lines) < 2:
        return None

    try:
        keys = next(csv.reader([lines[0].rstrip()]))
        vals = next(csv.reader([lines[1].rstrip()]))
    except Exception as e:
        _log('  [InsoleMap] 元数据行解析失败: ' + str(e), verbose)
        return None

    meta_raw = {}
    for k, v in zip(keys, vals):
        meta_raw[str(k).strip()] = str(v).strip()

    def as_float(key, default=None):
        try:
            return float(meta_raw.get(key, ''))
        except (TypeError, ValueError):
            return default

    def as_int(key, default=None):
        value = as_float(key, None)
        return default if value is None else int(round(value))

    n_cols = as_int('cell_count_x')
    n_rows = as_int('cell_count_y')
    dx_mm = as_float('cell_size_mm_x')
    dy_mm = as_float('cell_size_mm_y')

    if not n_cols or not n_rows or not dx_mm or not dy_mm:
        _log('  [InsoleMap] 元数据缺少网格尺寸字段: ' + str(file_path), verbose)
        return None

    dx_cm = dx_mm / 10.0
    dy_cm = dy_mm / 10.0

    return {
        'type': meta_raw.get('type', ''),
        'name': meta_raw.get('name', ''),
        'frequency': as_float('frequency'),
        'count': as_int('count'),
        'units': meta_raw.get('units', ''),
        'begin_time': as_float('begin_time', 0.0),
        'n_cols': n_cols,
        'n_rows': n_rows,
        'cell_dx_cm': dx_cm,
        'cell_dy_cm': dy_cm,
        'cell_area_cm2': dx_cm * dy_cm,
        'width_cm': n_cols * dx_cm,
        'length_cm': n_rows * dy_cm,
    }


def _parse_pressure_frames(lines, n_rows, n_cols, verbose=True):
    """解析帧块，返回 (time, matrix)，matrix 形状 (n_frames, n_rows, n_cols)。

    行数不完整的帧会被丢弃并计数告警。
    """
    times = []
    frames = []
    state = {'rows': None, 'time': None, 'bad': 0}

    def close_frame():
        rows = state['rows']
        if rows is None:
            return
        if len(rows) == n_rows and state['time'] is not None:
            times.append(state['time'])
            frames.append(rows)
        else:
            state['bad'] += 1
        state['rows'] = None
        state['time'] = None

    for line in lines:
        stripped = line.strip()
        if stripped == '' or set(stripped) <= {','}:
            close_frame()
            continue

        try:
            parts = next(csv.reader([line.rstrip()]))
        except Exception:
            continue
        if not parts:
            continue

        head = parts[0].strip()

        if head != '':
            close_frame()
            try:
                state['time'] = float(head)
            except ValueError:
                state['time'] = None
                continue
            state['rows'] = []

        if state['rows'] is None:
            continue

        row = []
        for v in parts[1:1 + n_cols]:
            v = v.strip()
            try:
                row.append(float(v) if v != '' else 0.0)
            except ValueError:
                row.append(0.0)
        while len(row) < n_cols:
            row.append(0.0)

        state['rows'].append(row)

    close_frame()

    if state['bad']:
        beauty_print(
            '  [InsoleMap] 丢弃行数不完整的帧: {} 帧'.format(state['bad']),
            type='warning')

    if not frames:
        return None, None

    return np.asarray(times, dtype=float), np.asarray(frames, dtype=float)


def load_pressure_map(file_path, verbose=True, toe_first=True,
                      min_force=None, return_matrix=True, time_offset=None):
    """读取逐点压力图文件，返回总力与逐帧压心。

    Parameters
    ----------
    file_path : str
    toe_first : bool, default True
        网格行号 0 是否位于足趾端。判定依据见 orientation.py。
    min_force : float, optional
        求 COP 的最小总力阈值，默认 MAP_MIN_FORCE_N。低于阈值的帧 COP 为 nan。
    return_matrix : bool, default True
        是否保留完整压强矩阵（较占内存）。
    time_offset : float, optional
        json 标定好的 insole_time_offset (s)。

    Returns
    -------
    dict or None
        time / force / cop_ant / cop_lat / pressure / meta
    """
    if not os.path.exists(file_path):
        beauty_print('  [InsoleMap] 文件不存在: ' + str(file_path),
                     type='warning')
        return None

    meta = read_pressure_map_header(file_path, verbose=verbose)
    if meta is None:
        beauty_print('  [InsoleMap] 元数据解析失败: ' + str(file_path),
                     type='warning')
        return None

    lines, used_enc = read_lines_any_encoding(file_path, verbose=verbose)
    if lines is None:
        return None

    n_rows = meta['n_rows']
    n_cols = meta['n_cols']

    time, matrix = _parse_pressure_frames(
        lines[MAP_HEADER_ROWS:], n_rows, n_cols, verbose=verbose)

    if matrix is None:
        beauty_print('  [InsoleMap] 未解析到任何完整帧: ' + str(file_path),
                     type='warning')
        return None

    declared = meta['count']
    if declared and len(time) != declared:
        beauty_print(
            '  [InsoleMap] 帧数与元数据不一致: 解析 {} / 声明 {}'.format(
                len(time), declared),
            type='warning')

    # 压强 -> 力。units 含 cm2 时需乘单元面积
    units = str(meta.get('units', '')).lower().replace(' ', '')
    if 'cm2' in units or 'cm^2' in units:
        scale = meta['cell_area_cm2']
    elif 'mm2' in units or 'mm^2' in units:
        scale = meta['cell_dx_cm'] * meta['cell_dy_cm'] * 100.0
    else:
        scale = 1.0
        beauty_print(
            '  [InsoleMap] 未识别的单位 {}，按已是力处理，'
            '不乘单元面积'.format(meta.get('units')),
            type='warning')

    force = matrix.sum(axis=(1, 2)) * scale

    # 逐帧压心
    thr = MAP_MIN_FORCE_N if min_force is None else min_force
    xs = (np.arange(n_cols) + 0.5) * meta['cell_dx_cm']
    ys = (np.arange(n_rows) + 0.5) * meta['cell_dy_cm']

    tot = matrix.sum(axis=(1, 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        cop_row = (matrix.sum(axis=2) @ ys) / tot
        cop_col = (matrix.sum(axis=1) @ xs) / tot

    invalid = ~(force >= thr)
    cop_row[invalid] = np.nan
    cop_col[invalid] = np.nan

    # 行号 0 在足趾端时，距足跟端的前向距离 = 全长 - 行向坐标
    cop_ant_cm = (meta['length_cm'] - cop_row) if toe_first else cop_row

    # 边缘列若有明显压强，说明脚踩到垫边，力有溢出丢失
    col_mean = matrix.mean(axis=(0, 1))
    edge = max(col_mean[0], col_mean[-1])
    if col_mean.max() > 0 and edge > 0.05 * col_mean.max():
        beauty_print(
            '  [InsoleMap] 边缘列存在明显压强 (边缘 {:.3f} vs 峰值 {:.3f})，'
            '脚可能踩出感应区，总力与 COP 均会偏低'.format(
                edge, col_mean.max()),
            type='warning')

    time = apply_time_offset(time, time_offset, verbose=verbose,
                             label=os.path.basename(file_path))

    n_valid = int((~invalid).sum())
    _log('  [InsoleMap] 已加载 ({}): {}  ({} frames, {}x{} cells, '
         '有效 COP {} 帧)'.format(
             used_enc, os.path.basename(file_path), len(time),
             n_rows, n_cols, n_valid), verbose)

    result = {
        'time': time,
        'force': force,
        'cop_ant': cop_ant_cm / 100.0,
        'cop_lat': cop_col / 100.0,
        'meta': meta,
    }
    if return_matrix:
        result['pressure'] = matrix

    return result


# ------------------------------------------------------------------
#  重采样
# ------------------------------------------------------------------

def resample(time, force, target_times):
    """把鞋垫力信号线性插值到目标时间轴。"""
    return np.interp(target_times, time, force,
                     left=force[0], right=force[-1])


def resample_nan_safe(time, values, target_times, max_gap_s=None):
    """把含 nan 的信号（例如悬空帧的 COP）插值到目标时间轴。

    为什么不能直接用 resample：np.interp 不认识 nan，一个 nan 会沿着
    两侧的线性段污染邻域，把 COP 拉到任意位置。这里只用【有效样本】
    建插值，并且：
      - 目标点落在有效样本覆盖范围之外时返回 nan，不做外推；
      - 若某目标点两侧最近有效样本的间隔超过 max_gap_s，也返回 nan。

    Returns
    -------
    np.ndarray or None -- 全无有效样本时返回 None
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    tgt = np.asarray(target_times, dtype=float)
    n = min(t.size, v.size)
    if n < 2:
        return None
    t, v = t[:n], v[:n]

    ok = np.isfinite(t) & np.isfinite(v)
    if int(ok.sum()) < 2:
        return None
    t_ok, v_ok = t[ok], v[ok]
    order = np.argsort(t_ok)
    t_ok, v_ok = t_ok[order], v_ok[order]

    out = np.interp(tgt, t_ok, v_ok)
    out[(tgt < t_ok[0]) | (tgt > t_ok[-1])] = np.nan

    if max_gap_s is not None and max_gap_s > 0:
        idx = np.searchsorted(t_ok, tgt).clip(1, t_ok.size - 1)
        gap = t_ok[idx] - t_ok[idx - 1]
        out[gap > max_gap_s] = np.nan

    return out