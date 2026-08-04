'''
example_insole_map_check.py

校验逐点压力图与汇总力文件是否一致。

这是使用逐帧 COP 之前的必要前置校验。两个文件来自同一次采集：

  Medilogic Insoles-XT Force.csv
      软件给出的总力，目前管线里的 grf_l / grf_r 就来自它。
  Medilogic Insoles-XT Medilogic Insoles G2. Foot.csv
      逐点压强，求和乘单元面积后应得到同一个总力。

两者必须吻合。若不吻合，说明 Force.csv 里含有压力图没有的标定（或反之），
那么由压强分布算出的 COP 就不能直接拿去替换 .sto 里的接触点，
必须先从标定链查起。

检查项：
  M1 文件存在性    两类文件都能找到且能解析
  M2 时间轴一致性  采样率、时长、起止时刻是否匹配
  M3 波形相关    两路总力的相关系数
  M4 增益一致    过原点回归斜率应为 1.0
  M5 残差大小    相对 RMS 残差
  M6 饱和检查    单元格是否在上限堆积
另外输出 COP 的描述性统计（均值 / 标准差 / 范围 / 与力的相关），
供判断恒定接触点近似误差有多大。本脚本不修改任何数据。
'''
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from digitaltwin.data.insole_processor import InsoleProcessor
from digitaltwin.utils.logger import beauty_print
from digitaltwin.visualization.insole_plot import (
    plot_load_pressure_cop, plot_cop_across_loads,
    global_pressure_vmax, global_force_range)
from digitaltwin.analysis.result_analysis import get_action_windows
from digitaltwin.config_manager import filter_load_keys, safe_load_key

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'

LOAD_KEYS = None                 # None = 全部
EXCLUDE_LOAD_KEYS = []

MIN_FORCE_N = 20.0               # 参与统计的最小力，低于此值视为悬空
SLOPE_TOL = 0.05                 # 过原点回归斜率允许偏离 1.0 的量
MIN_CORR = 0.98                  # 最低相关系数
REL_RMS_TOL = 0.05               # 相对 RMS 残差上限
DURATION_TOL_S = 0.20            # 时长差异允许值
START_TOL_S = 0.10               # 起始时刻差异允许值
SAT_FRAC_WARN = 0.002            # 达到峰值 99% 的单元格占比告警阈值

TOE_FIRST = True                 # 网格行 0 位于足趾端
SIDES = (('l', 'insole_file_l', 'insole_map_l'),
         ('r', 'insole_file_r', 'insole_map_r'))

# 按负载模式筛选。None = 全部；('isotonic',) = 只跑定负载组；
# 也可以写 ('isokinetic', 'isometric') 只看等速与等长组。
# 用这个代替以前硬写的 EXCLUDE_LOAD_KEYS=['0.15','0.3']，
# 以后新增等长组不用再改脚本。
LOAD_MODES_FILTER = None

# 只用深蹲动作期间的帧做 COP 统计 / 饱和统计 / 热图。
# 试次前后受试者会走动、上下杠、调整站位，那些帧同样有完整的压力分布，
# 但不属于被评估的动作，会同时抬高 COP 摆动范围并污染组间比较。
# 窗口直接复用 pipeline 已有的动作切片，与 example_validate_mot.py 同一套。
# 注意：启用窗口后，鞋垫时间轴必须先对齐到机器人时钟，
# 所以两个鞋垫文件都改成 use_info_timestamp=True 并传入 robot_file。
RESTRICT_TO_SQUAT = True
SQUAT_MOVEMENT_TYPES = ('upward', 'downward')

# 等长（isometric）组杆不动，vel_l 恒为 0，movement_type 切不出任何片段，
# 因此改用「力超过阈值的连续窗口」划定发力区间。
# 阈值 = ISOMETRIC_FORCE_FRAC x 本组总力的 95 分位数。
ISOMETRIC_FORCE_FRAC = 0.3
ISOMETRIC_MIN_DURATION_S = 0.5

PLOT_HEATMAP = True              # 足底平均压强热图 + COP 轨迹，每组一张
PLOT_ACROSS_LOADS = True         # 各组 COP 均值对比图
PLOT_SAVE_DIR = None             # 给出路径则保存 png
PLOT_SHOW = True                 # 结束时统一 plt.show()


# ----------------------------------------------------------------------
# 路径工具
# ----------------------------------------------------------------------
def get_base_dir():
    '''向上找到含 digitaltwin 包的项目根目录。'''
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, 'digitaltwin')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.abspath(__file__))


def get_config_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, CONFIG_FILE)),
        os.path.abspath(os.path.join(here, '..', CONFIG_FILE)),
        os.path.abspath(CONFIG_FILE),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def resolve_insole_path(config, rel_path):
    '''folder + insole_folder + 相对路径。'''
    if not rel_path:
        return None
    if os.path.isabs(rel_path):
        return rel_path

    folder = config.get('folder', '')
    insole_folder = config.get('modeling_file', {}).get('insole_folder', '')

    candidates = [
        os.path.join(folder, insole_folder, rel_path),
        os.path.join(folder, rel_path),
        rel_path,
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.normpath(path)
    return os.path.normpath(candidates[0])


def get_load_keys(config):
    '''按负载模式筛选采集组。'''
    return filter_load_keys(config, load_keys=LOAD_KEYS,
                            modes=LOAD_MODES_FILTER,
                            exclude=EXCLUDE_LOAD_KEYS)


def describe_en(config, load_key):
    '''
    图上标题用的英文说明。

    config_manager.describe_load_key 返回的是中文（「等长」「等速」），
    matplotlib 默认字体 DejaVu Sans 没有汉字字形，传进 suptitle 就会
    变成方块并刷出 missing glyph 告警。之前只把绘图模块内部的字面量
    改成了英文，漏了调用方传进去的这一串，所以前三组（等长/等速）
    图里仍有中文。此处直接从 config 的 mode 字段拼英文。
    '''
    group = config.get('modeling_file', {}).get('data', {}).get(load_key, {})
    mode = str(group.get('mode', '') or '').lower()

    if mode == 'isometric':
        height = group.get('bar_height')
        tail = '' if height is None else \
            ' at bar height {:.2f} m'.format(float(height))
        return 'group {} - isometric{}'.format(load_key, tail)

    if mode == 'isokinetic':
        vel = group.get('target_velocity')
        tail = '' if vel is None else \
            ' capped at {:.2f} m/s'.format(float(vel))
        return 'group {} - isokinetic{}'.format(load_key, tail)

    kg = group.get('load_kg')
    tail = '' if kg is None else ' {:.0f} kg'.format(float(kg))
    return 'group {} - isotonic{}'.format(load_key, tail)


# ----------------------------------------------------------------------
# 判定记录
# ----------------------------------------------------------------------
def get_squat_windows(config_path, load_keys):
    '''
    {load_key: (t0, t1)}：每组动作的时间窗（机器人时钟）。

    切分逻辑不在这里实现，而是调 result_analysis.get_action_windows：
      - 定负载 / 等速组：用 movement_type 的 upward/downward 切片包络。
        等速只限了最高速度，杆照样上下走，这套切分依然适用。
      - 等长组：杆不动，速度过零点切不出片段，改用力阈值的连续区间。
    get_action_windows 还会自动发现「缓存里的组名是旧名」这种静默失败
    （如 0.3 -> IK-0.3 改名后）并重建缓存。
    '''
    if not RESTRICT_TO_SQUAT:
        return {}

    try:
        info = get_action_windows(
            config_path, load_keys,
            movement_types=SQUAT_MOVEMENT_TYPES,
            force_frac=ISOMETRIC_FORCE_FRAC,
            min_duration=ISOMETRIC_MIN_DURATION_S,
            debug=False)
    except Exception as exc:
        beauty_print(
            '无法取得动作窗口（{}）；本次 COP 统计与热图会包含试次前后的'
            '走动帧，均值与摆动范围都会偏大，不可用于跳组比较。'.format(exc),
            type='warning')
        return {}

    windows = {}
    for load_key, item in info.items():
        if item.get('window') is None:
            beauty_print(
                '组 {} 未取到动作窗口（{}），该组按整段统计'.format(
                    load_key, item.get('detail', '')),
                type='warning')
            continue
        print('  [切片] load={:<11} {:<9} {}'.format(
            load_key, item.get('source', ''), item.get('detail', '')))
        windows[str(load_key)] = item['window']
    return windows


def slice_result_to_window(result, window, load_key, side):
    '''
    把 load_pressure_map 的返回值裁到深蹲窗口。返回 (result, 说明)。

    如果鞋垫时间轴与窗口几乎不重叠，说明 info.csv 对齐失败（而不是
    真的没有深蹲），这种情况必须报错，不能静静地裁出一段空数据。
    '''
    if result is None:
        return result, '无数据'

    t = np.asarray(result['time'], dtype=float)
    if window is None:
        return result, '未启用窗口，按整段 {} 帧统计'.format(len(t))

    t0, t1 = window
    span = max(t1 - t0, 1e-9)
    overlap = min(t[-1], t1) - max(t[0], t0)

    if overlap <= 0 or overlap < 0.5 * span:
        beauty_print(
            '  [窗口] load={} {} : 鞋垫时间轴 {:.2f}-{:.2f}s 与深蹲窗口 '
            '{:.2f}-{:.2f}s 重叠只有 {:.0%}。info.csv 对齐很可能失败，'
            '本组退回整段统计，COP 均值不可用于跳组比较。'.format(
                load_key, side.upper(), t[0], t[-1], t0, t1,
                max(overlap, 0.0) / span),
            type='warning')
        return result, '窗口与数据不重叠，退回整段'

    mask = (t >= t0) & (t <= t1)
    if mask.sum() < 10:
        beauty_print(
            '  [窗口] load={} {} : 窗口内仅 {} 帧，退回整段统计'.format(
                load_key, side.upper(), int(mask.sum())),
            type='warning')
        return result, '窗口内样本不足，退回整段'

    out = dict(result)
    for key in ('time', 'force', 'cop_ant', 'cop_lat'):
        arr = result.get(key)
        if arr is not None:
            out[key] = np.asarray(arr)[mask]
    if result.get('pressure') is not None:
        out['pressure'] = np.asarray(result['pressure'])[mask]

    return out, '深蹲窗口 {:.2f}-{:.2f}s，保留 {}/{} 帧 ({:.0%})'.format(
        t0, t1, int(mask.sum()), len(t), mask.mean())


class Verdicts:
    def __init__(self):
        self.items = []

    def add(self, name, load_key, side, ok, detail):
        self.items.append({
            'name': name, 'load': load_key, 'side': side,
            'ok': bool(ok), 'detail': detail,
        })
        tag = '[PASS]' if ok else '[FAIL]'
        line = '  {} {:<12} load={:<5} {} : {}'.format(
            tag, name, load_key, side.upper(), detail)
        if ok:
            print(line)
        else:
            beauty_print(line, type='warning')

    def summary(self):
        failed = [it for it in self.items if not it['ok']]
        print('')
        print('=' * 78)
        print('校验汇总: 共 {} 项，通过 {}，未通过 {}'.format(
            len(self.items), len(self.items) - len(failed), len(failed)))
        print('=' * 78)
        if not failed:
            print('全部通过。压力图积分与 Force.csv 一致，'
                  '可以放心用压力图求逐帧 COP。')
            return
        for it in failed:
            beauty_print('  {:<12} load={:<5} {} : {}'.format(
                it['name'], it['load'], it['side'].upper(), it['detail']),
                type='warning')
        print('')
        beauty_print('存在未通过项。在查清之前，不要用压力图算出的 COP '
                     '去替换 .sto 里的接触点。', type='warning')


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _fs_from_time(time):
    if time is None or len(time) < 3:
        return float('nan')
    dt = np.median(np.diff(np.asarray(time, dtype=float)))
    return float('nan') if dt <= 0 else 1.0 / dt


def _slope_through_origin(x, y):
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom > 0 else float('nan')


def _linfit(x, y):
    if len(x) < 2:
        return float('nan'), float('nan')
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def check_saturation(pressure, load_key, side, verdicts):
    '''判断单元格是否在上限堆积。'''
    if pressure is None or pressure.size == 0:
        return

    peak = float(pressure.max())
    if peak <= 0:
        verdicts.add('M6 饱和', load_key, side, False, '压强全为零')
        return

    n_all = pressure.size
    n99 = int((pressure >= 0.99 * peak).sum())
    n95 = int((pressure >= 0.95 * peak).sum())
    frac = n99 / n_all

    detail = ('峰值 {:.2f}，>=99%峰值 {} 个 ({:.4f}%)，>=95% {} 个，'
              '逐帧最大压强均值 {:.2f}').format(
        peak, n99, 100.0 * frac, n95, float(pressure.max(axis=(1, 2)).mean()))

    verdicts.add('M6 饱和', load_key, side, frac <= SAT_FRAC_WARN, detail)


def report_cop(result, load_key, side):
    '''输出 COP 描述性统计，不参与判定。'''
    cop = np.asarray(result['cop_ant'], dtype=float)
    lat = np.asarray(result['cop_lat'], dtype=float)
    force = np.asarray(result['force'], dtype=float)

    ok = np.isfinite(cop) & np.isfinite(force)
    if ok.sum() < 10:
        beauty_print('  [COP]  load={} {} : 有效帧不足，无法统计'.format(
            load_key, side.upper()), type='warning')
        return

    c = cop[ok]
    f = force[ok]
    corr = float(np.corrcoef(f, c)[0, 1]) if len(c) > 2 else float('nan')
    slope, _ = _linfit(f, c)

    print('  [COP]  load={:<5} {} : 前向 均值 {:.1f} cm  sd {:.2f} cm  '
          '范围 {:.1f}-{:.1f} cm ({:.1f} cm)'.format(
              load_key, side.upper(), 100 * c.mean(), 100 * c.std(),
              100 * c.min(), 100 * c.max(), 100 * (c.max() - c.min())))
    print('         横向 sd {:.2f} cm；与力的相关 {:+.3f}，'
          '回归 {:+.2f} cm / 1000 N'.format(
              100 * np.nanstd(lat[ok]), corr, 100 * slope * 1000.0))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def compare_one_side(config, load_key, side, force_rel, map_rel, verdicts,
                     window=None, robot_file=None):
    force_path = resolve_insole_path(config, force_rel)
    map_path = resolve_insole_path(config, map_rel)

    # ---- M1 文件存在性 ----
    if not map_rel:
        verdicts.add('M1 文件', load_key, side, False,
                     'config 中缺少压力图字段，请补上 insole_map_{}'.format(side))
        return
    if force_path is None or not os.path.exists(force_path):
        verdicts.add('M1 文件', load_key, side, False,
                     '找不到汇总力文件: {}'.format(force_path))
        return
    if not os.path.exists(map_path):
        verdicts.add('M1 文件', load_key, side, False,
                     '找不到压力图文件: {}'.format(map_path))
        return

    # 两个文件来自同一套采集软件，共用同一个相对时钟，
    # 因此这里不做 info.csv 对齐，直接比原始时间轴
    # 两个文件在同一个文件夹里，共用同一份 info.csv 与同一个 robot_file，
    # 所以两边拿到的时间修正量完全相同，M2 比的仍然是两者的相对关系。
    # 之前用 use_info_timestamp=False 是因为只需要相对比较；现在要用深蹲窗口
    # 截取，必须先对齐到机器人时钟，否则窗口和数据不在同一个时间基准上。
    t_force, f_force = InsoleProcessor.load(
        force_path, verbose=False, use_info_timestamp=True,
        robot_file=robot_file, folder=config.get('folder', ''))
    if t_force is None:
        verdicts.add('M1 文件', load_key, side, False,
                     '汇总力文件解析失败: {}'.format(
                         os.path.basename(force_path)))
        return

    result = InsoleProcessor.load_pressure_map(
        map_path, verbose=False, use_info_timestamp=True,
        robot_file=robot_file, folder=config.get('folder', ''),
        toe_first=TOE_FIRST, min_force=MIN_FORCE_N, return_matrix=True)
    if result is None:
        verdicts.add('M1 文件', load_key, side, False,
                     '压力图文件解析失败: {}'.format(
                         os.path.basename(map_path)))
        return

    t_map = np.asarray(result['time'], dtype=float)
    f_map = np.asarray(result['force'], dtype=float)
    meta = result['meta']

    verdicts.add('M1 文件', load_key, side, True,
                 '力 {} 帧 / 压力图 {} 帧，网格 {}x{}，'
                 '单元 {:.3f}x{:.3f} cm，感应面 {:.1f}x{:.1f} cm'.format(
                     len(t_force), len(t_map), meta['n_rows'], meta['n_cols'],
                     meta['cell_dx_cm'], meta['cell_dy_cm'],
                     meta['width_cm'], meta['length_cm']))

    # ---- M2 时间轴一致性 ----
    fs_force = _fs_from_time(t_force)
    fs_map = _fs_from_time(t_map)
    dur_force = float(t_force[-1] - t_force[0])
    dur_map = float(t_map[-1] - t_map[0])

    d_start = abs(float(t_map[0] - t_force[0]))
    d_dur = abs(dur_map - dur_force)
    ok_time = (d_start <= START_TOL_S) and (d_dur <= DURATION_TOL_S)

    verdicts.add('M2 时间轴', load_key, side, ok_time,
                 '力 {:.1f} Hz / {:.2f} s，压力图 {:.1f} Hz / {:.2f} s '
                 '(声明 {:.1f} Hz)；起点差 {:.3f} s，时长差 {:.3f} s'.format(
                     fs_force, dur_force, fs_map, dur_map,
                     meta['frequency'] or float('nan'), d_start, d_dur))

    # ---- 重采样到共同时间轴 ----
    t0 = max(t_force[0], t_map[0])
    t1 = min(t_force[-1], t_map[-1])
    if t1 - t0 < 1.0:
        verdicts.add('M3 相关', load_key, side, False,
                     '两文件时间轴重叠不足 1 s，无法比较')
        return

    # 以采样率较低的一路为基准，避免插值造出虚假的一致性
    if fs_map <= fs_force:
        grid = t_map[(t_map >= t0) & (t_map <= t1)]
        y_map = np.interp(grid, t_map, f_map)
        y_force = np.interp(grid, t_force, f_force)
    else:
        grid = t_force[(t_force >= t0) & (t_force <= t1)]
        y_force = np.interp(grid, t_force, f_force)
        y_map = np.interp(grid, t_map, f_map)

    use = (y_force >= MIN_FORCE_N) | (y_map >= MIN_FORCE_N)
    if use.sum() < 20:
        verdicts.add('M3 相关', load_key, side, False,
                     '超过阈值的帧不足 20，无法统计')
        return

    x = y_map[use]
    y = y_force[use]

    # ---- M3 波形相关 ----
    corr = float(np.corrcoef(x, y)[0, 1])
    verdicts.add('M3 相关', load_key, side, corr >= MIN_CORR,
                 '相关系数 {:.4f}（阈值 {:.2f}，共 {} 帧）'.format(
                     corr, MIN_CORR, int(use.sum())))

    # ---- M4 增益一致 ----
    k0 = _slope_through_origin(x, y)
    k1, b1 = _linfit(x, y)
    ok_gain = abs(k0 - 1.0) <= SLOPE_TOL
    verdicts.add('M4 增益', load_key, side, ok_gain,
                 '过原点斜率 {:.4f}（Force/压力图，理想 1.0）；'
                 '带截距拟合 {:.4f}x {:+.1f} N'.format(k0, k1, b1))

    # ---- M5 残差 ----
    diff = y - x
    scale = float(np.sqrt(np.mean(y * y)))
    rel_rms = float(np.sqrt(np.mean(diff * diff)) / scale) if scale > 0 \
        else float('nan')
    verdicts.add('M5 残差', load_key, side, rel_rms <= REL_RMS_TOL,
                 '相对 RMS 残差 {:.2%}（阈值 {:.0%}），'
                 '均值偏差 {:+.1f} N，最大 {:.1f} N；'
                 '均值 Force {:.1f} N / 压力图 {:.1f} N'.format(
                     rel_rms, REL_RMS_TOL, float(diff.mean()),
                     float(np.abs(diff).max()), float(y.mean()),
                     float(x.mean())))

    # ---- M6 饱和 ----
    # 以下统计只用深蹲窗口内的帧。M1-M5 比的是两个文件全长的一致性，
    # 故意不加窗；而 COP / 饱和 / 热图是在描述受试者的动作，必须加窗。
    windowed, win_note = slice_result_to_window(
        result, window, load_key, side)
    print('  [窗口]  load={:<11} {} : {}'.format(
        load_key, side.upper(), win_note))

    check_saturation(windowed.get('pressure'), load_key, side, verdicts)

    # ---- COP 描述性统计 ----
    report_cop(windowed, load_key, side)

    return windowed


def main():
    config_path = get_config_path()
    if not os.path.exists(config_path):
        beauty_print('找不到配置文件: {}'.format(config_path), type='warning')
        return

    with open(config_path, 'r', encoding='utf-8') as fh:
        config = json.load(fh)

    print('=' * 78)
    print('压力图 vs 汇总力文件 一致性校验')
    print('=' * 78)
    print('config : {}'.format(config_path))
    print('base   : {}'.format(get_base_dir()))

    data = config.get('modeling_file', {}).get('data', {})
    load_keys = get_load_keys(config)
    if not load_keys:
        beauty_print('config 中没有可用的负载组', type='warning')
        return

    # 模型里的常数接触点，画在热图上作为对照
    contact = config.get('opensim_settings', {}).get('insole_contact_point')
    contact_point = float(contact[0]) if contact else None

    windows = get_squat_windows(config_path, load_keys)

    verdicts = Verdicts()
    all_results = {}

    for load_key in load_keys:
        seg = data.get(load_key, {})
        print('')
        print('-' * 78)
        print('load = {}'.format(load_key))
        print('-' * 78)

        side_results = {}
        for side, force_key, map_key in SIDES:
            side_results[side] = compare_one_side(
                config, load_key, side,
                seg.get(force_key), seg.get(map_key), verdicts,
                window=windows.get(str(load_key)),
                robot_file=seg.get('robot_file'))

        all_results[load_key] = side_results

    # 热图必须等所有组都读完再画：色标上限要取全体的最大值，
    # 各组颜色才代表同一个压强。若每组自己归一化，看起来深浅差不多，
    # 实际可能差好几倍，跳组比较全是错的。COP 轨迹的力配色同理。
    if PLOT_HEATMAP:
        vmax = global_pressure_vmax(all_results, min_force=MIN_FORCE_N)
        force_range = global_force_range(all_results)
        print('')
        print('统一色标：平均压强 0-{} N/cm2；COP 轨迹配色总力 {}'.format(
            'N/A' if vmax is None else '{:.2f}'.format(vmax),
            'N/A' if force_range is None else
            '{:.0f}-{:.0f} N'.format(force_range[0], force_range[1])))
        for load_key, side_results in all_results.items():
            plot_load_pressure_cop(
                side_results, load_key=load_key,
                contact_point_m=contact_point,
                min_force=MIN_FORCE_N,
                describe=describe_en(config, load_key),
                save_dir=PLOT_SAVE_DIR, show=False,
                vmax=vmax, force_range=force_range)

    if PLOT_ACROSS_LOADS and all_results:
        plot_cop_across_loads(all_results, contact_point_m=contact_point,
                              save_dir=PLOT_SAVE_DIR, show=False)

    verdicts.summary()

    print('')
    print('判读提示：')
    print('  - M4 斜率接近 1.0：两路同源且标定一致，压力图 COP 可直接使用。')
    print('  - M4 斜率系统地大于 1.0：Force.csv 里含有额外标定（或压力图漏掉了')
    print('    部分区域的力）。此时 COP 的位置仍可用，但幅值不可直接当总力用。')
    print('  - M3 相关低：两个文件可能不是同一次采集，先查 config 路径。')
    print('  - M6 不通过：单元格在上限堆积，高负荷下总力会被削顶。')
    print('  - [窗口] 一行显示“退回整段”：该组的 COP 统计里混了非深蹲帧，'
          '不可用于跳组比较。')

    if PLOT_SHOW and (PLOT_HEATMAP or PLOT_ACROSS_LOADS):
        plt.show()


if __name__ == '__main__':
    main()