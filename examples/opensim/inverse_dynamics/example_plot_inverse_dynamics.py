"""
example_plot_inverse_dynamics.py

在运行完 example_inverse_dynamics.py（已生成每个 load 的 inverse_dynamics.sto）后，
把每个负载的运动学 / 动力学时间序列画在两张图上：

【图 1】运动学 + GRF + ID 关节力矩
  每个负载占一行子图，子图内包含所有曲线（全部采用 baseline 减基准法）：
    - 位置 pos_l (m)              减去窗口末尾均值作为基准，保留实际量纲
    - 机器人力 force_l (N)        同上，减基准
    - 加速度 acc_l (m/s²)         同上，减基准
    - 鞋垫 GRF grf_l / grf_r (N)  同上，减基准（grf_r 用虚线）
    - ID 关节力矩 hip / knee / ankle (N·m)  同上，减基准

【图 2】力平衡分析
  同样的负载排布，子图内包含以下曲线（全部减基准）：
    - Fbar    = force_l + force_r (N)
    - Fgrf    = grf_l  + grf_r  (N)
    - 惯性力  = subject_mass * a_y (N)
                a_y 读的是 Xsens Center of Mass 表插值到 OpenSim y 轴的
                质心加速度（与 example_shear_reconstruction.py 里
                _cached_xsens_com 完全同一份数据、同一份缓存），
                不是机器人杆的加速度。
    - 合力    = Fgrf - Fbar - m*g - m*a_y (N)
                与 example_shear_reconstruction.py 的 [V0] 里
                gap = grf_y + bar_y - m*g - m*a_y 代数等价
                （这里 Fbar = force_l+force_r，bar_y = -Fbar），
                因此这条曲线与之前 [V0] 分析的差额完全相同。
    - pos_l (m，机器人)、acc_l (m/s²，机器人) 放在最后画，用浅色，
      并放大 100 倍以便看出变化；两者只是参考，不参与力平衡计算。

  每个负载的合力均值（切片段平均）在运行结束后打印。

数据来源：
  使用标准切片数据 cutted_data（缓存保证含 grf_l / grf_r），
  ID 力矩通过 interpolate_column_to_segment 插值到同一批时间点。
  按 segment_id 分段绘制，避免相邻运动周期之间被连成一条直线。

用法：
    python example_plot_inverse_dynamics.py
"""
import os
import sys
import json
import math

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

_BASE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from digitaltwin.osim.inverse_dynamics import run_step3_inverse_dynamics
from digitaltwin.pipelines.standard_analysis import (
    run_standard_data_pipeline,
    load_or_create_cutted_pipeline_results,
)
from digitaltwin.analysis.result_analysis import (
    build_left_joint_coordinate_map,
    get_load_keys,
    get_inverse_dynamics_path,
    get_segment_from_results,
    read_opensim_table,
    interpolate_column_to_segment,
    find_id_moment_column,
)
from digitaltwin.utils.data_io import canonical_load_key as _canonical_load_key
from digitaltwin.utils.logger import beauty_print

# 复用 example_shear_reconstruction.py 里的 Xsens 质心加速度读取，
# 保证图 2 的“惯性力”与该脚本 [V0] 里的 m*a_y 完全同源，不是另算一套。
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)
from example_shear_reconstruction import (
    resolve_xsens_path,
    _cached_xsens_com,
)


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../../config/20260513_squat_FTS09_xsens.json'

LOAD_KEYS = None
EXCLUDE_LOAD_KEYS = []

RUN_INVERSE_DYNAMICS = False

# grf 的对齐完全依赖 config 里的 insole_time_offset（由 example_insole_sync_offset.py
# 标定）。result_folder 里的 cutted_data.csv 缓存可能是旧标定生成的，其中的 grf_l / grf_r
# 会与机器人 force 不同步。因此默认每次重跑完整流水线（与 example_data_analysis_insoles.py
# 一致），保证 grf 用当前 config 的 offset 重新注入。
# 只有当你确认缓存是用当前 config 生成的，才改成 False 走缓存以加快速度。
RUN_PIPELINE_FRESH = True

FORCE_REBUILD_CUTTED_CACHE = False
CUTTED_CACHE_NAME = 'cutted_data.csv'

USE_EXTERNAL_FORCES = True
MB = 0.0
OUTPUT_BODY_FORCES = False

NCOLS = 1
FIG_WIDTH = 12.0
FIG_HEIGHT_PER_ROW = 3.4
LINEWIDTH = 1.4

SAVE_FIGURE_1 = 'plot_kinematics_grf_moments_by_load.png'
SAVE_FIGURE_2 = 'plot_force_balance_by_load.png'
SHOW_FIGURE = True


# ============================================================
#  路径
# ============================================================

def get_base_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


# ============================================================
#  图 1 曲线定义
# ============================================================

SERIES_SPEC = [
    ('force_l',              'force_l',        '-'),
    ('pos_l',                'pos_l',          '-'),
    ('acc_l',                'acc_l',          '-'),
    ('grf_l',                'grf_l',          '-'),
    ('grf_r',                'grf_r',          '--'),
    ('hip_flexion_l_moment', 'hip moment',     '-'),
    ('knee_angle_l_moment',  'knee moment',    '-'),
    ('ankle_angle_l_moment', 'ankle moment',   '-'),
]
MOMENT_SPEC = {
    'hip_flexion_l_moment':  'hip_flexion_l',
    'knee_angle_l_moment':   'knee_angle_l',
    'ankle_angle_l_moment':  'ankle_angle_l',
}

_DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
SERIES_COLORS = {
    spec[0]: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
    for i, spec in enumerate(SERIES_SPEC)
}

# 图 2 使用不同配色序列
# pos_l / acc_l 是机器人（杆）的运动学，只用作参考，放在最后画、用浅色，
# 并放大 100 倍才能看出明显变化；其余四条才是真正参与力平衡的量。
_FIG2_KEYS = ['Fbar', 'Fgrf', 'inertia', 'net', 'grf_l', 'grf_r', 'pos_l', 'acc_l']
_FIG2_MAIN_COLORS = {
    'Fbar': _DEFAULT_COLORS[0], 'Fgrf': _DEFAULT_COLORS[1],
    'inertia': _DEFAULT_COLORS[2], 'net': _DEFAULT_COLORS[3],
    'grf_l': _DEFAULT_COLORS[4], 'grf_r': _DEFAULT_COLORS[5],
}
_FIG2_LIGHT_COLORS = {'pos_l': '#d0d0d0', 'acc_l': '#bcdcf0'}
_FIG2_COLORS = {**_FIG2_MAIN_COLORS, **_FIG2_LIGHT_COLORS}
_FIG2_SCALE = {'pos_l': 500.0, 'acc_l': 100.0}
_FIG2_LABELS = {
    'Fbar':    'Fbar = force_l+force_r (N)',
    'Fgrf':    'Fgrf-mg = grf_l+grf_r-mg (N)',
    'inertia': 'inertia = m·a_y(Xsens CoM) (N)',
    'net':     'net = Fgrf−Fbar−mg−m·a_y (N)',
    'grf_l':   'grf_l (N)',
    'grf_r':   'grf_r (N)',
    'pos_l':   'pos_l ×500 (m, robot, ref only)',
    'acc_l':   'acc_l ×100 (m/s², robot, ref only)',
}
_FIG2_LS = {'net': '--', 'grf_l': '--', 'grf_r': '--'}


# ============================================================
#  缩放：全部曲线减基准（减窗口末尾均值），保留实际量纲
# ============================================================

def scale_baseline(values, n_tail=10):
    """减去末尾 n_tail 帧的均值，保留实际量纲和正负号。"""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if int(ok.sum()) == 0:
        return v
    tail = v[ok][-n_tail:]
    baseline = float(np.nanmean(tail)) if len(tail) > 0 else 0.0
    return v - baseline


def scale_baseline_norm(values, n_tail=10):
    """减基准后再除以最大绝对值，统一落在 [-1, 1]（保留正负号，丢失实际量纲）。
    仅用于图 1：图 1 只关心形态/相位对比，图 2 仍然保留真实量纲以便做力平衡计算。"""
    v = scale_baseline(values, n_tail=n_tail)
    ok = np.isfinite(v)
    if int(ok.sum()) == 0:
        return v
    denom = float(np.max(np.abs(v[ok]))) + 1e-10
    return v / denom


# ============================================================
#  分段绘制工具
# ============================================================

def _plot_segs(ax, t, vals, seg_ids, color, linestyle, label):
    first = True
    for seg in np.unique(seg_ids):
        mask = seg_ids == seg
        tt, vv = t[mask], vals[mask]
        valid = np.isfinite(tt) & np.isfinite(vv)
        if int(valid.sum()) < 2:
            continue
        ax.plot(tt[valid], vv[valid], color=color, linestyle=linestyle,
                linewidth=LINEWIDTH, label=label if first else None)
        first = False


# ============================================================
#  收集图 1 所需曲线
# ============================================================

def collect_load_series(segment_df, id_df):
    t = segment_df['time'].values.astype(float)
    seg_ids = (segment_df['segment_id'].values.astype(int)
               if 'segment_id' in segment_df.columns
               else np.zeros(len(segment_df), dtype=int))

    series = {}
    for key, _label, _ls in SERIES_SPEC:
        if key in segment_df.columns:
            series[key] = (t, segment_df[key].values.astype(float))
            continue
        coord = MOMENT_SPEC.get(key)
        if coord is None:
            continue
        id_col = find_id_moment_column(id_df, coord)
        if id_col is None:
            continue
        vals = interpolate_column_to_segment(id_df, segment_df, id_col)
        if vals is not None:
            series[key] = (t, np.asarray(vals, dtype=float))

    return seg_ids, series


# ============================================================
#  诊断：Xsens 质心竖直加速度 a_y 与机器人杆加速度 acc_l 的关系
# ============================================================

def diagnose_ay_vs_acc(segment_df, xacc, load_key):
    """核查 Xsens 质心竖直加速度 a_y 与机器人杆加速度 acc_l 的关系。

    两者是不同的物理量：acc_l 是杆（机器人夹具）的加速度，a_y 是全身质心的
    竖直加速度。人握着杆一起运动，两者相关是正常的，但并不应该完全重合：
    人体质心的运动由椫关节骰关节跳躝共同决定，并不是杆位置的线性函数。
    需要核实的三个可能问题：
      1) Xsens 质心加速度没有正确插值/对齐，退化成了机器人信号；
      2) 时间轴没有对齐，恰好在某个滞后下强相关；
      3) 等速（IK）阶段杆速度恒定、acc_l≈0，但人体质心仍可以有真实的
         竖直加速度（椫伸/屈膝节奏）；若 a_y 在这些窗口里也場到 0，
         反而值得怀疑它只是被机器人信号污染了。
    """
    t = segment_df['time'].values.astype(float)
    acc_l = (segment_df['acc_l'].values.astype(float)
             if 'acc_l' in segment_df.columns else np.full(len(t), np.nan))
    vel_l = (segment_df['vel_l'].values.astype(float)
             if 'vel_l' in segment_df.columns else None)

    if xacc is None or 'time' not in xacc or 'acc' not in xacc:
        beauty_print('  [诊断] load={}: 没有 Xsens 质心加速度，无法比对。'
                     .format(load_key), type='warning')
        return

    ay = np.interp(t, xacc['time'], xacc['acc'][:, 1])

    ok = np.isfinite(ay) & np.isfinite(acc_l)
    if int(ok.sum()) < 10:
        print('  [诊断] load={}: 有效帧数不足，跳过 a_y vs acc_l 核查。'
              .format(load_key))
        return

    if np.std(ay[ok]) > 1e-9 and np.std(acc_l[ok]) > 1e-9:
        r = float(np.corrcoef(ay[ok], acc_l[ok])[0, 1])
    else:
        r = float('nan')
    print('  [诊断] load={}: corr(a_y, acc_l) = {}'.format(
        load_key, 'N/A' if not np.isfinite(r) else '{:+.3f}'.format(r)))
    if np.isfinite(r) and abs(r) > 0.9:
        beauty_print('  [诊断] load={}: corr(a_y, acc_l) = {:+.3f}，接近 1。'
                     '这两个本该独立的信号过度重合，请核实 a_y 是否真的'
                     '来自 Xsens 质心表，不是插值时时间轴对齐错误地'
                     '退化回机器人信号。'.format(load_key, r),
                     type='warning')

    if vel_l is None:
        beauty_print('  [诊断] load={}: 没有 vel_l 列，无法检查匀速窗口。'
                     .format(load_key), type='warning')
        return

    win = 25
    sd_vel = pd.Series(vel_l).rolling(
        win, center=True, min_periods=max(3, win // 2)).std().values
    finite_vel = vel_l[np.isfinite(vel_l)]
    quiet_speed = (0.05 * float(np.nanmedian(np.abs(finite_vel)))
                  if len(finite_vel) else 0.01)
    quiet_speed = max(quiet_speed, 0.005)
    const_mask = ((np.nan_to_num(sd_vel, nan=1e9) < quiet_speed)
                  & (np.abs(vel_l) > 0.02))

    if int(np.sum(const_mask)) < 10:
        print('  [诊断] load={}: 没找到足够长的匀速窗口，跳过匀速期核查。'
              .format(load_key))
        return

    acc_l_q = acc_l[const_mask]
    ay_q = ay[const_mask]
    acc_l_q = acc_l_q[np.isfinite(acc_l_q)]
    ay_q = ay_q[np.isfinite(ay_q)]
    acc_l_mean = float(np.mean(acc_l_q)) if len(acc_l_q) else float('nan')
    ay_mean = float(np.mean(ay_q)) if len(ay_q) else float('nan')
    ay_sd = float(np.std(ay_q)) if len(ay_q) else float('nan')
    print('  [诊断] load={}: 匀速窗口 {} 帧，acc_l 均值 {:+.3f} m/s²'
          '（应≈ 0），a_y 均值 {:+.3f} m/s²，a_y 标准差 {:.3f} m/s²'
          .format(load_key, int(np.sum(const_mask)), acc_l_mean, ay_mean, ay_sd))

    if len(ay_q) and np.isfinite(ay_mean) and np.isfinite(ay_sd) \
            and abs(ay_mean) > 3.0 * (ay_sd + 1e-6) and abs(ay_mean) > 0.3:
        beauty_print('  [诊断] load={}: 匀速窗口内 a_y 均值 {:+.3f} m/s²明显'
                     '偏离 0（超过 3 倍标准差，不是随机噪声量级），而杆'
                     '加速度已接近 0。这可能是质心竖直加速度在此期间'
                     '确实不为 0（身体仍在踌伸/调整），但也可能是'
                     'Xsens 数据没有对齐或存在恒定偏置，建议进一步核查。'
                     .format(load_key, ay_mean), type='warning')


# ============================================================
#  收集图 2 所需曲线（力平衡）
# ============================================================

def collect_force_balance_series(segment_df, subject_mass, xacc, load_key):
    """返回 (seg_ids, dict{key: (t, values)}, net 数组)。

    惯性力 m·a_y 读的是 Xsens Center of Mass 表插值到 OpenSim y 轴的加速度
    （与 example_shear_reconstruction.py 的 [V0] 里 s['ay'] 完全同一份数据、
    同一次缓存），不是机器人杆的加速度。net 的公式
        net = Fgrf − Fbar − m*g − m*a_y
    与该脚本 V0 的 gap = grf_y + bar_y - m*g - m*a_y 代数等价
    （这里 Fbar = force_l+force_r，bar_y = -Fbar，两者符号相反、
    数值相同）。
    """
    t = segment_df['time'].values.astype(float)
    seg_ids = (segment_df['segment_id'].values.astype(int)
               if 'segment_id' in segment_df.columns
               else np.zeros(len(segment_df), dtype=int))
    n = len(t)

    def col(name):
        if name in segment_df.columns:
            return segment_df[name].values.astype(float)
        return np.full(n, np.nan)

    pos_l  = col('pos_l')
    acc_l  = col('acc_l')
    fl     = col('force_l')
    fr     = col('force_r')
    grf_l  = col('grf_l')
    grf_r  = col('grf_r')

    fbar = fl + fr                     # N，杆向下作用于人
    fgrf = grf_l + grf_r              # N，鞋垫向上作用于人
    mg   = subject_mass * 9.81         # N，常数

    if xacc is not None and 'time' in xacc and 'acc' in xacc:
        ay = np.interp(t, xacc['time'], xacc['acc'][:, 1])
    else:
        beauty_print('  [WARN] load={}: 没有可用的 Xsens 质心加速度，'
                     '惯性力退化为机器人 acc_l（与 V0 不再同源）。'
                     .format(load_key), type='warning')
        ay = acc_l

    inertia = subject_mass * ay         # N，m·a_y（Xsens 质心）
    net = fgrf - fbar - mg - inertia   # 应趋于 0

    # 对照组：将 a_y 换成 0.5*acc_l（机器人杆加速度的一半），
    # 重新计算 inertia_alt 和 net_alt，用于对比与 Xsens 质心加速度
    # 的差异。
    ay_alt = 0.5 * acc_l
    inertia_alt = subject_mass * ay_alt
    net_alt = fgrf - fbar - mg - inertia_alt

    # 图 2 中的 'Fgrf' 曲线实际画的是 Fgrf-mg（已减去重力），
    # 便于与 Fbar / inertia 直接比较。net 的计算仍用未减 mg 前的
    # 原始 fgrf，保证代数与 V0 完全一致。
    series = {
        'pos_l':   (t, pos_l),
        'acc_l':   (t, acc_l),
        'Fbar':    (t, fbar),
        'Fgrf':    (t, fgrf - mg),
        'inertia': (t, inertia),
        'net':     (t, net),
        'grf_l':   (t, grf_l),
        'grf_r':   (t, grf_r),
    }
    return seg_ids, series, net, inertia, net_alt, inertia_alt


# ============================================================
#  绘制图 1 子图
# ============================================================

def plot_load_subplot_fig1(ax, load_key, seg_ids, series):
    baselined = {}
    for key, label, ls in SERIES_SPEC:
        data = series.get(key)
        if data is None:
            continue
        t, v = data
        baselined[key] = (t, scale_baseline(v))

    # grf_l / grf_r 必须共用同一个归一化分母，否则两条曲线各自被拉伸到
    # [-1,1]，看上去幅度一样，实际数值比例被救坏，无法比较左右差异。
    grf_keys = [k for k in ('grf_l', 'grf_r') if k in baselined]
    shared_denom = None
    if grf_keys:
        shared_denom = 0.0
        for k in grf_keys:
            _, v = baselined[k]
            ok = np.isfinite(v)
            if int(ok.sum()):
                shared_denom = max(shared_denom, float(np.max(np.abs(v[ok]))))
        shared_denom += 1e-10

    for key, label, ls in SERIES_SPEC:
        if key not in baselined:
            continue
        t, v = baselined[key]
        if key in ('grf_l', 'grf_r') and shared_denom:
            scaled = v / shared_denom
        else:
            ok = np.isfinite(v)
            denom = (float(np.max(np.abs(v[ok]))) + 1e-10
                     if int(ok.sum()) else 1.0)
            scaled = v / denom
        _plot_segs(ax, t, scaled, seg_ids, SERIES_COLORS[key], ls, label)

    ax.set_title(str(load_key), fontsize=13)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('normalized signal [-1, 1]')
    ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.4)
    ax.grid(True, alpha=0.3)


# ============================================================
#  绘制图 2 子图
# ============================================================

def plot_load_subplot_fig2(ax, load_key, seg_ids, series2):
    # 图 2 不减基准，都用原始数值；只有 pos_l/acc_l 额外乘 100（见 _FIG2_SCALE）。
    for key in _FIG2_KEYS:
        data = series2.get(key)
        if data is None:
            continue
        t, v = data
        ls = _FIG2_LS.get(key, '-')
        color = _FIG2_COLORS[key]
        label = _FIG2_LABELS[key]
        scaled = np.asarray(v, dtype=float) * _FIG2_SCALE.get(key, 1.0)
        _plot_segs(ax, t, scaled, seg_ids, color, ls, label)

    ax.set_title(str(load_key), fontsize=13)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('force / signal (raw, pos_l & acc_l ×100)')
    ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.4)
    ax.grid(True, alpha=0.3)


# ============================================================
#  主程序
# ============================================================

def main():
    base_dir   = get_base_dir()
    config_path = get_config_path()

    print('配置文件: {}'.format(config_path))
    print('基准目录: {}'.format(base_dir))

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    osim_cfg     = config.get('opensim_settings', {})
    subject_mass = float(osim_cfg.get('subject_mass', config.get('subject_mass', 70.0)))
    print('受试者质量: {:.2f} kg（来自 opensim_settings.subject_mass）'.format(subject_mass))

    load_keys = get_load_keys(config, LOAD_KEYS)
    if EXCLUDE_LOAD_KEYS:
        excluded  = {_canonical_load_key(k) for k in EXCLUDE_LOAD_KEYS}
        load_keys = [k for k in load_keys if _canonical_load_key(k) not in excluded]
    if not load_keys:
        raise ValueError('没有可绘制的负载 key，请检查 LOAD_KEYS / EXCLUDE_LOAD_KEYS。')

    coord_map = build_left_joint_coordinate_map(config)
    print('将绘制以下负载:', ', '.join(load_keys))

    # 1) 确保 ID 输出存在
    missing = [k for k in load_keys
               if not os.path.exists(get_inverse_dynamics_path(config, base_dir, k))]
    if RUN_INVERSE_DYNAMICS or missing:
        if missing:
            print('以下负载缺少 inverse_dynamics.sto，自动运行 Step3: {}'.format(missing))
        run_step3_inverse_dynamics(
            config=config, base_dir=base_dir,
            use_external_forces=USE_EXTERNAL_FORCES,
            Mb=MB, output_body_forces=OUTPUT_BODY_FORCES, verbose=True)

    # 2) 切片数据
    if RUN_PIPELINE_FRESH:
        # 每次重跑流水线：grf 按当前 insole_time_offset 重新注入，与 force 同步
        subject, _pipeline, results = run_standard_data_pipeline(
            config_path, include_xsens=False, include_insole=True,
            debug=True)
    else:
        subject, _pipeline, results = load_or_create_cutted_pipeline_results(
            config_path, include_xsens=False, include_insole=True,
            debug=True, force_rebuild=FORCE_REBUILD_CUTTED_CACHE,
            cache_name=CUTTED_CACHE_NAME)

    # 3) 逐 load 收集
    per_load        = {}   # 图 1
    per_load_f2     = {}   # 图 2
    net_means       = {}   # 合力均值打印
    inertia_means   = {}   # 惯性力均值打印
    net_means_alt   = {}   # ay=0.5*acc_l 时的合力均值打印
    inertia_means_alt = {} # ay=0.5*acc_l 时的惯性力均值打印

    for load_key in load_keys:
        segment_df = get_segment_from_results(
            results, load_key, movement_types=None)
        if segment_df is None or len(segment_df) == 0:
            print('[WARN] load={}: 没有可用切片数据，跳过。'.format(load_key))
            continue

        id_path = get_inverse_dynamics_path(config, base_dir, load_key)
        id_df   = read_opensim_table(id_path)
        if id_df is None or 'time' not in id_df.columns:
            print('[WARN] load={}: inverse_dynamics 文件不可读，跳过。'.format(load_key))
            continue

        seg_ids, series = collect_load_series(segment_df, id_df)
        if series:
            per_load[load_key] = (seg_ids, series)

        file_info  = config['modeling_file']['data'].get(str(load_key), {})
        xsens_path = resolve_xsens_path(config, file_info)
        xacc       = _cached_xsens_com(xsens_path, base_dir, config, load_key)

        diagnose_ay_vs_acc(segment_df, xacc, load_key)

        seg_ids2, series2, net_arr, inertia_arr, net_arr_alt, inertia_arr_alt = (
            collect_force_balance_series(segment_df, subject_mass, xacc, load_key))
        per_load_f2[load_key] = (seg_ids2, series2)

        # 合力均值（有限值）
        net_valid = net_arr[np.isfinite(net_arr)]
        net_means[load_key] = float(np.mean(net_valid)) if len(net_valid) else float('nan')

        # 惯性力均值（有限值）
        inertia_valid = inertia_arr[np.isfinite(inertia_arr)]
        inertia_means[load_key] = (float(np.mean(inertia_valid))
                                   if len(inertia_valid) else float('nan'))

        # ay=0.5*acc_l 时的合力均值与惯性力均值（有限值）
        net_valid_alt = net_arr_alt[np.isfinite(net_arr_alt)]
        net_means_alt[load_key] = (float(np.mean(net_valid_alt))
                                   if len(net_valid_alt) else float('nan'))
        inertia_valid_alt = inertia_arr_alt[np.isfinite(inertia_arr_alt)]
        inertia_means_alt[load_key] = (float(np.mean(inertia_valid_alt))
                                       if len(inertia_valid_alt) else float('nan'))

    if not per_load and not per_load_f2:
        raise RuntimeError('没有任何 load 成功收集到数据，无法绘图。')

    # 4) 图 1
    plot_loads = [k for k in load_keys if k in per_load]
    if plot_loads:
        n1    = len(plot_loads)
        nrows = int(math.ceil(n1 / NCOLS))
        fig1, axes1 = plt.subplots(
            nrows, NCOLS,
            figsize=(FIG_WIDTH, max(3.0, FIG_HEIGHT_PER_ROW * nrows)),
            squeeze=False)
        axes1 = axes1.ravel()

        for ax, lk in zip(axes1, plot_loads):
            plot_load_subplot_fig1(ax, lk, *per_load[lk])
        for ax in axes1[n1:]:
            ax.set_visible(False)

        handles, labels = axes1[0].get_legend_handles_labels()
        if handles:
            fig1.legend(handles, labels, loc='lower center',
                        ncol=max(1, len(handles) // 2), fontsize=10, frameon=False)
        fig1.suptitle(
            '{}: kinematics + GRF + ID moments per load (baseline-subtracted)'.format(
                config['experiment_label']),
            fontsize=14)
        fig1.tight_layout(rect=(0, 0.05, 1, 0.97))

        if SAVE_FIGURE_1:
            out_dir = os.path.join(base_dir, 'result', config['experiment_label'],
                                   'opensim', 'inverse_dynamics')
            os.makedirs(out_dir, exist_ok=True)
            p1 = os.path.join(out_dir, SAVE_FIGURE_1)
            fig1.savefig(p1, dpi=150, bbox_inches='tight')
            print('图 1 已保存: {}'.format(p1))

    # 5) 图 2
    plot_loads2 = [k for k in load_keys if k in per_load_f2]
    if plot_loads2:
        n2    = len(plot_loads2)
        nrows = int(math.ceil(n2 / NCOLS))
        fig2, axes2 = plt.subplots(
            nrows, NCOLS,
            figsize=(FIG_WIDTH, max(3.0, FIG_HEIGHT_PER_ROW * nrows)),
            squeeze=False)
        axes2 = axes2.ravel()

        for ax, lk in zip(axes2, plot_loads2):
            plot_load_subplot_fig2(ax, lk, *per_load_f2[lk])
        for ax in axes2[n2:]:
            ax.set_visible(False)

        handles2, labels2 = axes2[0].get_legend_handles_labels()
        if handles2:
            fig2.legend(handles2, labels2, loc='lower center',
                        ncol=max(1, len(handles2) // 2), fontsize=10, frameon=False)
        fig2.suptitle(
            '{}: force balance per load  [net = Fgrf − Fbar − mg − m·acc_l]'.format(
                config['experiment_label']),
            fontsize=14)
        fig2.tight_layout(rect=(0, 0.05, 1, 0.97))

        if SAVE_FIGURE_2:
            out_dir = os.path.join(base_dir, 'result', config['experiment_label'],
                                   'opensim', 'inverse_dynamics')
            os.makedirs(out_dir, exist_ok=True)
            p2 = os.path.join(out_dir, SAVE_FIGURE_2)
            fig2.savefig(p2, dpi=150, bbox_inches='tight')
            print('图 2 已保存: {}'.format(p2))

    # 6) 打印合力均值
    print('\n' + '=' * 60)
    print('各负载的合力均值（切片段内）')
    print('  net = Fgrf − Fbar − m*g − m*a_y')
    print('  m = {:.2f} kg，g = 9.81 m/s²，mg = {:.1f} N'.format(
        subject_mass, subject_mass * 9.81))
    print('  a_y 取自 Xsens Center of Mass 表（与 example_shear_reconstruction.py')
    print('  的 [V0] 同源），与该脚本九组的竖直力差额应逐位一致。')
    print('=' * 60)
    print('{:<12}{:>18}{:>18}'.format('load', 'net mean (N)', 'inertia mean (N)'))
    for lk in load_keys:
        v = net_means.get(lk, float('nan'))
        iv = inertia_means.get(lk, float('nan'))
        print('{:<12}{:>18.2f}{:>18.2f}'.format(str(lk), v, iv))
    print('=' * 60)

    # 7) 对照组：将 a_y 换为 0.5*acc_l 后重新计算的合力均值与惯性力均值
    print('\n' + '=' * 60)
    print('对照组：将 a_y 换成 0.5*acc_l 后的合力均值与惯性力均值')
    print('  ay_alt = 0.5 * acc_l（acc_l 为机器人杆加速度，不再用 Xsens 质心）')
    print('  inertia_alt = m * ay_alt，net_alt = Fgrf − Fbar − m*g − inertia_alt')
    print('=' * 60)
    print('{:<12}{:>22}{:>22}'.format(
        'load', 'net_alt mean (N)', 'inertia_alt mean (N)'))
    for lk in load_keys:
        v = net_means_alt.get(lk, float('nan'))
        iv = inertia_means_alt.get(lk, float('nan'))
        print('{:<12}{:>22.2f}{:>22.2f}'.format(str(lk), v, iv))
    print('=' * 60)

    if SHOW_FIGURE:
        plt.show()


if __name__ == '__main__':
    main()