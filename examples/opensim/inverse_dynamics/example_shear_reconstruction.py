'''
example_shear_reconstruction.py

重建足底地面反力的水平（剪切）分量，并验证它能否修复膝关节力矩的单调性。

背景
----
鞋垫只测法向压强，物理上给不出剪切力，所以 external_forces.py 写出的
.sto 里 grf_l_vx / grf_l_vz / grf_r_vx / grf_r_vz 恒为 0。后果是：地面反力
被强行变成纯竖直的，只能通过“COP 到关节中心的水平距离”产生力矩。
对膝关节而言这个距离极短（实测等效力臂 dM/dF 约 1.2 mm），所以
合力从 1130 N 涨到 1723 N（+52%）时，膝力矩只涨了 1.8%。
而剪切分量的力臂是“足底到膝”的竖直距离，约 0.45 m，大了两个量级。

三条重建路线（全部算出来互相对账，而不是选一条相信）
------------------------------------------------
M1 com_accel    质心加速度法。用 OpenSim 缩放模型 + .mot 逐帧算全身质心，
                低通后二次微分，再用牛顿第二定律：
                    F_shear = M * a_COM - F_bar_horizontal
                优点是质量分布用的就是 ID 用的那个模型，内部一致；
                缺点是两次微分对噪声敏感，结果依赖滤波截止频率。

M2 segment_accel  分段加速度法。直接读 Xsens 的 Segment Acceleration 表，
                按人体测量学质量分数加权求和得到 a_COM。
                关键：这一路不做任何微分，IMU 直接输出加速度，
                噪声特性与 M1 完全独立。M1 与 M2 吻合 = 最有力的内部交叉验证。
                另：重力是竖直的，水平分量不受“Xsens 加速度含不含重力”
                这个歧义影响，所以这里不需要猜它的约定。

M3 quasi_static 准静态几何法。深蹲速度慢，GRF 矢量近似沿“COP -> 质心”连线：
                    F_x = F_y * (x_COM - x_COP) / (y_COM - y_COP)
                完全不含微分，最稳健，用来卡 M1/M2 的量级是否离谱。

验证（按证据强度排序）
------------------
V1 骨盆残差（最强，且不需要任何新假设）
   模型骨盆是自由的，ID 会把“外力与运动学不闭合”的那部分全部丢到
   pelvis_tx/ty/tz 的残差力上。现在剪切恒为 0，那么 pelvis_tx 残差
   在数值上就等于缺失的剪切。补上之后残差应当大幅下降。
   这是 OpenSim 社区的标准判据（residual < 5% 体重）。
   ※ 注意：这项会先单独跑一遍。如果残差本来就很小，剪切假说当场被
     证伪，后面的重建就不必做了。

V2 三法一致性   M1/M2/M3 两两相关 + RMS 差异。

V3 等长组零对照 IM-1 杆不动、质心不动，剪切理论上应该 ~0。
                若重建出可观剪切，那就是微分噪声而不是物理量。免费的空白对照。

V4 量级合理性   剪切/竖直 比值应在 5-10% 量级（深蹲文献值）。

V5 单调性       补上剪切重跑 ID，看膝力矩对负载的 Spearman 是否提升、
                绝对量级是否进入 1-2 N·m/kg 体重的合理区间。

安全性
------
不修改任何已有文件。补了剪切的 .sto / .xml 写到
  result/{label}/opensim/external_forces_shear/{load_key}/
重跑的 ID 写到
  result/{label}/opensim/inverse_dynamics_shear/{load_key}/
原有结果原封不动，可以随时回到对照。确认有效后再把逻辑合入
 external_forces.py 的主流程。

路径约定跟 example_inverse_dynamics.py 一致（CONFIG_FILE 用 '../../config/...'，
base_dir 向上三级）。若放在别的目录，改这两个常量即可。
'''
import os
import json

import numpy as np
import pandas as pd
import opensim as osim
from scipy.signal import butter, filtfilt

from digitaltwin.osim.mot_pipeline import get_mot_files, get_scaled_model
from digitaltwin.osim.external_forces import get_ext_forces_dir
from digitaltwin.osim.inverse_dynamics import run_inverse_dynamics
from digitaltwin.pipelines.standard_analysis import (
    load_or_create_cutted_pipeline_results,
)
from digitaltwin.analysis.result_analysis import (
    get_load_keys,
    get_segment_from_results,
    read_opensim_table,
    get_inverse_dynamics_path,
    find_id_moment_column,
)
from digitaltwin.utils.data_io import canonical_load_key
from digitaltwin.utils.logger import beauty_print


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../../config/20260513_squat_FTS09_xsens.json'

# None = 全部九组（含 IM / IK）
LOAD_KEYS = None
EXCLUDE_LOAD_KEYS = []

# 取数窗口：等长组没有 upward/downward，
# get_segment_from_results 会自动回退到等长段，因此九组都能进来。
MOVEMENT_TYPES = ('upward', 'downward')
CACHE_NAME = 'cutted_data.csv'
FORCE_REBUILD_CUTTED_CACHE = False

# 低通滤波（仅 M1 需要）。零相位 filtfilt，避免相位延迟污染相位一致性。
# 6 Hz 是步态/深蹲文献的惯用值；若结果对它敏感，说明 M1 不可靠，应以 M2 为准。
FILTER_CUTOFF_HZ = 6.0
FILTER_ORDER = 4

# ---- mot 低通滤波 ----
# IK 逐帧独立求解，帧间没有任何平滑约束，所以 .mot 里带着高频抖动。
# ID 要对它做二次微分，噪声按 (2*pi*f)^2 放大：6 Hz 的抖动被放大 1400 倍，
# 30 Hz 的被放大 35000 倍。这正是 V2（M1 与 M2 只有 0.43-0.54 相关）与
# validate_mot 的“非物理跳变”最可能的共同病因 —— M1 要对质心做二次微分，
# M2 直接读 IMU 加速度，只有前者会被 mot 噪声污染。
# 滤波后的 mot 另存到 result/{label}/opensim/mot_filtered/，原文件不动。
FILTER_MOT = True
MOT_FILTER_CUTOFF_HZ = 6.0
MOT_FILTER_ORDER = 4
# 已有滤波结果且比源文件新时直接复用。必须这样做：每跑一次都重写会刷新
# mtime，导致下游的 cache_com_cop 永远命中不了（那一步是全程最慢的）。
REUSE_FILTERED_MOT = True

# 写入 .sto 时采用哪一路重建结果
#   'segment_accel' | 'com_accel' | 'quasi_static'
# V1b 实测：M1(com_accel) 对 FX 残差的相关 +0.55~+0.66，九组全部稳定高于
# M2(segment_accel) 的 +0.34~+0.42（106 组甚至 -0.26），M3 几乎为 0。
# 所以改用 M1。修正 Xsens 手性与朝向之后实测 k = +1.09~+1.26，幅值本身
# 就是对的（只偏大 10-25%），V1 九组 FX 残差一致下降 16-25%。
# r 约 0.63 时理论最大降幅 1-sqrt(1-r^2) = 22%，实测已经贴着这个上限，
# 说明 M1 能解释的那部分剪切已经补完，剩下的残差来自别的地方。
SHEAR_METHOD = 'com_accel'

# 体重：None = 用缩放模型的总质量。
# S6 标定给出的实测体重约 72 kg，config 里写的是 70.0，两者差 3%。
BODY_MASS_OVERRIDE = None

# Xsens 全局系是 (前, 左, 上)，OpenSim 是 (前, 上, 右)，两者都是右手系。
# 只交换 y/z 而不取负会得到左手系，相当于把人镜像了一次，左右会互换。
# 除此之外，Xsens 的“前”是标定时的朝向，未必是受试者当下的朝向，
# 所以还要绕竖直轴转一个朝向角 psi。
# 置 False 则只做手性修正、不做朝向旋转，可用来看这一项影响有多大。
APPLY_XSENS_HEADING = True

WRITE_PATCHED_STO = True
RUN_ID_WITH_SHEAR = True
SHEAR_TAG = 'shear'

# 判据阈值
RESIDUAL_WARN_FRAC = 0.05        # 残差力超过体重的这个比例就报警
CROSS_METHOD_MIN_CORR = 0.70     # 三法两两相关下限
ISOMETRIC_SHEAR_WARN_N = 30.0    # 等长组剪切应近于 0
SHEAR_FRAC_RANGE = (0.02, 0.20)  # 剪切/竖直 合理区间
COP_OFFSET_WARN_M = 0.03         # 准静态下质心与压心的前后错位上限

# COP 敏感性实测值：膝力矩对 COP 前后位置的偏导，单位 N·m/m。
# 用来把“COP 配准偏差”直接换算成“膝力矩误差”，判断它够不够解释单调性问题。
KNEE_COP_SENSITIVITY = 491.6

# ---- 静止站立标定（鞋垫竖直增益）----
# 静止站立时全身加速度为 0，竖直方向严格成立：GRF_y = m*g + |bar_y|。
# 这是唯一一个不含微分、不含 COP、不含任何模型假设的绝对基准，
# 所以它是标定鞋垫增益最干净的入口。
# 判定“静止”用四道闸门同时把关：杆力平稳、GRF 平稳、各关节角不动、质心不动。
QUIET_CAL_KEYS = ('IM-1',)     # 空元组 = 对所有组都试一遍
QUIET_WIN_S = 0.5              # 判定“不动”用的滑窗长度
QUIET_MIN_DUR_S = 0.8          # 合格静止窗口的最短时长
QUIET_BAR_SD_N = 15.0          # 杆力波动上限（对应“Robot force 不变”）
# |bar_y| 上限。注意 external_forces 里 F_bar 含 Mb*g，杆架在架子上时
# 这一项未必是 0，所以这里不强求它为 0，而是把实测 bar_y 计入“应有值”，
# 同时把 bar_y 的均值和最大值打出来，让人自己判断杆到底卸没卸。
QUIET_BAR_MAX_N = 250.0
QUIET_GRF_SD_N = 25.0          # 鞋垫总力波动上限
QUIET_ANGLE_SD_DEG = 1.5       # 各关节角波动上限（对应“angle 几乎不变”）
QUIET_COM_SD_M = 0.004         # 质心漂移上限（对应“position/velocity 不变”）
QUIET_ONLY_BEFORE_SEGMENT = True   # 只在第一段动作之前找（用户指的“前一小段”）
QUIET_K_SPREAD_WARN = 0.05     # 多个静止窗口标出的 k 若差这么多，说明不可信
QUIET_ANGLE_COORDS = ('knee_angle_l', 'knee_angle_r',
                      'hip_flexion_l', 'hip_flexion_r',
                      'ankle_angle_l', 'ankle_angle_r')

# ---- insole_heel_offset_x 标定 ----
# 各组反推出的偏置若离散超过这个值，就不是一个常数配准误差，不能用常数修。
HEEL_OFFSET_SPREAD_WARN_M = 0.02


# 人体测量学质量分数（Winter，占总体重比例），键 = Xsens 节段名。
# 躯干 0.497 沿脊柱分成五段；肩段归入躯干；足尖归入足。
# 这些分数只用于 M2，而 M2 只是交叉验证，几个百分点的误差不影响结论。
MASS_FRACTIONS = {
    'Pelvis': 0.142, 'L5': 0.090, 'L3': 0.090, 'T12': 0.090, 'T8': 0.085,
    'Neck': 0.020, 'Head': 0.061,
    'Right Shoulder': 0.0, 'Left Shoulder': 0.0,
    'Right Upper Arm': 0.028, 'Left Upper Arm': 0.028,
    'Right Forearm': 0.016, 'Left Forearm': 0.016,
    'Right Hand': 0.006, 'Left Hand': 0.006,
    'Right Upper Leg': 0.100, 'Left Upper Leg': 0.100,
    'Right Lower Leg': 0.0465, 'Left Lower Leg': 0.0465,
    'Right Foot': 0.0145, 'Left Foot': 0.0145,
    'Right Toe': 0.0, 'Left Toe': 0.0,
}


# ============================================================
#  路径
# ============================================================

def get_base_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


_canon_load_key = canonical_load_key


def get_shear_ext_dir(config, base_dir, load_key):
    return os.path.join(base_dir, 'result', config['experiment_label'],
                        'opensim', 'external_forces_' + SHEAR_TAG,
                        str(load_key))


def get_shear_id_dir(config, base_dir, load_key):
    return os.path.join(base_dir, 'result', config['experiment_label'],
                        'opensim', 'inverse_dynamics_' + SHEAR_TAG,
                        str(load_key))


# ============================================================
#  信号处理小工具
# ============================================================

def _lowpass(x, fs, cutoff=FILTER_CUTOFF_HZ, order=FILTER_ORDER):
    '''零相位低通。长度不够或含 nan 时原样返回。'''
    x = np.asarray(x, dtype=float)
    if len(x) < 3 * (order + 1) or not np.all(np.isfinite(x)):
        return x
    nyq = 0.5 * fs
    wn = min(cutoff / nyq, 0.99)
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, x)


def _second_derivative(t, x):
    '''二阶导数。np.gradient 两次，允许非均匀时间轴。'''
    v = np.gradient(np.asarray(x, dtype=float), t)
    return np.gradient(v, t)


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None

    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        return r

    rx, ry = rank(x), rank(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def window_mask_from_segment(times, seg):
    '''用切片结果逐段构造布尔并集掩码。

    不用 [min, max] 包络：两次深蹲之间的站立、调整站位都在包络里，
    而那些帧的剪切与残差统计意义不同。
    '''
    times = np.asarray(times, dtype=float)
    if seg is None or len(seg) == 0 or 'time' not in seg.columns:
        return np.ones(times.shape, dtype=bool)
    if 'segment_id' in seg.columns:
        mask = np.zeros(times.shape, dtype=bool)
        for sid in seg['segment_id'].unique():
            sub = seg[seg['segment_id'] == sid]
            if len(sub) < 10:
                continue
            mask |= ((times >= float(sub['time'].min()))
                     & (times <= float(sub['time'].max())))
        if mask.any():
            return mask
    return ((times >= float(seg['time'].min()))
            & (times <= float(seg['time'].max())))


# ============================================================
#  mot 低通滤波
# ============================================================

def _read_mot_raw(path):
    '''把 .mot 拆成 (header 行列表, 列名, 数值矩阵)。

    不用 osim.Storage 读写：Storage 会重排列、丢掉原 header，
    而 ID / IK 下游对列序和 inDegrees 标志是敏感的。
    直接按文本处理，header 原样抄回去最安全。
    '''
    try:
        with open(path, 'r') as fh:
            lines = fh.read().splitlines()
    except Exception:
        return None
    end = None
    for i, ln in enumerate(lines):
        if ln.strip().lower() == 'endheader':
            end = i
            break
    if end is None or end + 2 >= len(lines):
        return None
    header = lines[:end + 1]
    cols = lines[end + 1].split()
    rows = []
    for ln in lines[end + 2:]:
        if not ln.strip():
            continue
        try:
            rows.append([float(v) for v in ln.split()])
        except ValueError:
            return None
    if not rows:
        return None
    return header, cols, np.asarray(rows, dtype=float)


def write_filtered_mot(src, dst, cutoff=MOT_FILTER_CUTOFF_HZ,
                       order=MOT_FILTER_ORDER, key=''):
    '''对 .mot 的每一列坐标做零相位低通，另存为新文件，返回新路径。

    失败时返回原路径（宁可不滤，也不能悄悄换成坏文件）。
    '''
    if REUSE_FILTERED_MOT and os.path.exists(dst):
        try:
            if os.path.getmtime(dst) >= os.path.getmtime(src):
                print('  [CACHE] {} 复用已滤波 mot。'.format(key))
                return dst
        except Exception:
            pass
    pack = _read_mot_raw(src)
    if pack is None:
        beauty_print('  [滤波] {} 的 mot 解析失败，改用未滤波文件。'.format(key),
                     type='warning')
        return src
    header, cols, data = pack
    if data.ndim != 2 or len(data) < 30:
        beauty_print('  [滤波] {} 的 mot 只有 {} 帧，太短，不滤波。'
                     .format(key, len(data)), type='warning')
        return src
    t = data[:, 0]
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        beauty_print('  [滤波] {} 的 mot 时间轴异常，不滤波。'.format(key),
                     type='warning')
        return src
    fs = 1.0 / dt
    if cutoff >= 0.5 * fs:
        beauty_print('  [滤波] {} 的采样率只有 {:.1f} Hz，截止 {:.1f} Hz 超过'
                     ' Nyquist，不滤波。'.format(key, fs, cutoff),
                     type='warning')
        return src

    out = data.copy()
    skipped = []
    worst_name, worst_val = '', 0.0
    for j in range(1, data.shape[1]):
        v = data[:, j]
        if not np.all(np.isfinite(v)):
            skipped.append(cols[j] if j < len(cols) else str(j))
            continue
        if float(np.std(v)) < 1e-12:
            continue
        f = _lowpass(v, fs, cutoff, order)
        out[:, j] = f
        d = float(np.max(np.abs(f - v)))
        if d > worst_val:
            worst_val, worst_name = d, (cols[j] if j < len(cols) else str(j))

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        with open(dst, 'w') as fh:
            for ln in header:
                fh.write(ln + '\n')
            fh.write('\t'.join(cols) + '\n')
            for row in out:
                fh.write('\t'.join('{:.8f}'.format(v) for v in row) + '\n')
    except Exception as exc:
        beauty_print('  [滤波] {} 写盘失败: {}，改用未滤波文件。'
                     .format(key, exc), type='warning')
        return src

    print('  [滤波] {:<8} {} 帧 @ {:.1f} Hz -> {:.1f} Hz 低通；'
          '改动最大的坐标是 {}，最大改动 {:.3f}。'.format(
              key, len(out), fs, cutoff, worst_name, worst_val))
    if skipped:
        beauty_print('  [滤波] {} 有 {} 列含 nan 未滤波: {}。'
                     '这些列会带着原始噪声进 ID。'.format(
                         key, len(skipped), skipped[:6]), type='warning')
    return dst


# ============================================================
#  鞋垫竖直增益标定：静止站立段
# ============================================================

def _rolling_sd(v, win):
    s = pd.Series(np.asarray(v, dtype=float))
    return s.rolling(win, center=True,
                     min_periods=max(3, win // 2)).std().values


def _find_runs(mask, min_len):
    '''返回 mask 里所有长度 >= min_len 的连续 True 区间 [(i0, i1), ...]。'''
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def calibrate_insole_gain(key, sto_df, kin, mot_path, body_mass, seg=None):
    '''用“站着不动、杆已卸载”的帧标定鞋垫竖直增益 k，返回 k（实测/应有）。

    原理
    ----
    静止时 a_COM = 0，竖直方向的牛顿方程退化成纯静力学：
        GRF_y_应有 = m*g + |bar_y|
    杆真的卸载时第二项也没了，于是应有值就是体重，一个不含微分、
    不含 COP、不含任何模型假设的绝对基准。定义
        k = GRF_y_实测 / GRF_y_应有
    k < 1 就是鞋垫欠读，修法是把鞋垫力整体除以 k。

    这一路与 V0-gain 完全独立：V0-gain 用九组不同负载的斜率（相对量），
    这一路用单组的绝对值。两者吻合才能确认“乘性欠读”这个结论。

    判定“静止”的四道闸门（对应用户提的三条：force / position、velocity / angle）
    ----------------------------------------------------------------
      1. 杆力平稳     滑窗内 |bar_y| 的标准差 < QUIET_BAR_SD_N，且均值不超上限
      2. 鞋垫总力平稳 滑窗内 GRF 标准差 < QUIET_GRF_SD_N，且人确实站在垫上
      3. 关节角不动   六个下肢角的滑窗标准差都 < QUIET_ANGLE_SD_DEG
      4. 质心不动     质心前后与高度的滑窗标准差 < QUIET_COM_SD_M
                      （质心不动 = position 与 velocity 都不变，比直接看
                        robot 的 pos/vel 更严，且与 .sto 同一时钟）
    四道闸门每一道单独的通过率都会打出来，某一道卡死时一眼能看到是哪一道。
    '''
    print('\n' + '=' * 80)
    print('[CAL-gain] {} 静止站立段标定鞋垫竖直增益'.format(key))
    print('=' * 80)

    if sto_df is None or 'time' not in sto_df.columns:
        beauty_print('  [CAL-gain] {} 没有可用的 .sto，跳过。'.format(key),
                     type='warning')
        return None

    t = sto_df['time'].values.astype(float)
    n = len(t)

    def col(name):
        return (sto_df[name].values.astype(float)
                if name in sto_df.columns else np.zeros(n))

    bar_y = col('bar_force_vy')
    grf_l = col('grf_l_vy')
    grf_r = col('grf_r_vy')
    grf = grf_l + grf_r
    mg = body_mass * 9.81

    dt = float(np.median(np.diff(t))) if n > 2 else 0.01
    fs = 1.0 / dt if dt > 0 else 100.0
    win = max(3, int(round(QUIET_WIN_S * fs)))
    min_len = max(win, int(round(QUIET_MIN_DUR_S * fs)))
    print('  采样率 {:.1f} Hz，滑窗 {:.2f} s（{} 帧），'
          '最短窗口 {:.2f} s（{} 帧），总帧数 {}。'.format(
              fs, QUIET_WIN_S, win, QUIET_MIN_DUR_S, min_len, n))
    print('  体重 m = {:.2f} kg，m*g = {:.1f} N。'.format(body_mass, mg))

    sd_bar = _rolling_sd(bar_y, win)
    sd_grf = _rolling_sd(grf, win)

    # 关节角：从 mot 插到 .sto 的时间轴上。
    sd_ang = np.zeros(n)
    ang_used = []
    mdf = read_opensim_table(mot_path) if mot_path else None
    ang_series = {}
    if mdf is not None and 'time' in mdf.columns:
        mt = mdf['time'].values.astype(float)
        for c in QUIET_ANGLE_COORDS:
            if c not in mdf.columns:
                continue
            v = np.interp(t, mt, mdf[c].values.astype(float))
            ang_series[c] = v
            sd_ang = np.maximum(sd_ang, np.nan_to_num(_rolling_sd(v, win),
                                                      nan=0.0))
            ang_used.append(c)
    if not ang_used:
        beauty_print('  [CAL-gain] mot 里找不到任何下肢角，'
                     '“关节角不动”这道闸门失效，只靠力与质心判定。',
                     type='warning')

    # 质心：前后与高度都要稳。
    sd_com = np.zeros(n)
    if kin is not None:
        for axis in (0, 1):
            v = np.interp(t, kin['time'], kin['com'][:, axis])
            sd_com = np.maximum(sd_com, np.nan_to_num(_rolling_sd(v, win),
                                                      nan=0.0))
    else:
        beauty_print('  [CAL-gain] 没有质心序列，“质心不动”这道闸门失效。',
                     type='warning')

    gates = [
        ('1 杆力平稳 sd<{:.0f}N'.format(QUIET_BAR_SD_N),
         ~(np.nan_to_num(sd_bar, nan=1e9) > QUIET_BAR_SD_N)),
        ('1b |bar_y|<{:.0f}N'.format(QUIET_BAR_MAX_N),
         np.abs(bar_y) <= QUIET_BAR_MAX_N),
        ('2 GRF平稳 sd<{:.0f}N'.format(QUIET_GRF_SD_N),
         ~(np.nan_to_num(sd_grf, nan=1e9) > QUIET_GRF_SD_N)),
        ('2b 人站在垫上', grf > 0.5 * mg),
        ('3 关节角 sd<{:.1f}°'.format(QUIET_ANGLE_SD_DEG),
         sd_ang <= QUIET_ANGLE_SD_DEG),
        ('4 质心 sd<{:.0f}mm'.format(1000 * QUIET_COM_SD_M),
         sd_com <= QUIET_COM_SD_M),
    ]

    print('\n  各道闸门单独的通过情况：')
    print('  {:<24}{:>10}{:>10}'.format('闸门', '通过帧', '比例'))
    for name, g in gates:
        print('  {:<24}{:>10}{:>9.1%}'.format(
            name, int(np.sum(g)), float(np.mean(g))))

    mask = np.ones(n, dtype=bool)
    for _, g in gates:
        mask &= g
    print('  {:<24}{:>10}{:>9.1%}'.format(
        '全部同时满足', int(np.sum(mask)), float(np.mean(mask))))

    limited = False
    if QUIET_ONLY_BEFORE_SEGMENT and seg is not None and len(seg) > 0 \
            and 'time' in seg.columns:
        t0_seg = float(seg['time'].min())
        before = mask & (t < t0_seg)
        if _find_runs(before, min_len):
            mask = before
            limited = True
            print('  只取第一段动作（t = {:.2f} s）之前的帧，'
                  '即用户说的“前一小段”。'.format(t0_seg))
        else:
            print('  第一段动作（t = {:.2f} s）之前没有够长的静止窗口，'
                  '放开到整条时间轴。'.format(t0_seg))

    runs = _find_runs(mask, min_len)
    if not runs:
        beauty_print('  [CAL-gain] {} 找不到长度 >= {:.2f} s 的静止窗口。'
                     '请看上面哪一道闸门通过率最低，再放宽对应的阈值。'
                     .format(key, QUIET_MIN_DUR_S), type='warning')
        return None

    print('\n  候选静止窗口（共 {} 个）：'.format(len(runs)))
    print('  {:>8}{:>8}{:>7}{:>8}{:>10}{:>9}{:>10}{:>9}{:>8}{:>8}'.format(
        't0(s)', 't1(s)', '时长', 'n', 'bar(N)', 'bar sd',
        'GRF(N)', 'GRF sd', 'L占比', 'k'))
    rows = []
    for i0, i1 in runs:
        sl = slice(i0, i1 + 1)
        b_m = float(np.mean(bar_y[sl]))
        b_s = float(np.std(bar_y[sl]))
        g_m = float(np.mean(grf[sl]))
        g_s = float(np.std(grf[sl]))
        l_m = float(np.mean(grf_l[sl]))
        r_m = float(np.mean(grf_r[sl]))
        need = mg + abs(b_m)
        kk = g_m / need if need > 1e-6 else float('nan')
        share = l_m / g_m if abs(g_m) > 1e-6 else float('nan')
        rows.append({'i0': i0, 'i1': i1, 'n': i1 - i0 + 1,
                     't0': t[i0], 't1': t[i1], 'dur': t[i1] - t[i0],
                     'bar': b_m, 'bar_sd': b_s, 'grf': g_m, 'grf_sd': g_s,
                     'l': l_m, 'r': r_m, 'need': need, 'k': kk})
        print('  {:>8.2f}{:>8.2f}{:>7.2f}{:>8}{:>10.1f}{:>9.1f}'
              '{:>10.1f}{:>9.1f}{:>8.1%}{:>8.4f}'.format(
                  t[i0], t[i1], t[i1] - t[i0], i1 - i0 + 1,
                  b_m, b_s, g_m, g_s, share, kk))

    ks = np.array([r['k'] for r in rows if np.isfinite(r['k'])], dtype=float)
    if len(ks) >= 2 and float(np.max(ks) - np.min(ks)) > QUIET_K_SPREAD_WARN:
        beauty_print('  [CAL-gain] {} 各静止窗口标出的 k 从 {:.4f} 到 {:.4f}，'
                     '相差 {:.3f}，超过 {:.3f}。同一次采集里增益不该变，'
                     '说明有窗口其实没静止，或鞋垫有漂移/蠕变。'
                     .format(key, float(np.min(ks)), float(np.max(ks)),
                             float(np.max(ks) - np.min(ks)),
                             QUIET_K_SPREAD_WARN), type='warning')

    # 选最长的那个窗口作为结论；最长 = 最稳，且平均得最充分。
    best = max(rows, key=lambda r: r['n'])
    sl = slice(best['i0'], best['i1'] + 1)
    print('\n  ---- 选中最长的窗口作为标定结果 ----')
    print('  时间区间          [{:.3f}, {:.3f}] s，共 {} 帧（{:.2f} s）'
          .format(best['t0'], best['t1'], best['n'], best['dur']))
    if limited:
        print('  （取自第一段动作之前的静止段）')
    print('  体重项 m*g        = {:9.1f} N'.format(mg))
    print('  杆力项 |bar_y|    = {:9.1f} N   （sd {:.1f} N，'
          '窗口内最大 {:.1f} N）'.format(
              abs(best['bar']), best['bar_sd'],
              float(np.max(np.abs(bar_y[sl])))))
    print('  应有 GRF_y        = {:9.1f} N   = m*g + |bar_y|'
          .format(best['need']))
    print('  实测 GRF_y        = {:9.1f} N   （sd {:.1f} N）'
          .format(best['grf'], best['grf_sd']))
    print('     左 {:.1f} N + 右 {:.1f} N，左占 {:.1%}'.format(
        best['l'], best['r'],
        best['l'] / best['grf'] if abs(best['grf']) > 1e-6 else float('nan')))
    print('  缺口              = {:+9.1f} N   （体重的 {:.1%}）'.format(
        best['grf'] - best['need'],
        abs(best['grf'] - best['need']) / mg if mg > 0 else float('nan')))
    print('  ==> 鞋垫增益 k = 实测 / 应有 = {:.4f}，欠读 {:+.1%}'.format(
        best['k'], 1.0 - best['k']))
    print('      修正系数 1/k = {:.4f}（把鞋垫竖直力乘这个数）'.format(
        1.0 / best['k'] if abs(best['k']) > 1e-6 else float('nan')))
    k_bw_only = best['grf'] / mg if mg > 0 else float('nan')
    print('      若认为杆已完全卸载（忽略 bar_y）：k0 = {:.4f}'.format(k_bw_only))

    if abs(best['bar']) > 20.0:
        beauty_print('  [CAL-gain] 选中窗口里 |bar_y| 均值仍有 {:.0f} N，'
                     '杆并没有完全卸载。此时 k 依赖 bar_y 记得对不对，'
                     '不再是纯粹的“只有体重”基准。若 k 与 k0 差得多，'
                     '要先确认 external_forces 里的 Mb 设置。'
                     .format(abs(best['bar'])), type='warning')

    if ang_used:
        print('\n  窗口内各关节角标准差（越小越静止）：')
        for c in ang_used:
            v = ang_series[c][sl]
            print('    {:<16}sd {:6.3f}°，极差 {:6.3f}°，均值 {:+8.2f}°'
                  .format(c, float(np.std(v)), float(np.ptp(v)),
                          float(np.mean(v))))
    if kin is not None:
        cx = np.interp(t, kin['time'], kin['com'][:, 0])[sl]
        cy = np.interp(t, kin['time'], kin['com'][:, 1])[sl]
        print('  窗口内质心：前后 sd {:.1f} mm（极差 {:.1f} mm），'
              '高度 sd {:.1f} mm（极差 {:.1f} mm）'.format(
                  1000 * float(np.std(cx)), 1000 * float(np.ptp(cx)),
                  1000 * float(np.std(cy)), 1000 * float(np.ptp(cy))))

    return best['k']


# ============================================================
#  insole_heel_offset_x 标定
# ============================================================

def recommend_heel_offset(stats, current_offset):
    '''用准静态下的 COM-COP 前后错位反推 insole_heel_offset_x。

    标定原理
    --------
    慢速深蹲近似准静态，人不倒就说明质心必须落在压心正上方，
    所以 mean(COM_x - COP_x) 就是 COP 配准误差的直接测量，
    不依赖任何动力学假设，也不需要额外采集。
    external_forces 里 px_side = ant + heel_x，heel_x 增大会把 COP 前移，
    于是修正量就是
        heel_x_新 = heel_x_旧 + mean(COM_x - COP_x)
    判据：各组反推出来的值应当彼此接近（它是鞋垫相对足跟的安装位置，
    与负载无关）。若离散很大，那就不是一个常数偏置，别用常数去修。
    '''
    print('\n' + '=' * 80)
    print('[CAL-heel] 标定 insole_heel_offset_x')
    print('=' * 80)
    print('  当前 insole_heel_offset_x = {:+.4f} m'.format(current_offset))
    vals, keys = [], []
    for k, s in stats.items():
        d = s.get('cop_dx', float('nan'))
        if np.isfinite(d):
            vals.append(d)
            keys.append(k)
    if len(vals) < 3:
        beauty_print('  [CAL-heel] 可用组不足三组，无法标定。', type='warning')
        return None
    vals = np.asarray(vals, dtype=float)
    print('  {:<10}{:>16}{:>20}'.format(
        'load', 'COM-COP dx(cm)', '该组建议 heel_x(m)'))
    for k, d in zip(keys, vals):
        print('  {:<10}{:>16.2f}{:>20.4f}'.format(
            k, 100.0 * d, current_offset + d))
    m = float(np.mean(vals))
    sd = float(np.std(vals))
    print('  {} 组均值 {:+.2f} cm，标准差 {:.2f} cm，极差 {:.2f} cm。'.format(
        len(vals), 100.0 * m, 100.0 * sd, 100.0 * float(np.ptp(vals))))
    print('  ==> 建议 insole_heel_offset_x = {:+.4f} m'.format(
        current_offset + m))
    print('      预计消除的膝力矩误差约 {:.1f} N·m'.format(
        abs(m) * KNEE_COP_SENSITIVITY))
    if sd > HEEL_OFFSET_SPREAD_WARN_M:
        beauty_print('  [CAL-heel] 各组反推值的标准差 {:.1f} cm，超过 {:.0f} cm。'
                     '鞋垫相对足跟的安装位置与负载无关，本该是常数；'
                     '离散这么大说明还混着别的误差（足印未覆盖区、'
                     '左右垫装反、或深蹲窗口里混进了非准静态帧），'
                     '此时用常数偏置只是把平均值糊过去。'
                     .format(100.0 * sd, 100.0 * HEEL_OFFSET_SPREAD_WARN_M),
                     type='warning')
    beauty_print('  [CAL-heel] 提醒：这是自洽标定 —— 改完之后 dx 会被强行做到 0，'
                 '所以之后不能再拿 dx 当验证。独立验证要用鞋垫足印本身：'
                 'example_insole_map_check 里的首/末接触行给出足印实际占用的'
                 '行范围，把它按 29.98 cm 的垫长换算成米，再与模型 calcn 的'
                 '解剖足长对齐，两条路给出的 heel_x 应当吻合。',
                 type='warning')
    return current_offset + m


# ============================================================
#  M1 / M3 的公共前置：逐帧质心与地面系 COP
# ============================================================

def compute_com_and_cop(model_path, mot_path, sto_df, verbose=True):
    '''逐帧计算全身质心（ground）与左右 COP 的地面系坐标。

    .sto 里的 grf_*_p{x,y,z} 是 calcn 局部坐标（point_expressed_in_body
    写的就是 calcn），要用它做准静态几何估计必须先变换到地面系。
    质心与两个作用点在同一个 state 循环里一次算完，避免重复装配。

    Returns
    -------
    dict or None
        {'time', 'com'(n,3), 'cop_l'(n,3), 'cop_r'(n,3), 'mass'}
    '''
    model = osim.Model(model_path)
    state = model.initSystem()
    mass = float(model.getTotalMass(state))

    storage = osim.Storage(mot_path)
    if storage.isInDegrees():
        model.getSimbodyEngine().convertDegreesToRadians(storage)

    labels = storage.getColumnLabels()
    name_to_idx = {}
    for i in range(labels.getSize()):
        name_to_idx[labels.get(i)] = i

    coords = model.getCoordinateSet()
    coord_idx = []
    for c in range(coords.getSize()):
        nm = coords.get(c).getName()
        idx = name_to_idx.get(nm)
        if idx is not None and idx >= 1:
            coord_idx.append((c, idx - 1))
    if not coord_idx:
        beauty_print('  [SHEAR] mot 与模型没有任何同名坐标，无法算质心。',
                     type='warning')
        return None

    bodies = model.getBodySet()
    try:
        calcn_l = bodies.get('calcn_l')
        calcn_r = bodies.get('calcn_r')
    except Exception:
        calcn_l = calcn_r = None

    sto_t = None
    if sto_df is not None and 'time' in sto_df.columns:
        sto_t = sto_df['time'].values.astype(float)

    def local_cop(prefix, t):
        if sto_t is None:
            return None
        cols = [prefix + '_p' + ax for ax in ('x', 'y', 'z')]
        if not all(c in sto_df.columns for c in cols):
            return None
        return [float(np.interp(t, sto_t, sto_df[c].values.astype(float)))
                for c in cols]

    n = storage.getSize()
    times = np.zeros(n)
    com = np.zeros((n, 3))
    cop_l = np.full((n, 3), np.nan)
    cop_r = np.full((n, 3), np.nan)

    for k in range(n):
        sv = storage.getStateVector(k)
        data = sv.getData()
        t = float(sv.getTime())
        times[k] = t
        for c, d in coord_idx:
            coords.get(c).setValue(state, float(data.get(d)), False)
        # 膝关节的 coupler 约束必须靠 assemble 才能满足；
        # 跳过它会让 calcn 位姿偏，直接毁掉 M3。
        model.assemble(state)
        model.realizePosition(state)

        p = model.calcMassCenterPosition(state)
        com[k] = [p.get(0), p.get(1), p.get(2)]

        for body, out in ((calcn_l, cop_l), (calcn_r, cop_r)):
            if body is None:
                continue
            prefix = 'grf_l' if out is cop_l else 'grf_r'
            loc = local_cop(prefix, t)
            if loc is None:
                continue
            g = body.findStationLocationInGround(
                state, osim.Vec3(loc[0], loc[1], loc[2]))
            out[k] = [g.get(0), g.get(1), g.get(2)]

    if verbose:
        print('  [SHEAR] 模型总质量 {:.2f} kg，质心帧数 {}'.format(mass, n))
    return {'time': times, 'com': com, 'cop_l': cop_l, 'cop_r': cop_r,
            'mass': mass}


# ============================================================
#  逐帧质心 / COP 的磁盘缓存
# ============================================================

def _cached_com_and_cop(model_path, mot_path, sto_path, sto_df,
                        base_dir, config, key):
    '''compute_com_and_cop 每一帧都要 model.assemble()，是全程最慢的一步
    （九组合计约九万帧，且只跑在单核上）。结果只取决于模型、mot 和
    .sto 里的 COP 列，与本脚本的判据参数无关，所以按这三个文件的修改
    时间做磁盘缓存，重跑时直接读回。改了模型或重跑了 IK 会自动失效。
    '''
    cache_dir = os.path.join(base_dir, 'result', config['experiment_label'],
                             'opensim', 'cache_com_cop')
    os.makedirs(cache_dir, exist_ok=True)
    marks = []
    for p in (model_path, mot_path, sto_path):
        try:
            marks.append('{:.0f}'.format(os.path.getmtime(p)))
        except Exception:
            marks.append('na')
    path = os.path.join(cache_dir, '{}_{}.npz'.format(key, '_'.join(marks)))
    if os.path.exists(path):
        try:
            z = np.load(path)
            print('  [CACHE] {} 复用质心/COP 缓存。'.format(key))
            return {'time': z['time'], 'com': z['com'],
                    'cop_l': z['cop_l'], 'cop_r': z['cop_r'],
                    'mass': float(z['mass'])}
        except Exception:
            pass
    kin_new = compute_com_and_cop(model_path, mot_path, sto_df, verbose=False)
    if kin_new is not None:
        try:
            np.savez_compressed(path, time=kin_new['time'], com=kin_new['com'],
                                cop_l=kin_new['cop_l'], cop_r=kin_new['cop_r'],
                                mass=kin_new['mass'])
        except Exception:
            pass
    return kin_new


# ============================================================
#  M2：Xsens Segment Acceleration
# ============================================================

def _cached_xsens_com(xsens_path, base_dir, config, key):
    '''pd.read_excel 每次都要用 openpyxl 把整本工作簿解析一遍。Xsens 的 xlsx
    有八九个表、上万行，单次读取就要几十秒，九组下来是分钟级开销。
    按文件修改时间缓存到 npz。
    '''
    if not xsens_path:
        return None
    cache_dir = os.path.join(base_dir, 'result', config['experiment_label'],
                             'opensim', 'cache_xsens_com')
    os.makedirs(cache_dir, exist_ok=True)
    try:
        stamp = '{:.0f}'.format(os.path.getmtime(xsens_path))
    except Exception:
        stamp = 'na'
    # v2 = 加入手性与朝向修正之后的口径。改口径必须换版本号，
    # 否则会读到按旧坐标算的缓存。
    path = os.path.join(cache_dir, '{}_{}_v2.npz'.format(key, stamp))
    if os.path.exists(path):
        try:
            z = np.load(path)
            print('  [CACHE] {} 复用 Xsens 质心缓存。'.format(key))
            pos = z['pos'] if bool(z['has_pos']) else None
            ps = (z['psi_stat'] if 'psi_stat' in z.files
                  else np.full(3, np.nan))
            _print_psi(ps, tag=str(key) + ' ')
            return {'time': z['time'], 'acc': z['acc'], 'pos': pos,
                    'weight': float(z['weight']),
                    'source': str(z['source'].item()),
                    'psi_stat': ps}
        except Exception:
            pass
    res = com_accel_from_xsens(xsens_path, verbose=True)
    if res is not None:
        try:
            has_pos = res.get('pos') is not None
            np.savez_compressed(
                path, time=res['time'], acc=res['acc'],
                pos=res['pos'] if has_pos else np.zeros((1, 3)),
                has_pos=has_pos, weight=res.get('weight', 1.0),
                psi_stat=res.get('psi_stat', np.full(3, np.nan)),
                source=res.get('source', ''))
        except Exception:
            pass
    return res


def _print_psi(psi_stat, tag=''):
    '''打印骨盆朝向角统计。缓存命中时也要打，否则重跑就看不到这一行，
    而它恰恰是判断“要不要做朝向旋转”的唯一依据。

    判读：
      均值近 0 且极差小 -> Xsens 的 +x 就是受试者的前方，朝向项可以不管；
      均值几十度        -> 标定朝向与实际站位差一个常数角，必须转；
      极差大            -> 人在过程中转身了，只能逐帧转。
    '''
    psi_stat = np.asarray(psi_stat, dtype=float)
    if psi_stat.size < 3 or not np.isfinite(psi_stat[0]):
        return
    print('  [SHEAR] {}骨盆朝向 psi：均值 {:+.1f}°，标准差 {:.1f}°，'
          '极差 {:.1f}°。均值远离 0 说明 Xsens 的 +x 不是受试者的前方。'
          .format(tag, psi_stat[0], psi_stat[1], psi_stat[2]))


def _heading_from_quat(xls):
    '''从 Segment Orientation - Quat 取骨盆绕竖直轴的朝向角 psi（弧度）。

    Xsens 全局系里竖直轴是 z，所以 psi 就是四元数的 yaw：
        psi = atan2(2(w*z + x*y), 1 - 2(y^2 + z^2))
    用实测第 0 帧 Pelvis 四元数 (0.998891, 0.029077, -0.01542, -0.03366)
    算得 psi = -3.91 度，与同一帧 Euler 表的第三个分量 -3.90862 吐合，
    可以确认列序是 (w, x, y, z)。
    '''
    try:
        dq = pd.read_excel(xls, sheet_name='Segment Orientation - Quat')
    except Exception:
        return None
    cands = [['Pelvis q0', 'Pelvis q1', 'Pelvis q2', 'Pelvis q3'],
             ['Pelvis q_w', 'Pelvis q_x', 'Pelvis q_y', 'Pelvis q_z'],
             ['Pelvis w', 'Pelvis x', 'Pelvis y', 'Pelvis z']]
    cols = None
    for c in cands:
        if all(x in dq.columns for x in c):
            cols = c
            break
    if cols is None:
        pel = [c for c in dq.columns if str(c).startswith('Pelvis')]
        cols = pel[:4] if len(pel) >= 4 else None
    if cols is None:
        return None
    w = dq[cols[0]].values.astype(float)
    x = dq[cols[1]].values.astype(float)
    y = dq[cols[2]].values.astype(float)
    z = dq[cols[3]].values.astype(float)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _fit_psi(psi, n):
    '''把朝向角重采样到长度 n（不同表的帧数可能差一两帧）。'''
    if psi is None:
        return None
    if len(psi) == n:
        return psi
    if len(psi) < 2:
        return None
    return np.interp(np.linspace(0.0, 1.0, n),
                     np.linspace(0.0, 1.0, len(psi)), psi)


def _to_opensim_horizontal(a_fwd, a_left, psi):
    '''把 Xsens 全局水平分量 (前, 左) 转到 OpenSim 的 (x=前, z=右)。

    psi = 受试者前向在 Xsens 水平面内相对 +x 轴的转角，向 +y（左）为正。
        e_前  = ( cos psi, sin psi)
        e_左  = (-sin psi, cos psi)
        x_os =  a · e_前
        z_os = -a · e_左      （OpenSim 的 z 是右，与左相反）
    psi = 0 时退化为 x_os = a_fwd, z_os = -a_left，即单纯的手性修正。
    '''
    if psi is None:
        c, s = 1.0, 0.0
    else:
        c, s = np.cos(psi), np.sin(psi)
    return c * a_fwd + s * a_left, s * a_fwd - c * a_left


def resolve_xsens_path(config, file_info):
    xf = file_info.get('xsens_file')
    if not xf:
        return None
    if os.path.isabs(xf) and os.path.exists(xf):
        return xf
    folder = config['folder']
    modeling = config['modeling_file']
    for cand in (os.path.join(folder, modeling.get('xsens_folder', ''), xf),
                 os.path.join(folder, xf)):
        if os.path.exists(cand):
            return cand
    return None


def com_accel_from_xsens(xsens_path, verbose=True):
    '''从 Segment Acceleration 表加权求质心加速度（不做微分）。

    坐标映射：
        opensim_x = +xsens_x   (前 = 前)
        opensim_y = +xsens_z   (上 = 上)
        opensim_z = -xsens_y   (右 = -左)   <-- 注意这个负号

    负号是必须的：Xsens 全局系 (前, 左, 上) 与 OpenSim (前, 上, 右) 都是右手系，
    只交换 y/z 而不取负会得到行列式为 -1 的变换，等于把人镜像一次，左右互换。
    xsens_processor._process_excel 里的裸交换之所以没出问题，是因为
    mot_pipeline 的 sign_flip 里带了 'pelvis_tz': -1，在后面补上了这个负号；
    本脚本直接读 Xsens，不经过那一步，必须自己取负。

    另外还要绕竖直轴转一个朝向角 psi（见 _to_opensim_horizontal）：
    Xsens 的 +x 是标定时的朝向，未必就是受试者的前方。
    重力是竖直的，不影响两个水平分量。
    '''
    try:
        # 一次打开工作簿，后面几个表共用，避免重复解析。
        xls = pd.ExcelFile(xsens_path)
        df_info = pd.read_excel(xls, sheet_name='General Information',
                                header=None)
        fs = 60.0
        for _, row in df_info.iterrows():
            if row[0] == 'Frame Rate':
                fs = float(row[1])
                break
        # 骨盆朝向角，用于把 Xsens 全局水平分量转到 OpenSim 的 前/右。
        psi = _heading_from_quat(xls) if APPLY_XSENS_HEADING else None
        # 统计量随缓存一起存盘，见 _print_psi。
        if psi is None:
            psi_stat = np.full(3, np.nan)
        else:
            psi_stat = np.array([np.degrees(np.mean(psi)),
                                 np.degrees(np.std(psi)),
                                 np.degrees(np.ptp(psi))])
        _print_psi(psi_stat)
        # 优先用 Xsens 自带的 Center of Mass 表：它是 Xsens 全身模型直接输出的
        # 质心位置/速度/加速度，既不需要 Winter 通用质量分数，也不需要微分，
        # 比按节段加权求和更可信。
        # 实测第 0 帧 CoM acc z ≈ -0.034 m/s² 而不是 ±9.81，说明它已扣除重力，
        # 因此竖直分量可以直接拿来做竖直力平衡。
        try:
            df_com = pd.read_excel(xls, sheet_name='Center of Mass')
        except Exception:
            df_com = None
        if df_com is not None:
            cols_acc = ['CoM acc x', 'CoM acc y', 'CoM acc z']
            cols_pos = ['CoM pos x', 'CoM pos y', 'CoM pos z']
            if all(c in df_com.columns for c in cols_acc):
                if 'Frame' in df_com.columns:
                    tcom = df_com['Frame'].values.astype(float) / fs
                else:
                    tcom = np.arange(len(df_com), dtype=float) / fs
                cx = df_com[cols_acc[0]].values.astype(float)   # 前
                cy = df_com[cols_acc[1]].values.astype(float)   # 左
                cz = df_com[cols_acc[2]].values.astype(float)   # 上
                # 手性修正 + 朝向旋转，见 _to_opensim_horizontal。
                gx, gz = _to_opensim_horizontal(cx, cy, _fit_psi(psi, len(cx)))
                acc_com = np.column_stack([gx, cz, gz])   # -> OpenSim xyz
                pos_com = None
                if all(c in df_com.columns for c in cols_pos):
                    px = df_com[cols_pos[0]].values.astype(float)
                    py = df_com[cols_pos[1]].values.astype(float)
                    pz = df_com[cols_pos[2]].values.astype(float)
                    # 位置只做手性修正，不做朝向旋转：Xsens 全局原点与 OpenSim
                    # ground 原点不是同一点，所以这组坐标只有差分才有意义。
                    pos_com = np.column_stack([px, pz, -py])
                if verbose:
                    print('  [SHEAR] 使用 Xsens Center of Mass 表（无需微分）。')
                return {'time': tcom, 'acc': acc_com, 'pos': pos_com,
                        'weight': 1.0, 'source': 'xsens_com',
                        'psi_stat': psi_stat}
        df = pd.read_excel(xls, sheet_name='Segment Acceleration')
    except Exception as exc:
        beauty_print('  [SHEAR] 读 Segment Acceleration 失败: {}'.format(exc),
                     type='warning')
        return None

    if 'Frame' in df.columns:
        time = df['Frame'].values.astype(float) / fs
    else:
        time = np.arange(len(df), dtype=float) / fs

    total_w = 0.0
    acc = np.zeros((len(df), 3))
    missing = []
    for seg, frac in MASS_FRACTIONS.items():
        if frac <= 0:
            continue
        cols = [seg + ' x', seg + ' y', seg + ' z']
        if not all(c in df.columns for c in cols):
            missing.append(seg)
            continue
        ax = df[cols[0]].values.astype(float)   # 前
        ay = df[cols[1]].values.astype(float)   # 左
        az = df[cols[2]].values.astype(float)   # 上
        gx, gz = _to_opensim_horizontal(ax, ay, _fit_psi(psi, len(ax)))
        acc[:, 0] += frac * gx
        acc[:, 1] += frac * az
        acc[:, 2] += frac * gz
        total_w += frac

    if total_w < 0.5:
        beauty_print('  [SHEAR] Segment Acceleration 可用节段质量只占 '
                     '{:.0%}，不足以代表全身质心。'.format(total_w),
                     type='warning')
        return None
    if missing and verbose:
        print('  [SHEAR] Segment Acceleration 缺失节段: {}'.format(missing))

    acc /= total_w        # 归一化，补偿缺失节段
    return {'time': time, 'acc': acc, 'pos': None, 'weight': total_w,
            'source': 'segment_sum', 'psi_stat': psi_stat}


# ============================================================
#  重建三路剪切
# ============================================================

def reconstruct_shear(kin, xsens_acc, sto_df, body_mass, verbose=True):
    '''返回 {method: (n,2) 水平剪切 [Fx, Fz]}，对齐到 .sto 的时间轴。

    符号约定：返回的是地面施加给人体的总剪切力。
    牛顿第二定律：M * a_COM = F_grf + F_bar，水平方向上
        F_shear = M * a_COM_horizontal - F_bar_horizontal
    目前杆力只有竖直分量，所以第二项为 0，但仍然显式减去，
    以免将来给杆力加了水平分量后这里静默出错。
    '''
    t = sto_df['time'].values.astype(float)
    n = len(t)
    out = {}

    def col(name):
        return (sto_df[name].values.astype(float)
                if name in sto_df.columns else np.zeros(n))

    bar_x = col('bar_force_vx')
    bar_z = col('bar_force_vz')

    # ---- M1 质心加速度 ----
    if kin is not None:
        kt = kin['time']
        dt = np.median(np.diff(kt)) if len(kt) > 2 else 1.0 / 60.0
        fs = 1.0 / dt if dt > 0 else 60.0
        ax = _second_derivative(kt, _lowpass(kin['com'][:, 0], fs))
        az = _second_derivative(kt, _lowpass(kin['com'][:, 2], fs))
        out['com_accel'] = np.column_stack([
            body_mass * np.interp(t, kt, ax) - bar_x,
            body_mass * np.interp(t, kt, az) - bar_z,
        ])

    # ---- M2 分段加速度 ----
    if xsens_acc is not None:
        xt = xsens_acc['time']
        ax = xsens_acc['acc'][:, 0]
        az = xsens_acc['acc'][:, 2]
        out['segment_accel'] = np.column_stack([
            body_mass * np.interp(t, xt, ax) - bar_x,
            body_mass * np.interp(t, xt, az) - bar_z,
        ])

    # ---- M3 准静态几何 ----
    if kin is not None:
        fy_l = col('grf_l_vy')
        fy_r = col('grf_r_vy')
        fy = fy_l + fy_r
        kt = kin['time']
        com_x = np.interp(t, kt, kin['com'][:, 0])
        com_y = np.interp(t, kt, kin['com'][:, 1])
        com_z = np.interp(t, kt, kin['com'][:, 2])

        def interp_cop(arr, axis):
            v = arr[:, axis]
            ok = np.isfinite(v)
            if ok.sum() < 2:
                return np.full(n, np.nan)
            return np.interp(t, kt[ok], v[ok])

        w_l = np.where(fy > 1.0, fy_l / np.maximum(fy, 1e-6), 0.5)
        w_r = 1.0 - w_l
        cop_x = w_l * interp_cop(kin['cop_l'], 0) + w_r * interp_cop(kin['cop_r'], 0)
        cop_y = w_l * interp_cop(kin['cop_l'], 1) + w_r * interp_cop(kin['cop_r'], 1)
        cop_z = w_l * interp_cop(kin['cop_l'], 2) + w_r * interp_cop(kin['cop_r'], 2)

        dy = com_y - cop_y
        # 质心低于/接近 COP 时这个几何估计失效（除以近零），置 nan。
        bad = ~np.isfinite(dy) | (dy < 0.3)
        with np.errstate(invalid='ignore', divide='ignore'):
            qx = fy * (com_x - cop_x) / dy
            qz = fy * (com_z - cop_z) / dy
        qx[bad] = np.nan
        qz[bad] = np.nan
        out['quasi_static'] = np.column_stack([qx, qz])

    return out


# ============================================================
#  写补了剪切的 .sto / .xml
# ============================================================

def write_patched_external_loads(config, base_dir, load_key,
                                 sto_df, shear, verbose=True):
    '''把剪切按各侧竖直力占比分配到左右足，写新的 .sto 与 .xml。

    按竖直力分配是最少假设的做法：摩擦上限正比于法向力，
    承重多的一侧能提供的剪切也多。它不会凭空造出左右不对称，
    但会把已有的竖直不对称传递到水平方向。
    '''
    ext_dir = get_shear_ext_dir(config, base_dir, load_key)
    os.makedirs(ext_dir, exist_ok=True)

    df = sto_df.copy()
    n = len(df)
    fy_l = (df['grf_l_vy'].values.astype(float)
            if 'grf_l_vy' in df.columns else np.zeros(n))
    fy_r = (df['grf_r_vy'].values.astype(float)
            if 'grf_r_vy' in df.columns else np.zeros(n))
    fy = fy_l + fy_r
    share_l = np.where(fy > 1.0, fy_l / np.maximum(fy, 1e-6), 0.5)
    share_r = 1.0 - share_l

    sx = np.nan_to_num(shear[:, 0], nan=0.0)
    sz = np.nan_to_num(shear[:, 1], nan=0.0)

    df['grf_l_vx'] = share_l * sx
    df['grf_l_vz'] = share_l * sz
    df['grf_r_vx'] = share_r * sx
    df['grf_r_vz'] = share_r * sz

    cols = list(df.columns)
    sto_path = os.path.join(ext_dir, 'bar_force_{}.sto'.format(load_key))
    with open(sto_path, 'w') as fh:
        fh.write('external_forces\n')
        fh.write('nRows={}\n'.format(n))
        fh.write('nColumns={}\n'.format(len(cols)))
        fh.write('inDegrees=no\n')
        fh.write('endheader\n')
        fh.write('\t'.join(cols) + '\n')
        values = df[cols].values.astype(float)
        for row in values:
            fh.write('\t'.join('{:.6f}'.format(v) for v in row) + '\n')

    # XML 直接沿用原来的，只换 datafile 指向。
    src_xml = os.path.join(get_ext_forces_dir(config, base_dir, load_key),
                           'bar_loads_{}.xml'.format(load_key))
    xml_path = os.path.join(ext_dir, 'bar_loads_{}.xml'.format(load_key))
    if not os.path.exists(src_xml):
        beauty_print('  [SHEAR] 找不到原 XML: {}，无法生成带剪切的外力配置。'
                     .format(src_xml), type='warning')
        return None
    with open(src_xml, 'r', encoding='utf-8') as fh:
        xml = fh.read()
    import re
    xml = re.sub(r'<datafile>.*?</datafile>',
                 '<datafile>{}</datafile>'.format(os.path.basename(sto_path)),
                 xml, flags=re.DOTALL)
    with open(xml_path, 'w', encoding='utf-8') as fh:
        fh.write(xml)

    if verbose:
        print('  [SHEAR] 已写入 {}'.format(sto_path))
    return xml_path


# ============================================================
#  验证 V1：骨盆残差
# ============================================================

def read_residuals(id_path, mask_times=None, seg=None):
    '''从 ID 结果里读骨盆平移自由度的残差力。

    对自由骨盆模型，ID 把 pelvis_tx/ty/tz 当作广义坐标，
    对应的广义力就是残差力（单位 N）。列名通常是 pelvis_tx_force。
    '''
    df = read_opensim_table(id_path)
    if df is None or 'time' not in df.columns:
        return None

    t = df['time'].values.astype(float)
    mask = np.ones(len(t), dtype=bool)
    if seg is not None:
        mask = window_mask_from_segment(t, seg)
    if not mask.any():
        mask = np.ones(len(t), dtype=bool)

    out = {}
    for axis in ('tx', 'ty', 'tz'):
        target = 'pelvis_' + axis
        hit = None
        for c in df.columns:
            cl = c.lower()
            if cl.startswith(target) and ('force' in cl or cl == target):
                hit = c
                break
        if hit is None:
            continue
        v = df[hit].values.astype(float)[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        out[axis] = {'mean': float(np.mean(v)),
                     'mean_abs': float(np.mean(np.abs(v))),
                     'rms': float(np.sqrt(np.mean(v ** 2))),
                     'max_abs': float(np.max(np.abs(v))),
                     'column': hit}
        if axis == 'tx':
            # 保留时间序列：FX 残差本身就是“缺失剪切”的逐帧测量，
            # 是检验重建结果最好的真值，比只看 rms 降没降有信息量得多。
            out['_tx_time'] = t[mask]
            out['_tx_series'] = df[hit].values.astype(float)[mask]
    return out if out else None


def print_residual_table(title, table, body_weight_n):
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)
    print('{:<10}{:>13}{:>13}{:>13}{:>13}{:>10}'.format(
        'load', 'FX rms(N)', 'FY rms(N)', 'FY mean(N)', 'FZ rms(N)', 'FY/体重'))
    for key, res in table.items():
        if not res:
            print('{:<10}  无残差列（模型骨盆可能被锁定）'.format(key))
            continue

        def g(axis):
            return res.get(axis, {}).get('rms', float('nan'))

        def gm(axis):
            return res.get(axis, {}).get('mean', float('nan'))

        fx_frac = g('tx') / body_weight_n if body_weight_n > 0 else float('nan')
        fy_frac = g('ty') / body_weight_n if body_weight_n > 0 else float('nan')
        print('{:<10}{:>13.1f}{:>13.1f}{:>13.1f}{:>13.1f}{:>10.1%}'.format(
            key, g('tx'), g('ty'), gm('ty'), g('tz'), fy_frac))
        fy_signed = gm('ty')
        # 均值 vs rms 是区分两种病因最省事的判据：
        #   均值显著非零   -> 少算或多算了一项恒定的力（账目错）
        #   均值近零而 rms 大 -> 零均值振荡，是 mot 被二次微分放大的噪声
        if np.isfinite(fy_signed) and abs(fy_signed) > 0.05 * body_weight_n:
            beauty_print('  [V0-bias] {} 的竖直残差均值 {:+.0f} N（rms {:.0f} N）。'
                         '均值显著非零，说明有一项恒定的力记错了，'
                         '不是微分噪声。请按 GRF / 杆力 / m·g 三项对账。'
                         .format(key, fy_signed, g('ty')), type='warning')
        elif (np.isfinite(fy_signed) and np.isfinite(g('ty'))
              and abs(fy_signed) < 0.2 * g('ty')):
            beauty_print('  [V0-noise] {} 的竖直残差均值只有 {:+.0f} N，'
                         '而 rms 达 {:.0f} N，几乎是零均值振荡。'
                         '这更像 mot 被二次微分放大的噪声，'
                         '而不是力的账目错误，应先滤波 mot 再看。'
                         .format(key, fy_signed, g('ty')), type='warning')
        if np.isfinite(fy_frac) and fy_frac > RESIDUAL_WARN_FRAC:
            beauty_print('  [V0] {} 的竖直残差达体重的 {:.0%}，远超 5% 判据。'
                         '竖直方向都没闭合时，水平剪切只是次级问题，'
                         '应先查竖直力平衡（模型质量 / 杆力符号 / '
                         '鞋垫总量欠读）。'.format(key, fy_frac),
                         type='warning')


# ============================================================
#  主程序
# ============================================================

def main():
    base_dir = get_base_dir()
    config_path = get_config_path()
    print('配置文件: {}'.format(config_path))
    print('基准目录: {}'.format(base_dir))

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    load_keys = get_load_keys(config, LOAD_KEYS)
    if EXCLUDE_LOAD_KEYS:
        drop = {_canon_load_key(k) for k in EXCLUDE_LOAD_KEYS}
        load_keys = [k for k in load_keys if _canon_load_key(k) not in drop]
    print('参与的组: {}'.format(load_keys))

    model_path = get_scaled_model(config, base_dir)
    if not os.path.exists(model_path):
        beauty_print('找不到缩放模型: {}'.format(model_path), type='warning')
        return

    mot_by_key = {_canon_load_key(k): v
                  for k, v in get_mot_files(config, base_dir).items()}

    # --------------------------------------------------------
    #  mot 低通滤波
    # --------------------------------------------------------
    # 放在最前面：后面所有环节（质心、M1、重跑 ID）都用 mot_by_key，
    # 在这里换掉路径，下游一处都不用改。
    if FILTER_MOT:
        print('\n' + '=' * 80)
        print('[滤波] mot 零相位低通 {:.1f} Hz（{} 阶 Butterworth，filtfilt）'
              .format(MOT_FILTER_CUTOFF_HZ, MOT_FILTER_ORDER))
        print('=' * 80)
        print('  IK 逐帧独立求解，帧间没有平滑约束；ID 的二次微分会把 f Hz 的'
              '抖动放大 (2*pi*f)^2 倍。原文件不动，滤波结果另存。')
        filt_dir = os.path.join(base_dir, 'result', config['experiment_label'],
                                'opensim', 'mot_filtered')
        for _ck in list(mot_by_key.keys()):
            _p = mot_by_key[_ck]
            if not _p or not os.path.exists(_p):
                continue
            _dst = os.path.join(
                filt_dir,
                os.path.splitext(os.path.basename(_p))[0]
                + '_lp{:.0f}hz.mot'.format(MOT_FILTER_CUTOFF_HZ))
            mot_by_key[_ck] = write_filtered_mot(_p, _dst, key=_ck)
        print('  滤波后的 mot 目录: {}'.format(filt_dir))

    subject, pipeline, pipeline_results = load_or_create_cutted_pipeline_results(
        config_path, include_xsens=False, debug=False,
        force_rebuild=FORCE_REBUILD_CUTTED_CACHE, cache_name=CACHE_NAME)

    modeling = config['modeling_file']

    # --------------------------------------------------------
    #  V1-before：先把缺口量化出来
    # --------------------------------------------------------
    seg_by_key = {}
    for key in load_keys:
        seg_by_key[key] = get_segment_from_results(
            pipeline_results, key, movement_types=MOVEMENT_TYPES)

    before = {}
    for key in load_keys:
        before[key] = read_residuals(
            get_inverse_dynamics_path(config, base_dir, key),
            seg=seg_by_key.get(key))

    _osim_cfg = config.get('opensim_settings', {})
    # subject_mass 在 config 里嵌在 opensim_settings 下（scaling.py 也是这么读的）。
    # 之前从顶层读，取不到就静默退回 70.0；数值上碰巧一致，
    # 但换个受试者就会错，而且错了也不会报。
    body_mass_guess = float(_osim_cfg.get('subject_mass',
                                          config.get('subject_mass', 70.0)))
    bw = body_mass_guess * 9.81

    # 质量核对：scaling.py 会把各 Body 质量按 subject_mass 整体缩放，
    # 但它读的是 opensim_settings.subject_mass。若那一项缺失，它会静默
    # 跳过质量调整，模型就停在通用模板的质量上。这里直接把实际值打出来。
    try:
        _m = osim.Model(model_path)
        _s = _m.initSystem()
        _total = float(_m.getTotalMass(_s))
        print('\n[质量核对] 模型总质量 {:.2f} kg，config subject_mass {:.2f} kg，'
              '差 {:+.2f} kg（{:+.1f} N）。'.format(
                  _total, body_mass_guess, _total - body_mass_guess,
                  (_total - body_mass_guess) * 9.81))
        if abs(_total - body_mass_guess) > 0.5:
            beauty_print('[质量核对] 模型总质量与 subject_mass 不符，'
                         '说明缩放时没有做质量调整。请重跑 example_scaling.py，'
                         '并确认 subject_mass 写在 opensim_settings 下。',
                         type='warning')
    except Exception as _exc:
        beauty_print('[质量核对] 读取模型质量失败: {}'.format(_exc),
                     type='warning')
    print_residual_table(
        '[V1-before] 骨盆残差力（当前 .sto 剪切恒为 0）', before, bw)
    print('\n判读: FX 残差在数值上就是缺失的前后剪切。若它本来就 < 5% 体重，')
    print('      剪切假说当场被证伪，后面的重建就不必做了。')

    # --------------------------------------------------------
    #  重建 + V2/V3/V4
    # --------------------------------------------------------
    print('\n' + '=' * 80)
    print('[V2/V3/V4] 三路重建与一致性')
    print('=' * 80)
    print('{:<10}{:>12}{:>12}{:>12}{:>10}{:>10}{:>10}{:>10}'.format(
        'load', 'M1 rms', 'M2 rms', 'M3 rms', 'r(M1,M2)', 'r(M1,M3)',
        'r(M2,M3)', '剪切/竖直'))

    shear_by_key = {}
    sto_by_key = {}
    stats = {}
    methods_by_key = {}
    kin_by_key = {}

    for key in load_keys:
        ckey = _canon_load_key(key)
        mot_path = mot_by_key.get(ckey)
        file_info = modeling['data'].get(str(key), {})
        sto_path = os.path.join(get_ext_forces_dir(config, base_dir, key),
                                'bar_force_{}.sto'.format(key))
        if mot_path is None or not os.path.exists(sto_path):
            beauty_print('  {} 缺少 mot 或外力 .sto，跳过。'.format(key),
                         type='warning')
            continue

        sto_df = read_opensim_table(sto_path)
        if sto_df is None:
            beauty_print('  {} 的 .sto 不可读，跳过。'.format(key),
                         type='warning')
            continue
        sto_by_key[key] = sto_df

        kin = _cached_com_and_cop(model_path, mot_path, sto_path, sto_df,
                                  base_dir, config, key)
        kin_by_key[key] = kin
        body_mass = (BODY_MASS_OVERRIDE if BODY_MASS_OVERRIDE
                     else (kin['mass'] if kin else body_mass_guess))

        xsens_path = resolve_xsens_path(config, file_info)
        xacc = _cached_xsens_com(xsens_path, base_dir, config, key)

        methods = reconstruct_shear(kin, xacc, sto_df, body_mass, verbose=False)
        if not methods:
            beauty_print('  {} 三路全部失败，跳过。'.format(key), type='warning')
            continue

        t = sto_df['time'].values.astype(float)
        mask = window_mask_from_segment(t, seg_by_key.get(key))

        # 切片审计：window_mask_from_segment 在拿不到段时会静默退化为全 True。
        # 那样本组就变成“整条时间轴”统计，与其他组不可比，必须显式报出来。
        seg_now = seg_by_key.get(key)
        cov = float(np.mean(mask)) if len(mask) else 0.0
        if seg_now is None or len(seg_now) == 0:
            beauty_print('  [切片] {} 没有取到深蹲段，本组退化为整条时间轴统计，'
                         '包含站立、调整站位等非深蹲帧，'
                         '不能与其他组直接比较。'.format(key), type='warning')
        elif cov > 0.95:
            beauty_print('  [切片] {} 的深蹲窗口覆盖了 {:.0%} 的帧，'
                         '几乎等于没切。请核对段时间与 mot 时间轴是否同一时钟。'
                         .format(key, cov), type='warning')
        else:
            print('  [切片] {} 取用 {}/{} 帧（{:.0%}）。'.format(
                key, int(np.sum(mask)), len(mask), cov))

        def rms(name):
            m = methods.get(name)
            if m is None:
                return float('nan')
            v = m[mask, 0]
            v = v[np.isfinite(v)]
            return float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')

        def cc(a, b):
            ma, mb = methods.get(a), methods.get(b)
            if ma is None or mb is None:
                return None
            return _corr(ma[mask, 0], mb[mask, 0])

        fy = np.zeros(len(t))
        for c in ('grf_l_vy', 'grf_r_vy'):
            if c in sto_df.columns:
                fy = fy + sto_df[c].values.astype(float)
        fy_mean = float(np.mean(np.abs(fy[mask]))) if mask.any() else float('nan')

        methods_by_key[key] = (methods, mask)

        # ---- V0 竖直力账目 ----
        # 肌肉是内力，在全身自由体图里成对抵消，改变不了全身动量：
        #     ΣF_外 = m · a_质心
        # 所以竖直方向一定闭合，残差只可能来自账目错误或运动学噪声。
        grf_y_mean = float(np.mean(fy[mask])) if mask.any() else float('nan')
        bar_y = (sto_df['bar_force_vy'].values.astype(float)
                 if 'bar_force_vy' in sto_df.columns else np.zeros(len(t)))
        bar_y_mean = float(np.mean(bar_y[mask])) if mask.any() else float('nan')
        ay_mean = float('nan')
        if xacc is not None:
            _ay = np.interp(t, xacc['time'], xacc['acc'][:, 1])
            ay_mean = float(np.mean(_ay[mask])) if mask.any() else float('nan')

        # 准静态配准检查：慢速深蹲时质心应当基本落在压心正上方。
        # 两者的水平错位是 COP 配准误差的直接测量，不依赖任何动力学假设。
        cop_dx = cop_dz = float('nan')
        if kin is not None:
            km = window_mask_from_segment(kin['time'], seg_by_key.get(key))

            def _mid(axis):
                l = kin['cop_l'][:, axis]
                r = kin['cop_r'][:, axis]
                both = np.isfinite(l) & np.isfinite(r)
                m = np.where(both, 0.5 * (l + r),
                             np.where(np.isfinite(l), l, r))
                return m

            for axis, name in ((0, 'x'), (2, 'z')):
                d = (kin['com'][:, axis] - _mid(axis))[km]
                d = d[np.isfinite(d)]
                if len(d):
                    if name == 'x':
                        cop_dx = float(np.mean(d))
                    else:
                        cop_dz = float(np.mean(d))

        chosen = methods.get(SHEAR_METHOD)
        if chosen is None:
            for alt in ('segment_accel', 'com_accel', 'quasi_static'):
                if methods.get(alt) is not None:
                    beauty_print('  {} 没有 {} 结果，改用 {}。'.format(
                        key, SHEAR_METHOD, alt), type='warning')
                    chosen = methods[alt]
                    break
        shear_by_key[key] = chosen

        frac = (rms(SHEAR_METHOD) / fy_mean
                if fy_mean and np.isfinite(fy_mean) and fy_mean > 1 else float('nan'))

        def fmt(v):
            return 'N/A' if v is None or not np.isfinite(v) else '{:.3f}'.format(v)

        print('{:<10}{:>12.1f}{:>12.1f}{:>12.1f}{:>10}{:>10}{:>10}{:>9.1%}'.format(
            key, rms('com_accel'), rms('segment_accel'), rms('quasi_static'),
            fmt(cc('com_accel', 'segment_accel')),
            fmt(cc('com_accel', 'quasi_static')),
            fmt(cc('segment_accel', 'quasi_static')), frac))

        stats[key] = {'grf_y': grf_y_mean,
                      'bar_y': bar_y_mean,
                      'ay': ay_mean,
                      'mass': body_mass,
                      'frac': frac,
                      'r12': cc('com_accel', 'segment_accel'),
                      'rms': rms(SHEAR_METHOD),
                      'fy': fy_mean,
                      'cop_dx': cop_dx,
                      'cop_dz': cop_dz}

    # ---- V2 判定 ----
    bad_corr = [k for k, s in stats.items()
                if s['r12'] is not None and s['r12'] < CROSS_METHOD_MIN_CORR]
    if bad_corr:
        beauty_print('[V2 FAIL] 以下组 M1 与 M2 相关 < {:.2f}: {}。'
                     '两条独立路线不吻合，重建结果不可信，'
                     '先查滤波截止频率与 Xsens 时间轴。'.format(
                         CROSS_METHOD_MIN_CORR, bad_corr), type='warning')
    else:
        print('\n[V2 PASS] 所有组的 M1/M2 相关均达标。')

    # ---- V3 等长组零对照 ----
    for key, s in stats.items():
        if not str(key).upper().startswith('IM'):
            continue
        if np.isfinite(s['rms']) and s['rms'] > ISOMETRIC_SHEAR_WARN_N:
            beauty_print('[V3 FAIL] 等长组 {} 重建出 {:.1f} N 剪切，'
                         '超过阈值 {:.0f} N。杆不动时质心不应有水平加速度，'
                         '这更像微分噪声或基线漂移。'.format(
                             key, s['rms'], ISOMETRIC_SHEAR_WARN_N),
                         type='warning')
        else:
            print('[V3 PASS] 等长组 {} 剪切 {:.1f} N，接近零。'.format(
                key, s['rms']))

    # ---- V4 量级 ----
    lo, hi = SHEAR_FRAC_RANGE
    odd = [k for k, s in stats.items()
           if np.isfinite(s['frac']) and not (lo <= s['frac'] <= hi)
           and not str(k).upper().startswith('IM')]
    if odd:
        beauty_print('[V4 WARN] 以下组的 剪切/竖直 比值跑出 '
                     '[{:.0%}, {:.0%}]: {}。深蹲的剪切典型在 5-10%，'
                     '偏离过大要怀疑体重或坐标映射。'.format(lo, hi, odd),
                     type='warning')
    else:
        print('[V4 PASS] 剪切/竖直 比值均在合理区间内。')

    # --------------------------------------------------------
    #  V1b：以 FX 残差为真值，直接回归三路重建
    # --------------------------------------------------------
    # 这一步比“补进去看 rms 有没有降”强得多：残差是逐帧的真值序列，
    # 相关系数低就说明重建的是噪声，此时补进去必然让 rms 上升
    # （两个不相关信号的和，rms 按平方和相加）。
    print('\n' + '=' * 80)
    print('[V1b] 以 FX 残差为真值回归三路重建（相关应高，k 应接近 1）')
    print('=' * 80)
    print('{:<10}{:>13}{:>10}{:>10}{:>10}{:>12}'.format(
        'load', 'FX res rms', 'r(M1)', 'r(M2)', 'r(M3)', 'k(选中法)'))
    for key in load_keys:
        res = before.get(key)
        pack = methods_by_key.get(key)
        if not res or pack is None or '_tx_series' not in res:
            continue
        methods, mask = pack
        t = sto_by_key[key]['time'].values.astype(float)
        target = np.interp(t, res['_tx_time'], res['_tx_series'])[mask]
        cells = []
        for name in ('com_accel', 'segment_accel', 'quasi_static'):
            m = methods.get(name)
            r = None if m is None else _corr(target, m[mask, 0])
            cells.append('N/A' if r is None else '{:+.3f}'.format(r))
        k = float('nan')
        m = methods.get(SHEAR_METHOD)
        if m is not None:
            e = m[mask, 0]
            ok = np.isfinite(e) & np.isfinite(target)
            if ok.sum() > 10 and float(np.sum(e[ok] ** 2)) > 1e-9:
                k = float(np.sum(target[ok] * e[ok]) / np.sum(e[ok] ** 2))
        print('{:<10}{:>13.1f}{:>10}{:>10}{:>10}{:>12}'.format(
            key, res['tx']['rms'], cells[0], cells[1], cells[2],
            'N/A' if not np.isfinite(k) else '{:+.2f}'.format(k)))

    # --------------------------------------------------------
    #  COP 配准审计
    # --------------------------------------------------------
    print('\n' + '=' * 80)
    print('[V0] 竖直力账目：GRF_y + bar_y - m*g - m*a_y 应为 0')
    print('     肌肉是内力，不进这个方程，所以竖直方向必须闭合。')
    print('=' * 80)
    print('{:<10}{:>12}{:>12}{:>12}{:>12}{:>12}'.format(
        'load', 'GRF_y', 'bar_y', 'm*g', 'm*a_y', '差额(N)'))
    for key in load_keys:
        s = stats.get(key)
        if not s or not np.isfinite(s.get('grf_y', float('nan'))):
            continue
        mg = s['mass'] * 9.81
        may = (s['mass'] * s['ay']
               if np.isfinite(s.get('ay', float('nan'))) else 0.0)
        gap = s['grf_y'] + s['bar_y'] - mg - may
        print('{:<10}{:>12.1f}{:>12.1f}{:>12.1f}{:>12.1f}{:>12.1f}'.format(
            key, s['grf_y'], s['bar_y'], mg, may, gap))
        if abs(gap) > 0.05 * bw:
            beauty_print('  [V0] {} 的竖直力差额 {:+.0f} N（体重的 {:.0%}）。'
                         '这一项与 ID 的 FY 残差同源，应先把它做到接近 0，'
                         '再讨论水平剪切。'.format(key, gap, abs(gap) / bw),
                         type='warning')

    # ---- V0-gain：把九组的竖直差额放在一起，区分“加性漏项”与“乘性欠读” ----
    # 静态（m*a_y 实测 < 2 N，可忽略）下必然有 GRF_y = m*g + |bar_y|。
    # 把实测 GRF_y 对这个应有值回归：
    #   斜率 k 明显 < 1 而截距近 0 -> 鞋垫按比例欠读（标定增益 / 未覆盖面积 / 饱和）
    #   斜率近 1 而截距显著非 0    -> 少记了一项恒定的力（例如杆重 m_bar*g）
    # 这两种病因的修法完全不同，靠单组数据分不开，必须跨组看。
    req, obs = [], []
    for key in load_keys:
        s = stats.get(key)
        if not s or not np.isfinite(s.get('grf_y', float('nan'))):
            continue
        req.append(s['mass'] * 9.81 - s['bar_y'])
        obs.append(s['grf_y'])
    if len(req) >= 3:
        req = np.asarray(req, dtype=float)
        obs = np.asarray(obs, dtype=float)
        k_org = float(np.sum(req * obs) / np.sum(req ** 2))
        A = np.column_stack([req, np.ones(len(req))])
        k_int, c_int = np.linalg.lstsq(A, obs, rcond=None)[0]
        rms_org = float(np.sqrt(np.mean((obs - k_org * req) ** 2)))
        rms_int = float(np.sqrt(np.mean((obs - (k_int * req + c_int)) ** 2)))
        off = obs - req
        rms_off = float(np.sqrt(np.mean((off - np.mean(off)) ** 2)))
        print('\n[V0-gain] 以 GRF_y = k*(m*g + |bar_y|) + c 拟合 {} 组：'
              .format(len(req)))
        print('  纯增益      k = {:.4f}                残差 rms {:.1f} N'
              .format(k_org, rms_org))
        print('  增益+截距   k = {:.4f}, c = {:+.1f} N   残差 rms {:.1f} N'
              .format(k_int, c_int, rms_int))
        print('  纯加性偏置（k 固定为 1）              残差 rms {:.1f} N'
              .format(rms_off))
        if rms_off > 1.5 * rms_org and abs(1.0 - k_org) > 0.05:
            beauty_print('[V0-gain] 乘性模型明显优于加性模型：k = {:.3f}，'
                         '说明鞋垫竖直力系统性欠读约 {:.0%}。'
                         '杆重、模型质量这类加性候选可以排除，'
                         '应去查鞋垫标定增益、足底未覆盖区域与右垫饱和。'
                         .format(k_org, 1.0 - k_org), type='warning')
        elif abs(c_int) > 0.05 * bw and abs(1.0 - k_int) < 0.05:
            beauty_print('[V0-gain] 斜率接近 1 而截距达 {:+.0f} N，'
                         '更像漏记了一项恒定的力，'
                         '优先查杆重 m_bar*g 与外力施加的 body。'
                         .format(c_int), type='warning')

    # ---- CAL-gain：用静止站立段单独标定鞋垫增益 ----
    # 与 V0-gain 互为独立验证：那一路是九组之间的斜率（相对量），
    # 这一路是单组内的绝对值。两者吻合才敢动 external_forces。
    _cal_keys = [k for k in load_keys
                 if (not QUIET_CAL_KEYS) or (str(k) in QUIET_CAL_KEYS)]
    k_cal = {}
    for key in _cal_keys:
        if key not in sto_by_key:
            continue
        _kk = calibrate_insole_gain(
            key, sto_by_key[key], kin_by_key.get(key),
            mot_by_key.get(_canon_load_key(key)),
            stats.get(key, {}).get('mass', body_mass_guess),
            seg=seg_by_key.get(key))
        if _kk is not None and np.isfinite(_kk):
            k_cal[key] = float(_kk)
    if k_cal:
        _ks = np.array(list(k_cal.values()), dtype=float)
        _kmean = float(np.mean(_ks))
        print('\n[CAL-gain] 汇总：{} 组标定出 k = {}，均值 {:.4f}，'
              '修正系数 1/k = {:.4f}。'.format(
                  len(_ks),
                  ', '.join('{:.4f}'.format(v) for v in _ks),
                  _kmean, 1.0 / _kmean if abs(_kmean) > 1e-6 else float('nan')))
        _kref = locals().get('k_org', float('nan'))
        if np.isfinite(_kref):
            print('           跨组回归 V0-gain 给出 k = {:.4f}，'
                  '静止标定给出 k = {:.4f}，两者相差 {:+.4f}。'.format(
                      _kref, _kmean, _kmean - _kref))
            if abs(_kmean - _kref) < 0.03:
                print('           两条独立路线吻合到 0.03 以内，'
                     '可以认定鞋垫是乘性欠读，直接按 1/k 修正鞋垫竖直力。')
            else:
                beauty_print('[CAL-gain] 两条路线相差 {:.3f}，超过 0.03。'
                             '静止段与深蹲段的增益不一致，'
                             '说明欠读不是单纯的标定系数问题，'
                             '更像与压力大小相关（未覆盖区随负载变化、'
                             '或右垫在大负载下饱和）。'
                             '这种情况不能用一个常数 k 修。'
                             .format(abs(_kmean - _kref)), type='warning')

    print('\n{:<10}{:>16}{:>16}{:>18}'.format(
        'load', 'COM-COP dx(cm)', 'COM-COP dz(cm)', '膝力矩影响(N·m)'))
    for key in load_keys:
        s = stats.get(key)
        if not s or not np.isfinite(s.get('cop_dx', float('nan'))):
            continue
        print('{:<10}{:>16.2f}{:>16.2f}{:>18.1f}'.format(
            key, 100.0 * s['cop_dx'], 100.0 * s['cop_dz'],
            abs(s['cop_dx']) * KNEE_COP_SENSITIVITY))
        if abs(s['cop_dx']) > COP_OFFSET_WARN_M:
            beauty_print('  [COP] {} 的质心与压心前后错位 {:.1f} cm。'
                         '慢速深蹲近似准静态，两者本应基本重合；'
                         '这个量级更像 COP 配准偏差，'
                         '而 insole_heel_offset_x 目前还是 0。'.format(
                             key, 100.0 * s['cop_dx']), type='warning')

    recommend_heel_offset(
        stats, float(_osim_cfg.get('insole_heel_offset_x', 0.0)))

    if not WRITE_PATCHED_STO:
        print('\nWRITE_PATCHED_STO=False，到此为止（仅离线量级估算）。')
        return

    # --------------------------------------------------------
    #  写回 .sto 并重跑 ID
    # --------------------------------------------------------
    print('\n' + '=' * 80)
    print('[写入] 生成带剪切的外力文件（法 = {}）'.format(SHEAR_METHOD))
    print('=' * 80)
    xml_by_key = {}
    for key in load_keys:
        if key not in shear_by_key or shear_by_key[key] is None:
            continue
        xml_by_key[key] = write_patched_external_loads(
            config, base_dir, key, sto_by_key[key], shear_by_key[key])

    if not RUN_ID_WITH_SHEAR:
        print('\nRUN_ID_WITH_SHEAR=False，未重跑 ID。')
        return

    print('\n' + '=' * 80)
    print('[重跑] 带剪切的逆动力学')
    print('=' * 80)
    for key, xml in xml_by_key.items():
        if xml is None:
            continue
        mot_path = mot_by_key.get(_canon_load_key(key))
        run_inverse_dynamics(
            model_path=model_path, mot_path=mot_path,
            output_dir=get_shear_id_dir(config, base_dir, key),
            external_load_file=xml,
            label='{}_{}_{}'.format(config['experiment_label'], key, SHEAR_TAG),
            verbose=False)

    # --------------------------------------------------------
    #  V1-after + V5
    # --------------------------------------------------------
    after = {}
    for key in xml_by_key:
        path = os.path.join(
            get_shear_id_dir(config, base_dir, key),
            '{}_{}_{}_InverseDynamics.sto'.format(
                config['experiment_label'], key, SHEAR_TAG))
        if not os.path.exists(path):
            # 不同 OpenSim 版本的命名略有差异，回退到目录里找。
            d = get_shear_id_dir(config, base_dir, key)
            cands = ([f for f in os.listdir(d) if f.endswith('.sto')]
                     if os.path.isdir(d) else [])
            if not cands:
                continue
            path = os.path.join(d, cands[0])
        after[key] = read_residuals(path, seg=seg_by_key.get(key))

    print_residual_table('[V1-after] 骨盆残差力（已补剪切）', after, bw)

    print('\n{:<10}{:>14}{:>14}{:>12}'.format(
        'load', 'FX before', 'FX after', '降幅'))
    for key in after:
        b = (before.get(key) or {}).get('tx', {}).get('rms')
        a = (after.get(key) or {}).get('tx', {}).get('rms')
        if b is None or a is None:
            continue
        drop = 1.0 - a / b if b > 1e-9 else float('nan')
        print('{:<10}{:>14.1f}{:>14.1f}{:>11.1%}'.format(key, b, a, drop))
        if a > b:
            beauty_print('  [V1 FAIL] {} 的残差反而变大了。剪切符号或坐标映射'
                         '可能搞反了，先把 x 轴方向对一遍。'.format(key),
                         type='warning')

    # ---- V5 单调性 ----
    print('\n' + '=' * 80)
    print('[V5] 膝力矩单调性（横轴用组均竖直 GRF，对九组都成立）')
    print('=' * 80)
    print('{:<10}{:>14}{:>16}{:>16}'.format(
        'load', 'GRF均值(N)', '膝|M| before', '膝|M| after'))

    xs, ys_b, ys_a = [], [], []
    for key in load_keys:
        seg = seg_by_key.get(key)
        fy = stats.get(key, {}).get('fy')
        if fy is None or not np.isfinite(fy):
            continue

        def knee_abs(id_path):
            df = read_opensim_table(id_path)
            if df is None or 'time' not in df.columns:
                return None
            colname = find_id_moment_column(df, 'knee_angle_l')
            if colname is None:
                return None
            t = df['time'].values.astype(float)
            m = window_mask_from_segment(t, seg)
            v = np.abs(df[colname].values.astype(float)[m])
            v = v[np.isfinite(v)]
            return float(np.mean(v)) if len(v) else None

        mb = knee_abs(get_inverse_dynamics_path(config, base_dir, key))
        ma = None
        d = get_shear_id_dir(config, base_dir, key)
        if os.path.isdir(d):
            cands = [f for f in os.listdir(d) if f.endswith('.sto')]
            if cands:
                ma = knee_abs(os.path.join(d, cands[0]))
        if mb is None:
            continue
        print('{:<10}{:>14.1f}{:>16.2f}{:>16}'.format(
            key, fy, mb, 'N/A' if ma is None else '{:.2f}'.format(ma)))
        xs.append(fy)
        ys_b.append(mb)
        if ma is not None:
            ys_a.append(ma)

    if len(xs) >= 3:
        rb = _spearman(xs, ys_b)
        print('\n膝力矩 vs GRF 的 Spearman  before: {}'.format(
            'N/A' if rb is None else '{:+.3f}'.format(rb)))
        if len(ys_a) == len(xs):
            ra = _spearman(xs, ys_a)
            print('膝力矩 vs GRF 的 Spearman  after : {}'.format(
                'N/A' if ra is None else '{:+.3f}'.format(ra)))
            if ra is not None and rb is not None and ra <= rb:
                beauty_print('[V5 FAIL] 补上剪切后单调性没有改善。'
                             '如果此时 V1 的残差确实下降了，说明剪切是对的，'
                             '但膝力矩的主因在别处（优先查 heel offset 标定）。',
                             type='warning')
            else:
                print('[V5 PASS] 单调性有改善。')

    print('\n完成。原有结果未被修改；带剪切的结果在 '
          'inverse_dynamics_{}/ 下。'.format(SHEAR_TAG))


if __name__ == '__main__':
    main()