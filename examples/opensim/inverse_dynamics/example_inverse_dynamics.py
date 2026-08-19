"""
example_inverse_dynamics_diagnostics.py

用于排查 inverse_dynamics 输出是否随深蹲负载合理变化。

流程：
  1. 对指定 load 运行 Step 3 InverseDynamics；
  2. 立即复用 example_data_analysis.py / MultiLoadPipeline 的标准运动切片；
  3. 只取标准 upward 阶段；
  4. 将 inverse_dynamics.sto 中的关节力矩按 time 插值到 upward 切片时间点；
  5. 打印每个 load、每个左腿关节的：
       - signed mean：有符号平均力矩
       - mean abs：平均绝对力矩

为什么同时打印 mean 和 mean abs？
  - signed mean 可能因为正负号或相位混合而相互抵消；
  - mean abs 更适合检查“负载增加时力矩绝对值是否上升”。

内置诊断（排查力矩非单调时按顺序看）：
  [诊断 1] 时间轴覆盖率：切片 time 是否落在 inverse_dynamics time 范围内。
           覆盖率 < 100% 时 np.interp 会用端点常值静默填充，结果不可信。
           同时打印 n_seg / n_cycle / 时长，判断各 load 样本量是否可比。
  [诊断 2] 相位一致性：机器人时钟与 Xsens/mot 时钟是否同零点。
           物理约束是“深蹲最低点(pos_l 最小) == 膝屈最大(knee_angle_l 最大)”，
           因此把 -pos_l 与 knee_angle_l 做互相关，最优滞后应接近 0。
  [诊断 3] 单调性报告：按负载数值排序，列出违反单调的具体位置与 Spearman 秩相关。

丢帧处理原则：
  Xsens 丢帧后被保持/线性插值的区间（图上呈一条直线）里，角速度与角加速度
  不可信，ID 的惯性项会错。这些区间由 mot_pipeline.detect_frozen_intervals()
  在 Step1 自动检测并写成 .dropouts.csv；本脚本只剔除落在其中的帧/段，
  同一 load 的其余循环照常参与统计——不丢掉整组数据。
  欧拉角绕接（wrap）翻转同样在 Step1 用 unwrap_degrees() 修正，而不在这里排除负载。
"""
import os
import sys
import json

import numpy as np
import pandas as pd

# 让脚本从 examples/... 目录直接运行时也能找到项目根下的 digitaltwin 包
_BASE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from digitaltwin.osim.inverse_dynamics import run_step3_inverse_dynamics
from digitaltwin.osim.mot_pipeline import (
    get_mot_files,
    load_dropout_intervals,
    in_intervals,
)
# 编排层（跑流水线 / 带缓存的切片装载 / 动作窗口）
from digitaltwin.pipelines.standard_analysis import (
    load_or_create_cutted_pipeline_results,
)
# 纯分析层
from digitaltwin.analysis.result_analysis import (
    build_left_joint_coordinate_map,
    build_bilateral_joint_coordinate_map,
    summarize_inverse_dynamics_moments,
    print_summary_table,
    get_load_keys,
    get_inverse_dynamics_path,
    get_segment_from_results,
    read_opensim_table,
    interpolate_column_to_segment,
    find_id_moment_column,
)


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../../config/20250409_squat_NCMP001_xsens.json'

# None = 全部；也可以指定，如 ['20', '38', '56']
LOAD_KEYS = None

# 排除非负载试验（例如 MVC / 空杆 / 标定 trial）。
EXCLUDE_LOAD_KEYS = None

# 诊断开关
RUN_DIAGNOSTICS = True
# 相位检查参考量：Xsens 侧关节角 + 机器人侧位置信号
PHASE_XSENS_COORD = 'knee_angle_l'
PHASE_ROBOT_SIGNAL = 'pos_l'
# 互相关搜索的最大滞后（秒）与时间分辨率（秒）
PHASE_MAX_LAG = 5.0
PHASE_DT = 0.01
# 滞后容差（秒）。Xsens 60Hz 下单帧 = 16.7ms，互相关在平滑准正弦信号上的
# 峰值很宽，因此 0.05s 量级的滞后属于估计噪声，不构成真正的不同步。
# 取 0.15s（约 9 帧，且远小于一个 upward 阶段的 10%）作为实际容差。
PHASE_LAG_TOLERANCE = 0.15

# 丢帧区间处理：剔除落在丢帧区间（±margin 秒）内的帧，而不丢整组数据。
# margin 留余量是因为 ID 的惯性项依赖二阶导数，区间边缘上一两帧也不可信。
DROP_FROZEN_FRAMES = True
DROPOUT_MARGIN = 0.10                       # 秒

# 姿态策略诊断要看的坐标
POSTURE_COORDS = ('pelvis_tilt', 'lumbar_extension', 'hip_flexion_l',
                  'knee_angle_l', 'ankle_angle_l')

# 匹配膝角对比：在每个 upward 段中取膝角最接近该值的时刻（单位：度）
MATCHED_KNEE_ANGLES = (50.0, 70.0, 90.0)
RUN_MATCHED_ANGLE = True

# None = 从 opensim_settings.muscle_analysis_coordinates 中自动取所有左腿关节
# 也可以指定，如 ['hip_flexion', 'knee_angle', 'ankle_angle']
JOINT_BASES_TO_PRINT = None

# 只统计标准 upward 阶段；也可改为 ('downward',) 或 ('upward', 'downward')
MOVEMENT_TYPES = ('upward',)

# 切片缓存设置：
#   False = 优先读取 result_folder/cutted_data.csv；
#           如果没有，则尝试用 aligned_data.csv 快速重新切片；
#           如果 aligned_data.csv 也没有，才运行完整 MultiLoadPipeline。
#   True  = 强制重新生成 cutted_data.csv。
FORCE_REBUILD_CUTTED_CACHE = False
CUTTED_CACHE_NAME = 'cutted_data.csv'

# ID 外力设置
USE_EXTERNAL_FORCES = True
MB = 0.0
OUTPUT_BODY_FORCES = False


# ============================================================
#  路径
# ============================================================

def get_base_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


# ============================================================
#  诊断工具
# ============================================================

def _canon_load_key(value):
    """统一 load key 字符串格式（与 result_analysis 内部规则保持一致）。"""
    try:
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f'{f:g}'
    except Exception:
        return str(value)


def diagnose_time_axis_coverage(config, base_dir, pipeline_results,
                                load_keys, movement_types):
    """
    [诊断 1] 切片时间点是否落在 inverse_dynamics 的时间范围内。

    覆盖率 < 100% 说明 np.interp 用端点常值填充了越界样本，该 load 不可信。
    n_seg / n_cycle 差异过大说明各 load 有效重复次数不同，均值不可直接比较。
    """
    print('\n' + '=' * 80)
    print('[诊断 1] 时间轴覆盖率')
    print('=' * 80)
    print(f'{"load":<8}{"seg_t0":>9}{"seg_t1":>9}{"id_t0":>9}{"id_t1":>9}'
          f'{"覆盖率":>9}{"n_seg":>7}{"n_cycle":>8}{"seg时长":>9}{"丢帧帧比":>10}')

    mot_by_key = {
        _canon_load_key(k): v
        for k, v in get_mot_files(config, base_dir).items()
    }
    report = {}
    for load_key in load_keys:
        segment_df = get_segment_from_results(
            pipeline_results, load_key, movement_types=movement_types)
        id_df = read_opensim_table(
            get_inverse_dynamics_path(config, base_dir, load_key))

        if segment_df is None or id_df is None or 'time' not in id_df.columns:
            print(f'{load_key:<8}  数据缺失（切片或 ID 文件不可读）')
            report[load_key] = None
            continue

        seg_t = segment_df['time'].values.astype(float)
        id_t = id_df['time'].values.astype(float)
        inside = int(np.sum((seg_t >= id_t.min()) & (seg_t <= id_t.max())))
        coverage = 100.0 * inside / len(seg_t)

        if 'cycle_id' in segment_df.columns:
            n_cycle = int(segment_df['cycle_id'].nunique())
        elif 'segment_id' in segment_df.columns:
            n_cycle = int(segment_df['segment_id'].nunique())
        else:
            n_cycle = -1

        duration = float(seg_t.max() - seg_t.min())

        # 丢帧占比：切片时间点中有多少落在丢帧/插值区间内
        drop_txt = f'{"N/A":>10}'
        mot_path = mot_by_key.get(_canon_load_key(load_key))
        if mot_path:
            mot_df = read_opensim_table(mot_path)
            intervals = load_dropout_intervals(mot_path, mot_df=mot_df)
            mask = in_intervals(seg_t, intervals, DROPOUT_MARGIN)
            drop_txt = f'{100.0 * float(mask.mean()):>9.1f}%'

        print(f'{load_key:<8}{seg_t.min():>9.3f}{seg_t.max():>9.3f}'
              f'{id_t.min():>9.3f}{id_t.max():>9.3f}'
              f'{coverage:>8.1f}%{len(seg_t):>7d}{n_cycle:>8d}{duration:>9.2f}'
              f'{drop_txt}')
        report[load_key] = coverage

    bad = [k for k, v in report.items() if v is not None and v < 99.999]
    if bad:
        print(f'\n[FAIL] 以下 load 覆盖率不足 100%，其统计值被端点常值污染: {bad}')
        print('       常见原因: mot 文件被截断 / 用错 xsens_file / Step1 未重跑。')
    else:
        print('\n[PASS] 所有 load 覆盖率均为 100%（仅说明区间重叠，不代表相位对齐）。')
    return report


def _load_aligned_data_by_load(subject, verbose=True):
    """读取 result_folder/aligned_data.csv，并按 load 分组返回。"""
    path = os.path.join(subject.result_folder, 'aligned_data.csv')
    if not os.path.exists(path):
        if verbose:
            print(f'[诊断 2] 未找到 {path}，跳过相位检查。')
        return None

    df = pd.read_csv(path)
    load_col = None
    for c in ('load_weight', 'load', 'load_value'):
        if c in df.columns:
            load_col = c
            break
    if load_col is None:
        return {'all': df}

    return {
        _canon_load_key(k): g.reset_index(drop=True)
        for k, g in df.groupby(load_col)
    }


def estimate_time_lag(t_ref, v_ref, t_test, v_test,
                      dt=0.01, max_lag=5.0, lag_step=0.02):
    """
    用互相关估计 v_test 相对 v_ref 的时间滞后。

    定义: 使 v_test(t + lag) 与 v_ref(t) 最相关的 lag。
    lag > 0 表示 test 信号事件发生得比 ref 更晚。

    Returns
    -------
    (best_lag, best_corr, zero_lag_corr)  任一项无法计算时返回 None
    """
    t_ref = np.asarray(t_ref, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)
    t_test = np.asarray(t_test, dtype=float)
    v_test = np.asarray(v_test, dtype=float)

    ok_ref = np.isfinite(t_ref) & np.isfinite(v_ref)
    ok_test = np.isfinite(t_test) & np.isfinite(v_test)
    if ok_ref.sum() < 20 or ok_test.sum() < 20:
        return None, None, None

    t_ref, v_ref = t_ref[ok_ref], v_ref[ok_ref]
    t_test, v_test = t_test[ok_test], v_test[ok_test]

    order = np.argsort(t_ref)
    t_ref, v_ref = t_ref[order], v_ref[order]
    order = np.argsort(t_test)
    t_test, v_test = t_test[order], v_test[order]

    grid = np.arange(t_ref.min(), t_ref.max(), dt)
    if len(grid) < 50:
        return None, None, None
    ref = np.interp(grid, t_ref, v_ref)
    if np.std(ref) < 1e-12:
        return None, None, None

    min_overlap = max(50, int(0.5 * len(grid)))
    best_lag, best_corr, zero_corr = None, -2.0, None

    for lag in np.arange(-max_lag, max_lag + lag_step, lag_step):
        probe = grid + lag
        mask = (probe >= t_test.min()) & (probe <= t_test.max())
        if int(mask.sum()) < min_overlap:
            continue
        a = ref[mask]
        b = np.interp(probe[mask], t_test, v_test)
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if abs(lag) < lag_step / 2.0:
            zero_corr = r
        if r > best_corr:
            best_corr, best_lag = r, float(lag)

    if best_lag is None:
        return None, None, None
    return best_lag, best_corr, zero_corr


def diagnose_phase_alignment(subject, config, base_dir, load_keys,
                             xsens_coord=PHASE_XSENS_COORD,
                             robot_signal=PHASE_ROBOT_SIGNAL,
                             max_lag=PHASE_MAX_LAG, dt=PHASE_DT):
    """
    [诊断 2] 机器人时钟与 Xsens/mot 时钟的相位一致性。

    深蹲最低点（robot pos_l 最小）必须与膝屈最大（knee_angle_l 最大）同时发生，
    因此 -pos_l 与 knee_angle_l 的互相关最优滞后应接近 0。
    若最优滞后是一个稳定的非零常数，则存在固定时间偏移，需要在插值前校正。
    """
    aligned = _load_aligned_data_by_load(subject)
    if aligned is None:
        return None

    mot_by_key = {
        _canon_load_key(k): v
        for k, v in get_mot_files(config, base_dir).items()
    }

    print('\n' + '=' * 80)
    print('[诊断 2] 相位一致性（机器人时钟 vs Xsens/mot 时钟）')
    print('=' * 80)
    print(f'参考信号: robot -{robot_signal}   vs   mot {xsens_coord}')
    print(f'{"load":<8}{"最优滞后(s)":>13}{"最优相关":>10}'
          f'{"零滞后相关":>12}{"robot最低点":>13}{"xsens最大屈":>13}{"判读":>10}')

    report = {}
    for load_key in load_keys:
        key = _canon_load_key(load_key)
        robot_df = aligned.get(key, aligned.get('all'))
        mot_path = mot_by_key.get(key)

        if robot_df is None or mot_path is None:
            print(f'{load_key:<8}  缺少 aligned_data 或 mot 文件，跳过')
            report[load_key] = None
            continue
        if robot_signal not in robot_df.columns or 'time' not in robot_df.columns:
            print(f'{load_key:<8}  aligned_data 缺少 {robot_signal}/time 列，跳过')
            report[load_key] = None
            continue

        mot_df = read_opensim_table(mot_path)
        if mot_df is None or xsens_coord not in mot_df.columns:
            print(f'{load_key:<8}  mot 文件缺少 {xsens_coord} 列，跳过')
            report[load_key] = None
            continue

        t_robot = robot_df['time'].values.astype(float)
        v_robot = -robot_df[robot_signal].values.astype(float)   # 低位 -> 大值
        t_mot = mot_df['time'].values.astype(float)
        v_mot = mot_df[xsens_coord].values.astype(float)

        lag, corr, zero_corr = estimate_time_lag(
            t_robot, v_robot, t_mot, v_mot, dt=dt, max_lag=max_lag)

        t_robot_bottom = float(t_robot[np.nanargmax(v_robot)])
        t_mot_bottom = float(t_mot[np.nanargmax(v_mot)])

        if lag is None:
            verdict = '无法计算'
            lag_s, corr_s, zero_s = 'N/A', 'N/A', 'N/A'
        else:
            lag_s = f'{lag:+.3f}'
            corr_s = f'{corr:.3f}'
            zero_s = 'N/A' if zero_corr is None else f'{zero_corr:.3f}'
            if corr < 0.5:
                verdict = '? 相关低'
            elif abs(lag) <= PHASE_LAG_TOLERANCE:
                verdict = '✓ 对齐'
            else:
                verdict = '✗ 有偏移'

        print(f'{load_key:<8}{lag_s:>13}{corr_s:>10}{zero_s:>12}'
              f'{t_robot_bottom:>13.3f}{t_mot_bottom:>13.3f}{verdict:>10}')
        report[load_key] = {'lag': lag, 'corr': corr, 'zero_corr': zero_corr}

    lags = [v['lag'] for v in report.values()
            if v and v['lag'] is not None and v['corr'] is not None and v['corr'] >= 0.5]
    if lags:
        lags = np.asarray(lags, dtype=float)
        print(f'\n滞后统计: mean={lags.mean():+.3f}s  std={lags.std():.3f}s  '
              f'范围=[{lags.min():+.3f}, {lags.max():+.3f}]s')
        if np.abs(lags).max() <= PHASE_LAG_TOLERANCE:
            print(f'[PASS] 滞后均在容差 {PHASE_LAG_TOLERANCE:.2f}s 内（约 '
                  f'{PHASE_LAG_TOLERANCE * 60:.0f} 帧 @60Hz），且零滞后相关与最优相关几乎相等，')
            print('       属于互相关估计噪声而非真正不同步；假设 3 可排除。')
        elif lags.std() <= 0.05:
            print('[FAIL] 存在稳定的固定偏移 Δ；修法: 插值前对 segment_df["time"] 统一加 Δ。')
        else:
            print('[FAIL] 滞后在各 load 间抖动，说明每次试验同步不可靠；')
            print('       应改用特征点（最低点/过零点）对齐，而非绝对时间对齐。')
    if any(v and v['corr'] is not None and v['corr'] < 0.5 for v in report.values()):
        print('[WARN] 某些 load 相关性 < 0.5：可能 mot 与 robot 记录的不是同一次试验。')
    return report


def _spearman(x, y):
    """无 scipy 依赖的 Spearman 秩相关。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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


def report_monotonicity(title, summary, load_keys):
    """
    [诊断 3] 按负载数值排序，检查统计量是否随负载单调，并定位违反处。
    """
    numeric = []
    for k in load_keys:
        try:
            numeric.append((float(k), str(k)))
        except (TypeError, ValueError):
            continue
    numeric.sort()
    if len(numeric) < 3:
        return

    print('\n' + '=' * 80)
    print(f'[诊断 3] 单调性报告 — {title}')
    print('=' * 80)
    print(f'负载顺序: {[k for _, k in numeric]}')
    print(f'{"joint":<20}{"Spearman":>10}{"违反次数":>10}   违反位置')

    for joint_base, load_values in summary.items():
        xs, ys, keys = [], [], []
        for val, key in numeric:
            v = load_values.get(key)
            if v is None or not np.isfinite(v):
                continue
            xs.append(val)
            ys.append(abs(float(v)))
            keys.append(key)

        if len(ys) < 3:
            print(f'{joint_base:<20}{"N/A":>10}{"N/A":>10}   样本不足')
            continue

        violations = [
            f'{keys[i]}->{keys[i + 1]}({ys[i + 1] - ys[i]:+.2f})'
            for i in range(len(ys) - 1) if ys[i + 1] < ys[i]
        ]
        rho = _spearman(xs, ys)
        rho_s = 'N/A' if rho is None else f'{rho:+.3f}'
        print(f'{joint_base:<20}{rho_s:>10}{len(violations):>10}   '
              f'{", ".join(violations) if violations else "—"}')

    print('\n判读: 使用绝对值排序；Spearman 接近 +1 表示单调上升。')
    print('      若仅个别相邻负载轻微反转，优先怀疑样本量/相位混合（诊断 1）；')
    print('      若整体 Spearman 接近 0 或为负，则问题在外力或运动学层面。')


def diagnose_posture_strategy(config, base_dir, pipeline_results, load_keys,
                              movement_types, coords=POSTURE_COORDS):
    """
    [诊断 4] 各 load 的 upward 姿态是否可比。

    膝力矩强烈依赖膝角与躯干前倾：负载增大时受试往往自发转为
    “高位主导（hip-dominant）”策略，即前倾变大、深蹲变浅，
    从而把力矩从膝转移到髓。这会造成“髓/趾单调、膝不单调”的真实生理现象，
    并非程序 bug。本诊断用于区分两者。
    """
    mot_by_key = {
        _canon_load_key(k): v
        for k, v in get_mot_files(config, base_dir).items()
    }

    print('\n' + '=' * 80)
    print('[诊断 4] upward 阶段姿态策略（单位：度）')
    print('=' * 80)
    header = f'{"load":<8}' + ''.join(f'{c:>18}' for c in coords) + f'{"峰值|膝角|":>12}'
    print(header)
    print('-' * len(header))

    for load_key in load_keys:
        key = _canon_load_key(load_key)
        segment_df = get_segment_from_results(
            pipeline_results, load_key, movement_types=movement_types)
        mot_path = mot_by_key.get(key)
        if segment_df is None or mot_path is None:
            print(f'{load_key:<8}  缺少切片或 mot 文件，跳过')
            continue

        mot_df = read_opensim_table(mot_path)
        if mot_df is None:
            print(f'{load_key:<8}  mot 不可读，跳过')
            continue

        # 丢帧区间内的运动学不可信，置为 NaN 后再统计（不丢整组数据）
        bad = np.zeros(len(segment_df), dtype=bool)
        if DROP_FROZEN_FRAMES and 'time' in segment_df.columns:
            intervals = load_dropout_intervals(mot_path, mot_df=mot_df)
            if intervals:
                bad = in_intervals(segment_df['time'].values, intervals,
                                   DROPOUT_MARGIN)

        row = f'{load_key:<8}'
        knee_peak = None
        for coord in coords:
            vals = interpolate_column_to_segment(mot_df, segment_df, coord)
            if vals is None:
                row += f'{"N/A":>18}'
                continue
            vals = np.asarray(vals, dtype=float).copy()
            vals[bad] = np.nan
            if not np.any(np.isfinite(vals)):
                row += f'{"N/A":>18}'
                continue
            row += f'{np.nanmean(vals):>18.2f}'
            if coord == 'knee_angle_l':
                knee_peak = np.nanmax(np.abs(vals))
        row += f'{knee_peak:>12.2f}' if knee_peak is not None else f'{"N/A":>12}'
        print(row)

    print('\n判读: 若 pelvis_tilt / lumbar_extension 随负载单调变大（前倾增加），')
    print('      或峰值|膝角|随负载下降（蹲得更浅），则膝力矩非单调是真实策略变化，')
    print('      必须改用“匹配膝角”比较（诊断 5）而非整段均值。')


def summarize_moments_at_matched_knee_angle(config, base_dir, pipeline_results,
                                            load_keys, coord_map,
                                            movement_types,
                                            target_angles=MATCHED_KNEE_ANGLES,
                                            knee_coord='knee_angle_l'):
    """
    [诊断 5] 在固定膝角处比较各 load 的 ID 力矩。

    整段均值会把不同深度/速度/重复次数的相位混在一起；
    固定膝角可以把“运动学差异”控制住，只看负载的效应。

    对每个 upward 段单独取膝角最接近目标值的那一帧，再在段间平均，
    因此不会因为某些 load 重复次数少而被加权。膝角统一取绝对值，
    避开模型与 Xsens 的屈/伸符号约定差异。
    """
    mot_by_key = {
        _canon_load_key(k): v
        for k, v in get_mot_files(config, base_dir).items()
    }

    group_col_candidates = ('segment_id', 'cycle_id')

    for target in target_angles:
        summary = {jb: {} for jb in coord_map.keys()}
        counts = {}
        dropped = {}

        for load_key in load_keys:
            key = _canon_load_key(load_key)
            segment_df = get_segment_from_results(
                pipeline_results, load_key, movement_types=movement_types)
            mot_path = mot_by_key.get(key)
            id_df = read_opensim_table(
                get_inverse_dynamics_path(config, base_dir, load_key))
            if segment_df is None or mot_path is None or id_df is None:
                continue

            mot_df = read_opensim_table(mot_path)
            knee = interpolate_column_to_segment(mot_df, segment_df, knee_coord)
            if knee is None:
                continue
            knee = np.abs(knee)

            bad = np.zeros(len(segment_df), dtype=bool)
            if DROP_FROZEN_FRAMES and 'time' in segment_df.columns:
                intervals = load_dropout_intervals(mot_path, mot_df=mot_df)
                if intervals:
                    bad = in_intervals(segment_df['time'].values, intervals,
                                       DROPOUT_MARGIN)

            group_col = next(
                (c for c in group_col_candidates if c in segment_df.columns), None)
            if group_col is None:
                groups = [np.arange(len(segment_df))]
            else:
                gv = segment_df[group_col].values
                groups = [np.where(gv == g)[0] for g in pd.unique(gv)]

            # 每段选一帧：膝角最接近 target，且不落在丢帧区间内
            picked = []
            n_drop = 0
            for idx in groups:
                keep = idx[~bad[idx]]
                if len(keep) == 0:
                    n_drop += 1               # 整段落在丢帧区间内
                    continue
                sub = knee[keep]
                if not np.any(np.isfinite(sub)):
                    n_drop += 1
                    continue
                j = int(np.nanargmin(np.abs(sub - target)))
                if abs(sub[j] - target) > 10.0:   # 该段未达到目标角度，舍弃
                    continue
                picked.append(int(keep[j]))
            counts[str(load_key)] = len(picked)
            dropped[str(load_key)] = n_drop
            if not picked:
                continue

            for joint_base, coord in coord_map.items():
                id_col = find_id_moment_column(id_df, coord)
                if id_col is None:
                    continue
                vals = interpolate_column_to_segment(id_df, segment_df, id_col)
                if vals is None:
                    continue
                sel = np.abs(np.asarray(vals, dtype=float)[picked])
                sel = sel[np.isfinite(sel)]
                if len(sel) > 0:
                    summary[joint_base][str(load_key)] = float(np.mean(sel))

        print_summary_table(
            title=f'ID |力矩| @ 膝角≈{target:.0f}°（每段取一帧，段间平均）',
            summary=summary,
            load_keys=load_keys,
            unit='N·m',
            note=('说明: 已控制膝角，因此此表应比整段均值更接近单调。'
                  f'参与统计的段数: {counts}；'
                  f'因丢帧剔除的段数: {dropped}')
        )
        report_monotonicity(f'@膝角{target:.0f}°', summary, load_keys)


# ============================================================
#  主程序
# ============================================================

def main():
    base_dir = get_base_dir()
    config_path = get_config_path()

    print(f'配置文件: {config_path}')
    print(f'基准目录: {base_dir}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    load_keys = get_load_keys(config, LOAD_KEYS)
    if EXCLUDE_LOAD_KEYS:
        excluded = {_canon_load_key(k) for k in EXCLUDE_LOAD_KEYS}
        removed = [k for k in load_keys if _canon_load_key(k) in excluded]
        load_keys = [k for k in load_keys if _canon_load_key(k) not in excluded]
        if removed:
            print(f'已排除非负载试验: {removed}')

    # 左右两侧共 6 个关节坐标（hip/knee/ankle × l/r）都要计算并打印，
    # 不能只统计左腿。
    coord_map = build_bilateral_joint_coordinate_map(
        config,
        joint_bases=JOINT_BASES_TO_PRINT,
    )

    if not coord_map:
        raise ValueError(
            '未找到可统计的左右关节坐标；请检查 '
            'opensim_settings.muscle_analysis_coordinates'
        )

    print('\n将统计以下左右关节坐标：')
    for joint_base, coord in coord_map.items():
        print(f'  {joint_base}: {coord}')

    # 1) 先运行 InverseDynamics
    # 注意：run_step3_inverse_dynamics 当前没有 load_keys 参数，
    # 因此这里运行配置中所有可用负载。后续统计表仍按 LOAD_KEYS 过滤打印。
    run_step3_inverse_dynamics(
        config=config,
        base_dir=base_dir,
        use_external_forces=USE_EXTERNAL_FORCES,
        Mb=MB,
        output_body_forces=OUTPUT_BODY_FORCES,
        verbose=True,
    )

    # 2) 再获得标准切片时间点
    #    优先读取 cutted_data.csv；若不存在，则用 aligned_data.csv 快速切片；
    #    两者都没有时，才运行完整 MultiLoadPipeline。
    subject, pipeline, pipeline_results = load_or_create_cutted_pipeline_results(
        config_path,
        include_xsens=False,
        debug=True,
        force_rebuild=FORCE_REBUILD_CUTTED_CACHE,
        cache_name=CUTTED_CACHE_NAME,
    )

    # 2.5) 诊断：时间轴覆盖率 + 相位一致性
    if RUN_DIAGNOSTICS:
        diagnose_time_axis_coverage(
            config=subject.config,
            base_dir=base_dir,
            pipeline_results=pipeline_results,
            load_keys=load_keys,
            movement_types=MOVEMENT_TYPES,
        )
        diagnose_phase_alignment(
            subject=subject,
            config=subject.config,
            base_dir=base_dir,
            load_keys=load_keys,
        )

    # 3) 打印 signed mean
    id_mean = summarize_inverse_dynamics_moments(
        config=subject.config,
        base_dir=base_dir,
        pipeline_results=pipeline_results,
        load_keys=load_keys,
        coordinates=coord_map,
        movement_types=MOVEMENT_TYPES,
        statistic='mean',
    )

    print_summary_table(
        title=f'ID 关节力矩 signed mean（标准切片: {MOVEMENT_TYPES}）',
        summary=id_mean,
        load_keys=load_keys,
        unit='N·m',
        note=('说明: 对 inverse_dynamics.sto 的 ID moment 按标准切片时间点插值，'
              '然后直接计算有符号平均值。')
    )

    # 4) 打印 mean abs，更适合检查负载增大时绝对力矩是否增大
    id_mean_abs = summarize_inverse_dynamics_moments(
        config=subject.config,
        base_dir=base_dir,
        pipeline_results=pipeline_results,
        load_keys=load_keys,
        coordinates=coord_map,
        movement_types=MOVEMENT_TYPES,
        statistic='mean_abs',
    )

    print_summary_table(
        title=f'ID 关节力矩 mean abs（标准切片: {MOVEMENT_TYPES}）',
        summary=id_mean_abs,
        load_keys=load_keys,
        unit='N·m',
        note=('说明: 对同一批 upward 时间点先取 |ID moment|，再计算平均值；'
              '该表更适合检查负载增加时关节力矩绝对值是否上升。')
    )

    # 5) 单调性报告
    if RUN_DIAGNOSTICS:
        report_monotonicity('mean abs', id_mean_abs, load_keys)

    # 6) 姿态策略与匹配膝角对比
    if RUN_DIAGNOSTICS:
        diagnose_posture_strategy(
            config=subject.config,
            base_dir=base_dir,
            pipeline_results=pipeline_results,
            load_keys=load_keys,
            movement_types=MOVEMENT_TYPES,
        )

    if RUN_MATCHED_ANGLE:
        summarize_moments_at_matched_knee_angle(
            config=subject.config,
            base_dir=base_dir,
            pipeline_results=pipeline_results,
            load_keys=load_keys,
            coord_map=coord_map,
            movement_types=MOVEMENT_TYPES,
        )


if __name__ == '__main__':
    main()