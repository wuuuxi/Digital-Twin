"""
example_xsens_to_mot.py

单独运行 Xsens Excel -> OpenSim .mot 的转换，并在写出 .mot 前严格约束
各个 OpenSim 坐标的范围。

用途：
  - 排查 Xsens -> mot 过程中是否出现超出 OpenSim 模型坐标范围的角度；
  - 生成已经 clamp 到模型坐标范围内的 .mot；
  - 保留一份 raw/unconstrained .mot 方便对比。

输出：
  result/{experiment_label}/opensim/mot/
    {xsens_stem}_opensim_raw.mot          # 未约束版本
    {xsens_stem}_opensim.mot              # 约束后版本，默认覆盖标准输出名

说明：
  1. read_xsens_excel_for_opensim() 负责从 Xsens Excel 读取并生成原始 OpenSim 坐标；
  2. 本脚本随后对每个坐标执行 np.clip(value, lower, upper)；
  3. 约束范围按 OpenSim Coordinate slider 范围设置，如截图所示。
"""
import os
import json
import shutil
from pathlib import Path

import numpy as np

from digitaltwin.utils.logger import beauty_print
from digitaltwin.osim.mot_pipeline import (
    read_xsens_excel_for_opensim,
    save_dropout_intervals,
    load_dropout_intervals,
)


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../../config/20260513_squat_FTS09_xsens.json'

# None = 处理 config 中所有 load；也可指定，如 ['20', '38', '56']
LOAD_KEYS = None

# True：约束后的文件保存为标准名 {xsens_stem}_opensim.mot，后续 OpenSim pipeline 会直接使用它。
# False：约束后的文件保存为 {xsens_stem}_opensim_constrained.mot，不覆盖标准文件。
OVERWRITE_STANDARD_MOT = True

# 若 OVERWRITE_STANDARD_MOT=True 且标准 .mot 已存在，是否先备份旧文件。
BACKUP_EXISTING_STANDARD_MOT = True

# 是否保留 raw/unconstrained .mot。
KEEP_RAW_MOT = True

# clip 不是“修复”，而是把超范围的角度削平：被削平的帧，其角速度与
# 角加速度是错的，ID 会在这些帧上算出错误的惯性项而不报任何错。
# 因此这里把“被削平的时间段”当成不可用数据，与 Step1 检测到的丢帧
# 区间合并后写入 .dropouts.csv，交给下游按时间段剔除（不丢整组数据）。
MARK_CLIPPED_AS_UNUSABLE = True

# 参与“不可用”判定的坐标（只关心进入下肢 ID 的那些）
CLIP_CRITICAL_COORDS = ('pelvis_tilt', 'pelvis_list',
                        'hip_flexion_l', 'hip_flexion_r',
                        'knee_angle_l', 'knee_angle_r',
                        'ankle_angle_l', 'ankle_angle_r')

# 短于该时长的被削平片段不单独上报（秒）
CLIP_MIN_DURATION = 0.10

# 某坐标超限的帧超过该比例时，明确警告该坐标的运动学不可信
CLIP_FRACTION_WARN = 0.005

# 是否真的把超限值削平（np.clip）。
#
# 默认 False。clip 从来不是“修复”，它只是把超限的值按平，结果是：
#   1. 被削平的连续帧变成一条水平线，角速度=0、角加速度=0，
#      ID 会在这些帧上算出错误的惯性项而不报任何错；
#   2. 左右踝同时被削平在 30.00° 时，“左右峰值差=0.0°”是伪造的对称，
#      反而堆盖了真实的不对称；
#   3. 超限本身是一个很好的错误探测器，削平等于把警报删掉。
# 因此默认只“检测 + 报告 + 标记为不可用时间段”，不修改数值。
# 超限的正确处理方式是对症下药：
#   模型限位过窄  -> 放宽 .osim 的 Coordinate range；
#   jmap 映射错误  -> 修映射/零点/符号；
#   Xsens 本身异常 -> 按时间段剔除。
ENFORCE_CLIP = False


# ============================================================
#  关节角 / 坐标范围约束
#  单位：角度为 deg，平移为 m
# ============================================================

COORD_LIMITS = {
    # pelvis orientation / translation
    # pelvis 是自由浮动关节，模型里并没有 ±90° 的物理限位；OpenSim slider
    # 的默认范围只是显示范围。把它 clip 到 ±90° 会做两件坏事：
    #   1. 把 yaw（pelvis_rotation）整段削平在 ±90.00；
    #   2. 把 Step1 里 unwrap 展开后的角度重新折回去，等于撤销了绕接修复。
    'pelvis_tilt':     (-180.0, 180.0),
    'pelvis_list':     (-180.0, 180.0),
    'pelvis_rotation': (-180.0, 180.0),
    'pelvis_tx':       (-5.0, 5.0),
    'pelvis_ty':       (-1.0, 2.0),
    'pelvis_tz':       (-3.0, 3.0),

    # right leg
    'hip_flexion_r':    (-120.0, 120.0),
    'hip_adduction_r':  (-120.0, 120.0),
    'hip_rotation_r':   (-120.0, 120.0),
    'knee_angle_r':     (0.0, 150.0),
    # 实测背屈峰值 33-36°，深蹲到底时超过 gait2392 默认的 30° 完全正常，
    # 这是模型限位过窄而不是数据异常，应同步放宽 .osim 的 range。
    'ankle_angle_r':    (-50.0, 50.0),
    'subtalar_angle_r': (-30.0, 30.0),
    'mtp_angle_r':      (-40.0, 40.0),

    # left leg
    'hip_flexion_l':    (-120.0, 120.0),
    'hip_adduction_l':  (-120.0, 120.0),
    'hip_rotation_l':   (-120.0, 120.0),
    'knee_angle_l':     (0.0, 150.0),
    'ankle_angle_l':    (-50.0, 50.0),
    'subtalar_angle_l': (-30.0, 30.0),
    'mtp_angle_l':      (-40.0, 40.0),

    # right shoulder / arm
    'SC_y':        (-30.0, 30.0),
    'SC_x':        (-45.0, 10.0),
    'SC_z':        (0.0, 35.0),
    'arm_flex_r':  (-90.0, 180.0),
    'arm_add_r':   (-120.0, 90.0),
    'arm_rot_r':   (-90.0, 90.0),
    'elbow_flex_r': (0.0, 150.0),
    'pro_sup_r':   (0.0, 90.0),
    'wrist_flex_r': (-70.0, 70.0),
    'wrist_dev_r':  (-25.0, 35.0),

    # left shoulder / arm
    'SC_y_l':       (-30.0, 30.0),
    'SC_x_l':       (-45.0, 10.0),
    'SC_z_l':       (-35.0, 0.0),
    'arm_flex_l':   (-90.0, 180.0),
    'arm_add_l':    (-120.0, 90.0),
    'arm_rot_l':    (-90.0, 90.0),
    'elbow_flex_l': (0.0, 150.0),
    'pro_sup_l':    (0.0, 90.0),
    'wrist_flex_l': (-70.0, 70.0),
    'wrist_dev_l':  (-25.0, 35.0),
}


# ============================================================
#  路径工具
# ============================================================

def get_base_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


def get_load_keys(config):
    if LOAD_KEYS is None:
        return [str(k) for k in config['modeling_file']['data'].keys()]
    return [str(k) for k in LOAD_KEYS]


# ============================================================
#  MOT 写入
# ============================================================

def write_mot(output_path, data_with_time, joint_names_order, name='constrained_mot'):
    """写出 OpenSim .mot 文件。"""
    n_samples = data_with_time.shape[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('first trial\n')
        f.write(f'nRows={n_samples}\n')
        f.write(f'nColumns={len(joint_names_order) + 1}\n\n')
        f.write('# SIMM Motion File Header:\n')
        f.write(f'name {name}\n')
        f.write(f'datacolumns {len(joint_names_order) + 1}\n')
        f.write(f'datarows {n_samples}\n')
        f.write('otherdata 1\n')
        # 显式声明角度单位，不依赖 OpenSim 对旧格式的默认猜测。
        f.write('inDegrees=yes\n')
        data_min = np.nanmin(data_with_time[:, 1:])
        data_max = np.nanmax(data_with_time[:, 1:])
        f.write(f'range {data_min:.4f} {data_max:.4f}\n')
        f.write('endheader\n')
        f.write('time\t' + '\t'.join(joint_names_order) + '\n')
        for row in data_with_time:
            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')


def mask_to_intervals(time, mask, min_duration=0.0):
    """把逐帧布尔掩码转成 [(t0, t1), ...]，短于 min_duration 的片段丢弃。"""
    time = np.asarray(time, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    intervals = []
    i, n = 0, time.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if time[j] - time[i] >= min_duration:
            intervals.append((float(time[i]), float(time[j])))
        i = j + 1
    return intervals


def merge_intervals(intervals):
    """合并重叠/相邻区间。"""
    if not intervals:
        return []
    out = []
    for t0, t1 in sorted(intervals):
        if out and t0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], t1))
        else:
            out.append((t0, t1))
    return [(float(a), float(b)) for a, b in out]


def apply_coordinate_limits(data_with_time, joint_names_order, limits,
                            clip=False):
    """
    检测（并可选地削平）超出坐标范围的帧。

    Parameters
    ----------
    clip : bool
        False（默认）只检测与统计，不修改任何数值；
        True 才真的执行 np.clip（会破坏该段的角速度/角加速度）。

    Returns
    -------
    constrained : np.ndarray
    report : list[dict]
    clip_mask : dict[str, np.ndarray]
        逐帧超限掉码，无论是否真的削平都会返回。
    """
    constrained = data_with_time.copy()
    report = []
    n_frames = data_with_time.shape[0]
    clip_mask = {}

    for jname, (lo, hi) in limits.items():
        if jname not in joint_names_order:
            report.append({
                'joint': jname,
                'status': 'missing',
                'n_clipped': None,
                'before_min': None,
                'before_max': None,
                'after_min': None,
                'after_max': None,
                'lo': lo,
                'hi': hi,
            })
            continue

        # +1 是因为第 0 列是 time
        col_idx = joint_names_order.index(jname) + 1
        before = constrained[:, col_idx].copy()
        clipped = np.clip(before, lo, hi)
        clipped_frames = np.abs(clipped - before) > 1e-10

        # 超限帧总是要标记；但是否真的把数值削平，由 clip 决定。
        if clip:
            constrained[:, col_idx] = clipped
            after = clipped
        else:
            after = before
        clip_mask[jname] = clipped_frames
        n_clip = int(np.sum(clipped_frames))
        report.append({
            'joint': jname,
            'status': 'ok',
            'n_clipped': n_clip,
            'frac_clipped': (n_clip / n_frames) if n_frames else 0.0,
            'before_min': float(np.nanmin(before)),
            'before_max': float(np.nanmax(before)),
            'after_min': float(np.nanmin(after)),
            'after_max': float(np.nanmax(after)),
            'lo': lo,
            'hi': hi,
        })

    return constrained, report, clip_mask


def print_clip_report(load_key, report):
    """打印裁剪统计。"""
    print(f'\n[load={load_key}] 坐标范围约束报告')
    print('-' * 96)
    print(f'{"joint":<20s}{"limit":>18s}{"before":>24s}{"after":>24s}'
          f'{"clipped":>10s}{"占比":>8s}')
    print('-' * 96)

    for r in report:
        if r['status'] == 'missing':
            print(f'{r["joint"]:<20s}{"MISSING":>18s}{"":>24s}{"":>24s}{"":>10s}')
            continue

        # 只打印被裁剪过的坐标，以及范围本来接近边界的坐标；如需全部打印可去掉该判断
        if r['n_clipped'] == 0:
            continue

        limit_s = f'[{r["lo"]:.1f}, {r["hi"]:.1f}]'
        before_s = f'[{r["before_min"]:.2f}, {r["before_max"]:.2f}]'
        after_s = f'[{r["after_min"]:.2f}, {r["after_max"]:.2f}]'
        frac = r.get('frac_clipped', 0.0)
        print(f'{r["joint"]:<20s}{limit_s:>18s}{before_s:>24s}{after_s:>24s}'
              f'{r["n_clipped"]:>10d}{frac:>8.1%}')

    print('-' * 96)

    bad = [r for r in report if r['status'] == 'ok'
           and r.get('frac_clipped', 0.0) >= CLIP_FRACTION_WARN]
    if bad:
        lines = [f'load={load_key}: 以下坐标超出模型范围的帧数占比过高，'
                 '其运动学不可信：']
        for r in sorted(bad, key=lambda x: -x['frac_clipped']):
            lines.append(
                f'  {r["joint"]:<18s}{r["frac_clipped"]:>7.1%}  '
                f'原始范围 [{r["before_min"]:.2f}, {r["before_max"]:.2f}] '
                f'超出限值 [{r["lo"]:.1f}, {r["hi"]:.1f}]')
        lines.append('  处理方式（不要用 clip 掩盖）: '
                     '模型限位过窄 -> 放宽 .osim 的 Coordinate range；'
                     'jmap 映射错误 -> 修映射/零点/符号；'
                     'Xsens 异常 -> 按时间段剔除。')
        beauty_print('\n'.join(lines), type="warning")


# ============================================================
#  单个 load 转换
# ============================================================

def convert_one_load(config, base_dir, load_key):
    experiment_label = config['experiment_label']
    folder = config['folder']
    modeling = config['modeling_file']
    xsens_dir = os.path.join(folder, modeling.get('xsens_folder', 'xsens'))

    file_info = modeling['data'].get(str(load_key))
    if file_info is None:
        print(f'[MISS] load={load_key}: config 中没有该负载')
        return None

    xsens_file = file_info.get('xsens_file')
    if not xsens_file:
        print(f'[MISS] load={load_key}: 无 xsens_file')
        return None

    xsens_path = os.path.join(xsens_dir, xsens_file)
    stem = Path(xsens_file).stem

    output_dir = os.path.join(base_dir, 'result', experiment_label, 'opensim', 'mot')
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, f'{stem}_opensim_raw.mot')
    standard_path = os.path.join(output_dir, f'{stem}_opensim.mot')
    constrained_path = (
        standard_path if OVERWRITE_STANDARD_MOT
        else os.path.join(output_dir, f'{stem}_opensim_constrained.mot')
    )

    if OVERWRITE_STANDARD_MOT and BACKUP_EXISTING_STANDARD_MOT and os.path.exists(standard_path):
        backup_path = os.path.join(output_dir, f'{stem}_opensim_before_constrained.mot')
        shutil.copy2(standard_path, backup_path)
        print(f'[backup] {standard_path} -> {backup_path}')

    print(f'\n[load={load_key}] Xsens -> raw MOT')
    print(f'  xsens: {xsens_path}')
    print(f'  raw  : {raw_path}')

    data_with_time, joint_names_order = read_xsens_excel_for_opensim(
        xsens_path,
        output_mot_path=raw_path,
    )

    constrained, report, clip_mask = apply_coordinate_limits(
        data_with_time,
        joint_names_order,
        COORD_LIMITS,
        clip=ENFORCE_CLIP,
    )

    write_mot(
        constrained_path,
        constrained,
        joint_names_order,
        name=f'{stem}_opensim_constrained',
    )
    print(f'  constrained: {constrained_path}')

    # Step1 的丢帧清单是写在 raw 文件旁边的；下游用的是这个 constrained
    # 文件，因此必须把清单（并入被削平的时间段）另存到 constrained 文件
    # 旁边，否则下游看不到任何 .dropouts.csv，会误以为 Step1 没有重跑。
    time_col = constrained[:, 0]
    intervals = list(load_dropout_intervals(raw_path))
    if MARK_CLIPPED_AS_UNUSABLE:
        crit = [c for c in CLIP_CRITICAL_COORDS if c in clip_mask]
        if crit:
            bad_frames = np.zeros(time_col.shape, dtype=bool)
            for c in crit:
                bad_frames |= clip_mask[c]
            intervals += mask_to_intervals(time_col, bad_frames,
                                           CLIP_MIN_DURATION)
    intervals = merge_intervals(intervals)
    sidecar = save_dropout_intervals(constrained_path, intervals)
    total = sum(t1 - t0 for t0, t1 in intervals)
    print(f'  不可用时间段清单: {sidecar}  ({len(intervals)} 段 / {total:.2f} s)')

    if not KEEP_RAW_MOT and os.path.exists(raw_path):
        os.remove(raw_path)
        print(f'  raw removed: {raw_path}')

    print_clip_report(load_key, report)

    return constrained_path


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

    load_keys = get_load_keys(config)
    print(f'处理负载: {load_keys}')

    out = {}
    for load_key in load_keys:
        path = convert_one_load(config, base_dir, load_key)
        if path is not None:
            out[str(load_key)] = path

    print('\n完成。约束后 MOT 文件：')
    for load_key, path in out.items():
        print(f'  {load_key}: {path}')


if __name__ == '__main__':
    main()