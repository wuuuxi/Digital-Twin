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

from digitaltwin.osim.mot_pipeline import read_xsens_excel_for_opensim


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'

# None = 处理 config 中所有 load；也可指定，如 ['20', '38', '56']
LOAD_KEYS = None

# True：约束后的文件保存为标准名 {xsens_stem}_opensim.mot，后续 OpenSim pipeline 会直接使用它。
# False：约束后的文件保存为 {xsens_stem}_opensim_constrained.mot，不覆盖标准文件。
OVERWRITE_STANDARD_MOT = True

# 若 OVERWRITE_STANDARD_MOT=True 且标准 .mot 已存在，是否先备份旧文件。
BACKUP_EXISTING_STANDARD_MOT = True

# 是否保留 raw/unconstrained .mot。
KEEP_RAW_MOT = True


# ============================================================
#  关节角 / 坐标范围约束
#  单位：角度为 deg，平移为 m
# ============================================================

COORD_LIMITS = {
    # pelvis orientation / translation
    'pelvis_tilt':     (-90.0, 90.0),
    'pelvis_list':     (-90.0, 90.0),
    'pelvis_rotation': (-90.0, 90.0),
    'pelvis_tx':       (-5.0, 5.0),
    'pelvis_ty':       (-1.0, 2.0),
    'pelvis_tz':       (-3.0, 3.0),

    # right leg
    'hip_flexion_r':    (-120.0, 120.0),
    'hip_adduction_r':  (-120.0, 120.0),
    'hip_rotation_r':   (-120.0, 120.0),
    'knee_angle_r':     (0.0, 120.0),
    'ankle_angle_r':    (-40.0, 30.0),
    'subtalar_angle_r': (-20.0, 20.0),
    'mtp_angle_r':      (-30.0, 30.0),

    # left leg
    'hip_flexion_l':    (-120.0, 120.0),
    'hip_adduction_l':  (-120.0, 120.0),
    'hip_rotation_l':   (-120.0, 120.0),
    'knee_angle_l':     (0.0, 120.0),
    'ankle_angle_l':    (-40.0, 30.0),
    'subtalar_angle_l': (-20.0, 20.0),
    'mtp_angle_l':      (-30.0, 30.0),

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
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))


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
        data_min = np.nanmin(data_with_time[:, 1:])
        data_max = np.nanmax(data_with_time[:, 1:])
        f.write(f'range {data_min:.4f} {data_max:.4f}\n')
        f.write('endheader\n')
        f.write('time\t' + '\t'.join(joint_names_order) + '\n')
        for row in data_with_time:
            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')


def apply_coordinate_limits(data_with_time, joint_names_order, limits):
    """
    对 data_with_time 中的坐标列执行 clip。

    Returns
    -------
    constrained : np.ndarray
    report : list[dict]
        每个被约束坐标的统计信息。
    """
    constrained = data_with_time.copy()
    report = []

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
        after = np.clip(before, lo, hi)
        constrained[:, col_idx] = after

        n_clip = int(np.sum(np.abs(after - before) > 1e-10))
        report.append({
            'joint': jname,
            'status': 'ok',
            'n_clipped': n_clip,
            'before_min': float(np.nanmin(before)),
            'before_max': float(np.nanmax(before)),
            'after_min': float(np.nanmin(after)),
            'after_max': float(np.nanmax(after)),
            'lo': lo,
            'hi': hi,
        })

    return constrained, report


def print_clip_report(load_key, report):
    """打印裁剪统计。"""
    print(f'\n[load={load_key}] 坐标范围约束报告')
    print('-' * 96)
    print(f'{"joint":<20s}{"limit":>18s}{"before":>24s}{"after":>24s}{"clipped":>10s}')
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
        print(f'{r["joint"]:<20s}{limit_s:>18s}{before_s:>24s}{after_s:>24s}{r["n_clipped"]:>10d}')

    print('-' * 96)


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

    constrained, report = apply_coordinate_limits(
        data_with_time,
        joint_names_order,
        COORD_LIMITS,
    )

    write_mot(
        constrained_path,
        constrained,
        joint_names_order,
        name=f'{stem}_opensim_constrained',
    )
    print(f'  constrained: {constrained_path}')

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