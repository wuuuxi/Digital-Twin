"""
example_cop_sensitivity.py

放于 example_inverse_dynamics.py 同一目录（examples/opensim/inverse_dynamics/）。

目的：量化“足底 GRF 作用点（COP）位置”对 ID 关节力矩的影响。

为什么要做：
  external_forces.py 目前把左右 GRF 施加在 insole_contact_point（默认 [0,0,0]，
  即 calcn 原点 ≈ 跟部）上，且逐帧不变。真实 COP 在深蹲中位于足中前部，
  并沿前后方向移动数厘米。由于

      M_joint ≈ F_GRF × (关节中心到 COP 的力臂)

  而 GRF 量级为 500-1500 N，0.03 m 的作用点误差就能产生 15-45 N·m 的力矩误差，
  与观测到的膝力矩非单调量（2-5 N·m）相比绰绰有余。先用单变量扰动实验确认
  这一量级，再决定是否值得去解鞑垫逐点力数据。

方法：
  1. 固定运动学（.mot）、固定缩放模型、固定切片与比较帧、固定力大小；
  2. 只改 insole_contact_point 的前后坐标 dx（calcn 局部 +X = 前向）；
  3. 每个 dx 重新生成外力并重跑 ID，输出到
     result/{label}/opensim/sensitivity/{tag}/{load}/（不覆盖正式结果）；
  4. 在固定膝角处取 |力矩|（每个 upward 段一帧，段间均值 ± 标准差）；
  5. 报告每个关节的敏感度 d|M|/d(COP)，以及哪个 dx 下力矩才随负载单调。

注意：本脚本不会永久修改 config（扰动只作用于内存 deepcopy），但
  generate_external_loads 会覆写 external_forces/{load}/ 下的共享文件，
  因此脚本结尾（finally）会用基线接触点重新生成一次，把共享文件恢复。
  另：本脚本绕过 run_step3_inverse_dynamics，直接调 run_inverse_dynamics，
  因为 Step3 会静默复用已存在的 bar_loads_{load}.xml，使扰动失效。

数据质量处理原则（不丢掉整组数据）：
  - 欧拉角绕接翻转（106 kg 的 pelvis_tilt = +36.9°，其余负载 -32~-38°）属于
    Step1 的转换缺陷，已在 mot_pipeline.read_xsens_excel_for_opensim 中用
    unwrap_degrees() 修正，不在这里排除负载。改完需重跑 Step1 + Step3。
  - 丢帧/插值成直线的区间（75 kg 的 15-25 s）由 mot_pipeline 的
    detect_frozen_intervals() 自动检测，写成 .dropouts.csv 旁路文件；
    本脚本只剔除落在这些区间内的比较帧（可加 DROPOUT_MARGIN 安全边界），
    其余循环照常参统计。既没有 sidecar 也能直接从 .mot 现算，
    因此无需为了做诊断而先重跑 Step1。
  - 只有非负载试次（0.15 / 0.3）默认排除，因为它们不属于负载序列。
"""
import os
import json
import copy
import shutil

import numpy as np
import pandas as pd

from digitaltwin.osim.mot_pipeline import (
    get_mot_files,
    get_scaled_model,
    load_dropout_intervals,
    in_intervals,
)
from digitaltwin.osim.external_forces import (
    generate_external_loads,
    get_ext_forces_dir,
)
from digitaltwin.osim.inverse_dynamics import run_inverse_dynamics
# 编排层（跑流水线 / 带缓存的切片装载 / 动作窗口）
from digitaltwin.pipelines.standard_analysis import (
    load_or_create_cutted_pipeline_results,
)
# 纯分析层
from digitaltwin.analysis.result_analysis import (
    build_left_joint_coordinate_map,
    get_load_keys,
    get_segment_from_results,
    read_opensim_table,
    interpolate_column_to_segment,
    find_id_moment_column,
    print_summary_table,
)

# 与 example_inverse_dynamics.py 同目录，直接复用其工具函数，保证两边口径一致。
from example_inverse_dynamics import (
    _canon_load_key,
    report_monotonicity,
    get_base_dir,
    get_config_path,
)


# ============================================================
#  配置
# ============================================================

LOAD_KEYS = None                            # None = 全部
EXCLUDE_LOAD_KEYS = ['0.15', '0.3']         # 只排除非负载试次

# 丢帧区间处理：剔除落在丢帧区间（±margin 秒）内的比较帧，而不丢整组数据。
# margin 留余量是因为 ID 的惯性项依赖二阶导数，区间边缘上一两帧也不可信。
DROP_FROZEN_FRAMES = True
DROPOUT_MARGIN = 0.10                       # 秒

# COP 前后位置扫描（calcn 局部 +X = 前向，m）
# 0.00 = 当前设置（跟部）；成人足长约 0.25 m，深蹲时真实 COP 大约在 0.09-0.15 m
COP_AP_OFFSETS = (0.00, 0.03, 0.06, 0.09, 0.12, 0.15)
COP_VERTICAL = 0.0        # 竖直位置（鞑垫厚度）
COP_MEDIOLATERAL = 0.0    # 内外侧位置

# 可选对照：同时扫描杆作用点的前后位置
RUN_BAR_SWEEP = False
BAR_AP_OFFSETS = (-0.13, -0.10, -0.07, -0.04, -0.01)

# 匹配膝角（度）与容差；70° 是所有负载都能达到的深度（90° 高负载达不到）
TARGET_KNEE_ANGLE = 70.0
ANGLE_TOLERANCE = 10.0
KNEE_COORD = 'knee_angle_l'
MOVEMENT_TYPES = ('upward',)

JOINT_BASES = None        # None = 自动取所有左腿关节
MB = 0.0                  # 会被 config.opensim_settings.bar_mass 覆盖
VERBOSE_ID = False


# ============================================================
#  工具
# ============================================================

def _tag_for(prefix, value):
    """生成可做目录名的标签，例如 cop_ap_p090mm / bar_ap_m100mm。"""
    mm = int(round(value * 1000))
    sign = 'p' if mm >= 0 else 'm'
    return f'{prefix}_{sign}{abs(mm):03d}mm'


def pick_frames_at_knee_angle(mot_df, segment_df, target=TARGET_KNEE_ANGLE,
                              tolerance=ANGLE_TOLERANCE,
                              knee_coord=KNEE_COORD,
                              dropout_intervals=None,
                              margin=DROPOUT_MARGIN):
    """
    在每个 upward 段中取膝角最接近 target 的一帧。

    - 膝角取绝对值，避开屈/伸符号约定差异；
    - 每段只取一帧，因此重复次数不同的 load 不会被不当加权；
    - 落在丢帧区间（±margin）内的帧不可用；若整段都不可用才舍弃该段，
      同一 load 的其余循环仍然参与统计；
    - 该帧选择只依赖运动学，与外力无关，所以在整个扫描中只算一次。

    Returns
    -------
    (picked : list[int], n_total : int, n_dropped : int)
    """
    knee = interpolate_column_to_segment(mot_df, segment_df, knee_coord)
    if knee is None:
        return [], 0, 0
    knee = np.abs(np.asarray(knee, dtype=float))

    n_rows = len(segment_df)
    bad = np.zeros(n_rows, dtype=bool)
    if DROP_FROZEN_FRAMES and dropout_intervals and 'time' in segment_df.columns:
        bad = in_intervals(segment_df['time'].values, dropout_intervals, margin)

    group_col = None
    for c in ('segment_id', 'cycle_id'):
        if c in segment_df.columns:
            group_col = c
            break

    if group_col is None:
        groups = [np.arange(n_rows)]
    else:
        gv = segment_df[group_col].values
        groups = [np.where(gv == g)[0] for g in pd.unique(gv)]

    picked = []
    n_dropped = 0
    for idx in groups:
        keep = idx[~bad[idx]]
        if len(keep) == 0:
            n_dropped += 1          # 整段落在丢帧区间内
            continue
        sub = knee[keep]
        if not np.any(np.isfinite(sub)):
            n_dropped += 1
            continue
        j = int(np.nanargmin(np.abs(sub - target)))
        if abs(sub[j] - target) > tolerance:
            continue                # 这一段没蹲到目标膝角，不算丢帧剔除
        picked.append(int(keep[j]))
    return picked, len(groups), n_dropped


def measure_moments(id_dir, segment_df, picked, coord_map):
    """
    从指定 ID 输出目录读力矩，在 picked 帧上统计 |力矩| 的均值、标准差、样本数。

    Returns
    -------
    dict  {joint_base: (mean_abs, std_abs, n)}
    """
    id_path = os.path.join(id_dir, 'inverse_dynamics.sto')
    if not os.path.exists(id_path):
        return {}

    id_df = read_opensim_table(id_path)
    if id_df is None or 'time' not in id_df.columns or not picked:
        return {}

    out = {}
    for joint_base, coord in coord_map.items():
        id_col = find_id_moment_column(id_df, coord)
        if id_col is None:
            continue
        vals = interpolate_column_to_segment(id_df, segment_df, id_col)
        if vals is None:
            continue
        sel = np.abs(np.asarray(vals, dtype=float)[picked])
        sel = sel[np.isfinite(sel)]
        if len(sel) == 0:
            continue
        out[joint_base] = (float(np.mean(sel)), float(np.std(sel)), int(len(sel)))
    return out


def run_id_with_contact_point(config, base_dir, load_keys, mot_files,
                              scaled_model, setting_key, point_xyz, tag,
                              mb=MB, verbose=VERBOSE_ID):
    """
    用指定接触点重新生成外力并重跑 ID。

    生成的 xml/sto 会被复制到 sensitivity/{tag}/{load}/，ID 也从那里运行，
    因此不依赖共享目录的中间状态，也不覆盖正式 ID 结果。

    Returns
    -------
    dict  {canonical_load_key: id_dir}
    """
    cfg = copy.deepcopy(config)
    cfg.setdefault('opensim_settings', {})[setting_key] = list(point_xyz)

    label = cfg['experiment_label']
    id_dirs = {}

    for load_key, mot_path in mot_files.items():
        key = _canon_load_key(load_key)
        if key not in load_keys:
            continue

        xml_path = generate_external_loads(
            config=cfg, base_dir=base_dir, load_key=load_key,
            mot_path=mot_path, Mb=mb, verbose=verbose)
        if xml_path is None:
            print(f'  [{tag}] load={load_key}: 外力生成失败，跳过')
            continue

        work_dir = os.path.join(base_dir, 'result', label, 'opensim',
                                'sensitivity', tag, str(load_key))
        os.makedirs(work_dir, exist_ok=True)

        # xml 中的 <datafile> 是相对名，所以 sto 必须与 xml 同目录
        src_dir = get_ext_forces_dir(cfg, base_dir, load_key)
        sto_name = f'bar_force_{load_key}.sto'
        local_xml = os.path.join(work_dir, os.path.basename(xml_path))
        try:
            shutil.copy2(xml_path, local_xml)
            shutil.copy2(os.path.join(src_dir, sto_name),
                         os.path.join(work_dir, sto_name))
        except OSError as e:
            print(f'  [{tag}] load={load_key}: 复制外力文件失败 {e}')
            continue

        ok = run_inverse_dynamics(
            model_path=scaled_model,
            mot_path=mot_path,
            output_dir=work_dir,
            external_load_file=local_xml,
            label=f'{label}_{load_key}',
            verbose=verbose)

        if ok:
            id_dirs[key] = work_dir
        else:
            print(f'  [{tag}] load={load_key}: ID 运行失败')

    return id_dirs


def sweep_contact_point(config, base_dir, load_keys, mot_files, scaled_model,
                        seg_cache, coord_map, setting_key, offsets,
                        tag_prefix, build_point):
    """
    对一组偏移量执行扫描，逐个打印力矩表、段间标准差与单调性报告。

    Returns
    -------
    dict  {offset: {joint_base: {load_key: mean_abs}}}
    """
    all_results = {}

    for offset in offsets:
        tag = _tag_for(tag_prefix, offset)
        point = build_point(offset)
        print('\n' + '#' * 80)
        print(f'# 扫描 {setting_key} = {point}   (tag={tag})')
        print('#' * 80)

        id_dirs = run_id_with_contact_point(
            config=config, base_dir=base_dir, load_keys=load_keys,
            mot_files=mot_files, scaled_model=scaled_model,
            setting_key=setting_key, point_xyz=point, tag=tag)

        summary = {jb: {} for jb in coord_map.keys()}
        detail = {}

        for key, id_dir in id_dirs.items():
            cached = seg_cache.get(key)
            if cached is None:
                continue
            stats = measure_moments(id_dir, cached['segment_df'],
                                    cached['picked'], coord_map)
            for joint_base, (mean_abs, std_abs, n) in stats.items():
                summary[joint_base][key] = mean_abs
                detail[(joint_base, key)] = (std_abs, n)

        print_summary_table(
            title=(f'ID |力矩| @ 膝角≈{TARGET_KNEE_ANGLE:.0f}°  |  '
                   f'{setting_key} = {point}'),
            summary=summary,
            load_keys=load_keys,
            unit='N·m',
            note='说明: 每个 upward 段取一帧（膝角最接近目标），段间平均。')

        # 段间离散度：用于判断相邻负载的差值是否显著
        print()
        print(f'{"joint":<16}' + ''.join(f'{k + " kg":>18}' for k in load_keys))
        for joint_base in coord_map.keys():
            row = f'{joint_base:<16}'
            for k in load_keys:
                d = detail.get((joint_base, k))
                cell = 'N/A' if d is None else f'±{d[0]:.1f} (n={d[1]})'
                row += f'{cell:>18}'
            print(row)
        print('上表为段间标准差。若相邻负载的均值差小于各自标准差，'
              '则该“违反单调”在统计上不显著，不需要用建模误差解释。')

        report_monotonicity(f'{tag_prefix}={offset * 100:+.0f}cm',
                            summary, load_keys)
        all_results[offset] = summary

    return all_results


def print_sensitivity_summary(all_results, load_keys, coord_map, tag_prefix):
    """汇总表：各偏移下的跳负载平均 |力矩|，以及敏感度 d|M|/d(offset)。"""
    offsets = sorted(all_results.keys())
    if len(offsets) < 2:
        return

    print('\n' + '=' * 80)
    print(f'[汇总] {tag_prefix} 敏感度（各负载平均 |力矩|，N·m）')
    print('=' * 80)
    header = f'{"joint":<16}' + ''.join(f'{o * 100:>+11.0f}cm' for o in offsets)
    print(header)
    print('-' * len(header))

    for joint_base in coord_map.keys():
        row = f'{joint_base:<16}'
        means = []
        for o in offsets:
            vals = [v for k, v in all_results[o].get(joint_base, {}).items()
                    if k in load_keys and v is not None and np.isfinite(v)]
            m = float(np.mean(vals)) if vals else float('nan')
            means.append(m)
            row += f'{m:>13.2f}' if np.isfinite(m) else f'{"N/A":>13}'
        print(row)

        finite = [(o, m) for o, m in zip(offsets, means) if np.isfinite(m)]
        if len(finite) >= 2:
            (o0, m0), (o1, m1) = finite[0], finite[-1]
            span = o1 - o0
            if abs(span) > 1e-9:
                slope = (m1 - m0) / span
                print(f'{"":<16}敏感度 ≈ {slope:+.1f} N·m/m '
                      f'= {slope * 0.03:+.2f} N·m 每 3cm'
                      f'   (全范围变化 {m1 - m0:+.2f} N·m)')

    print('\n判读:')
    print('  1) 若膝关节敏感度（N·m 每 3cm）大于观测到的单调违反量（2-5 N·m），')
    print('     则固定 COP 足以解释非单调 → 转入鞑垫逐点力计算真实 COP。')
    print('  2) 若膝敏感度远小于违反量，则 COP 不是主因，应转向杆力作用点、')
    print('     GRF 重复计数（鞑垫力 + 杆力是否重叠）或运动学噪声。')
    print('  3) 足踝力矩对 COP 最敏感，它的敏感度可作为量级标尺；')
    print('     当前设置下踝力矩“漂亮的单调”只是因为力臂恒定，不是正确性证据。')
    print('  4) 若某个非零 offset 下 Spearman 明显接近 +1，那个位置就是真实 COP 的一个估计。')


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
    excluded = {_canon_load_key(k) for k in EXCLUDE_LOAD_KEYS}
    removed = [k for k in load_keys if _canon_load_key(k) in excluded]
    load_keys = [k for k in load_keys if _canon_load_key(k) not in excluded]
    if removed:
        print(f'已排除负载: {removed}')
    if len(load_keys) < 3:
        raise ValueError('参与敏感性测试的负载少于 3 个，无法判断单调性')
    print(f'参与负载: {load_keys}')

    coord_map = build_left_joint_coordinate_map(config, joint_bases=JOINT_BASES)
    if not coord_map:
        raise ValueError('未找到可统计的左腿关节坐标')

    mot_files = get_mot_files(config, base_dir)
    scaled_model = get_scaled_model(config, base_dir)
    if not os.path.exists(scaled_model):
        raise FileNotFoundError(f'找不到缩放模型: {scaled_model}')

    # 切片与“固定膝角帧”只算一次：它们与外力无关，
    # 必须在整个扫描中保持完全一致，否则就不是单变量实验。
    subject, _, pipeline_results = load_or_create_cutted_pipeline_results(
        config_path, include_xsens=False, debug=False)

    mot_by_key = {_canon_load_key(k): v for k, v in mot_files.items()}
    seg_cache = {}
    print(f'\n固定膝角帧预处理（目标 {TARGET_KNEE_ANGLE:.0f}°，容差 {ANGLE_TOLERANCE:.0f}°）:')
    for load_key in load_keys:
        key = _canon_load_key(load_key)
        segment_df = get_segment_from_results(
            pipeline_results, load_key, movement_types=MOVEMENT_TYPES)
        mot_path = mot_by_key.get(key)
        if segment_df is None or mot_path is None:
            print(f'  load={load_key}: 缺少切片或 mot，跳过')
            continue
        mot_df = read_opensim_table(mot_path)
        dropouts = load_dropout_intervals(mot_path, mot_df=mot_df)
        picked, n_total, n_dropped = pick_frames_at_knee_angle(
            mot_df, segment_df, dropout_intervals=dropouts)

        drop_txt = ''
        if dropouts:
            total = sum(t1 - t0 for t0, t1 in dropouts)
            spans = ', '.join(f'{t0:.1f}-{t1:.1f}s' for t0, t1 in dropouts[:4])
            if len(dropouts) > 4:
                spans += ', ...'
            drop_txt = (f'  | 丢帧 {len(dropouts)} 段/{total:.1f}s [{spans}]'
                        f' → 剔除 {n_dropped} 段')
        print(f'  load={load_key}: 可用段数 = {len(picked)}/{n_total}{drop_txt}')
        if picked:
            seg_cache[key] = {'segment_df': segment_df, 'picked': picked}

    if not seg_cache:
        raise ValueError(f'没有任何负载在膝角 {TARGET_KNEE_ANGLE}° 处有可用帧')

    osim_cfg = config.get('opensim_settings', {})
    baseline_insole = list(osim_cfg.get('insole_contact_point', [0.0, 0.0, 0.0]))
    baseline_bar = list(osim_cfg.get('bar_contact_point', [-0.07, 0.30, 0.0]))
    print(f'\n基线 insole_contact_point = {baseline_insole}')
    print(f'基线 bar_contact_point    = {baseline_bar}')

    try:
        cop_results = sweep_contact_point(
            config=config, base_dir=base_dir, load_keys=load_keys,
            mot_files=mot_files, scaled_model=scaled_model,
            seg_cache=seg_cache, coord_map=coord_map,
            setting_key='insole_contact_point',
            offsets=COP_AP_OFFSETS,
            tag_prefix='cop_ap',
            build_point=lambda dx: [dx, COP_VERTICAL, COP_MEDIOLATERAL],
        )
        print_sensitivity_summary(cop_results, load_keys, coord_map, 'cop_ap')

        if RUN_BAR_SWEEP:
            bar_results = sweep_contact_point(
                config=config, base_dir=base_dir, load_keys=load_keys,
                mot_files=mot_files, scaled_model=scaled_model,
                seg_cache=seg_cache, coord_map=coord_map,
                setting_key='bar_contact_point',
                offsets=BAR_AP_OFFSETS,
                tag_prefix='bar_ap',
                build_point=lambda dx: [dx, baseline_bar[1], baseline_bar[2]],
            )
            print_sensitivity_summary(bar_results, load_keys, coord_map, 'bar_ap')

    finally:
        # 恢复共享外力文件，避免后续正式流程用到被扰动的 xml/sto
        print('\n' + '=' * 80)
        print('[恢复] 用基线接触点重新生成 external_forces/ 共享文件')
        print('=' * 80)
        restore_cfg = copy.deepcopy(config)
        restore_cfg.setdefault('opensim_settings', {})
        restore_cfg['opensim_settings']['insole_contact_point'] = baseline_insole
        restore_cfg['opensim_settings']['bar_contact_point'] = baseline_bar
        for load_key, mot_path in mot_files.items():
            generate_external_loads(
                config=restore_cfg, base_dir=base_dir, load_key=load_key,
                mot_path=mot_path, Mb=MB, verbose=False)
        print('已恢复。敏感性结果单独保存在 opensim/sensitivity/ 下，')
        print('正式 inverse_dynamics/ 输出未被修改。')


if __name__ == '__main__':
    main()