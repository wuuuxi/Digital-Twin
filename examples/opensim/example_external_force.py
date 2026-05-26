"""
example_external_force.py

单独生成 / 读取 external force，并统计标准 upward 切片阶段内的外力平均值。

流程：
  1. 读取 config；
  2. 找到每个 load 对应的 OpenSim .mot；
  3. 调用 generate_external_loads() 生成：
       result/{experiment_label}/opensim/external_forces/{load_key}/
         bar_force_{load_key}.sto
         bar_loads_{load_key}.xml
  4. 复用标准切片逻辑读取 upward 阶段时间点；
  5. 对 external force .sto 中的每个外力前缀分别统计 upward 阶段均值：
       - mean_vx / mean_vy / mean_vz
       - mean_abs_vx / mean_abs_vy / mean_abs_vz
       - mean_mag = mean(sqrt(vx^2 + vy^2 + vz^2))

说明：
  external force 可能包含多个 force object，例如：
    - bar_force
    - grf_l
    - grf_r

  本脚本会自动从 .sto 列名中识别 *_vx, *_vy, *_vz 三元组。
"""
import os
import json
import numpy as np

from digitaltwin.osim.mot_pipeline import get_mot_files
from digitaltwin.osim.external_forces import (
    generate_external_loads,
    get_ext_forces_dir,
)
from digitaltwin.analysis.result_analysis import (
    load_or_create_cutted_pipeline_results,
    get_segment_from_results,
    read_opensim_table,
    interpolate_column_to_segment,
    get_load_keys,
)


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'

# None = 全部；也可以指定，如 ['20', '38', '56']
LOAD_KEYS = None

# 只统计标准 upward 阶段；也可改为 ('downward',) 或 ('upward', 'downward')
MOVEMENT_TYPES = ('upward',)

# 外力设置
MB = 20.0
REGENERATE_EXTERNAL_FORCES = True

# 切片缓存设置
# False = 优先读取 cutted_data.csv；没有则用 aligned_data.csv 重新切片；
#         都没有才运行完整 MultiLoadPipeline。
# True  = 强制重新生成 cutted_data.csv。
FORCE_REBUILD_CUTTED_CACHE = False
CUTTED_CACHE_NAME = 'cutted_data.csv'

# 只打印平均 magnitude 大于该阈值的 external force。
# 设为 0 会打印所有识别到的 force，包括全 0 的 GRF。
MIN_MEAN_MAG_TO_PRINT = 1e-6


# ============================================================
#  路径工具
# ============================================================

def get_base_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


def get_external_force_sto_path(config, base_dir, load_key):
    return os.path.join(
        get_ext_forces_dir(config, base_dir, load_key),
        f'bar_force_{load_key}.sto',
    )


# ============================================================
#  external force 识别与统计
# ============================================================

def discover_force_prefixes(ext_df):
    """
    从 external force .sto 列名中识别外力前缀。

    例如：
      bar_force_vx, bar_force_vy, bar_force_vz -> bar_force
      grf_l_vx, grf_l_vy, grf_l_vz             -> grf_l
    """
    prefixes = []
    cols = set(ext_df.columns)

    for col in ext_df.columns:
        if not col.endswith('_vx'):
            continue
        prefix = col[:-3]  # remove "_vx"
        if (
            f'{prefix}_vx' in cols and
            f'{prefix}_vy' in cols and
            f'{prefix}_vz' in cols
        ):
            prefixes.append(prefix)

    return prefixes


def summarize_external_forces_for_load(config, base_dir, load_key,
                                       mot_path, pipeline_results):
    """
    生成 / 读取一个 load 的 external force，并统计 upward 阶段均值。

    Returns
    -------
    list[dict]
        每个 external force prefix 一行统计。
    """
    if REGENERATE_EXTERNAL_FORCES:
        generate_external_loads(
            config=config,
            base_dir=base_dir,
            load_key=load_key,
            mot_path=mot_path,
            Mb=MB,
            verbose=True,
        )

    segment_df = get_segment_from_results(
        pipeline_results,
        load_key,
        movement_types=MOVEMENT_TYPES,
    )
    if segment_df is None or len(segment_df) == 0:
        print(f'[WARN] load={load_key}: 无标准切片数据 {MOVEMENT_TYPES}')
        return []

    sto_path = get_external_force_sto_path(config, base_dir, load_key)
    ext_df = read_opensim_table(sto_path)
    if ext_df is None or 'time' not in ext_df.columns:
        print(f'[MISS] load={load_key}: external force sto 不可读: {sto_path}')
        return []

    prefixes = discover_force_prefixes(ext_df)
    if not prefixes:
        print(f'[WARN] load={load_key}: 未识别到 *_vx/_vy/_vz external force 列')
        return []

    rows = []
    for prefix in prefixes:
        vx = interpolate_column_to_segment(ext_df, segment_df, f'{prefix}_vx')
        vy = interpolate_column_to_segment(ext_df, segment_df, f'{prefix}_vy')
        vz = interpolate_column_to_segment(ext_df, segment_df, f'{prefix}_vz')

        if vx is None or vy is None or vz is None:
            continue

        valid = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
        if valid.sum() == 0:
            continue

        vx_v = vx[valid]
        vy_v = vy[valid]
        vz_v = vz[valid]
        mag = np.sqrt(vx_v ** 2 + vy_v ** 2 + vz_v ** 2)

        row = {
            'load_key': str(load_key),
            'force': prefix,
            'n': int(valid.sum()),
            'mean_vx': float(np.nanmean(vx_v)),
            'mean_vy': float(np.nanmean(vy_v)),
            'mean_vz': float(np.nanmean(vz_v)),
            'mean_abs_vx': float(np.nanmean(np.abs(vx_v))),
            'mean_abs_vy': float(np.nanmean(np.abs(vy_v))),
            'mean_abs_vz': float(np.nanmean(np.abs(vz_v))),
            'mean_mag': float(np.nanmean(mag)),
        }
        rows.append(row)

    return rows


def print_external_force_summary(rows):
    """
    按 external force 名称分别打印 upward 均值表。

    例如分别打印：
      - bar_force 一个表
      - grf_l 一个表
      - grf_r 一个表
    """
    if not rows:
        print('\n无 external force 统计结果。')
        return

    rows = [
        r for r in rows
        if r['mean_mag'] >= MIN_MEAN_MAG_TO_PRINT
    ]

    if not rows:
        print('\n所有 external force 的 mean_mag 均低于打印阈值。')
        return

    def load_sort_key(load_key):
        try:
            return float(load_key)
        except Exception:
            return 999999.0

    force_names = sorted(set(r['force'] for r in rows))

    for force_name in force_names:
        force_rows = [r for r in rows if r['force'] == force_name]
        force_rows = sorted(force_rows, key=lambda r: load_sort_key(r['load_key']))

        print('\n' + '=' * 104)
        print(f'External force 均值: {force_name}（标准切片: {MOVEMENT_TYPES}）')
        print('=' * 104)
        print(
            f'{"load":>8s}  {"n":>7s}  '
            f'{"mean_vx":>12s}  {"mean_vy":>12s}  {"mean_vz":>12s}  '
            f'{"mean|vx|":>12s}  {"mean|vy|":>12s}  {"mean|vz|":>12s}  '
            f'{"mean_mag":>12s}'
        )
        print('-' * 104)

        for r in force_rows:
            print(
                f'{r["load_key"]:>8s}  {r["n"]:>7d}  '
                f'{r["mean_vx"]:>12.3f}  {r["mean_vy"]:>12.3f}  {r["mean_vz"]:>12.3f}  '
                f'{r["mean_abs_vx"]:>12.3f}  {r["mean_abs_vy"]:>12.3f}  {r["mean_abs_vz"]:>12.3f}  '
                f'{r["mean_mag"]:>12.3f}'
            )

    print('\n单位: N')
    print('说明:')
    print('  - 每个 external force 单独一个表，例如 bar_force / grf_l / grf_r。')
    print('  - mean_v* 是有符号分量均值。')
    print('  - mean|v*| 是分量绝对值均值。')
    print('  - mean_mag 是三维外力向量模长的均值。')
    print('  - OpenSim 中 y 轴向上；bar_force_vy 通常为负，代表杆件向下作用力。')


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

    # 1) 读取或生成标准切片缓存
    subject, pipeline, pipeline_results = load_or_create_cutted_pipeline_results(
        config_path,
        include_xsens=False,
        debug=True,
        force_rebuild=FORCE_REBUILD_CUTTED_CACHE,
        cache_name=CUTTED_CACHE_NAME,
    )

    # 2) 找到 .mot 文件
    mot_files = get_mot_files(config, base_dir)
    if not mot_files:
        raise FileNotFoundError('未找到 mot 文件，请先运行 Xsens -> MOT 转换。')

    # 3) 对每个 load 生成 external force 并统计 upward 均值
    all_rows = []
    for load_key in load_keys:
        if str(load_key) not in {str(k) for k in mot_files.keys()}:
            print(f'[MISS] load={load_key}: 找不到对应 mot 文件')
            continue

        # 兼容 mot_files 的 key 是 int / str
        mot_path = None
        for k, v in mot_files.items():
            if str(k) == str(load_key):
                mot_path = v
                break

        print(f'\n{"=" * 60}')
        print(f'load={load_key}')
        print('=' * 60)
        print(f'mot: {mot_path}')

        rows = summarize_external_forces_for_load(
            config=subject.config,
            base_dir=base_dir,
            load_key=str(load_key),
            mot_path=mot_path,
            pipeline_results=pipeline_results,
        )
        all_rows.extend(rows)

    # 4) 打印汇总表
    print_external_force_summary(all_rows)


if __name__ == '__main__':
    main()