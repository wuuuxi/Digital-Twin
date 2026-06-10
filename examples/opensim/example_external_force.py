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
  5. 对 external force 分别统计 upward 阶段均值：
       - bar_force 仍从 external force .sto 中按 time 插值到标准切片；
       - grf_l / grf_r 改为与 example_data_analysis.py 完全相同：
         先在 MultiLoadPipeline 中把鞋垫数据插值到 aligned_data 的 time 轴，
         生成 grf_l / grf_r，再进行标准切片，统计时直接读取切片后的列。
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
import matplotlib.pyplot as plt

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
# False = 优先读取 cutted_data_with_grf.csv；
#         没有则运行完整 MultiLoadPipeline，并按 example_data_analysis.py
#         的方式将鞋垫 GRF 插值到 aligned_data 的 time 轴后再切片。
# True  = 强制重新生成 cutted_data_with_grf.csv。
FORCE_REBUILD_CUTTED_CACHE = False
INCLUDE_INSOLE_GRF = True

# 鞋垫时间戳处理：默认 True。
# True  = 使用 info.csv measurement_date + robot_file 第一帧时间修正鞋垫时间；
# False = 退回鞋垫文件原始相对时间。
USE_INSOLE_INFO_TIMESTAMP = True

CUTTED_CACHE_NAME = (
    'cutted_data_with_grf_info_time.csv'
    if USE_INSOLE_INFO_TIMESTAMP else
    'cutted_data_with_grf_raw_time.csv'
)

# 只打印平均 magnitude 大于该阈值的 external force。
# 设为 0 会打印所有识别到的 force，包括全 0 的 GRF。
MIN_MEAN_MAG_TO_PRINT = 1e-6

# 是否将每个 external force 的 y 方向值画成 Height-Vy 散点图。
PLOT_Y_SCATTER = True
SCATTER_POINT_SIZE = 8
SCATTER_ALPHA = 0.45


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
            use_insole_info_timestamp=USE_INSOLE_INFO_TIMESTAMP,
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
        # grf_l / grf_r 与 example_data_analysis.py 保持一致：
        # 使用 MultiLoadPipeline 注入到 aligned_data 后再切片得到的列，
        # 而不是使用 external force .sto 中已经重采样到 mot_times 的列。
        if prefix in ('grf_l', 'grf_r') and prefix in segment_df.columns:
            vy = segment_df[prefix].values.astype(float)
            vx = np.zeros_like(vy)
            vz = np.zeros_like(vy)
        else:
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

        # 保存绘图用数据：每个 external force 的 y 方向值 vs 高度
        if 'pos_l' in segment_df.columns:
            height = segment_df['pos_l'].values.astype(float)
            row['_height'] = height[valid]
            row['_vy'] = vy_v

        # bar_force 的来源项：原始机器人 force_l / force_r
        # 注意这里是标准 upward 切片内的原始 force_l / force_r 均值，
        # 尚未加 Mb*g 和 Mb*avg_acc，也没有取 OpenSim y 方向负号。
        if prefix == 'bar_force':
            if 'force_l' in segment_df.columns:
                force_l_raw = segment_df['force_l'].values.astype(float)[valid]
                row['mean_force_l_raw'] = float(np.nanmean(force_l_raw))
                row['mean_abs_force_l_raw'] = float(np.nanmean(np.abs(force_l_raw)))
            else:
                row['mean_force_l_raw'] = None
                row['mean_abs_force_l_raw'] = None

            if 'force_r' in segment_df.columns:
                force_r_raw = segment_df['force_r'].values.astype(float)[valid]
                row['mean_force_r_raw'] = float(np.nanmean(force_r_raw))
                row['mean_abs_force_r_raw'] = float(np.nanmean(np.abs(force_r_raw)))
            else:
                row['mean_force_r_raw'] = None
                row['mean_abs_force_r_raw'] = None

            if ('force_l' in segment_df.columns and
                    'force_r' in segment_df.columns):
                force_sum_raw = force_l_raw + force_r_raw
                row['mean_force_sum_raw'] = float(np.nanmean(force_sum_raw))
                row['mean_abs_force_sum_raw'] = float(np.nanmean(np.abs(force_sum_raw)))
            else:
                row['mean_force_sum_raw'] = None
                row['mean_abs_force_sum_raw'] = None

        rows.append(row)

    return rows


def _fmt_value(v):
    """表格打印辅助。"""
    if v is None:
        return 'N/A'
    try:
        if not np.isfinite(v):
            return 'N/A'
        return f'{v:.3f}'
    except Exception:
        return 'N/A'


def print_external_force_summary(rows):
    """
    按 external force 名称分别打印 upward 均值表。

    例如分别打印：
      - bar_force 一个表
      - grf_l 一个表
      - grf_r 一个表

    对 bar_force，额外打印原始机器人 force_l / force_r 的 upward 均值。
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

        if force_name == 'bar_force':
            width = 166
            print('\n' + '=' * width)
            print(f'External force 均值: {force_name}（标准切片: {MOVEMENT_TYPES}）')
            print('=' * width)
            print(
                f'{"load":>8s}  {"n":>7s}  '
                f'{"raw force_l":>12s}  {"raw force_r":>12s}  {"raw sum":>12s}  '
                f'{"mean_vx":>12s}  {"mean_vy":>12s}  {"mean_vz":>12s}  '
                f'{"mean|vx|":>12s}  {"mean|vy|":>12s}  {"mean|vz|":>12s}  '
                f'{"mean_mag":>12s}'
            )
            print('-' * width)

            for r in force_rows:
                print(
                    f'{r["load_key"]:>8s}  {r["n"]:>7d}  '
                    f'{_fmt_value(r.get("mean_force_l_raw")):>12s}  '
                    f'{_fmt_value(r.get("mean_force_r_raw")):>12s}  '
                    f'{_fmt_value(r.get("mean_force_sum_raw")):>12s}  '
                    f'{r["mean_vx"]:>12.3f}  {r["mean_vy"]:>12.3f}  {r["mean_vz"]:>12.3f}  '
                    f'{r["mean_abs_vx"]:>12.3f}  {r["mean_abs_vy"]:>12.3f}  {r["mean_abs_vz"]:>12.3f}  '
                    f'{r["mean_mag"]:>12.3f}'
                )
        else:
            width = 104
            print('\n' + '=' * width)
            print(f'External force 均值: {force_name}（标准切片: {MOVEMENT_TYPES}）')
            print('=' * width)
            print(
                f'{"load":>8s}  {"n":>7s}  '
                f'{"mean_vx":>12s}  {"mean_vy":>12s}  {"mean_vz":>12s}  '
                f'{"mean|vx|":>12s}  {"mean|vy|":>12s}  {"mean|vz|":>12s}  '
                f'{"mean_mag":>12s}'
            )
            print('-' * width)

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
    print('  - bar_force 表中的 raw force_l / raw force_r 是 upward 切片内原始机器人力均值。')
    print('  - mean_v* 是有符号分量均值。')
    print('  - mean|v*| 是分量绝对值均值。')
    print('  - mean_mag 是三维外力向量模长的均值。')
    print('  - OpenSim 中 y 轴向上；bar_force_vy 通常为负，代表杆件向下作用力。')


def plot_external_force_y_scatter(rows, result_folder):
    """
    将每个 external force 的 y 方向值画成横轴为高度的散点图。

    每个 force 单独保存一张图：
      result/{experiment_label}/external_force_y_scatter/{force}_vy_vs_height.png
    """
    if not PLOT_Y_SCATTER or not rows:
        return

    plot_rows = [
        r for r in rows
        if '_height' in r and '_vy' in r and r['mean_mag'] >= MIN_MEAN_MAG_TO_PRINT
    ]
    if not plot_rows:
        print('[plot] 无可绘制的 external force y scatter 数据')
        return

    save_dir = os.path.join(result_folder, 'external_force_y_scatter')
    os.makedirs(save_dir, exist_ok=True)

    def load_sort_key(load_key):
        try:
            return float(load_key)
        except Exception:
            return 999999.0

    force_names = sorted(set(r['force'] for r in plot_rows))
    for force_name in force_names:
        force_rows = [r for r in plot_rows if r['force'] == force_name]
        force_rows = sorted(force_rows, key=lambda r: load_sort_key(r['load_key']))

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for r in force_rows:
            ax.scatter(
                r['_height'], r['_vy'],
                s=SCATTER_POINT_SIZE,
                alpha=SCATTER_ALPHA,
                label=f'{r["load_key"]} kg',
            )

        ax.set_xlabel('Height / pos_l (m)')
        ax.set_ylabel(f'{force_name}_vy (N)')
        ax.set_title(f'{force_name}: Vy vs Height ({MOVEMENT_TYPES})')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()

        out_path = os.path.join(save_dir, f'{force_name}_vy_vs_height.png')
        fig.savefig(out_path, dpi=200)
        # plt.close(fig)
        print(f'[plot] 已保存: {out_path}')


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
        include_insole=INCLUDE_INSOLE_GRF,
        use_insole_info_timestamp=USE_INSOLE_INFO_TIMESTAMP,
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

    # 5) 绘制每个 external force 的 y 方向值 vs 高度散点图
    plot_external_force_y_scatter(all_rows, subject.result_folder)

    plt.show()


if __name__ == '__main__':
    main()