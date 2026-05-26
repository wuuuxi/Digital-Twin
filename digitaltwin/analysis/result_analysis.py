"""
result_analysis.py

OpenSim 结果文件与标准运动切片的对齐/统计工具。

用途：
  - 复用 example_data_analysis.py / MultiLoadPipeline 的标准切片逻辑；
  - 将 inverse_dynamics / muscle_analysis 等 OpenSim .sto/.mot 结果
    按 time 插值到标准切片后的 upward/downward 阶段；
  - 打印每个 load、每个关节坐标的均值 / 平均绝对值 / RMS 等统计量。

说明：
  inverse_dynamics 与 muscle_analysis 输出的 time 通常与 Xsens -> mot 的时间轴一致，
  因此可以直接用标准切片数据中的 time 去插值 OpenSim 结果。
"""
import os
import numpy as np
import pandas as pd

from digitaltwin.subject import Subject
from digitaltwin.pipeline import MultiLoadPipeline
from digitaltwin.analysis.alignment import DataAligner


def _canonical_load_key(value):
    """
    统一 load key 的字符串格式。

    CSV 读写后，原来的 "20" 可能变成 20.0 / "20.0"。
    这里统一成 "20"，避免缓存切片数据时 load key 对不上。
    """
    try:
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f'{f:g}'
    except Exception:
        return str(value)


# ============================================================
#  OpenSim 表格读取
# ============================================================

def resolve_optional_extension_path(path):
    """
    解析 OpenSim .sto / .mot / 无后缀同格式文件路径。

    按顺序尝试：
      1. 原始路径
      2. 去掉后缀后的路径
      3. 补 .sto
      4. 补 .mot
      5. 同目录下 basename 相同、后缀为 '', '.sto', '.mot' 的文件
    """
    if path is None:
        return None

    path = os.path.normpath(path)
    folder = os.path.dirname(path)
    base = os.path.basename(path)
    root, ext = os.path.splitext(path)

    candidates = [path]
    if ext:
        candidates.append(root)
    candidates.extend([root + '.sto', root + '.mot'])

    seen = set()
    for p in candidates:
        p = os.path.normpath(p)
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p

    if folder and os.path.isdir(folder):
        target_stem = os.path.splitext(base)[0]
        for fname in os.listdir(folder):
            fstem, fext = os.path.splitext(fname)
            if fstem == target_stem and fext.lower() in ('', '.sto', '.mot'):
                candidate = os.path.join(folder, fname)
                if os.path.isfile(candidate):
                    return candidate

    return None


def read_opensim_table(path):
    """
    读取 OpenSim .sto / .mot / 无后缀同格式文件为 pandas.DataFrame。
    """
    resolved = resolve_optional_extension_path(path)
    if resolved is None:
        return None

    with open(resolved, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == 'endheader':
            header_start = i + 1
            break

    if header_start is None:
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('time'):
                header_start = i
                break

    if header_start is None:
        raise ValueError(f'无法识别 OpenSim 表头: {resolved}')

    from io import StringIO
    return pd.read_csv(
        StringIO(''.join(lines[header_start:])),
        sep=r'\s+',
        engine='python',
    )


# ============================================================
#  标准数据处理 / 运动切片
# ============================================================

def run_standard_data_pipeline(config_path, include_xsens=False, debug=True):
    """
    复用 example_data_analysis.py 中的标准处理流程：

      Subject -> MultiLoadPipeline.run() -> DataAligner.cut_aligned_data()

    Returns
    -------
    subject : Subject
    pipeline : MultiLoadPipeline
    results : dict
    """
    subject = Subject(config_path)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = debug
    results = pipeline.run(include_xsens=include_xsens)
    return subject, pipeline, results


def _segments_to_pipeline_results(cutted_df):
    """
    将切片缓存表还原成 summarize_* 函数需要的 pipeline_results 结构。

    只需要每个 load 的 cutted_data，因此不还原 robot/emg/xsens 等完整字段。
    """
    if cutted_df is None or len(cutted_df) == 0:
        return {}

    load_col = None
    for c in ('load_weight', 'load', 'load_value'):
        if c in cutted_df.columns:
            load_col = c
            break

    if load_col is None:
        return {'all': {'cutted_data': cutted_df}}

    results = {}
    for load_key, df_load in cutted_df.groupby(load_col):
        key = _canonical_load_key(load_key)
        results[key] = {
            'cutted_data': df_load.reset_index(drop=True),
        }
    return results


def _collect_cutted_from_pipeline_results(pipeline_results):
    """
    将 MultiLoadPipeline.run() 返回的 results 中的 cutted_data 合并成一个表。
    """
    frames = []
    for load_key, result in pipeline_results.items():
        cd = result.get('cutted_data')
        if cd is None:
            continue
        if isinstance(cd, list):
            if not cd:
                continue
            cd = pd.concat(cd, ignore_index=True)
        if cd is None or len(cd) == 0:
            continue

        df = cd.copy()
        if 'load_weight' not in df.columns:
            df['load_weight'] = str(load_key)
        if 'load_value' not in df.columns:
            try:
                df['load_value'] = float(load_key)
            except Exception:
                pass
        frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_or_create_cutted_pipeline_results(config_path,
                                           include_xsens=False,
                                           debug=True,
                                           force_rebuild=False,
                                           cache_name='cutted_data.csv'):
    """
    快速获得标准运动切片数据，并缓存到 CSV。

    优先级：
      1. 若 result_folder/cutted_data.csv 存在，直接读取；
      2. 若 result_folder/aligned_data.csv 存在，则直接对 aligned_data 重新切片并保存 cutted_data.csv；
      3. 若两者都不存在，才运行完整 MultiLoadPipeline.run()，然后保存 cutted_data.csv。

    这样像 example_inverse_dynamics.py 这类只需要切片时间点的脚本，
    后续运行时不必反复重新读取/处理 robot + EMG。
    """
    subject = Subject(config_path)
    result_folder = subject.result_folder
    os.makedirs(result_folder, exist_ok=True)

    cutted_cache_path = os.path.join(result_folder, cache_name)
    aligned_cache_path = os.path.join(result_folder, 'aligned_data.csv')

    if os.path.exists(cutted_cache_path) and not force_rebuild:
        if debug:
            print(f'[cache] 读取切片缓存: {cutted_cache_path}')
        cutted_df = pd.read_csv(cutted_cache_path)
        return subject, None, _segments_to_pipeline_results(cutted_df)

    # 如果已有 aligned_data.csv，不再跑完整 pipeline，只重新做标准切片
    if os.path.exists(aligned_cache_path) and not force_rebuild:
        if debug:
            print(f'[cache] 读取 aligned_data 并重新切片: {aligned_cache_path}')
        aligned_df = pd.read_csv(aligned_cache_path)

        load_col = None
        for c in ('load_weight', 'load', 'load_value'):
            if c in aligned_df.columns:
                load_col = c
                break

        aligner = DataAligner()
        frames = []

        if load_col is None:
            groups = [('all', aligned_df)]
        else:
            groups = list(aligned_df.groupby(load_col))

        for load_key, df_load in groups:
            # DataAligner 内部使用位置索引，因此每个 load 必须 reset_index
            df_load = df_load.reset_index(drop=True)
            cd = aligner.cut_aligned_data(df_load)
            if cd is None:
                continue
            if isinstance(cd, list):
                if not cd:
                    continue
                cd = pd.concat(cd, ignore_index=True)
            if len(cd) == 0:
                continue

            cd = cd.copy()
            key = _canonical_load_key(load_key)
            cd['load_weight'] = key
            if 'load_value' not in cd.columns:
                try:
                    cd['load_value'] = float(key)
                except Exception:
                    pass
            frames.append(cd)

        if frames:
            cutted_df = pd.concat(frames, ignore_index=True)
            cutted_df.to_csv(cutted_cache_path, index=False)
            if debug:
                print(f'[cache] 切片缓存已保存: {cutted_cache_path}')
            return subject, None, _segments_to_pipeline_results(cutted_df)

    # 最后才运行完整 pipeline
    if debug:
        print('[cache] 未找到可用缓存，运行完整 MultiLoadPipeline...')
    subject, pipeline, results = run_standard_data_pipeline(
        config_path,
        include_xsens=include_xsens,
        debug=debug,
    )
    cutted_df = _collect_cutted_from_pipeline_results(results)
    if cutted_df is not None and len(cutted_df) > 0:
        cutted_df.to_csv(cutted_cache_path, index=False)
        if debug:
            print(f'[cache] 切片缓存已保存: {cutted_cache_path}')

    return subject, pipeline, results


def get_segment_from_results(pipeline_results, load_key,
                             movement_types=('upward',)):
    """
    从 MultiLoadPipeline.run() 的结果中取出指定 load 的运动切片。

    Parameters
    ----------
    pipeline_results : dict
        MultiLoadPipeline.run() 的返回值。
    load_key : str
        负载 key。
    movement_types : tuple/list/None
        例如 ('upward',), ('downward',), ('upward', 'downward')。
        None 表示不过滤 movement_type。

    Returns
    -------
    pd.DataFrame or None
    """
    query_key = _canonical_load_key(load_key)
    result = pipeline_results.get(query_key)
    if result is None:
        for k, v in pipeline_results.items():
            if _canonical_load_key(k) == query_key:
                result = v
                break

    if result is None:
        return None

    cutted = result.get('cutted_data')
    if cutted is None:
        return None

    if isinstance(cutted, list):
        if not cutted:
            return None
        cutted = pd.concat(cutted, ignore_index=True)

    if cutted is None or len(cutted) == 0:
        return None

    df = cutted.copy()

    if movement_types is not None and 'movement_type' in df.columns:
        df = df[df['movement_type'].isin(list(movement_types))].copy()

    if len(df) == 0:
        return None

    if 'time' not in df.columns:
        return None

    return df


def interpolate_column_to_segment(table_df, segment_df, value_col,
                                  time_col='time'):
    """
    将 OpenSim 结果表中某列按 time 插值到标准切片 segment_df 的时间点。

    inverse_dynamics / muscle_analysis 的 time 与 Xsens mot 时间轴一致时，
    直接使用该函数即可。

    Returns
    -------
    np.ndarray or None
    """
    if table_df is None or segment_df is None:
        return None
    if time_col not in table_df.columns or time_col not in segment_df.columns:
        return None
    if value_col not in table_df.columns:
        return None

    src_t = table_df[time_col].values.astype(float)
    src_v = table_df[value_col].values.astype(float)
    dst_t = segment_df[time_col].values.astype(float)

    valid_src = np.isfinite(src_t) & np.isfinite(src_v)
    valid_dst = np.isfinite(dst_t)
    if valid_src.sum() < 2 or valid_dst.sum() == 0:
        return None

    out = np.full(len(dst_t), np.nan, dtype=float)
    out[valid_dst] = np.interp(
        dst_t[valid_dst],
        src_t[valid_src],
        src_v[valid_src],
        left=src_v[valid_src][0],
        right=src_v[valid_src][-1],
    )
    return out


# ============================================================
#  ID 结果路径 / 坐标列识别
# ============================================================

def get_load_keys(config, load_keys=None):
    """获取负载 key 列表，统一转为 str。"""
    if load_keys is None:
        return [
            _canonical_load_key(k)
            for k in config.get('modeling_file', {}).get('data', {}).keys()
        ]
    return [_canonical_load_key(k) for k in load_keys]


def build_left_joint_coordinate_map(config, joint_bases=None):
    """
    从 opensim_settings.muscle_analysis_coordinates 构建左侧关节坐标映射。

    Returns
    -------
    dict
        {joint_base: coord_l}
        例如 {'knee_angle': 'knee_angle_l'}
    """
    coords = config.get('opensim_settings', {}).get(
        'muscle_analysis_coordinates', [])
    out = {}
    for coord in coords:
        if coord.endswith('_l'):
            base = coord[:-2]
            if joint_bases is None or base in joint_bases:
                out[base] = coord
    return out


def get_inverse_dynamics_path(config, base_dir, load_key):
    """默认 inverse_dynamics.sto 路径。"""
    label = config['experiment_label']
    return os.path.join(
        base_dir, 'result', label,
        'opensim', 'inverse_dynamics', str(load_key),
        'inverse_dynamics.sto',
    )


def find_id_moment_column(id_df, coord):
    """
    在 inverse_dynamics 输出中查找指定坐标的 moment 列。

    兼容：
      knee_angle_l_moment
      knee_angle_l/moment
      以及大小写差异
    """
    if id_df is None:
        return None

    coord_l = coord.lower()
    exact_candidates = [
        f'{coord}_moment',
        f'{coord}/moment',
        f'{coord}.moment',
    ]
    lower_map = {c.lower(): c for c in id_df.columns}
    for cand in exact_candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for c in id_df.columns:
        cl = c.lower().replace('/', '_').replace('.', '_')
        if coord_l in cl and 'moment' in cl:
            return c

    return None


def _stat(values, statistic):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    statistic = statistic.lower()
    if statistic == 'mean':
        return float(np.nanmean(values))
    if statistic == 'mean_abs':
        return float(np.nanmean(np.abs(values)))
    if statistic == 'rms':
        return float(np.sqrt(np.nanmean(values ** 2)))

    raise ValueError(f'未知 statistic: {statistic}')


def summarize_inverse_dynamics_moments(config, base_dir, pipeline_results,
                                       load_keys=None,
                                       coordinates=None,
                                       movement_types=('upward',),
                                       statistic='mean'):
    """
    统计每个 load、每个关节坐标在标准运动切片阶段内的 ID 力矩。

    Parameters
    ----------
    config : dict
    base_dir : str
    pipeline_results : dict
        MultiLoadPipeline.run() 的结果，用于取得标准切片时间范围。
    load_keys : list[str] or None
    coordinates : dict/list/None
        - dict: {joint_base: coord}
        - list: [coord1, coord2, ...]，joint_base 自动由 coord 去掉 _l/_r
        - None: 使用 build_left_joint_coordinate_map(config)
    movement_types : tuple/list/None
    statistic : {'mean', 'mean_abs', 'rms'}

    Returns
    -------
    dict
        {joint_base: {load_key: value}}
    """
    load_keys = get_load_keys(config, load_keys)

    if coordinates is None:
        coord_map = build_left_joint_coordinate_map(config)
    elif isinstance(coordinates, dict):
        coord_map = coordinates
    else:
        coord_map = {}
        for coord in coordinates:
            coord = str(coord)
            if coord.endswith('_l') or coord.endswith('_r'):
                base = coord[:-2]
            else:
                base = coord
            coord_map[base] = coord

    summary = {joint_base: {} for joint_base in coord_map.keys()}

    for load_key in load_keys:
        segment_df = get_segment_from_results(
            pipeline_results, load_key,
            movement_types=movement_types,
        )
        if segment_df is None or len(segment_df) == 0:
            available = ', '.join(sorted(str(k) for k in pipeline_results.keys()))
            print(f'[WARN] load={load_key}: 无标准切片数据 {movement_types}；'
                  f'可用 load keys=[{available}]')
            continue

        id_path = get_inverse_dynamics_path(config, base_dir, load_key)
        id_df = read_opensim_table(id_path)
        if id_df is None or 'time' not in id_df.columns:
            print(f'[MISS] load={load_key}: inverse_dynamics 文件不可读: {id_path}')
            continue

        for joint_base, coord in coord_map.items():
            id_col = find_id_moment_column(id_df, coord)
            if id_col is None:
                print(f'[WARN] load={load_key}: ID 文件中未找到 {coord} moment 列')
                summary[joint_base][str(load_key)] = None
                continue

            values = interpolate_column_to_segment(id_df, segment_df, id_col)
            summary[joint_base][str(load_key)] = _stat(values, statistic)

    return summary


def print_summary_table(title, summary, load_keys, unit='N·m', note=None):
    """
    打印 joint × load 表格。
    """
    if not summary:
        return

    load_keys = [str(k) for k in load_keys]

    print('\n' + '=' * 60)
    print(title)
    print('=' * 60)

    header = f'{"joint":<20s}' + ''.join(
        f'{str(lk) + " kg":>14s}' for lk in load_keys
    )
    print(header)
    print('-' * len(header))

    for joint_base, load_values in summary.items():
        row = f'{joint_base:<20s}'
        for lk in load_keys:
            v = load_values.get(str(lk))
            if v is None or not np.isfinite(v):
                row += f'{"N/A":>14s}'
            else:
                row += f'{v:>14.4f}'
        print(row)

    print(f'\n单位: {unit}')
    if note:
        print(note)