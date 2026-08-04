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
from digitaltwin.utils.logger import beauty_print


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

def run_standard_data_pipeline(config_path, include_xsens=False,
                               include_insole=False,
                               use_insole_info_timestamp=True,
                               debug=True):
    """
    复用 example_data_analysis.py 中的标准处理流程：

      Subject -> MultiLoadPipeline.run() -> DataAligner.cut_aligned_data()

    Parameters
    ----------
    include_xsens : bool
        是否注入 Xsens 数据。
    include_insole : bool
        是否按 example_data_analysis.py / MultiLoadPipeline 的方式注入
        鞋垫 GRF，即将 insole_file_l / insole_file_r 插值到 aligned_data
        的 time 轴，生成 grf_l / grf_r，再进行标准切片。

    Returns
    -------
    subject : Subject
    pipeline : MultiLoadPipeline
    results : dict
    """
    subject = Subject(config_path)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = debug
    results = pipeline.run(
        include_xsens=include_xsens,
        include_insole=include_insole,
        use_insole_info_timestamp=use_insole_info_timestamp,
    )
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
                                           include_insole=False,
                                           use_insole_info_timestamp=True,
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

    # 如果已有 aligned_data.csv，不再跑完整 pipeline，只重新做标准切片。
    # 但当需要鞋垫 GRF 且缓存里没有 grf_l / grf_r 时，必须重新跑 pipeline，
    # 否则无法补回鞋垫列。
    if os.path.exists(aligned_cache_path) and not force_rebuild:
        if debug:
            print(f'[cache] 读取 aligned_data 并重新切片: {aligned_cache_path}')
        aligned_df = pd.read_csv(aligned_cache_path)

        can_use_aligned_cache = True
        if include_insole and not {'grf_l', 'grf_r'}.issubset(aligned_df.columns):
            can_use_aligned_cache = False
            if debug:
                print('[cache] aligned_data 缺少 grf_l/grf_r，改为运行完整 MultiLoadPipeline...')

        if can_use_aligned_cache:
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
        include_insole=include_insole,
        use_insole_info_timestamp=use_insole_info_timestamp,
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


def find_force_windows(time, force, min_force=None, force_frac=0.3,
                       min_duration=0.5, merge_gap=1.0):
    '''
    按「力超过阈值的连续区间」切段。

    用于等长（isometric）试次：杆不动，vel_l 几乎恒为 0，
    DataAligner.cut_aligned_data 靠速度过零点切不出任何有效片段。
    但发力与不发力在力信号上是分得很开的，所以改用力阈值。

    Parameters
    ----------
    time, force : array-like
        同一时间轴上的时间与力（通常是 force_l + force_r）。
    min_force : float, optional
        给定时直接用它作为绝对阈值 (N)。
    force_frac : float
        未给 min_force 时，阈值 = force_frac x 力的 95 分位数。
        用分位数而不是最大值，是为了不被单帧尖刺拉高阈值。
    min_duration : float
        短于此时长的区间丢弃（s），滤掉碰一下、调整姿势等杂帧。
    merge_gap : float
        相邻区间间隔小于此值时合并（s）。力在阈值附近抖动
        会把一次发力切碎，不合并就会得到几十个碎片。

    Returns
    -------
    list[tuple[float, float]]
        [(t0, t1), ...]，按时间升序。没有符合条件的区间时返回 []。
    '''
    time = np.asarray(time, dtype=float)
    force = np.asarray(force, dtype=float)
    ok = np.isfinite(time) & np.isfinite(force)
    if int(ok.sum()) < 10:
        return []

    t = time[ok]
    f = force[ok]
    order = np.argsort(t)
    t, f = t[order], f[order]

    if min_force is None:
        thr = float(force_frac) * float(np.nanpercentile(f, 95))
    else:
        thr = float(min_force)
    if not np.isfinite(thr) or thr <= 0:
        return []

    mask = f >= thr
    if not mask.any():
        return []

    flips = np.flatnonzero(np.diff(mask.astype(int)))
    starts = [0] if mask[0] else []
    ends = []
    for i in flips:
        if mask[i + 1]:
            starts.append(i + 1)
        else:
            ends.append(i)
    if mask[-1]:
        ends.append(len(mask) - 1)

    merged = []
    for i0, i1 in zip(starts, ends):
        w = (float(t[i0]), float(t[i1]))
        if merged and w[0] - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], w[1])
        else:
            merged.append(w)

    return [w for w in merged if (w[1] - w[0]) >= min_duration]


def _aligned_data_for_load(subject, pipeline_results, load_key):
    '''取单组的 aligned_data（未切片）。先看 pipeline 结果，再读缓存 CSV。'''
    key = _canonical_load_key(load_key)

    result = (pipeline_results or {}).get(key)
    if result is None:
        for k, v in (pipeline_results or {}).items():
            if _canonical_load_key(k) == key:
                result = v
                break
    if result is not None:
        aligned = result.get('aligned_data')
        if aligned is not None and len(aligned) > 0:
            return aligned

    path = os.path.join(subject.result_folder, 'aligned_data.csv')
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    for col in ('load_weight', 'load', 'load_value'):
        if col in df.columns:
            sel = df[df[col].map(_canonical_load_key) == key]
            if len(sel) > 0:
                return sel.reset_index(drop=True)
    return None


def get_action_windows(config_path, load_keys,
                       movement_types=('upward', 'downward'),
                       force_cols=('force_l', 'force_r'),
                       min_force=None, force_frac=0.3,
                       min_duration=0.5, merge_gap=1.0,
                       include_insole=False,
                       cache_name='cutted_data.csv',
                       debug=False):
    '''
    每组的动作时间窗（机器人时钟），自动适配三种负载模式。

    优先用运动切片（定负载与等速都适用：等速只限了最高速度，
    杆照样上下走）；取不到时退回力阈值窗口（等长组）。

    同时处理一个很容易静默出错的坑：切片缓存是按 load_weight 存的，
    config 里的组名一旦改过（如 0.3 -> IK-0.3），旧缓存里就一个都对不上，
    表现为每组都「未取到窗口」。这里会检测到并自动 force_rebuild。

    Returns
    -------
    dict
        {load_key: {'window': (t0, t1) or None,
                    'source': 'movement' | 'force' | None,
                    'sub_windows': list[(t0, t1)]  # 仅 force 模式
                    'detail': str}}
        window 是首至末的包络区间。
    '''
    def _load_results(force_rebuild):
        '''取切片结果；流水线整体失败时不让异常穿出去。

        一个组的问题（例如 float('IM-1') 抛 ValueError）不应该把所有组
        的窗口一起带掉。拿不到切片时至少要保证 subject 可用，
        后面还能读 aligned_data.csv 走力阈值切段。
        '''
        try:
            return load_or_create_cutted_pipeline_results(
                config_path, include_xsens=False,
                include_insole=include_insole, debug=debug,
                force_rebuild=force_rebuild, cache_name=cache_name)
        except Exception as exc:
            beauty_print(
                '切片流水线整体失败（{}: {}）。\n'
                '这会让所有组一起失去动作窗口，所以改为只读 '
                'aligned_data，各组退回力阈值切段。'.format(
                    type(exc).__name__, exc),
                type="warning")
            try:
                return Subject(config_path), None, {}
            except Exception:
                return None, None, {}

    subject, _pipeline, results = _load_results(False)

    wanted = [str(k) for k in load_keys]
    available = {_canonical_load_key(k) for k in (results or {}).keys()}
    missing = [k for k in wanted if _canonical_load_key(k) not in available]

    if missing:
        beauty_print(
            '切片缓存 {} 里没有这些组: {}\n'
            '缓存里实际有的是: {}\n'
            '最常见的原因是 config 里的组名改过（例如 0.3 -> IK-0.3），'
            '而缓存里的 load_weight 还是旧名，于是每组都取不到窗口。\n'
            '正在用 force_rebuild=True 重建缓存（会重跑一次完整 pipeline）。'.format(
                cache_name, missing, sorted(available)),
            type="warning")
        subject_new, pipeline_new, results_new = _load_results(True)
        if results_new:
            subject, _pipeline, results = subject_new, pipeline_new, results_new
        elif subject_new is not None:
            # 重建失败：保留旧 results（至少旧名的组还能用），
            # 但 subject 要用新的，否则连 aligned_data 都读不到。
            subject = subject_new

    out = {}
    for load_key in wanted:
        try:
            seg = get_segment_from_results(results, load_key,
                                           movement_types=movement_types)
        except Exception as exc:
            beauty_print('组 {} 取运动切片时出错（{}: {}），'
                         '改用力阈值切段。'.format(
                             load_key, type(exc).__name__, exc),
                         type="warning")
            seg = None
        if seg is not None and len(seg) > 0 and 'time' in seg.columns:
            out[load_key] = {
                'window': (float(seg['time'].min()), float(seg['time'].max())),
                'source': 'movement',
                'sub_windows': [],
                'detail': '运动切片 {} -> {:.2f}-{:.2f}s，{} 帧'.format(
                    list(movement_types), float(seg['time'].min()),
                    float(seg['time'].max()), len(seg)),
            }
            continue

        # 运动切片拿不到：很可能是等长组。改用未切片的 aligned_data + 力阈值。
        aligned = _aligned_data_for_load(subject, results, load_key)
        if aligned is None or 'time' not in getattr(aligned, 'columns', []):
            out[load_key] = {
                'window': None, 'source': None, 'sub_windows': [],
                'detail': '既无运动切片，也拿不到 aligned_data'}
            beauty_print(
                '组 {} 既没有运动切片，也拿不到 aligned_data，'
                '无法划定动作窗口。'.format(load_key), type="warning")
            continue

        cols = [c for c in force_cols if c in aligned.columns]
        if not cols:
            out[load_key] = {
                'window': None, 'source': None, 'sub_windows': [],
                'detail': 'aligned_data 中没有力列 {}'.format(list(force_cols))}
            beauty_print(
                '组 {} 的 aligned_data 里没有 {}，无法用力阈值切段。'.format(
                    load_key, list(force_cols)), type="warning")
            continue

        total = aligned[cols].sum(axis=1).values.astype(float)
        windows = find_force_windows(
            aligned['time'].values, total, min_force=min_force,
            force_frac=force_frac, min_duration=min_duration,
            merge_gap=merge_gap)

        if not windows:
            out[load_key] = {
                'window': None, 'source': None, 'sub_windows': [],
                'detail': '力阈值没切出任何区间（力列 {}）'.format(cols)}
            beauty_print(
                '组 {} 用力阈值也没切出区间；请检查 {} 是否全为零或异常。'.format(
                    load_key, cols), type="warning")
            continue

        out[load_key] = {
            'window': (windows[0][0], windows[-1][1]),
            'source': 'force',
            'sub_windows': windows,
            'detail': '力阈值切出 {} 段，包络 {:.2f}-{:.2f}s（力列 {}，'
                      '总时长 {:.2f}s）'.format(
                          len(windows), windows[0][0], windows[-1][1], cols,
                          sum(w[1] - w[0] for w in windows)),
        }

    return out


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

    # 越界保护：np.interp 会用 left/right 端点常值静默填充越界样本，
    # 这会在不报错的情况下抹平负载效应，因此必须显式警告。
    src_lo = float(src_t[valid_src].min())
    src_hi = float(src_t[valid_src].max())
    dst_valid = dst_t[valid_dst]
    n_out = int(np.sum((dst_valid < src_lo) | (dst_valid > src_hi)))
    if n_out > 0:
        print(f'[WARN] interpolate_column_to_segment: '
              f'{n_out}/{len(dst_valid)} '
              f'({100.0 * n_out / len(dst_valid):.1f}%) 个目标时间点越界，'
              f'源区间=[{src_lo:.3f}, {src_hi:.3f}], '
              f'目标区间=[{dst_valid.min():.3f}, {dst_valid.max():.3f}], '
              f'列={value_col}；这些点已被端点常值填充。')

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