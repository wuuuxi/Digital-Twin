"""
标准数据分析流水线编排层。

从 analysis/result_analysis.py 拆出，负责「运行数据流水线」这一层编排：
  - run_standard_data_pipeline: Subject -> MultiLoadPipeline.run() -> 切片
  - load_or_create_cutted_pipeline_results: 切片结果带 CSV 缓存
  - get_action_windows: 每组的动作时间窗（内部调用上面的流水线）

纯分析函数（read_opensim_table / get_segment_from_results / find_force_windows /
summarize_inverse_dynamics_moments 等）仍在 analysis/result_analysis.py，
本层通过参数接收 pipeline results，不再在 analysis 与 pipeline 之间互相依赖。
"""
import os

import numpy as np
import pandas as pd

from digitaltwin.subject import Subject
from digitaltwin.analysis.alignment import DataAligner
from digitaltwin.analysis.result_analysis import (
    get_segment_from_results,
    find_force_windows,
    _aligned_data_for_load,
)
from digitaltwin.pipelines.multi_load import MultiLoadPipeline
from digitaltwin.utils.data_io import (
    canonical_load_key as _canonical_load_key,
    read_data_csv as _read_data_csv,
)
from digitaltwin.utils.logger import beauty_print
from digitaltwin.models import PipelineResults, TrialMetadata, TrialResult


def run_standard_data_pipeline(config_path, include_xsens=False,
                               include_insole=False,
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
    results : PipelineResults
    """
    subject = Subject(config_path)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = debug
    results = pipeline.run(
        include_xsens=include_xsens,
        include_insole=include_insole,
    )
    return subject, pipeline, results


def _segments_to_pipeline_results(cutted_df):
    """
    将切片缓存表还原成 summarize_* 函数需要的 pipeline_results 结构。

    只需要每个 load 的 cutted_data，因此不还原 robot/emg/xsens 等完整字段。
    """
    if cutted_df is None or len(cutted_df) == 0:
        return PipelineResults()

    load_col = None
    for c in ('load_weight', 'load', 'load_value'):
        if c in cutted_df.columns:
            load_col = c
            break

    if load_col is None:
        return PipelineResults({
            'all': TrialResult(
                metadata=TrialMetadata(load_weight='all'),
                segments=cutted_df,
            )
        })

    results = {}
    for load_key, df_load in cutted_df.groupby(load_col):
        key = _canonical_load_key(load_key)
        numeric_value = None
        try:
            numeric_value = float(load_key)
        except (TypeError, ValueError):
            pass
        results[key] = TrialResult(
            metadata=TrialMetadata(
                load_weight=key,
                load_value=numeric_value,
            ),
            segments=df_load.reset_index(drop=True),
        )
    return PipelineResults(results)


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
        cutted_df = _read_data_csv(cutted_cache_path)

        # 与 aligned_data 分支同样的检查：需要鞋垫 GRF 但缓存里没有
        # grf_l / grf_r 时不能直接用，否则调用方拿不到鞋垫列，
        # 会静默回退到别的数据源。
        if include_insole and not {'grf_l', 'grf_r'}.issubset(cutted_df.columns):
            if debug:
                print('[cache] 切片缓存缺少 grf_l/grf_r，改为重新生成...')
            cutted_df = None
        if cutted_df is not None and len(cutted_df) > 0:
            return subject, None, _segments_to_pipeline_results(cutted_df)

    # 如果已有 aligned_data.csv，不再跑完整 pipeline，只重新做标准切片。
    # 但当需要鞋垫 GRF 且缓存里没有 grf_l / grf_r 时，必须重新跑 pipeline，
    # 否则无法补回鞋垫列。
    if os.path.exists(aligned_cache_path) and not force_rebuild:
        if debug:
            print(f'[cache] 读取 aligned_data 并重新切片: {aligned_cache_path}')
        aligned_df = _read_data_csv(aligned_cache_path)

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
        debug=debug,
    )
    cutted_df = _collect_cutted_from_pipeline_results(results)
    if cutted_df is not None and len(cutted_df) > 0:
        cutted_df.to_csv(cutted_cache_path, index=False)
        if debug:
            print(f'[cache] 切片缓存已保存: {cutted_cache_path}')

    return subject, pipeline, results


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
