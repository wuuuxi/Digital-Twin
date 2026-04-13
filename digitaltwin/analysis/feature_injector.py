"""
EMG / Xsens 特征注入模块。

负责将 MDF、RMS、Xsens 关节角度等特征插值到 aligned_data 的时间轴上，
使其与 pos_l / vel_l / emg_* 共享同一行索引，
经 cut_aligned_data 切片后 cutted_data 自然包含所有特征列。

从 pipeline.py 中拆出，避免 MultiLoadPipeline 过于庞大。
"""
import numpy as np
import pandas as pd

from digitaltwin.data.emg_processor import EMGProcessor


# ==============================================================
#  EMG 特征注入（MDF + RMS）
# ==============================================================

def inject_emg_features(aligned, emg_data, fs,
                        window_size=256, overlap=128):
    """
    将每块肌肉的 MDF 和 RMS 插值到 aligned_data 的时间轴上，
    作为新列 mdf_<muscle> 和 rms_<muscle> 写入 DataFrame。

    优先使用 emg_data 中预计算的 mdf_signals / rms_signals；
    若不存在则回退到从 raw_signals 重新计算。

    Parameters
    ----------
    aligned : pd.DataFrame
        对齐后的 DataFrame，必须包含 'time' 列。
    emg_data : dict
        EMGProcessor.process() 返回的字典，包含
        'time', 'raw_signals', 'mdf_signals', 'rms_signals' 等。
    fs : int
        EMG 采样率 (Hz)。
    window_size : int
        STFT 窗口大小（回退计算时使用）。
    overlap : int
        窗口重叠样本数（回退计算时使用）。

    Returns
    -------
    pd.DataFrame
        添加了 mdf_* 和 rms_* 列的 aligned DataFrame。
    """
    if aligned is None or emg_data is None:
        return aligned
    if 'time' not in aligned.columns:
        return aligned

    aligned_time = aligned['time'].values
    emg_time = emg_data.get('time')
    if emg_time is None or len(emg_time) == 0:
        return aligned

    t0 = emg_time[0]
    raw_signals = emg_data.get('raw_signals', {})
    mdf_signals = emg_data.get('mdf_signals', {})
    rms_signals = emg_data.get('rms_signals', {})

    new_cols = {}
    for musc_name in raw_signals.keys():
        # --- MDF ---
        if musc_name in mdf_signals:
            mdf_times = mdf_signals[musc_name]['time']
            mdf_vals = mdf_signals[musc_name]['values']
        else:
            mdf_times, mdf_vals = EMGProcessor.compute_median_frequency(
                raw_signals[musc_name], fs,
                window_size=window_size, overlap=overlap)

        if len(mdf_times) == 0:
            new_cols[f'mdf_{musc_name}'] = np.full(len(aligned_time), np.nan)
        else:
            mdf_abs = mdf_times + t0
            new_cols[f'mdf_{musc_name}'] = np.interp(
                aligned_time, mdf_abs, mdf_vals,
                left=np.nan, right=np.nan)

        # --- RMS ---
        if musc_name in rms_signals:
            rms_times = rms_signals[musc_name]['time']
            rms_vals = rms_signals[musc_name]['values']
        else:
            rms_times, rms_vals = EMGProcessor.compute_rms(
                raw_signals[musc_name], fs)

        if len(rms_times) == 0:
            new_cols[f'rms_{musc_name}'] = np.full(len(aligned_time), np.nan)
        else:
            rms_abs = rms_times + t0
            new_cols[f'rms_{musc_name}'] = np.interp(
                aligned_time, rms_abs, rms_vals,
                left=np.nan, right=np.nan)

    if new_cols:
        aligned = pd.concat(
            [aligned, pd.DataFrame(new_cols, index=aligned.index)],
            axis=1)

    return aligned


# ==============================================================
#  Xsens 关节角度注入
# ==============================================================

def inject_xsens_features(aligned, xsens_data, start_time=0):
    """
    将 Xsens 关节角度插值到 aligned_data 的时间轴上，
    作为新列 xsens_<joint_name> 写入 DataFrame。
    使用 pd.concat 一次性合并，避免逐列赋值导致碎片化。

    Parameters
    ----------
    aligned : pd.DataFrame
        对齐后的 DataFrame，必须包含 'time' 列。
    xsens_data : dict
        XsensProcessor.process() 返回的字典。
    start_time : float
        Xsens 时间轴偏移（秒），与 EMG 的 start_time 含义相同。
        即 xsens_time_abs = xsens_time + start_time。

    Returns
    -------
    pd.DataFrame
        添加了 xsens_* 列的 aligned DataFrame。
    """
    if aligned is None or xsens_data is None:
        return aligned
    if 'time' not in aligned.columns:
        return aligned

    joint_df = xsens_data.get('joint_angles')
    if joint_df is None or 'time' not in joint_df.columns:
        return aligned

    aligned_time = aligned['time'].values
    xsens_time = joint_df['time'].values + start_time  # 应用时间偏移

    # Xsens 采样间隔（用于计算角速度）
    xsens_fs = xsens_data.get('metadata', {}).get('fs', 60.0)
    dt = 1.0 / xsens_fs

    new_cols = {}
    for col in joint_df.columns:
        if col == 'time':
            continue
        vals = joint_df[col].values

        # 关节角度插值
        new_cols[f'xsens_{col}'] = np.interp(
            aligned_time, xsens_time, vals,
            left=np.nan, right=np.nan)

        # 关节角速度：中心差分 (deg/s)
        vel = np.gradient(vals, dt)
        new_cols[f'xsens_vel_{col}'] = np.interp(
            aligned_time, xsens_time, vel,
            left=np.nan, right=np.nan)

    if new_cols:
        aligned = pd.concat(
            [aligned, pd.DataFrame(new_cols, index=aligned.index)],
            axis=1)

    return aligned


# ==============================================================
#  批量 MDF 计算（独立于 pipeline 的工具函数）
# ==============================================================

def compute_mdf_for_results(results, musc_label, fs,
                            window_size=256, overlap=128):
    """
    对每个负载、每块肌肉计算 EMG 中值频率（MDF）时间序列。

    Returns
    -------
    dict
        {load_weight: {muscle_name: {'time': ndarray, 'mdf': ndarray}}}
    """
    mdf_results = {}
    for load_weight, result in results.items():
        emg_data = result.get('emg_data')
        if emg_data is None:
            continue
        mdf_results[load_weight] = {}
        for musc in musc_label:
            raw_signal = emg_data['raw_signals'].get(musc)
            if raw_signal is None:
                continue
            times, mdf = EMGProcessor.compute_median_frequency(
                raw_signal, fs,
                window_size=window_size, overlap=overlap)
            mdf_results[load_weight][musc] = {'time': times, 'mdf': mdf}
    return mdf_results


def compute_segmented_mdf_for_results(results, musc_label, fs,
                                      window_size=256, overlap=128):
    """
    计算运动切片后的 MDF，并与位置对齐。

    Returns
    -------
    dict
        {load_weight: {muscle_name: {'position': ndarray, 'mdf': ndarray}}}
    """
    seg_mdf = {}
    for load_weight, result in results.items():
        emg_data = result.get('emg_data')
        aligned = result.get('aligned_data')
        if emg_data is None or aligned is None:
            continue
        if 'time' not in aligned.columns or 'pos_l' not in aligned.columns:
            continue

        aligned_time = aligned['time'].values
        aligned_pos = aligned['pos_l'].values
        seg_mdf[load_weight] = {}

        for musc in musc_label:
            raw_signal = emg_data['raw_signals'].get(musc)
            if raw_signal is None:
                continue
            mdf_times, mdf_vals = EMGProcessor.compute_median_frequency(
                raw_signal, fs,
                window_size=window_size, overlap=overlap)
            if len(mdf_times) == 0:
                continue

            emg_time = emg_data['time']
            mdf_abs = mdf_times + (emg_time[0] if len(emg_time) > 0 else 0)
            mdf_positions = np.interp(
                mdf_abs, aligned_time, aligned_pos,
                left=np.nan, right=np.nan)
            valid = ~np.isnan(mdf_positions)
            seg_mdf[load_weight][musc] = {
                'position': mdf_positions[valid],
                'mdf': mdf_vals[valid]
            }
    return seg_mdf