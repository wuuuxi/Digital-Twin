"""
数组与信号处理工具
提供数组查找、数据重采样、连续索引段提取等通用数值工具函数。
"""
import numpy as np
import pandas as pd


def find_nearest_idx(array, value, axis=None):
    """
    Find nearest index for a value in array.

    Parameters
    ----------
    array : array_like
        Input array (1-D or N-D).
    value : float
        Target value.
    axis : int, optional
        If None (default), search over the flattened array and return
        a scalar index (1-D) or unraveled tuple (N-D).
        If given, compute the nearest index along that axis and return
        an array of indices with the same shape as *array* minus *axis*.

    Returns
    -------
    int or tuple of int or np.ndarray
    """
    array = np.asarray(array)
    if axis is not None:
        return np.abs(array - value).argmin(axis=axis)
    flat_idx = np.abs(array - value).argmin()
    if array.ndim == 1:
        return int(flat_idx)
    return np.unravel_index(flat_idx, array.shape)


def resample_data(data, target_length, time_col='time'):
    """
    将数据重采样到目标长度

    Parameters
    ----------
    data : pd.DataFrame
        原始数据
    target_length : int
        目标长度
    time_col : str
        时间列名

    Returns
    -------
    pd.DataFrame
        重采样后的数据
    """
    if len(data) == target_length:
        return data.copy()

    if time_col not in data.columns:
        raise ValueError(f"数据中缺少时间列: {time_col}")

    original_time = data[time_col].values
    new_time = np.linspace(original_time[0], original_time[-1], target_length)

    resampled_data = pd.DataFrame()
    resampled_data[time_col] = new_time

    for column in data.columns:
        if column != time_col:
            original_values = data[column].values
            interpolated_values = np.interp(new_time, original_time, original_values)
            resampled_data[column] = interpolated_values

    return resampled_data


def extract_continuous_segments(indices):
    """提取连续的索引段"""
    if len(indices) == 0:
        return []

    segments = []
    current_segment = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_segment.append(indices[i])
        else:
            segments.append(current_segment)
            current_segment = [indices[i]]

    segments.append(current_segment)
    return segments


def select_longest_segment(data, segments, min_length=10):
    """选择最长的连续段"""
    if not segments:
        return None

    longest_segment = max(segments, key=len)
    if len(longest_segment) < min_length:
        return None

    return data.iloc[longest_segment].reset_index(drop=True)