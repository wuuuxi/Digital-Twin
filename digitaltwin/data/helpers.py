"""
工具函数模块
提供通用的辅助函数
"""
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import shutil
import sys


def find_nearest_idx(array, value):
    """Find nearest index for a value in array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def resample_data(data, target_length, time_col='time'):
    """
    将数据重采样到目标长度

    Parameters:
    -----------
    data : pd.DataFrame
        原始数据
    target_length : int
        目标长度
    time_col : str
        时间列名

    Returns:
    --------
    pd.DataFrame
        重采样后的数据
    """
    if len(data) == target_length:
        return data.copy()

    if time_col not in data.columns:
        raise ValueError(f"数据中缺少时间列: {time_col}")

    original_time = data[time_col].values
    new_time = np.linspace(original_time[0], original_time[-1], target_length)

    # 创建新的DataFrame
    resampled_data = pd.DataFrame()
    resampled_data[time_col] = new_time

    # 对每一列进行插值重采样
    for column in data.columns:
        if column != time_col:
            original_values = data[column].values
            # 线性插值
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
        if indices[i] == indices[i - 1] + 1:  # 连续
            current_segment.append(indices[i])
        else:  # 不连续，开始新段
            segments.append(current_segment)
            current_segment = [indices[i]]

    segments.append(current_segment)
    return segments


def select_longest_segment(data, segments, min_length=10):
    """选择最长的连续段"""
    if not segments:
        return None

    longest_segment = max(segments, key=len)
    if len(longest_segment) < min_length:  # 太短的段忽略
        return None

    return data.iloc[longest_segment].reset_index(drop=True)


def print_debug_info(message, print_label=True):
    """打印调试信息"""
    if print_label:
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")


def save_pickle(data, filepath):
    """保存pickle文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    """加载pickle文件"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def make_result_folder(experiment_label):
    current_time = datetime.now()
    folder_name = current_time.strftime('%Y%m%d_%H%M%S')
    result_folder = f'result/{experiment_label}/{folder_name}/'
    os.makedirs(result_folder, exist_ok=True)

    code_dir = os.path.join(result_folder, 'code')
    os.makedirs(code_dir, exist_ok=True)
    current_file = sys.argv[0]
    file_name = os.path.basename(current_file)
    target_path = os.path.join(code_dir, file_name)
    shutil.copy2(current_file, target_path)
    return result_folder