"""
EMG数据处理模块
专门处理肌电信号
"""
# import numpy as np
# import pandas as pd
# import scipy.signal
# import os
# from digitaltwin.data.utils import find_nearest_idx, print_debug_info, save_pickle, load_pickle

import numpy as np
import pandas as pd
import scipy.signal
import os


class EMGProcessor:
    """
    EMG数据加载、整流和归一化处理器。
    统一提供 EMG 信号处理功能，供 data_manager 和离线分析流水线共用。
    """

    def __init__(self, fs=1000, musc_mvc=None, musc_label=None):
        self.fs = fs
        self.musc_mvc = musc_mvc or []
        self.musc_label = musc_label or []

    def process(self, emg_file, load_weight, emg_folder, folder,
                motion_flag='all', remove_leading_zeros=False):
        """
        处理单个负载的 EMG 数据文件。

        Parameters
        ----------
        emg_file : str
            EMG 文件名
        load_weight : str
            负载重量标识
        emg_folder : str
            EMG 文件目录
        folder : str
            实验根目录
        motion_flag : str
            运动模式 ('all' 或 'squat')
        remove_leading_zeros : bool
            是否移除前导零

        Returns
        -------
        dict or None
            包含 time, raw_signals, norm_signals, metadata 的字典
        """
        try:
            # 构建完整路径
            file_path = self._resolve_file_path(emg_file, emg_folder, folder)
            if file_path is None:
                return None

            # 加载原始 EMG 数据
            emg_raw = self.load_from_csv(file_path, motion=motion_flag,
                                         new=remove_leading_zeros)
            emg_raw = emg_raw.T

            # 构建结果字典
            emg_data = {
                'time': emg_raw[0],
                'raw_signals': {},
                'norm_signals': {},
                'metadata': {}
            }

            # 对每个肌肉通道进行整流和归一化
            for i, muscle_name in enumerate(self.musc_label):
                if i + 1 >= emg_raw.shape[0]:
                    break

                raw_signal = emg_raw[i + 1]
                rectified = self.rectification(raw_signal, self.fs)

                # MVC 归一化
                if i < len(self.musc_mvc) and self.musc_mvc[i] > 0:
                    norm_signal = rectified / self.musc_mvc[i]
                else:
                    norm_signal = rectified

                emg_data['raw_signals'][muscle_name] = raw_signal
                emg_data['norm_signals'][muscle_name] = norm_signal

            emg_data['metadata'] = {
                'load_weight': load_weight,
                'fs': self.fs,
                'muscle_labels': self.musc_label,
                'mvc_values': self.musc_mvc
            }
            return emg_data

        except Exception as e:
            print(f"EMG数据处理错误: {e}")
            # 后备简单加载
            try:
                file_path = self._resolve_file_path(emg_file, emg_folder, folder)
                return self.load_simple(file_path, load_weight)
            except:
                return None

    def load_from_csv(self, file_path, motion='all', new=False):
        """
        从 CSV 加载 EMG 原始数据。

        Parameters
        ----------
        file_path : str
            CSV 文件路径
        motion : str
            运动模式，'all' 或 'squat'
        new : bool
            是否移除前导零

        Returns
        -------
        np.ndarray
            EMG 数据矩阵 (samples x channels)
        """
        # emg_start_time = 0
        if new is not True:
            states = pd.read_csv(file_path, low_memory=False, skiprows=2, encoding='GB2312')
            raw_data = np.squeeze(np.asarray([states]))
            if motion == 'all':
                raw_data = raw_data[3:, 0:25].astype(float)  # 前musc_num个肌肉, 包括第一列时间
            elif motion == 'squat':
                raw_data = np.concatenate([raw_data[3:, 0:7], raw_data[3:, 13:19]], axis=1).astype(float)
        else:
            states = pd.read_csv(file_path, low_memory=False, skiprows=1, encoding='GB2312')
            # print(states.columns)
            # emg_start_time = pd.to_datetime(states['AbsTime'][0], format='%Y-%m-%d_%H-%M-%S.%f')
            raw_data = np.squeeze(np.asarray([states]))
            if motion == 'all':
                raw_data = np.concatenate([raw_data[3:, 24:25], raw_data[3:, 0:24]], axis=1).astype(float)
            elif motion == 'squat':
                emg_data = np.concatenate([raw_data[3:, 24:25], raw_data[3:, 0:6]], axis=1).astype(float)
                raw_data = np.concatenate([emg_data, raw_data[3:, 12:18]], axis=1).astype(float)
            raw_data, time_delay = self._remove_leading_zeros(raw_data)
            # emg_start_time = emg_start_time + pd.Timedelta(seconds=time_delay)
            # print(time_delay)
            # print(raw_data.shape)

        return raw_data

        # states = pd.read_csv(file_path, low_memory=False, skiprows=2,
        #                      encoding='GB2312')
        # raw_data = np.squeeze(np.asarray([states]))
        #
        # if motion == 'all':
        #     raw_data = raw_data[3:, 0:25].astype(float)
        # elif motion == 'squat':
        #     raw_data = np.concatenate(
        #         [raw_data[3:, 0:7], raw_data[3:, 13:19]], axis=1
        #     ).astype(float)
        #
        # if new:
        #     # 移除前导零区域
        #     first_nonzero = 0
        #     for i in range(raw_data.shape[0]):
        #         if np.any(raw_data[i, 1:] != 0):
        #             first_nonzero = i
        #             break
        #     raw_data = raw_data[first_nonzero:]
        #
        # return raw_data

    @staticmethod
    def rectification(signal, fs, ref=1.0):
        """
        EMG 信号整流流水线：带通滤波 → 全波整流 → 低通包络 → 归一化。

        Parameters
        ----------
        signal : np.ndarray
            原始 EMG 信号
        fs : int
            采样率 (Hz)
        ref : float
            MVC 参考值（归一化用）

        Returns
        -------
        np.ndarray
            处理后的 EMG 信号
        """
        nyquist = fs / 2
        low_cutoff = 20 / nyquist
        high_cutoff = 450 / nyquist

        # 带通滤波 (20-450 Hz)
        b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff],
                                   btype='band', analog=False)
        filtered = scipy.signal.filtfilt(b, a, signal)

        # 去均值 + 全波整流
        filtered = filtered - np.mean(filtered)
        emg_fwr = np.abs(filtered)

        # 低通滤波包络 (4 Hz)
        wn = 4 / (fs / 2)
        b_lp, a_lp = scipy.signal.butter(3, wn, 'low')
        envelope = scipy.signal.filtfilt(
            b_lp, a_lp, emg_fwr,
            padtype='odd', padlen=3 * (max(len(b_lp), len(a_lp)) - 1)
        )

        # MVC 归一化
        normalized = envelope / ref

        # 神经激活滤波
        neural_activation = np.zeros_like(normalized)
        for i in range(2, len(normalized)):
            neural_activation[i] = (2.25 * normalized[i]
                                    - 1.0 * neural_activation[i - 1]
                                    - 0.25 * neural_activation[i - 2])
        return neural_activation

    def load_simple(self, file_path, load_weight):
        """
        简单的 EMG 数据加载方法（后备）。
        """
        try:
            df = pd.read_csv(file_path)
            emg_data = {
                'time': np.arange(len(df)) / self.fs,
                'raw_signals': {},
                'norm_signals': {},
                'metadata': {}
            }
            for i, col in enumerate(df.columns):
                if i == 0 and 'time' in col.lower():
                    continue
                signal = df[col].values
                muscle_name = (self.musc_label[i]
                               if i < len(self.musc_label)
                               else f"Muscle_{i}")
                rectified = np.abs(signal)
                if i < len(self.musc_mvc) and self.musc_mvc[i] > 0:
                    norm_signal = rectified / self.musc_mvc[i]
                else:
                    norm_signal = rectified / (np.max(rectified) + 1e-10)
                emg_data['raw_signals'][muscle_name] = signal
                emg_data['norm_signals'][muscle_name] = norm_signal
            return emg_data
        except Exception as e:
            print(f"简单EMG加载也失败: {e}")
            return None

    @staticmethod
    def _resolve_file_path(emg_file, emg_folder, folder):
        """解析 EMG 文件的完整路径"""
        if os.path.isabs(emg_file):
            if os.path.exists(emg_file):
                return emg_file
            return None
        file_path = os.path.join(emg_folder, emg_file)
        if os.path.exists(file_path):
            return file_path
        file_path = os.path.join(folder, emg_file)
        if os.path.exists(file_path):
            return file_path
        print(f"EMG文件不存在: {emg_file}")
        return None

    @staticmethod
    def _remove_leading_zeros(data):
        # a = data[:, 1]
        # b = np.nonzero(data[:, 1])
        nonzero_indices = np.nonzero(data[:, 1])[0]
        if len(nonzero_indices) > 0:
            first_nonzero = nonzero_indices[0]
            data = data[first_nonzero:]
            time_delay = data[0, 0]
            data[:, 0] = data[:, 0] - time_delay
            return data, time_delay
        return data, data[0, 0]


# class EMGProcessor:
#     """EMG数据处理器"""
#
#     def __init__(self, fs=1000, musc_label=None, musc_mvc=None):
#         """
#         初始化EMG处理器
#
#         Parameters:
#         -----------
#         fs : int
#             采样率，默认1000Hz
#         musc_label : list
#             肌肉名称列表
#         musc_mvc : list
#             肌肉MVC值列表
#         """
#         self.fs = fs
#         self.musc_label = musc_label or []
#         self.musc_mvc = musc_mvc or []
#         self.emg_data = None
#         self.processed_data = None
#
#     def load_from_csv(self, file_path, motion='all', remove_leading_zeros=True):
#         """
#         从CSV文件加载EMG数据
#
#         Parameters:
#         -----------
#         file_path : str
#             CSV文件路径
#         motion : str
#             运动类型，'all'或'squat'
#         remove_leading_zeros : bool
#             是否移除前导零
#
#         Returns:
#         --------
#         tuple
#             (emg_raw, start_time)
#         """
#         try:
#             print_debug_info(f"加载EMG文件: {os.path.basename(file_path)}")
#
#             # 读取CSV文件
#             states = pd.read_csv(file_path, low_memory=False, skiprows=2, encoding='GB2312')
#             raw_data = np.squeeze(np.asarray([states]))
#
#             # 根据运动类型选择肌肉
#             if motion == 'all':
#                 raw_data = raw_data[3:, 0:25].astype(float)  # 前musc_num个肌肉, 包括第一列时间
#             elif motion == 'squat':
#                 # 筛选特定肌肉群
#                 raw_data = np.concatenate([raw_data[3:, 0:7], raw_data[3:, 13:19]], axis=1).astype(float)
#
#             # 移除前导零
#             if remove_leading_zeros:
#                 first_nonzero = np.where(np.any(raw_data[1:] != 0, axis=0))[0]
#                 if len(first_nonzero) > 0:
#                     raw_data = raw_data[:, first_nonzero[0]:]
#
#             emg_raw = raw_data.T
#             start_time = emg_raw[0, 0]
#
#             print_debug_info(f"EMG数据加载成功: {emg_raw.shape[1]}个样本, {emg_raw.shape[0] - 1}个通道")
#
#             return emg_raw, start_time
#
#         except Exception as e:
#             print_debug_info(f"EMG加载失败: {e}")
#             return None, None
#
#     def rectify_signal(self, signal, mvc_value=None):
#         """
#         EMG信号整流和归一化
#
#         Parameters:
#         -----------
#         signal : np.array
#             原始EMG信号
#         mvc_value : float
#             MVC值，用于归一化
#
#         Returns:
#         --------
#         np.array
#             处理后的信号
#         """
#         # 带通滤波
#         nyquist = self.fs / 2
#         low_cutoff = 20 / nyquist
#         high_cutoff = 450 / nyquist
#         [b, a] = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band', analog=False)
#         x = scipy.signal.filtfilt(b, a, signal)
#
#         # 去除直流分量
#         x_mean = np.mean(x)
#         raw = x - x_mean
#
#         # 全波整流
#         EMGFWR = np.abs(raw)
#
#         # 低通滤波获取包络
#         NUMPASSES = 3
#         LOWPASSRATE = 4
#         Wn = LOWPASSRATE / (self.fs / 2)
#         [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
#         EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd',
#                                       padlen=3 * (max(len(b), len(a)) - 1))
#
#         # 归一化
#         if mvc_value and mvc_value > 0:
#             normalized_EMG = EMGLE / mvc_value
#         else:
#             normalized_EMG = EMGLE
#
#         # 神经激活模型
#         neural_activation = np.zeros_like(normalized_EMG)
#         for i in range(len(normalized_EMG)):
#             if i > 1:
#                 neural_activation[i] = 2.25 * normalized_EMG[i] - 1 * neural_activation[i - 1] - 0.25 * \
#                                        neural_activation[i - 2]
#
#         return neural_activation
#
#     def process_emg_data(self, emg_raw, mvc_values=None):
#         """
#         处理完整的EMG数据
#
#         Parameters:
#         -----------
#         emg_raw : np.array
#             原始EMG数据，第一列为时间
#         mvc_values : list
#             MVC值列表
#
#         Returns:
#         --------
#         dict
#             处理后的EMG数据
#         """
#         if emg_raw is None:
#             return None
#
#         if mvc_values is None:
#             mvc_values = self.musc_mvc
#
#         emg_raw = emg_raw.T
#         time = emg_raw[0]
#
#         emg_data = {
#             'time': time,
#             'raw_signals': {},
#             'norm_signals': {},
#             'metadata': {
#                 'fs': self.fs,
#                 'muscle_labels': self.musc_label.copy(),
#                 'mvc_values': mvc_values.copy() if mvc_values else []
#             }
#         }
#
#         # 处理每个肌肉通道
#         for i, muscle_name in enumerate(self.musc_label):
#             if i + 1 < emg_raw.shape[0]:
#                 raw_signal = emg_raw[i + 1]
#                 mvc = mvc_values[i] if i < len(mvc_values) else None
#
#                 # 整流和归一化
#                 norm_signal = self.rectify_signal(raw_signal, mvc)
#
#                 emg_data['raw_signals'][muscle_name] = raw_signal
#                 emg_data['norm_signals'][muscle_name] = norm_signal
#
#         self.processed_data = emg_data
#         return emg_data
#
#     def align_to_robot_time(self, robot_time):
#         """
#         将EMG数据对齐到机器人时间
#
#         Parameters:
#         -----------
#         robot_time : np.array
#             机器人时间序列
#
#         Returns:
#         --------
#         dict
#             对齐后的EMG数据
#         """
#         if self.processed_data is None:
#             return None
#
#         emg_time = self.processed_data['time']
#         aligned_signals = {}
#
#         for muscle_name, signal in self.processed_data['norm_signals'].items():
#             aligned = []
#             for t in robot_time:
#                 idx = find_nearest_idx(emg_time, t)
#                 aligned.append(signal[idx])
#             aligned_signals[muscle_name] = np.array(aligned)
#
#         return aligned_signals
#
#     def save_processed_data(self, save_path):
#         """保存处理后的EMG数据"""
#         if self.processed_data:
#             save_pickle(self.processed_data, save_path)
#             print_debug_info(f"EMG数据已保存到: {save_path}")
#
#     def load_processed_data(self, load_path):
#         """加载处理后的EMG数据"""
#         data = load_pickle(load_path)
#         if data:
#             self.processed_data = data
#             print_debug_info(f"EMG数据已加载: {load_path}")
#         return data