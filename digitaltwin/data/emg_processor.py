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
                'mdf_signals': {},
                'rms_signals': {},
                'norm_signals': {},
                'metadata': {}
            }

            # 对每个肌肉通道进行整流和归一化
            for i, muscle_name in enumerate(self.musc_label):
                if i + 1 >= emg_raw.shape[0]:
                    break

                raw_signal = emg_raw[i + 1]
                filtered = self._bandpass_filter(raw_signal, self.fs)
                mdf_time, mdf_signal = self.compute_median_frequency(filtered, self.fs)
                rms_time, rms_signal = self.compute_rms(filtered, self.fs)

                if i < len(self.musc_mvc) and self.musc_mvc[i] > 0:
                    norm_signal = self.compute_envelope(raw_signal, self.fs, self.musc_mvc[i])
                else:
                    norm_signal = self.compute_envelope(raw_signal, self.fs)

                emg_data['raw_signals'][muscle_name] = raw_signal
                emg_data['mdf_signals'][muscle_name] = {'time': mdf_time, 'values': mdf_signal}
                emg_data['rms_signals'][muscle_name] = {'time': rms_time, 'values': rms_signal}
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
                return None
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
    def compute_envelope(signal, fs, ref=1.0):
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
        b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band', analog=False)
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

        # # 神经激活滤波
        # neural_activation = np.zeros_like(normalized)
        # for i in range(2, len(normalized)):
        #     neural_activation[i] = (2.25 * normalized[i]
        #                             - 1.0 * neural_activation[i - 1]
        #                             - 0.25 * neural_activation[i - 2])
        return normalized

    @staticmethod
    def compute_median_frequency(signal, fs, window_size=512, overlap=384, bandpass=True):
        """
        计算 EMG 信号的中值频率（Median Frequency, MDF）时间序列。

        使用短时傅里叶变换（STFT）在滑动窗口上计算功率谱密度，
        并求出将功率谱面积一分为二的频率。

        Parameters
        ----------
        signal : np.ndarray
            原始 EMG 信号（建议带通滤波后的信号）
        fs : int
            采样率 (Hz)
        window_size : int
            FFT 窗口大小（样本数），默认 256
        overlap : int
            窗口重叠样本数，默认 128
        bandpass : bool
            是否对输入信号进行带通滤波（20-450 Hz）。
            如果为 False，则对信号进行滤波；如果为 True，则假设信号已经滤波。
            默认为 True。

        Returns
        -------
        times : np.ndarray
            每个窗口中心对应的时间 (s)
        mdf : np.ndarray
            中值频率序列 (Hz)
        """

        if bandpass is False:
            nyquist = fs / 2
            low_cutoff = 20 / nyquist
            high_cutoff = 450 / nyquist
            b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            signal = scipy.signal.filtfilt(b, a, signal)

        step = window_size - overlap
        n_windows = (len(signal) - window_size) // step + 1
        if n_windows <= 0:
            return np.array([]), np.array([])

        times = np.zeros(n_windows)
        mdf = np.zeros(n_windows)
        window_func = np.hanning(window_size)

        for i in range(n_windows):
            start = i * step
            end = start + window_size
            segment = signal[start:end]

            # # 带通滤波（与 rectification 一致的 20-450 Hz 范围）
            # filtered = scipy.signal.filtfilt(b, a, segment)

            # 加汉宁窗
            windowed = segment * window_func

            # FFT → 功率谱
            fft_vals = np.fft.rfft(windowed)
            power = np.abs(fft_vals) ** 2
            freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)

            # 中值频率：累积功率达到总功率 50% 的频率
            cumulative_power = np.cumsum(power)
            total_power = cumulative_power[-1]
            if total_power > 1e-6:
                median_idx = np.searchsorted(cumulative_power, total_power / 2)
                mdf[i] = freqs[min(median_idx, len(freqs) - 1)]
            else:
                mdf[i] = np.nan

            times[i] = (start + window_size / 2) / fs

        return times, mdf

    @staticmethod
    def compute_rms(signal, fs, window_seconds=0.25, overlap_seconds=0.125, bandpass=True):
        """
        计算滑动窗口RMS（用于疲劳分析）

        Parameters
        ----------
        signal : np.ndarray
            原始EMG信号
        fs : int
            采样率 (Hz)
        window_seconds : float
            窗口长度（秒），默认0.25秒
        overlap_seconds : float
            重叠长度（秒），默认0.125秒（50%重叠）
        bandpass : bool
            是否对输入信号进行带通滤波（20-450 Hz）。
            如果为 False，则对信号进行滤波；如果为 True，则假设信号已经滤波。
            默认为 True。

        Returns
        -------
        times : np.ndarray
            每个窗口中心的时间 (s)
        rms : np.ndarray
            RMS值序列（原始幅值，未归一化）
        """
        if bandpass is False:
            nyquist = fs / 2
            low_cutoff = 20 / nyquist
            high_cutoff = 450 / nyquist
            b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            signal = scipy.signal.filtfilt(b, a, signal)

        window_size = int(window_seconds * fs)
        step = int((window_seconds - overlap_seconds) * fs)

        # if window_size % 2 == 0:
        #     window_size += 1  # 确保奇数，方便对齐

        n_windows = (len(signal) - window_size) // step + 1
        if n_windows <= 0:
            return np.array([]), np.array([])

        times = np.zeros(n_windows)
        rms_values = np.zeros(n_windows)

        for i in range(n_windows):
            start = i * step
            end = start + window_size
            window = signal[start:end]

            # RMS = sqrt(mean(x^2))
            rms_values[i] = np.sqrt(np.mean(window ** 2))
            times[i] = (start + window_size / 2) / fs

        return times, rms_values

    @staticmethod
    def compute_rms_envelope(signal, fs, window_seconds=0.1):
        """
        计算密集RMS（每个样本点一个值，用于与包络对比）

        通过滑动窗口为每个样本点计算RMS值
        """
        window_size = int(window_seconds * fs)
        half_win = window_size // 2

        rms_envelope = np.zeros_like(signal, dtype=float)
        padded = np.pad(signal, half_win, mode='reflect')

        for i in range(len(signal)):
            start = i
            end = i + window_size
            window = padded[start:end]
            rms_envelope[i] = np.sqrt(np.mean(window ** 2))

        return rms_envelope

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

    @staticmethod
    def compute_mvc_from_files(emg_files, emg_folder, folder, fs, musc_label,
                               motion_flag='all', remove_leading_zeros=False,
                               percentile_low=95, percentile_high=99):
        """
        从多个 EMG 文件计算每块肌肉的 MVC 值。

        处理流程：
          1. 对每个文件的每块肌肉：原始信号 → compute_envelope(ref=1.0) → 取包络的
             第 percentile_low ~ percentile_high 百分位的平均值
          2. 跨所有文件取每块肌肉的最大值作为 MVC

        Parameters
        ----------
        emg_files : list[str]
            EMG 文件名列表
        emg_folder : str
            EMG 文件目录
        folder : str
            实验根目录
        fs : int
            采样率
        musc_label : list[str]
            肌肉名称列表
        motion_flag : str
            运动模式 ('all' 或 'squat')
        remove_leading_zeros : bool
            是否移除前导零
        percentile_low : float
            百分位下限（默认 95）
        percentile_high : float
            百分位上限（默认 99）

        Returns
        -------
        dict
            {
                'musc_mvc': list[float],  # 每块肌肉的 MVC 值
                'per_file': dict,  # {filename: {musc: {envelope, percentile_avg, raw, filtered, artifact_pct}}}
            }
        """
        processor = EMGProcessor(fs=fs, musc_mvc=[], musc_label=musc_label)
        n_muscles = len(musc_label)
        # 每个文件每块肌肉的百分位平均值
        file_results = {}  # {filename: {musc: {...}}}
        musc_max = np.zeros(n_muscles)  # 跨文件最大值

        for emg_file in emg_files:
            file_path = EMGProcessor._resolve_file_path(
                emg_file, emg_folder, folder)
            if file_path is None:
                print(f"MVC 计算跳过: {emg_file}")
                continue

            try:
                emg_raw = processor.load_from_csv(
                    file_path, motion=motion_flag, new=remove_leading_zeros)
                emg_raw = emg_raw.T
            except Exception as e:
                print(f"MVC 加载失败 {emg_file}: {e}")
                continue

            file_data = {}
            for i, musc in enumerate(musc_label):
                if i + 1 >= emg_raw.shape[0]:
                    break

                raw_signal = emg_raw[i + 1]
                filtered = EMGProcessor._bandpass_filter(raw_signal, fs)
                envelope = EMGProcessor.compute_envelope(raw_signal, fs, ref=1.0)

                # 取第 percentile_low ~ percentile_high 百分位的平均值
                p_low = np.percentile(envelope, percentile_low)
                p_high = np.percentile(envelope, percentile_high)
                mask = (envelope >= p_low) & (envelope <= p_high)
                percentile_avg = np.mean(envelope[mask]) if np.any(mask) else np.max(envelope)

                # 运动伪影百分比：先去均值，再算 |demeaned - filtered| 的能量占 demeaned 能量的比例
                demeaned = raw_signal - np.mean(raw_signal)
                artifact = demeaned - filtered
                demeaned_energy = np.sum(demeaned ** 2)
                artifact_pct = (np.sum(artifact ** 2) / (demeaned_energy + 1e-15)) * 100

                file_data[musc] = {
                    'raw': raw_signal,
                    'filtered': filtered,
                    'envelope': envelope,
                    'percentile_avg': percentile_avg,
                    'artifact_pct': artifact_pct,
                }

                musc_max[i] = max(musc_max[i], percentile_avg)

            file_results[emg_file] = file_data

        # 保留小数点后四位
        musc_mvc_rounded = [round(v, 4) for v in musc_max.tolist()]

        return {
            'musc_mvc': musc_mvc_rounded,
            'per_file': file_results,
        }

    @staticmethod
    def _bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4):
        """带通滤波"""
        nyquist = fs / 2
        b, a = scipy.signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
        return scipy.signal.filtfilt(b, a, signal)
