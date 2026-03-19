import numpy as np
import pandas as pd
import scipy
import os

from .emg_processor import EMGProcessor

class DataManager:
    """数据加载和管理类"""

    def __init__(self, data_settings, motion_label):
        self.folder = data_settings.get("folder")
        self.data_files = data_settings.get("data_files")
        self.musc_mvc = data_settings.get("musc_mvc")
        self.Fs = data_settings.get("fs", 1000)
        self.motion_label = motion_label

        self.input_data = []
        self.current_index = 0
        self.data_loaded = False
        self.original_lift_duration = 0
        self.original_lower_duration = 0
        self.current_data_index = 0
        self.loop_mode = True

        self.emg_processor = EMGProcessor(fs=self.Fs, musc_mvc=self.musc_mvc)

        # Configuration - modify these paths and parameters as needed
        self.ArmLength = 0.37

    def load_data(self):
        """Load and process squat data from files instead of UDP"""

        # emg_flag = 'benchpress'
        print("Loading bench press data from files...")
        # combined_data = load_benchpress_data(folder, data_files, musc_mvc, Fs)
        combined_data = self.load_benchpress_data()

        # Convert to input format expected by the visualization
        for _, row in combined_data.iterrows():
            # Convert position data to the format expected by squat simulation
            # Original UDP format: [x, y, right_height, left_height]
            pos_right = row.get('pos_l', 0.68)  # Default to 1.0 if not found
            pos_left = row.get('pos_l', 0.68)  # Use same value for both legs initially

            # # Apply constraints similar to original UDP code
            # pos_right = max(0.68, min(0.68 + self.ArmLength, pos_right))
            # pos_left = max(0.68, min(0.68 + self.ArmLength, pos_left))
            if self.motion_label == 'benchpress':
                pos_right = max(0.68, min(0.68 + self.ArmLength, pos_right))
                pos_left = max(0.68, min(0.68 + self.ArmLength, pos_left))
            elif self.motion_label == 'squat':
                pos_right = max(0.95, min(1.4, pos_right))
                pos_left = max(0.95, min(1.4, pos_left))

            phase = row.get('phase', 'lift')
            pos_data = [0.0, 0.0, pos_right, pos_left, phase]
            self.input_data.append(pos_data)

        self.data_loaded = True
        print(f"Loaded {len(self.input_data)} data points successfully")

        #
        # try:
        #     print("Loading bench press data from files...")
        #     # combined_data = load_benchpress_data(folder, data_files, musc_mvc, Fs)
        #     combined_data = load_benchpress_data(folder, data_files, musc_mvc, Fs)
        #
        #     # Convert to input format expected by the visualization
        #     input_data = []
        #     for _, row in combined_data.iterrows():
        #         # Convert position data to the format expected by squat simulation
        #         # Original UDP format: [x, y, right_height, left_height]
        #         pos_right = row.get('pos_l', 0.68)  # Default to 1.0 if not found
        #         pos_left = row.get('pos_l', 0.68)  # Use same value for both legs initially
        #
        #         # Apply constraints similar to original UDP code
        #         pos_right = max(0.68, min(0.68 + ArmLength, pos_right))
        #         pos_left = max(0.68, min(0.68 + ArmLength, pos_left))
        #
        #         phase = row.get('phase', 'lift')
        #         pos_data = [0.0, 0.0, pos_right, pos_left, phase]
        #         input_data.append(pos_data)
        #
        #     data_loaded = True
        #     print(f"Loaded {len(input_data)} data points successfully")
        #
        # except Exception as e:
        #     print(f"Error loading data: {e}")
        #     # Fallback to dummy data for testing
        #     input_data = [[0.0, 0.0, 1.0, 1.0, 'lift']] * 100
        #     data_loaded = True
        #     print("Using dummy data for testing")

    def load_benchpress_data(self, start_height=1.1, end_height=1.5):
        combined_data = pd.DataFrame()
        emg_data = pd.DataFrame()
        max_data = pd.DataFrame()
        max_xsens_data = pd.DataFrame()
        for file, indices in self.data_files.items():
            start_time, end_time, xsens_file, emg_file = indices

            file_name = os.path.join(self.folder, xsens_file)
            # position_data = pd.read_excel(file_name, sheet_name='Segment Position', engine='xlrd')
            position_data = pd.read_excel(file_name, sheet_name='Segment Position')
            velocity_data = pd.read_excel(file_name, sheet_name='Segment Velocity')
            hand_position = position_data[['Left Hand z']].rename(columns={'Left Hand z': 'pos_l'})
            hand_velocity = velocity_data[['Left Hand z']].rename(columns={'Left Hand z': 'vel_l'})
            time = np.linspace(0, hand_position.shape[0] - 1, hand_position.shape[0]) / 60
            xsens_data = pd.concat([pd.Series(time, name='time'), hand_position, hand_velocity], axis=1)
            xsens_data['load'] = float(file)

            start = self._find_nearest_idx(time, start_time)
            end = self._find_nearest_idx(time, end_time)
            xsens_data = xsens_data[start:end].copy()

            file_name_emg = os.path.join(self.folder, emg_file)
            emg_raw = self._load_emg_from_csv(file_name_emg).T
            for i in range(len(self.musc_mvc)):
                emg_raw[i + 1] = self._emg_rectification(emg_raw[i + 1], self.musc_mvc[i])

            time_robot = np.asarray(xsens_data['time'])
            emg = []
            for i in range(len(time_robot)):
                t = time_robot[i]
                idx = self._find_nearest_idx(emg_raw[0], t)
                emg.append(emg_raw[:, idx])
            emg = np.asarray(emg)
            for i in range(len(self.musc_mvc)):
                xsens_data[f'emg{i}'] = np.asarray(emg)[:, i + 1]

            xsens_data['time'] = xsens_data['time'] - start_time
            # xsens_data = xsens_data[xsens_data['vel_l'] > 0.05]
            # xsens_data['pos_l'] = xsens_data['pos_l'] - (xsens_data['pos_l'].min() + xsens_data['pos_l'].max()) / 2
            combined_data = pd.concat([combined_data, xsens_data], ignore_index=True)
            # max_data = pd.concat([max_data, max_xsens_data], ignore_index=True)

        data = combined_data[combined_data['vel_l'] > 0.05]
        # max_data = max_data[max_data['vel_l'] > 0.05]
        # data = combined_data

        # 修改这里：提取上升和下降两个单独的动作
        lift_data, lower_data = self._extract_separate_benchpress_phases(combined_data)

        # 调整两个动作的时间长度一致（按照短的）
        lift_data, lower_data = self._synchronize_phase_durations(lift_data, lower_data)

        # 合并两个动作，添加阶段标记
        single_cycle_data = self._combine_phases_with_markers(lift_data, lower_data)

        # 计算阶段时长
        lift_duration = lift_data['time'].iloc[-1] - lift_data['time'].iloc[0] if len(lift_data) > 0 else 0
        lower_duration = lower_data['time'].iloc[-1] - lower_data['time'].iloc[0] if len(lower_data) > 0 else 0

        # 存储时长信息
        self.original_lift_duration = lift_duration
        self.original_lower_duration = lower_duration

        # print(f"  Phase durations after sync: Lift={lift_duration:.1f}s, Lower={lower_duration:.1f}s")

        return single_cycle_data

    def _emg_rectification(self, x, ref):
        nyquist_frequency = self.Fs / 2
        low_cutoff = 20 / nyquist_frequency
        high_cutoff = 450 / nyquist_frequency
        [b, a] = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band', analog=False)
        x = scipy.signal.filtfilt(b, a, x)  #零相位滤波

        x_mean = np.mean(x)
        raw = x - x_mean * np.ones_like(x)
        t = np.arange(0, raw.size / self.Fs, 1 / self.Fs)
        EMGFWR = abs(raw)

        NUMPASSES = 3
        LOWPASSRATE = 4  # 低通滤波2—10Hz得到包络线
        Wn = LOWPASSRATE / (self.Fs / 2)
        [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
        EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

        # 归一化处理
        normalized_EMG = EMGLE / ref
        y = normalized_EMG

        neural_activation = np.zeros_like(normalized_EMG)
        for i in range(len(normalized_EMG)):
            if i > 1:
                neural_activation[i] = 2.25 * y[i] - 1 * neural_activation[i - 1] - 0.25 * neural_activation[i - 2]
        u = neural_activation
        return u

    def get_next_input(self):
        if not self.data_loaded or len(self.input_data) == 0:
            return [0.0, 0.0, 0.68, 0.68, 'lift']

        if self.current_data_index >= len(self.input_data):
            if self.loop_mode:
                self.current_data_index = 0
                # print(f"Loop completed, restarting... (Duration: {benchpress_duration}s)")
            else:
                self.current_data_index = len(self.input_data) - 1

        current_input = self.input_data[self.current_data_index]
        self.current_data_index += 1

        return current_input

    def _find_nearest_idx(self, array, value):
        """Find nearest index for a value in array"""
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _combine_phases_with_markers(self, lift_data, lower_data):
        """
        合并上升和下降阶段，并添加阶段标记
        """
        if lift_data is None and lower_data is None:
            return pd.DataFrame()

        combined_data = pd.DataFrame()

        # 添加上升阶段
        if lift_data is not None:
            lift_data_with_phase = lift_data.copy()
            lift_data_with_phase['phase'] = 'lift'
            combined_data = pd.concat([combined_data, lift_data_with_phase], ignore_index=True)

        # 添加下降阶段
        if lower_data is not None:
            lower_data_with_phase = lower_data.copy()
            lower_data_with_phase['phase'] = 'lower'
            combined_data = pd.concat([combined_data, lower_data_with_phase], ignore_index=True)

        # 重新计算总时间
        if len(combined_data) > 0:
            total_duration = combined_data['time'].iloc[-1] - combined_data['time'].iloc[0]
            combined_data['time'] = np.linspace(0, total_duration, len(combined_data))

        return combined_data

    def _load_emg_from_csv(self, file, motion='squat'):
        states = pd.read_csv(file, low_memory=False, skiprows=2, encoding='GB2312')
        raw_data = np.squeeze(np.asarray([states]))
        if motion == 'all':
            raw_data = raw_data[3:, 0:25].astype(float)  # 前musc_num个肌肉, 包括第一列时间
        elif motion == 'squat':
            # 筛选特定肌肉群
            raw_data = np.concatenate([raw_data[3:, 0:7], raw_data[3:, 13:19]], axis=1).astype(float)
            # emg_data = np.concatenate([raw_data[3:, 24:25], raw_data[3:, 0:6]], axis=1).astype(float)
            # raw_data = np.concatenate([emg_data, raw_data[3:, 12:18]], axis=1).astype(float)
        return raw_data

    def _extract_separate_benchpress_phases(self, data):
        """
        将卧推数据分离为上升和下降两个独立的动作
        """
        pos = data['pos_l'].values
        vel = data['vel_l'].values

        # 使用速度符号判断运动方向
        lift_indices = np.where(vel > 0.01)[0]  # 上升阶段（速度为正）
        lower_indices = np.where(vel < -0.01)[0]  # 下降阶段（速度为负）

        # 如果速度判断不充分，使用位置变化判断
        if len(lift_indices) == 0 or len(lower_indices) == 0:
            lift_indices = []
            lower_indices = []
            for i in range(1, len(pos)):
                if pos[i] > pos[i - 1]:  # 位置增加 - 上升
                    lift_indices.append(i)
                elif pos[i] < pos[i - 1]:  # 位置减少 - 下降
                    lower_indices.append(i)

        # 提取连续的运动段
        lift_segments = self._extract_continuous_segments(lift_indices)
        lower_segments = self._extract_continuous_segments(lower_indices)

        # 选择最长的连续段作为代表性动作
        lift_data = self._select_longest_segment(data, lift_segments)
        lower_data = self._select_longest_segment(data, lower_segments)

        # 重置时间
        if lift_data is not None:
            lift_duration = lift_data['time'].iloc[-1] - lift_data['time'].iloc[0]
            lift_data = lift_data.copy()
            lift_data['time'] = np.linspace(0, lift_duration, len(lift_data))

        if lower_data is not None:
            lower_duration = lower_data['time'].iloc[-1] - lower_data['time'].iloc[0]
            lower_data = lower_data.copy()
            lower_data['time'] = np.linspace(0, lower_duration, len(lower_data))

        return lift_data, lower_data

    def _extract_continuous_segments(self, indices):
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

    def _select_longest_segment(self, data, segments):
        """选择最长的连续段"""
        if not segments:
            return None

        longest_segment = max(segments, key=len)
        if len(longest_segment) < 10:  # 太短的段忽略
            return None

        return data.iloc[longest_segment].reset_index(drop=True)

    def _synchronize_phase_durations(self, lift_data, lower_data):
        """
        调整上升和下降阶段的时间长度一致（按照短的）
        """
        if lift_data is None or lower_data is None:
            return lift_data, lower_data

        lift_len = len(lift_data)
        lower_len = len(lower_data)

        # 确定目标长度（取较短的那个）
        target_len = min(lift_len, lower_len)

        # print(f"  Before sync: Lift={lift_len} frames, Lower={lower_len} frames")
        # print(f"  Target length: {target_len} frames")

        # 重采样到相同长度
        if lift_len != target_len:
            lift_data = self._resample_data(lift_data, target_len)

        if lower_len != target_len:
            lower_data = self._resample_data(lower_data, target_len)

        return lift_data, lower_data

    def _resample_data(self, data, target_length):
        """将数据重采样到目标长度"""
        if len(data) == target_length:
            return data

        original_time = data['time'].values
        new_time = np.linspace(original_time[0], original_time[-1], target_length)

        # 创建新的DataFrame
        resampled_data = pd.DataFrame()
        resampled_data['time'] = new_time

        # 对每一列进行插值重采样
        for column in data.columns:
            if column != 'time':
                original_values = data[column].values
                # 线性插值
                interpolated_values = np.interp(new_time, original_time, original_values)
                resampled_data[column] = interpolated_values

        return resampled_data