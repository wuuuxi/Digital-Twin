import numpy as np
import pandas as pd
import os

from .emg_processor import EMGProcessor
from digitaltwin.utils.array_tools import (
    find_nearest_idx, resample_data,
    extract_continuous_segments, select_longest_segment,
)


class DataManager:
    """数据加载和管理类（实时可视化用）"""

    # 各运动类型的位置约束范围
    POSITION_LIMITS = {
        'benchpress': (0.68, 0.37),   # (base_height, arm_length)
        'squat': (0.95, 0.45),        # (min_height, max_offset)
    }

    def __init__(self, data_settings, motion_label):
        self.folder = data_settings.get("folder")
        self.data_files = data_settings.get("data_files")
        self.musc_mvc = data_settings.get("musc_mvc")
        self.fs = data_settings.get("fs", 1000)
        self.motion_label = motion_label

        self.input_data = []
        self.current_data_index = 0
        self.data_loaded = False
        self.original_lift_duration = 0
        self.original_lower_duration = 0
        self.loop_mode = True

        self.emg_processor = EMGProcessor(fs=self.fs, musc_mvc=self.musc_mvc)

    def load_data(self):
        """加载并处理运动数据"""
        print(f"Loading {self.motion_label} data from files...")
        combined_data = self._load_benchpress_data()

        for _, row in combined_data.iterrows():
            pos_right = row.get('pos_l', 0.68)
            pos_left = row.get('pos_l', 0.68)

            # 根据运动类型约束位置范围
            pos_right, pos_left = self._clip_positions(pos_right, pos_left)

            phase = row.get('phase', 'lift')
            self.input_data.append([0.0, 0.0, pos_right, pos_left, phase])

        self.data_loaded = True
        print(f"Loaded {len(self.input_data)} data points successfully")

    def get_next_input(self):
        """获取下一帧输入数据"""
        if not self.data_loaded or len(self.input_data) == 0:
            return [0.0, 0.0, 0.68, 0.68, 'lift']

        if self.current_data_index >= len(self.input_data):
            if self.loop_mode:
                self.current_data_index = 0
            else:
                self.current_data_index = len(self.input_data) - 1

        current_input = self.input_data[self.current_data_index]
        self.current_data_index += 1
        return current_input

    # ------------------------------------------------------------------
    #  内部方法
    # ------------------------------------------------------------------

    def _clip_positions(self, pos_right, pos_left):
        """根据运动类型约束位置范围"""
        if self.motion_label == 'benchpress':
            base, arm_len = self.POSITION_LIMITS['benchpress']
            pos_right = np.clip(pos_right, base, base + arm_len)
            pos_left = np.clip(pos_left, base, base + arm_len)
        elif self.motion_label == 'squat':
            lo, hi_offset = self.POSITION_LIMITS['squat']
            pos_right = np.clip(pos_right, lo, lo + hi_offset)
            pos_left = np.clip(pos_left, lo, lo + hi_offset)
        return pos_right, pos_left

    def _load_benchpress_data(self):
        """加载并预处理卧推/深蹲原始数据，返回单周期数据"""
        combined_data = pd.DataFrame()

        for file, indices in self.data_files.items():
            start_time, end_time, xsens_file, emg_file = indices

            # 加载 Xsens 运动捕捉数据
            file_path = os.path.join(self.folder, xsens_file)
            position_data = pd.read_excel(file_path, sheet_name='Segment Position')
            velocity_data = pd.read_excel(file_path, sheet_name='Segment Velocity')

            hand_position = position_data[['Left Hand z']].rename(
                columns={'Left Hand z': 'pos_l'})
            hand_velocity = velocity_data[['Left Hand z']].rename(
                columns={'Left Hand z': 'vel_l'})
            time = np.linspace(0, hand_position.shape[0] - 1,
                               hand_position.shape[0]) / 60

            xsens_data = pd.concat(
                [pd.Series(time, name='time'), hand_position, hand_velocity],
                axis=1)
            xsens_data['load'] = float(file)

            # 按时间窗截取
            start_idx = find_nearest_idx(time, start_time)
            end_idx = find_nearest_idx(time, end_time)
            xsens_data = xsens_data[start_idx:end_idx].copy()

            # 加载并对齐 EMG 数据
            emg_file_path = os.path.join(self.folder, emg_file)
            emg_raw = self.emg_processor.load_from_csv(emg_file_path).T
            for i in range(len(self.musc_mvc)):
                emg_raw[i + 1] = EMGProcessor.compute_envelope(
                    emg_raw[i + 1], self.fs, ref=self.musc_mvc[i])

            robot_time = np.asarray(xsens_data['time'])
            emg_aligned = []
            for t in robot_time:
                idx = find_nearest_idx(emg_raw[0], t)
                emg_aligned.append(emg_raw[:, idx])
            emg_aligned = np.asarray(emg_aligned)

            for i in range(len(self.musc_mvc)):
                xsens_data[f'emg{i}'] = emg_aligned[:, i + 1]

            xsens_data['time'] = xsens_data['time'] - start_time
            combined_data = pd.concat([combined_data, xsens_data],
                                      ignore_index=True)

        # 分离上升/下降阶段并同步
        lift_data, lower_data = self._extract_separate_phases(combined_data)
        lift_data, lower_data = self._synchronize_phase_durations(
            lift_data, lower_data)
        single_cycle_data = self._combine_phases_with_markers(
            lift_data, lower_data)

        # 记录阶段时长
        if lift_data is not None and len(lift_data) > 0:
            self.original_lift_duration = (
                lift_data['time'].iloc[-1] - lift_data['time'].iloc[0])
        if lower_data is not None and len(lower_data) > 0:
            self.original_lower_duration = (
                lower_data['time'].iloc[-1] - lower_data['time'].iloc[0])

        return single_cycle_data

    def _extract_separate_phases(self, data):
        """将数据分离为上升和下降两个独立阶段"""
        vel = data['vel_l'].values
        pos = data['pos_l'].values

        lift_indices = np.where(vel > 0.01)[0]
        lower_indices = np.where(vel < -0.01)[0]

        # 速度判断不充分时，使用位置变化
        if len(lift_indices) == 0 or len(lower_indices) == 0:
            lift_indices, lower_indices = [], []
            for i in range(1, len(pos)):
                if pos[i] > pos[i - 1]:
                    lift_indices.append(i)
                elif pos[i] < pos[i - 1]:
                    lower_indices.append(i)

        # 提取最长连续段
        lift_segments = extract_continuous_segments(lift_indices)
        lower_segments = extract_continuous_segments(lower_indices)
        lift_data = select_longest_segment(data, lift_segments)
        lower_data = select_longest_segment(data, lower_segments)

        # 重置时间轴
        for phase_data in [lift_data, lower_data]:
            if phase_data is not None:
                duration = phase_data['time'].iloc[-1] - phase_data['time'].iloc[0]
                phase_data = phase_data.copy()
                phase_data['time'] = np.linspace(0, duration, len(phase_data))

        return lift_data, lower_data

    def _synchronize_phase_durations(self, lift_data, lower_data):
        """将两个阶段重采样到相同长度（取较短的）"""
        if lift_data is None or lower_data is None:
            return lift_data, lower_data

        target_len = min(len(lift_data), len(lower_data))

        if len(lift_data) != target_len:
            lift_data = resample_data(lift_data, target_len)
        if len(lower_data) != target_len:
            lower_data = resample_data(lower_data, target_len)

        return lift_data, lower_data

    @staticmethod
    def _combine_phases_with_markers(lift_data, lower_data):
        """合并上升和下降阶段，添加 phase 标记列"""
        parts = []
        if lift_data is not None:
            lift_copy = lift_data.copy()
            lift_copy['phase'] = 'lift'
            parts.append(lift_copy)
        if lower_data is not None:
            lower_copy = lower_data.copy()
            lower_copy['phase'] = 'lower'
            parts.append(lower_copy)

        if not parts:
            return pd.DataFrame()

        combined = pd.concat(parts, ignore_index=True)
        total_duration = combined['time'].iloc[-1] - combined['time'].iloc[0]
        combined['time'] = np.linspace(0, total_duration, len(combined))
        return combined