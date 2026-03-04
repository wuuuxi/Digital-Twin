# from opensim import *
import opensim as osim
import time
import numpy as np
import pandas as pd
import math
import threading
import scipy
import scipy.signal
import pygame
import json
import os

# os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

from typing import Dict, Any, Optional


class ConfigManager:
    """配置文件管理器"""

    def __init__(self, config_path: str = "config.json"):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

    def load_config(self) -> bool:
        """从文件加载配置"""
        try:
            if not os.path.exists(self.config_path):
                print(f"配置文件 {self.config_path} 不存在，使用默认配置")
                self._create_default_config()
                return True

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"配置文件加载成功: {self.config_path}")
            return True
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self._create_default_config()
            return False

    def save_config(self, config_path: str = None) -> bool:
        """保存配置到文件"""
        save_path = config_path or self.config_path
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"配置文件保存成功: {save_path}")
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False

    def _create_default_config(self):
        """创建默认配置"""
        self.config = {
            "data_settings": {
                "folder": "",
                "data_files": {},
                "musc_mvc": [],
                "fs": 1000
            },
            "opensim_settings": {
                "model_path": "workspace",
                "geometry_path": "workspace/Geometry"
            },
            "audio_settings": {
                "sound_path": "workspace",
                "fixed_beep_count": 5
            },
            "playback_settings": {
                "target_lift_duration": None,
                "target_lower_duration": None,
                "lift_speed_ratio": 1.0,
                "lower_speed_ratio": 1.0
            },
            "visualization_settings": {
                "display_multiplier": 23,
                "sample_num": 100,
                "arm_length": 0.37,
                "shoulder_height": 0.45
            }
        }

    def get_data_settings(self) -> Dict[str, Any]:
        """获取数据设置"""
        return self.config.get("data_settings", {})

    def get_opensim_settings(self) -> Dict[str, Any]:
        """获取OpenSim设置"""
        return self.config.get("opensim_settings", {})

    def get_audio_settings(self) -> Dict[str, Any]:
        """获取音频设置"""
        return self.config.get("audio_settings", {})

    def get_playback_settings(self) -> Dict[str, Any]:
        """获取播放设置"""
        return self.config.get("playback_settings", {})

    def get_visualization_settings(self) -> Dict[str, Any]:
        """获取可视化设置"""
        return self.config.get("visualization_settings", {})

    def update_setting(self, section: str, key: str, value: Any) -> bool:
        """更新特定设置"""
        try:
            if section in self.config:
                self.config[section][key] = value
                return True
            return False
        except Exception as e:
            print(f"更新设置失败: {e}")
            return False

class DataManager:
    """数据加载和管理类"""

    def __init__(self, data_settings):
        self.folder = data_settings.get("folder")
        self.data_files = data_settings.get("data_files")
        self.musc_mvc = data_settings.get("musc_mvc")
        self.Fs = data_settings.get("fs", 1000)

        self.input_data = []
        self.current_index = 0
        self.data_loaded = False
        self.original_lift_duration = 0
        self.original_lower_duration = 0
        self.current_data_index = 0
        self.loop_mode = True

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

            # Apply constraints similar to original UDP code
            pos_right = max(0.68, min(0.68 + self.ArmLength, pos_right))
            pos_left = max(0.68, min(0.68 + self.ArmLength, pos_left))

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

class OpenSimModel:
    """OpenSim模型管理类"""

    def __init__(self, opensim_settings):
        self.model_path = opensim_settings.get("model_path")
        self.model_name = opensim_settings.get("model_name", 'whole body model_yuetian_realtime.osim')
        self.modelFilePath = self.model_path + '/' + self.model_name

        self.geometry_path = opensim_settings.get("geometry_path", self.model_path + "/Geometry")

        self.model = None
        self.state = None
        self.visualizer = None
        self.muscle_state = MuscleStateManager()

        self.load_model()

    def load_model(self):
        """加载OpenSim模型"""
        try:
            self.model = osim.Model(self.modelFilePath)
            osim.ModelVisualizer.addDirToGeometrySearchPaths(self.geometry_path)

            self.model.setUseVisualizer(True)
            self.state = self.model.initSystem()
            init_state = osim.State(self.state)


            self.visualizer = self.model.updVisualizer()
            sviz = self.visualizer.updSimbodyVisualizer()
            self.visualizer.show(init_state)

            print("OpenSim model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading OpenSim model: {e}")
            return False

    def initBenchPressQ(self):
        """Initialize squat pose - same as original but renamed for clarity"""
        # Squat initial position coordinates
        Coord_Seq = [1, 8, 11, 13, 18, 53, 60, 55,
                     62]  # what does each item in q represent in model coordinate (in seq # way)
        value_Seq = [90, -25, -25, 85, 85, 90, 90, 90, 90]
        q_in_rad = np.zeros(len(value_Seq))
        for i in range(len(value_Seq)):
            q_in_rad[i] = value_Seq[i] / 180 * math.pi

        for i in range(len(Coord_Seq)):
            Coord = self.model.updCoordinateSet().get(Coord_Seq[i] - 1)
            Coord.setValue(self.state, q_in_rad[i])

        # Set pelvis height so that feet touch the ground
        Coord = self.model.updCoordinateSet().get(5 - 1)
        Coord.setValue(self.state, 0.52)
        return

    def updStateQ(self, input):
        """Update state variable q for selected coordinates - same as original"""
        Coord_Seq = [52, 51, 59, 58, 54, 61]

        if len(input) == 4:
            qt_r, qt_l = self.muscle_state.calculate_joint_angle(input)

            q_in_rad_r = np.zeros(7)
            q_in_rad_l = np.zeros(7)

            for i in range(7):
                q_in_rad_r[i] = qt_r[i] / 180 * math.pi
                q_in_rad_l[i] = qt_l[i] / 180 * math.pi

            right_coords = [51, 52, 53, 54, 55, 56, 57]
            for i, coord_idx in enumerate(right_coords):
                Coord = self.model.updCoordinateSet().get(coord_idx - 1)
                Coord.setValue(self.state, q_in_rad_r[i])

            left_coords = [58, 59, 60, 61, 62, 63, 64]
            for i, coord_idx in enumerate(left_coords):
                Coord = self.model.updCoordinateSet().get(coord_idx - 1)
                Coord.setValue(self.state, q_in_rad_l[i])

        return

    def updStateAct(self, input):
        """Update state activation level for all/selected muscles - same as original"""
        DisplayMultiplier = 23

        act = self.muscle_state.calculate_activation(input)
        for i in range(len(act)):
            act[i] = act[i] * DisplayMultiplier

        # left muscle groups
        MuscleGroup1 = [30, 31, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
        MuscleGroup2 = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        MuscleGroup3 = [12, 13, 14, 15]
        MuscleGroup4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        MuscleGroup5 = [215, 216, 217, 218, 219, 220]
        MuscleGroup6 = [242, 243, 244, 245, 246, 247]

        # left muscle groups
        MuscleGroup7 = [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156]
        MuscleGroup8 = [131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144]
        MuscleGroup9 = [127, 128, 129, 130]
        MuscleGroup10 = [116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126]
        MuscleGroup11 = [227, 228, 229, 230, 231, 232]
        MuscleGroup12 = [465, 466, 467, 468, 469, 470]

        muscle_groups = [MuscleGroup1, MuscleGroup2, MuscleGroup3, MuscleGroup4,
                         MuscleGroup5, MuscleGroup6, MuscleGroup7, MuscleGroup8,
                         MuscleGroup9, MuscleGroup10, MuscleGroup11, MuscleGroup12]

        MuscleSet = self.model.updForceSet()
        Muscles = MuscleSet.getMuscles()

        for group_idx, muscle_group in enumerate(muscle_groups):
            if group_idx < len(act):
                for i in range(len(muscle_group)):
                    Act = Muscles.get(muscle_group[i] - 1)
                    Act.setActivation(self.state, act[group_idx])

        return

    def update_visualization(self):
        """更新可视化"""
        if not self.model:
            return

        silo = self.model.getVisualizer().getInputSilo()
        self.model.equilibrateMuscles(self.state)

        self.model.updForceSet()
        self.model.updControls(self.state)
        self.model.realizeDynamics(self.state)
        self.model.getVisualizer().show(self.state)

class MuscleStateManager:
    """肌肉激活计算与管理类"""

    def __init__(self):
        self.armlenth = 0.37
        UpperLimbMusR1 = [-2.83847443e-02, 3.78972232e-04, 5.73952879e-04,
                          -4.93123956e-07, -4.39014837e-06, -1.86593838e-06]
        UpperLimbMusR2 = [-1.59330326e-01, 2.30448754e-03, 2.35025582e-03,
                          -5.08526370e-06, -2.24674247e-05, -6.13742225e-07]
        UpperLimbMusR3 = [-1.37639327e-01, 2.00722246e-03, 2.18952440e-03,
                          -4.10670254e-06, -2.10775861e-05, -2.03264374e-06]
        UpperLimbMusR4 = [-2.37691947e-03, 7.78713575e-05, 5.35114834e-05,
                          5.08632500e-08, -8.37050041e-07, -6.00804626e-08]
        UpperLimbMusR5 = [-1.01442551e-01, 1.47043898e-03, 1.56184263e-03,
                          -3.23510388e-06, -1.43861358e-05, -1.25572653e-06]
        UpperLimbMusR6 = [-1.07357358e-03, 1.15538266e-04, 2.14702498e-04,
                          -7.47505564e-07, -1.02141327e-07, -2.59922512e-06]
        UpperLimbMusR = [UpperLimbMusR1, UpperLimbMusR2, UpperLimbMusR3,
                         UpperLimbMusR4, UpperLimbMusR5, UpperLimbMusR6]
        self.UpperLimbMusR = UpperLimbMusR

    def _height_to_joint_angles(self, height):
        # height_remap = (height-0.68) / self.armlenth *37 + 0.68
        if height < 0.68:
            height = 0.68
        elif height > 0.68 + self.armlenth:
            height = 0.68 + self.armlenth
        P1 = [-88.62, 431.6, -283.8]  # arm_flex
        P2 = [507.7, -786.8, 255.1]  # armabd
        P3 = [-90.15, 379.6, -255.7]  # arm rot
        P4 = [-442.9, 416.9, 51.31]  # elb flex
        P5 = [-154, 333.8, -63.38]  # prosup
        P6 = [-48.62, 83.46, -45.51]  # wstFlex
        P7 = [-17.15, 23.32, -6.703]  # wstDev
        P = [P1, P2, P3, P4, P5, P6, P7]

        qt = np.zeros(7)

        for i in range(7):
            qt[i] = P[i][0] * height ** 2 + P[i][1] * height + P[i][2]

        return qt

    def calculate_joint_angle(self, input):
        qt_r = self._height_to_joint_angles(input[2])
        qt_l = self._height_to_joint_angles(input[3])
        return qt_r, qt_l

    def calculate_activation(self, input):
        qt_r, qt_l = self.calculate_joint_angle(input)

        if len(input) == 4:
            q_R_Shou_Flex = qt_r[3]
            q_R_Elbow_Flex = qt_r[0]
            q_L_Shou_Flex = qt_l[3]
            q_L_Elbow_Flex = qt_l[0]
        else:
            q_R_Shou_Flex = 0
            q_R_Elbow_Flex = 0
            q_L_Shou_Flex = 0
            q_L_Elbow_Flex = 0

        w1 = 1
        w2 = 1
        act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(6):
            tempAct = (self.UpperLimbMusR[i][0] +
                       self.UpperLimbMusR[i][1] * q_R_Shou_Flex * w1 +
                       self.UpperLimbMusR[i][2] * q_R_Elbow_Flex * w2 +
                       self.UpperLimbMusR[i][3] * q_R_Shou_Flex ** 2 +
                       self.UpperLimbMusR[i][4] * q_R_Shou_Flex * q_R_Elbow_Flex +
                       self.UpperLimbMusR[i][5] * q_R_Elbow_Flex ** 2)
            act[i] = max(0.0, min(1.0, tempAct))

        for i in range(0, 6):
            tempAct = (self.UpperLimbMusR[i][0] +
                       self.UpperLimbMusR[i][1] * q_L_Shou_Flex * w1 +
                       self.UpperLimbMusR[i][2] * q_L_Elbow_Flex * w2 +
                       self.UpperLimbMusR[i][3] * q_L_Shou_Flex ** 2 +
                       self.UpperLimbMusR[i][4] * q_L_Shou_Flex * q_L_Elbow_Flex +
                       self.UpperLimbMusR[i][5] * q_L_Elbow_Flex ** 2)
            act[i + 6] = max(0.0, min(1.0, tempAct))

        return act


class MetronomePlayer:
    """节拍器音效播放器 - 使用 pygame"""

    def __init__(self, sound_dir=None):
        self.enabled = True
        self.sound_files_ready = False

        if sound_dir is None:
            self.sound_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.sound_dir = sound_dir

        self._init_pygame()
        self._load_sounds()

    def _init_pygame(self):
        """初始化 pygame 音频"""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            return True
        except Exception as e:
            print(f"  Failed to initialize pygame: {e}")
            return False

    def _load_sounds(self):
        """加载音效文件"""

        beep_high_path = os.path.join(self.sound_dir, 'beep_high.wav')
        beep_low_path = os.path.join(self.sound_dir, 'beep_low.wav')

        if os.path.exists(beep_high_path) and os.path.exists(beep_low_path):
            try:
                self.beep_high = pygame.mixer.Sound(beep_high_path)
                self.beep_low = pygame.mixer.Sound(beep_low_path)
                self.sound_files_ready = True
                # print(f"  Sounds loaded successfully")
            except Exception as e:
                print(f"  Failed to load sounds: {e}")
        else:
            print("  Sound files not found")

    def play(self, frequency=1000, duration=0.15, sound_type=None):

        if not self.enabled or not self.sound_files_ready:
            return

        # 判断音效类型
        is_finish = frequency >= 1000
        if sound_type is None:
            sound_type = "Finish" if is_finish else "Motion"

        timestamp = time.strftime("%H:%M:%S")

        try:
            sound = self.beep_high if is_finish else self.beep_low
            sound.play()
            print(f"[{timestamp}] 🔔 {sound_type} ♪ ({frequency}Hz)")
        except Exception as e:
            print(f"[error: {timestamp}] 🔔 {sound_type} (error: {e})")

class AudioCueManager:

    def __init__(self, speed_controller, audio_settings):

        self.sound_dir = audio_settings.get("sound_path", None)
        self.fixed_beep_count = audio_settings.get("fixed_beep_count", 5)

        if self.sound_dir is None:
            self.sound_dir = os.path.dirname(os.path.abspath(__file__))

        self.current_phase_speed = None

        self.speed_controller = speed_controller
        self.metronome = MetronomePlayer(self.sound_dir)

        self.current_phase = None
        self.phase_start_time = 0
        self.last_beep_time = 0
        self.current_phase_beep_times = []
        self.current_beep_index = 0
        self.phase_duration = 0

    def play_beep(self, frequency=800, duration=0.15, sound_type=None):
        """播放提示音 - 使用节拍器音效"""

        # 根据音效类型确定显示文本
        if sound_type is None:
            sound_type = "Finish" if frequency >= 1000 else "Motion"

        self.metronome.play(frequency, duration, sound_type)

    # def check_and_play_beep(self, current_phase_name, cycle_elapsed_time):
    #     """
    #     检查是否需要播放提示音
    #     current_phase_name: 'lift' or 'lower'
    #     cycle_elapsed_time: 当前循环经过的时间(秒)
    #     frame_index: 当前帧索引（用于调试）
    #     """
    #
    #     # 检测阶段切换
    #     if self.current_phase != current_phase_name:
    #         old_phase = self.current_phase
    #         self.current_phase = current_phase_name
    #         self.phase_start_time = cycle_elapsed_time
    #         self.last_beep_time = cycle_elapsed_time
    #
    #         # 重置阶段内的音效计数器
    #         if self.current_phase == 'lift':
    #             phase_duration = self.speed_controller.lift_duration
    #             self.current_phase_speed = self.speed_controller.lift_playback_speed
    #         else:
    #             phase_duration = self.speed_controller.lower_duration
    #             self.current_phase_speed = self.speed_controller.lower_playback_speed
    #
    #         # 计算该阶段的音效时间点
    #         self.current_phase_beep_times = self._calculate_phase_beep_times(phase_duration)
    #         self.current_beep_index = 0  # 重置索引
    #     else:
    #         # 在同一阶段，检查是否到达下一个提示音时间
    #         phase_elapsed = cycle_elapsed_time - self.phase_start_time
    #
    #         # 检查是否有音效需要播放
    #         if self.current_phase_beep_times and self.current_beep_index < len(self.current_phase_beep_times):
    #             next_beep_time = self.current_phase_beep_times[self.current_beep_index]
    #
    #             # 使用小容差检查时间点
    #             if phase_elapsed >= next_beep_time - 0.05:  # 50ms容差
    #                 # 检查是否是最后一个音效
    #                 is_last_beep = (self.current_beep_index == len(self.current_phase_beep_times) - 1)
    #
    #                 if not is_last_beep:
    #                     # 前n-1个为Motion音效 (800Hz)
    #                     self.play_beep(frequency=800, duration=0.15, sound_type="Motion")
    #                 else:
    #                     # 最后一个为Finish音效 (1200Hz)
    #                     self.play_beep(frequency=1200, duration=0.15, sound_type="Finish")
    #                     # print(f"    Finish beep triggered at {phase_elapsed:.1f}s")
    #
    #                 self.current_beep_index += 1
    #
    #                 # 调试信息
    #                 # print(f"    Beep {self.metronome.current_beep_index-1}/{len(current_phase_beep_times)} triggered at {phase_elapsed:.1f}s")
    #
    #         # 特别检查：如果阶段即将结束但最后一个音效还未触发，强制触发
    #         if (self.current_phase_beep_times and
    #                 self.current_beep_index < len(self.current_phase_beep_times) and
    #                 self.current_beep_index == len(self.current_phase_beep_times) - 1 and
    #                 phase_elapsed >= self.current_phase_beep_times[-1] - 0.2):  # 在最后一个音效时间点前200ms开始检查
    #
    #             # 强制触发最后一个音效（Finish）
    #             self.play_beep(frequency=1200, duration=0.15, sound_type="Finish")
    #             # print(f"    FORCED Finish beep triggered at {phase_elapsed:.1f}s (before phase end)")
    #             self.current_beep_index += 1

    def check_and_play_beep(self, current_phase_name, cycle_elapsed_time, test_flag=False):
        """检查是否需要播放提示音"""

        # 检测阶段切换
        if self.current_phase != current_phase_name:
            old_phase = self.current_phase
            self.current_phase = current_phase_name

            # 重置阶段起始时间
            self.phase_start_time = cycle_elapsed_time
            self.last_beep_time = cycle_elapsed_time

            # 获取新阶段的持续时间
            if self.current_phase == 'lift':
                self.phase_duration = self.speed_controller.lift_duration
                self.current_phase_speed = self.speed_controller.lift_playback_speed
            else:
                self.phase_duration = self.speed_controller.lower_duration
                self.current_phase_speed = self.speed_controller.lower_playback_speed

            # 计算音效时间点
            relative_beep_times = self._calculate_phase_beep_times(self.phase_duration)

            # 转换为绝对时间点（基于cycle_elapsed_time）
            self.current_phase_beep_times = [
                self.phase_start_time + t for t in relative_beep_times
            ]

            # 重置音效索引
            self.current_beep_index = 0

            # 重要：如果当前时间已经超过了第一个音效时间点，调整索引
            if self.current_phase_beep_times:
                while (self.current_beep_index < len(self.current_phase_beep_times) and
                       cycle_elapsed_time >= self.current_phase_beep_times[self.current_beep_index] - 0.05):
                    self.current_beep_index += 1

            if test_flag is True:
                # 调试信息
                print(f"Phase switched: {old_phase} -> {current_phase_name}")
                print(f"  Start time: {self.phase_start_time:.2f}s")
                print(f"  Phase duration: {self.phase_duration:.2f}s")
                print(f"  Beep times: {[f'{t:.2f}' for t in self.current_phase_beep_times]}")
                print(f"  Starting index: {self.current_beep_index}")

        # 检查是否到达下一个提示音时间
        if self.current_phase_beep_times and self.current_beep_index < len(self.current_phase_beep_times):
            next_beep_time = self.current_phase_beep_times[self.current_beep_index]

            # 使用小容差检查时间点
            if cycle_elapsed_time >= next_beep_time - 0.05:  # 50ms容差
                # 确定音效类型
                total_beeps = len(self.current_phase_beep_times)
                current_beep_num = self.current_beep_index + 1

                is_last_beep = (current_beep_num == total_beeps)

                # 播放音效
                if not is_last_beep:
                    self.play_beep(frequency=800, duration=0.15, sound_type="Motion")
                    if test_flag is True:
                        print(f"Motion beep at {cycle_elapsed_time:.2f}s (beep {current_beep_num}/{total_beeps})")
                else:
                    self.play_beep(frequency=1200, duration=0.15, sound_type="Finish")
                    if test_flag is True:
                        print(f"Finish beep at {cycle_elapsed_time:.2f}s (beep {current_beep_num}/{total_beeps})")

                self.current_beep_index += 1

                if test_flag is True:
                    # 调试：显示时间间隔
                    if self.last_beep_time > 0:
                        interval = cycle_elapsed_time - self.last_beep_time
                        print(f"  Interval since last beep: {interval:.2f}s")

                self.last_beep_time = cycle_elapsed_time

        # 强制检查：如果阶段即将结束但最后一个音效还未触发，强制触发
        # 注意：现在使用 self.phase_duration 而不是局部变量 phase_duration
        if (self.current_phase_beep_times and
                self.current_beep_index < len(self.current_phase_beep_times) and
                self.current_beep_index == len(self.current_phase_beep_times) - 1):

            # 计算阶段结束时间
            phase_end_time = self.phase_start_time + self.phase_duration

            # 如果快到阶段结束了，强制播放Finish音效
            if cycle_elapsed_time >= phase_end_time - 0.2:  # 在结束前200ms
                self.play_beep(frequency=1200, duration=0.15, sound_type="Finish")
                if test_flag is True:
                    print(f"FORCED Finish beep at {cycle_elapsed_time:.2f}s (near phase end)")
                self.current_beep_index = len(self.current_phase_beep_times)
                self.last_beep_time = cycle_elapsed_time

    def _calculate_phase_beep_times(self, phase_duration):
        """计算阶段内的音效时间点（返回相对时间）"""
        if phase_duration <= 0:
            return []

        # 固定音效数量
        num_beeps = self.fixed_beep_count

        if num_beeps <= 1:
            return [phase_duration]  # 只有一个音效就放在最后

        # 关键：计算均匀间隔
        # 从0开始计算，包括起始点吗？我们不包括起始点，只在中间和结束播放
        interval = phase_duration / num_beeps

        # 生成音效时间点
        beep_times = [interval * (i + 1) for i in range(num_beeps)]

        # 确保最后一个音效正好在阶段结束时
        if beep_times and abs(beep_times[-1] - phase_duration) > 0.01:
            beep_times[-1] = phase_duration

        # 调试信息
        # print(f"  Phase duration: {phase_duration:.2f}s, {num_beeps} beeps")
        # print(f"  Interval: {interval:.2f}s")
        # print(f"  Beep times: {[f'{t:.2f}' for t in beep_times]}")

        return beep_times

    # def _calculate_phase_beep_times(self, phase_duration):
    #     """
    #     计算阶段内的音效时间点
    #     """
    #     if phase_duration <= 0:
    #         return []
    #
    #     # num_beeps = max(1, int(round(phase_duration / self.beep_interval)))
    #     num_beeps = self.fixed_beep_count
    #
    #     # 计算时间间隔
    #     interval = phase_duration / num_beeps
    #
    #     # 生成音效时间点
    #     beep_times = [interval * (i + 1) for i in range(num_beeps)]
    #
    #     # 确保最后一个音效正好在阶段结束时
    #     if beep_times and abs(beep_times[-1] - phase_duration) > 0.01:
    #         beep_times[-1] = phase_duration
    #
    #     motion_count = max(0, num_beeps - 1)
    #     finish_count = 1 if num_beeps > 0 else 0
    #
    #     # print(f"    Phase duration: {phase_duration:.1f}s, {num_beeps} beep times: {[f'{t:.1f}' for t in beep_times]}")
    #     # print(f"    Beep distribution: {motion_count} Motion + {finish_count} Finish")
    #
    #     return beep_times

    def reset(self):
        self.current_phase = None
        self.phase_start_time = 0
        self.last_beep_time = 0
        self.current_phase_beep_times = []
        self.current_beep_index = 0
        self.phase_duration = 0

class SpeedController:
    """播放速度控制类"""

    def __init__(self, data_manager, playback_settings):
        self.input_data = data_manager.input_data

        self.original_lift_duration = data_manager.original_lift_duration
        self.original_lower_duration = data_manager.original_lower_duration

        self.target_lift_duration = playback_settings.get("target_lift_duration", None)
        self.target_lower_duration = playback_settings.get("target_lower_duration", None)
        self.lift_speed_ratio = playback_settings.get("lift_speed_ratio", None)
        self.lower_speed_ratio = playback_settings.get("lower_speed_ratio", None)

        self.lift_duration = 10
        self.lower_duration = 10
        self.lift_playback_speed = 1.0
        self.lower_playback_speed = 1.0

        self.calculate_and_set_timings()

    def calculate_and_set_timings(self):
        """
        Calculates and sets the final durations and playback speeds for each phase
        based on user inputs.

        This is the core of the new timing logic. It follows these rules:
        1. If speed_ratio is provided for a phase, it takes precedence over target_duration.
        2. If only target_duration is provided, speed_ratio is calculated from it.
        3. If neither is provided for a phase, it raises an error.
        4. It then calculates the required frame delay (playback_speed) for the visualizer.
        """

        print("\n--- Calculating Timings and Speeds ---")
        print(f"User Inputs:")
        print(f"  - Target Lift Duration: {self.target_lift_duration} s")
        print(f"  - Lift Speed Ratio: {self.lift_speed_ratio} x")
        print(f"  - Target Lower Duration: {self.target_lower_duration} s")
        print(f"  - Lower Speed Ratio: {self.lower_speed_ratio} x")
        print("-" * 20)

        # --- LIFT PHASE CALCULATION ---
        if self.lift_speed_ratio is None and self.target_lift_duration is None:
            raise ValueError(
                "For the LIFT phase, you must provide either 'target_lift_duration' or 'lift_speed_ratio'. ")

        final_lift_ratio = 1.0
        if self.lift_speed_ratio is not None:
            final_lift_ratio = self.lift_speed_ratio
            self.lift_duration = self.original_lift_duration / final_lift_ratio
        else:  # target_lift_duration is not None
            self.lift_duration = self.target_lift_duration
            if self.original_lift_duration > 0:
                final_lift_ratio = self.original_lift_duration / self.lift_duration
            else:
                final_lift_ratio = 1.0

        # --- LOWER PHASE CALCULATION ---
        if self.lower_speed_ratio is None and self.target_lower_duration is None:
            raise ValueError(
                "For the LOWER phase, you must provide either 'target_lower_duration' or 'lower_speed_ratio'. Both cannot be None.")

        final_lower_ratio = 1.0
        if self.lower_speed_ratio is not None:
            final_lower_ratio = self.lower_speed_ratio
            self.lower_duration = self.original_lower_duration / final_lower_ratio
        else:  # target_lower_duration is not None
            self.lower_duration = self.target_lower_duration
            if self.original_lower_duration > 0:
                final_lower_ratio = self.original_lower_duration / self.lower_duration
            else:
                final_lower_ratio = 1.0

        # --- CALCULATE PLAYBACK SPEEDS (FRAME DELAYS) ---
        if len(self.input_data) > 0:
            lift_frames = len([d for d in self.input_data if d[4] == 'lift'])
            lower_frames = len([d for d in self.input_data if d[4] == 'lower'])

            if self.lift_duration > 0 and lift_frames > 0:
                lift_target_fps = lift_frames / self.lift_duration
                self.lift_playback_speed = 1.0 / lift_target_fps
            else:
                self.lift_playback_speed = 0.05  # Default fallback

            if self.lower_duration > 0 and lower_frames > 0:
                lower_target_fps = lower_frames / self.lower_duration
                self.lower_playback_speed = 1.0 / lower_target_fps
            else:
                self.lower_playback_speed = 0.05  # Default fallback

        # --- FINAL OUTPUT ---
        print("Final Calculated Parameters:")
        print(f"  Lift Phase:")
        print(f"    - Duration: {self.original_lift_duration:.2f}s --> {self.lift_duration:.2f}s")
        print(f"    - Effective Speed Ratio: {final_lift_ratio:.2f}x")
        print(f"    - Target FPS: {1.0 / self.lift_playback_speed:.1f}")
        print(f"  Lower Phase:")
        print(f"    - Duration: {self.original_lower_duration:.2f}s --> {self.lower_duration:.2f}s")
        print(f"    - Effective Speed Ratio: {final_lower_ratio:.2f}x")
        print(f"    - Target FPS: {1.0 / self.lower_playback_speed:.1f}")
        print("--------------------------------------\n")


class BenchPressVisualizerApp:
    """卧推可视化主应用程序"""

    def __init__(self, config_path: str = "config.json"):

        self.config_manager = ConfigManager(config_path)
        if not self.config_manager.load_config():
            print("警告：使用默认配置")

        data_settings = self.config_manager.get_data_settings()
        self.audio_settings = self.config_manager.get_audio_settings()
        self.playback_settings = self.config_manager.get_playback_settings()
        self.opensim_settings = self.config_manager.get_opensim_settings()

        self.data_manager = DataManager(data_settings)

        self.speed_controller = None
        self.metronome = None
        self.opensim_model = None

        self.loop_mode = True
        self.is_running = False


    def initialize(self):
        """初始化应用程序"""
        print("Initializing BenchPress Visualizer...")

        # Start data loading in a thread
        data_thread = threading.Thread(target=self.data_manager.load_data)
        data_thread.daemon = True
        data_thread.start()

        # Wait until data is loaded and original durations are known
        while not self.data_manager.data_loaded:
            time.sleep(0.1)

        # 设置阶段速度
        self.speed_controller = SpeedController(self.data_manager, self.playback_settings)
        self.metronome = AudioCueManager(self.speed_controller, self.audio_settings)
        self.opensim_model = OpenSimModel(self.opensim_settings)

    def run_visualization(self):
        """Main visualization function - modified to use file data"""

        self.opensim_model.initBenchPressQ()  # Initialize squat position
        self.opensim_model.update_visualization()

        print("Benchpress visualization started. Press Ctrl+C to stop.")

        cycle_start_time = time.time()
        self.metronome.reset()

        while True:
            # try:
            frame_start_time = time.time()  # 帧开始时间
            prev_index = self.data_manager.current_data_index

            current_input = self.data_manager.get_next_input()  # 获取输入数据
            current_phase_name = current_input[4] if len(current_input) > 4 else 'lift'  # 获取当前阶段

            # Detect start of a new loop
            if prev_index > 0 and self.data_manager.current_data_index == 0:
                # 重置所有计时器和状态
                cycle_start_time = time.time()
                self.metronome.reset()

            # 计算当前循环经过的时间
            cycle_elapsed_time = time.time() - cycle_start_time
            # 检查并播放提示音
            self.metronome.check_and_play_beep(current_phase_name, cycle_elapsed_time)

            self.opensim_model.updStateQ(current_input[:4])  # 更新关节角度
            # self.opensim_model.updStateAct(current_input[:4])  # 更新肌肉激活状态
            self.opensim_model.update_visualization()

            target_frame_time = self.metronome.current_phase_speed if hasattr(self.metronome,
                                                                              'current_phase_speed') else 0.05
            frame_elapsed = time.time() - frame_start_time
            sleep_time = max(0, target_frame_time - frame_elapsed)

            # frame_elapsed = time.time() - frame_start_time
            # sleep_time = max(0, self.metronome.current_phase_speed - frame_elapsed)
            time.sleep(sleep_time)
            # print(sleep_time)

            # except KeyboardInterrupt:
            #     print("Visualization stopped by user")
            #     break
            # except Exception as e:
            #     print(f"Error in visualization: {e}")
            #     break

if __name__ == "__main__":
    # deep_patch_numpy_typing()

    app = BenchPressVisualizerApp('config/bp_config.json')
    app.initialize()
    app.run_visualization()

    # Start visualization
    # try:
    #     app.run_visualization()
    # except KeyboardInterrupt:
    #     print("Program stopped by user")
    # except Exception as e:
    #     print(f"Program error: {e}")
