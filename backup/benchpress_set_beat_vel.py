# -*- coding: utf-8 -*-
"""
Modified Squat script to load data from files instead of UDP connection
"""
from opensim import *
import time
import numpy as np
import pandas as pd
import os
import math
import threading
from datetime import datetime
import scipy
import scipy.signal
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d, binary_closing
import wave
import struct
import pygame
import types
import os

os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# Global variables
state = None
model = None
input_data = []
current_data_index = 0
data_loaded = False
ShouHeight = 0.45  # Shoulder height parameter
ArmLength = 0.37
sample_num = 100

# Time control variables
benchpress_duration = 10.0  # 默认一个卧推循环10秒
resting_time = 2
playback_speed = 1.0
loop_mode = True

# 分别控制上升和下降阶段的速度
lift_playback_speed = 1.0
lower_playback_speed = 1.0
current_phase_speed = 1.0  # 当前阶段的速度

# Audio cue variables
beep_interval = 0.5  # 提示音间隔（秒）
current_phase = None  # 'lift' or 'lower'
phase_start_time = 0
last_beep_time = 0
lift_duration = 10
lower_duration = 10

# 添加这两个全局变量
original_lift_duration = 0
original_lower_duration = 0


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


def deep_patch_numpy_typing():
    """深度修补numpy.typing模块"""

    # 如果numpy.typing不存在，创建完整的typing模块
    if not hasattr(np, 'typing'):
        print("创建numpy.typing模块...")
        typing_module = types.ModuleType('typing')

        # 创建NDArray类
        class _NDArrayMeta(type):
            def __getitem__(cls, item):
                return np.ndarray

        class NDArray(metaclass=_NDArrayMeta):
            pass

        # 添加必要的属性到typing模块
        typing_module.NDArray = NDArray
        typing_module.ArrayLike = np.ndarray

        # 将typing模块附加到numpy
        np.typing = typing_module
        sys.modules['numpy.typing'] = typing_module

        print("numpy.typing模块创建完成")

    # 确保NDArray属性存在
    if not hasattr(np.typing, 'NDArray'):
        class _NDArrayMeta(type):
            def __getitem__(cls, item):
                return np.ndarray

        class NDArray(metaclass=_NDArrayMeta):
            pass

        np.typing.NDArray = NDArray

    print("numpy.typing.NDArray修补完成")


def load_emg_from_csv(file, motion='squat'):
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


def emg_rectification(x, Fs, ref):
    nyquist_frequency = Fs / 2
    low_cutoff = 20 / nyquist_frequency
    high_cutoff = 450 / nyquist_frequency
    [b, a] = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band', analog=False)
    x = scipy.signal.filtfilt(b, a, x)  #零相位滤波

    x_mean = np.mean(x)
    raw = x - x_mean * np.ones_like(x)
    t = np.arange(0, raw.size / Fs, 1 / Fs)
    EMGFWR = abs(raw)

    NUMPASSES = 3
    LOWPASSRATE = 4  # 低通滤波2—10Hz得到包络线
    Wn = LOWPASSRATE / (Fs / 2)
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


def load_benchpress_data(folder, robot_files, musc_mvc, Fs, start_height=1.1, end_height=1.5):
    combined_data = pd.DataFrame()
    emg_data = pd.DataFrame()
    max_data = pd.DataFrame()
    max_xsens_data = pd.DataFrame()
    for file, indices in robot_files.items():
        start_time, end_time, xsens_file, emg_file = indices

        file_name = os.path.join(folder, xsens_file)
        # position_data = pd.read_excel(file_name, sheet_name='Segment Position', engine='xlrd')
        position_data = pd.read_excel(file_name, sheet_name='Segment Position')
        velocity_data = pd.read_excel(file_name, sheet_name='Segment Velocity')
        hand_position = position_data[['Left Hand z']].rename(columns={'Left Hand z': 'pos_l'})
        hand_velocity = velocity_data[['Left Hand z']].rename(columns={'Left Hand z': 'vel_l'})
        time = np.linspace(0, hand_position.shape[0] - 1, hand_position.shape[0]) / 60
        xsens_data = pd.concat([pd.Series(time, name='time'), hand_position, hand_velocity], axis=1)
        xsens_data['load'] = float(file)

        start = find_nearest_idx(time, start_time)
        end = find_nearest_idx(time, end_time)
        xsens_data = xsens_data[start:end].copy()

        max_xsens_data['pos_l'] = np.linspace(start_height, end_height, sample_num)
        max_vel_list = []
        for i in range(sample_num):
            def tar_pos(i):
                return start_height + i * (end_height - start_height) / sample_num

            data = xsens_data[(xsens_data['pos_l'] > tar_pos(i)) & (xsens_data['pos_l'] < tar_pos(i + 1))]['vel_l']
            if not data.empty:
                max_vel = data.max()  # 获取最大速度
            else:
                max_vel = np.nan  # 或者选择其他适当的值
            # max_vel = max(data)
            max_vel_list.append(max_vel)
        max_xsens_data['vel_l'] = max_vel_list
        # max_robot_data['load'] = np.random.normal(float(file), 0.1, len(max_vel_list))
        max_xsens_data['load'] = float(file)
        max_xsens_data['ori_load'] = float(file)

        file_name_emg = os.path.join(folder, emg_file)
        emg_raw = load_emg_from_csv(file_name_emg).T
        for i in range(len(musc_mvc)):
            emg_raw[i + 1] = emg_rectification(emg_raw[i + 1], Fs, musc_mvc[i])

        time_robot = np.asarray(xsens_data['time'])
        emg = []
        for i in range(len(time_robot)):
            t = time_robot[i]
            idx = find_nearest_idx(emg_raw[0], t)
            emg.append(emg_raw[:, idx])
        emg = np.asarray(emg)
        for i in range(len(musc_mvc)):
            xsens_data[f'emg{i}'] = np.asarray(emg)[:, i + 1]

        xsens_data['time'] = xsens_data['time'] - start_time
        # xsens_data = xsens_data[xsens_data['vel_l'] > 0.05]
        # xsens_data['pos_l'] = xsens_data['pos_l'] - (xsens_data['pos_l'].min() + xsens_data['pos_l'].max()) / 2
        combined_data = pd.concat([combined_data, xsens_data], ignore_index=True)
        max_data = pd.concat([max_data, max_xsens_data], ignore_index=True)

    data = combined_data[combined_data['vel_l'] > 0.05]
    max_data = max_data[max_data['vel_l'] > 0.05]
    # data = combined_data

    # 修改这里：提取上升和下降两个单独的动作
    lift_data, lower_data = extract_separate_benchpress_phases(combined_data)

    # 调整两个动作的时间长度一致（按照短的）
    lift_data, lower_data = synchronize_phase_durations(lift_data, lower_data)

    # 合并两个动作，添加阶段标记
    single_cycle_data = combine_phases_with_markers(lift_data, lower_data)

    # 计算阶段时长
    lift_duration = lift_data['time'].iloc[-1] - lift_data['time'].iloc[0] if len(lift_data) > 0 else 0
    lower_duration = lower_data['time'].iloc[-1] - lower_data['time'].iloc[0] if len(lower_data) > 0 else 0

    # 存储时长信息（全局变量）
    lift_duration = lift_duration
    lower_duration = lower_duration

    # print(f"  Phase durations after sync: Lift={lift_duration:.1f}s, Lower={lower_duration:.1f}s")

    return single_cycle_data


def extract_separate_benchpress_phases(data):
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
    lift_segments = extract_continuous_segments(lift_indices)
    lower_segments = extract_continuous_segments(lower_indices)

    # 选择最长的连续段作为代表性动作
    lift_data = select_longest_segment(data, lift_segments)
    lower_data = select_longest_segment(data, lower_segments)

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


def select_longest_segment(data, segments):
    """选择最长的连续段"""
    if not segments:
        return None

    longest_segment = max(segments, key=len)
    if len(longest_segment) < 10:  # 太短的段忽略
        return None

    return data.iloc[longest_segment].reset_index(drop=True)


def synchronize_phase_durations(lift_data, lower_data):
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
        lift_data = resample_data(lift_data, target_len)

    if lower_len != target_len:
        lower_data = resample_data(lower_data, target_len)

    return lift_data, lower_data


def resample_data(data, target_length):
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


def combine_phases_with_markers(lift_data, lower_data):
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


def find_nearest_idx(array, value):
    """Find nearest index for a value in array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def check_and_play_beep(current_phase_name, cycle_elapsed_time, frame_index, is_first_frame=False):
    """
    检查是否需要播放提示音
    current_phase_name: 'lift' or 'lower'
    cycle_elapsed_time: 当前循环经过的时间(秒)
    frame_index: 当前帧索引（用于调试）
    """
    global current_phase, phase_start_time, last_beep_time, lift_duration, lower_duration
    global current_phase_beep_times, current_beep_index
    global lift_playback_speed, lower_playback_speed, current_phase_speed

    # 检测阶段切换
    if current_phase != current_phase_name:
        old_phase = current_phase
        current_phase = current_phase_name
        phase_start_time = cycle_elapsed_time
        last_beep_time = cycle_elapsed_time

        # 重置阶段内的音效计数器
        if current_phase == 'lift':
            phase_duration = lift_duration
            current_phase_speed = lift_playback_speed
        else:
            phase_duration = lower_duration
            current_phase_speed = lower_playback_speed

        # 计算该阶段的音效时间点
        current_phase_beep_times = calculate_phase_beep_times(phase_duration)
        current_beep_index = 0  # 重置索引

        # print(f"  Phase change: {old_phase} -> {current_phase}, duration: {phase_duration:.1f}s")
        # print(f"  Speed change: {current_phase_speed:.3f}s/frame")
    else:
        # 在同一阶段，检查是否到达下一个提示音时间
        phase_elapsed = cycle_elapsed_time - phase_start_time

        # 检查是否有音效需要播放
        if current_phase_beep_times and current_beep_index < len(current_phase_beep_times):
            next_beep_time = current_phase_beep_times[current_beep_index]

            # 使用小容差检查时间点
            if phase_elapsed >= next_beep_time - 0.05:  # 50ms容差
                # 检查是否是最后一个音效
                is_last_beep = (current_beep_index == len(current_phase_beep_times) - 1)

                if not is_last_beep:
                    # 前n-1个为Motion音效 (800Hz)
                    play_beep(frequency=800, duration=0.15, sound_type="Motion")
                else:
                    # 最后一个为Finish音效 (1200Hz)
                    play_beep(frequency=1200, duration=0.15, sound_type="Finish")
                    # print(f"    Finish beep triggered at {phase_elapsed:.1f}s")

                current_beep_index += 1

                # 调试信息
                # print(f"    Beep {current_beep_index-1}/{len(current_phase_beep_times)} triggered at {phase_elapsed:.1f}s")

        # 特别检查：如果阶段即将结束但最后一个音效还未触发，强制触发
        if (current_phase_beep_times and
                current_beep_index < len(current_phase_beep_times) and
                current_beep_index == len(current_phase_beep_times) - 1 and
                phase_elapsed >= current_phase_beep_times[-1] - 0.2):  # 在最后一个音效时间点前200ms开始检查

            # 强制触发最后一个音效（Finish）
            play_beep(frequency=1200, duration=0.15, sound_type="Finish")
            # print(f"    FORCED Finish beep triggered at {phase_elapsed:.1f}s (before phase end)")
            current_beep_index += 1


def calculate_phase_beep_times(phase_duration):
    """
    计算阶段内的音效时间点
    根据阶段时长和beep_interval计算音效个数，前n-1个为Motion，第n个为Finish
    """
    if phase_duration <= 0:
        return []

    # 根据阶段时长和beep_interval计算音效个数
    # 确保至少有一个音效（Finish音效）
    num_beeps = max(1, int(round(phase_duration / beep_interval)))

    # 计算时间间隔
    interval = phase_duration / num_beeps

    # 生成音效时间点
    beep_times = [interval * (i + 1) for i in range(num_beeps)]

    # 确保最后一个音效正好在阶段结束时
    if beep_times and abs(beep_times[-1] - phase_duration) > 0.01:
        beep_times[-1] = phase_duration

    motion_count = max(0, num_beeps - 1)
    finish_count = 1 if num_beeps > 0 else 0

    # print(f"    Phase duration: {phase_duration:.1f}s, {num_beeps} beep times: {[f'{t:.1f}' for t in beep_times]}")
    # print(f"    Beep distribution: {motion_count} Motion + {finish_count} Finish")

    return beep_times


def play_beep(frequency=800, duration=0.15, sound_type=None):
    """播放提示音 - 使用节拍器音效"""
    global metronome
    if metronome is None:
        metronome = MetronomePlayer()

    # 根据音效类型确定显示文本
    if sound_type is None:
        sound_type = "Finish" if frequency >= 1000 else "Motion"

    metronome.play(frequency, duration, sound_type)


def initBenchPressQ():
    """Initialize squat pose - same as original but renamed for clarity"""
    global state, model
    # Squat initial position coordinates
    Coord_Seq = [1, 8, 11, 13, 18, 53, 60, 55,
                 62]  # what does each item in q represent in model coordinate (in seq # way)
    value_Seq = [90, -25, -25, 85, 85, 90, 90, 90, 90]
    q_in_rad = np.zeros(len(value_Seq))
    for i in range(len(value_Seq)):
        q_in_rad[i] = value_Seq[i] / 180 * math.pi

    for i in range(len(Coord_Seq)):
        Coord = model.updCoordinateSet().get(Coord_Seq[i] - 1)
        Coord.setValue(state, q_in_rad[i])

    # Set pelvis height so that feet touch the ground
    Coord = model.updCoordinateSet().get(5 - 1)
    Coord.setValue(state, 0.52)
    return


def q_Transform(height):
    global armlenth
    armlenth = 0.37
    # height_remap = (height-0.68) / armlenth *37 + 0.68
    if height < 0.68:
        height = 0.68
    elif height > 0.68 + armlenth:
        height = 0.68 + armlenth
    P1 = [-88.62, 431.6, -283.8];  # arm_flex
    P2 = [507.7, -786.8, 255.1];  # armabd
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


def updStateQ(input):
    """Update state variable q for selected coordinates - same as original"""
    global state, model
    Coord_Seq = [52, 51, 59, 58, 54, 61]

    if len(input) == 4:
        qt_r = q_Transform(input[2])
        qt_l = q_Transform(input[3])

        q_in_rad_r = np.zeros(7)
        q_in_rad_l = np.zeros(7)

        for i in range(7):
            q_in_rad_r[i] = qt_r[i] / 180 * math.pi
            q_in_rad_l[i] = qt_l[i] / 180 * math.pi

        right_coords = [51, 52, 53, 54, 55, 56, 57]
        for i, coord_idx in enumerate(right_coords):
            Coord = model.updCoordinateSet().get(coord_idx - 1)
            Coord.setValue(state, q_in_rad_r[i])

        left_coords = [58, 59, 60, 61, 62, 63, 64]
        for i, coord_idx in enumerate(left_coords):
            Coord = model.updCoordinateSet().get(coord_idx - 1)
            Coord.setValue(state, q_in_rad_l[i])

    return


def updStateAct(act):
    """Update state activation level for all/selected muscles - same as original"""
    DisplayMultiplier = 23
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

    global state, model
    MuscleSet = model.updForceSet()
    Muscles = MuscleSet.getMuscles()

    for group_idx, muscle_group in enumerate(muscle_groups):
        if group_idx < len(act):
            for i in range(len(muscle_group)):
                Act = Muscles.get(muscle_group[i] - 1)
                Act.setActivation(state, act[group_idx])

    return


def calculate_Activation(input):
    qt_r = q_Transform(input[2])
    qt_l = q_Transform(input[3])

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

    act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    UpperLimbMusR1 = [-2.83847443e-02, 3.78972232e-04, 5.73952879e-04, -4.93123956e-07, -4.39014837e-06,
                      -1.86593838e-06]
    UpperLimbMusR2 = [-1.59330326e-01, 2.30448754e-03, 2.35025582e-03, -5.08526370e-06, -2.24674247e-05,
                      -6.13742225e-07]
    UpperLimbMusR3 = [-1.37639327e-01, 2.00722246e-03, 2.18952440e-03, -4.10670254e-06, -2.10775861e-05,
                      -2.03264374e-06]
    UpperLimbMusR4 = [-2.37691947e-03, 7.78713575e-05, 5.35114834e-05, 5.08632500e-08, -8.37050041e-07, -6.00804626e-08]
    UpperLimbMusR5 = [-1.01442551e-01, 1.47043898e-03, 1.56184263e-03, -3.23510388e-06, -1.43861358e-05,
                      -1.25572653e-06]
    UpperLimbMusR6 = [-1.07357358e-03, 1.15538266e-04, 2.14702498e-04, -7.47505564e-07, -1.02141327e-07,
                      -2.59922512e-06]
    UpperLimbMusR = [UpperLimbMusR1, UpperLimbMusR2, UpperLimbMusR3, UpperLimbMusR4, UpperLimbMusR5, UpperLimbMusR6]

    w1 = 1
    w2 = 1
    for i in range(6):
        tempAct = (UpperLimbMusR[i][0] +
                   UpperLimbMusR[i][1] * q_R_Shou_Flex * w1 +
                   UpperLimbMusR[i][2] * q_R_Elbow_Flex * w2 +
                   UpperLimbMusR[i][3] * q_R_Shou_Flex ** 2 +
                   UpperLimbMusR[i][4] * q_R_Shou_Flex * q_R_Elbow_Flex +
                   UpperLimbMusR[i][5] * q_R_Elbow_Flex ** 2)
        act[i] = max(0.0, min(1.0, tempAct))

    for i in range(0, 6):
        tempAct = (UpperLimbMusR[i][0] +
                   UpperLimbMusR[i][1] * q_L_Shou_Flex * w1 +
                   UpperLimbMusR[i][2] * q_L_Elbow_Flex * w2 +
                   UpperLimbMusR[i][3] * q_L_Shou_Flex ** 2 +
                   UpperLimbMusR[i][4] * q_L_Shou_Flex * q_L_Elbow_Flex +
                   UpperLimbMusR[i][5] * q_L_Elbow_Flex ** 2)
        act[i + 6] = max(0.0, min(1.0, tempAct))

    return act


def DataLoader():
    """Load and process squat data from files instead of UDP"""
    global input_data, data_loaded, current_data_index, ShouHeight, lift_duration, lower_duration

    # Configuration - modify these paths and parameters as needed
    folder = ("D:/OneDrive - The Chinese University of Hong Kong/Experiment/"
              "20240604_HKSI_benchpress_yuetian/20240604 Yuetian bp_xlsx/")
    data_files = {
        # start, end, robot_file, emg_file
        '20': [6.650, 19.032, '20kg/MVN-004.xlsx', '20kg/test 2024_06_04 16_16_22.csv'],
        # '30': [13.449, 25.915, '30kg/MVN-005.xlsx', '30kg/test 2024_06_04 16_21_47.csv'],
        # '40': [9.683, 17.949, '40kg/MVN-006.xlsx', '40kg/test 2024_06_04 16_23_42.csv'],
        # '50': [7.850, 17.399, '50kg/MVN-007.xlsx', '50kg/test 2024_06_04 16_26_07.csv'],
        # '60': [23.915, 31.582, '60kg/MVN-008.xlsx', '60kg/test 2024_06_04 16_29_21.csv'],
    }

    musc_mvc = [0.1276, 0.2980, 0.3023, 0.3392, 0.3792, 0.3061,
                0.7502, 0.2241, 0.2260, 0.3334, 0.0539, 0.0480]
    Fs = 1000
    # emg_flag = 'benchpress'
    print("Loading bench press data from files...")
    # combined_data = load_benchpress_data(folder, data_files, musc_mvc, Fs)
    combined_data = load_benchpress_data(folder, data_files, musc_mvc, Fs)

    # Convert to input format expected by the visualization
    input_data = []
    for _, row in combined_data.iterrows():
        # Convert position data to the format expected by squat simulation
        # Original UDP format: [x, y, right_height, left_height]
        pos_right = row.get('pos_l', 0.68)  # Default to 1.0 if not found
        pos_left = row.get('pos_l', 0.68)  # Use same value for both legs initially

        # Apply constraints similar to original UDP code
        pos_right = max(0.68, min(0.68 + ArmLength, pos_right))
        pos_left = max(0.68, min(0.68 + ArmLength, pos_left))

        phase = row.get('phase', 'lift')
        pos_data = [0.0, 0.0, pos_right, pos_left, phase]
        input_data.append(pos_data)

    data_loaded = True
    print(f"Loaded {len(input_data)} data points successfully")
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


def set_benchpress_duration(duration_seconds):
    """设置卧推循环时间"""
    global benchpress_duration, playback_speed, input_data, lift_duration, lower_duration
    global lift_playback_speed, lower_playback_speed, original_lift_duration, original_lower_duration

    benchpress_duration = duration_seconds

    if len(input_data) > 0:
        total_data_points = len(input_data)

        # 重新按比例计算阶段时长
        original_total = lift_duration + lower_duration
        if original_total > 0:
            scale_factor = benchpress_duration / original_total
            lift_duration = lift_duration * scale_factor
            lower_duration = lower_duration * scale_factor
        else:
            lift_duration = benchpress_duration / 2
            lower_duration = benchpress_duration / 2

        # 保存原始的阶段持续时间（用于速度比例计算）
        original_lift_duration = lift_duration
        original_lower_duration = lower_duration

        # 计算每个阶段的帧数
        lift_frames = len([x for x in input_data if x[4] == 'lift'])
        lower_frames = len([x for x in input_data if x[4] == 'lower'])

        # 分别计算每个阶段的播放速度
        if lift_frames > 0:
            lift_target_fps = lift_frames / lift_duration
            lift_playback_speed = 1.0 / lift_target_fps if lift_target_fps > 0 else 0.05
        else:
            lift_playback_speed = 0.05

        if lower_frames > 0:
            lower_target_fps = lower_frames / lower_duration
            lower_playback_speed = 1.0 / lower_target_fps if lower_target_fps > 0 else 0.05
        else:
            lower_playback_speed = 0.05

        # 计算每个阶段的提示音次数
        lift_beeps = max(1, int(round(lift_duration / beep_interval)))
        lower_beeps = max(1, int(round(lower_duration / beep_interval)))

        lift_motion = max(0, lift_beeps - 1)
        lift_finish = 1 if lift_beeps > 0 else 0
        lower_motion = max(0, lower_beeps - 1)
        lower_finish = 1 if lower_beeps > 0 else 0

        # print(f"  Actual phase durations: Lift={lift_duration:.1f}s, Lower={lower_duration:.1f}s")
        # print(f"  Lift: {lift_duration:.1f}s ({lift_beeps} beeps: {lift_motion} Motion + {lift_finish} Finish)")
        # print(f"  Lower: {lower_duration:.1f}s ({lower_beeps} beeps: {lower_motion} Motion + {lower_finish} Finish)")
        # print(f"  Playback speeds: Lift={1.0/lift_playback_speed:.1f} fps, Lower={1.0/lower_playback_speed:.1f} fps\n")


def set_phase_speeds(lift_speed_ratio=1.0, lower_speed_ratio=1.0):
    """
    设置上升和下降阶段的播放速度比例
    lift_speed_ratio: 上升阶段速度比例 (默认1.0)
    lower_speed_ratio: 下降阶段速度比例 (默认1.0)
    """
    global lift_playback_speed, lower_playback_speed, lift_duration, lower_duration

    # 计算基础速度
    lift_frames = len([x for x in input_data if x[4] == 'lift'])
    lower_frames = len([x for x in input_data if x[4] == 'lower'])

    if lift_frames > 0:
        lift_target_fps = lift_frames / lift_duration
        base_lift_speed = 1.0 / lift_target_fps if lift_target_fps > 0 else 0.05
    else:
        base_lift_speed = 0.05

    if lower_frames > 0:
        lower_target_fps = lower_frames / lower_duration
        base_lower_speed = 1.0 / lower_target_fps if lower_target_fps > 0 else 0.05
    else:
        base_lower_speed = 0.05

    # 应用速度比例
    lift_playback_speed = base_lift_speed / lift_speed_ratio
    lower_playback_speed = base_lower_speed / lower_speed_ratio

    # 关键修复：根据速度比例调整阶段持续时间
    # 速度越快，阶段持续时间越短
    original_lift_duration = lift_duration
    original_lower_duration = lower_duration

    lift_duration = original_lift_duration / lift_speed_ratio
    lower_duration = original_lower_duration / lower_speed_ratio

    # 计算每个阶段的提示音次数（基于调整后的持续时间）
    lift_beeps = max(1, int(round(lift_duration / beep_interval)))
    lower_beeps = max(1, int(round(lower_duration / beep_interval)))

    lift_motion = max(0, lift_beeps - 1)
    lift_finish = 1 if lift_beeps > 0 else 0
    lower_motion = max(0, lower_beeps - 1)
    lower_finish = 1 if lower_beeps > 0 else 0

    # print(f"Phase speeds adjusted:")
    # print(f"  Lift: {1.0/lift_playback_speed:.1f} fps (ratio: {lift_speed_ratio:.1f})")
    # print(f"  Lower: {1.0/lower_playback_speed:.1f} fps (ratio: {lower_speed_ratio:.1f})")
    print(f"  Adjusted phase durations: Lift={lift_duration:.1f}s, Lower={lower_duration:.1f}s")
    print(f"  Lift: {lift_duration:.1f}s ({lift_beeps} beeps: {lift_motion} Motion + {lift_finish} Finish)")
    print(f"  Lower: {lower_duration:.1f}s ({lower_beeps} beeps: {lower_motion} Motion + {lower_finish} Finish)\n")


def get_next_input():
    global input_data, current_data_index, data_loaded, loop_mode

    if not data_loaded or len(input_data) == 0:
        return [0.0, 0.0, 0.68, 0.68, 'lift']

    if current_data_index >= len(input_data):
        if loop_mode:
            current_data_index = 0
            print(f"Loop completed, restarting... (Duration: {benchpress_duration}s)")
            print(f"Resting for {resting_time} seconds...\n")
            time.sleep(resting_time)
        else:
            current_data_index = len(input_data) - 1

    current_input = input_data[current_data_index]
    current_data_index += 1

    return current_input


def debug_model_loading():
    print("\n=== 系统兼容性检查 ===")

    import platform
    import sys

    print(f"操作系统: {platform.platform()}")
    print(f"Python版本: {sys.version}")
    print(f"系统架构: {platform.architecture()}")

    # 检查OpenSim版本
    try:
        print(f"OpenSim版本: {opensim.__version__}")
    except:
        print("无法获取OpenSim版本")

    # 检查可能的环境变量
    env_vars = ['PATH', 'OPENSIM_HOME', 'SIMBODY_HOME']
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        if len(value) > 100:  # 避免输出太长的路径
            value = value[:100] + "..."
        print(f"{var}: {value}")

    print("\n=== 硬件问题排查 ===")
    try:
        import psutil
        # 检查内存
        memory = psutil.virtual_memory()
        print(f"可用内存: {memory.available / (1024 ** 3):.1f} GB")
        print(f"总内存: {memory.total / (1024 ** 3):.1f} GB")

        # 检查GPU（如果可能）
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"GPU: {gpu.name}, 显存: {gpu.memoryFree}MB 空闲 / {gpu.memoryTotal}MB 总量")
        except:
            print("无法获取GPU信息")

    except ImportError:
        print("安装psutil和GPUtil以获得更详细的硬件信息")


def Visualizer():
    """Main visualization function - modified to use file data"""
    global state, model, benchpress_duration, current_phase, phase_start_time, last_beep_time
    global current_phase_beep_times, current_beep_index
    global lift_playback_speed, lower_playback_speed, current_phase_speed

    # debug_model_loading()

    current_phase_beep_times = []
    current_beep_index = 0
    current_phase_speed = lift_playback_speed

    # Initialize OpenSim modelE:/OpenSim/Documents/Models/whole_body
    # modelFilePath = "E:/OpenSim/Documents/Models/whole_body"
    # modelPath = modelFilePath + "/model.osim"
    modelFilePath = ("workspace")
    modelPath = modelFilePath + "/whole body model_yuetian_realtime.osim"
    model = Model(modelPath)
    geomPath = modelFilePath + "/Geometry"
    ModelVisualizer.addDirToGeometrySearchPaths(geomPath)

    model.setUseVisualizer(True)
    state = model.initSystem()
    initState = State(state)
    sviz = model.updVisualizer().updSimbodyVisualizer()
    model.getVisualizer().show(initState)
    initBenchPressQ()  # Initialize squat position

    silo = model.getVisualizer().getInputSilo()
    model.equilibrateMuscles(state)
    model.realizeDynamics(state)
    model.getVisualizer().show(state)
    MuscleSet = model.updForceSet()
    Muscles = MuscleSet.getMuscles()

    print("Benchpress visualization started. Press Ctrl+C to stop.")
    loop_count = 0
    loop_start_time = time.time()
    cycle_start_time = time.time()

    # 初始化音效状态
    current_phase = None
    phase_start_time = 0
    last_beep_time = 0
    is_first_frame_of_cycle = True
    current_phase_beep_times = []  # 重置音效时间点
    current_beep_index = 0

    while True:
        try:
            frame_start = time.time()
            prev_index = current_data_index
            current_input = get_next_input()
            current_phase_name = current_input[4] if len(current_input) > 4 else 'lift'

            # print(current_input)
            if prev_index > 0 and current_data_index == 0:
                loop_count += 1
                elapsed = time.time() - loop_start_time

                # 重置所有计时器和状态
                loop_start_time = time.time()
                cycle_start_time = time.time()
                current_phase = None
                phase_start_time = 0
                last_beep_time = 0
                current_phase_beep_times = []  # 重置音效时间点
                is_first_frame_of_cycle = True
                current_beep_index = 0

            # 计算当前循环经过的时间
            cycle_elapsed_time = time.time() - cycle_start_time
            # 检查并播放提示音
            check_and_play_beep(current_phase_name, cycle_elapsed_time, prev_index, is_first_frame_of_cycle)

            # Act = calculate_Activation(current_input[:4])
            updStateQ(current_input[:4])
            # updStateAct(Act)  # 更新肌肉激活状态

            model.updForceSet()
            model.updControls(state)
            model.realizeDynamics(state)
            model.getVisualizer().show(state)

            frame_elapsed = time.time() - frame_start
            sleep_time = max(0, current_phase_speed - frame_elapsed)
            time.sleep(sleep_time)
            # print(sleep_time)

        except KeyboardInterrupt:
            print("Visualization stopped by user")
            break
        except Exception as e:
            print(f"Error in visualization: {e}")
            break


def main():
    """Main function to run the modified squat visualization"""
    global metronome, benchpress_duration
    # 初始化节拍器
    WavPath = 'workspace'
    metronome = MetronomePlayer(WavPath)

    # print("Starting Benchpress visualization with file input...")
    # if len(sys.argv) > 1:
    #     try:
    #         global benchpress_duration
    #         benchpress_duration = float(sys.argv[1])
    #     except ValueError:
    #         pass
    # else:
    #     try:
    #         duration_input = input("Enter bench press duration in seconds (default 10): ").strip()
    #         if duration_input:
    #             benchpress_duration = float(duration_input)
    #     except:
    #         pass

    lift_speed_ratio = lift_playback_speed
    lower_speed_ratio = lower_playback_speed
    # Start data loading in a separate thread
    data_thread = threading.Thread(target=DataLoader)
    data_thread.daemon = True
    data_thread.start()

    while not data_loaded:
        time.sleep(0.1)

    set_benchpress_duration(benchpress_duration)

    # 设置阶段速度
    set_phase_speeds(lift_speed_ratio, lower_speed_ratio)

    time.sleep(0.5)

    # Start visualization
    try:
        Visualizer()
    except KeyboardInterrupt:
        print("Program stopped by user")
    except Exception as e:
        print(f"Program error: {e}")


if __name__ == "__main__":
    # deep_patch_numpy_typing()
    main()
