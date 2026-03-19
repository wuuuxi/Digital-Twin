import pygame
import time
import os
from digitaltwin.utils.logger import beauty_print

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
            beauty_print(f"  Failed to initialize pygame: {e}", type='warning')
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
                beauty_print(f"  Failed to load sounds: {e}", type='warning')
        else:
            beauty_print("  Sound files not found", type='warning')

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
            # print(f"[{timestamp}] 🔔 {sound_type} ♪ ({frequency}Hz)")
            print(f"[{timestamp}] 🔔 {sound_type} ♪")
        except Exception as e:
            beauty_print(f"[error: {timestamp}] 🔔 {sound_type} (error: {e})", type='warning')

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

    def reset(self):
        self.current_phase = None
        self.phase_start_time = 0
        self.last_beep_time = 0
        self.current_phase_beep_times = []
        self.current_beep_index = 0
        self.phase_duration = 0