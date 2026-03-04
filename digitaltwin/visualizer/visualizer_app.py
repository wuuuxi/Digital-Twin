import threading
import time
import os

from .audio import MetronomePlayer

class GlobalAudioScheduler(threading.Thread):
    """全局音效调度器 - 作为主时间控制器"""

    def __init__(self, speed_controller, audio_settings, data_manager=None):
        super().__init__()
        self.daemon = True

        self.sound_dir = audio_settings.get("sound_path", None)
        self.fixed_beep_count = audio_settings.get("fixed_beep_count", 5)

        if self.sound_dir is None:
            self.sound_dir = os.path.dirname(os.path.abspath(__file__))

        self.speed_controller = speed_controller
        self.data_manager = data_manager
        self.metronome = MetronomePlayer(self.sound_dir)

        # 同步机制
        self.current_cycle_time = 0  # 当前循环时间（秒）
        self.current_phase = 'lift'  # 当前阶段
        self.frame_update_callback = None  # 帧更新回调
        self.cycle_start_time = time.time()
        self.is_running = True

        # 音效时间表
        self.lift_schedule = []
        self.lower_schedule = []
        self.all_beep_times = []  # 所有音效时间点

        # 计算音效时间表
        self.calculate_schedule()

        # 线程安全锁
        self.lock = threading.Lock()

    def calculate_schedule(self):
        """计算音效时间表"""
        # 计算上升阶段音效
        self.lift_schedule = self._calculate_phase_beep_times(
            self.speed_controller.lift_duration,
            0, 'lift'
        )

        # 计算下降阶段音效
        self.lower_schedule = self._calculate_phase_beep_times(
            self.speed_controller.lower_duration,
            self.speed_controller.lift_duration, 'lower'
        )

        # 合并所有音效时间点
        self.all_beep_times = sorted(self.lift_schedule + self.lower_schedule)

        print(f"\n=== Audio-Visual Sync Schedule ===")
        print(f"Lift: {self.speed_controller.lift_duration:.2f}s, {len(self.lift_schedule)} beeps")
        print(f"Lower: {self.speed_controller.lower_duration:.2f}s, {len(self.lower_schedule)} beeps")
        print(f"Total cycle: {self.speed_controller.lift_duration + self.speed_controller.lower_duration:.2f}s")
        print("=" * 40)

    def _calculate_phase_beep_times(self, phase_duration, start_offset, phase_name):
        """计算阶段音效时间点"""
        if phase_duration <= 0 or self.fixed_beep_count <= 0:
            return []

        interval = phase_duration / self.fixed_beep_count
        beep_times = [start_offset + interval * (i + 1) for i in range(self.fixed_beep_count)]

        # 确保最后一个音效在阶段结束时
        if beep_times:
            phase_end = start_offset + phase_duration
            if abs(beep_times[-1] - phase_end) > 0.01:
                beep_times[-1] = phase_end

        return beep_times

    def set_frame_update_callback(self, callback):
        """设置帧更新回调函数"""
        self.frame_update_callback = callback

    def get_current_time_info(self):
        """获取当前时间信息（线程安全）"""
        with self.lock:
            return {
                'cycle_time': self.current_cycle_time,
                'phase': self.current_phase,
                'next_beep_time': self._get_next_beep_time(),
                'beep_index': self._get_current_beep_index()
            }

    def _get_next_beep_time(self):
        """获取下一个音效时间"""
        if not self.all_beep_times:
            return None

        current_time = self.current_cycle_time
        for beep_time in self.all_beep_times:
            if beep_time > current_time:
                return beep_time

        return self.all_beep_times[0]  # 回到第一个

    def _get_current_beep_index(self):
        """获取当前音效索引"""
        if not self.all_beep_times:
            return -1

        current_time = self.current_cycle_time
        for i, beep_time in enumerate(self.all_beep_times):
            if beep_time > current_time:
                return max(0, i - 1)

        return len(self.all_beep_times) - 1

    def run(self):
        """主调度循环"""
        print("Audio-Visual Sync Scheduler started")

        beep_index = 0
        last_update_time = time.time()
        self.cycle_start_time = time.time()

        while self.is_running:
            try:
                # 更新当前循环时间
                current_real_time = time.time()
                self.current_cycle_time = current_real_time - self.cycle_start_time

                # 确定当前阶段
                if self.current_cycle_time < self.speed_controller.lift_duration:
                    new_phase = 'lift'
                else:
                    new_phase = 'lower'

                # 阶段切换时更新
                if new_phase != self.current_phase:
                    self.current_phase = new_phase
                    # print(f"[SYNC] Phase changed to {self.current_phase.upper()} at {self.current_cycle_time:.2f}s")

                # 检查音效播放
                if beep_index < len(self.all_beep_times):
                    next_beep_time = self.all_beep_times[beep_index]

                    if self.current_cycle_time >= next_beep_time - 0.005:  # 5ms容差
                        # 确定音效类型
                        if beep_index == len(self.lift_schedule) - 1 or beep_index == len(self.all_beep_times) - 1:
                            frequency = 1200
                            sound_type = "Finish"
                        else:
                            frequency = 800
                            sound_type = "Motion"

                        # 播放音效
                        self.metronome.play(frequency=frequency, duration=0.15, sound_type=sound_type)

                        # print(f"[SYNC] {sound_type} at {self.current_cycle_time:.2f}s "
                        #       f"(phase: {self.current_phase}, index: {beep_index + 1}/{len(self.all_beep_times)})")

                        beep_index += 1

                # 调用帧更新回调（如果设置）
                if self.frame_update_callback:
                    self.frame_update_callback(self.current_cycle_time, self.current_phase)

                # 检查循环是否完成
                total_cycle_time = self.speed_controller.lift_duration + self.speed_controller.lower_duration
                if self.current_cycle_time >= total_cycle_time:
                    # 重置循环
                    self.cycle_start_time = time.time()
                    self.current_cycle_time = 0
                    beep_index = 0
                    self.current_phase = 'lift'
                    # print("[SYNC] Cycle completed, restarting")

                # 控制更新频率（约100Hz）
                elapsed = time.time() - last_update_time
                sleep_time = max(0, 0.01 - elapsed)  # 10ms间隔
                if sleep_time > 0:
                    time.sleep(sleep_time)

                last_update_time = time.time()

            except Exception as e:
                print(f"[SYNC] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def stop(self):
        """停止调度器"""
        self.is_running = False

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

# class VisualizerApp:
#     """卧推可视化主应用程序"""
#
#     def __init__(self, config_path: str = "config.json"):
#
#         self.config_manager = ConfigManager(config_path)
#         if not self.config_manager.load_config():
#             print("警告：使用默认配置")
#
#         self.motion = self.config_manager.get_motion()
#
#         data_settings = self.config_manager.get_data_settings()
#         self.audio_settings = self.config_manager.get_audio_settings()
#         self.playback_settings = self.config_manager.get_playback_settings()
#         self.opensim_settings = self.config_manager.get_opensim_settings()
#
#         self.data_manager = DataManager(data_settings)
#
#         self.speed_controller = None
#         self.audio_scheduler = None
#         self.opensim_model = None
#
#         self.loop_mode = True
#         self.is_running = False
#
#
#     def initialize(self):
#         """初始化应用程序"""
#         print("Initializing BenchPress Visualizer...")
#
#         # Start data loading in a thread
#         data_thread = threading.Thread(target=self.data_manager.load_data)
#         data_thread.daemon = True
#         data_thread.start()
#
#         # Wait until data is loaded and original durations are known
#         while not self.data_manager.data_loaded:
#             time.sleep(0.1)
#
#         # 设置阶段速度
#         self.speed_controller = SpeedController(self.data_manager, self.playback_settings)
#         # self.metronome = AudioCueManager(self.speed_controller, self.audio_settings)
#         # self.audio_scheduler = GlobalAudioScheduler(self.speed_controller, self.audio_settings)
#
#         self.audio_scheduler = GlobalAudioScheduler(
#             self.speed_controller,
#             self.audio_settings,
#             self.data_manager
#         )
#         self.audio_scheduler.set_frame_update_callback(self.update_visualization_frame)
#
#         self.opensim_model = OpenSimModel(self.opensim_settings, self.motion)
#
#         self.prepare_data_lookup()
#
#     def prepare_data_lookup(self):
#         """准备数据查找表，用于根据时间查找对应的数据帧"""
#         print("Preparing data lookup table...")
#
#         # 获取所有数据点
#         all_data = self.data_manager.input_data
#
#         # 计算每个数据点的时间
#         total_frames = len(all_data)
#
#         # 分离上升和下降阶段数据
#         self.lift_data = [d for d in all_data if d[4] == 'lift']
#         self.lower_data = [d for d in all_data if d[4] == 'lower']
#
#         lift_frames = len(self.lift_data)
#         lower_frames = len(self.lower_data)
#
#         # 计算每帧的时间间隔
#         if lift_frames > 0:
#             self.lift_frame_time = self.speed_controller.lift_duration / lift_frames
#         else:
#             self.lift_frame_time = 0.05
#
#         if lower_frames > 0:
#             self.lower_frame_time = self.speed_controller.lower_duration / lower_frames
#         else:
#             self.lower_frame_time = 0.05
#
#         print(f"Data prepared: Lift={lift_frames} frames, Lower={lower_frames} frames")
#         print(f"Frame times: Lift={self.lift_frame_time:.3f}s, Lower={self.lower_frame_time:.3f}s")
#
#     def get_data_for_time(self, cycle_time, phase):
#         """根据时间和阶段获取对应的数据帧"""
#         if phase == 'lift':
#             if not self.lift_data:
#                 return [0.0, 0.0, 0.68, 0.68, 'lift']
#
#             # 计算帧索引
#             frame_index = int(cycle_time / self.lift_frame_time)
#             frame_index = min(frame_index, len(self.lift_data) - 1)
#
#             return self.lift_data[frame_index]
#
#         else:  # lower phase
#             if not self.lower_data:
#                 return [0.0, 0.0, 0.68, 0.68, 'lower']
#
#             # 计算相对于下降阶段开始的时间
#             lower_start_time = self.speed_controller.lift_duration
#             lower_elapsed = max(0, cycle_time - lower_start_time)
#
#             # 计算帧索引
#             frame_index = int(lower_elapsed / self.lower_frame_time)
#             frame_index = min(frame_index, len(self.lower_data) - 1)
#
#             return self.lower_data[frame_index]
#
#     def update_visualization_frame(self, cycle_time, phase):
#         """音频调度器调用的帧更新函数"""
#         try:
#             # 获取对应时间的数据
#             current_data = self.get_data_for_time(cycle_time, phase)
#
#             # 更新模型
#             self.opensim_model.updStateQ(current_data[:4])  # 更新关节角度
#             # self.opensim_model.updStateAct(current_data[:4])
#             self.opensim_model.update_visualization()
#
#             # 可选：显示当前时间信息
#             # print(f"[VISUAL] Time: {cycle_time:.2f}s, Phase: {phase}, "
#             #       f"Pos: {current_data[2]:.2f}m")
#
#         except Exception as e:
#             print(f"[VISUAL] Error updating frame: {e}")
#             import traceback
#             traceback.print_exc()
#
#     def run_visualization(self):
#         """运行可视化（现在由音频调度器驱动）"""
#         # self.opensim_model.initBenchPressQ()
#         self.opensim_model.initQ()
#         self.opensim_model.update_visualization()
#
#         print("Starting Audio-Visual synchronized visualization...")
#         print("Press Ctrl+C to stop.")
#
#         # 启动音频调度器（它会驱动可视化）
#         self.audio_scheduler.start()
#
#         try:
#             # 主线程现在只需要等待
#             while True:
#                 # 显示当前状态
#                 time_info = self.audio_scheduler.get_current_time_info()
#                 # print(f"\rTime: {time_info['cycle_time']:.2f}s | "
#                 #       f"Phase: {time_info['phase'].upper()} | "
#                 #       f"Next beep: {time_info['next_beep_time']:.2f}s", end="")
#
#                 time.sleep(0.1)
#
#         except KeyboardInterrupt:
#             print("\nVisualization stopped by user")
#         except Exception as e:
#             print(f"\nVisualization error: {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             self.audio_scheduler.stop()