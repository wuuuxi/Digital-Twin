import time
import threading
import os
# import digitaltwin as dt

from digitaltwin.osim.osim_model import OpenSimModel
from digitaltwin.config_manager import ConfigManager
from digitaltwin.visualization.realtime import SpeedController, GlobalAudioScheduler
from digitaltwin.data.data_manager import DataManager

os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'


class BenchPressVisualizerApp:
    """卧推可视化主应用程序"""

    def __init__(self, config_path: str = "config.json"):

        self.config_manager = ConfigManager(config_path)
        if not self.config_manager.load_config():
            print("警告：使用默认配置")

        self.motion = self.config_manager.get_motion()

        data_settings = self.config_manager.get_data_settings()
        self.audio_settings = self.config_manager.get_audio_settings()
        self.playback_settings = self.config_manager.get_playback_settings()
        self.opensim_settings = self.config_manager.get_opensim_settings()

        self.data_manager = DataManager(data_settings, self.motion)

        self.speed_controller = None
        self.audio_scheduler = None
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
        # self.metronome = AudioCueManager(self.speed_controller, self.audio_settings)
        # self.audio_scheduler = GlobalAudioScheduler(self.speed_controller, self.audio_settings)

        self.audio_scheduler = GlobalAudioScheduler(
            self.speed_controller,
            self.audio_settings,
            self.data_manager
        )
        self.audio_scheduler.set_frame_update_callback(self.update_visualization_frame)

        self.opensim_model = OpenSimModel(self.opensim_settings, self.motion)

        self.prepare_data_lookup()

    # def run_visualization(self):
    #     """Main visualization function - modified to use file data"""
    #
    #     self.opensim_model.initBenchPressQ()  # Initialize squat position
    #     self.opensim_model.update_visualization()
    #
    #     print("Benchpress visualization started. Press Ctrl+C to stop.")
    #
    #     cycle_start_time = time.time()
    #     self.metronome.reset()
    #
    #     while True:
    #         # try:
    #         frame_start_time = time.time()  # 帧开始时间
    #         prev_index = self.data_manager.current_data_index
    #
    #         current_input = self.data_manager.get_next_input()  # 获取输入数据
    #         current_phase_name = current_input[4] if len(current_input) > 4 else 'lift'  # 获取当前阶段
    #
    #         # Detect start of a new loop
    #         if prev_index > 0 and self.data_manager.current_data_index == 0:
    #             # 重置所有计时器和状态
    #             cycle_start_time = time.time()
    #             self.metronome.reset()
    #
    #         # 计算当前循环经过的时间
    #         cycle_elapsed_time = time.time() - cycle_start_time
    #         # 检查并播放提示音
    #         self.metronome.check_and_play_beep(current_phase_name, cycle_elapsed_time)
    #
    #         self.opensim_model.updStateQ(current_input[:4])  # 更新关节角度
    #         self.opensim_model.updStateAct(current_input[:4])  # 更新肌肉激活状态
    #         self.opensim_model.update_visualization()
    #
    #         target_frame_time = self.metronome.current_phase_speed if hasattr(self.metronome,
    #                                                                           'current_phase_speed') else 0.05
    #         frame_elapsed = time.time() - frame_start_time
    #         sleep_time = max(0, target_frame_time - frame_elapsed)
    #
    #         # frame_elapsed = time.time() - frame_start_time
    #         # sleep_time = max(0, self.metronome.current_phase_speed - frame_elapsed)
    #         time.sleep(sleep_time)
    #         # print(sleep_time)
    #
    #         # except KeyboardInterrupt:
    #         #     print("Visualization stopped by user")
    #         #     break
    #         # except Exception as e:
    #         #     print(f"Error in visualization: {e}")
    #         #     break

    # def run_visualization(self):
    #     """Main visualization function - modified to use file data"""
    #
    #     self.opensim_model.initBenchPressQ()  # Initialize squat position
    #     self.opensim_model.update_visualization()
    #
    #     print("Benchpress visualization started. Press Ctrl+C to stop.")
    #
    #     # 启动全局音效调度器
    #     self.audio_scheduler.start()
    #
    #     # 主可视化循环不再处理音效
    #     while True:
    #         try:
    #             frame_start_time = time.time()  # 帧开始时间
    #             prev_index = self.data_manager.current_data_index
    #
    #             current_input = self.data_manager.get_next_input()  # 获取输入数据
    #
    #             # 检测循环重置
    #             if prev_index > 0 and self.data_manager.current_data_index == 0:
    #                 # 通知音频调度器重置（如果需要的话）
    #                 pass
    #
    #             self.opensim_model.updStateQ(current_input[:4])  # 更新关节角度
    #             self.opensim_model.updStateAct(current_input[:4])  # 更新肌肉激活状态
    #             self.opensim_model.update_visualization()
    #
    #             # 根据当前阶段决定延迟时间
    #             current_phase_name = current_input[4] if len(current_input) > 4 else 'lift'
    #
    #             # 更新音频调度器的当前阶段
    #             if hasattr(self.audio_scheduler, 'update_visual_phase'):
    #                 self.audio_scheduler.update_visual_phase(current_phase_name)
    #
    #             if current_phase_name == 'lift':
    #                 target_frame_time = self.speed_controller.lift_playback_speed
    #             else:
    #                 target_frame_time = self.speed_controller.lower_playback_speed
    #
    #             frame_elapsed = time.time() - frame_start_time
    #             sleep_time = max(0, target_frame_time - frame_elapsed)
    #
    #             # 精确控制帧率
    #             if sleep_time > 0:
    #                 time.sleep(sleep_time)
    #
    #         except KeyboardInterrupt:
    #             print("Visualization stopped by user")
    #             break
    #         except Exception as e:
    #             print(f"Error in visualization: {e}")
    #             break

    def prepare_data_lookup(self):
        """准备数据查找表，用于根据时间查找对应的数据帧"""
        print("Preparing data lookup table...")

        # 获取所有数据点
        all_data = self.data_manager.input_data

        # 计算每个数据点的时间
        total_frames = len(all_data)

        # 分离上升和下降阶段数据
        self.lift_data = [d for d in all_data if d[4] == 'lift']
        self.lower_data = [d for d in all_data if d[4] == 'lower']

        lift_frames = len(self.lift_data)
        lower_frames = len(self.lower_data)

        # 计算每帧的时间间隔
        if lift_frames > 0:
            self.lift_frame_time = self.speed_controller.lift_duration / lift_frames
        else:
            self.lift_frame_time = 0.05

        if lower_frames > 0:
            self.lower_frame_time = self.speed_controller.lower_duration / lower_frames
        else:
            self.lower_frame_time = 0.05

        print(f"Data prepared: Lift={lift_frames} frames, Lower={lower_frames} frames")
        print(f"Frame times: Lift={self.lift_frame_time:.3f}s, Lower={self.lower_frame_time:.3f}s")

    def get_data_for_time(self, cycle_time, phase):
        """根据时间和阶段获取对应的数据帧"""
        if phase == 'lift':
            if not self.lift_data:
                return [0.0, 0.0, 0.68, 0.68, 'lift']

            # 计算帧索引
            frame_index = int(cycle_time / self.lift_frame_time)
            frame_index = min(frame_index, len(self.lift_data) - 1)

            return self.lift_data[frame_index]

        else:  # lower phase
            if not self.lower_data:
                return [0.0, 0.0, 0.68, 0.68, 'lower']

            # 计算相对于下降阶段开始的时间
            lower_start_time = self.speed_controller.lift_duration
            lower_elapsed = max(0, cycle_time - lower_start_time)

            # 计算帧索引
            frame_index = int(lower_elapsed / self.lower_frame_time)
            frame_index = min(frame_index, len(self.lower_data) - 1)

            return self.lower_data[frame_index]

    def update_visualization_frame(self, cycle_time, phase):
        """音频调度器调用的帧更新函数"""
        try:
            # 获取对应时间的数据
            current_data = self.get_data_for_time(cycle_time, phase)

            # 更新模型
            self.opensim_model.update_state_q(current_data[:4])  # 更新关节角度
            self.opensim_model.update_state_activation(current_data[:4])  # 更新肌肉激活状态
            self.opensim_model.update_visualization()

            # 可选：显示当前时间信息
            # print(f"[VISUAL] Time: {cycle_time:.2f}s, Phase: {phase}, "
            #       f"Pos: {current_data[2]:.2f}m")

        except Exception as e:
            print(f"[VISUAL] Error updating frame: {e}")
            import traceback
            traceback.print_exc()

    def run_visualization(self):
        """运行可视化（现在由音频调度器驱动）"""
        self.opensim_model.initialize_pose()
        self.opensim_model.update_visualization()

        print("Starting Audio-Visual synchronized visualization...")
        print("Press Ctrl+C to stop.")

        # 启动音频调度器（它会驱动可视化）
        self.audio_scheduler.start()

        try:
            # 主线程现在只需要等待
            while True:
                # 显示当前状态
                time_info = self.audio_scheduler.get_current_time_info()
                # print(f"\rTime: {time_info['cycle_time']:.2f}s | "
                #       f"Phase: {time_info['phase'].upper()} | "
                #       f"Next beep: {time_info['next_beep_time']:.2f}s", end="")

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nVisualization stopped by user")
        except Exception as e:
            print(f"\nVisualization error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.audio_scheduler.stop()


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
