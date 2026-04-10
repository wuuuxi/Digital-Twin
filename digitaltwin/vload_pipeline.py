"""
变负载数据处理流水线。

从 pipeline.py 中拆出，负责 run_vload 的全部逻辑：
加载变负载的 robot、EMG、Xsens 数据 → 对齐 → 特征注入 → 切片。
"""
import numpy as np

from digitaltwin.data.robot_processor import RobotProcessor, RobotOriginProcessor
from digitaltwin.data.emg_processor import EMGProcessor
from digitaltwin.data.xsens_processor import XsensProcessor
from digitaltwin.analysis.alignment import DataAligner
from digitaltwin.analysis.feature_injector import (
    inject_emg_features, inject_xsens_features
)


class VLoadPipeline:
    """
    变负载数据处理流水线。

    加载并处理 subject.vload_data 中的每组变负载实验数据，
    复用 EMGProcessor、DataAligner 和 feature_injector，
    处理流程与固定负载完全一致：加载 → 对齐 → 特征注入 → 切片。

    variable_load_file.data 格式（来自 JSON 配置）：
        {"FibLon_0.2_0.5": {"robot_file": ..., "emg_file": ...,
         "target_activation": ..., "start_time": ...,
         "load_range": ..., "target_muscle": ..., "xsens_file": ...}}
    """

    def __init__(self, subject, emg_processor=None, debug=False):
        """
        Parameters
        ----------
        subject : Subject
            实验配置实例
        emg_processor : EMGProcessor, optional
            复用已有的 EMGProcessor，不传则自动创建
        debug : bool
            是否打印调试信息
        """
        self.subject = subject
        self.debug = debug
        self.emg_processor = emg_processor or EMGProcessor(
            fs=subject.emg_fs,
            musc_mvc=subject.musc_mvc,
            musc_label=subject.musc_label
        )
        self.aligner = DataAligner()
        self.vload_results = {}

    def run(self):
        """
        执行变负载数据处理流水线。

        Returns
        -------
        dict
            {label: {'aligned_data', 'cutted_data', 'emg_data',
                     'robot_data', 'xsens_data', 'metadata'}}
        """
        vload_data = self.subject.vload_data
        if not vload_data:
            self._log('配置中未找到 variable_load_file 数据')
            self.vload_results = {}
            return {}

        self._log(f'\n开始处理 {len(vload_data)} 组变负载数据...')
        vload_results = {}

        for label, params in vload_data.items():
            robot_file = params.get("robot_file", "")
            emg_file = params.get("emg_file", "")
            start_time = params.get("start_time", 0)
            target_muscle = params.get("target_muscle")
            goal_activation = params.get("target_activation")
            load_range = params.get("load_range", self.subject.load_range)
            xsens_file = params.get("xsens_file", None)

            self._log(f'\n加载变负载: {label} (目标肌肉: {target_muscle})')

            try:
                # 1. 加载机器人数据
                read_ori = self.subject.read_ori_robot
                end_time = (
                    load_range[1]
                    if load_range is not None and len(load_range) > 1
                    else -1)

                if read_ori:
                    robot_data = RobotOriginProcessor.process(
                        robot_file,
                        self.subject.vload_robot_folder,
                        self.subject.folder,
                        turn_position=self.subject.turn_position,
                        start_time=start_time,
                        end_time=end_time)
                else:
                    robot_data = RobotProcessor.process(
                        robot_file, '0',
                        self.subject.vload_robot_folder,
                        self.subject.folder,
                        turn_position=self.subject.turn_position)

                if robot_data is None:
                    self._log(f'  机器人数据加载失败: {robot_file}')
                    continue
                robot_data['load'] = 0.0
                robot_data['load_weight'] = label
                self._log(f'  机器人数据加载成功: {len(robot_data)} 样本')

                # 2. 加载 EMG 数据
                emg_data = self.emg_processor.process(
                    emg_file, label,
                    self.subject.vload_emg_folder, self.subject.folder,
                    motion_flag=self.subject.motion_flag,
                    remove_leading_zeros=self.subject.remove_leading_zeros)
                if emg_data is None:
                    self._log(f'  EMG数据加载失败: {emg_file}')
                    continue

                # 应用 emg_delay
                if start_time and start_time > 0 and emg_data.get('time') is not None:
                    emg_data['time'] = emg_data['time'] + start_time

                # 3. 加载 Xsens 数据（可选）
                xsens_data = None
                if xsens_file:
                    xsens_data = XsensProcessor.process(
                        xsens_file, label, self.subject.folder,
                        xsens_folder=self.subject.vload_xsens_folder)

                # 4. 对齐
                aligned = self.aligner.align_robot_emg(robot_data, emg_data)
                if aligned is None:
                    self._log(f'  对齐失败: {label}')
                    continue

                # 5. 注入 EMG 特征（MDF + RMS）
                aligned = inject_emg_features(
                    aligned, emg_data, self.subject.emg_fs)

                # 6. 注入 Xsens 关节角度
                if xsens_data and xsens_data.get('joint_angles') is not None:
                    aligned = inject_xsens_features(
                        aligned, xsens_data, start_time=start_time)

                # 7. 运动切片
                cutted = self.aligner.cut_aligned_data(aligned)

                vload_results[label] = {
                    'aligned_data': aligned,
                    'cutted_data': cutted,
                    'emg_data': emg_data,
                    'robot_data': robot_data,
                    'xsens_data': xsens_data,
                    'metadata': {
                        'target_muscle': target_muscle,
                        'goal_activation': goal_activation,
                        'load_range': load_range,
                    }
                }
                n_cut = len(cutted) if cutted is not None else 0
                self._log(f'  成功: aligned {len(aligned)} 样本, '
                          f'cutted {n_cut} 样本')

            except Exception as e:
                self._log(f'  变负载 {label} 处理失败: {e}')
                import traceback
                traceback.print_exc()

        self.vload_results = vload_results
        self._log(f'\n变负载处理完成: {len(vload_results)}/{len(vload_data)} 成功')
        return vload_results

    def _log(self, msg):
        if self.debug:
            print(msg)