"""
多负载数据处理流水线。
统一调度 RobotProcessor、EMGProcessor、XsensProcessor、
DataAligner、CurveAnalyzer 和 CurvePlotter，完成从数据加载到
曲线分析、RBF 热力图生成和变负载优化的全流程。
"""
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from digitaltwin.data.robot_processor import RobotProcessor, RobotOriginProcessor
from digitaltwin.data.emg_processor import EMGProcessor
from digitaltwin.data.xsens_processor import XsensProcessor
from digitaltwin.analysis.alignment import DataAligner
from digitaltwin.analysis.curve_analysis import CurveAnalyzer
from digitaltwin.analysis.rbf_fitting import (
    fit_activation_map, fit_activation_map_3d,
    save_rbf_params, load_rbf_params,
    compute_rmse_percentage, compute_rmse_by_load,
)
from digitaltwin.analysis.variable_load import generate_variable_load
from digitaltwin.visualization.plot_curves import CurvePlotter
from digitaltwin.visualization.heatmap import (
    plot_activation_3d, draw_heatmap_2d,
)
from digitaltwin.visualization.variable_load_plot import plot_variable_load_result


class MultiLoadPipeline:
    """
    多负载数据处理流水线。
    统一调度 RobotProcessor、EMGProcessor、XsensProcessor、
    DataAligner、CurveAnalyzer 和 CurvePlotter，完成从数据加载到曲线分析的全流程。
    """

    def __init__(self, subject):
        """
        Parameters
        ----------
        subject : Subject
            实验配置实例
        """
        self.subject = subject
        self.emg_processor = EMGProcessor(
            fs=subject.emg_fs,
            musc_mvc=subject.musc_mvc,
            musc_label=subject.musc_label
        )
        self.aligner = DataAligner()
        self.curve_analyzer = CurveAnalyzer()
        self.plotter = CurvePlotter(subject=subject)

        self.results = {}
        self.aligned_data = None
        self.debug = False

    # ==================== 核心流水线 ====================

    def run(self, robot_files=None, include_xsens=True):
        """
        执行完整的多负载数据处理流水线。

        Parameters
        ----------
        robot_files : dict, optional
            负载文件字典，格式: {'weight': [robot_file, emg_file, start_time]}
        include_xsens : bool
            是否处理 Xsens 数据

        Returns
        -------
        dict
            每个负载的处理结果
        """
        if robot_files is None:
            robot_files = self.subject.modeling_data

        self._log(f"开始处理 {len(robot_files)} 个负载的数据...")
        all_results = {}

        for load_weight, file_info in robot_files.items():
            self._log(f"处理负载 {load_weight}kg...")
            result = self._process_single_load(
                load_weight, file_info, include_xsens)
            if result is not None:
                all_results[load_weight] = result

        self.results = all_results
        self.plotter.set_results(all_results)
        self._log(f"成功处理 {len(all_results)}/{len(robot_files)} 个负载")

        if all_results:
            self._align_all_loads()

        return all_results

    def _process_single_load(self, load_weight, file_info, include_xsens):
        """处理单个负载的数据"""
        robot_file = file_info.get("robot_file", "")
        emg_file = file_info.get("emg_file", "")
        start_time = file_info.get("start_time", 0)
        xsens_file = file_info.get("xsens_file", None)

        result = {
            'load_weight': load_weight,
            'load_value': float(load_weight),
            'robot_data': None,
            'emg_data': None,
            'xsens_data': None,
            'aligned_data': None,
            'metadata': {}
        }

        try:
            # 1. 机器人数据
            robot_data = RobotProcessor.process(
                robot_file, load_weight,
                self.subject.modeling_robot_folder, self.subject.folder,
                turn_position=self.subject.turn_position)
            if robot_data is None:
                self._log(f"负载 {load_weight}kg: 机器人数据处理失败")
                return None
            result['robot_data'] = robot_data
            result['metadata']['robot_samples'] = len(robot_data)

            # 2. EMG 数据
            emg_data = self.emg_processor.process(
                emg_file, load_weight,
                self.subject.modeling_emg_folder, self.subject.folder,
                motion_flag=self.subject.motion_flag,
                remove_leading_zeros=self.subject.remove_leading_zeros)
            if emg_data is None:
                self._log(f"负载 {load_weight}kg: EMG数据处理失败")
                return None
            result['emg_data'] = emg_data
            result['metadata']['emg_samples'] = len(emg_data['time'])

            # 3. Xsens 数据（可选）
            if include_xsens:
                xsens_data = XsensProcessor.process(
                    xsens_file, load_weight, self.subject.folder)
                result['xsens_data'] = xsens_data

            # 4. 对齐机器人和 EMG 数据
            aligned = self.aligner.align_robot_emg(robot_data, emg_data)

            # 4.5 计算每块肌肉的 MDF 并注入 aligned_data
            #     使 MDF 与 pos_l / vel_l 等共享同一时间轴，
            #     后续 cut_aligned_data 切片后 cutted_data 中即有 mdf_* 列。
            aligned = self._inject_mdf_columns(
                aligned, emg_data, self.subject.emg_fs)

            result['aligned_data'] = aligned

            # 5. 运动分割
            cutted = self.aligner.cut_aligned_data(aligned)
            result['cutted_data'] = cutted

            # 6. 计算平均曲线
            average = self.curve_analyzer.process_for_curves(cutted)
            result['average_data'] = average

            result['metadata']['load_weight'] = load_weight
            result['metadata']['processing_time'] = (
                datetime.now().isoformat())
            return result

        except Exception as e:
            self._log(f"负载 {load_weight}kg 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _align_all_loads(self):
        """对齐所有负载的数据并保存"""
        all_aligned = []
        for load_weight, result in self.results.items():
            if result['aligned_data'] is not None:
                df = result['aligned_data'].copy()
                df['load_weight'] = load_weight
                df['load_value'] = float(load_weight)
                all_aligned.append(df)

        if all_aligned:
            self.aligned_data = pd.concat(all_aligned, ignore_index=True)
            if 'pos_l' in self.aligned_data.columns:
                self.subject.height_range = [
                    self.aligned_data['pos_l'].min(),
                    self.aligned_data['pos_l'].max()
                ]
            self._log(f"所有负载对齐完成，共 {len(self.aligned_data)} 个样本")
            self._save_aligned_data()

    def _save_aligned_data(self, save_path=None):
        """保存对齐后的数据"""
        if self.aligned_data is None:
            return
        if save_path is None:
            save_path = os.path.join(
                self.subject.result_folder, 'aligned_data.csv')
        self.aligned_data.to_csv(save_path, index=False)
        pkl_path = save_path.replace('.csv', '.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.aligned_data, f)
        self._log(f"对齐数据已保存: {save_path}")

    def plot(self, save_path=None):
        """绘制平均曲线"""
        for load_weight, result in self.results.items():
            if 'average_data' in result and result['average_data']:
                self.plotter.plot_average_curves(
                    result['average_data'], save_path=save_path)

    # ==================== 可视化接口 ====================

    def visualize_alignment(self, **kwargs):
        """可视化数据对齐"""
        self.plotter.visualize_alignment(
            results=self.results, subject=self.subject, **kwargs)

    def visualize_movement_segments(self, **kwargs):
        """可视化运动切片"""
        self.plotter.visualize_movement_segments(
            results=self.results, subject=self.subject, **kwargs)

    def visualize_test_3d_scatter(self, **kwargs):
        """3D散点图"""
        self.plotter.visualize_test_3d_scatter(
            results=self.results, subject=self.subject, **kwargs)

    def visualize_muscle_analysis(self, **kwargs):
        """肌肉分析图"""
        self.plotter.visualize_muscle_analysis(
            results=self.results, subject=self.subject, **kwargs)

    def visualize_analyze_kinematic_emg_errors_by_position(self, **kwargs):
        """基于位置的误差分析"""
        return self.plotter.visualize_analyze_kinematic_emg_errors_by_position(
            results=self.results, subject=self.subject, **kwargs)

    def analyze_muscle_kinematic_errors_individual(self, **kwargs):
        """单肌肉误差分析"""
        return self.plotter.analyze_muscle_kinematic_errors_individual(
            results=self.results, subject=self.subject, **kwargs)

    def get_muscle_data(self, muscle_name, load_weights=None):
        """获取指定肌肉的所有负载数据"""
        if self.aligned_data is None:
            return None
        emg_col = f'emg_{muscle_name}'
        if emg_col not in self.aligned_data.columns:
            print(f"未找到肌肉 {muscle_name} 的数据")
            return None
        cols = ['load', 'pos_l', 'vel_l', 'force_l', emg_col]
        available_cols = [c for c in cols if c in self.aligned_data.columns]
        if load_weights is None:
            data = self.aligned_data[available_cols].copy()
        else:
            mask = self.aligned_data['load'].isin([float(lw) for lw in load_weights])
            data = self.aligned_data.loc[mask, available_cols].copy()
        if emg_col in data.columns:
            data.rename(columns={emg_col: 'emg_activation'}, inplace=True)
        return data

    # ==================== MDF 注入 ====================

    @staticmethod
    def _inject_mdf_columns(aligned, emg_data, fs,
                            window_size=256, overlap=128):
        """
        将每块肌肉的 MDF 和 RMS 插值到 aligned_data 的时间轴上，
        作为新列 mdf_<muscle> 和 rms_<muscle> 写入 DataFrame。

        优先使用 emg_data 中预计算的 mdf_signals / rms_signals；
        若不存在则回退到从 raw_signals 重新计算。

        这样经过后续 cut_aligned_data 切片后，cutted_data 中自然
        包含 mdf_* 和 rms_* 列，与 pos_l / vel_l / emg_* 完全共享同一行索引。

        Parameters
        ----------
        aligned : pd.DataFrame
            对齐后的 DataFrame，必须包含 'time' 列。
        emg_data : dict
            EMGProcessor.process() 返回的字典，包含
            'time', 'raw_signals', 'mdf_signals', 'rms_signals' 等。
        fs : int
            EMG 采样率 (Hz)。
        window_size : int
            STFT 窗口大小（回退计算时使用）。
        overlap : int
            窗口重叠样本数（回退计算时使用）。

        Returns
        -------
        pd.DataFrame
            添加了 mdf_* 和 rms_* 列的 aligned DataFrame。
        """
        if aligned is None or emg_data is None:
            return aligned
        if 'time' not in aligned.columns:
            return aligned

        aligned_time = aligned['time'].values
        emg_time = emg_data.get('time')
        if emg_time is None or len(emg_time) == 0:
            return aligned

        t0 = emg_time[0]
        raw_signals = emg_data.get('raw_signals', {})
        mdf_signals = emg_data.get('mdf_signals', {})
        rms_signals = emg_data.get('rms_signals', {})

        for musc_name in raw_signals.keys():
            # --- MDF 注入 ---
            if musc_name in mdf_signals:
                mdf_times = mdf_signals[musc_name]['time']
                mdf_vals = mdf_signals[musc_name]['values']
            else:
                mdf_times, mdf_vals = EMGProcessor.compute_median_frequency(
                    raw_signals[musc_name], fs,
                    window_size=window_size, overlap=overlap)

            if len(mdf_times) == 0:
                aligned[f'mdf_{musc_name}'] = np.nan
            else:
                mdf_abs = mdf_times + t0
                aligned[f'mdf_{musc_name}'] = np.interp(
                    aligned_time, mdf_abs, mdf_vals,
                    left=np.nan, right=np.nan)

            # --- RMS 注入 ---
            if musc_name in rms_signals:
                rms_times = rms_signals[musc_name]['time']
                rms_vals = rms_signals[musc_name]['values']
            else:
                rms_times, rms_vals = EMGProcessor.compute_rms(
                    raw_signals[musc_name], fs)

            if len(rms_times) == 0:
                aligned[f'rms_{musc_name}'] = np.nan
            else:
                rms_abs = rms_times + t0
                aligned[f'rms_{musc_name}'] = np.interp(
                    aligned_time, rms_abs, rms_vals,
                    left=np.nan, right=np.nan)

        return aligned

    # ==================== 热力图 / RBF 拟合 ====================

    def load_training_robot_data(self):
        """
        加载并合并所有负载的训练机器人数据。

        Returns
        -------
        pd.DataFrame
            合并后的 DataFrame，包含 pos_l, vel_l, load 等列。
        """
        if self.aligned_data is not None:
            return self.aligned_data

        cache_path = os.path.join(
            self.subject.result_folder, 'aligned_data.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.aligned_data = pickle.load(f)
            self._log(f"从缓存加载对齐数据: {cache_path}")
            return self.aligned_data

        self.run(include_xsens=False)
        return self.aligned_data

    def generate_heatmaps(self, muscles=None, save_dir=None,
                          data_len=50, sigma=1.0,
                          num_centers=20, fit_3d=False):
        """
        对每块肌肉进行 RBF 拟合并生成热力图。

        Parameters
        ----------
        muscles : list[str], optional
            要分析的肌肉名称列表，默认使用 subject.musc_label。
        save_dir : str, optional
            图片 / 参数保存目录，默认 subject.result_folder/heatmap。
        data_len : int
            网格每维的采样点数。
        sigma : float
            RBF 核宽度参数。
        num_centers : int
            KMeans 聚类中心数。
        fit_3d : bool
            是否同时进行 3-D 拟合。

        Returns
        -------
        dict
            {muscle_name: rbf_params_dict}
        """
        data = self.load_training_robot_data()
        if data is None:
            self._log("无法加载训练数据，热力图生成终止。")
            return {}

        if muscles is None:
            muscles = self.subject.musc_label
        if save_dir is None:
            save_dir = os.path.join(self.subject.result_folder, 'heatmap')
        os.makedirs(save_dir, exist_ok=True)

        load_col = 'load' if 'load' in data.columns else 'load_value'

        all_params = {}
        for musc in muscles:
            emg_col = f'emg_{musc}'
            if emg_col not in data.columns:
                self._log(f"跳过肌肉 {musc}：列 {emg_col} 不存在")
                continue

            self._log(f"拟合肌肉 {musc} 的激活热力图...")

            params = fit_activation_map(
                data, pos_col='pos_l', load_col=load_col, emg_col=emg_col,
                num_centers=num_centers, sigma=sigma, data_len=data_len)
            all_params[musc] = params

            plot_activation_3d(
                data, params, pos_col='pos_l', load_col=load_col, emg_col=emg_col,
                label=musc, result_folder=save_dir)

            draw_heatmap_2d(params, label=musc, result_folder=save_dir)

            if fit_3d and 'vel_l' in data.columns:
                params_3d = fit_activation_map_3d(
                    data, 'pos_l', load_col, emg_col, 'vel_l',
                    num_centers=num_centers, sigma=sigma, data_len=data_len)
                all_params[f'{musc}_3d'] = params_3d

            params_dir = os.path.join(save_dir, 'params')
            os.makedirs(params_dir, exist_ok=True)
            save_rbf_params(
                params['centers'], params['weights'],
                params['scaler'], params['sigma'],
                os.path.join(params_dir, f'{musc}_rbf_params.pkl'))

        for musc, p in all_params.items():
            if '_3d' not in musc:
                emg_col = f'emg_{musc}'
                rmse_pct = compute_rmse_percentage(
                    data, 'pos_l', load_col, emg_col,
                    p['centers'], p['weights'], p['scaler'], p['sigma'])
                self._log(f"{musc} RMSE%: {rmse_pct:.2f}%")

        self._log(f"热力图已保存至 {save_dir}")
        return all_params

    # ==================== EMG 中值频率分析 ====================

    def compute_median_frequencies(self, muscles=None,
                                   window_size=256, overlap=128):
        """
        对每个负载、每块肌肉计算 EMG 中值频率（MDF）时间序列。

        Parameters
        ----------
        muscles : list[str], optional
            要分析的肌肉列表，默认 subject.musc_label。
        window_size : int
            STFT 窗口大小（样本数）。
        overlap : int
            窗口重叠样本数。

        Returns
        -------
        dict
            结构: {load_weight: {muscle_name: {'time': ndarray, 'mdf': ndarray}}}
        """
        if not self.results:
            self._log("请先调用 run() 加载数据。")
            return {}

        if muscles is None:
            muscles = self.subject.musc_label

        fs = self.subject.emg_fs
        mdf_results = {}

        for load_weight, result in self.results.items():
            emg_data = result.get('emg_data')
            if emg_data is None:
                continue

            mdf_results[load_weight] = {}
            for musc in muscles:
                raw_signal = emg_data['raw_signals'].get(musc)
                if raw_signal is None:
                    continue

                times, mdf = EMGProcessor.compute_median_frequency(
                    raw_signal, fs,
                    window_size=window_size, overlap=overlap)
                mdf_results[load_weight][musc] = {
                    'time': times, 'mdf': mdf
                }

            self._log(f"负载 {load_weight}kg: "
                      f"已计算 {len(mdf_results[load_weight])} 块肌肉的 MDF")

        return mdf_results

    def compute_segmented_mdf(self, muscles=None,
                              window_size=256, overlap=128):
        """
        计算运动切片后的 MDF，并与位置对齐。

        对每个负载的已切片数据，将每个运动周期的 EMG 原始信号
        分段计算 MDF，然后通过时间插值将 MDF 对齐到位置序列。

        Parameters
        ----------
        muscles : list[str], optional
            要分析的肌肉列表。
        window_size : int
            STFT 窗口大小。
        overlap : int
            窗口重叠样本数。

        Returns
        -------
        dict
            结构: {load_weight: {muscle_name: {'position': ndarray, 'mdf': ndarray}}}
        """
        if not self.results:
            self._log("请先调用 run() 加载数据。")
            return {}

        if muscles is None:
            muscles = self.subject.musc_label

        fs = self.subject.emg_fs
        seg_mdf = {}

        for load_weight, result in self.results.items():
            emg_data = result.get('emg_data')
            aligned = result.get('aligned_data')
            if emg_data is None or aligned is None:
                continue

            seg_mdf[load_weight] = {}
            # 获取对齐后的时间和位置
            if 'time' in aligned.columns and 'pos_l' in aligned.columns:
                aligned_time = aligned['time'].values
                aligned_pos = aligned['pos_l'].values
            else:
                continue

            for musc in muscles:
                raw_signal = emg_data['raw_signals'].get(musc)
                if raw_signal is None:
                    continue

                mdf_times, mdf_vals = EMGProcessor.compute_median_frequency(
                    raw_signal, fs,
                    window_size=window_size, overlap=overlap)

                if len(mdf_times) == 0:
                    continue

                # 将 MDF 时间序列插值到对齐后的时间点，获得对应位置
                emg_time = emg_data['time']
                # mdf_times 基于 EMG 采样时间，需映射到对齐时间
                # 使用 EMG 原始时间的起始偏移
                mdf_abs_times = mdf_times + (emg_time[0] if len(emg_time) > 0 else 0)

                # 插值：在对齐时间轴上找到 MDF 对应的位置
                mdf_positions = np.interp(mdf_abs_times, aligned_time, aligned_pos,
                                          left=np.nan, right=np.nan)

                # 去掉超出范围的点
                valid = ~np.isnan(mdf_positions)
                seg_mdf[load_weight][musc] = {
                    'position': mdf_positions[valid],
                    'mdf': mdf_vals[valid]
                }

        return seg_mdf

    # ==================== 变负载数据加载 ====================

    def run_vload(self):
        """
        加载并处理 subject.vload_data 中的每组变负载实验数据。
        复用本 pipeline 的 EMGProcessor、DataAligner 和 _inject_mdf_columns，
        处理流程与固定负载完全一致：加载 → 对齐 → MDF注入 → 切片。

        结果存储在 self.vload_results 中，结构与 self.results 同构：
            {label: {'aligned_data', 'cutted_data', 'emg_data',
                     'robot_data', 'metadata'}}

        variable_load_file.data 格式（来自 JSON 配置）：
            {"FibLon_0.2_0.5": {"robot_file": ..., "emg_file": ...,
             "target_activation": ..., "start_time": ...,
             "load_range": ..., "target_muscle": ...}}

        Returns
        -------
        dict
            变负载处理结果
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

            self._log(f'\n加载变负载: {label} (目标肌肉: {target_muscle})')

            try:
                # 1. 加载机器人数据
                #    根据 subject.read_ori_robot 选择处理器：
                #    - True:  RobotOriginProcessor（原始格式，skiprows=17）
                #    - False: RobotProcessor（标准格式）
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

                # 应用 emg_delay（start_time 在旧代码中即 emg_delay）
                if start_time and start_time > 0 and emg_data.get('time') is not None:
                    emg_data['time'] = emg_data['time'] + start_time

                # 3. 对齐
                aligned = self.aligner.align_robot_emg(robot_data, emg_data)
                if aligned is None:
                    self._log(f'  对齐失败: {label}')
                    continue

                # 4. 注入 MDF 列
                aligned = self._inject_mdf_columns(
                    aligned, emg_data, self.subject.emg_fs)

                # 5. 运动切片
                cutted = self.aligner.cut_aligned_data(aligned)

                vload_results[label] = {
                    'aligned_data': aligned,
                    'cutted_data': cutted,
                    'emg_data': emg_data,
                    'robot_data': robot_data,
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

    # ==================== 变负载优化 ====================

    def run_variable_load_optimization(self, variable_mode=1):
        """
        基于 RBF 模型运行变负载优化。

        Parameters
        ----------
        variable_mode : int
            优化模式：1=目标激活, 2=最小化, 3=效率最大化。
        """
        self._log("开始变负载优化...")
        generate_variable_load(
            self.subject,
            variable_mode=variable_mode,
            plot_fn=plot_variable_load_result,
        )
        self._log("变负载优化完成。")

    # ==================== 日志 ====================

    def _log(self, msg):
        if self.debug:
            print(msg)