"""
多负载数据处理流水线（瘦身版）。

核心职责：
  - run(): 固定负载数据加载 → 对齐 → 特征注入 → 切片
  - run_vload(): 委托给 VLoadPipeline
  - 可视化代理：转发给 CurvePlotter
  - 热力图生成
  - 变负载优化

特征注入逻辑已拆分到 analysis/feature_injector.py
变负载处理已拆分到 vload_pipeline.py
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
from digitaltwin.analysis.feature_injector import (
    inject_emg_features, inject_xsens_features,
    compute_mdf_for_results, compute_segmented_mdf_for_results,
)
from digitaltwin.analysis.rbf_fitting import (
    fit_activation_map, fit_activation_map_3d,
    save_rbf_params, load_rbf_params,
    compute_rmse_percentage, compute_rmse_by_load,
)
from digitaltwin.analysis.variable_load import generate_variable_load
from digitaltwin.vload_pipeline import VLoadPipeline
from digitaltwin.visualization.plot_curves import CurvePlotter
from digitaltwin.visualization.heatmap import (
    plot_activation_3d, draw_heatmap_2d, draw_load_sensitivity_heatmap_2d,
)
from digitaltwin.visualization.variable_load_plot import plot_variable_load_result


class MultiLoadPipeline:
    """
    多负载数据处理流水线。
    统一调度数据加载、对齐、特征注入、切片和可视化。
    """

    def __init__(self, subject):
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
        self.vload_results = {}
        self.debug = False

    # ==================== 核心流水线 ====================

    def run(self, robot_files=None, include_xsens=True):
        """
        执行完整的多负载数据处理流水线。

        Parameters
        ----------
        robot_files : dict, optional
            负载文件字典，默认使用 subject.modeling_data
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
            if include_xsens and xsens_file:
                xsens_data = XsensProcessor.process(
                    xsens_file, load_weight, self.subject.folder,
                    xsens_folder=self.subject.modeling_xsens_folder)
                result['xsens_data'] = xsens_data

            # 4. 对齐机器人和 EMG 数据
            aligned = self.aligner.align_robot_emg(robot_data, emg_data)

            # 5. 注入 EMG 特征（MDF + RMS）
            aligned = inject_emg_features(
                aligned, emg_data, self.subject.emg_fs)

            # 6. 注入 Xsens 关节角度
            if include_xsens and xsens_file and result.get('xsens_data'):
                aligned = inject_xsens_features(
                    aligned, result['xsens_data'], start_time=start_time)

            result['aligned_data'] = aligned

            # 7. 运动分割
            cutted = self.aligner.cut_aligned_data(aligned)
            result['cutted_data'] = cutted

            # 8. 计算平均曲线
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

    # ==================== 变负载（委托给 VLoadPipeline） ====================

    def run_vload(self):
        """加载并处理变负载实验数据（委托给 VLoadPipeline）"""
        vload = VLoadPipeline(
            self.subject,
            emg_processor=self.emg_processor,
            debug=self.debug)
        self.vload_results = vload.run()
        return self.vload_results

    # ==================== 可视化代理 ====================

    def plot(self, save_path=None):
        """绘制平均曲线"""
        for load_weight, result in self.results.items():
            if 'average_data' in result and result['average_data']:
                self.plotter.plot_average_curves(
                    result['average_data'], save_path=save_path)

    def visualize_alignment(self, **kwargs):
        self.plotter.visualize_alignment(
            results=self.results, subject=self.subject, **kwargs)

    def visualize_movement_segments(self, **kwargs):
        self.plotter.visualize_movement_segments(
            results=self.results, subject=self.subject, **kwargs)

    def visualize_test_3d_scatter(self, **kwargs):
        self.plotter.visualize_test_3d_scatter(
            results=self.results, subject=self.subject, **kwargs)

    def visualize_muscle_analysis(self, **kwargs):
        self.plotter.visualize_muscle_analysis(
            results=self.results, subject=self.subject, **kwargs)

    def visualize_analyze_kinematic_emg_errors_by_position(self, **kwargs):
        return self.plotter.visualize_analyze_kinematic_emg_errors_by_position(
            results=self.results, subject=self.subject, **kwargs)

    def analyze_muscle_kinematic_errors_individual(self, **kwargs):
        return self.plotter.analyze_muscle_kinematic_errors_individual(
            results=self.results, subject=self.subject, **kwargs)

    # ==================== 数据访问 ====================

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
            mask = self.aligned_data['load'].isin(
                [float(lw) for lw in load_weights])
            data = self.aligned_data.loc[mask, available_cols].copy()
        if emg_col in data.columns:
            data.rename(columns={emg_col: 'emg_activation'}, inplace=True)
        return data

    # ==================== MDF 计算（委托给 feature_injector） ====================

    def compute_median_frequencies(self, muscles=None,
                                   window_size=256, overlap=128):
        """对每个负载、每块肌肉计算 MDF 时间序列"""
        if not self.results:
            self._log("请先调用 run() 加载数据。")
            return {}
        musc_label = muscles or self.subject.musc_label
        return compute_mdf_for_results(
            self.results, musc_label, self.subject.emg_fs,
            window_size=window_size, overlap=overlap)

    def compute_segmented_mdf(self, muscles=None,
                              window_size=256, overlap=128):
        """计算运动切片后的 MDF，并与位置对齐"""
        if not self.results:
            self._log("请先调用 run() 加载数据。")
            return {}
        musc_label = muscles or self.subject.musc_label
        return compute_segmented_mdf_for_results(
            self.results, musc_label, self.subject.emg_fs,
            window_size=window_size, overlap=overlap)

    # ==================== 热力图 / RBF 拟合 ====================

    def load_training_robot_data(self):
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

    def _collect_cutted_data(self, movement_types=None):
        """
        从 self.results 收集切片数据并按运动阶段过滤。

        Parameters
        ----------
        movement_types : list[str] or None
            要保留的运动阶段列表，如 ['upward'], ['downward'],
            或 ['upward', 'downward']。
            None 表示不过滤，保留所有阶段。

        Returns
        -------
        pd.DataFrame or None
        """
        if not self.results:
            self._log("请先调用 run() 加载数据。")
            return None

        frames = []
        for load_weight, result in self.results.items():
            cd = result.get('cutted_data')
            if cd is None or (hasattr(cd, '__len__') and len(cd) == 0):
                continue
            if isinstance(cd, list):
                cd = pd.concat(cd, ignore_index=True)
            df = cd.copy()
            if 'load' not in df.columns:
                df['load'] = float(load_weight)
            frames.append(df)

        if not frames:
            self._log("没有可用的切片数据。")
            return None

        combined = pd.concat(frames, ignore_index=True)

        if movement_types is not None and 'movement_type' in combined.columns:
            before = len(combined)
            combined = combined[combined['movement_type'].isin(movement_types)]
            self._log(f"运动阶段过滤 {movement_types}: {before} -> {len(combined)} 行")

        if len(combined) == 0:
            self._log("过滤后没有剩余数据。")
            return None

        return combined

    def generate_heatmaps(self, muscles=None, save_dir=None,
                          data_len=50, sigma=1.0,
                          num_centers=20, fit_3d=False,
                          movement_types=None):
        """
        生成肌肉激活热力图。

        Parameters
        ----------
        muscles : list[str], optional
            要生成热力图的肌肉列表，默认使用全部。
        save_dir : str, optional
            保存路径。
        data_len, sigma, num_centers : RBF 拟合参数。
        fit_3d : bool
            是否额外拟合 3D（含速度维度）。
        movement_types : list[str] or None
            使用哪些运动阶段的切片数据。
            默认 None -> 仅使用上升阶段 ['upward']。
            传入 ['upward', 'downward'] 可同时使用两个阶段。
        """
        if movement_types is None:
            movement_types = ['upward']

        # 确保已有 results
        if not self.results:
            self.run(include_xsens=False)

        data = self._collect_cutted_data(movement_types=movement_types)
        if data is None:
            self._log("无法加载切片数据，热力图生成终止。")
            return {}

        # 打印切片数据的高度范围（方便填写 heatmap_settings.height_range）
        if 'pos_l' in data.columns:
            h_min = round(float(data['pos_l'].min()), 4)
            h_max = round(float(data['pos_l'].max()), 4)
            print(f'height_range: [{h_min}, {h_max}]')

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
                num_centers=num_centers, sigma=sigma, data_len=data_len,
                height_range=self.subject.height_range)
            all_params[musc] = params

            plot_activation_3d(
                data, params, pos_col='pos_l', load_col=load_col,
                emg_col=emg_col, label=musc, result_folder=save_dir)
            draw_heatmap_2d(params, label=musc, result_folder=save_dir)
            draw_load_sensitivity_heatmap_2d(
                params, label=musc, result_folder=save_dir)

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

    # ==================== 变负载优化 ====================

    def run_variable_load_optimization(self, variable_mode=1):
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