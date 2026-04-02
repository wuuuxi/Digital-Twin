"""
Digital Twin 主入口文件。

用法：
    from digitaltwin.subject import Subject
    from digitaltwin.run import MultiLoadPipeline

    subject = Subject('20250708_BenchPress_Chenzui')
    pipeline = MultiLoadPipeline(subject)
    results = pipeline.run()
"""
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from digitaltwin.subject import Subject
from digitaltwin.data.robot_processor import RobotProcessor
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
            robot_files = self.subject.robot_files

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

        # 对齐所有负载的数据
        if all_results:
            self._align_all_loads()

        return all_results

    def _process_single_load(self, load_weight, file_info, include_xsens):
        """处理单个负载的数据"""
        robot_file, emg_file, start_time = file_info[:3]
        xsens_file = file_info[3] if len(file_info) > 3 else None

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
                self.subject.robot_folder, self.subject.folder,
                turn_position=self.subject.turn_position)
            if robot_data is None:
                self._log(f"负载 {load_weight}kg: 机器人数据处理失败")
                return None
            result['robot_data'] = robot_data
            result['metadata']['robot_samples'] = len(robot_data)

            # 2. EMG 数据
            emg_data = self.emg_processor.process(
                emg_file, load_weight,
                self.subject.emg_folder, self.subject.folder,
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

        # 如果尚未运行 pipeline，尝试从缓存加载
        cache_path = os.path.join(
            self.subject.result_folder, 'aligned_data.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.aligned_data = pickle.load(f)
            self._log(f"从缓存加载对齐数据: {cache_path}")
            return self.aligned_data

        # 否则执行完整 pipeline
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
            网格每维的采样点数（对应 rbf_fitting 的 data_len）。
        sigma : float
            RBF 核宽度参数。
        num_centers : int
            KMeans 聚类中心数。
        fit_3d : bool
            是否同时进行 3-D (position, velocity, load) 拟合。

        Returns
        -------
        dict
            {muscle_name: rbf_params_dict} 的字典。
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

            # 2-D RBF (position × load → activation)
            params = fit_activation_map(
                data, pos_col='pos_l', load_col=load_col, emg_col=emg_col,
                num_centers=num_centers, sigma=sigma, data_len=data_len)
            all_params[musc] = params

            # 可视化：3-D 散点 + 曲面图
            plot_activation_3d(
                data, pos_col='pos_l', load_col=load_col, emg_col=emg_col,
                label=musc, num_centers=num_centers, sigma=sigma,
                data_len=data_len, result_folder=save_dir)

            # 可视化：2-D 热力图（用 RBF 预测网格）
            draw_heatmap_2d(params['zi'])

            # 可选 3-D 拟合 (position × load × velocity)
            if fit_3d and 'vel_l' in data.columns:
                params_3d = fit_activation_map_3d(
                    data, 'pos_l', load_col, emg_col, 'vel_l',
                    num_centers=num_centers, sigma=sigma, data_len=data_len)
                all_params[f'{musc}_3d'] = params_3d

            # 保存 RBF 参数
            params_dir = os.path.join(save_dir, 'params')
            os.makedirs(params_dir, exist_ok=True)
            save_rbf_params(
                params['centers'], params['weights'],
                params['scaler'], params['sigma'],
                os.path.join(params_dir, f'{musc}_rbf_params.pkl'))

        # RMSE 报告
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
        """
        基于 RBF 模型运行变负载优化。

        需先执行 generate_heatmaps() 以在 result_folder 中生成
        各肌肉的 *_rbf_params.pkl 文件。优化配置读取自 subject
        的 titles, goal, epsilons, max_iter 等属性。

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

    def _log(self, msg):
        if self.debug:
            print(msg)


if __name__ == '__main__':
    # 示例用法
    # subject = Subject('config/20251009_BenchPress_Yuetian.json')
    subject = Subject('config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True
    # results = pipeline.run(include_xsens=False)

    # # 绘图 - 可选择不同的可视化函数
    # pipeline.plot()                                           # 平均曲线
    # pipeline.visualize_alignment()                            # 对齐可视化
    # pipeline.visualize_movement_segments()                    # 运动切片
    # pipeline.visualize_test_3d_scatter()                      # 3D散点图
    # pipeline.visualize_muscle_analysis()                      # 肌肉分析
    # pipeline.visualize_analyze_kinematic_emg_errors_by_position()  # 位置误差分析
    # pipeline.analyze_muscle_kinematic_errors_individual()     # 单肌肉误差分析

    # 热力图 + 变负载优化
    pipeline.generate_heatmaps()
    # pipeline.run_variable_load_optimization()
    plt.show()