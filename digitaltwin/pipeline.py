"""
多负载数据处理流水线（瘦身版）。

核心职责：
  - run(): 固定负载数据加载 → 对齐 → 特征注入 → 切片
  - run_vload(): 委托给 VLoadPipeline
  - 可视化代理：转发给 CurvePlotter
  - 热力图生成：委托给 HeatmapGenerator
  - 变负载优化

特征注入逻辑已拆分到 analysis/feature_injector.py
变负载处理已拆分到 vload_pipeline.py
热力图生成已拆分到 analysis/heatmap/heatmap_generator.py
"""
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from digitaltwin.data.robot_processor import RobotProcessor
from digitaltwin.data.emg_processor import EMGProcessor
from digitaltwin.data.xsens_processor import XsensProcessor
from digitaltwin.data.insole_processor import InsoleProcessor
from digitaltwin.analysis.alignment import DataAligner
from digitaltwin.analysis.curve_analysis import CurveAnalyzer
from digitaltwin.analysis.feature_injector import (
    inject_emg_features, inject_xsens_features,
    compute_mdf_for_results, compute_segmented_mdf_for_results,
)
from digitaltwin.analysis.heatmap.heatmap_generator import (
    HeatmapGenerator, estimate_load_from_df,
)
from digitaltwin.analysis.vload.variable_load import generate_variable_load
from digitaltwin.vload_pipeline import VLoadPipeline
from digitaltwin.visualization.plot_curves import CurvePlotter
from digitaltwin.visualization.vload.variable_load_plot import plot_variable_load_result
from digitaltwin.utils.logger import beauty_print


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
        self.heatmap_generator = HeatmapGenerator(self)

        self.results = {}
        self.aligned_data = None
        self.vload_results = {}
        self.debug = False

    # ==================== 核心流水线 ====================

    def run(self, robot_files=None, include_xsens=True,
            include_insole=False, use_insole_info_timestamp=True):
        """
        执行完整的多负载数据处理流水线。

        Parameters
        ----------
        robot_files : dict, optional
            负载文件字典，默认使用 subject.modeling_data
        include_xsens : bool
            是否处理 Xsens 数据
        include_insole : bool
            是否可选注入鞋垫 GRF 数据。
            若 True，会读取 modeling_file.data[*].insole_file_l / insole_file_r，
            并插值到 aligned_data 的 time 轴，生成 grf_l / grf_r 列。
        use_insole_info_timestamp : bool, default True
            是否使用鞋垫文件同目录 info.csv 中的 measurement_date，
            结合 robot_file 第一帧时间修正鞋垫时间轴。默认开启；
            如需退回鞋垫文件原始相对时间，可置为 False。

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
            # 不再拼 "kg"：等长 / 等速组的组名是 IM-1 / IK-0.3，写成
            # "IM-1kg" 会让日志看起来像是 1 kg 的定负载组。
            self._log(f"处理负载 {load_weight}...")
            result = self._process_single_load(
                load_weight, file_info, include_xsens, include_insole,
                use_insole_info_timestamp)
            if result is not None:
                all_results[load_weight] = result

        self.results = all_results
        self.plotter.set_results(all_results)
        self._log(f"成功处理 {len(all_results)}/{len(robot_files)} 个负载")

        if all_results:
            self._align_all_loads()

        return all_results

    @staticmethod
    def _numeric_load_value(load_weight, file_info=None):
        """把组名解析成数值负载 (kg)；解析不出来返回 nan。

        为什么不能直接 float(load_weight)：等长 / 等速组的组名是
        'IM-1' / 'IK-0.3'，float() 会抛 ValueError。而且这一步在 try 外面，
        异常会直接穿出 run()，把整个流水线（包括所有定负载组）一起打挂。

        也不能从组名里抽数字：'IM-1' 的 1 是杆高 1.0 m，
        'IK-0.3' 的 0.3 是最高速度 0.3 m/s，都不是负载。误用会把
        等长组当成 1 kg 的定负载组静静带进热力图拟合。
        这两类组的实际负载必须由受力反推，因此先给 nan。
        """
        info = file_info or {}
        for key in ('load_kg', 'load'):
            value = info.get(key)
            if value is None:
                continue
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                return f
        try:
            return float(load_weight)
        except (TypeError, ValueError):
            return float('nan')

    def _process_single_load(self, load_weight, file_info, include_xsens,
                             include_insole=False,
                             use_insole_info_timestamp=True):
        """处理单个负载的数据"""
        robot_file = file_info.get("robot_file", "")
        emg_file = file_info.get("emg_file", "")
        start_time = file_info.get("start_time", 0)
        xsens_file = file_info.get("xsens_file", None)

        result = {
            'load_weight': load_weight,
            'load_value': self._numeric_load_value(load_weight, file_info),
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
                turn_position=self.subject.turn_position,
                load_value=result['load_value'])
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

            # 6. 可选注入其他传感器，例如鞋垫 GRF
            if include_insole:
                aligned = self._inject_insole_grf(
                    aligned, file_info, load_weight,
                    use_insole_info_timestamp=use_insole_info_timestamp)

            # 7. 注入 Xsens 关节角度
            if include_xsens and xsens_file and result.get('xsens_data'):
                aligned = inject_xsens_features(
                    aligned, result['xsens_data'], start_time=start_time)

            result['aligned_data'] = aligned

            # 8. 运动分割
            cutted = self.aligner.cut_aligned_data(aligned)
            result['cutted_data'] = cutted

            # 9. 计算平均曲线（切片为空时跳过，不能让异常把整组数据打掉）
            if cutted is None or (hasattr(cutted, '__len__') and len(cutted) == 0):
                beauty_print(
                    '负载 {}：切片为空，跳过平均曲线计算。该组仍保留 robot_data / '
                    'aligned_data，可用力阈值窗口（result_analysis.get_action_windows）'
                    '继续分析。'.format(load_weight),
                    type="warning")
                result['average_data'] = {}
            else:
                result['average_data'] = self.curve_analyzer.process_for_curves(cutted)

            result['metadata']['load_weight'] = load_weight
            result['metadata']['processing_time'] = (
                datetime.now().isoformat())
            return result

        except Exception as e:
            self._log(f"负载 {load_weight}kg 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _resolve_modeling_insole_path(self, insole_file):
        """解析 modeling_file 中的鞋垫文件路径。"""
        if not insole_file:
            return None
        if os.path.isabs(insole_file):
            return insole_file if os.path.exists(insole_file) else None

        modeling = self.subject.config.get('modeling_file', {})
        insole_folder = modeling.get('insole_folder', 'Sorted')
        if os.path.isabs(insole_folder):
            candidates = [
                os.path.join(insole_folder, insole_file),
                os.path.join(self.subject.folder, insole_file),
            ]
        else:
            candidates = [
                os.path.join(self.subject.folder, insole_folder, insole_file),
                os.path.join(self.subject.folder, insole_file),
            ]

        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _inject_insole_grf(self, aligned, file_info, load_weight,
                           use_insole_info_timestamp=True):
        """
        将鞋垫 GRF 数据插值到 aligned_data 时间轴。

        生成列：
          - grf_l: 左脚 GRF，+Y 向上，单位 N
          - grf_r: 右脚 GRF，+Y 向上，单位 N
        """
        if aligned is None or 'time' not in aligned.columns:
            return aligned

        out = aligned.copy()
        target_times = out['time'].values.astype(float)

        for side, key, col in [
            ('L', 'insole_file_l', 'grf_l'),
            ('R', 'insole_file_r', 'grf_r'),
        ]:
            insole_file = file_info.get(key)
            if not insole_file:
                self._log(f"负载 {load_weight}kg: 无 {key}，跳过 {col}")
                continue

            insole_path = self._resolve_modeling_insole_path(insole_file)
            if insole_path is None:
                self._log(f"负载 {load_weight}kg: 鞋垫文件不存在 {insole_file}")
                continue

            t_s, f_s = InsoleProcessor.load(
                insole_path,
                verbose=self.debug,
                use_info_timestamp=use_insole_info_timestamp,
                robot_file=file_info.get('robot_file'),
                robot_folder=self.subject.modeling_robot_folder,
                folder=self.subject.folder)
            if t_s is None or f_s is None:
                self._log(f"负载 {load_weight}kg: {side} 鞋垫数据读取失败")
                continue

            out[col] = InsoleProcessor.resample(t_s, f_s, target_times)
            self._log(
                f"负载 {load_weight}kg: 已注入 {col} "
                f"[{np.nanmin(out[col]):.1f}, {np.nanmax(out[col]):.1f}] N")

        return out

    def _align_all_loads(self):
        """对齐所有负载的数据并保存"""
        all_aligned = []
        for load_weight, result in self.results.items():
            if result['aligned_data'] is not None:
                df = result['aligned_data'].copy()
                df['load_weight'] = load_weight
                df['load_value'] = result.get(
                    'load_value', self._numeric_load_value(load_weight))
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

    def visualize_load_estimation(self, movement_types=None, **kwargs):
        """可视化估算负载（三联图）：位置-速度 / 位置-估算负载 / 位置-交互力均值。"""
        if movement_types is None:
            movement_types = ['upward']
        cutted = self._collect_cutted_data(movement_types=movement_types)
        if cutted is not None:
            cutted = cutted.copy()
            cutted['estimated_load'] = MultiLoadPipeline.estimate_load_from_df(cutted)
        self.plotter.plot_load_estimation(
            cutted=cutted, subject=self.subject, **kwargs)

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
            wanted = []
            for lw in load_weights:
                try:
                    wanted.append(float(lw))
                except (TypeError, ValueError):
                    wanted.append(lw)
            mask = self.aligned_data['load'].isin(wanted)
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
        return self.heatmap_generator.collect_cutted_data(
            movement_types=movement_types)

    def generate_heatmaps(self, muscles=None, save_dir=None,
                          data_len=50, sigma=1.0,
                          num_centers=20, fit_3d=False,
                          movement_types=None,
                          pspline_n_basis_h=20,
                          pspline_n_basis_l=10,
                          pspline_degree=3,
                          pspline_lambda_h=0.1,
                          pspline_lambda_l=1.0,
                          pspline_solver='auto',
                          pspline_max_iter=2000):
        """
        生成肌肉激活热力图（委托给 HeatmapGenerator）。

        默认主曲面使用 2D 张量积单调 P-spline 拟合，同时跑 RBF 基线对比，
        并输出 RMSE 报告。参数说明见 HeatmapGenerator.generate_heatmaps()。
        """
        return self.heatmap_generator.generate_heatmaps(
            muscles=muscles, save_dir=save_dir,
            data_len=data_len, sigma=sigma, num_centers=num_centers,
            fit_3d=fit_3d, movement_types=movement_types,
            pspline_n_basis_h=pspline_n_basis_h,
            pspline_n_basis_l=pspline_n_basis_l,
            pspline_degree=pspline_degree,
            pspline_lambda_h=pspline_lambda_h,
            pspline_lambda_l=pspline_lambda_l,
            pspline_solver=pspline_solver,
            pspline_max_iter=pspline_max_iter,
            load_source='nominal')

    # ==================== 以估算负载生成热力图 ====================

    def generate_heatmaps_with_estimated_load(self, muscles=None, save_dir=None,
                                               movement_types=None, g=9.81,
                                               data_len=50, sigma=1.0,
                                               num_centers=20,
                                               pspline_n_basis_h=20,
                                               pspline_n_basis_l=10,
                                               pspline_degree=3,
                                               pspline_lambda_h=0.1,
                                               pspline_lambda_l=1.0,
                                               pspline_solver='auto',
                                               pspline_max_iter=2000):
        """
        以逐样本估算负载替代 JSON 固定负载，生成肌肉激活热力图
        （委托给 HeatmapGenerator，load_source='estimated'）。

        估算公式：
            estimated_load (kg) = (force_l + force_r) / ((acc_l + acc_r) / 2 + g)

        参数说明见 HeatmapGenerator.generate_heatmaps()。
        """
        return self.heatmap_generator.generate_heatmaps(
            muscles=muscles, save_dir=save_dir,
            movement_types=movement_types,
            data_len=data_len, sigma=sigma, num_centers=num_centers,
            pspline_n_basis_h=pspline_n_basis_h,
            pspline_n_basis_l=pspline_n_basis_l,
            pspline_degree=pspline_degree,
            pspline_lambda_h=pspline_lambda_h,
            pspline_lambda_l=pspline_lambda_l,
            pspline_solver=pspline_solver,
            pspline_max_iter=pspline_max_iter,
            load_source='estimated', g=g)

    # ==================== 变负载优化 ====================

    def run_variable_load_optimization(self, variable_mode=1,
                                       use_pspline=True, tee=None):
        """
        Parameters
        ----------
        variable_mode : int
            1=目标跟踪, 2=最小化, 3=效率（仅 RBF 路径）。
        use_pspline : bool, default True
            True 时使用 P-spline 曲面（由 generate_heatmaps 自动产出的
            {musc}_pspline_params.pkl），在 Pyomo 中通过截断幂次基作
            C² 光滑的符号求值；False 时回退到 RBF 路径。
        tee : bool or None, default None
            是否把 ipopt 的求解日志打到 stdout。None 时随 self.debug 自动开关，
            方便出现“bad status: error”之类问题时直接看到 ipopt 的退出原因。
        """
        if tee is None:
            tee = bool(self.debug)
        self._log(
            f"开始变负载优化 (use_pspline={use_pspline}, tee={tee})...")
        generate_variable_load(
            self.subject,
            variable_mode=variable_mode,
            plot_fn=plot_variable_load_result,
            use_pspline=use_pspline,
            tee=tee,
        )
        self._log("变负载优化完成。")

    # ==================== 负载估算 ====================

    @staticmethod
    def estimate_load_from_df(df, g=9.81):
        """
        根据左右交互力与加速度逐样本估算实际负载。

        公式：(force_l + force_r) / ((acc_l + acc_r) / 2 + g)

        Parameters
        ----------
        df : pd.DataFrame
            需包含 force_l, force_r, acc_l, acc_r 列。
        g : float
            重力加速度 (m/s²)，默认 9.81。

        Returns
        -------
        pd.Series
            逐样本估算负载 (kg)。
        """
        return estimate_load_from_df(df, g=g)

    # ==================== 日志 ====================

    def _log(self, msg):
        if self.debug:
            print(msg)