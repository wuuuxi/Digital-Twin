"""
热力图生成器：从 pipeline.py 拆出。

统一调度肌肉激活热力图拟合（RBF 基线 + 单调 P-spline 主曲面）、
对比图生成和 RMSE 报告。

generate_heatmaps() 合并了原 MultiLoadPipeline 的两个方法：
  - generate_heatmaps()                 → load_source='nominal'
  - generate_heatmaps_with_estimated_load() → load_source='estimated'
两者共用一套 muscle 循环，区别只在负载列来源与输出路径。
"""
import os
import pickle

import numpy as np
import pandas as pd

from digitaltwin.analysis.heatmap.rbf_fitting import (
    fit_activation_map, fit_activation_map_3d,
    save_rbf_params, compute_rmse_percentage, predict_at,
)
from digitaltwin.visualization.heatmap import (
    plot_activation_3d, draw_heatmap_2d, draw_load_sensitivity_heatmap_2d,
    plot_compare_activation_3d, plot_compare_heatmap_2d,
    plot_compare_load_sensitivity_2d,
)
from digitaltwin.utils.logger import beauty_print


def _float_or_nan(value):
    """把值解析成数值负载 (kg)；解析不出来返回 nan。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return f if np.isfinite(f) else float('nan')


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
    required = ['force_l', 'force_r', 'acc_l', 'acc_r']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'estimate_load_from_df: 缺少列 {missing}')
    force_total = df['force_l'] + df['force_r']
    acc_avg = (df['acc_l'] + df['acc_r']) / 2.0
    denominator = acc_avg + g
    denominator = denominator.where(denominator.abs() > 1e-3, other=np.nan)
    return force_total / denominator


def collect_cutted_data(results, movement_types=None, log=None,
                        numeric_load=_float_or_nan):
    """
    从 results 收集切片数据并按运动阶段过滤。

    Parameters
    ----------
    results : dict
        {load_weight: result}，result 需含 'cutted_data' 与 'load_value'。
    movement_types : list[str] or None
        要保留的运动阶段列表，如 ['upward'], ['downward'],
        或 ['upward', 'downward']。None 表示不过滤。
    log : callable, optional
        调试日志回调。
    numeric_load : callable, optional
        把组名解析成数值负载的函数，默认 float 直转。

    Returns
    -------
    pd.DataFrame or None
    """
    if not results:
        if log:
            log('请先调用 run() 加载数据。')
        return None

    frames = []
    for load_weight, result in results.items():
        cd = result.get('cutted_data')
        if cd is None or (hasattr(cd, '__len__') and len(cd) == 0):
            continue
        if isinstance(cd, list):
            cd = pd.concat(cd, ignore_index=True)
        df = cd.copy()
        if 'load' not in df.columns:
            df['load'] = result.get('load_value', numeric_load(load_weight))
        frames.append(df)

    if not frames:
        if log:
            log('没有可用的切片数据。')
        return None

    combined = pd.concat(frames, ignore_index=True)

    if movement_types is not None and 'movement_type' in combined.columns:
        before = len(combined)
        # 只有 upward/downward 才属于"运动阶段"，isometric / isokinetic 是
        # 模式而非阶段：等长组的 movement_type 直接记为 'isometric'（杆不动，
        # 走力阈值切段，见 alignment._cut_by_force_threshold），没有任何
        # 向上/向下阶段。因此请求阶段过滤时，必须把这些模式的组一并保留，
        # 否则 isometric 会被 ['upward'] 之类全部过滤掉、图上不显示。
        keep_phase = combined['movement_type'].isin(movement_types)
        keep_mode = combined['movement_type'].isin(('isometric', 'isokinetic'))
        combined = combined[keep_phase | keep_mode]
        if log:
            log(f'运动阶段过滤 {movement_types}: {before} -> {len(combined)} 行')

    if len(combined) == 0:
        if log:
            log('过滤后没有剩余数据。')
        return None

    return combined


class HeatmapGenerator:
    """
    肌肉激活热力图生成器。

    持有 pipeline 引用（组合而非继承），从中读取 subject / results / debug，
    并复用其 run() 懒加载数据。默认主曲面使用 2D 张量积单调 P-spline 拟合
    （保存到 save_dir/），同时跑原始 RBF 基线作为对比（保存到 save_dir/rbf/）。

    Parameters
    ----------
    pipeline : MultiLoadPipeline
        已（或将要）持有 results 的流水线实例。
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.subject = pipeline.subject

    # ==================== 数据访问 ====================

    def collect_cutted_data(self, movement_types=None):
        """从 pipeline.results 收集切片数据并按运动阶段过滤。"""
        return collect_cutted_data(
            self.pipeline.results, movement_types=movement_types,
            log=self._log)

    def _ensure_results(self):
        """确保 pipeline 已有 results，必要时先跑固定负载流水线。"""
        if not self.pipeline.results:
            self.pipeline.run(include_xsens=False)

    # ==================== 热力图生成 ====================

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
                          pspline_max_iter=2000,
                          load_source='nominal',
                          g=9.81):
        """
        生成肌肉激活热力图。

        默认主曲面使用 2D 张量积单调 P-spline 拟合（保存到 save_dir/）；
        同时跑一个原始 RBF 基线作为对比（保存到 save_dir/rbf/）。
        每块肌肉额外生成 3 张 RBF vs P-spline 1×2 对比图：
          - {musc}_compare_3D.png
          - {musc}_compare_2D.png
          - {musc}_compare_load_sensitivity_2D.png

        Parameters
        ----------
        muscles : list[str], optional
        save_dir : str, optional
        data_len, sigma, num_centers : RBF 拟合参数。
        fit_3d : bool
            是否额外拟合 3D（含速度维度）。
        movement_types : list[str] or None
            使用哪些运动阶段的切片数据。默认 ['upward']。
        pspline_n_basis_h, pspline_n_basis_l : int
            高度 / 负载方向 B-spline basis 个数。
        pspline_degree : int
            B-spline 阶数（3=三次）。
        pspline_lambda_h, pspline_lambda_l : float
            高度 / 负载方向二阶差分平滑权重。
        pspline_solver : {'auto', 'cvxpy', 'lbfgs'}
            'auto' 优先用 cvxpy 严格 QP，缺失时回退 L-BFGS-B。
        pspline_max_iter : int
            L-BFGS-B 最大迭代次数。
        load_source : {'nominal', 'estimated'}
            'nominal'   : 使用 JSON 固定负载（等长 / 等速组的无标称负载行会被剔除）。
            'estimated' : 使用逐样本估算负载
                          estimated_load = (force_l + force_r) /
                                           ((acc_l + acc_r) / 2 + g)。
        g : float
            重力加速度 (m/s²)，仅在 load_source='estimated' 时使用。

        Returns
        -------
        dict
            {musc: params}，RBF 基线以 '{musc}_rbf' 为键。
        """
        if movement_types is None:
            movement_types = ['upward']

        self._ensure_results()

        data = self.collect_cutted_data(movement_types=movement_types)
        if data is None:
            self._log('无法加载切片数据，热力图生成终止。')
            return {}

        if load_source == 'estimated':
            data = self._prepare_estimated_load_data(data, g=g)
            if data is None:
                return {}
            save_dir = save_dir or os.path.join(
                self.subject.result_folder, 'heatmap_estimated_load')
            params_suffix = '_est'
            load_col = 'load'
        elif load_source == 'nominal':
            data = self._drop_nan_loads(data)
            if data is None:
                return {}
            save_dir = save_dir or os.path.join(
                self.subject.result_folder, 'heatmap')
            params_suffix = ''
            load_col = 'load' if 'load' in data.columns else 'load_value'
        else:
            raise ValueError(
                f'load_source 必须是 "nominal" 或 "estimated"，got {load_source!r}')

        # 打印切片数据的高度范围（方便填写 heatmap_settings.height_range）
        if 'pos_l' in data.columns:
            self._print_height_range(data, estimated=(load_source == 'estimated'))

        if muscles is None:
            muscles = self.subject.musc_label

        os.makedirs(save_dir, exist_ok=True)
        rbf_dir = os.path.join(save_dir, 'rbf')
        os.makedirs(rbf_dir, exist_ok=True)
        params_dir = os.path.join(save_dir, 'params')
        os.makedirs(params_dir, exist_ok=True)

        common_kw = dict(num_centers=num_centers, sigma=sigma,
                         data_len=data_len,
                         height_range=self.subject.height_range)
        psp_kw = dict(**common_kw, use_pspline=True,
                      pspline_n_basis_h=pspline_n_basis_h,
                      pspline_n_basis_l=pspline_n_basis_l,
                      pspline_degree=pspline_degree,
                      pspline_lambda_h=pspline_lambda_h,
                      pspline_lambda_l=pspline_lambda_l,
                      pspline_solver=pspline_solver,
                      pspline_max_iter=pspline_max_iter)
        est = (load_source == 'estimated')

        all_params = {}
        for musc in muscles:
            emg_col = f'emg_{musc}'
            if emg_col not in data.columns:
                self._log(f'跳过肌肉 {musc}：列 {emg_col} 不存在')
                continue

            self._log(f'拟合肌肉 {musc} 的激活热力图...')

            # ---- RBF 基线（用于对比）----
            params_rbf = fit_activation_map(
                data, pos_col='pos_l', load_col=load_col, emg_col=emg_col,
                **common_kw)
            all_params[f'{musc}_rbf'] = params_rbf

            plot_activation_3d(
                data, params_rbf, pos_col='pos_l', load_col=load_col,
                emg_col=emg_col, label=musc, result_folder=rbf_dir)
            draw_heatmap_2d(params_rbf, label=musc, result_folder=rbf_dir)
            if not est:
                draw_load_sensitivity_heatmap_2d(
                    params_rbf, label=musc, result_folder=rbf_dir)
                save_rbf_params(
                    params_rbf['centers'], params_rbf['weights'],
                    params_rbf['scaler'], params_rbf['sigma'],
                    os.path.join(params_dir, f'{musc}_rbf_params.pkl'))

            # ---- P-spline 主曲面（默认）----
            params_psp = fit_activation_map(
                data, pos_col='pos_l', load_col=load_col, emg_col=emg_col,
                **psp_kw)
            all_params[musc] = params_psp

            if not est:
                plot_activation_3d(
                    data, params_psp, pos_col='pos_l', load_col=load_col,
                    emg_col=emg_col, label=musc, result_folder=save_dir)
                draw_heatmap_2d(params_psp, label=musc, result_folder=save_dir)
                draw_load_sensitivity_heatmap_2d(
                    params_psp, label=musc, result_folder=save_dir)
            with open(os.path.join(
                      params_dir,
                      f'{musc}{params_suffix}_pspline_params.pkl'),
                      'wb') as f:
                pickle.dump(params_psp, f)

            # ---- 1×2 对比图：RBF (左) vs P-spline (右) ----
            plot_compare_activation_3d(
                data, params_rbf, params_psp, pos_col='pos_l',
                load_col=load_col, emg_col=emg_col, label=musc,
                result_folder=save_dir)
            plot_compare_heatmap_2d(
                params_rbf, params_psp, label=musc, result_folder=save_dir)
            if not est:
                plot_compare_load_sensitivity_2d(
                    params_rbf, params_psp, label=musc,
                    result_folder=save_dir)

            # ---- 可选 3D（速度维）----
            if fit_3d and not est and 'vel_l' in data.columns:
                params_3d = fit_activation_map_3d(
                    data, 'pos_l', load_col, emg_col, 'vel_l',
                    num_centers=num_centers, sigma=sigma, data_len=data_len)
                all_params[f'{musc}_3d'] = params_3d

        self._print_rmse_table(all_params, data, load_col,
                               movement_types=movement_types)

        self._log(f'热力图已保存至 {save_dir}')
        return all_params

    # ==================== 数据准备 ====================

    def _prepare_estimated_load_data(self, data, g=9.81):
        """用逐样本估算负载替换 load 列；缺列时返回 None。"""
        data = data.copy()
        data['load'] = estimate_load_from_df(data, g=g)
        data = data.dropna(subset=['load'])
        if len(data) == 0:
            self._log('估算负载后数据为空（缺少 force/acc 列？），终止。')
            return None
        if 'pos_l' in data.columns:
            self.subject.height_range = [
                float(data['pos_l'].min()), float(data['pos_l'].max())]
        return data

    def _drop_nan_loads(self, data):
        """剔除无标称负载的组（等长 / 等速组），空数据返回 None。"""
        load_col_check = 'load' if 'load' in data.columns else 'load_value'
        if load_col_check in data.columns:
            bad = ~np.isfinite(pd.to_numeric(data[load_col_check],
                                             errors='coerce'))
            if bool(bad.any()):
                dropped = sorted(set(
                    data.loc[bad, 'load_weight'].astype(str)
                )) if 'load_weight' in data.columns else ['?']
                beauty_print(
                    '热力图已剔除无标称负载的组: {}（{} 行）。\n'
                    '等长 / 等速组的负载靠受力反推，不能当成定负载直接拟合；'
                    '需要纳入时请改用 load_source="estimated"。'.format(
                        dropped, int(bad.sum())),
                    type="warning")
                data = data[~bad].reset_index(drop=True)
            if len(data) == 0:
                self._log('剔除无标称负载的组后数据为空，热力图生成终止。')
                return None
        return data

    def _print_height_range(self, data, estimated=False):
        """打印切片数据的高度范围，方便填写 heatmap_settings.height_range。"""
        h_min = round(float(data['pos_l'].min()), 4)
        h_max = round(float(data['pos_l'].max()), 4)
        if estimated:
            print(f'height_range (estimated load): [{h_min}, {h_max}]')
        else:
            print(f'height_range: [{h_min}, {h_max}]')

    def _print_rmse_table(self, all_params, data, load_col,
                          movement_types=None):
        """输出 RBF / P-spline 的 RMSE% 表格。"""
        rmse_rows = {}
        for musc, p in all_params.items():
            if '_3d' in musc:
                continue
            if musc.endswith('_rbf'):
                base_musc = musc[:-len('_rbf')]
                tag = 'RBF'
            else:
                base_musc = musc
                tag = 'P-spline'

            emg_col = f'emg_{base_musc}'
            if emg_col not in data.columns:
                continue

            if p.get('model') == 'pspline':
                pred = predict_at(
                    p, data['pos_l'].values, data[load_col].values)
                actual = data[emg_col].values
                rmse = float(np.sqrt(np.nanmean((pred - actual) ** 2)))
                mean_pred = float(np.nanmean(np.abs(pred)))
                rmse_pct = (rmse / mean_pred * 100
                            if mean_pred > 1e-8 else float('inf'))
            else:
                rmse_pct = compute_rmse_percentage(
                    data, 'pos_l', load_col, emg_col,
                    p['centers'], p['weights'],
                    p['scaler'], p['sigma'])

            rmse_rows.setdefault(base_musc, {})[tag] = rmse_pct

        if not rmse_rows:
            return

        print('\n' + '=' * 64)
        print(f'Heatmap RMSE% 表格（movement_types={movement_types}）')
        print('=' * 64)
        print(f'{"muscle":<16s}{"RBF RMSE%":>16s}{"P-spline RMSE%":>20s}')
        print('-' * 64)
        for base_musc in sorted(rmse_rows.keys()):
            rbf_v = rmse_rows[base_musc].get('RBF')
            psp_v = rmse_rows[base_musc].get('P-spline')
            rbf_s = 'N/A' if rbf_v is None else f'{rbf_v:.2f}'
            psp_s = 'N/A' if psp_v is None else f'{psp_v:.2f}'
            print(f'{base_musc:<16s}{rbf_s:>16s}{psp_s:>20s}')
        print('=' * 64)
        print('单位: %')

    # ==================== 日志 ====================

    def _log(self, msg):
        if getattr(self.pipeline, 'debug', False):
            print(msg)
