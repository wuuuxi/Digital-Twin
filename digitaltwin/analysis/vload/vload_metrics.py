"""
变负载预测 RMSE 计算工具

计算 RBF / P-spline 等 heatmap 拟合曲面在变负载工况下的预测与实测 EMG
的 RMSE，供 vload 对比示例中的表格 / legend / 柱状图使用。

主要函数：
  - compute_rmse_at_actual_points: 单组变负载上多个预测器的 RMSE
  - compute_groups_rmse_for_muscle: 某块肌肉在所有实验组 (固定 + 变负载) 上的 RMSE
  - format_rmse_for_legend: 输出 ' [RMSE=0.0432]' 样的字符串
预测器以 overlays 列表传入：list of (key, _label, params, _color, _ls)。
key 除 'rbf'/'pspline' 等之外，'expected' 是保留关键字，表示使用规划
Activation(Height) 插值作为预测（不从 overlays 中读取 params）。
常从 heatmap_io.load_heatmap_params_by_mode() 获得 overlays。
"""
import numpy as np

from digitaltwin.analysis.alignment import filter_movement_types
from digitaltwin.analysis.heatmap.rbf_fitting import predict_at
from digitaltwin.analysis.vload.vload_planning import load_planned_vload
from digitaltwin.utils.array_tools import interp_sorted, rmse_with_count


def compute_rmse_at_actual_points(cutted, planned_df, heatmap_overlays,
                                  target_muscle):
    """
    在实测数据点上计算各预测器与实测 EMG 的 RMSE。

    实测数据不记录瞬时负载：在实测高度上插值出规划负载 Load(Height)
    作为该点负载。Expected 预测则使用 Activation(Height) 插值。

    Parameters
    ----------
    cutted : pd.DataFrame
        实测切片数据 (含 pos_l, emg_<muscle>)。
    planned_df : pd.DataFrame
        规划数据 (含 Height, Load, Activation)。
    heatmap_overlays : list of (key, _label, params, _color, _ls)
        load_heatmap_params_by_mode() 返回的 overlays；extra fields ignored。
    target_muscle : str

    Returns
    -------
    dict
        {key: (rmse, n_points)}，key 取值 'expected', 'rbf', 'pspline' 等。
    """
    emg_col = f'emg_{target_muscle}'
    rmse_dict = {}

    if (cutted is None
            or emg_col not in cutted.columns
            or 'pos_l' not in cutted.columns
            or planned_df is None):
        return rmse_dict

    h_actual = cutted['pos_l'].values
    emg_actual = cutted[emg_col].values

    # Expected: 插值规划 Activation(Height)
    if {'Height', 'Activation'}.issubset(planned_df.columns):
        a_expected = interp_sorted(
            h_actual, planned_df['Height'].values,
            planned_df['Activation'].values)
        rmse_dict['expected'] = rmse_with_count(a_expected, emg_actual)

    # heatmap overlays。在实测高度上插值出规划负载
    if {'Height', 'Load'}.issubset(planned_df.columns) and heatmap_overlays:
        l_at_actual = interp_sorted(
            h_actual, planned_df['Height'].values,
            planned_df['Load'].values)
        for key, _label, params, _color, _ls in heatmap_overlays:
            if params is None:
                continue
            try:
                pred = predict_at(params, h_actual, l_at_actual)
                rmse_dict[key] = rmse_with_count(pred, emg_actual)
            except Exception as e:
                print(f'  {key} 预测失败: {e}')

    return rmse_dict


def compute_groups_rmse_for_muscle(pipeline, subject, vload_results,
                                   muscle, movement_types,
                                   heatmap_overlays):
    """
    对指定肌肉，在所有固定负载组 + 变负载组上计算各预测器的 RMSE。

    Parameters
    ----------
    pipeline : MultiLoadPipeline
        需已调用 run() 加载过固定负载数据。
    subject : Subject
    vload_results : dict
        pipeline.run_vload() 返回值。
    muscle : str
    movement_types : list[str] or None
    heatmap_overlays : list of (key, _label, params, _color, _ls)

    Returns
    -------
    list of dict
        每个 dict: {'label', 'kind' ('fixed'|'vload'), <key>: (rmse, n)}。
        <key> 对应 heatmap_overlays 中的 key。
    """
    emg_col = f'emg_{muscle}'
    keys_params = [(ov[0], ov[2]) for ov in heatmap_overlays
                   if ov[2] is not None]
    keys = [kp[0] for kp in keys_params]

    def _empty():
        return {k: (float('nan'), 0) for k in keys}

    groups = []

    # ---- 固定负载 ----
    h_min = h_max = None
    if subject.height_range is not None:
        h_min, h_max = subject.height_range
    fixed_keys = sorted(subject.modeling_data.keys(), key=lambda k: float(k))
    for lw in fixed_keys:
        rec = {'label': f'{lw}kg', 'kind': 'fixed', **_empty()}
        result = pipeline.results.get(lw)
        cutted = result.get('cutted_data') if result else None
        cutted = filter_movement_types(cutted, movement_types)
        if (cutted is None or emg_col not in cutted.columns
                or 'pos_l' not in cutted.columns):
            groups.append(rec)
            continue
        if h_min is not None:
            cutted = cutted[(cutted['pos_l'] >= h_min)
                            & (cutted['pos_l'] <= h_max)]
        load_col = 'load' if 'load' in cutted.columns else 'load_value'
        h = cutted['pos_l'].values
        l = cutted[load_col].values
        actual = cutted[emg_col].values
        for k, params in keys_params:
            try:
                rec[k] = rmse_with_count(predict_at(params, h, l), actual)
            except Exception as e:
                print(f'  {k} 预测失败 ({lw}kg): {e}')
        groups.append(rec)

    # ---- 变负载 ----
    for vlabel, vparams in subject.vload_data.items():
        rec = {'label': f'VL:{vlabel}', 'kind': 'vload', **_empty()}
        vresult = vload_results.get(vlabel) if vload_results else None
        cutted = vresult.get('cutted_data') if vresult else None
        cutted = filter_movement_types(cutted, movement_types)
        if (cutted is None or emg_col not in cutted.columns
                or 'pos_l' not in cutted.columns):
            groups.append(rec)
            continue
        planned_df = load_planned_vload(
            subject, vparams.get('vload_file'), verbose=False)
        if planned_df is None or not {'Height', 'Load'}.issubset(
                planned_df.columns):
            groups.append(rec)
            continue
        h = cutted['pos_l'].values
        l = interp_sorted(h, planned_df['Height'].values,
                          planned_df['Load'].values)
        actual = cutted[emg_col].values
        for k, params in keys_params:
            try:
                rec[k] = rmse_with_count(predict_at(params, h, l), actual)
            except Exception as e:
                print(f'  {k} 预测失败 (VL:{vlabel}): {e}')
        groups.append(rec)

    return groups


def format_rmse_for_legend(rmse_dict, key, with_n=False):
    """生成 legend 后缀字符串，例如 ' [RMSE=0.0432]' 或 ' [RMSE=0.0432, n=128]'。

    仅在 key 存在且 rmse 有限时返回非空串。
    """
    if key not in rmse_dict:
        return ''
    rmse, n = rmse_dict[key]
    if not np.isfinite(rmse):
        return ''
    if with_n:
        return f' [RMSE={rmse:.4f}, n={n}]'
    return f' [RMSE={rmse:.4f}]'