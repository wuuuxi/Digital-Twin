"""
Heatmap 拟合参数 I/O 工具

提供基于 Subject 配置的 RBF / P-spline 拟合参数加载函数。
拟合参数由 pipeline.generate_heatmaps() 或 example_heatmap.py 生成，保存在：
  - subject.muscle_folder         (load_previous_data=True 时)
  - result_folder/heatmap/params/ (新拟合默认)

用法示例:
    from digitaltwin.analysis.heatmap_io import (
        load_rbf_params, load_pspline_params, load_heatmap_params_by_mode,
    )
    rbf_p = load_rbf_params(subject, 'GL')
    psp_p = load_pspline_params(subject, 'GL')
    overlays = load_heatmap_params_by_mode(subject, 'GL', 'both')
"""
import os
import pickle


def heatmap_param_dir(subject):
    """返回 heatmap 拟合参数所在目录。"""
    return (subject.muscle_folder if subject.load_previous_data
            else os.path.join(subject.result_folder, 'heatmap/params'))


def load_rbf_params(subject, muscle, verbose=True):
    """加载原始 RBF 参数 ({muscle}_rbf_params.pkl)。返回 dict 或 None。"""
    path = os.path.join(heatmap_param_dir(subject),
                        f'{muscle}_rbf_params.pkl')
    if not os.path.exists(path):
        if verbose:
            print(f'  未找到 {muscle} 的 RBF 参数: {path}')
        return None
    with open(path, 'rb') as f:
        p = pickle.load(f)
    p = dict(p)
    p['use_pspline'] = False
    p['model'] = 'rbf'
    return p


def load_pspline_params(subject, muscle, verbose=True):
    """加载 P-spline 拟合后的完整 params ({muscle}_pspline_params.pkl)。返回 dict 或 None。"""
    path = os.path.join(heatmap_param_dir(subject),
                        f'{muscle}_pspline_params.pkl')
    if not os.path.exists(path):
        if verbose:
            print(f'  未找到 {muscle} 的 P-spline 参数: {path}')
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_heatmap_params_by_mode(subject, muscle, mode):
    """根据 mode 加载 heatmap 拟合参数，返回 overlays 列表
    (供 plot_vload_overlay / compute_rmse_at_actual_points 使用)。

    Parameters
    ----------
    subject : Subject
    muscle  : str
    mode    : {'both', 'rbf', 'pspline', 'none'}

    Returns
    -------
    list of (key, label, params, color, linestyle)
        key      -- 'rbf' 或 'pspline'，用于 RMSE 表中的识别。
        label    -- legend 名称。
        params   -- predict_at 可用的参数字典。
        color, linestyle -- 供刷预测曲线。
    """
    if mode == 'none' or muscle is None:
        return []
    overlays = []
    if mode in ('rbf', 'both'):
        p = load_rbf_params(subject, muscle)
        if p is not None:
            overlays.append(('rbf', 'Heatmap (RBF)', p, 'C3', '--'))
    if mode in ('pspline', 'both'):
        p = load_pspline_params(subject, muscle)
        if p is not None:
            overlays.append(
                ('pspline', 'Heatmap (P-spline)', p, 'C2', '-.'))
    return overlays