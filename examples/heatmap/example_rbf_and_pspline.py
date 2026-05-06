"""
RBF vs P-spline 对比示例

在 example_vload_result.py 的基础上同时对比 RBF 与 P-spline 两种预测。

每个变负载实验（= 一块目标肌肉）一张 1×2 图：
  左：Activation vs Height
        * 灰色散点 = 实测 EMG
        * Expected (vload_file)
        * Goal 水平线
        * RBF 预测曲线
        * P-spline 预测曲线
  右：该肌肉在 5 组固定负载 + 3 组变负载下，实测 EMG 与
       RBF / P-spline 两种预测的 RMSE 柱状图。

说明：
  - 运行本脚本前需先跑过 example_heatmap.py 生成
    {muscle}_rbf_params.pkl 与 {muscle}_pspline_params.pkl。
用法：
    python example_rbf_and_pspline.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.analysis.rbf_fitting import predict_at


# ---- 选项 ----
MOVEMENT_TYPES = ['upward']

# 柱状图配色：固定负载用更浅的色，突出变负载组
RBF_FIXED_COLOR = '#f4d4d4'      # 极浅粉红
RBF_VLOAD_COLOR = '#e89090'      # 较深粉红
PSP_FIXED_COLOR = '#d6ead7'     # 极浅薄荷绿
PSP_VLOAD_COLOR = '#85bf90'     # 较深薄荷绿
BAR_EDGE_COLOR = '#888888'


# =========================================================
#  通用工具
# =========================================================

def filter_movement_types(df, movement_types):
    if df is None or movement_types is None:
        return df
    if 'movement_type' not in df.columns:
        return df
    return df[df['movement_type'].isin(movement_types)]


def load_planned_vload(subject, vload_file):
    if not vload_file:
        return None
    path = os.path.join(subject.vload_load_folder, vload_file)
    if not os.path.exists(path):
        print(f'  规划文件不存在: {path}')
        return None
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]
    return df


def _heatmap_param_dir(subject):
    return (subject.muscle_folder if subject.load_previous_data
            else os.path.join(subject.result_folder, 'heatmap/params'))


def load_rbf_params(subject, muscle):
    """加载原始 RBF 参数 ({muscle}_rbf_params.pkl)。"""
    path = os.path.join(_heatmap_param_dir(subject),
                        f'{muscle}_rbf_params.pkl')
    if not os.path.exists(path):
        print(f'  未找到 {muscle} 的 RBF 参数: {path}')
        return None
    with open(path, 'rb') as f:
        p = pickle.load(f)
    p = dict(p)
    p['use_pspline'] = False
    p['model'] = 'rbf'
    return p


def load_pspline_params(subject, muscle):
    """加载 P-spline 拟合后的完整 params ({muscle}_pspline_params.pkl)。"""
    path = os.path.join(_heatmap_param_dir(subject),
                        f'{muscle}_pspline_params.pkl')
    if not os.path.exists(path):
        print(f'  未找到 {muscle} 的 P-spline 参数: {path}')
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _interp_sorted(x_query, x_known, y_known):
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    order = np.argsort(x_known)
    return np.interp(x_query, x_known[order], y_known[order])


def _rmse(pred, actual):
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() == 0:
        return float('nan'), 0
    diff = pred[mask] - actual[mask]
    return float(np.sqrt(np.mean(diff ** 2))), int(mask.sum())


def _format_rmse_for_legend(rmse_dict, key):
    if key not in rmse_dict:
        return ''
    rmse, _n = rmse_dict[key]
    if not np.isfinite(rmse):
        return ''
    return f' [RMSE={rmse:.4f}]'


# =========================================================
#  变负载实测点上的 RMSE（用于左图 legend 后缀）
# =========================================================

def compute_vload_rmse_at_actual(cutted, planned_df, rbf_p, psp_p,
                                 target_muscle):
    emg_col = f'emg_{target_muscle}'
    rmse = {}
    if (cutted is None or emg_col not in cutted.columns
            or 'pos_l' not in cutted.columns or planned_df is None):
        return rmse

    h_actual = cutted['pos_l'].values
    emg_actual = cutted[emg_col].values

    if {'Height', 'Activation'}.issubset(planned_df.columns):
        a_exp = _interp_sorted(h_actual, planned_df['Height'].values,
                               planned_df['Activation'].values)
        rmse['expected'] = _rmse(a_exp, emg_actual)

    if {'Height', 'Load'}.issubset(planned_df.columns):
        l_at_actual = _interp_sorted(
            h_actual, planned_df['Height'].values,
            planned_df['Load'].values)
        if rbf_p is not None:
            try:
                pred = predict_at(rbf_p, h_actual, l_at_actual)
                rmse['rbf'] = _rmse(pred, emg_actual)
            except Exception as e:
                print(f'  RBF 预测失败: {e}')
        if psp_p is not None:
            try:
                pred = predict_at(psp_p, h_actual, l_at_actual)
                rmse['pspline'] = _rmse(pred, emg_actual)
            except Exception as e:
                print(f'  P-spline 预测失败: {e}')
    return rmse


# =========================================================
#  柱状图所需的各组 RMSE（以某块肌肉为主）
# =========================================================

def compute_groups_rmse_for_muscle(pipeline, subject, vload_results,
                                   muscle, movement_types,
                                   rbf_p, psp_p):
    """
    对指定肌肉，计算 5 组固定负载 + 3 组变负载下，
    RBF / P-spline 两种预测与实测 EMG 的 RMSE。

    Returns
    -------
    list of dict
        每个 dict 含字段: 'label', 'kind' ('fixed'|'vload'),
        'rbf' (rmse, n), 'pspline' (rmse, n)。
    """
    emg_col = f'emg_{muscle}'
    groups = []

    # ---- 固定负载（从 pipeline.results 拿 cutted，按 height_range 过滤）----
    h_min = h_max = None
    if subject.height_range is not None:
        h_min, h_max = subject.height_range

    fixed_keys = sorted(subject.modeling_data.keys(), key=lambda k: float(k))
    for lw in fixed_keys:
        rec = {'label': f'{lw}kg', 'kind': 'fixed'}
        result = pipeline.results.get(lw)
        cutted = result.get('cutted_data') if result else None
        cutted = filter_movement_types(cutted, movement_types)
        if (cutted is None or emg_col not in cutted.columns
                or 'pos_l' not in cutted.columns):
            rec['rbf'] = (float('nan'), 0)
            rec['pspline'] = (float('nan'), 0)
            groups.append(rec)
            continue
        if h_min is not None:
            cutted = cutted[(cutted['pos_l'] >= h_min)
                            & (cutted['pos_l'] <= h_max)]
        load_col = 'load' if 'load' in cutted.columns else 'load_value'
        h = cutted['pos_l'].values
        l = cutted[load_col].values
        actual = cutted[emg_col].values
        rec['rbf'] = (_rmse(predict_at(rbf_p, h, l), actual)
                      if rbf_p is not None else (float('nan'), 0))
        rec['pspline'] = (_rmse(predict_at(psp_p, h, l), actual)
                            if psp_p is not None else (float('nan'), 0))
        groups.append(rec)

    # ---- 变负载（3 组，负载从规划 vload_file 插值）----
    for vlabel, vparams in subject.vload_data.items():
        rec = {'label': f'VL:{vlabel}', 'kind': 'vload'}
        vresult = vload_results.get(vlabel) if vload_results else None
        cutted = vresult.get('cutted_data') if vresult else None
        cutted = filter_movement_types(cutted, movement_types)
        if (cutted is None or emg_col not in cutted.columns
                or 'pos_l' not in cutted.columns):
            rec['rbf'] = (float('nan'), 0)
            rec['pspline'] = (float('nan'), 0)
            groups.append(rec)
            continue
        planned_df = load_planned_vload(subject, vparams.get('vload_file'))
        if planned_df is None or not {'Height', 'Load'}.issubset(
                planned_df.columns):
            rec['rbf'] = (float('nan'), 0)
            rec['pspline'] = (float('nan'), 0)
            groups.append(rec)
            continue
        h = cutted['pos_l'].values
        l = _interp_sorted(h, planned_df['Height'].values,
                           planned_df['Load'].values)
        actual = cutted[emg_col].values
        rec['rbf'] = (_rmse(predict_at(rbf_p, h, l), actual)
                      if rbf_p is not None else (float('nan'), 0))
        rec['pspline'] = (_rmse(predict_at(psp_p, h, l), actual)
                            if psp_p is not None else (float('nan'), 0))
        groups.append(rec)

    return groups


# =========================================================
#  每个股肉/变负载实验一张 1×2 图
# =========================================================

def plot_per_muscle_figure(target_label, vload_result, planned_df_target,
                           rbf_p, psp_p,
                           target_muscle, target_activation,
                           pipeline, subject, vload_results, movement_types):
    """
    以某块肌肉 (= target_muscle) 为主制作一张图：
      - 左侧：N 个子图 (N = 变负载组数) 从上到下排列，每个子图
            对应一个变负载实验期间，画该肌肉的实测 EMG 与
            RBF / P-spline 两种预测的对比 (只在 target 变负载那个
            子图上额外画 Expected 与 Goal，因为 vload_file 中的
            Activation 是为其 target_muscle 计划的)。
      - 右侧：该肌肉在所有实验组 (固定 + 变负载) 上的 RMSE 柱状图。
    """
    emg_col = f'emg_{target_muscle}'

    # target 变负载上的 RMSE，用于该子图的 legend
    target_cutted = filter_movement_types(
        vload_result.get('cutted_data'), movement_types)
    rmse_at_actual_target = compute_vload_rmse_at_actual(
        target_cutted, planned_df_target, rbf_p, psp_p, target_muscle)

    vload_items = list(subject.vload_data.items())
    n_vload = len(vload_items)

    fig = plt.figure(figsize=(14, max(4.5, 2.0 * n_vload)))
    gs = GridSpec(n_vload, 2, width_ratios=[1.0, 1.3],
                  wspace=0.25, hspace=0.32, figure=fig)

    mt_label = '+'.join(movement_types) if movement_types else 'all'
    fig.suptitle(
        f'{target_label}  (target muscle: {target_muscle}, '
        f'goal: {target_activation}, movement: {mt_label})',
        fontsize=12, fontweight='bold')

    # ---------- 左：N 个子图 ----------
    left_axes = []
    for i, (vlabel, vparams) in enumerate(vload_items):
        sharex = left_axes[0] if left_axes else None
        ax = fig.add_subplot(gs[i, 0], sharex=sharex)
        left_axes.append(ax)

        is_target = (vlabel == target_label)
        vresult = vload_results.get(vlabel)
        cutted_v = filter_movement_types(
            vresult.get('cutted_data') if vresult else None, movement_types)
        if vlabel == target_label:
            planned_df_v = planned_df_target
        else:
            planned_df_v = load_planned_vload(
                subject, vparams.get('vload_file'))

        # 实测 EMG（主肌肉在该 vload 期间的记录）
        if (cutted_v is not None
                and emg_col in cutted_v.columns
                and 'pos_l' in cutted_v.columns):
            ax.scatter(cutted_v['pos_l'], cutted_v[emg_col],
                       s=8, alpha=0.35, color='gray',
                       label='Actual EMG')

        # Expected / Goal：只在该变负载的 target_muscle 与本图肌肉一致时画
        if (is_target and planned_df_v is not None
                and {'Height', 'Activation'}.issubset(planned_df_v.columns)):
            h = planned_df_v['Height'].values
            a = planned_df_v['Activation'].values
            order = np.argsort(h)
            ax.plot(h[order], a[order], color='C0', linewidth=2,
                    label=f'Expected (vload_file)'
                          f'{_format_rmse_for_legend(rmse_at_actual_target, "expected")}')
        if is_target and target_activation is not None:
            ax.axhline(target_activation, color='C1', linestyle=':',
                       linewidth=1.5,
                       label=f'Goal = {target_activation}')

        # RBF / P-spline：在该 vload 的规划 (Height, Load) 上预测本图肌肉的激活
        if (planned_df_v is not None
                and {'Height', 'Load'}.issubset(planned_df_v.columns)):
            h = planned_df_v['Height'].values
            l = planned_df_v['Load'].values
            order = np.argsort(h)

            if is_target:
                rmse_local = rmse_at_actual_target
            else:
                rmse_local = compute_vload_rmse_at_actual(
                    cutted_v, planned_df_v, rbf_p, psp_p, target_muscle)

            if rbf_p is not None:
                try:
                    pred = predict_at(rbf_p, h, l)
                    ax.plot(h[order], pred[order], color='C3',
                            linewidth=1.5, linestyle='--',
                            label=f'RBF'
                                  f'{_format_rmse_for_legend(rmse_local, "rbf")}')
                except Exception as e:
                    print(f'  RBF 画图失败 ({vlabel}): {e}')
            if psp_p is not None:
                try:
                    pred = predict_at(psp_p, h, l)
                    ax.plot(h[order], pred[order], color='C2',
                            linewidth=1.5, linestyle='-.',
                            label=f'P-spline'
                                  f'{_format_rmse_for_legend(rmse_local, "pspline")}')
                except Exception as e:
                    print(f'  P-spline 画图失败 ({vlabel}): {e}')

        ax.set_ylabel(f'{target_muscle}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        marker = '  ★' if is_target else ''
        ax.set_title(f'During {vlabel}{marker}', fontsize=9)
        if i < n_vload - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Height (m)')

    # ---------- 右：该肌肉各组 RMSE 柱状图 ----------
    ax = fig.add_subplot(gs[:, 1])
    groups = compute_groups_rmse_for_muscle(
        pipeline, subject, vload_results, target_muscle,
        movement_types, rbf_p, psp_p)

    labels = [g['label'] for g in groups]
    rbf_vals = [g['rbf'][0] for g in groups]
    psp_vals = [g['pspline'][0] for g in groups]
    ns = [max(g['rbf'][1], g['pspline'][1]) for g in groups]

    x = np.arange(len(groups))
    width = 0.38
    rbf_colors = [RBF_VLOAD_COLOR if g['kind'] == 'vload'
                  else RBF_FIXED_COLOR for g in groups]
    psp_colors = [PSP_VLOAD_COLOR if g['kind'] == 'vload'
                   else PSP_FIXED_COLOR for g in groups]
    b1 = ax.bar(x - width / 2, rbf_vals, width,
                color=rbf_colors,
                edgecolor=BAR_EDGE_COLOR, linewidth=0.5)
    b2 = ax.bar(x + width / 2, psp_vals, width,
                color=psp_colors,
                edgecolor=BAR_EDGE_COLOR, linewidth=0.5)
    for bars, vals in [(b1, rbf_vals), (b2, psp_vals)]:
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_xlabel('Experiment group  (5 fixed loads + 3 variable loads)')
    ax.set_ylabel(f'RMSE (predicted vs actual EMG of {target_muscle})')
    ax.set_title(f'RMSE per group for {target_muscle}')
    legend_handles = [
        Patch(facecolor=RBF_VLOAD_COLOR, edgecolor=BAR_EDGE_COLOR,
              label='RBF (variable load)'),
        Patch(facecolor=RBF_FIXED_COLOR, edgecolor=BAR_EDGE_COLOR,
              label='RBF (fixed load)'),
        Patch(facecolor=PSP_VLOAD_COLOR, edgecolor=BAR_EDGE_COLOR,
              label='P-spline (variable load)'),
        Patch(facecolor=PSP_FIXED_COLOR, edgecolor=BAR_EDGE_COLOR,
              label='P-spline (fixed load)'),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    return fig, rmse_at_actual_target, groups


# =========================================================
#  打印表格
# =========================================================

def _print_vload_at_actual_summary(summary):
    if not summary:
        return
    keys = ['expected', 'rbf', 'pspline']
    headers = {'expected': 'Expected', 'rbf': 'RBF',
               'pspline': 'P-spline'}
    col_w = max(20, max(len(label) for label in summary) + 2)
    print('\n========== Variable-load (at actual points) RMSE summary ==========')
    head = ('Label'.ljust(col_w)
            + ''.join(headers[k].rjust(15) for k in keys) + '   n')
    print(head)
    print('-' * len(head))
    for label, d in summary.items():
        row = label.ljust(col_w)
        n_print = ''
        for k in keys:
            if k in d and np.isfinite(d[k][0]):
                row += f'{d[k][0]:>15.4f}'
                n_print = d[k][1]
            else:
                row += '              -'
        row += f'   {n_print}'
        print(row)
    print('=' * len(head))


def _print_groups_summary(muscle, groups):
    print(f'\n========== Per-group RMSE for {muscle} ==========')
    head = ('Group'.ljust(20) + 'RBF'.rjust(12) + 'P-spline'.rjust(12)
            + '   n')
    print(head)
    print('-' * len(head))
    for g in groups:
        row = g['label'].ljust(20)
        n_print = max(g['rbf'][1], g['pspline'][1])
        for k in ('rbf', 'pspline'):
            v = g[k][0]
            if np.isfinite(v):
                row += f'{v:>12.4f}'
            else:
                row += '           -'
        row += f'   {n_print}'
        print(row)
    print('=' * len(head))


# =========================================================
#  主函数
# =========================================================

def main():
    subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # 加载固定负载 + 变负载数据
    pipeline.run(include_xsens=False)
    vload_results = pipeline.run_vload()

    if not vload_results:
        print('未加载到变负载实际数据，终止。')
        return

    rmse_at_actual_summary = {}

    for label, params in subject.vload_data.items():
        if label not in vload_results:
            print(f'\n>>> 跳过 {label}：未成功处理')
            continue

        vload_file = params.get('vload_file')
        target_muscle = params.get('target_muscle')
        target_activation = params.get('target_activation')
        if target_muscle is None:
            print(f'\n>>> 跳过 {label}：未指定 target_muscle')
            continue

        print(f'\n>>> 处理 {label} '
              f'(muscle={target_muscle}, goal={target_activation})')

        planned_df = load_planned_vload(subject, vload_file)
        if planned_df is None:
            print('  跳过：未找到 vload_file 规划数据')
            continue

        rbf_p = load_rbf_params(subject, target_muscle)
        psp_p = load_pspline_params(subject, target_muscle)

        _, rmse_at_actual, groups = plot_per_muscle_figure(
            label, vload_results[label], planned_df, rbf_p, psp_p,
            target_muscle, target_activation,
            pipeline, subject, vload_results, MOVEMENT_TYPES)

        rmse_at_actual_summary[label] = rmse_at_actual
        for k in ('expected', 'rbf', 'pspline'):
            if k in rmse_at_actual:
                rmse, n = rmse_at_actual[k]
                if np.isfinite(rmse):
                    print(f'  RMSE [{k:>9}] (vload at actual)'
                          f' = {rmse:.4f}  (n={n})')
        _print_groups_summary(target_muscle, groups)

    _print_vload_at_actual_summary(rmse_at_actual_summary)

    plt.show()


if __name__ == '__main__':
    main()