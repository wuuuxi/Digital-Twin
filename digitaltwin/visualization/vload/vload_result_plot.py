"""
变负载结果对比可视化。

provides:
  - plot_vload_overlay
      单组变负载 1×2 图 (实测 + 各预测曲线 + 规划负载)。
  - plot_vload_per_muscle_compare
      以某块肌肉为主的对比图：左侧 N 个变负载子图 + 右侧
      该肌肉在所有实验组 (固定 + 变负载) 上的 RMSE 柱状图。
  - print_vload_rmse_summary, print_groups_rmse
      控制台表格。
预测器统一以 overlays 列表传入：list of (key, label, params, color, linestyle)。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from digitaltwin.analysis.alignment import filter_movement_types
from digitaltwin.analysis.heatmap.rbf_fitting import predict_at
from digitaltwin.analysis.vload.vload_planning import load_planned_vload
from digitaltwin.analysis.vload.vload_metrics import (
    compute_rmse_at_actual_points,
    compute_groups_rmse_for_muscle,
    format_rmse_for_legend,
)


# 默认柱状图配色 (固定负载用更浅的色，突出变负载组)
DEFAULT_BAR_COLORS = {
    'rbf':     {'fixed': '#f4d4d4', 'vload': '#e89090', 'name': 'RBF'},
    'pspline': {'fixed': '#d6ead7', 'vload': '#85bf90', 'name': 'P-spline'},
}
BAR_EDGE_COLOR = '#888888'


def plot_vload_overlay(label, vload_result, planned_df, heatmap_overlays,
                       target_muscle, target_activation,
                       movement_types=None):
    """
    单组变负载实验 1×2 图。

    左：Activation vs Height (实测散点 + Expected/Goal/heatmap 预测)。
    右：Planned Load vs Height。

    Returns
    -------
    (fig, rmse_dict)
        rmse_dict = compute_rmse_at_actual_points() 返回值。
    """
    cutted = filter_movement_types(
        vload_result.get('cutted_data'), movement_types)
    emg_col = f'emg_{target_muscle}'

    rmse_dict = compute_rmse_at_actual_points(
        cutted, planned_df, heatmap_overlays, target_muscle)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    mt_label = '+'.join(movement_types) if movement_types else 'all'
    fig.suptitle(
        f'{label}  (target muscle: {target_muscle}, '
        f'goal: {target_activation}, movement: {mt_label})',
        fontsize=12, fontweight='bold')

    # -------- 左 --------
    ax = axes[0]
    if (cutted is not None
            and emg_col in cutted.columns
            and 'pos_l' in cutted.columns):
        ax.scatter(cutted['pos_l'], cutted[emg_col], s=8, alpha=0.35,
                   color='gray', label='Actual EMG')

    if (planned_df is not None
            and {'Height', 'Activation'}.issubset(planned_df.columns)):
        h = planned_df['Height'].values
        a = planned_df['Activation'].values
        order = np.argsort(h)
        ax.plot(h[order], a[order], color='C0', linewidth=2,
                label='Expected (vload_file)' + format_rmse_for_legend(
                    rmse_dict, 'expected', with_n=True))

    if target_activation is not None:
        ax.axhline(target_activation, color='C1', linestyle=':',
                   linewidth=1.5, label=f'Goal = {target_activation}')

    if (heatmap_overlays
            and planned_df is not None
            and {'Height', 'Load'}.issubset(planned_df.columns)):
        h = planned_df['Height'].values
        l = planned_df['Load'].values
        order = np.argsort(h)
        for key, ov_label, ov_params, ov_color, ov_ls in heatmap_overlays:
            if ov_params is None:
                continue
            try:
                pred = predict_at(ov_params, h, l)
                ax.plot(h[order], pred[order], color=ov_color, linewidth=2,
                        linestyle=ov_ls,
                        label=ov_label + format_rmse_for_legend(
                            rmse_dict, key, with_n=True))
            except Exception as e:
                print(f'  {ov_label} 画图预测失败: {e}')

    ax.set_xlabel('Height (m)')
    ax.set_ylabel(f'{target_muscle} activation')
    ax.set_title('Activation vs Height')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # -------- 右 --------
    ax = axes[1]
    if (planned_df is not None
            and {'Height', 'Load'}.issubset(planned_df.columns)):
        h = planned_df['Height'].values
        l = planned_df['Load'].values
        order = np.argsort(h)
        ax.plot(h[order], l[order], color='C0', linewidth=2,
                label='Planned load')
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Load (kg)')
    ax.set_title('Planned Load vs Height')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, rmse_dict


def plot_vload_per_muscle_compare(target_label, vload_result,
                                  planned_df_target, heatmap_overlays,
                                  target_muscle, target_activation,
                                  pipeline, subject, vload_results,
                                  movement_types,
                                  bar_colors=None):
    """
    以某块肌肉为主的对比图：
      - 左侧：N 个子图 (N = subject.vload_data 中的变负载组数)。
        每个子图画该变负载期间 target_muscle 的实测 EMG 与各预测。
        只在 target_label 那个子图上额外画 Expected 与 Goal。
      - 右侧：该肌肉在所有实验组 (固定 + 变负载) 上的 RMSE 柱状图。

    Parameters
    ----------
    target_label : str
        当前变负载实验的 label (subject.vload_data 中的 key)。
    vload_result : dict
        vload_results[target_label]。
    planned_df_target : pd.DataFrame
    heatmap_overlays : list of (key, label, params, color, linestyle)
        预测器列表；key 需出现在 bar_colors 里才会画柱状图。
    bar_colors : dict, optional
        见 DEFAULT_BAR_COLORS。

    Returns
    -------
    (fig, rmse_at_actual_target, groups)
    """
    if bar_colors is None:
        bar_colors = DEFAULT_BAR_COLORS

    emg_col = f'emg_{target_muscle}'

    target_cutted = filter_movement_types(
        vload_result.get('cutted_data'), movement_types)
    rmse_at_actual_target = compute_rmse_at_actual_points(
        target_cutted, planned_df_target, heatmap_overlays, target_muscle)

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

    # ---------- 左侧：N 个子图 ----------
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
                subject, vparams.get('vload_file'), verbose=False)

        if (cutted_v is not None
                and emg_col in cutted_v.columns
                and 'pos_l' in cutted_v.columns):
            ax.scatter(cutted_v['pos_l'], cutted_v[emg_col],
                       s=8, alpha=0.35, color='gray',
                       label='Actual EMG')

        # Expected / Goal: 只在 target 子图上画
        if (is_target and planned_df_v is not None
                and {'Height', 'Activation'}.issubset(planned_df_v.columns)):
            h = planned_df_v['Height'].values
            a = planned_df_v['Activation'].values
            order = np.argsort(h)
            ax.plot(h[order], a[order], color='C0', linewidth=2,
                    label='Expected (vload_file)' + format_rmse_for_legend(
                        rmse_at_actual_target, 'expected'))
        if is_target and target_activation is not None:
            ax.axhline(target_activation, color='C1', linestyle=':',
                       linewidth=1.5,
                       label=f'Goal = {target_activation}')

        # 预测曲线：从 overlays 逐个画
        if (planned_df_v is not None
                and {'Height', 'Load'}.issubset(planned_df_v.columns)):
            h = planned_df_v['Height'].values
            l = planned_df_v['Load'].values
            order = np.argsort(h)

            if is_target:
                rmse_local = rmse_at_actual_target
            else:
                rmse_local = compute_rmse_at_actual_points(
                    cutted_v, planned_df_v, heatmap_overlays, target_muscle)

            for key, ov_label, ov_params, ov_color, ov_ls in heatmap_overlays:
                if ov_params is None:
                    continue
                cinfo = bar_colors.get(key)
                short_name = cinfo['name'] if cinfo else key
                try:
                    pred = predict_at(ov_params, h, l)
                    ax.plot(h[order], pred[order], color=ov_color,
                            linewidth=1.5, linestyle=ov_ls,
                            label=short_name
                                  + format_rmse_for_legend(rmse_local, key))
                except Exception as e:
                    print(f'  {short_name} 画图失败 ({vlabel}): {e}')

        ax.set_ylabel(f'{target_muscle}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        marker = '  ★' if is_target else ''
        ax.set_title(f'During {vlabel}{marker}', fontsize=9)
        if i < n_vload - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Height (m)')

    # ---------- 右侧：RMSE 柱状图 ----------
    ax = fig.add_subplot(gs[:, 1])
    groups = compute_groups_rmse_for_muscle(
        pipeline, subject, vload_results, target_muscle,
        movement_types, heatmap_overlays)

    plot_keys = [ov[0] for ov in heatmap_overlays
                 if ov[2] is not None and ov[0] in bar_colors]
    n_keys = max(len(plot_keys), 1)
    width = 0.8 / n_keys
    x = np.arange(len(groups))

    legend_handles = []
    for ki, k in enumerate(plot_keys):
        cinfo = bar_colors[k]
        vals = [g.get(k, (float('nan'), 0))[0] for g in groups]
        colors = [cinfo['vload'] if g['kind'] == 'vload' else cinfo['fixed']
                  for g in groups]
        offset = (ki - (n_keys - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=0.5)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7)
        legend_handles.append(Patch(
            facecolor=cinfo['vload'], edgecolor=BAR_EDGE_COLOR,
            label=f"{cinfo['name']} (variable load)"))
        legend_handles.append(Patch(
            facecolor=cinfo['fixed'], edgecolor=BAR_EDGE_COLOR,
            label=f"{cinfo['name']} (fixed load)"))

    labels = [g['label'] for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_xlabel('Experiment group  (fixed loads + variable loads)')
    ax.set_ylabel(f'RMSE (predicted vs actual EMG of {target_muscle})')
    ax.set_title(f'RMSE per group for {target_muscle}')
    ax.legend(handles=legend_handles, fontsize=7, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    return fig, rmse_at_actual_target, groups


def plot_vload_overlay_est_load(label, vload_result, planned_df,
                                heatmap_overlays, est_load_params,
                                target_muscle, target_activation,
                                movement_types=None, g=9.81, n_bins=40):
    """
    与 plot_vload_overlay 相同的 1×2 图，额外支持：

    左图新增一条预测曲线：
        使用 heatmap_estimated_load 参数对参數，
        以变负载实测数据中逻样本估算负载（交互力 / 加速度 + g）
        对每个样本进行预测，再按位置分筱平均后画出曲线。

    右图新增一组散点：
        变负载实测数据中逻样本估算负载 vs 高度。

    Parameters
    ----------
    est_load_params : dict or None
        由 generate_heatmaps_with_estimated_load() 产出的 P-spline（或 RBF）
        参数字典，通常加载自
        result_folder/heatmap_estimated_load/params/{musc}_est_pspline_params.pkl。
    g : float
        重力加速度 (m/s²)。
    n_bins : int
        左图预测曲线按位置分筱的筱数。

    Returns
    -------
    (fig, rmse_dict)
        rmse_dict 新增 key 'est_load':(rmse, n)。
    """
    cutted = filter_movement_types(
        vload_result.get('cutted_data'), movement_types)
    emg_col = f'emg_{target_muscle}'

    rmse_dict = compute_rmse_at_actual_points(
        cutted, planned_df, heatmap_overlays, target_muscle)

    # --- 逻样本估算负载 ---
    est_load_series = None
    est_pred_series = None
    rmse_est = None
    if (cutted is not None
            and all(c in cutted.columns
                    for c in ['force_l', 'force_r', 'acc_l', 'acc_r'])
            and est_load_params is not None):
        force_total = cutted['force_l'] + cutted['force_r']
        acc_avg = (cutted['acc_l'] + cutted['acc_r']) / 2.0
        denom = acc_avg + g
        denom = denom.where(denom.abs() > 1e-3, other=np.nan)
        est_load_series = force_total / denom

        try:
            pos_vals = cutted['pos_l'].values
            est_vals = est_load_series.values
            valid_mask = np.isfinite(est_vals) & np.isfinite(pos_vals)
            if valid_mask.sum() > 0:
                pred_all = predict_at(est_load_params, pos_vals, est_vals)
                est_pred_series = pred_all
                if emg_col in cutted.columns:
                    actual = cutted[emg_col].values
                    valid = (valid_mask
                             & np.isfinite(actual)
                             & np.isfinite(pred_all))
                    if valid.sum() > 0:
                        rmse_est = float(np.sqrt(
                            np.mean((pred_all[valid] - actual[valid]) ** 2)))
                        rmse_dict['est_load'] = (rmse_est, int(valid.sum()))
        except Exception as e:
            print(f'  est_load 预测失败: {e}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    mt_label = '+'.join(movement_types) if movement_types else 'all'
    fig.suptitle(
        f'{label}  (target muscle: {target_muscle}, '
        f'goal: {target_activation}, movement: {mt_label})',
        fontsize=12, fontweight='bold')

    # -------- 左图 --------
    ax = axes[0]
    if (cutted is not None
            and emg_col in cutted.columns
            and 'pos_l' in cutted.columns):
        ax.scatter(cutted['pos_l'], cutted[emg_col], s=8, alpha=0.35,
                   color='gray', label='Actual EMG')

    if (planned_df is not None
            and {'Height', 'Activation'}.issubset(planned_df.columns)):
        h = planned_df['Height'].values
        a = planned_df['Activation'].values
        order = np.argsort(h)
        ax.plot(h[order], a[order], color='C0', linewidth=2,
                label='Expected (vload_file)' + format_rmse_for_legend(
                    rmse_dict, 'expected', with_n=True))

    if target_activation is not None:
        ax.axhline(target_activation, color='C1', linestyle=':',
                   linewidth=1.5, label=f'Goal = {target_activation}')

    if (heatmap_overlays
            and planned_df is not None
            and {'Height', 'Load'}.issubset(planned_df.columns)):
        h = planned_df['Height'].values
        l = planned_df['Load'].values
        order = np.argsort(h)
        for key, ov_label, ov_params, ov_color, ov_ls in heatmap_overlays:
            if ov_params is None:
                continue
            try:
                pred = predict_at(ov_params, h, l)
                ax.plot(h[order], pred[order], color=ov_color, linewidth=2,
                        linestyle=ov_ls,
                        label=ov_label + format_rmse_for_legend(
                            rmse_dict, key, with_n=True))
            except Exception as e:
                print(f'  {ov_label} 画图预测失败: {e}')

    # 基于估算负载的预测曲线（分筱平均）
    if (est_pred_series is not None
            and cutted is not None
            and 'pos_l' in cutted.columns):
        pos_vals = cutted['pos_l'].values
        valid = np.isfinite(est_pred_series) & np.isfinite(pos_vals)
        if valid.sum() > 0:
            pv = pos_vals[valid]
            prv = est_pred_series[valid]
            bins = np.linspace(pv.min(), pv.max(), n_bins + 1)
            bin_centers, bin_means = [], []
            for i in range(n_bins):
                mask = (pv >= bins[i]) & (pv < bins[i + 1])
                if mask.sum() > 0:
                    bin_centers.append((bins[i] + bins[i + 1]) / 2)
                    bin_means.append(float(np.mean(prv[mask])))
            if bin_centers:
                bc = np.array(bin_centers)
                bm = np.array(bin_means)
                rmse_str = (f'  RMSE={rmse_est:.4f}'
                            if rmse_est is not None else '')
                ax.plot(bc, bm, color='C3', linewidth=2, linestyle='-.',
                        label=f'Est.Load Heatmap{rmse_str}')

    ax.set_xlabel('Height (m)')
    ax.set_ylabel(f'{target_muscle} activation')
    ax.set_title('Activation vs Height')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # -------- 右图 --------
    ax = axes[1]
    if (planned_df is not None
            and {'Height', 'Load'}.issubset(planned_df.columns)):
        h = planned_df['Height'].values
        l = planned_df['Load'].values
        order = np.argsort(h)
        ax.plot(h[order], l[order], color='C0', linewidth=2,
                label='Planned load')

    # 实际估算负载散点
    if (est_load_series is not None
            and cutted is not None
            and 'pos_l' in cutted.columns):
        pos_vals = cutted['pos_l'].values
        est_vals = est_load_series.values
        valid = np.isfinite(est_vals) & np.isfinite(pos_vals)
        if valid.sum() > 0:
            ax.scatter(pos_vals[valid], est_vals[valid],
                       s=6, alpha=0.35, color='C3',
                       label='Estimated load (actual)')

    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Load (kg)')
    ax.set_title('Planned & Estimated Load vs Height')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, rmse_dict


def plot_vload_overlay_est_load(label, vload_result, planned_df,
                                heatmap_overlays, est_load_params,
                                target_muscle, target_activation,
                                movement_types=None, g=9.81, n_bins=40):
    """
    与 plot_vload_overlay 相同的 1×2 图，额外支持：

    左图新增一条预测曲线：
        使用 heatmap_estimated_load 参数，
        以变负载实测数据中逻样本估算负载（交互力 / 加速度 + g）
        对每个样本进行预测，再按位置分筱平均后画出曲线。
        颜色使用 C4（紫色），与 RBF (C3/红) 、P-spline (C2/绿) 区分。

    右图新增一组散点：
        变负载实测数据中逻样本估算负载 vs 高度（C4/紫色）。

    Parameters
    ----------
    est_load_params : dict or None
        由 generate_heatmaps_with_estimated_load() 产出的 P-spline 参数，
        通常加载自
        result_folder/heatmap_estimated_load/params/{musc}_est_pspline_params.pkl。
    g : float
        重力加速度 (m/s²)。
    n_bins : int
        左图预测曲线按位置分筱的筱数。

    Returns
    -------
    (fig, rmse_dict)
        rmse_dict 新增 key 'est_load': (rmse, n)。
    """
    EST_COLOR = 'C4'   # 紫色，区别于 RBF(C3) 、P-spline(C2) 、Expected(C0) 、Goal(C1)

    cutted = filter_movement_types(
        vload_result.get('cutted_data'), movement_types)
    emg_col = f'emg_{target_muscle}'

    rmse_dict = compute_rmse_at_actual_points(
        cutted, planned_df, heatmap_overlays, target_muscle)

    # --- 逻样本估算负载 ---
    est_load_series = None
    est_pred_series = None
    rmse_est = None
    if (cutted is not None
            and all(c in cutted.columns
                    for c in ['force_l', 'force_r', 'acc_l', 'acc_r'])
            and est_load_params is not None):
        force_total = cutted['force_l'] + cutted['force_r']
        acc_avg = (cutted['acc_l'] + cutted['acc_r']) / 2.0
        denom = acc_avg + g
        denom = denom.where(denom.abs() > 1e-3, other=np.nan)
        est_load_series = force_total / denom

        try:
            pos_vals = cutted['pos_l'].values
            est_vals = est_load_series.values
            valid_mask = np.isfinite(est_vals) & np.isfinite(pos_vals)
            if valid_mask.sum() > 0:
                pred_all = predict_at(est_load_params, pos_vals, est_vals)
                est_pred_series = pred_all
                if emg_col in cutted.columns:
                    actual = cutted[emg_col].values
                    valid = (valid_mask
                             & np.isfinite(actual)
                             & np.isfinite(pred_all))
                    if valid.sum() > 0:
                        rmse_est = float(np.sqrt(
                            np.mean((pred_all[valid] - actual[valid]) ** 2)))
                        rmse_dict['est_load'] = (rmse_est, int(valid.sum()))
        except Exception as e:
            print(f'  est_load 预测失败: {e}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    mt_label = '+'.join(movement_types) if movement_types else 'all'
    fig.suptitle(
        f'{label}  (target muscle: {target_muscle}, '
        f'goal: {target_activation}, movement: {mt_label})',
        fontsize=12, fontweight='bold')

    # -------- 左图 --------
    ax = axes[0]
    if (cutted is not None
            and emg_col in cutted.columns
            and 'pos_l' in cutted.columns):
        ax.scatter(cutted['pos_l'], cutted[emg_col], s=8, alpha=0.35,
                   color='gray', label='Actual EMG')

    if (planned_df is not None
            and {'Height', 'Activation'}.issubset(planned_df.columns)):
        h = planned_df['Height'].values
        a = planned_df['Activation'].values
        order = np.argsort(h)
        ax.plot(h[order], a[order], color='C0', linewidth=2,
                label='Expected (vload_file)' + format_rmse_for_legend(
                    rmse_dict, 'expected', with_n=True))

    if target_activation is not None:
        ax.axhline(target_activation, color='C1', linestyle=':',
                   linewidth=1.5, label=f'Goal = {target_activation}')

    if (heatmap_overlays
            and planned_df is not None
            and {'Height', 'Load'}.issubset(planned_df.columns)):
        h = planned_df['Height'].values
        l = planned_df['Load'].values
        order = np.argsort(h)
        for key, ov_label, ov_params, ov_color, ov_ls in heatmap_overlays:
            if ov_params is None:
                continue
            try:
                pred = predict_at(ov_params, h, l)
                ax.plot(h[order], pred[order], color=ov_color, linewidth=2,
                        linestyle=ov_ls,
                        label=ov_label + format_rmse_for_legend(
                            rmse_dict, key, with_n=True))
            except Exception as e:
                print(f'  {ov_label} 画图预测失败: {e}')

    # 基于估算负载的预测曲线（分筱平均）
    if (est_pred_series is not None
            and cutted is not None
            and 'pos_l' in cutted.columns):
        pos_vals = cutted['pos_l'].values
        valid = np.isfinite(est_pred_series) & np.isfinite(pos_vals)
        if valid.sum() > 0:
            pv = pos_vals[valid]
            prv = est_pred_series[valid]
            bins = np.linspace(pv.min(), pv.max(), n_bins + 1)
            bin_centers, bin_means = [], []
            for i in range(n_bins):
                mask = (pv >= bins[i]) & (pv < bins[i + 1])
                if mask.sum() > 0:
                    bin_centers.append((bins[i] + bins[i + 1]) / 2)
                    bin_means.append(float(np.mean(prv[mask])))
            if bin_centers:
                bc = np.array(bin_centers)
                bm = np.array(bin_means)
                rmse_str = (f'  RMSE={rmse_est:.4f}'
                            if rmse_est is not None else '')
                ax.plot(bc, bm, color=EST_COLOR, linewidth=2, linestyle='-.',
                        label=f'Est.Load Heatmap{rmse_str}')

    ax.set_xlabel('Height (m)')
    ax.set_ylabel(f'{target_muscle} activation')
    ax.set_title('Activation vs Height')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # -------- 右图 --------
    ax = axes[1]
    if (planned_df is not None
            and {'Height', 'Load'}.issubset(planned_df.columns)):
        h = planned_df['Height'].values
        l = planned_df['Load'].values
        order = np.argsort(h)
        ax.plot(h[order], l[order], color='C0', linewidth=2,
                label='Planned load')

    if (est_load_series is not None
            and cutted is not None
            and 'pos_l' in cutted.columns):
        pos_vals = cutted['pos_l'].values
        est_vals = est_load_series.values
        valid = np.isfinite(est_vals) & np.isfinite(pos_vals)
        if valid.sum() > 0:
            ax.scatter(pos_vals[valid], est_vals[valid],
                       s=6, alpha=0.35, color=EST_COLOR,
                       label='Estimated load (actual)')

    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Load (kg)')
    ax.set_title('Planned & Estimated Load vs Height')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, rmse_dict


def print_vload_rmse_summary(rmse_summary,
                             keys=('expected', 'rbf', 'pspline'),
                             header_map=None,
                             title='RMSE summary (actual EMG vs predictions)'):
    """打印各变负载实验在实测点上的 RMSE 总表。"""
    if not rmse_summary:
        return
    if header_map is None:
        header_map = {'expected': 'Expected',
                      'rbf': 'Heatmap (RBF)',
                      'pspline': 'Heatmap (P-spline)'}
    col_w = max(20, max(len(label) for label in rmse_summary) + 2)
    print(f'\n========== {title} ==========')
    head = ('Label'.ljust(col_w)
            + ''.join(header_map.get(k, k).rjust(22) for k in keys) + '   n')
    print(head)
    print('-' * len(head))
    for label, d in rmse_summary.items():
        row = label.ljust(col_w)
        n_print = ''
        for k in keys:
            if k in d and np.isfinite(d[k][0]):
                row += f'{d[k][0]:>22.4f}'
                n_print = d[k][1]
            else:
                row += '                     -'
        row += f'   {n_print}'
        print(row)
    print('=' * len(head))


def print_groups_rmse(muscle, groups, keys=('rbf', 'pspline'),
                      header_map=None):
    """打印某块肌肉在所有实验组上的 RMSE 表。"""
    if header_map is None:
        header_map = {'rbf': 'RBF', 'pspline': 'P-spline'}
    print(f'\n========== Per-group RMSE for {muscle} ==========')
    head = ('Group'.ljust(20)
            + ''.join(header_map.get(k, k).rjust(12) for k in keys) + '   n')
    print(head)
    print('-' * len(head))
    for g in groups:
        row = g['label'].ljust(20)
        ns = [g[k][1] for k in keys if k in g]
        n_print = max(ns) if ns else 0
        for k in keys:
            v = g.get(k, (float('nan'), 0))[0]
            if np.isfinite(v):
                row += f'{v:>12.4f}'
            else:
                row += '           -'
        row += f'   {n_print}'
        print(row)
    print('=' * len(head))