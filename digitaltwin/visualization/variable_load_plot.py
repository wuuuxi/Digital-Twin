"""
变负载优化结果可视化模块
提供优化负载曲线、激活曲线和热力图叠加的绘图函数。

原始实现位于 src/variable_load.py 的绘图部分，
现已重构并集成到 Digital Twin 框架中。

用法示例:
    from digitaltwin.visualization.variable_load_plot import (
        plot_variable_load_result,
        plot_variable_load_result_multi_muscles,
    )
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os


def plot_variable_load_result(subject, xi, yi, zi, heights, opti_loads,
                              activations, epsilons, idx, goal, title,
                              result_folder=None):
    """
    绘制单肌肉变负载优化结果（热力图 + 优化曲线 + 激活曲线 + 负载曲线）。

    Parameters
    ----------
    subject : Subject
        实验配置（需要 height_range）
    xi, yi, zi : np.ndarray
        RBF 网格数据
    heights : list of np.ndarray
        每个 epsilon 的高度序列
    opti_loads : list of np.ndarray
        每个 epsilon 的优化负载
    activations : list of np.ndarray
        每个 epsilon 的激活值
    epsilons : list of float
    idx : int
        肌肉索引
    goal : float
        目标激活值
    title : str
    result_folder : str, optional
    """
    fig = plt.figure(figsize=(5, 7.5))
    plt.subplots_adjust(left=0.152, bottom=0.102, right=0.905, top=0.93)
    gs = GridSpec(3, 1, height_ratios=[6, 2, 2], hspace=0.4)
    line_styles = ['-', '--', '-.', ':']
    color = ['#1f77b4', '#2ca02c', '#9467bd']

    # --- 热力图 + 优化曲线 ---
    ax1 = fig.add_subplot(gs[0, :])
    contour = ax1.contourf(xi, yi, zi, levels=100, cmap='hot')
    plt.colorbar(contour, ax=ax1)
    ax1.contour(xi, yi, zi, levels=[goal], colors='white',
                linewidths=1.5, linestyles='--')
    for i in range(len(epsilons)):
        ax1.plot(heights[i], opti_loads[i], color='#1f77b4',
                 linestyle=line_styles[i % 4], label='Optimization')
    ax1.set_xlim(subject.height_range[0], subject.height_range[1])
    ax1.set_xlabel('Height (m)'); ax1.set_ylabel('Load (kg)')
    ax1.set_title(title)

    # --- 激活曲线 ---
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(heights[0], goal * np.ones_like(activations[0][:, idx]),
             '--', label='Goal', color='#ff7f0e')
    for i in range(len(epsilons)):
        ax2.plot(heights[i], activations[i][:, idx], color=color[i % 3],
                 linestyle=line_styles[i % 4], label='Optimization')
    ax2.legend(loc='lower left', frameon=True, ncol=4, columnspacing=0.5)
    ax2.set_xlim(subject.height_range[0], subject.height_range[1])
    ax2.set_ylabel('Activation'); ax2.set_xlabel('Height (m)')

    # --- 负载曲线 ---
    ax3 = fig.add_subplot(gs[2, :])
    for i in range(len(epsilons)):
        ax3.plot(heights[i], opti_loads[i], color=color[i % 3],
                 linestyle=line_styles[i % 4])
    ax3.set_xlim(subject.height_range[0], subject.height_range[1])
    ax3.set_xlabel('Height (m)'); ax3.set_ylabel('Load (kg)')

    if result_folder:
        plt.savefig(os.path.join(result_folder, f'{title}_{goal}.png'))

    # 导出 CSV
    df = pd.DataFrame({
        'Load': opti_loads[0], 'Height': heights[0],
        'Activation': activations[0][:, idx],
    })
    csv_path = (os.path.join(result_folder, f'{title}_{goal}_{epsilons[0]}.csv')
                if result_folder else f'{title}_{goal}_{epsilons[0]}.csv')
    df.to_csv(csv_path)

    # --- 遮罩热力图 ---
    fig2 = plt.figure(figsize=(5, 7.5))
    plt.subplots_adjust(left=0.152, bottom=0.102, right=0.905, top=0.93)
    gs2 = GridSpec(3, 1, height_ratios=[6, 2, 2], hspace=0.4)

    ax_mask = fig2.add_subplot(gs2[1:, :])
    masked_zi = zi.copy()
    for i in range(len(epsilons)):
        for j, x_val in enumerate(xi[0, :]):
            k_idx = np.argmin(np.abs(heights[i] - x_val))
            curve_y = opti_loads[i][k_idx]
            k = np.argmin(np.abs(yi[:, j] - curve_y))
            masked_zi[k + 1:, j] = np.nan
            masked_zi[:k, j] = masked_zi[k, j]
    vmin, vmax = np.nanmin(zi), np.nanmax(zi)
    contour_m = ax_mask.contourf(xi, yi, masked_zi, levels=100, cmap='hot',
                                 vmin=vmin, vmax=vmax)
    plt.colorbar(contour_m, ax=ax_mask)
    for i in range(len(epsilons)):
        ax_mask.plot(heights[i], opti_loads[i], color='#1f77b4',
                     linestyle=line_styles[i % 4], linewidth=2)
    ax_mask.set_title(title); ax_mask.set_axis_off()

    ax_full = fig2.add_subplot(gs2[0, :])
    contour_f = ax_full.contourf(xi, yi, zi, levels=100, cmap='hot')
    plt.colorbar(contour_f, ax=ax_full)
    ax_full.set_axis_off()

    if result_folder:
        plt.savefig(os.path.join(result_folder, f'{title}_{goal}_masked.png'))
    plt.close('all')


def plot_variable_load_result_multi_muscles(xi, yi, zi_all, heights, opti_loads,
                                            activations, epsilons, idx, goal,
                                            title, result_folder=None):
    """
    绘制多肌肉变负载优化结果（含危险区域标注）。

    Parameters
    ----------
    xi, yi : np.ndarray
        网格
    zi_all : list of np.ndarray
        各肌肉的 RBF 预测网格
    heights, opti_loads, activations, epsilons, idx, goal, title
        与 plot_variable_load_result 相同
    result_folder : str, optional
    """
    zi = zi_all[0]
    line_styles = ['-', '--', '-.', ':']

    fig = plt.figure(figsize=(5, 7.5))
    plt.subplots_adjust(left=0.152, bottom=0.102, right=0.905, top=0.93)
    gs = GridSpec(3, 1, height_ratios=[6, 2, 2], hspace=0.4)

    # 遮罩热力图
    ax1 = fig.add_subplot(gs[1:, :])
    masked_zi = zi.copy()
    for i in range(len(epsilons)):
        for j, x_val in enumerate(xi[0, :]):
            k_idx = np.argmin(np.abs(heights[i] - x_val))
            curve_y = opti_loads[i][k_idx]
            k = np.argmin(np.abs(yi[:, j] - curve_y))
            masked_zi[k + 1:, j] = np.nan
            masked_zi[:k, j] = masked_zi[k, j]

    vmin, vmax = np.nanmin(zi), np.nanmax(zi)
    contour = ax1.contourf(xi, yi, masked_zi, levels=100, cmap='hot',
                           vmin=vmin, vmax=vmax)
    plt.colorbar(contour, ax=ax1)
    for i in range(len(epsilons)):
        ax1.plot(heights[i], opti_loads[i], color='#1f77b4',
                 linestyle=line_styles[i % 4], linewidth=2)
    ax1.set_title(title); ax1.set_axis_off()

    # 含过激活区域标注的热力图
    ax2 = fig.add_subplot(gs[0, :])
    contour2 = ax2.contourf(xi, yi, zi, levels=100, cmap='hot')
    plt.colorbar(contour2, ax=ax2)

    # 标注各肌肉过激活区域
    hatch_styles = ['//', '\\\\', 'xx']
    for m_idx, zi_m in enumerate(zi_all):
        if m_idx < len(hatch_styles):
            thresh = np.percentile(zi_m, 85)
            ax2.contourf(xi, yi, zi_m, levels=[thresh, np.max(zi_m)],
                         colors='none', hatches=[hatch_styles[m_idx]], alpha=0)

    ax2.set_axis_off()

    if result_folder:
        plt.savefig(os.path.join(result_folder,
                    f'{title}_{goal}_multi_muscles.png'))
    plt.close('all')


def plot_danger_area(subject, xi, yi, zi, heights, opti_loads, activations,
                    epsilons, idx, goal, title, danger_muscle_files=None,
                    dangerous_th=None, training_th=None, result_folder=None):
    """
    绘制含危险区域和低效区域标注的变负载结果图。

    Parameters
    ----------
    danger_muscle_files : list of str, optional
        危险肌肉的 RBF 参数文件
    dangerous_th : list of float, optional
        过激活阈值
    training_th : list of float, optional
        低效阈值
    """
    from digitaltwin.analysis.rbf_fitting import rbf_predict as _rbf_predict

    line_styles = ['-', '--', '-.', ':']
    color = ['#1f77b4', '#2ca02c', '#9467bd']

    # 加载危险肌肉的 RBF 数据
    zi_all = []
    if danger_muscle_files is not None:
        centers, weights, scaler, sigma = [], [], [], []
        for file in danger_muscle_files:
            base = (subject.muscle_folder if subject.load_previous_data
                    else subject.result_folder)
            import pickle
            with open(os.path.join(base, file), 'rb') as f:
                p = pickle.load(f)
                centers.append(p['centers'])
                weights.append(p['weights'])
                scaler.append(p['scaler'])
                sigma.append(p['sigma'])
        for i in range(len(danger_muscle_files)):
            z = _rbf_predict((xi.flatten(), yi.flatten()),
                             centers[i], weights[i], scaler[i], sigma[i])
            zi_all.append(z.reshape(xi.shape))

    fig = plt.figure(figsize=(5, 7.5))
    plt.subplots_adjust(left=0.152, bottom=0.102, right=0.905, top=0.93)
    gs = GridSpec(3, 1, height_ratios=[6, 2, 2], hspace=0.4)

    ax1 = fig.add_subplot(gs[0, :])
    contour = ax1.contourf(xi, yi, zi, levels=100, cmap='hot')
    plt.colorbar(contour, ax=ax1)
    legend_elements = []

    if zi_all and dangerous_th:
        for i in range(min(len(zi_all), len(dangerous_th))):
            ax1.contourf(xi, yi, zi_all[i],
                         levels=[dangerous_th[i], np.max(zi_all[i])],
                         colors='none', hatches=['//'], alpha=0)
        legend_elements.append(
            Patch(facecolor='none', edgecolor='black',
                  hatch='//', label='Over-Activate Area'))

        if training_th and idx < len(training_th):
            contour_below = ax1.contourf(
                xi, yi, zi, levels=[np.min(zi), training_th[idx]],
                colors='none', alpha=0)
            contour_below.set_edgecolor('white')
            contour_below.set_linewidth(0)
            contour_below.set_hatch('\\\\\\\\')
            contour_below.set_alpha(1.0)
            contour_below.set_facecolor('none')
            legend_elements.append(
                Patch(facecolor='none', edgecolor='white',
                      hatch='\\\\', label='Inefficient Area'))

    ax1.contour(xi, yi, zi, levels=[goal], colors='white',
                linewidths=1.5, linestyles='--')
    legend_elements.append(
        Line2D([0], [0], color='white', linestyle='--',
               linewidth=1.5, label='Goal'))

    for i in range(len(epsilons)):
        line, = ax1.plot(heights[i], opti_loads[i], color='#1f77b4',
                         linestyle=line_styles[i % 4], label='Load Profile')
        legend_elements.append(line)

    ax1.set_xlim(subject.height_range[0], subject.height_range[1])
    ax1.set_xlabel('Height (m)'); ax1.set_ylabel('Load (kg)')
    ax1.set_title(title)
    ax1.legend(handles=legend_elements, loc='lower right',
               frameon=False, labelcolor='white', fontsize=12)

    # 激活曲线
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(heights[0], goal * np.ones_like(activations[0][:, idx]),
             '--', label='Goal', color='#ff7f0e')
    for i in range(len(epsilons)):
        ax2.plot(heights[i], activations[i][:, idx], color=color[i % 3],
                 linestyle=line_styles[i % 4], label='Optimization')
    ax2.legend(loc='lower left', frameon=True, ncol=4, columnspacing=0.5)
    ax2.set_xlim(subject.height_range[0], subject.height_range[1])
    ax2.set_ylabel('Activation'); ax2.set_xlabel('Height (m)')

    # 负载曲线
    ax3 = fig.add_subplot(gs[2, :])
    for i in range(len(epsilons)):
        ax3.plot(heights[i], opti_loads[i], color=color[i % 3],
                 linestyle=line_styles[i % 4])
    ax3.set_xlim(subject.height_range[0], subject.height_range[1])
    ax3.set_xlabel('Height (m)'); ax3.set_ylabel('Load (kg)')

    if result_folder:
        plt.savefig(os.path.join(result_folder, f'{title}_{goal}.png'))
    plt.close('all')