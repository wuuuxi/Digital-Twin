"""
变负载对比可视化模块（Robot 运动学 + EMG 肌肉激活）。

提供：
  - plot_robot_kinematics_bar: 不同负载下 Robot 位置/速度/加速度均值柱状图
  - plot_emg_activation_bar:  不同负载下肌肉激活均值柱状图
"""
import matplotlib.pyplot as plt
import numpy as np

LOAD_COLORS = plt.cm.tab10.colors
VLOAD_COLORS = plt.cm.Set2.colors


def plot_robot_kinematics_bar(fixed_results, vload_results=None):
    """
    不同负载下 Robot 位置、速度、加速度的均值 ± 标准差柱状图。
    1行3列：pos_l / vel_l / acc_l
    """
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys()) if vload_results else []
    cols = ['pos_l', 'vel_l', 'acc_l']
    labels = ['Position (m)', 'Velocity (m/s)', 'Acceleration (m/s\u00b2)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Robot Kinematics by Load: Fixed + Variable Loads',
                 fontsize=14, fontweight='bold')

    for ci, (col, label) in enumerate(zip(cols, labels)):
        ax = axes[ci]
        bm, bs, bl, bc, bh = [], [], [], [], []

        for i, lw in enumerate(fixed_keys):
            cd = fixed_results[lw].get('cutted_data')
            if cd is None or col not in cd.columns: continue
            vals = cd[col].dropna().values
            if len(vals) == 0: continue
            bm.append(np.mean(np.abs(vals))); bs.append(np.std(np.abs(vals)))
            bl.append(f'{lw} kg'); bc.append(LOAD_COLORS[i % len(LOAD_COLORS)]); bh.append('')

        for j, vk in enumerate(vload_keys):
            cd = vload_results[vk].get('cutted_data')
            if cd is None or col not in cd.columns: continue
            vals = cd[col].dropna().values
            if len(vals) == 0: continue
            bm.append(np.mean(np.abs(vals))); bs.append(np.std(np.abs(vals)))
            bl.append(f'VL:{vk}'); bc.append(VLOAD_COLORS[j % len(VLOAD_COLORS)]); bh.append('//')

        if not bm:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label); continue

        x = np.arange(len(bm))
        bars = ax.bar(x, bm, yerr=bs, capsize=4, color=bc, alpha=0.8, edgecolor='black', linewidth=0.5)
        for bi, h in enumerate(bh): bars[bi].set_hatch(h)
        for xi, m, s in zip(x, bm, bs):
            ax.text(xi, m + s + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(bl, fontsize=7, rotation=30, ha='right')
        ax.set_title(label); ax.set_ylabel(label); ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


def plot_emg_activation_bar(fixed_results, musc_label, vload_results=None):
    """
    不同负载下肌肉激活均值 ± 标准差柱状图。
    每块肌肉一个子图，X 轴为不同负载。
    """
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys()) if vload_results else []
    n_muscles = len(musc_label)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows),
                             squeeze=False)
    fig.suptitle('Mean EMG Activation by Load: Fixed + Variable Loads',
                 fontsize=14, fontweight='bold')

    for idx, musc in enumerate(musc_label):
        ax = axes[idx // n_cols][idx % n_cols]
        emg_col = f'emg_{musc}'
        bm, bs, bl, bc, bh = [], [], [], [], []

        for i, lw in enumerate(fixed_keys):
            cd = fixed_results[lw].get('cutted_data')
            if cd is None or emg_col not in cd.columns: continue
            vals = cd[emg_col].dropna().values
            if len(vals) == 0: continue
            bm.append(np.mean(vals)); bs.append(np.std(vals))
            bl.append(f'{lw} kg'); bc.append(LOAD_COLORS[i % len(LOAD_COLORS)]); bh.append('')

        for j, vk in enumerate(vload_keys):
            cd = vload_results[vk].get('cutted_data')
            if cd is None or emg_col not in cd.columns: continue
            vals = cd[emg_col].dropna().values
            if len(vals) == 0: continue
            bm.append(np.mean(vals)); bs.append(np.std(vals))
            bl.append(f'VL:{vk}'); bc.append(VLOAD_COLORS[j % len(VLOAD_COLORS)]); bh.append('//')

        if not bm:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(musc); continue

        x = np.arange(len(bm))
        bars = ax.bar(x, bm, yerr=bs, capsize=4, color=bc, alpha=0.8, edgecolor='black', linewidth=0.5)
        for bi, h in enumerate(bh): bars[bi].set_hatch(h)
        for xi, m, s in zip(x, bm, bs):
            ax.text(xi, m + s + 0.005, f'{m:.3f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(bl, fontsize=6, rotation=30, ha='right')
        ax.set_title(musc); ax.set_ylabel('Activation'); ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=7)

    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()
    return fig