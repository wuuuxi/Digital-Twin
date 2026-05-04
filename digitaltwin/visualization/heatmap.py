"""
激活图可视化模块
提供位置-负载-肌肉激活关系的 3D 散点图、曲面图和热力图。

原始实现位于 src/heatmap.py 的绘图部分，
现已重构并集成到 Digital Twin 框架中。

用法示例:
    from digitaltwin.visualization.heatmap import (
        plot_activation_3d, compare_activation_maps, draw_heatmap_2d,
    )
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import seaborn as sns
import os

from digitaltwin.analysis.rbf_fitting import rbf_fit, rbf_predict, predict_at

DEFAULT_DATA_LEN = 50


def plot_activation_3d(data, params, pos_col='pos_l', load_col='load', emg_col='emg0',
                       label=None, cmap='hot', result_folder=None):
    """
    拟合 RBF 并绘制位置-负载-激活的 3D 散点图 + 曲面图。

    Parameters
    ----------
    data : pd.DataFrame
    pos_col, load_col, emg_col : str
    label : str, optional
    cmap : str
    result_folder : str, optional
    num_centers, sigma, data_len : RBF 参数

    Returns
    -------
    list
        [xi, yi, zi, centers, weights, scaler, sigma]
    """

    xi, yi, zi = params['xi'], params['yi'], params['zi']

    # ---------- 原始散点 3D ----------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[pos_col], data[load_col], data[emg_col],
                         c=data[emg_col], cmap=cmap, s=10)
    ax.set_xlabel('Height (m)'); ax.set_ylabel('Load (kg)')
    ax.set_zlabel('Activation'); ax.view_init(elev=30, azim=-135)
    plt.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cax, label='Activation')
    z_min, z_max = ax.get_zlim()
    cbar_min, cbar_max = scatter.get_clim()
    if result_folder:
        plt.savefig(os.path.join(result_folder,
                    f'{label}_original.png'))
    # plt.close(fig)

    # ---------- RBF 散点 3D ----------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xi, yi, zi, c=zi, cmap=cmap, s=10,
                         vmin=cbar_min, vmax=cbar_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('Height (m)'); ax.set_ylabel('Load (kg)')
    ax.set_zlabel('Activation'); ax.view_init(elev=30, azim=-135)
    plt.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cax, label='Activation')
    if result_folder:
        plt.savefig(os.path.join(result_folder,
                    f'{label}_RBF_scatter.png'))
    # plt.close(fig)

    # ---------- RBF 曲面 3D ----------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap=cmap,
                           vmin=cbar_min, vmax=cbar_max, edgecolor='none')
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('Height (m)'); ax.set_ylabel('Load (kg)')
    ax.set_zlabel('Activation'); ax.view_init(elev=30, azim=-135)
    plt.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(surf, cax=cax, label='Activation')
    if result_folder:
        plt.savefig(os.path.join(result_folder,
                    f'{label}_RBF.png'))
    # plt.close(fig)


def compare_activation_maps(data_robot, data_smith, pos_col='pos_l',
                            load_col='load', emg_col='emg0',
                            label=None, cmap='viridis', result_folder=None,
                            num_centers=20, sigma=1.0, data_len=DEFAULT_DATA_LEN):
    """
    对比两组数据（如机器人 vs 史密斯机）的激活图。

    Returns
    -------
    list
        [xi, yi, zi_smith, centers, weights, scaler, sigma]
    """
    x1, y1, z1 = data_smith[pos_col], data_smith[load_col], data_smith[emg_col]
    x2, y2, z2 = data_robot[pos_col], data_robot[load_col], data_robot[emg_col]

    xi = np.linspace(min(min(x1), min(x2)), max(max(x1), max(x2)), data_len)
    yi = np.linspace(min(min(y1), min(y2)), max(max(y1), max(y2)), data_len)
    xi, yi = np.meshgrid(xi, yi)

    c1, w1, s1, sig1 = rbf_fit((x1, y1), z1, num_centers=num_centers, sigma=sigma)
    zi_1 = rbf_predict((xi.flatten(), yi.flatten()), c1, w1, s1, sig1).reshape(xi.shape)

    c2, w2, s2, sig2 = rbf_fit((x2, y2), z2, num_centers=num_centers, sigma=sigma)
    zi_2 = rbf_predict((xi.flatten(), yi.flatten()), c2, w2, s2, sig2).reshape(xi.shape)

    # 绘图
    fig = plt.figure(figsize=(14, 6))
    if label:
        fig.suptitle(label, fontweight='bold')

    for idx, (data_src, zi, title) in enumerate([
        (data_smith, zi_1, 'Smith'), (data_robot, zi_2, 'Robot')
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        scatter = ax.scatter(data_src[pos_col], data_src[load_col],
                             data_src[emg_col], c=data_src[emg_col],
                             cmap=cmap, s=10)
        plt.colorbar(scatter, ax=ax)
        ax.set_xlabel('Height (m)'); ax.set_ylabel('Load (%1RM)')
        ax.set_zlabel('Activation'); ax.set_title(title)
        ax.view_init(elev=30, azim=-135)

    if result_folder and label:
        plt.savefig(os.path.join(result_folder, f'{label}.png'))
    # plt.close(fig)

    # RBF 曲面对比
    fig = plt.figure(figsize=(14, 6))
    for idx, (zi, title, z_min, z_max) in enumerate([
        (zi_1, 'Smith', None, None), (zi_2, 'Robot', None, None)
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap=cmap, edgecolor='none')
        plt.colorbar(surf, ax=ax)
        ax.set_xlabel('Height (m)'); ax.set_ylabel('Load (%1RM)')
        ax.set_zlabel('Activation'); ax.set_title(title)
        ax.view_init(elev=30, azim=-135)

    if result_folder and label:
        plt.savefig(os.path.join(result_folder,
                    f'{label}_RBF_pos-load-activation.png'))
    # plt.close(fig)

    return [xi, yi, zi_1, c1, w1, s1, sig1]


def draw_heatmap_2d(params, sigma_smooth=1, label=None, result_folder=None):
    """
    绘制简单的 2D 高斯平滑热力图。

    Parameters
    ----------
    data : np.ndarray
        二维数据矩阵
    sigma_smooth : float
        高斯平滑 sigma
    label : str, optional
    result_folder : str, optional
    """
    xi, yi, zi = params['xi'], params['yi'], params['zi']
    zi = gaussian_filter(zi, sigma=sigma_smooth)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xi, yi, zi, levels=100, cmap='hot')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('Height (m)'); ax.set_ylabel('Load (kg)')
    ax.set_title(label)
    if result_folder:
        plt.savefig(os.path.join(result_folder,
                                 f'{label}_RBF_2D.png'))
    # plt.close(fig)
    # plt.show()


def plot_load_slices_comparison(data, params_orig, params_mono, muscle,
                                pos_col='pos_l', load_col='load',
                                result_folder=None):
    """
    绘制每个负载下原始数据散点与两个拟合曲线的对比。

    每个子图对应一个负载：
    - 灰色散点：该负载下的原始高度-激活数据
    - 蓝色实线：原始 RBF 拟合在该负载下的预测曲线
    - 红色虚线：平滑单调投影后的预测曲线

    Parameters
    ----------
    data : pd.DataFrame
        原始数据，需包含 pos_col, load_col, emg_<muscle> 列。
    params_orig : dict
        原始 RBF 拟合参数 (fit_activation_map 返回值)。
    params_mono : dict or None
        平滑单调投影后的参数字典；为 None 时不画。
    muscle : str
        肌肉名，emg 列名为 f'emg_{muscle}'。
    pos_col, load_col : str
    result_folder : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    emg_col = f'emg_{muscle}'
    if emg_col not in data.columns:
        return None

    loads = sorted(data[load_col].unique())
    n = len(loads)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(f'{muscle} — Activation vs Height (per Load)',
                 fontsize=14, fontweight='bold')

    # 从 params_orig 的 xi 网格提取一维高度序列
    xi = np.asarray(params_orig['xi'])
    h_grid = xi[0, :] if xi.ndim == 2 else np.unique(xi)

    # 统一 y 轴范围，便于子图间对比
    y_min = float(data[emg_col].min())
    y_max = float(data[emg_col].max())
    y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.05

    for idx, load in enumerate(loads):
        ax = axes[idx // n_cols][idx % n_cols]
        subset = data[data[load_col] == load]
        ax.scatter(subset[pos_col], subset[emg_col], s=8, alpha=0.4,
                   color='gray', label='Raw data')

        y_fixed = np.full_like(h_grid, float(load), dtype=float)

        # 原始 RBF 预测曲线
        z_orig = predict_at(params_orig, h_grid, y_fixed)
        ax.plot(h_grid, z_orig, color='C0', linewidth=2,
                label='RBF (original)')

        # 平滑单调投影后的预测曲线（来自后处理网格插值）
        if params_mono is not None:
            z_mono = predict_at(params_mono, h_grid, y_fixed)
            ax.plot(h_grid, z_mono, color='C3', linewidth=2,
                    linestyle='--', label='Smooth monotonic')

        ax.set_xlabel('Height (m)')
        ax.set_ylabel('Activation')
        ax.set_title(f'Load = {load} kg')
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc='best')

    # 隐藏多余子图
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if result_folder:
        plt.savefig(os.path.join(result_folder,
                                 f'{muscle}_load_slices.png'))
    return fig


def draw_load_sensitivity_heatmap_2d(params, sigma_smooth=1, label=None,
                                     result_folder=None):
    """
    绘制负载敏感度热力图：∂activation/∂load。

    对 RBF 拟合曲面沿负载轴（y 轴）求数值偏导数，
    表征在不同高度和不同负载下，肌肉激活对负载变化的敏感程度。
    值越大说明在该位置-负载组合下，增加负载能更有效地提高激活。

    Parameters
    ----------
    params : dict
        fit_activation_map 返回的参数字典，包含 xi, yi, zi 网格。
    sigma_smooth : float
        高斯平滑 sigma。
    label : str, optional
        肌肉名称标签。
    result_folder : str, optional
        保存路径。
    """
    xi, yi, zi = params['xi'], params['yi'], params['zi']

    # yi 沿 axis=0 变化（meshgrid 的行方向）
    yi_1d = yi[:, 0]  # 一维负载数组
    dz_dload = np.gradient(zi, yi_1d, axis=0)
    dz_dload = gaussian_filter(dz_dload, sigma=sigma_smooth)

    fig, ax = plt.subplots(figsize=(8, 6))
    # 使用发散色图，正值=激活随负载增加，负值=激活随负载减少
    vmax = np.max(np.abs(dz_dload))
    contour = ax.contourf(xi, yi, dz_dload, levels=100,
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(contour, ax=ax, label='∂Activation / ∂Load')
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Load (kg)')
    ax.set_title(f'{label} — Load Sensitivity' if label else 'Load Sensitivity')
    if result_folder:
        plt.savefig(os.path.join(result_folder,
                                 f'{label}_load_sensitivity_2D.png'))
    # plt.close(fig)