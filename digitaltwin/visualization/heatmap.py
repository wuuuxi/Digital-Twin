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

from digitaltwin.analysis.rbf_fitting import rbf_fit, rbf_predict

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
    plt.close(fig)

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
    plt.close(fig)

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
    plt.close(fig)


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
    plt.close(fig)

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
    plt.close(fig)

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
    plt.close(fig)
    # plt.show()