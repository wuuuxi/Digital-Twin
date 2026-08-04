"""
负载/力/功率 - 肌肉激活曲线（固定高度切片）

在文件顶部用 X_AXIS 选择横轴指标，仅支持三选一：
  - 'load'  : 横轴 = 负载 (kg)，并叠加 heatmap 拟合曲线（P-spline + RBF）。
  - 'force' : 横轴 = 总交互力 force_l + force_r (N)，仅画原始散点。
  - 'power' : 横轴 = 功率 (force_l + force_r) * vel_l (W)，仅画原始散点。

即：选择 force / power 时不拟合、不画曲线，只画原始数据点。

选定指标后输出四张大图（行 = 肌肉，列 = 三个固定高度）：
  图 1：横轴 = 指标本身，纵轴 = activation；
  图 2：横轴 = log(指标)，纵轴 = activation；
  图 3：横轴 = 指标本身，纵轴 = log(RMS 激活)；
  图 4：横轴 = log(指标)，纵轴 = log(RMS 激活)。
各子图坐标轴自动缩放（不再统一行/列范围）。

选择 'force' / 'power' 时，额外再输出一张『不分高度、所有点』的散点图，
仍按肌肉分子图（横轴 = 指标，纵轴 = activation，散点按负载着色）。

用法：
    python example_load_activation_curve.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.analysis.heatmap.rbf_fitting import (
    fit_activation_map, predict_at)


# ============================================================
#  选项
# ============================================================
X_AXIS = 'power'   # 横轴指标，只能选 'load' / 'force' / 'power'

CONFIG_FILE = '../config/20260513_squat_FTS09_mvc.json'
TARGET_MUSCLES = ['RBF', 'RVL', 'RGlutMax', 'RVM']
# CONFIG_FILE = '../config/20250409_squat_NCMP001.json'
# TARGET_MUSCLES = ["FibLon", "GL", "VL"]
MOVEMENT_TYPES = ['upward']

# HEIGHT_FRACTIONS = [0.70, 0.75, 0.80]   # 在高度范围内取的相对位置
HEIGHT_FRACTIONS = [0.25, 0.50, 0.75]
HEIGHT_WINDOW_FRAC = 0.05               # 散点高度窗口半宽（占总高度范围比例）
N_LOAD_GRID = 100                       # 横轴方向曲线采样点数（仅 load 画曲线时用）
FORCE_COLUMNS = ['force_l', 'force_r']  # 相加得到总交互力 (N)
VELOCITY_COLUMN = 'vel_l'               # 速度列 (m/s)，power = force * 速度
LOAD_COLUMNS = ['load', 'load_value', 'load_weight']  # 负载列候选名
LOAD_COLORS = plt.cm.tab10.colors       # 不同负载使用不同颜色的散点

# 每个横轴指标的单位 / 标题 / 文件名前缀
X_AXIS_SPECS = {
    'load':  {'unit': 'kg', 'title': 'Load',  'file': 'load'},
    'force': {'unit': 'N',  'title': 'Force', 'file': 'force'},
    'power': {'unit': 'W',  'title': 'Power', 'file': 'power'},
}


def safe_log10(arr):
    """对非正元素返回 nan（不参与绘图），正元素取自然对数 ln（以 e 为底）。"""
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    pos = arr > 0
    out[pos] = np.log(arr[pos])
    return out


def _fit_surfaces(data, x_col, emg_col, height_range):
    """在 (pos_l, x_col) 空间上拟合 P-spline 主曲面 + RBF 基线。"""
    params_rbf = fit_activation_map(
        data, pos_col='pos_l', load_col=x_col, emg_col=emg_col,
        num_centers=20, sigma=1.0, data_len=50,
        height_range=height_range)
    params_psp = fit_activation_map(
        data, pos_col='pos_l', load_col=x_col, emg_col=emg_col,
        num_centers=20, sigma=1.0, data_len=50,
        height_range=height_range,
        use_pspline=True,
        pspline_n_basis_h=20,
        pspline_n_basis_l=10,
        pspline_degree=3,
        pspline_lambda_h=0.1,
        pspline_lambda_l=1.0,
        pspline_solver='auto',
        pspline_max_iter=2000)
    return params_psp, params_rbf


def _draw_figure(cutted, params, heights, h_window, x_grid, x_grid_log,
                 x_log, y_log, x_label, suptitle, out_path, draw_fit):
    """画一张 肌肉 × 高度 的大图。

    x_log / y_log 控制横纵轴是否取 log（自然对数 ln）；draw_fit 控制是否叠加拟合曲线。
    """
    n_rows = len(TARGET_MUSCLES)
    n_cols = len(heights)
    grid = x_grid_log if x_log else x_grid

    # 检测负载列，为每个负载分配一种颜色（散点按负载着色）
    load_col = next((col for col in LOAD_COLUMNS if col in cutted.columns), None)
    if load_col is not None:
        load_values = sorted(cutted[load_col].dropna().unique())
        load_color_map = {lv: LOAD_COLORS[i % len(LOAD_COLORS)]
                          for i, lv in enumerate(load_values)}
    else:
        load_values = []
        load_color_map = {}

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(suptitle, fontsize=15, fontweight='bold')

    for r, musc in enumerate(TARGET_MUSCLES):
        emg_col = f'emg_{musc}'
        params_psp = params.get(musc)
        params_rbf = params.get(f'{musc}_rbf')

        if emg_col not in cutted.columns:
            for c in range(n_cols):
                ax = axes[r][c]
                ax.set_axis_off()
                ax.set_title(f'{musc}: 无 {emg_col} 列')
            continue

        for c, h in enumerate(heights):
            ax = axes[r][c]
            frac = HEIGHT_FRACTIONS[c]

            mask = ((cutted['pos_l'] >= h - h_window) &
                    (cutted['pos_l'] <= h + h_window))
            if x_log:
                mask = mask & (cutted['xval'] > 0)
            if y_log:
                mask = mask & (cutted[emg_col] > 0)
            near = cutted[mask]

            if load_col is not None:
                # 按负载分组着色，每个负载一种颜色
                for lv in load_values:
                    sub = near[near[load_col] == lv]
                    if len(sub) == 0:
                        continue
                    x_raw = np.log(sub['xval']) if x_log else sub['xval']
                    y_raw = safe_log10(sub[emg_col]) if y_log else sub[emg_col]
                    ax.scatter(x_raw, y_raw, s=12, alpha=0.5,
                               color=load_color_map[lv],
                               label=(f'{lv:g} kg'
                                      if (r == 0 and c == 0) else None))
            else:
                x_raw = np.log(near['xval']) if x_log else near['xval']
                y_raw = safe_log10(near[emg_col]) if y_log else near[emg_col]
                ax.scatter(x_raw, y_raw, s=12, alpha=0.4,
                           color='gray', label='Raw data')

            if draw_fit:
                h_fixed = np.full_like(grid, float(h), dtype=float)
                x_plot = np.log(grid) if x_log else grid
                if params_psp is not None:
                    z = predict_at(params_psp, h_fixed, grid)
                    z = safe_log10(z) if y_log else z
                    ax.plot(x_plot, z, color='C3', linewidth=2,
                            linestyle='-', label='Heatmap (P-spline)')
                if params_rbf is not None:
                    z = predict_at(params_rbf, h_fixed, grid)
                    z = safe_log10(z) if y_log else z
                    ax.plot(x_plot, z, color='C0', linewidth=2,
                            linestyle='--', label='Heatmap (RBF)')

            ax.set_xlabel(x_label)
            ax.set_ylabel('log(RMS activation)' if y_log else 'Activation')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{musc} @ h={h:.3f} m ({int(frac * 100)}% range)')
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc='best')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _draw_all_points_figure(cutted, x_label, suptitle, out_path):
    """画一张『不分高度、所有点』的散点大图：每个肌肉一个子图。

    横轴 = 指标 (xval)，纵轴 = activation，散点按负载着色（不做高度窗口过滤）。
    """
    n_cols = len(TARGET_MUSCLES)

    # 检测负载列，为每个负载分配一种颜色
    load_col = next((col for col in LOAD_COLUMNS if col in cutted.columns), None)
    if load_col is not None:
        load_values = sorted(cutted[load_col].dropna().unique())
        load_color_map = {lv: LOAD_COLORS[i % len(LOAD_COLORS)]
                          for i, lv in enumerate(load_values)}
    else:
        load_values = []
        load_color_map = {}

    fig, axes = plt.subplots(1, n_cols,
                             figsize=(5 * n_cols, 4),
                             squeeze=False)
    fig.suptitle(suptitle, fontsize=15, fontweight='bold')

    for c, musc in enumerate(TARGET_MUSCLES):
        ax = axes[0][c]
        emg_col = f'emg_{musc}'
        if emg_col not in cutted.columns:
            ax.set_axis_off()
            ax.set_title(f'{musc}: 无 {emg_col} 列')
            continue

        if load_col is not None:
            # 按负载分组着色，每个负载一种颜色
            for lv in load_values:
                sub = cutted[cutted[load_col] == lv]
                if len(sub) == 0:
                    continue
                ax.scatter(sub['xval'], sub[emg_col], s=12, alpha=0.5,
                           color=load_color_map[lv],
                           label=(f'{lv:g} kg' if c == 0 else None))
        else:
            ax.scatter(cutted['xval'], cutted[emg_col], s=12, alpha=0.4,
                       color='gray', label='Raw data')

        ax.set_xlabel(x_label)
        ax.set_ylabel('Activation')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{musc} (all heights)')
        if c == 0:
            ax.legend(fontsize=8, loc='best')

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def main():
    if X_AXIS not in X_AXIS_SPECS:
        raise ValueError(
            f"X_AXIS 只能是 {list(X_AXIS_SPECS)}，当前为 {X_AXIS!r}")
    spec = X_AXIS_SPECS[X_AXIS]
    draw_fit = (X_AXIS == 'load')   # 仅 load 叠加拟合曲线

    subject = Subject(CONFIG_FILE)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # ---- 步骤 1：收集原始切片数据并按高度范围过滤 ----
    if not pipeline.results:
        pipeline.run(include_xsens=False)
    cutted = pipeline._collect_cutted_data(movement_types=MOVEMENT_TYPES)
    if cutted is None or len(cutted) == 0:
        print('未收集到切片数据，终止。')
        return
    cutted = cutted.copy()

    if subject.height_range is not None:
        h_min, h_max = subject.height_range
    else:
        h_min = float(cutted['pos_l'].min())
        h_max = float(cutted['pos_l'].max())
    cutted = cutted[(cutted['pos_l'] >= h_min) & (cutted['pos_l'] <= h_max)]

    # ---- 步骤 2：根据 X_AXIS 构造横轴变量列 'xval' ----
    if X_AXIS == 'load':
        load_col = 'load' if 'load' in cutted.columns else 'load_value'
        if load_col not in cutted.columns:
            print('切片数据缺少 load / load_value 列，终止。')
            return
        cutted['xval'] = cutted[load_col].astype(float)
    elif X_AXIS == 'force':
        avail = [c for c in FORCE_COLUMNS if c in cutted.columns]
        if not avail:
            print(f'切片数据缺少力列 {FORCE_COLUMNS}，终止。')
            return
        cutted['xval'] = cutted[avail].sum(axis=1)
    else:  # 'power'
        avail = [c for c in FORCE_COLUMNS if c in cutted.columns]
        if not avail:
            print(f'切片数据缺少力列 {FORCE_COLUMNS}，终止。')
            return
        if VELOCITY_COLUMN not in cutted.columns:
            print(f'切片数据缺少速度列 {VELOCITY_COLUMN}，终止。')
            return
        cutted['xval'] = cutted[avail].sum(axis=1) * cutted[VELOCITY_COLUMN]

    # ---- 步骤 3：仅 load 时在 (位置, 负载) 空间上拟合曲面 ----
    params = {}
    if draw_fit:
        for musc in TARGET_MUSCLES:
            emg_col = f'emg_{musc}'
            if emg_col not in cutted.columns:
                print(f'跳过肌肉 {musc}：列 {emg_col} 不存在')
                continue
            print(f'拟合肌肉 {musc} 的 {spec["title"]}-激活 曲面...')
            params_psp, params_rbf = _fit_surfaces(
                cutted, 'xval', emg_col, subject.height_range)
            params[musc] = params_psp
            params[f'{musc}_rbf'] = params_rbf

    # ---- 步骤 4：固定高度 + 横轴采样网格 ----
    heights = [h_min + frac * (h_max - h_min) for frac in HEIGHT_FRACTIONS]
    h_window = (h_max - h_min) * HEIGHT_WINDOW_FRAC

    x_min = float(cutted['xval'].min())
    x_max = float(cutted['xval'].max())
    x_grid = np.linspace(x_min, x_max, N_LOAD_GRID)

    eps = 1e-6
    x_min_pos = x_min if x_min > 0 else eps
    x_grid_log = np.logspace(
        np.log(max(x_min_pos, eps)),
        np.log(max(x_max, x_min_pos + eps)),
        N_LOAD_GRID,
        base=np.e,
    )

    # ---- 步骤 5：输出目录 + 标签 ----
    save_dir = os.path.join(subject.result_folder, 'heatmap')
    os.makedirs(save_dir, exist_ok=True)

    title = spec['title']
    unit = spec['unit']
    file_key = spec['file']
    x_label_lin = f'{title} ({unit})'
    x_label_log = f'log({title})  [{title} in {unit}]'

    # ---- 步骤 6：四张图 ----
    # 图 1：横轴线性，纵轴 activation
    _draw_figure(
        cutted, params, heights, h_window, x_grid, x_grid_log,
        x_log=False, y_log=False, x_label=x_label_lin,
        suptitle=f'{title} \u2014 Activation at fixed heights',
        out_path=os.path.join(
            save_dir, f'{file_key}_activation_curves_by_height.png'),
        draw_fit=draw_fit)

    # 图 2：横轴 log10，纵轴 activation
    _draw_figure(
        cutted, params, heights, h_window, x_grid, x_grid_log,
        x_log=True, y_log=False, x_label=x_label_log,
        suptitle=f'log({title}) \u2014 Activation at fixed heights',
        out_path=os.path.join(
            save_dir, f'log_{file_key}_activation_curves_by_height.png'),
        draw_fit=draw_fit)

    # 图 3：横轴线性，纵轴 log10(RMS)
    _draw_figure(
        cutted, params, heights, h_window, x_grid, x_grid_log,
        x_log=False, y_log=True, x_label=x_label_lin,
        suptitle=f'{title} \u2014 log(RMS) at fixed heights',
        out_path=os.path.join(
            save_dir, f'{file_key}_logrms_curves_by_height.png'),
        draw_fit=draw_fit)

    # 图 4：横轴 log10，纵轴 log10(RMS)
    _draw_figure(
        cutted, params, heights, h_window, x_grid, x_grid_log,
        x_log=True, y_log=True, x_label=x_label_log,
        suptitle=f'log({title}) \u2014 log(RMS) at fixed heights',
        out_path=os.path.join(
            save_dir, f'log{file_key}_logrms_curves_by_height.png'),
        draw_fit=draw_fit)

    # ---- 步骤 7：force / power 额外输出『不分高度』的全点散点图（分肌肉子图）----
    if X_AXIS in ('force', 'power'):
        _draw_all_points_figure(
            cutted, x_label_lin,
            suptitle=f'{title} — Activation (all heights, all points)',
            out_path=os.path.join(
                save_dir, f'{file_key}_activation_all_points.png'))

    plt.show()


if __name__ == '__main__':
    main()