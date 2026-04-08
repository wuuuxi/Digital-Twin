"""
EMG 中值频率（MDF）分析示例

图 1：各肌肉的 MDF 随时间变化（散点图），不同负载用不同颜色区分
图 2：对齐切片后 MDF 与位置的关系（散点图），不同负载用不同颜色区分
      数据来自 cutted_data 的 mdf_* 列，与 pos_l 完全同行
图 3：参考 visualize_test_3d_scatter 最后一张图，
      3 行 x N 列（每列一个负载）：位置-速度 / 位置-肌肉激活 / 位置-MDF，
      向心/离心用蓝/橙区分。
      支持传入单个或多个肌肉名称（如 'VL' 或 ['VL', 'RF']），
      多肌肉时对每块肌肉各画一张图。

用法：
    python example_emg_frequency.py
"""
import matplotlib.pyplot as plt
import numpy as np
from digitaltwin import Subject, MultiLoadPipeline

# 高对比度颜色（与 example_data_analysis 一致，取自 tab10）
LOAD_COLORS = plt.cm.tab10.colors


def plot_mdf_vs_time(pipeline, muscles, title_prefix=''):
    """
    参考 visualize_movement_segments 的布局，绘制 MDF-时间图。
    布局：n_muscles 行 × n_loads 列，同一行（同一肌肉）共享 Y 轴范围。
    灰色底层：aligned_data 的完整 MDF 时间序列。
    彩色叠加：cutted_data 按 cycle_id 分段，每段一个颜色，
    向心实线 / 离心虚线。

    Parameters
    ----------
    pipeline : MultiLoadPipeline
        已调用 run() 的流水线实例
    muscles : list[str]
        肌肉名称列表（不含 'emg_' 前缀），如 ['VL', 'RF']
    """
    results = pipeline.results
    load_weights = sorted(results.keys(), key=lambda x: float(x))
    n_loads = len(load_weights)
    n_muscles = len(muscles)
    if n_loads == 0 or n_muscles == 0:
        print('No data to plot'); return

    # ---------- 全局 Y 轴范围（每块肌肉跨所有负载共享） ----------
    gy = {}  # {musc: [ymin, ymax]}
    for musc in muscles:
        mdf_col = f'mdf_{musc}'
        ymin, ymax = float('inf'), float('-inf')
        for lw in load_weights:
            res = results.get(lw)
            if res is None:
                continue
            # aligned_data 范围
            ad = res.get('aligned_data')
            if ad is not None and mdf_col in ad.columns:
                vals = ad[mdf_col].dropna().values
                if len(vals) > 0:
                    ymin = min(ymin, np.min(vals))
                    ymax = max(ymax, np.max(vals))
            # cutted_data 范围
            cd = res.get('cutted_data')
            if cd is not None and mdf_col in cd.columns:
                vals = cd[mdf_col].dropna().values
                if len(vals) > 0:
                    ymin = min(ymin, np.min(vals))
                    ymax = max(ymax, np.max(vals))
        if ymin != float('inf'):
            r = (ymax - ymin) * 0.1
            gy[musc] = [ymin - r, ymax + r]
        else:
            gy[musc] = [0, 1]

    # ---------- 绘图 ----------
    fig, axes = plt.subplots(n_muscles, n_loads,
                             figsize=(5 * n_loads, 3 * n_muscles),
                             squeeze=False)
    fig.suptitle(f'{title_prefix}MDF vs Time (Segmented)',
                 fontsize=14, fontweight='bold')

    cycle_colors = plt.cm.tab10.colors

    for mi, musc in enumerate(muscles):
        mdf_col = f'mdf_{musc}'
        for li, lw in enumerate(load_weights):
            ax = axes[mi][li]
            res = results.get(lw)
            if res is None:
                ax.set_visible(False)
                continue

            ad = res.get('aligned_data')
            cd = res.get('cutted_data')

            # 灰色底层：完整 aligned_data
            if ad is not None and mdf_col in ad.columns and 'time' in ad.columns:
                ax.plot(ad['time'], ad[mdf_col], 'k-', alpha=0.3, linewidth=0.8)

            # 彩色叠加：cutted_data 按 cycle_id 分段
            if cd is not None and mdf_col in cd.columns and 'time' in cd.columns:
                if 'cycle_id' in cd.columns:
                    n_cycles = int(cd['cycle_id'].max()) + 1
                    for ci in range(n_cycles):
                        seg = cd[cd['cycle_id'] == ci]
                        clr = cycle_colors[ci % len(cycle_colors)]
                        valid = ~np.isnan(seg[mdf_col].values)
                        if 'movement_type' in seg.columns:
                            up = seg[seg['movement_type'] == 'upward']
                            dn = seg[seg['movement_type'] == 'downward']
                            vu = ~np.isnan(up[mdf_col].values)
                            vd = ~np.isnan(dn[mdf_col].values)
                            if np.any(vu):
                                ax.plot(up['time'].values[vu],
                                        up[mdf_col].values[vu],
                                        '-', color=clr, linewidth=1.5)
                            if np.any(vd):
                                ax.plot(dn['time'].values[vd],
                                        dn[mdf_col].values[vd],
                                        '--', color=clr, linewidth=1.5,
                                        alpha=0.6)
                        else:
                            if np.any(valid):
                                ax.plot(seg['time'].values[valid],
                                        seg[mdf_col].values[valid],
                                        '-', color=clr, linewidth=1.5)
                else:
                    # 无 cycle_id，直接用单色
                    valid = ~np.isnan(cd[mdf_col].values)
                    ax.plot(cd['time'].values[valid],
                            cd[mdf_col].values[valid],
                            '-', color='tab:blue', linewidth=1.5)

            ax.set_ylim(gy[musc][0], gy[musc][1])
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)

            # 标题和标签
            if mi == 0:
                ax.set_title(f'{lw} kg')
            if li == 0:
                ax.set_ylabel(f'{musc}\nMDF (Hz)')

            # 缩小 X 范围到切片区间 ± 边距
            if cd is not None and 'time' in cd.columns and len(cd) > 0:
                st = cd['time'].min()
                et = cd['time'].max()
                ax.set_xlim(st - 1, et + 1)

    fig.tight_layout()


def plot_mdf_vs_position(pipeline, muscles, title_prefix=''):
    """
    绘制对齐切片后的 MDF-位置散点图。
    数据来自 cutted_data 的 mdf_<muscle> 和 pos_l 列，完全同行。

    Parameters
    ----------
    pipeline : MultiLoadPipeline
        已调用 run() 的流水线实例
    muscles : list[str]
        肌肉名称列表（不含 'emg_' 前缀），如 ['VL', 'RF']
    """
    results = pipeline.results
    load_weights = sorted(results.keys(), key=lambda x: float(x))
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(f'{title_prefix}EMG Median Frequency vs Position (cutted)', fontsize=14)

    for idx, musc in enumerate(muscles):
        ax = axes[idx // n_cols][idx % n_cols]
        mdf_col = f'mdf_{musc}'
        for i, lw in enumerate(load_weights):
            res = results.get(lw)
            if res is None:
                continue
            cd = res.get('cutted_data')
            if cd is None or mdf_col not in cd.columns or 'pos_l' not in cd.columns:
                continue
            pos = cd['pos_l'].values
            mdf = cd[mdf_col].values
            valid = ~np.isnan(mdf)
            if not np.any(valid):
                continue
            ax.scatter(pos[valid], mdf[valid],
                       color=LOAD_COLORS[i % len(LOAD_COLORS)],
                       alpha=0.5, s=8, label=f'{lw} kg')
        ax.set_title(musc)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('MDF (Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()


def plot_pos_vel_emg_mdf_grid(pipeline, muscles=None):
    """
    参考 visualize_test_3d_scatter 的最后一张图（fig5），
    绘制 3行 x N列 的散点图矩阵：
      Row 1: Position vs |Velocity|
      Row 2: Position vs |EMG Activation|
      Row 3: Position vs MDF
    每列为一个负载，Concentric(蓝) / Eccentric(橙) 分色。

    MDF 数据来自 cutted_data 的 mdf_<muscle> 列，
    与 pos_l / vel_l / emg_* 完全共享同一行索引。

    Parameters
    ----------
    pipeline : MultiLoadPipeline
        已调用 run() 的流水线实例
    muscles : str or list[str]
        肌肉名称（不含 'emg_' 前缀），如 'VL' 或 ['VL', 'RF']。
        传入多个肌肉时，对每块肌肉各生成一张图。
    """
    results = pipeline.results
    subject = pipeline.subject
    load_weights = sorted(results.keys(), key=lambda x: float(x))
    n_loads = len(load_weights)
    if n_loads == 0:
        print('No results to plot'); return

    # --- 归一化输入 ---
    if muscles is None:
        muscles = ['PMSte'] if subject and subject.target_motion == 'benchpress' else ['TA']
    if isinstance(muscles, str):
        muscles = [muscles]

    CON_COLOR = '#1f77b4'
    ECC_COLOR = '#ff7f0e'

    for muscle_name in muscles:
        emg_col = f'emg_{muscle_name}'
        mdf_col = f'mdf_{muscle_name}'

        # ---------- 全局 Y 轴范围 ----------
        gv = [float('inf'), float('-inf')]
        ge = [float('inf'), float('-inf')]
        gm = [float('inf'), float('-inf')]

        for lw in load_weights:
            res = results.get(lw)
            if res is None:
                continue
            cd = res.get('cutted_data')
            if cd is None or emg_col not in cd.columns:
                continue

            vel = np.abs(cd['vel_l'].values)
            emg = np.abs(cd[emg_col].values)
            gv = [min(gv[0], np.min(vel)), max(gv[1], np.max(vel))]
            ge = [min(ge[0], np.min(emg)), max(ge[1], np.max(emg))]

            if mdf_col in cd.columns:
                mdf_vals = cd[mdf_col].dropna().values
                if len(mdf_vals) > 0:
                    gm = [min(gm[0], np.min(mdf_vals)),
                          max(gm[1], np.max(mdf_vals))]

        for g in [gv, ge, gm]:
            if g[0] != float('inf'):
                r = (g[1] - g[0]) * 0.1
                g[0] = max(0, g[0] - r)
                g[1] += r
            else:
                g[0], g[1] = 0, 1

        # ---------- 绘图 ----------
        fig, axes = plt.subplots(3, n_loads, figsize=(4 * n_loads, 8),
                                 squeeze=False)
        fig.suptitle(f'Position–Velocity / EMG / MDF  ({muscle_name})',
                     fontsize=14, fontweight='bold')

        for li, lw in enumerate(load_weights):
            res = results.get(lw)
            if res is None:
                for r in range(3):
                    axes[r][li].set_visible(False)
                continue

            cd = res.get('cutted_data')
            if cd is None or emg_col not in cd.columns:
                for r in range(3):
                    axes[r][li].set_visible(False)
                continue

            pos = cd['pos_l'].values
            vel = cd['vel_l'].values
            emg = cd[emg_col].values
            pmask = vel > 0
            nmask = vel <= 0

            # Row 1: Position vs |Velocity|
            ax1 = axes[0][li]
            ax1.scatter(pos[pmask], np.abs(vel[pmask]), alpha=0.5, s=10,
                        color=CON_COLOR, label='Concentric')
            ax1.scatter(pos[nmask], np.abs(vel[nmask]), alpha=0.5, s=10,
                        color=ECC_COLOR, label='Eccentric')
            ax1.set_ylim(gv[0], gv[1])
            ax1.set_title(f'{lw} kg')
            ax1.set_xlabel('Position (m)')
            if li == 0:
                ax1.set_ylabel('|Velocity| (m/s)')
            if li == n_loads - 1:
                ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Row 2: Position vs |EMG Activation|
            ax2 = axes[1][li]
            ax2.scatter(pos[pmask], np.abs(emg[pmask]), alpha=0.5, s=10,
                        color=CON_COLOR, label='Concentric')
            ax2.scatter(pos[nmask], np.abs(emg[nmask]), alpha=0.5, s=10,
                        color=ECC_COLOR, label='Eccentric')
            ax2.set_ylim(ge[0], ge[1])
            ax2.set_xlabel('Position (m)')
            if li == 0:
                ax2.set_ylabel(f'{muscle_name} Activation')
            if li == n_loads - 1:
                ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            # Row 3: Position vs MDF
            ax3 = axes[2][li]
            if mdf_col in cd.columns:
                mdf = cd[mdf_col].values
                valid = ~np.isnan(mdf)
                ax3.scatter(pos[pmask & valid], mdf[pmask & valid],
                            alpha=0.5, s=10,
                            color=CON_COLOR, label='Concentric')
                ax3.scatter(pos[nmask & valid], mdf[nmask & valid],
                            alpha=0.5, s=10,
                            color=ECC_COLOR, label='Eccentric')
            ax3.set_ylim(gm[0], gm[1])
            ax3.set_xlabel('Position (m)')
            if li == 0:
                ax3.set_ylabel('MDF (Hz)')
            if li == n_loads - 1:
                ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

        fig.tight_layout()


def plot_mdf_bar_by_load(pipeline, muscles, title_prefix=''):
    """
    柱状图：每块肌肉一个子图，X 轴为不同负载，
    Y 轴为该肌肉在该负载下切片后 MDF 的平均值 ± 标准差。

    Parameters
    ----------
    pipeline : MultiLoadPipeline
        已调用 run() 的流水线实例
    muscles : list[str]
        肌肉名称列表（不含 'emg_' 前缀），如 ['VL', 'RF']
    """
    results = pipeline.results
    load_weights = sorted(results.keys(), key=lambda x: float(x))
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(f'{title_prefix}Mean MDF by Load', fontsize=14)

    for idx, musc in enumerate(muscles):
        ax = axes[idx // n_cols][idx % n_cols]
        mdf_col = f'mdf_{musc}'

        means = []
        stds = []
        labels = []
        colors = []

        for i, lw in enumerate(load_weights):
            res = results.get(lw)
            if res is None:
                continue
            cd = res.get('cutted_data')
            if cd is None or mdf_col not in cd.columns:
                continue
            mdf_vals = cd[mdf_col].dropna().values
            if len(mdf_vals) == 0:
                continue
            means.append(np.mean(mdf_vals))
            stds.append(np.std(mdf_vals))
            labels.append(f'{lw}')
            colors.append(LOAD_COLORS[i % len(LOAD_COLORS)])

        if not means:
            ax.set_title(musc)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=colors, alpha=0.8, edgecolor='black',
                      linewidth=0.5)

        # 在柱顶标注均值
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + 0.5, f'{m:.1f}',
                    ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{lb} kg' for lb in labels], fontsize=9)
        ax.set_title(musc)
        ax.set_xlabel('Load')
        ax.set_ylabel('MDF (Hz)')
        ax.grid(True, alpha=0.3, axis='y')

    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()


def main():
    # --- 配置 ---
    subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # --- 数据加载 ---
    pipeline.run(include_xsens=False)

    # 选择要分析的肌肉（不含 'emg_' 前缀）
    muscles = subject.musc_label[:6]  # 取前 6 块肌肉示例

    # --- 图 1：MDF vs Time（运动切片风格：灰色底层 + 彩色切片） ---
    plot_mdf_vs_time(pipeline, muscles)

    # --- 图 2：MDF vs Position 散点图（对齐切片后数据） ---
    plot_mdf_vs_position(pipeline, muscles)

    # --- 图 3：Position-Velocity / Position-EMG / Position-MDF ---
    # 传入 str 或 list[str]，多肌肉时每块各画一张图
    plot_pos_vel_emg_mdf_grid(pipeline, ['VL', 'RF'])

    # --- 图 4：各肌肉 MDF 均值 ± 标准差 柱状图 ---
    plot_mdf_bar_by_load(pipeline, muscles)

    plt.show()


if __name__ == '__main__':
    main()