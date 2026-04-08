"""
EMG 中值频率（MDF）分析示例——固定负载 + 变负载对比

在 example_emg_frequency.py 的基础上，额外加载并分析 JSON 配置中
`vload_parameters` 指定的变负载实验数据，与固定负载结果一起对比。

vload_parameters 格式：
    {"label": [robot_file, emg_file, start_time, goal, load_range, mode]}
    label 约定为 "<muscle>_<goal>_<epsilon>"

图 1：MDF vs Time（运动切片风格）——固定负载 + 变负载
图 2：MDF vs Position 散点图——固定负载 + 变负载
图 3：3行×N列 Position-Velocity / EMG / MDF 网格图
图 4：MDF 均值柱状图（固定负载 + 变负载）

用法：
    python example_emg_frequency_vload.py
"""
import matplotlib.pyplot as plt
import numpy as np
from digitaltwin import Subject, MultiLoadPipeline

# 高对比度颜色
LOAD_COLORS = plt.cm.tab10.colors
VLOAD_COLORS = plt.cm.Set2.colors  # 变负载用另一套颜色


# ==============================================================
#  图 1：MDF vs Time（固定负载 + 变负载）
# ==============================================================

def plot_mdf_vs_time_combined(pipeline, vload_results, muscles):
    """
    运动切片风格 MDF-时间图。
    布局：n_muscles 行 × (n_fixed_loads + n_vloads) 列。
    前 N 列为固定负载，后 M 列为变负载。
    同一行共享 Y 轴范围。
    """
    results = pipeline.results
    fixed_keys = sorted(results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys())
    all_keys = fixed_keys + vload_keys
    n_cols = len(all_keys)
    n_muscles = len(muscles)
    if n_cols == 0 or n_muscles == 0:
        return

    # 合并数据源
    def get_data(key):
        if key in results:
            return results[key]
        return vload_results.get(key)

    # 全局 Y 范围
    gy = {}
    for musc in muscles:
        mdf_col = f'mdf_{musc}'
        ymin, ymax = float('inf'), float('-inf')
        for key in all_keys:
            d = get_data(key)
            if d is None:
                continue
            for df_key in ['aligned_data', 'cutted_data']:
                df = d.get(df_key)
                if df is not None and mdf_col in df.columns:
                    vals = df[mdf_col].dropna().values
                    if len(vals) > 0:
                        ymin = min(ymin, np.min(vals))
                        ymax = max(ymax, np.max(vals))
        if ymin != float('inf'):
            r = (ymax - ymin) * 0.1
            gy[musc] = [ymin - r, ymax + r]
        else:
            gy[musc] = [0, 1]

    fig, axes = plt.subplots(n_muscles, n_cols,
                             figsize=(4 * n_cols, 3 * n_muscles),
                             squeeze=False)
    fig.suptitle('MDF vs Time: Fixed Loads + Variable Loads',
                 fontsize=14, fontweight='bold')
    cycle_colors = plt.cm.tab10.colors

    for mi, musc in enumerate(muscles):
        mdf_col = f'mdf_{musc}'
        for ci, key in enumerate(all_keys):
            ax = axes[mi][ci]
            d = get_data(key)
            if d is None:
                ax.set_visible(False)
                continue

            ad = d.get('aligned_data')
            cd = d.get('cutted_data')

            # 灰色底层
            if ad is not None and mdf_col in ad.columns and 'time' in ad.columns:
                ax.plot(ad['time'], ad[mdf_col], 'k-', alpha=0.3, linewidth=0.8)

            # 彩色切片
            if cd is not None and mdf_col in cd.columns and 'time' in cd.columns:
                if 'cycle_id' in cd.columns:
                    n_cyc = int(cd['cycle_id'].max()) + 1
                    for ci2 in range(n_cyc):
                        seg = cd[cd['cycle_id'] == ci2]
                        clr = cycle_colors[ci2 % len(cycle_colors)]
                        if 'movement_type' in seg.columns:
                            up = seg[seg['movement_type'] == 'upward']
                            dn = seg[seg['movement_type'] == 'downward']
                            vu = ~np.isnan(up[mdf_col].values)
                            vd = ~np.isnan(dn[mdf_col].values)
                            if np.any(vu):
                                ax.plot(up['time'].values[vu], up[mdf_col].values[vu],
                                        '-', color=clr, linewidth=1.5)
                            if np.any(vd):
                                ax.plot(dn['time'].values[vd], dn[mdf_col].values[vd],
                                        '--', color=clr, linewidth=1.5, alpha=0.6)
                        else:
                            valid = ~np.isnan(seg[mdf_col].values)
                            if np.any(valid):
                                ax.plot(seg['time'].values[valid], seg[mdf_col].values[valid],
                                        '-', color=clr, linewidth=1.5)
                else:
                    valid = ~np.isnan(cd[mdf_col].values)
                    ax.plot(cd['time'].values[valid], cd[mdf_col].values[valid],
                            '-', color='tab:blue', linewidth=1.5)

            ax.set_ylim(gy[musc][0], gy[musc][1])
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)

            if mi == 0:
                is_vload = key in vload_results
                title = f'VLoad: {key}' if is_vload else f'{key} kg'
                ax.set_title(title, fontsize=9,
                             color='green' if is_vload else 'black')
            if ci == 0:
                ax.set_ylabel(f'{musc}\nMDF (Hz)')

            if cd is not None and 'time' in cd.columns and len(cd) > 0:
                ax.set_xlim(cd['time'].min() - 1, cd['time'].max() + 1)

    fig.tight_layout()


# ==============================================================
#  图 2：MDF vs Position（固定 + 变负载）
# ==============================================================

def plot_mdf_vs_position_combined(pipeline, vload_results, muscles):
    """
    MDF-位置散点图，固定负载用实心点，变负载用空心三角。
    """
    results = pipeline.results
    fixed_keys = sorted(results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys())
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle('MDF vs Position: Fixed + Variable Loads', fontsize=14)

    for idx, musc in enumerate(muscles):
        ax = axes[idx // n_cols][idx % n_cols]
        mdf_col = f'mdf_{musc}'

        # 固定负载
        for i, lw in enumerate(fixed_keys):
            cd = results[lw].get('cutted_data')
            if cd is None or mdf_col not in cd.columns:
                continue
            pos = cd['pos_l'].values
            mdf = cd[mdf_col].values
            valid = ~np.isnan(mdf)
            if np.any(valid):
                ax.scatter(pos[valid], mdf[valid],
                           color=LOAD_COLORS[i % len(LOAD_COLORS)],
                           alpha=0.4, s=8, label=f'{lw} kg')

        # 变负载
        for j, vk in enumerate(vload_keys):
            vd = vload_results[vk]
            cd = vd.get('cutted_data')
            if cd is None or mdf_col not in cd.columns:
                continue
            pos = cd['pos_l'].values
            mdf = cd[mdf_col].values
            valid = ~np.isnan(mdf)
            if np.any(valid):
                ax.scatter(pos[valid], mdf[valid],
                           color=VLOAD_COLORS[j % len(VLOAD_COLORS)],
                           alpha=0.6, s=15, marker='^',
                           label=f'VL: {vk}')

        ax.set_title(musc)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('MDF (Hz)')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()


# ==============================================================
#  图 3：3行×N列 Position-Vel/EMG/MDF 网格（固定 + 变负载）
# ==============================================================

def plot_grid_combined(pipeline, vload_results, muscles=None):
    """
    3行(Vel/EMG/MDF) × (n_fixed + n_vload)列 的散点图。
    固定负载列在前，变负载列在后。
    """
    results = pipeline.results
    subject = pipeline.subject
    fixed_keys = sorted(results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys())
    all_keys = fixed_keys + vload_keys
    n_cols = len(all_keys)
    if n_cols == 0:
        return

    if muscles is None:
        muscles = ['PMSte'] if subject and subject.target_motion == 'benchpress' else ['TA']
    if isinstance(muscles, str):
        muscles = [muscles]

    CON_COLOR = '#1f77b4'
    ECC_COLOR = '#ff7f0e'

    def get_cd(key):
        if key in results:
            return results[key].get('cutted_data')
        vd = vload_results.get(key)
        return vd.get('cutted_data') if vd else None

    for muscle_name in muscles:
        emg_col = f'emg_{muscle_name}'
        mdf_col = f'mdf_{muscle_name}'

        # 全局 Y 范围
        gv = [float('inf'), float('-inf')]
        ge = [float('inf'), float('-inf')]
        gm = [float('inf'), float('-inf')]

        for key in all_keys:
            cd = get_cd(key)
            if cd is None or emg_col not in cd.columns:
                continue
            vel = np.abs(cd['vel_l'].values)
            emg = np.abs(cd[emg_col].values)
            gv = [min(gv[0], np.min(vel)), max(gv[1], np.max(vel))]
            ge = [min(ge[0], np.min(emg)), max(ge[1], np.max(emg))]
            if mdf_col in cd.columns:
                mv = cd[mdf_col].dropna().values
                if len(mv) > 0:
                    gm = [min(gm[0], np.min(mv)), max(gm[1], np.max(mv))]

        for g in [gv, ge, gm]:
            if g[0] != float('inf'):
                r = (g[1] - g[0]) * 0.1
                g[0] = max(0, g[0] - r); g[1] += r
            else:
                g[0], g[1] = 0, 1

        fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 8),
                                 squeeze=False)
        fig.suptitle(f'Pos-Vel / EMG / MDF ({muscle_name}) '
                     f'Fixed + VLoad', fontsize=13, fontweight='bold')

        for ci, key in enumerate(all_keys):
            cd = get_cd(key)
            is_vload = key in vload_results

            if cd is None or emg_col not in cd.columns:
                for r in range(3):
                    axes[r][ci].set_visible(False)
                continue

            pos = cd['pos_l'].values
            vel = cd['vel_l'].values
            emg = cd[emg_col].values
            pmask = vel > 0
            nmask = vel <= 0

            for ri, (data_arr, ylim, ylabel) in enumerate([
                (np.abs(vel), gv, '|Velocity|'),
                (np.abs(emg), ge, f'{muscle_name} Act'),
            ]):
                ax = axes[ri][ci]
                ax.scatter(pos[pmask], data_arr[pmask], alpha=0.5, s=10,
                           color=CON_COLOR, label='Con')
                ax.scatter(pos[nmask], data_arr[nmask], alpha=0.5, s=10,
                           color=ECC_COLOR, label='Ecc')
                ax.set_ylim(ylim[0], ylim[1])
                if ci == 0:
                    ax.set_ylabel(ylabel)
                if ci == n_cols - 1:
                    ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            # Row 3: MDF
            ax3 = axes[2][ci]
            if mdf_col in cd.columns:
                mdf = cd[mdf_col].values
                valid = ~np.isnan(mdf)
                ax3.scatter(pos[pmask & valid], mdf[pmask & valid],
                            alpha=0.5, s=10, color=CON_COLOR, label='Con')
                ax3.scatter(pos[nmask & valid], mdf[nmask & valid],
                            alpha=0.5, s=10, color=ECC_COLOR, label='Ecc')
            ax3.set_ylim(gm[0], gm[1])
            ax3.set_xlabel('Position (m)')
            if ci == 0:
                ax3.set_ylabel('MDF (Hz)')
            if ci == n_cols - 1:
                ax3.legend(fontsize=7)
            ax3.grid(True, alpha=0.3)

            # 标题
            title = f'VL: {key}' if is_vload else f'{key} kg'
            axes[0][ci].set_title(title, fontsize=9,
                                  color='green' if is_vload else 'black')

        fig.tight_layout()


# ==============================================================
#  图 4：MDF 均值柱状图（固定 + 变负载）
# ==============================================================

def plot_mdf_bar_combined(pipeline, vload_results, muscles):
    """
    柱状图：每块肌肉一个子图，固定负载柱 + 变负载柱并排。
    变负载柱用斜线填充以便区分。
    """
    results = pipeline.results
    fixed_keys = sorted(results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys())
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle('Mean MDF: Fixed Loads vs Variable Loads', fontsize=14)

    for idx, musc in enumerate(muscles):
        ax = axes[idx // n_cols][idx % n_cols]
        mdf_col = f'mdf_{musc}'

        bar_means, bar_stds, bar_labels, bar_colors, bar_hatches = [], [], [], [], []

        # 固定负载
        for i, lw in enumerate(fixed_keys):
            cd = results[lw].get('cutted_data')
            if cd is None or mdf_col not in cd.columns:
                continue
            vals = cd[mdf_col].dropna().values
            if len(vals) == 0:
                continue
            bar_means.append(np.mean(vals))
            bar_stds.append(np.std(vals))
            bar_labels.append(f'{lw} kg')
            bar_colors.append(LOAD_COLORS[i % len(LOAD_COLORS)])
            bar_hatches.append('')

        # 变负载
        for j, vk in enumerate(vload_keys):
            vd = vload_results[vk]
            cd = vd.get('cutted_data')
            if cd is None or mdf_col not in cd.columns:
                continue
            vals = cd[mdf_col].dropna().values
            if len(vals) == 0:
                continue
            bar_means.append(np.mean(vals))
            bar_stds.append(np.std(vals))
            bar_labels.append(f'VL:{vk}')
            bar_colors.append(VLOAD_COLORS[j % len(VLOAD_COLORS)])
            bar_hatches.append('//')

        if not bar_means:
            ax.set_title(musc)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        x = np.arange(len(bar_means))
        bars = ax.bar(x, bar_means, yerr=bar_stds, capsize=4,
                      color=bar_colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        for bi, h in enumerate(bar_hatches):
            bars[bi].set_hatch(h)

        for xi, m, s in zip(x, bar_means, bar_stds):
            ax.text(xi, m + s + 0.5, f'{m:.1f}',
                    ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=7, rotation=30, ha='right')
        ax.set_title(musc)
        ax.set_ylabel('MDF (Hz)')
        ax.grid(True, alpha=0.3, axis='y')

    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()


# ==============================================================
#  main
# ==============================================================

def main():
    # --- 配置 ---
    subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # --- Step 1: 固定负载数据 ---
    pipeline.run(include_xsens=False)

    # --- Step 2: 变负载数据（处理逻辑已下沉到 pipeline.run_vload()） ---
    vload_results = pipeline.run_vload()

    # 选择要分析的肌肉（不含 'emg_' 前缀）
    muscles = subject.musc_label[:6]

    # --- 图 1: MDF vs Time ---
    plot_mdf_vs_time_combined(pipeline, vload_results, muscles)

    # --- 图 2: MDF vs Position ---
    plot_mdf_vs_position_combined(pipeline, vload_results, muscles)

    # --- 图 3: Position-Vel/EMG/MDF 网格 ---
    plot_grid_combined(pipeline, vload_results, ['VL', 'RF'])

    # --- 图 4: MDF 均值柱状图 ---
    plot_mdf_bar_combined(pipeline, vload_results, muscles)

    plt.show()


if __name__ == '__main__':
    main()