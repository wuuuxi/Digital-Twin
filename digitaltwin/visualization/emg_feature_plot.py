"""
EMG 特征（MDF / RMS 等）通用可视化模块

所有绘图函数通过 `feature` 参数（如 'mdf' 或 'rms'）和 `feature_label`
参数（如 'MDF (Hz)' 或 'RMS (mV)'）实现复用，避免 MDF/RMS 版本间的代码重复。

提供 4 种基础图（仅固定负载）和 4 种组合图（固定 + 变负载）：
  - feature_vs_time         / feature_vs_time_combined
  - feature_vs_position     / feature_vs_position_combined
  - pos_vel_emg_feature_grid / pos_vel_emg_feature_grid_combined
  - feature_bar_by_load     / feature_bar_combined
"""
import matplotlib.pyplot as plt
import numpy as np

LOAD_COLORS = plt.cm.tab10.colors
VLOAD_COLORS = plt.cm.Set2.colors
CON_COLOR = '#1f77b4'
ECC_COLOR = '#ff7f0e'


# ==============================================================
#  内部工具
# ==============================================================

def _col(feature, musc):
    """构建列名，如 'mdf_VL' 或 'rms_VL'"""
    return f'{feature}_{musc}'


def _global_ylim(all_data, col_name, margin=0.1):
    """跨多组数据计算某列的全局 Y 轴范围（忽略 NaN）"""
    ymin, ymax = float('inf'), float('-inf')
    for d in all_data:
        if d is None:
            continue
        for df_key in ('aligned_data', 'cutted_data'):
            df = d.get(df_key) if isinstance(d, dict) else d
            if df is None:
                continue
            if hasattr(df, 'columns') and col_name in df.columns:
                vals = df[col_name].dropna().values
                if len(vals) > 0:
                    ymin = min(ymin, np.min(vals))
                    ymax = max(ymax, np.max(vals))
    if ymin == float('inf'):
        return [0, 1]
    r = (ymax - ymin) * margin
    return [ymin - r, ymax + r]


def _plot_segmented_line(ax, cd, col_name, cycle_colors):
    """在 ax 上绘制按 cycle_id 分段的彩色线（向心实线/离心虚线）"""
    if cd is None or col_name not in cd.columns or 'time' not in cd.columns:
        return
    if 'cycle_id' in cd.columns:
        n_cyc = int(cd['cycle_id'].max()) + 1
        for ci in range(n_cyc):
            seg = cd[cd['cycle_id'] == ci]
            clr = cycle_colors[ci % len(cycle_colors)]
            if 'movement_type' in seg.columns:
                for subset, style in [(seg[seg['movement_type'] == 'upward'], '-'),
                                      (seg[seg['movement_type'] == 'downward'], '--')]:
                    v = ~np.isnan(subset[col_name].values)
                    if np.any(v):
                        alpha = 1.0 if style == '-' else 0.6
                        ax.plot(subset['time'].values[v], subset[col_name].values[v],
                                style, color=clr, linewidth=1.5, alpha=alpha)
            else:
                v = ~np.isnan(seg[col_name].values)
                if np.any(v):
                    ax.plot(seg['time'].values[v], seg[col_name].values[v],
                            '-', color=clr, linewidth=1.5)
    else:
        v = ~np.isnan(cd[col_name].values)
        if np.any(v):
            ax.plot(cd['time'].values[v], cd[col_name].values[v],
                    '-', color='tab:blue', linewidth=1.5)


# ==============================================================
#  图 1：Feature vs Time（运动切片风格）
# ==============================================================

def plot_feature_vs_time(results, muscles, feature='mdf',
                         feature_label='MDF (Hz)', title_prefix=''):
    """
    n_muscles 行 × n_loads 列。灰色底层 aligned_data，彩色切片 cutted_data。

    Parameters
    ----------
    results : dict
        pipeline.results 或合并后的 {key: result_dict}
    muscles : list[str]
    feature : str  ('mdf' 或 'rms')
    feature_label : str  Y 轴标签
    """
    load_weights = sorted(results.keys(), key=lambda x: float(x))
    n_loads = len(load_weights)
    n_muscles = len(muscles)
    if n_loads == 0 or n_muscles == 0:
        print('No data to plot'); return

    gy = {}
    for musc in muscles:
        col = _col(feature, musc)
        gy[musc] = _global_ylim([results[k] for k in load_weights], col)

    fig, axes = plt.subplots(n_muscles, n_loads,
                             figsize=(3 * n_loads, 2 * n_muscles), squeeze=False)
    fig.suptitle(f'{title_prefix}{feature.upper()} vs Time (Segmented)',
                 fontsize=14, fontweight='bold')
    cycle_colors = plt.cm.tab10.colors

    for mi, musc in enumerate(muscles):
        col = _col(feature, musc)
        for li, lw in enumerate(load_weights):
            ax = axes[mi][li]
            res = results.get(lw)
            if res is None:
                ax.set_visible(False); continue
            ad = res.get('aligned_data')
            cd = res.get('cutted_data')
            if ad is not None and col in ad.columns and 'time' in ad.columns:
                v = ~np.isnan(ad[col].values)
                ax.plot(ad['time'].values[v], ad[col].values[v], 'k-', alpha=0.3, linewidth=0.8)
            _plot_segmented_line(ax, cd, col, cycle_colors)
            ax.set_ylim(gy[musc]); ax.set_xlabel('Time (s)'); ax.grid(True, alpha=0.3)
            if mi == 0: ax.set_title(f'{lw} kg')
            if li == 0: ax.set_ylabel(f'{musc}\n{feature_label}')
            if cd is not None and 'time' in cd.columns and len(cd) > 0:
                ax.set_xlim(cd['time'].min() - 1, cd['time'].max() + 1)
    fig.tight_layout()


# ==============================================================
#  图 2：Feature vs Position 散点图
# ==============================================================

def plot_feature_vs_position(results, muscles, feature='mdf',
                             feature_label='MDF (Hz)', title_prefix=''):
    load_weights = sorted(results.keys(), key=lambda x: float(x))
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(f'{title_prefix}{feature.upper()} vs Position (cutted)', fontsize=14)

    for idx, musc in enumerate(muscles):
        ax = axes[idx // n_cols][idx % n_cols]
        col = _col(feature, musc)
        for i, lw in enumerate(load_weights):
            cd = results[lw].get('cutted_data') if results.get(lw) else None
            if cd is None or col not in cd.columns or 'pos_l' not in cd.columns:
                continue
            pos, vals = cd['pos_l'].values, cd[col].values
            v = ~np.isnan(vals)
            if np.any(v):
                ax.scatter(pos[v], vals[v], color=LOAD_COLORS[i % len(LOAD_COLORS)],
                           alpha=0.5, s=8, label=f'{lw} kg')
        ax.set_title(musc); ax.set_xlabel('Position (m)'); ax.set_ylabel(feature_label)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()


# ==============================================================
#  图 3：3行×N列 Position-Vel / EMG / Feature 网格
# ==============================================================

def plot_pos_vel_emg_feature_grid(results, muscles, subject=None,
                                  feature='mdf', feature_label='MDF (Hz)'):
    load_weights = sorted(results.keys(), key=lambda x: float(x))
    n_loads = len(load_weights)
    if n_loads == 0:
        print('No results'); return
    if muscles is None:
        muscles = ['PMSte'] if subject and subject.target_motion == 'benchpress' else ['TA']
    if isinstance(muscles, str):
        muscles = [muscles]

    for muscle_name in muscles:
        emg_col = f'emg_{muscle_name}'
        feat_col = _col(feature, muscle_name)
        gv, ge, gf = ([float('inf'), float('-inf')] for _ in range(3))
        for lw in load_weights:
            cd = results[lw].get('cutted_data') if results.get(lw) else None
            if cd is None or emg_col not in cd.columns: continue
            vel, emg = np.abs(cd['vel_l'].values), np.abs(cd[emg_col].values)
            gv = [min(gv[0], np.min(vel)), max(gv[1], np.max(vel))]
            ge = [min(ge[0], np.min(emg)), max(ge[1], np.max(emg))]
            if feat_col in cd.columns:
                fv = cd[feat_col].dropna().values
                if len(fv) > 0:
                    gf = [min(gf[0], np.min(fv)), max(gf[1], np.max(fv))]
        for g in [gv, ge, gf]:
            if g[0] != float('inf'):
                r = (g[1] - g[0]) * 0.1; g[0] = max(0, g[0] - r); g[1] += r
            else:
                g[0], g[1] = 0, 1

        fig, axes = plt.subplots(3, n_loads, figsize=(3 * n_loads, 6), squeeze=False)
        fig.suptitle(f'Pos-Vel / EMG / {feature.upper()} ({muscle_name})',
                     fontsize=14, fontweight='bold')
        for li, lw in enumerate(load_weights):
            cd = results[lw].get('cutted_data') if results.get(lw) else None
            if cd is None or emg_col not in cd.columns:
                for r in range(3): axes[r][li].set_visible(False); continue
            pos, vel, emg = cd['pos_l'].values, cd['vel_l'].values, cd[emg_col].values
            pm, nm = vel > 0, vel <= 0
            for ri, (arr, ylim, yl) in enumerate([
                (np.abs(vel), gv, '|Velocity| (m/s)'),
                (np.abs(emg), ge, f'{muscle_name} Activation'),
            ]):
                ax = axes[ri][li]
                ax.scatter(pos[pm], arr[pm], alpha=0.5, s=10, color=CON_COLOR, label='Con')
                ax.scatter(pos[nm], arr[nm], alpha=0.5, s=10, color=ECC_COLOR, label='Ecc')
                ax.set_ylim(ylim); ax.grid(True, alpha=0.3)
                if li == 0: ax.set_ylabel(yl)
                if li == n_loads - 1: ax.legend(fontsize=7)
            ax3 = axes[2][li]
            if feat_col in cd.columns:
                fv = cd[feat_col].values; valid = ~np.isnan(fv)
                ax3.scatter(pos[pm & valid], fv[pm & valid], alpha=0.5, s=10, color=CON_COLOR, label='Con')
                ax3.scatter(pos[nm & valid], fv[nm & valid], alpha=0.5, s=10, color=ECC_COLOR, label='Ecc')
            ax3.set_ylim(gf); ax3.set_xlabel('Position (m)'); ax3.grid(True, alpha=0.3)
            if li == 0: ax3.set_ylabel(feature_label)
            if li == n_loads - 1: ax3.legend(fontsize=7)
            axes[0][li].set_title(f'{lw} kg')
        fig.tight_layout()


# ==============================================================
#  图 4：Feature 均值柱状图
# ==============================================================

def plot_feature_bar_by_load(results, muscles, feature='mdf',
                             feature_label='MDF (Hz)', title_prefix='',
                             fmt=None):
    """
    Parameters
    ----------
    fmt : str, optional
        柱顶标注格式，默认根据 feature 自动选择
        ('mdf' → '.1f', 'rms' → '.4f')
    """
    if fmt is None:
        fmt = '.1f' if feature == 'mdf' else '.4f'
    load_weights = sorted(results.keys(), key=lambda x: float(x))
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(f'{title_prefix}Mean {feature.upper()} by Load', fontsize=14)

    for idx, musc in enumerate(muscles):
        ax = axes[idx // n_cols][idx % n_cols]
        col = _col(feature, musc)
        means, stds, labels, colors = [], [], [], []
        for i, lw in enumerate(load_weights):
            cd = results[lw].get('cutted_data') if results.get(lw) else None
            if cd is None or col not in cd.columns: continue
            vals = cd[col].dropna().values
            if len(vals) == 0: continue
            means.append(np.mean(vals)); stds.append(np.std(vals))
            labels.append(f'{lw}'); colors.append(LOAD_COLORS[i % len(LOAD_COLORS)])
        if not means:
            ax.set_title(musc); ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); continue
        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + (m * 0.02 + 0.001), f'{m:{fmt}}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f'{lb} kg' for lb in labels], fontsize=9)
        ax.set_title(musc); ax.set_xlabel('Load'); ax.set_ylabel(feature_label)
        ax.grid(True, alpha=0.3, axis='y')
    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()


# ==============================================================
#  组合图（固定负载 + 变负载）
# ==============================================================

def plot_feature_vs_time_combined(fixed_results, vload_results, muscles,
                                  feature='mdf', feature_label='MDF (Hz)'):
    """图 1 组合版：固定负载列 + 变负载列"""
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys())
    all_keys = fixed_keys + vload_keys
    n_cols = len(all_keys)
    n_muscles = len(muscles)
    if n_cols == 0 or n_muscles == 0: return

    def get_data(key):
        return fixed_results.get(key) or vload_results.get(key)

    gy = {}
    for musc in muscles:
        col = _col(feature, musc)
        gy[musc] = _global_ylim([get_data(k) for k in all_keys], col)

    fig, axes = plt.subplots(n_muscles, n_cols,
                             figsize=(3 * n_cols, 2 * n_muscles), squeeze=False)
    fig.suptitle(f'{feature.upper()} vs Time: Fixed + Variable Loads',
                 fontsize=14, fontweight='bold')
    cycle_colors = plt.cm.tab10.colors

    for mi, musc in enumerate(muscles):
        col = _col(feature, musc)
        for ci, key in enumerate(all_keys):
            ax = axes[mi][ci]
            d = get_data(key)
            if d is None: ax.set_visible(False); continue
            ad, cd = d.get('aligned_data'), d.get('cutted_data')
            if ad is not None and col in ad.columns and 'time' in ad.columns:
                v = ~np.isnan(ad[col].values)
                ax.plot(ad['time'].values[v], ad[col].values[v], 'k-', alpha=0.3, linewidth=0.8)
            _plot_segmented_line(ax, cd, col, cycle_colors)
            ax.set_ylim(gy[musc]); ax.set_xlabel('Time (s)'); ax.grid(True, alpha=0.3)
            if mi == 0:
                is_vl = key in vload_results
                ax.set_title(f'VLoad: {key}' if is_vl else f'{key} kg',
                             fontsize=9, color='green' if is_vl else 'black')
            if ci == 0: ax.set_ylabel(f'{musc}\n{feature_label}')
            if cd is not None and 'time' in cd.columns and len(cd) > 0:
                ax.set_xlim(cd['time'].min() - 1, cd['time'].max() + 1)
    fig.tight_layout()


def plot_feature_vs_position_combined(fixed_results, vload_results, muscles,
                                     feature='mdf', feature_label='MDF (Hz)'):
    """图 2 组合版：固定负载实心点 + 变负载空心三角"""
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys())
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(f'{feature.upper()} vs Position: Fixed + Variable Loads', fontsize=14)

    for idx, musc in enumerate(muscles):
        ax = axes[idx // n_cols][idx % n_cols]
        col = _col(feature, musc)
        for i, lw in enumerate(fixed_keys):
            cd = fixed_results[lw].get('cutted_data')
            if cd is None or col not in cd.columns: continue
            pos, vals = cd['pos_l'].values, cd[col].values
            v = ~np.isnan(vals)
            if np.any(v):
                ax.scatter(pos[v], vals[v], color=LOAD_COLORS[i % len(LOAD_COLORS)],
                           alpha=0.4, s=8, label=f'{lw} kg')
        for j, vk in enumerate(vload_keys):
            cd = vload_results[vk].get('cutted_data')
            if cd is None or col not in cd.columns: continue
            pos, vals = cd['pos_l'].values, cd[col].values
            v = ~np.isnan(vals)
            if np.any(v):
                ax.scatter(pos[v], vals[v], color=VLOAD_COLORS[j % len(VLOAD_COLORS)],
                           alpha=0.6, s=15, marker='^', label=f'VL: {vk}')
        ax.set_title(musc); ax.set_xlabel('Position (m)'); ax.set_ylabel(feature_label)
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)
    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()


def plot_pos_vel_emg_feature_grid_combined(fixed_results, vload_results, muscles,
                                           subject=None, feature='mdf',
                                           feature_label='MDF (Hz)'):
    """图 3 组合版：3行 × (n_fixed + n_vload) 列"""
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys())
    all_keys = fixed_keys + vload_keys
    n_cols = len(all_keys)
    if n_cols == 0: return
    if muscles is None:
        muscles = ['PMSte'] if subject and subject.target_motion == 'benchpress' else ['TA']
    if isinstance(muscles, str): muscles = [muscles]

    def get_cd(key):
        d = fixed_results.get(key) or vload_results.get(key)
        return d.get('cutted_data') if d else None

    for muscle_name in muscles:
        emg_col, feat_col = f'emg_{muscle_name}', _col(feature, muscle_name)
        gv, ge, gf = ([float('inf'), float('-inf')] for _ in range(3))
        for key in all_keys:
            cd = get_cd(key)
            if cd is None or emg_col not in cd.columns: continue
            vel, emg = np.abs(cd['vel_l'].values), np.abs(cd[emg_col].values)
            gv = [min(gv[0], np.min(vel)), max(gv[1], np.max(vel))]
            ge = [min(ge[0], np.min(emg)), max(ge[1], np.max(emg))]
            if feat_col in cd.columns:
                fv = cd[feat_col].dropna().values
                if len(fv) > 0: gf = [min(gf[0], np.min(fv)), max(gf[1], np.max(fv))]
        for g in [gv, ge, gf]:
            if g[0] != float('inf'):
                r = (g[1] - g[0]) * 0.1; g[0] = max(0, g[0] - r); g[1] += r
            else: g[0], g[1] = 0, 1

        fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 6), squeeze=False)
        fig.suptitle(f'Pos-Vel / EMG / {feature.upper()} ({muscle_name}) Fixed + VLoad',
                     fontsize=13, fontweight='bold')
        for ci, key in enumerate(all_keys):
            cd = get_cd(key)
            is_vl = key in vload_results
            if cd is None or emg_col not in cd.columns:
                for r in range(3): axes[r][ci].set_visible(False); continue
            pos, vel, emg = cd['pos_l'].values, cd['vel_l'].values, cd[emg_col].values
            pm, nm = vel > 0, vel <= 0
            for ri, (arr, ylim, yl) in enumerate([
                (np.abs(vel), gv, '|Velocity|'), (np.abs(emg), ge, f'{muscle_name} Act')]):
                ax = axes[ri][ci]
                ax.scatter(pos[pm], arr[pm], alpha=0.5, s=10, color=CON_COLOR, label='Con')
                ax.scatter(pos[nm], arr[nm], alpha=0.5, s=10, color=ECC_COLOR, label='Ecc')
                ax.set_ylim(ylim); ax.grid(True, alpha=0.3)
                if ci == 0: ax.set_ylabel(yl)
                if ci == n_cols - 1: ax.legend(fontsize=7)
            ax3 = axes[2][ci]
            if feat_col in cd.columns:
                fv = cd[feat_col].values; valid = ~np.isnan(fv)
                ax3.scatter(pos[pm & valid], fv[pm & valid], alpha=0.5, s=10, color=CON_COLOR, label='Con')
                ax3.scatter(pos[nm & valid], fv[nm & valid], alpha=0.5, s=10, color=ECC_COLOR, label='Ecc')
            ax3.set_ylim(gf); ax3.set_xlabel('Position (m)'); ax3.grid(True, alpha=0.3)
            if ci == 0: ax3.set_ylabel(feature_label)
            if ci == n_cols - 1: ax3.legend(fontsize=7)
            axes[0][ci].set_title(f'VL: {key}' if is_vl else f'{key} kg',
                                  fontsize=9, color='green' if is_vl else 'black')
        fig.tight_layout()


def plot_feature_bar_combined(fixed_results, vload_results, muscles,
                             feature='mdf', feature_label='MDF (Hz)', fmt=None):
    """图 4 组合版：固定负载柱 + 变负载斜线柱"""
    if fmt is None:
        fmt = '.1f' if feature == 'mdf' else '.4f'
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys())
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(f'Mean {feature.upper()}: Fixed vs Variable Loads', fontsize=14)

    for idx, musc in enumerate(muscles):
        ax = axes[idx // n_cols][idx % n_cols]
        col = _col(feature, musc)
        bm, bs, bl, bc, bh = [], [], [], [], []
        for i, lw in enumerate(fixed_keys):
            cd = fixed_results[lw].get('cutted_data')
            if cd is None or col not in cd.columns: continue
            vals = cd[col].dropna().values
            if len(vals) == 0: continue
            bm.append(np.mean(vals)); bs.append(np.std(vals))
            bl.append(f'{lw} kg'); bc.append(LOAD_COLORS[i % len(LOAD_COLORS)]); bh.append('')
        for j, vk in enumerate(vload_keys):
            cd = vload_results[vk].get('cutted_data')
            if cd is None or col not in cd.columns: continue
            vals = cd[col].dropna().values
            if len(vals) == 0: continue
            bm.append(np.mean(vals)); bs.append(np.std(vals))
            bl.append(f'VL:{vk}'); bc.append(VLOAD_COLORS[j % len(VLOAD_COLORS)]); bh.append('//')
        if not bm:
            ax.set_title(musc); ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); continue
        x = np.arange(len(bm))
        bars = ax.bar(x, bm, yerr=bs, capsize=4, color=bc, alpha=0.8, edgecolor='black', linewidth=0.5)
        for bi, h in enumerate(bh): bars[bi].set_hatch(h)
        for xi, m, s in zip(x, bm, bs):
            ax.text(xi, m + s + (m * 0.02 + 0.001), f'{m:{fmt}}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(bl, fontsize=7, rotation=30, ha='right')
        ax.set_title(musc); ax.set_ylabel(feature_label); ax.grid(True, alpha=0.3, axis='y')
    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()