"""
example_emg_force_ratio.py

计算所有固定负载与变负载数据中各肌肉的 EMG-力比值与 EMG-占比指标，
并可视化为棋盘格热图及高度-占比曲线图。

三个指标（针对每个 肌肉 × 数据集 的组合）：
  A. mean(emg) / mean(|force|)
       先对该数据集中所有时刻取平均 EMG，再除以平均力的绝对值
  B. mean(emg / |force|)
       先对每个时刻计算 emg/|force|，再取均值
  C. mean(emg / sum_all_emg)
       每个时刻将指定肌肉激活除以所有 target_muscles 激活之和，
       再取均值（占比）

输出：
  - 控制台：三张表格（行=肌肉，列=固定负载+变负载标签）
  - 图窗 1：三个棋盘格热图（A / B / C）
  - 图窗 2：居民 C（EMG 占比）vs 高度的散点图，每块肌肉一个子图

用法：
    python example_emg_force_ratio.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from digitaltwin import Subject, MultiLoadPipeline


# 力列：使用左右交互力的绝对值之和
FORCE_COLS = ['force_l', 'force_r']

# 是否包含变负载数据
INCLUDE_VLOAD = True


# ============================================================
#  工具函数
# ============================================================

def get_combined_force_abs(df):
    """取可用的力列绝对值之和；若两列均不存在则返回 None。"""
    available = [c for c in FORCE_COLS if c in df.columns]
    if not available:
        return None
    return df[available].abs().sum(axis=1)


def resolve_emg_col(df, muscle):
    """
    查找肌肉对应的实际列名，支持 LFibLon / RFibLon 等变体。
    优先精确匹配 emg_{muscle}，然后搜索包含 muscle 的列。
    """
    exact = f'emg_{muscle}'
    if exact in df.columns:
        return exact
    candidates = [c for c in df.columns
                  if c.startswith('emg_') and muscle in c]
    return candidates[0] if candidates else None


def compute_force_ratios(df, emg_col):
    """
    计算两个力相关指标（力均取绝对值）。

    Returns
    -------
    ratio_mean : float  mean(emg) / mean(|force|)
    mean_ratio : float  mean(emg / |force|)
    n : int
    """
    if emg_col not in df.columns:
        return np.nan, np.nan, 0
    force_abs = get_combined_force_abs(df)
    if force_abs is None:
        return np.nan, np.nan, 0
    emg = df[emg_col]
    valid = np.isfinite(emg) & np.isfinite(force_abs) & (force_abs > 1e-6)
    n = valid.sum()
    if n == 0:
        return np.nan, np.nan, 0
    emg_v = emg[valid].values
    force_v = force_abs[valid].values
    ratio_mean = float(np.mean(emg_v)) / float(np.mean(force_v))
    mean_ratio = float(np.mean(emg_v / force_v))
    return ratio_mean, mean_ratio, int(n)


def compute_emg_share(df, emg_col, all_emg_cols):
    """
    计算指定肌肉占所有目标肌肉激活之和的平均占比。

    share_i = emg_i / sum(emg_j  for j in all_emg_cols)  # 每个时刻
    返回所有时刻 share_i 的均值。

    Returns
    -------
    mean_share : float
    n : int
    """
    if emg_col not in df.columns:
        return np.nan, 0
    avail = [c for c in all_emg_cols if c in df.columns]
    if not avail:
        return np.nan, 0
    total = df[avail].sum(axis=1)
    emg = df[emg_col]
    valid = np.isfinite(emg) & (total > 1e-8)
    n = valid.sum()
    if n == 0:
        return np.nan, 0
    share = emg[valid].values / total[valid].values
    return float(np.mean(share)), int(n)


def collect_fixed_load_data(results):
    """从固定负载 results 字典中提取切片数据。"""
    out = {}
    for load_weight, result in results.items():
        cd = result.get('cutted_data')
        if cd is None:
            continue
        if isinstance(cd, list):
            if not cd:
                continue
            cd = pd.concat(cd, ignore_index=True)
        if len(cd) == 0:
            continue
        out[f'{load_weight}kg'] = cd
    return out


def collect_vload_data(vload_results):
    """从变负载 vload_results 字典中提取切片数据。"""
    out = {}
    for label, result in vload_results.items():
        cd = result.get('cutted_data')
        if cd is None:
            continue
        if isinstance(cd, list):
            if not cd:
                continue
            cd = pd.concat(cd, ignore_index=True)
        if len(cd) == 0:
            continue
        out[label] = cd
    return out


def build_ratio_tables(datasets, muscles):
    """
    计算所有 (肌肉, 数据集) 组合的三个指标。

    Returns
    -------
    df_A : 行=肌肉, 列=数据集, mean(emg)/mean(|force|)
    df_B : 行=肌肉, 列=数据集, mean(emg/|force|)
    df_C : 行=肌肉, 列=数据集, mean(emg / sum_all_emg)
    df_n : 行=肌肉, 列=数据集, 样本数
    """
    col_labels = list(datasets.keys())
    A_data, B_data, C_data, n_data = {}, {}, {}, {}

    # 预先构造每个数据集中所有目标肌肉的实际列名（内圈）
    def get_all_emg_cols(df, muscles):
        cols = []
        for m in muscles:
            c = resolve_emg_col(df, m)
            if c is not None:
                cols.append(c)
        return cols

    for muscle in muscles:
        a_row, b_row, c_row, n_row = [], [], [], []
        for label, df in datasets.items():
            emg_col = resolve_emg_col(df, muscle)
            all_emg = get_all_emg_cols(df, muscles)
            if emg_col is None:
                a_row.append(np.nan)
                b_row.append(np.nan)
                c_row.append(np.nan)
                n_row.append(0)
            else:
                rm, mr, n = compute_force_ratios(df, emg_col)
                ms, _   = compute_emg_share(df, emg_col, all_emg)
                a_row.append(rm)
                b_row.append(mr)
                c_row.append(ms)
                n_row.append(n)
        A_data[muscle] = a_row
        B_data[muscle] = b_row
        C_data[muscle] = c_row
        n_data[muscle] = n_row

    def make_df(d):
        return pd.DataFrame(d, index=col_labels).T

    return make_df(A_data), make_df(B_data), make_df(C_data), make_df(n_data)


# ============================================================
#  打印工具
# ============================================================

def print_table(df, title, float_fmt='.6f'):
    print(f'\n{"-"*60}')
    print(f'  {title}')
    print(f'{"-"*60}')
    print(df.to_string(float_format=lambda x: format(x, float_fmt)))


# ============================================================
#  棋盘格热图
# ============================================================

def plot_checkerboard(df, title, ax=None, cmap='YlOrRd',
                      fmt='.4f', annot=True):
    """
    绘制棋盘格热图（行=肌肉, 列=数据集）。
    """
    data = df.values.astype(float)
    rows, cols = data.shape
    row_labels = list(df.index)
    col_labels = list(df.columns)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6, cols * 1.4), max(4, rows * 0.8)))

    valid_vals = data[np.isfinite(data)]
    vmin = valid_vals.min() if len(valid_vals) else 0
    vmax = valid_vals.max() if len(valid_vals) else 1
    if vmin == vmax:
        vmax = vmin + 1e-9

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            color = (cmap_obj(norm(val)) if np.isfinite(val)
                     else (0.85, 0.85, 0.85, 1.0))
            ax.add_patch(plt.Rectangle(
                [j, rows - 1 - i], 1, 1,
                facecolor=color, edgecolor='white', linewidth=1.5))
            if annot:
                text_val = f'{val:{fmt}}' if np.isfinite(val) else 'N/A'
                if np.isfinite(val):
                    r, g, b = color[:3]
                    txt_color = 'white' if (0.299*r + 0.587*g + 0.114*b) < 0.5 else 'black'
                else:
                    txt_color = 'gray'
                ax.text(j + 0.5, rows - 1 - i + 0.5, text_val,
                        ha='center', va='center', fontsize=8, color=txt_color)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(col_labels, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(row_labels[::-1], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Dataset', fontsize=9)
    ax.set_ylabel('Muscle', fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    return ax


# ============================================================
#  高度 - EMG占比 曲线图
# ============================================================

def plot_emg_share_vs_height(datasets, muscles, n_cols=3):
    """
    居民 C (emg / sum_all_emg) vs 高度 的散点图。

    居民【每块肌肉一个子图】，每个子图中不同数据集用不同颜色。

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
    muscles  : list[str]
    n_cols   : int  每行子图数
    """
    n_muscles = len(muscles)
    n_rows = int(np.ceil(n_muscles / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.5, n_rows * 3.2),
        squeeze=False)
    fig.suptitle('EMG Share (emg / Σall_emg) vs Height',
                 fontsize=13, fontweight='bold')

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {label: prop_cycle[i % len(prop_cycle)]
              for i, label in enumerate(datasets.keys())}

    # 预先构建每个数据集中目标肌肉列名
    def get_all_emg_cols(df):
        return [c for m in muscles
                for c in [resolve_emg_col(df, m)] if c is not None]

    for idx, muscle in enumerate(muscles):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        for label, df in datasets.items():
            if 'pos_l' not in df.columns:
                continue
            emg_col = resolve_emg_col(df, muscle)
            if emg_col is None:
                continue
            all_emg = get_all_emg_cols(df)
            if not all_emg:
                continue
            total = df[all_emg].sum(axis=1)
            emg   = df[emg_col]
            valid = np.isfinite(emg) & (total > 1e-8)
            if valid.sum() == 0:
                continue
            height = df.loc[valid, 'pos_l'].values
            share  = emg[valid].values / total[valid].values
            ax.scatter(height, share, s=4, alpha=0.6,
                       color=colors[label], label=label)

        ax.set_title(muscle, fontsize=10, fontweight='bold')
        ax.set_xlabel('Height (m)', fontsize=8)
        ax.set_ylabel('EMG share', fontsize=8)
        ax.legend(fontsize=7, loc='best', markerscale=2)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(n_muscles, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================
#  主程序
# ============================================================

def main():
    # subject = Subject('../config/20250409_squat_NCMP001_mvc.json')
    subject = Subject('../config/20260513_squat_FTS09_xsens.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # target_muscles = ['VL', 'RF', 'FibLon', 'GL', 'SOL']
    target_muscles = ["LGL", "LFibLon", "LVL", "LRF", "LVM", "LBF", "LGlutMax",]
                      # "RGL", "RFibLon", "RVL", "RRF", "RVM", "RBF", "RGlutMax"]
    # target_muscles = ["LTA", "LGL", "LFibLon", "LVL", "LRF", "LVM", "LAddl", "LBF", "LGlutMax", "LGlutMed"]

    # ---- 加载固定负载数据 ----
    print('\n[1/2] 加载固定负载数据...')
    results = pipeline.run(include_xsens=False)
    datasets = collect_fixed_load_data(results)

    # ---- 加载变负载数据 ----
    if INCLUDE_VLOAD:
        print('\n[2/2] 加载变负载数据...')
        vload_results = pipeline.run_vload()
        datasets.update(collect_vload_data(vload_results))

    if not datasets:
        print('未找到任何数据，退出。')
        return

    print(f'\n数据集列表: {list(datasets.keys())}')
    print(f'目标肌肉  : {target_muscles}')

    # ---- 计算比值表 ----
    df_A, df_B, df_C, df_n = build_ratio_tables(datasets, target_muscles)

    # ---- 打印表格 ----
    print_table(df_A, 'A: mean(EMG) / mean(|Force|)  [行=肌肉, 列=数据集]')
    print_table(df_B, 'B: mean(EMG / |Force|)         [行=肌肉, 列=数据集]')
    print_table(df_C, 'C: mean(EMG / Σall_EMG)        [行=肌肉, 列=数据集]')
    print_table(df_n, '样本数 (n)                   [行=肌肉, 列=数据集]',
                float_fmt='.0f')

    # ---- 图窗 1：三张棋盘格热图 ----
    n_ds = len(datasets)
    n_ms = len(target_muscles)
    w = max(8, n_ds * 1.5)
    h = max(4, n_ms * 0.9)

    fig1, axes1 = plt.subplots(1, 3, figsize=(w * 3 + 3, h + 1))
    fig1.suptitle('EMG Ratio & Share Checkerboard',
                  fontsize=13, fontweight='bold')
    plot_checkerboard(df_A,
                      'A: mean(EMG) / mean(|Force|)',
                      ax=axes1[0], cmap='YlOrRd')
    plot_checkerboard(df_B,
                      'B: mean(EMG / |Force|)',
                      ax=axes1[1], cmap='YlGnBu')
    plot_checkerboard(df_C,
                      'C: mean(EMG / Σall_EMG)',
                      ax=axes1[2], cmap='PuBuGn')
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # ---- 图窗 2：高度 - EMG占比 曲线图 ----
    plot_emg_share_vs_height(datasets, target_muscles)

    plt.show()


if __name__ == '__main__':
    main()