"""
example_muscle_moment_contribution.py

针对每个左腿关节绘制「肌肉力矩贡献堆叠图」。

数据来源：
  - MuscleAnalysis 输出的 MuscleAnalysis_Moment_{coord_l}.sto
      使用文件中所有左腿肌肉列（列名含 '_l'）。
  - 逆动力学结果 inverse_dynamics.sto 中对应关节的净力矩曲线。
  - robot 数据中的 pos_l（高度），作为横轴。

只使用 upward 阶段的数据，每个关节输出两张图（每张图内每个 load 一个子图）：

  图1 — 全贡献图（all）：
    Y- 堆叠（0 到 负力矩总和）：提供负力矩的左腿肌肉。
    Y+ 堆叠（负力矩总和 到 正力矩总和）：提供正力矩的左腿肌肉。
      → 正堆叠顶端 = 净力矩，可直接对比 ID 曲线。
    黑色实线：逆动力学净力矩曲线（用于验证吻合程度）。

  图2 — 仅负力矩贡献图（negative）：
    只有 Y- 堆叠（0 到 负力矩总和）。

横轴均为机器人高度 pos_l（m）。
输出：result/{experiment_label}/test/
"""

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd

from digitaltwin import Subject, MultiLoadPipeline


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'

LOAD_KEYS = None               # None = 全部
# LOAD_KEYS = ['20']
JOINT_BASES_TO_PLOT = None     # None = 从 muscle_analysis_coordinates 自动推断
N_HEIGHT_BINS = 50
UPWARD_VELOCITY_THRESHOLD = 0.005
MOMENT_EPS = 0.01              # N·m，整体绝对值低于此值的肌肉忽略
SMOOTH_WINDOW = 5
N_COLS = 4
DPI = 200

# 在色块上直接标注肌肉名的最小峰值阈值（N·m）
# 色块峰值 > 该值时，在峰值处写肌肉名
LABEL_MIN_PEAK = 1.0
LABEL_FONT_SIZE = 7


# ============================================================
#  路径工具
# ============================================================

def get_base_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


def resolve_path(path):
    """
    解析 OpenSim 表格文件路径。

    MuscleAnalysis / InverseDynamics 输出有时会没有 .sto/.mot 后缀，
    但文件内容仍是标准 sto/mot 文本格式。因此这里按顺序尝试：
      1) 原始路径
      2) 去掉后缀后的路径（例如 xxx.sto -> xxx）
      3) 补 .sto
      4) 补 .mot
      5) 在同目录下做一次同 basename 的兜底查找
    """
    if path is None:
        return None

    path = os.path.normpath(path)
    folder = os.path.dirname(path)
    base = os.path.basename(path)
    root, ext = os.path.splitext(path)

    candidates = [path]

    # 如果传入 xxx.sto / xxx.mot，但真实文件没有后缀
    if ext:
        candidates.append(root)

    # 如果传入无后缀路径，但真实文件有 .sto / .mot
    candidates += [root + '.sto', root + '.mot']

    seen = set()
    for p in candidates:
        p = os.path.normpath(p)
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p

    # 兜底：同目录下查找 basename 相同、忽略 .sto/.mot 后缀的文件
    # 例如期望 xxx.sto，但实际存在 xxx；或期望 xxx，但实际存在 xxx.sto
    if folder and os.path.isdir(folder):
        target_stem = os.path.splitext(base)[0]
        for fname in os.listdir(folder):
            fstem, fext = os.path.splitext(fname)
            if fstem == target_stem and fext.lower() in ('', '.sto', '.mot'):
                candidate = os.path.join(folder, fname)
                if os.path.isfile(candidate):
                    return candidate

    return None


def read_opensim_file(path):
    """读取 OpenSim .sto/.mot（或无后缀同格式文件）为 DataFrame。"""
    resolved = resolve_path(path)
    if resolved is None:
        return None
    with open(resolved, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    header_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == 'endheader':
            header_start = i + 1
            break
    if header_start is None:
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('time'):
                header_start = i
                break
    if header_start is None:
        raise ValueError(f'无法识别表头: {resolved}')
    from io import StringIO
    import pandas as pd
    return pd.read_csv(
        StringIO(''.join(lines[header_start:])),
        sep=r'\s+', engine='python'
    )


# ============================================================
#  配置解析
# ============================================================

def build_joint_groups(config):
    """从 muscle_analysis_coordinates 构建左腿 {joint_base: coord_l}。"""
    coords = config.get('opensim_settings', {}).get(
        'muscle_analysis_coordinates', [])
    groups = {}
    for coord in coords:
        if coord.endswith('_l'):
            base = coord[:-2]
            groups[base] = coord
    if JOINT_BASES_TO_PLOT is not None:
        groups = {b: groups[b] for b in JOINT_BASES_TO_PLOT if b in groups}
    return groups  # {joint_base: coord_l}


def get_load_keys(config):
    if LOAD_KEYS is None:
        return list(config['modeling_file']['data'].keys())
    return [str(k) for k in LOAD_KEYS]


# ============================================================
#  文件路径
# ============================================================

def get_moment_path(config, base_dir, load_key, coord):
    label = config['experiment_label']
    return os.path.join(
        base_dir, 'result', label,
        'opensim', 'muscle_analysis', str(load_key),
        f'{label}_{load_key}_MuscleAnalysis_Moment_{coord}.sto'
    )


def get_id_path(config, base_dir, load_key):
    """逆动力学结果文件路径（OpenSim 默认输出 inverse_dynamics.sto）。"""
    label = config['experiment_label']
    return os.path.join(
        base_dir, 'result', label,
        'opensim', 'inverse_dynamics', str(load_key),
        'inverse_dynamics.sto'
    )


# ============================================================
#  标准数据处理 / 运动切片
# ============================================================

def run_standard_data_pipeline(config_path):
    """
    直接复用 example_data_analysis.py 中的标准处理流程：

      Subject -> MultiLoadPipeline.run(include_xsens=False)
      -> DataAligner.cut_aligned_data()

    这样 upward/downward 阶段与主数据分析脚本完全一致，
    避免本脚本用简单速度阈值重新切片导致 upward 数据不足。
    """
    subject = Subject(config_path)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True
    results = pipeline.run(include_xsens=False)
    return subject, pipeline, results


def get_upward_segment_from_results(pipeline_results, load_key):
    """
    从 MultiLoadPipeline.run() 的结果中取出指定 load 的 upward 切片。

    返回值是已经包含 robot/EMG 对齐结果、movement_type、cycle_id 等字段的 DataFrame。
    后续会按该 DataFrame 的 time 列插值加入 MuscleAnalysis / ID 结果。
    """
    result = pipeline_results.get(str(load_key))
    if result is None:
        # 兼容 key 是 int / float 的情况
        for k, v in pipeline_results.items():
            if str(k) == str(load_key):
                result = v
                break

    if result is None:
        return None

    cutted = result.get('cutted_data')
    if cutted is None:
        return None

    if isinstance(cutted, list):
        if not cutted:
            return None
        cutted = pd.concat(cutted, ignore_index=True)

    if cutted is None or len(cutted) == 0:
        return None

    df = cutted.copy()
    if 'movement_type' not in df.columns:
        print(f'[WARN] load={load_key} 切片结果中没有 movement_type 列')
        return None

    upward = df[df['movement_type'] == 'upward'].copy()
    if len(upward) == 0:
        return None

    return upward


# ============================================================
#  数据收集
# ============================================================

def collect_load_data(config, base_dir, coord_l, load_keys, pipeline_results):
    """
    读取每个 load 的左腿肌肉力矩数据、逆动力学净力矩，并按高度分箱。

    Returns
    -------
    data : dict
        load_key -> {
            'height_bins' : ndarray (N_HEIGHT_BINS,)
            'muscles'     : [(muscle_col, binned_moment)]
            'id_moment'   : ndarray (N_HEIGHT_BINS,) or None
        }
    """
    data = {}

    for load_key in load_keys:
        upward_df = get_upward_segment_from_results(pipeline_results, load_key)
        if upward_df is None or len(upward_df) < 10:
            print(f'[WARN] load={load_key} upward 切片数据不足（使用 MultiLoadPipeline/DataAligner）')
            continue

        if 'time' not in upward_df.columns or 'pos_l' not in upward_df.columns:
            print(f'[WARN] load={load_key} upward 切片缺少 time 或 pos_l 列')
            continue

        t_up = upward_df['time'].values.astype(float)
        h_up = upward_df['pos_l'].values.astype(float)
        valid_up = np.isfinite(t_up) & np.isfinite(h_up)
        upward_df = upward_df.loc[valid_up].copy()
        t_up = upward_df['time'].values.astype(float)
        h_up = upward_df['pos_l'].values.astype(float)

        if len(t_up) < 10:
            print(f'[WARN] load={load_key} upward 有效数据不足')
            continue

        # ── 读取 Moment .sto ──────────────────────────────────
        mpath = get_moment_path(config, base_dir, load_key, coord_l)
        m_df = read_opensim_file(mpath)
        if m_df is None or 'time' not in m_df.columns:
            print(f'[MISS] Moment 文件: {mpath}')
            continue

        m_t = m_df['time'].values.astype(float)

        # 取所有列名含 '_l' 的肌肉列（左腿肌肉）
        muscle_cols = [
            c for c in m_df.columns
            if c != 'time' and '_l' in c.lower()
        ]
        if not muscle_cols:
            print(f'[WARN] load={load_key} Moment 文件中无左腿肌肉列')
            continue

        # 在标准 upward 切片 DataFrame 中加入对应时刻的 MuscleAnalysis moment
        seg_df = upward_df.copy()

        muscle_records = []
        ma_cols = []   # 所有左腿肌肉 moment 列，用于计算所有肌肉力矩之和
        for col in muscle_cols:
            vals = m_df[col].values.astype(float)
            ma_col = f'ma_moment_{col}'
            seg_df[ma_col] = np.interp(
                t_up, m_t, vals,
                left=vals[0], right=vals[-1]
            )
            ma_cols.append(ma_col)

            valid = np.isfinite(seg_df['pos_l'].values) & np.isfinite(seg_df[ma_col].values)
            if valid.sum() == 0:
                continue
            if np.nanmax(np.abs(seg_df.loc[valid, ma_col].values)) < MOMENT_EPS:
                continue

            muscle_records.append((
                col,
                seg_df.loc[valid, 'pos_l'].values.astype(float),
                seg_df.loc[valid, ma_col].values.astype(float),
            ))

        # 所有左腿肌肉力矩逐时刻求和（不因 MOMENT_EPS 过滤而丢掉小肌肉）
        muscle_total_raw = None
        muscle_total_mean_upward = None
        if ma_cols:
            muscle_total_raw = seg_df[ma_cols].sum(axis=1, skipna=True).values.astype(float)
            if np.isfinite(muscle_total_raw).any():
                muscle_total_mean_upward = float(np.nanmean(muscle_total_raw))

        if not muscle_records:
            print(f'[WARN] load={load_key} coord={coord_l} 无有效肌肉')
            continue

        # ── 读取逆动力学净力矩 ─────────────────────────────────
        id_df = read_opensim_file(get_id_path(config, base_dir, load_key))
        id_moment_raw = None
        if id_df is not None and 'time' in id_df.columns:
            # 列名可能是 knee_angle_l_moment 或 knee_angle_l/moment
            coord_base = coord_l  # e.g. 'knee_angle_l'
            id_col = None
            for c in id_df.columns:
                cl = c.lower().replace('/', '_')
                if coord_base in cl and 'moment' in cl:
                    id_col = c
                    break
            if id_col:
                id_t = id_df['time'].values.astype(float)
                id_v = id_df[id_col].values.astype(float)
                id_analysis_col = f'id_moment_{coord_l}'
                seg_df[id_analysis_col] = np.interp(
                    t_up, id_t, id_v,
                    left=id_v[0], right=id_v[-1]
                )
                id_moment_raw = seg_df[id_analysis_col].values.astype(float)
            else:
                print(f'[WARN] load={load_key} 逆动力学文件中未找到 {coord_l} 列')
        else:
            print(f'[MISS] 逆动力学文件: {get_id_path(config, base_dir, load_key)}')

        # ── 按高度分箱 ─────────────────────────────────────────
        all_h = np.concatenate([r[1] for r in muscle_records])
        h_min, h_max = float(np.nanmin(all_h)), float(np.nanmax(all_h))
        if h_max <= h_min:
            continue

        edges = np.linspace(h_min, h_max, N_HEIGHT_BINS + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        def bin_series(h_arr, v_arr):
            b = np.full(N_HEIGHT_BINS, np.nan)
            for bi in range(N_HEIGHT_BINS):
                mask = (h_arr >= edges[bi]) & (h_arr < edges[bi + 1])
                if mask.sum() > 0:
                    b[bi] = float(np.nanmean(v_arr[mask]))
            idx_v = np.where(np.isfinite(b))[0]
            if len(idx_v) >= 2:
                b = np.interp(np.arange(N_HEIGHT_BINS), idx_v, b[idx_v])
            elif len(idx_v) == 1:
                b[:] = b[idx_v[0]]
            return b

        binned_muscles = [(col, bin_series(h, v)) for col, h, v in muscle_records]

        binned_id = None
        if id_moment_raw is not None:
            valid_id = np.isfinite(h_up) & np.isfinite(id_moment_raw)
            if valid_id.sum() >= 2:
                binned_id = bin_series(h_up[valid_id], id_moment_raw[valid_id])

        data[str(load_key)] = {
            'height_bins': centers,
            'muscles': binned_muscles,
            'id_moment': binned_id,
            # 用于判断主力矩方向：与图中黑色 IK 曲线一致的分箱后均值
            'id_mean': float(np.nanmean(binned_id)) if binned_id is not None else None,
            # 用于最后打印：标准 upward 切片中原始插值 IK 力矩均值
            'id_mean_upward': (
                float(np.nanmean(id_moment_raw))
                if id_moment_raw is not None else None
            ),
            # 用于最后打印：标准 upward 切片中所有左腿肌肉 moment 的逐时刻求和均值
            'muscle_total_mean_upward': muscle_total_mean_upward,
        }
        print(f'[OK] load={load_key} coord={coord_l}  '
              f'n_muscles={len(binned_muscles)}  '
              f'id={"yes" if binned_id is not None else "no"}')

    return data


# ============================================================
#  颜色分配
# ============================================================

def build_color_map(all_muscle_cols):
    """给所有出现过的肌肉列分配固定颜色。"""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    extra = plt.cm.tab20.colors  # 备用 20 色
    palette = list(colors) + [c for c in extra if c not in colors]
    unique = sorted(set(all_muscle_cols))
    return {col: palette[i % len(palette)] for i, col in enumerate(unique)}


def _add_band_label(ax, x, vals, bottom, color, col):
    """
    若色块峰值 > LABEL_MIN_PEAK，在峰值处的色块中央写肌肉名。

    Parameters
    ----------
    x      : ndarray  横轴
    vals   : ndarray  本层增量（正/负）
    bottom : ndarray  本层起始基线
    color  : str      色块颜色
    col    : str      肌肉名

    Returns
    -------
    bool  是否添加了标签
    """
    abs_v = np.abs(vals)
    peak = float(np.nanmax(abs_v))
    if peak < LABEL_MIN_PEAK:
        return False
    # 在峰值处放标签
    pidx = int(np.nanargmax(abs_v))
    mid_y = float(bottom[pidx]) + float(vals[pidx]) / 2.0
    # 判断背景亮度，决定字体颜色
    import colorsys
    try:
        r, g, b = plt.matplotlib.colors.to_rgb(color)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        fc = 'white' if luma < 0.55 else 'black'
    except Exception:
        fc = 'white'
    ax.text(
        x[pidx], mid_y, col,
        ha='center', va='center',
        fontsize=LABEL_FONT_SIZE, fontweight='bold', color=fc,
        clip_on=True,
        bbox=dict(boxstyle='round,pad=0.15', facecolor=color,
                  edgecolor='none', alpha=0.75),
        zorder=10,
    )
    return True


# ============================================================
#  方向判断
# ============================================================

def _determine_dominant_positive(data):
    """
    根据各 load 的 IK 均值判断关节力矩的主要方向。

    Returns
    -------
    bool  True = 主要为正力矩；False = 主要为负力矩
    """
    id_means = [
        e['id_mean'] for e in data.values()
        if e.get('id_mean') is not None
    ]
    if not id_means:
        return True
    return float(np.nanmean(id_means)) >= 0


# ============================================================
#  合并图：每个 load 一行，左列 = 主力矩，右列 = 次力矩
#  两列共享 Y 轴范围，纵轴完全一致
# ============================================================

def plot_contributions_combined(joint_base, coord_l, data, color_map,
                                dominant_positive=True, output_path=None):
    """
    每个 load 一行，共 2 列：
      左列（主力矩）：
        dominant_positive=True  → 只画正力矩肌肉，基线从 neg_sum 向上堆叠至 total_sum；
                                   橙线 = total_sum，黑虚线 = IK 净力矩。
        dominant_positive=False → 只画负力矩肌肉，Y 轴反转；
                                   橙线 = total_sum，黑虚线 = IK 净力矩。
      右列（次力矩）：
        dominant_positive=True  → 只画负力矩肌肉，从 0 向下堆叠。
        dominant_positive=False → 只画正力矩肌肉，从 0 向上堆叠。

    两列的 Y 轴范围完全一致（sharey）。
    字体略大以适应宽幅布局。
    """
    load_keys = list(data.keys())
    if not load_keys:
        return

    n_loads = len(load_keys)
    # 每个 load 一行，2 列（主 | 次），共享 Y 轴
    fig, axes = plt.subplots(
        n_loads, 2,
        figsize=(11.0, n_loads * 4.2),
        squeeze=False,
        sharey='row',          # 同一行两列共享完全相同的 Y 轴范围
    )
    dir_main = 'Positive' if dominant_positive else 'Negative (Y-inv)'
    dir_minor = 'Negative' if dominant_positive else 'Positive'
    fig.suptitle(
        f'Muscle Moment Contribution  —  {coord_l}\n'
        f'Left: Main ({dir_main})   |   Right: Minor ({dir_minor})',
        fontsize=13, fontweight='bold'
    )

    legend_handles_main  = []
    legend_labeled_main  = set()
    legend_handles_minor = []
    legend_labeled_minor = set()
    seen_line_labels = set()

    FONT = 10   # 稍大字号

    for row, lk in enumerate(load_keys):
        ax_main  = axes[row][0]
        ax_minor = axes[row][1]
        entry    = data[lk]
        x        = entry['height_bins']
        muscles  = entry['muscles']
        id_moment = entry['id_moment']

        # ── 三个累计量 ──────────────────────────────────────────
        pos_sum = sum(
            np.nan_to_num(np.where(b > 0, b, 0.0)) for _, b in muscles
        )
        neg_sum = sum(
            np.nan_to_num(np.where(b < 0, b, 0.0)) for _, b in muscles
        )
        total_sum = pos_sum + neg_sum

        # ── 左列：主力矩 ────────────────────────────────────────
        if dominant_positive:
            main_list = [
                (col, np.where(b > 0, b, 0.0)) for col, b in muscles
                if np.nanmax(np.where(b > 0, b, 0.0)) > MOMENT_EPS * 0.1
            ]
            bottom = neg_sum.copy()
            for col, vals in main_list:
                v = np.nan_to_num(vals)
                color = color_map.get(col, 'gray')
                ax_main.fill_between(x, bottom, bottom + v,
                                     color=color, alpha=0.82, linewidth=0)
                labeled = _add_band_label(ax_main, x, v, bottom, color, col)
                if not labeled and col not in legend_labeled_main:
                    legend_labeled_main.add(col)
                    legend_handles_main.append(mpatches.Patch(color=color, label=col))
                bottom += v
        else:
            main_list = [
                (col, np.where(b < 0, b, 0.0)) for col, b in muscles
                if np.nanmin(np.where(b < 0, b, 0.0)) < -MOMENT_EPS * 0.1
            ]
            bottom = pos_sum.copy()
            for col, vals in main_list:
                v = np.nan_to_num(vals)
                color = color_map.get(col, 'gray')
                ax_main.fill_between(x, bottom + v, bottom,
                                     color=color, alpha=0.82, linewidth=0)
                labeled = _add_band_label(ax_main, x, v, bottom, color, col)
                if not labeled and col not in legend_labeled_main:
                    legend_labeled_main.add(col)
                    legend_handles_main.append(mpatches.Patch(color=color, label=col))
                bottom += v
            ax_main.invert_yaxis()

        # 总力矩线 + IK 线
        ax_main.plot(x, total_sum, color='tab:orange', linewidth=1.8,
                     linestyle='-', zorder=6)
        if 'Muscle total' not in seen_line_labels:
            seen_line_labels.add('Muscle total')
            legend_handles_main.append(
                plt.Line2D([0], [0], color='tab:orange', linewidth=1.8,
                           linestyle='-', label='Muscle total')
            )
        if id_moment is not None:
            ax_main.plot(x, id_moment, color='black', linewidth=1.5,
                         linestyle='--', zorder=5)
            if 'ID net moment' not in seen_line_labels:
                seen_line_labels.add('ID net moment')
                legend_handles_main.append(
                    plt.Line2D([0], [0], color='black', linewidth=1.5,
                               linestyle='--', label='ID net moment')
                )

        ax_main.axhline(0, color='gray', linewidth=0.6, alpha=0.5)
        ax_main.set_title(f'{lk} kg  —  Main ({dir_main})', fontsize=FONT, fontweight='bold')
        ax_main.set_xlabel('Height / pos_l (m)', fontsize=FONT - 1)
        ax_main.set_ylabel('Moment (N·m)', fontsize=FONT - 1)
        ax_main.grid(True, alpha=0.25)
        ax_main.tick_params(labelsize=FONT - 1)

        # ── 右列：次力矩 ────────────────────────────────────────
        if dominant_positive:
            minor_list = [
                (col, np.where(b < 0, b, 0.0)) for col, b in muscles
                if np.nanmin(np.where(b < 0, b, 0.0)) < -MOMENT_EPS * 0.1
            ]
            bottom_m = np.zeros(len(x))
            for col, vals in minor_list:
                v = np.nan_to_num(vals)
                color = color_map.get(col, 'gray')
                ax_minor.fill_between(x, bottom_m + v, bottom_m,
                                      color=color, alpha=0.82, linewidth=0)
                labeled = _add_band_label(ax_minor, x, v, bottom_m, color, col)
                if not labeled and col not in legend_labeled_minor:
                    legend_labeled_minor.add(col)
                    legend_handles_minor.append(mpatches.Patch(color=color, label=col))
                bottom_m += v
        else:
            minor_list = [
                (col, np.where(b > 0, b, 0.0)) for col, b in muscles
                if np.nanmax(np.where(b > 0, b, 0.0)) > MOMENT_EPS * 0.1
            ]
            bottom_m = np.zeros(len(x))
            for col, vals in minor_list:
                v = np.nan_to_num(vals)
                color = color_map.get(col, 'gray')
                ax_minor.fill_between(x, bottom_m, bottom_m + v,
                                      color=color, alpha=0.82, linewidth=0)
                labeled = _add_band_label(ax_minor, x, v, bottom_m, color, col)
                if not labeled and col not in legend_labeled_minor:
                    legend_labeled_minor.add(col)
                    legend_handles_minor.append(mpatches.Patch(color=color, label=col))
                bottom_m += v

        ax_minor.axhline(0, color='gray', linewidth=0.6, alpha=0.5)
        ax_minor.set_title(f'{lk} kg  —  Minor ({dir_minor})', fontsize=FONT, fontweight='bold')
        ax_minor.set_xlabel('Height / pos_l (m)', fontsize=FONT - 1)
        # 共享 Y 轴时右列不重复画纵轴标签
        ax_minor.set_ylabel('Moment (N·m)', fontsize=FONT - 1)
        ax_minor.grid(True, alpha=0.25)
        ax_minor.tick_params(labelsize=FONT - 1)

    # ── 图例：放在最后一行两列各自内 ────────────────────────────
    if legend_handles_main:
        axes[-1][0].legend(handles=legend_handles_main,
                           fontsize=7, loc='best', ncol=2)
    if legend_handles_minor:
        axes[-1][1].legend(handles=legend_handles_minor,
                           fontsize=7, loc='best', ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f'[Saved] {output_path}')


# ============================================================
#  主程序
# ============================================================

def main():
    base_dir = get_base_dir()
    config_path = get_config_path()

    print(f'配置文件: {config_path}')
    print(f'基准目录: {base_dir}')

    # 复用 example_data_analysis.py 的标准数据处理和运动切片流程
    subject, pipeline, pipeline_results = run_standard_data_pipeline(config_path)
    config = subject.config

    # 只分析左腿关节
    joint_groups = build_joint_groups(config)
    if not joint_groups:
        raise ValueError(
            '没有可分析的左腿关节；'
            '请确保 opensim_settings.muscle_analysis_coordinates 中包含 _l 后缀坐标'
        )

    print('\n左腿关节坐标：')
    for base, coord in joint_groups.items():
        print(f'  {base}: {coord}')

    load_keys = get_load_keys(config)
    experiment_label = config['experiment_label']
    out_dir = os.path.join(base_dir, 'result', experiment_label, 'test')

    # 记录每个负载、每个关节在标准 upward 切片中的平均力矩
    id_mean_summary = {}
    muscle_total_mean_summary = {}

    for joint_base, coord_l in joint_groups.items():
        print(f'\n{"=" * 60}')
        print(f'关节: {joint_base}  坐标: {coord_l}')
        print('=' * 60)

        data = collect_load_data(
            config=config,
            base_dir=base_dir,
            coord_l=coord_l,
            load_keys=load_keys,
            pipeline_results=pipeline_results,
        )

        if not data:
            print('  [SKIP] 无数据')
            continue

        id_mean_summary[joint_base] = {
            lk: entry.get('id_mean_upward', entry.get('id_mean'))
            for lk, entry in data.items()
        }
        muscle_total_mean_summary[joint_base] = {
            lk: entry.get('muscle_total_mean_upward')
            for lk, entry in data.items()
        }

        # 收集所有肌肉列名以建立全局颜色映射
        all_cols = [col for entry in data.values()
                    for col, _ in entry['muscles']]
        color_map = build_color_map(all_cols)

        # 根据 IK 均值判断主力矩方向
        dominant_positive = _determine_dominant_positive(data)
        dir_label = '正(positive)' if dominant_positive else '负(negative)'
        print(f'  主力矩方向: {dir_label}')

        plot_contributions_combined(
            joint_base, coord_l, data, color_map,
            dominant_positive=dominant_positive,
            output_path=os.path.join(
                out_dir, f'moment_contribution_{joint_base}.png'
            )
        )

    def print_summary_table(title, summary, load_keys, note):
        """按 joint × load 打印均值表。"""
        if not summary:
            return
        print('\n' + '=' * 60)
        print(title)
        print('=' * 60)

        header = f'{"joint":<20s}' + ''.join(
            f'{str(lk) + " kg":>14s}' for lk in load_keys
        )
        print(header)
        print('-' * len(header))

        for joint_base, load_means in summary.items():
            row = f'{joint_base:<20s}'
            for lk in load_keys:
                v = load_means.get(str(lk))
                if v is None or not np.isfinite(v):
                    row += f'{"N/A":>14s}'
                else:
                    row += f'{v:>14.4f}'
            print(row)

        print('\n单位: N·m')
        print(note)

    # ── 最后打印每个负载、每个关节在 upward 阶段的 ID 平均关节力矩 ──
    print_summary_table(
        title='ID 关节力矩平均值（仅标准 upward 切片阶段）',
        summary=id_mean_summary,
        load_keys=load_keys,
        note=('说明: 均值基于 MultiLoadPipeline/DataAligner 切出的 upward 阶段，'
              '再按对应时间插值 inverse_dynamics.sto 的 ID moment。')
    )

    # ── 类似地打印所有左腿肌肉力矩逐时刻求和后的平均值 ──
    print_summary_table(
        title='所有左腿肌肉力矩之和的平均值（仅标准 upward 切片阶段）',
        summary=muscle_total_mean_summary,
        load_keys=load_keys,
        note=('说明: 先在标准 upward 阶段按时间插值 MuscleAnalysis_Moment，'
              '再对该关节下所有左腿肌肉 moment 逐时刻求和，最后对时间取平均。')
    )

    plt.show()


if __name__ == '__main__':
    main()