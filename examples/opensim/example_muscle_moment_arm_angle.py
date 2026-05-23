"""
example_muscle_moment_arm_angle.py

将 OpenSim MuscleAnalysis 生成的肌肉力臂结果与 .mot 中的关节角对齐，
绘制“关节角 - 肌肉力臂”散点图，并叠加拟合函数。

这里的“每个关节角一张图”指按左右合并后的关节名称分组：
  - knee_angle 一张图
      左腿肌肉使用 knee_angle_l
      右腿肌肉使用 knee_angle_r
  - hip_flexion 一张图
      左腿肌肉使用 hip_flexion_l
      右腿肌肉使用 hip_flexion_r
  - ankle_angle 一张图
      左腿肌肉使用 ankle_angle_l
      右腿肌肉使用 ankle_angle_r

核心功能：
  1. 使用相对路径读取 examples/config/*.json
  2. 从 JSON 中读取：
       - emg_settings.musc_label
       - opensim_settings.muscle_analysis_muscles
       - opensim_settings.muscle_analysis_coordinates
  3. muscle_analysis_muscles 与 musc_label 一一对应；
     不再自动猜 OpenSim 肌肉名。
  4. 每个“关节基名”画一张大图，例如 knee_angle / hip_flexion / ankle_angle。
  5. 左腿肌肉自动取该关节的 *_l 坐标，右腿肌肉自动取 *_r 坐标。
  6. 使用散点图：x = 关节角度，y = 肌肉力臂。
  7. 对散点进行多项式或傅里叶级数拟合，并将拟合曲线叠加到散点图上。
  8. 如果某条肌肉在某个关节上的力臂整体范围都小于 MOMENT_ARM_EPS，则整条数据忽略；
     如果正常曲线经过 0 或包含小于 0.001 的局部区间，则不会删除这些点。

输出：
  result/{experiment_label}/test/
    moment_arm_vs_knee_angle.png
    moment_arm_vs_hip_flexion.png
    ...
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'

# 只画这些负载；None 表示画 config 中所有 load
LOAD_KEYS = None
# LOAD_KEYS = ["20", "38", "56", "75", "94", "106"]

# 只画这些关节基名；None 表示从 opensim_settings.muscle_analysis_coordinates 自动合并
# 例如 muscle_analysis_coordinates = ["knee_angle_l", "knee_angle_r"]
# 会自动合并为 joint_base = "knee_angle"
JOINT_BASES_TO_PLOT = None
# JOINT_BASES_TO_PLOT = ["knee_angle", "hip_flexion", "ankle_angle"]

# 每行子图数量
N_COLS = 4

# 如果某个 EMG 标签对应多个 OpenSim 肌肉，是否显示每个分束的浅色散点
SHOW_COMPONENT_POINTS = True

# 散点大小与透明度
SCATTER_SIZE = 7
SCATTER_ALPHA = 0.35
COMPONENT_ALPHA = 0.12

# 若某条肌肉力臂整体最大绝对值小于该阈值，则认为该肌肉不跨该关节并整体忽略。
# 注意：不会删除曲线中局部接近 0 的正常点。
MOMENT_ARM_EPS = 0.001

# 拟合方法：
#   "poly"    多项式拟合
#   "fourier" 傅里叶级数拟合
FIT_METHOD = "poly"

# 多项式阶数；会根据有效点数量自动降低
POLY_DEGREE = 6

# 傅里叶阶数；会根据有效点数量自动降低
FOURIER_ORDER = 3

# 每个拟合曲线最少有效点数
MIN_FIT_POINTS = 12

# 拟合曲线采样点数
FIT_N_POINTS = 250

# 是否将所有 load 合并后为每个肌肉拟合一条总曲线
FIT_COMBINED_LOADS = True

# 是否额外为每个 load 单独拟合曲线
FIT_EACH_LOAD = False

# 输出图片 DPI
DPI = 200


# ============================================================
#  路径与基础工具
# ============================================================

def get_base_dir():
    """本文件位于 examples/opensim/，项目根目录为向上两级。"""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


def resolve_optional_extension_path(path):
    """
    解析 OpenSim 输出文件路径。

    有些 OpenSim / MuscleAnalysis 输出文件可能没有 .sto/.mot 后缀，
    但文件内容格式完全一样。因此读取时按以下顺序尝试：
      1. 原路径
      2. 去掉后缀后的路径
      3. .sto
      4. .mot
    """
    if path is None:
        return None

    root, ext = os.path.splitext(path)
    candidates = [path]
    if ext:
        candidates.append(root)
    candidates.extend([root + '.sto', root + '.mot'])

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p
    return None


def read_sto_or_mot(path):
    """
    读取 OpenSim .sto / .mot 文件为 DataFrame。

    兼容：
      - 有 endheader 的文件
      - tab 或空格分隔
      - 文件缺少 .sto/.mot 后缀，但内容格式相同
    """
    resolved_path = resolve_optional_extension_path(path)
    if resolved_path is None:
        return None

    with open(resolved_path, 'r', encoding='utf-8', errors='ignore') as f:
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
        raise ValueError(f'无法识别表头: {resolved_path}')

    from io import StringIO
    table_text = ''.join(lines[header_start:])
    return pd.read_csv(StringIO(table_text), sep=r'\s+', engine='python')


def flatten_muscle_item(item):
    """
    支持 muscle_analysis_muscles 中每一项为：
      "vas_lat_r"
      ["glut_max1_r", "glut_max2_r", "glut_max3_r"]
    """
    if item is None:
        return []
    if isinstance(item, (list, tuple)):
        return [str(x) for x in item if x]
    return [str(item)]


def infer_side_from_label(label):
    """
    从 EMG 标签推断左右侧：
      LTA -> l
      RTA -> r
    """
    if not label:
        return None
    first = label[0].upper()
    if first == 'L':
        return 'l'
    if first == 'R':
        return 'r'
    return None


def get_target_muscle_pairs(config):
    """
    返回 [(emg_label, side, [opensim_muscle_names...]), ...]

    要求：
      len(emg_settings.musc_label) == len(opensim_settings.muscle_analysis_muscles)
    """
    musc_labels = config.get('emg_settings', {}).get('musc_label', [])
    osim_cfg = config.get('opensim_settings', {})
    ma_muscles = (
        osim_cfg.get('muscle_analysis_muscles')
        or config.get('muscle_analysis_muscles')
    )

    if not musc_labels:
        raise ValueError("JSON 中缺少 emg_settings.musc_label")
    if not ma_muscles:
        raise ValueError("JSON 中缺少 opensim_settings.muscle_analysis_muscles")

    if len(musc_labels) != len(ma_muscles):
        raise ValueError(
            "musc_label 与 muscle_analysis_muscles 数量不一致："
            f"{len(musc_labels)} vs {len(ma_muscles)}"
        )

    pairs = []
    for label, muscle_item in zip(musc_labels, ma_muscles):
        side = infer_side_from_label(label)
        muscles = flatten_muscle_item(muscle_item)
        if side is None:
            print(f"[WARN] 无法从 EMG 标签判断左右侧，跳过: {label}")
            continue
        if muscles:
            pairs.append((label, side, muscles))
    return pairs


def get_load_keys(config):
    if LOAD_KEYS is None:
        return list(config['modeling_file']['data'].keys())
    return [str(k) for k in LOAD_KEYS]


def get_mot_path(config, base_dir, load_key):
    """
    根据 config 中的 xsens_file 构造 Step 1 生成的 .mot 路径。
    """
    experiment_label = config['experiment_label']
    file_info = config['modeling_file']['data'].get(str(load_key))
    if file_info is None:
        return None
    xsens_file = file_info.get('xsens_file')
    if not xsens_file:
        return None

    mot_name = os.path.splitext(os.path.basename(xsens_file))[0] + '_opensim.mot'
    return os.path.join(
        base_dir, 'result', experiment_label,
        'opensim', 'mot', mot_name
    )


def get_moment_arm_path(config, base_dir, load_key, coord):
    """
    构造 MuscleAnalysis MomentArm 文件路径。

    默认文件名带 .sto；若实际生成文件没有后缀，read_sto_or_mot()
    会自动尝试无后缀版本。
    """
    experiment_label = config['experiment_label']
    return os.path.join(
        base_dir, 'result', experiment_label,
        'opensim', 'muscle_analysis', str(load_key),
        f'{experiment_label}_{load_key}_MuscleAnalysis_MomentArm_{coord}.sto'
    )


# ============================================================
#  关节基名分组
# ============================================================

def split_coord_side(coord):
    """
    将 knee_angle_l / knee_angle_r 分成：
      ("knee_angle", "l")
      ("knee_angle", "r")

    非左右坐标，例如 flex_extension：
      ("flex_extension", None)
    """
    if coord.endswith('_l'):
        return coord[:-2], 'l'
    if coord.endswith('_r'):
        return coord[:-2], 'r'
    return coord, None


def build_joint_groups(config):
    """
    从 muscle_analysis_coordinates 构建关节分组。

    Returns
    -------
    groups : dict
      joint_base -> {"l": coord_l, "r": coord_r, None: coord_no_side}
    """
    coords = config.get('opensim_settings', {}).get(
        'muscle_analysis_coordinates', []
    )

    groups = {}
    for coord in coords:
        base, side = split_coord_side(coord)
        groups.setdefault(base, {})
        groups[base][side] = coord

    if JOINT_BASES_TO_PLOT is not None:
        groups = {b: groups.get(b, {}) for b in JOINT_BASES_TO_PLOT}

    return groups


def coord_for_label(joint_group, side):
    """
    对于左右关节：
      L 肌肉 -> *_l
      R 肌肉 -> *_r

    对于非左右关节：
      使用 None 对应的 coord。
    """
    if side in joint_group:
        return joint_group[side]
    if None in joint_group:
        return joint_group[None]
    return None


# ============================================================
#  数据收集
# ============================================================

def collect_joint_base_data(config, base_dir, joint_base, joint_group,
                            muscle_pairs, load_keys):
    """
    收集一个关节基名下所有目标肌肉的 angle - moment arm 数据。

    例如 joint_base = knee_angle：
      L 肌肉读取 knee_angle_l
      R 肌肉读取 knee_angle_r

    Returns
    -------
    plot_data : dict
      emg_label -> list[dict]
    """
    plot_data = {label: [] for label, _, _ in muscle_pairs}

    for load_key in load_keys:
        mot_path = get_mot_path(config, base_dir, load_key)
        mot_df = read_sto_or_mot(mot_path)

        print(f'\n[{joint_base}] load={load_key}')
        if mot_df is None:
            print(f'  [MISS] mot: {mot_path}')
            continue
        if 'time' not in mot_df.columns:
            print(f'  [WARN] .mot 中缺少 time: {mot_path}')
            continue

        # 同一 load 内缓存不同 coord 的 MomentArm 文件
        ma_cache = {}

        for label, side, muscles in muscle_pairs:
            coord = coord_for_label(joint_group, side)
            if coord is None:
                continue

            if coord not in mot_df.columns:
                print(f'  [WARN] .mot 中缺少 {coord}: {mot_path}')
                continue

            if coord not in ma_cache:
                ma_path = get_moment_arm_path(config, base_dir, load_key, coord)
                ma_cache[coord] = read_sto_or_mot(ma_path)
                if ma_cache[coord] is None:
                    print(f'  [MISS] moment arm: {ma_path}')

            ma_df = ma_cache[coord]
            if ma_df is None or 'time' not in ma_df.columns:
                continue

            cols = [m for m in muscles if m in ma_df.columns]
            if not cols:
                # 该肌肉不跨该关节时，MomentArm_{coord}.sto 中不会有对应列
                continue

            mot_t = mot_df['time'].values.astype(float)
            angle = mot_df[coord].values.astype(float)

            ma_t = ma_df['time'].values.astype(float)
            angle_on_ma = np.interp(
                ma_t, mot_t, angle,
                left=angle[0], right=angle[-1]
            )

            comp = ma_df[cols].values.astype(float)
            moment_arm = np.nanmean(comp, axis=1)

            valid = np.isfinite(angle_on_ma) & np.isfinite(moment_arm)
            if valid.sum() == 0:
                continue

            # 如果该肌肉在该关节上的力臂整体都非常小，说明基本不跨该关节，整体忽略。
            # 如果只是曲线局部经过 0 或小于阈值，则保留这些点。
            if np.nanmax(np.abs(moment_arm[valid])) < MOMENT_ARM_EPS:
                continue

            plot_data[label].append({
                'load_key': str(load_key),
                'side': side,
                'coord': coord,
                'angle': angle_on_ma[valid],
                'moment_arm': moment_arm[valid],
                'component_cols': cols,
                'component_values': comp[valid, :],
            })

            print(f'  [OK] {label:9s} uses {coord:14s} -> cols={cols}, n={valid.sum()}')

    return plot_data


# ============================================================
#  拟合工具
# ============================================================

def _prepare_fit_xy(x, y):
    """
    清理拟合数据：
      - 去掉 nan / inf
      - 若整体力臂范围过小则忽略
      - 按 x 排序
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return x, y

    # 只在整条曲线都接近 0 时忽略；不要删除局部过零点。
    if np.nanmax(np.abs(y)) < MOMENT_ARM_EPS:
        return np.array([]), np.array([])

    order = np.argsort(x)
    return x[order], y[order]


def fit_polynomial_curve(x, y, degree=POLY_DEGREE, n_points=FIT_N_POINTS):
    """
    多项式拟合 y = p(x)。

    返回:
      x_fit, y_fit, label
    """
    x, y = _prepare_fit_xy(x, y)
    if len(x) < MIN_FIT_POINTS:
        return None

    deg = min(int(degree), len(x) - 1)
    if deg < 1:
        return None

    coeff = np.polyfit(x, y, deg=deg)
    x_fit = np.linspace(np.min(x), np.max(x), n_points)
    y_fit = np.polyval(coeff, x_fit)
    return x_fit, y_fit, f'poly{deg}'


def _fourier_design_matrix(x_scaled, order):
    """
    构建傅里叶级数设计矩阵：
      y = a0 + Σ [ak cos(kx) + bk sin(kx)]

    x_scaled 应为 [0, 2π]。
    """
    cols = [np.ones_like(x_scaled)]
    for k in range(1, order + 1):
        cols.append(np.cos(k * x_scaled))
        cols.append(np.sin(k * x_scaled))
    return np.column_stack(cols)


def fit_fourier_curve(x, y, order=FOURIER_ORDER, n_points=FIT_N_POINTS):
    """
    傅里叶级数拟合。

    注意：这里将当前关节角范围线性映射到 [0, 2π]，
    用作经验平滑拟合；并不表示关节角本身一定是周期变量。
    """
    x, y = _prepare_fit_xy(x, y)
    if len(x) < MIN_FIT_POINTS:
        return None

    x_min, x_max = float(np.min(x)), float(np.max(x))
    if abs(x_max - x_min) < 1e-9:
        return None

    # 参数数量 = 1 + 2 * order，要求点数更多，否则降低阶数
    max_order = max(1, (len(x) - 2) // 2)
    order = min(int(order), max_order)
    if order < 1:
        return None

    x_scaled = (x - x_min) / (x_max - x_min) * 2 * np.pi
    A = _fourier_design_matrix(x_scaled, order)
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)

    x_fit = np.linspace(x_min, x_max, n_points)
    x_fit_scaled = (x_fit - x_min) / (x_max - x_min) * 2 * np.pi
    y_fit = _fourier_design_matrix(x_fit_scaled, order) @ coeff
    return x_fit, y_fit, f'fourier{order}'


def fit_curve(x, y):
    """
    根据 FIT_METHOD 选择拟合方式。
    """
    method = FIT_METHOD.lower()
    if method == 'poly':
        return fit_polynomial_curve(x, y)
    if method == 'fourier':
        return fit_fourier_curve(x, y)
    raise ValueError(f'未知 FIT_METHOD: {FIT_METHOD}')


def collect_xy_from_entries(entries):
    """
    将某个肌肉的所有 load 数据合并为 x/y。
    """
    xs, ys = [], []
    for e in entries:
        xs.append(e['angle'])
        ys.append(e['moment_arm'])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


# ============================================================
#  绘图
# ============================================================

def plot_joint_base_moment_arm(joint_base, plot_data, output_path=None):
    """
    对一个关节基名画一张图：
      - knee_angle 一张图
      - L 肌肉使用 knee_angle_l 的角度和力臂
      - R 肌肉使用 knee_angle_r 的角度和力臂
      - 使用散点图
      - 叠加多项式或傅里叶拟合曲线
    """
    labels = [k for k, v in plot_data.items() if len(v) > 0]
    if not labels:
        print(f'[WARN] {joint_base}: 没有可绘制的数据')
        return None

    n = len(labels)
    n_rows = math.ceil(n / N_COLS)

    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(N_COLS * 4.2, n_rows * 3.2),
        squeeze=False
    )
    fig.suptitle(
        f'Muscle Moment Arm vs Joint Angle: {joint_base}',
        fontsize=15,
        fontweight='bold'
    )

    # 固定不同 load 的颜色
    all_loads = []
    for entries in plot_data.values():
        for e in entries:
            if e['load_key'] not in all_loads:
                all_loads.append(e['load_key'])
    try:
        all_loads = sorted(all_loads, key=lambda x: float(x))
    except Exception:
        all_loads = sorted(all_loads)

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {
        load: prop_cycle[i % len(prop_cycle)]
        for i, load in enumerate(all_loads)
    }

    # 左右侧用不同 marker；颜色仍表示 load
    marker_map = {'l': 'o', 'r': '^', None: 's'}

    for idx, label in enumerate(labels):
        r, c = divmod(idx, N_COLS)
        ax = axes[r][c]

        entries = plot_data[label]

        for e in entries:
            color = color_map[e['load_key']]
            marker = marker_map.get(e['side'], 'o')
            x = e['angle']
            y = e['moment_arm']

            if SHOW_COMPONENT_POINTS and e['component_values'].shape[1] > 1:
                comp = e['component_values']
                for j in range(comp.shape[1]):
                    comp_y = comp[:, j]
                    comp_valid = np.isfinite(x) & np.isfinite(comp_y)
                    # 分束也只在整体都接近 0 时忽略；保留局部过零点。
                    if comp_valid.sum() > 0 and np.nanmax(np.abs(comp_y[comp_valid])) >= MOMENT_ARM_EPS:
                        ax.scatter(
                            x[comp_valid], comp_y[comp_valid],
                            s=max(2, SCATTER_SIZE * 0.55),
                            color=color,
                            marker=marker,
                            alpha=COMPONENT_ALPHA,
                            linewidths=0
                        )

            ax.scatter(
                x, y,
                s=SCATTER_SIZE,
                color=color,
                marker=marker,
                alpha=SCATTER_ALPHA,
                linewidths=0,
                label=f"{e['load_key']} kg ({e['coord']})"
            )

            # 可选：每个 load 单独拟合
            if FIT_EACH_LOAD:
                fit = fit_curve(x, y)
                if fit is not None:
                    xf, yf, fit_label = fit
                    ax.plot(
                        xf, yf,
                        color=color,
                        linewidth=1.2,
                        alpha=0.75,
                        linestyle='--',
                        label=f"{e['load_key']} {fit_label}"
                    )

        # 默认：合并所有 load 后拟合一条黑色曲线
        if FIT_COMBINED_LOADS:
            x_all, y_all = collect_xy_from_entries(entries)
            fit = fit_curve(x_all, y_all)
            if fit is not None:
                xf, yf, fit_label = fit
                ax.plot(
                    xf, yf,
                    color='black',
                    linewidth=2.0,
                    alpha=0.9,
                    linestyle='-',
                    label=f'all loads {fit_label}'
                )

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel(f'{joint_base} angle (deg)', fontsize=8)
        ax.set_ylabel('Moment arm (m)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # 避免每个子图 legend 太拥挤，只在第一个子图显示
        if idx == 0:
            handles, labels_legend = ax.get_legend_handles_labels()
            by_label = dict(zip(labels_legend, handles))
            ax.legend(by_label.values(), by_label.keys(),
                      fontsize=6, loc='best', markerscale=1.8)

    # 隐藏多余子图
    for idx in range(n, n_rows * N_COLS):
        r, c = divmod(idx, N_COLS)
        axes[r][c].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f'[Saved] {output_path}')

    return fig


# ============================================================
#  主程序
# ============================================================

def main():
    base_dir = get_base_dir()
    config_path = get_config_path()

    print(f'配置文件: {config_path}')
    print(f'基准目录: {base_dir}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    muscle_pairs = get_target_muscle_pairs(config)
    print('\n目标肌肉：')
    for label, side, muscles in muscle_pairs:
        print(f'  {label:9s} ({side}) -> {muscles}')

    joint_groups = build_joint_groups(config)
    if not joint_groups:
        raise ValueError('没有可绘制关节；请设置 opensim_settings.muscle_analysis_coordinates')

    print('\n关节分组：')
    for joint_base, group in joint_groups.items():
        print(f'  {joint_base}: {group}')

    load_keys = get_load_keys(config)

    experiment_label = config['experiment_label']
    out_dir = os.path.join(
        base_dir, 'result', experiment_label, 'opensim/moment_arm_angle'
    )

    for joint_base, joint_group in joint_groups.items():
        plot_data = collect_joint_base_data(
            config=config,
            base_dir=base_dir,
            joint_base=joint_base,
            joint_group=joint_group,
            muscle_pairs=muscle_pairs,
            load_keys=load_keys,
        )

        out_fig = os.path.join(out_dir, f'moment_arm_vs_{joint_base}.png')
        plot_joint_base_moment_arm(joint_base, plot_data, output_path=out_fig)

    plt.show()


if __name__ == '__main__':
    main()