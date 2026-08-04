"""
疲劳实验（fatigue_file）机器人数据分析

数据来源：
    配置文件 fatigue_file.fatigue_data 中的三组数据（vl / cl / el），
    每组含 robot_file（+ emg_file，本脚本只用机器人数据）。
    机器人文件位于 folder/fatigue_file.robot_folder 目录下。

绘图内容：
    图1  原始信号大图：分三个子图（每组一个），画机器人记录的
         力（force_l + force_r）、速度（vel_l）、位置（pos_l），
         三者各自做 [0,1] 归一化后画在同一坐标系，从而画在一个图上。
    图2  切片数据大图：对每组机器人数据做运动切片
         （DataAligner.cut_aligned_data），同样分三个子图画归一化后的
         力 / 速度 / 位置（散点图，只保留有效运动片段）。
    图3  切片数据的功率累加曲线：power = |force_total| * |vel_l|，
         按时间累加 power*dt（≈累计功，只累计上升阶段 vel>0），横轴为时间，三组画在同一图。
    图4  总功 / rep 数柱状对比图：左为每组累计功最终值，右为完成的 rep 数。
    图5  切片数据的平均交互力 / 平均速度（速率）柱状对比图。
    图6  rep 数 - 各 rep 上升阶段平均速度折线图（三组）。
    图7  rep 数 - 速度丢失率折线图：(v_max - v_rep)/v_max（v_max 为该组最高平均速度）。
    图8  速度丢失率 - 做完该 rep 的累积功折线图（三组）。
    图9  两个子图：左=高度-负载曲线（三条，无散点，来自各组 load_file：
         第一行=半负载*2，第二行=高度-0.7+0.7）；右=高度-交互力散点图
         （每组真实高度 vs force_total）；每组两图同色。

说明：
    - power = |force| * |velocity|（力与速度均取绝对值）；累积功仅累计上升阶段（vel>0）。
    - 切片数据时间不连续，功率累加时对异常/间断的 dt 用中位数 dt 修正。

用法：
    python example_fatigue_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from digitaltwin import Subject
from digitaltwin.data.robot_processor import RobotProcessor
from digitaltwin.analysis.alignment import DataAligner

# ============================================================
#  选项
# ============================================================
CONFIG_FILE = '../config/20260513_squat_FTS09.json'
FORCE_COLUMNS = ['force_l', 'force_r']   # 相加得到总力 (N)
VELOCITY_COLUMN = 'vel_l'                # 速度列 (m/s)
POSITION_COLUMN = 'pos_l'                # 位置列 (m)
GROUP_COLORS = plt.cm.tab10.colors


def _normalize(arr):
    """min-max 归一化到 [0,1]；常数或全 nan 时返回全 0。"""
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _add_force_total(df):
    """添加 force_total = force_l + force_r 列。"""
    avail = [c for c in FORCE_COLUMNS if c in df.columns]
    if avail:
        df = df.copy()
        df['force_total'] = df[avail].sum(axis=1)
    return df


def _load_fatigue_groups(subject):
    """读取 fatigue_file 三组机器人数据，返回 [(label, robot_df), ...]。"""
    fatigue_cfg = subject.config.get('fatigue_file', {})
    fatigue_data = fatigue_cfg.get('fatigue_data', {})
    robot_subfolder = fatigue_cfg.get('robot_folder', 'fatigue/robot')
    robot_folder = os.path.join(subject.folder, robot_subfolder)

    groups = []
    for label, info in fatigue_data.items():
        robot_file = info.get('robot_file', '')
        # load_weight 在疲劳实验中无意义，传 0 仅占位
        # 疲劳数据固定 turn_position=False（其他实验仍用默认 subject.turn_position）
        robot_df = RobotProcessor.process(
            robot_file, 0, robot_folder, subject.folder,
            turn_position=False)
        if robot_df is None:
            print(f'[{label}] 机器人数据加载失败，跳过。')
            continue
        if not any(c in robot_df.columns for c in FORCE_COLUMNS):
            print(f'[{label}] 缺少力列 {FORCE_COLUMNS}，跳过。')
            continue
        groups.append((label, _add_force_total(robot_df)))
    return groups


def _cut_group(robot_df):
    """对单组机器人数据做运动切片，返回切片后的 DataFrame 或 None。"""
    aligner = DataAligner()
    try:
        cutted = aligner.cut_aligned_data(robot_df)
    except Exception as e:
        print(f'切片失败: {e}')
        return None
    if cutted is None:
        return None
    if isinstance(cutted, list):
        if len(cutted) == 0:
            return None
        cutted = pd.concat(cutted, ignore_index=True)
    if len(cutted) == 0:
        return None
    return _add_force_total(cutted)


def _cumulative_work(df):
    """计算 power = |force_total| * |vel| 的按时间累计（power*dt），只累计上升阶段（vel>0）。
    返回 (elapsed_time, cumulative_work)。
    """
    t = np.asarray(df['time'].values, dtype=float)
    dt = np.diff(t, prepend=t[0])
    pos_dt = dt[dt > 0]
    med = np.median(pos_dt) if pos_dt.size else 0.01
    if not np.isfinite(med) or med <= 0:
        med = 0.01
    # 修正切片造成的时间间断（非正或异常大的 dt 用中位数代替）
    dt = np.where((dt <= 0) | (dt > 5 * med), med, dt)
    vel_raw = np.asarray(df[VELOCITY_COLUMN].values, dtype=float)
    force = np.abs(np.asarray(df['force_total'].values, dtype=float))
    vel = np.abs(vel_raw)
    power = force * vel
    dW = power * dt
    # 只累计上升阶段（vel > 0）的功
    dW = np.where(vel_raw > 0, dW, 0.0)
    return np.cumsum(dt), np.cumsum(dW)


def _draw_normalized_signals(groups, out_path, title, scatter=False):
    """分三个子图（每组一个），画归一化后的 力 / 速度 / 位置。"""
    n = len(groups)
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), squeeze=False)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    for i, (label, df) in enumerate(groups):
        ax = axes[i][0]
        t = np.asarray(df['time'].values, dtype=float)
        t = t - t[0]
        series = [
            ('Force (L+R)', _normalize(df['force_total'].values), 'tab:red'),
            ('Velocity', _normalize(df[VELOCITY_COLUMN].values), 'tab:blue'),
            ('Position', _normalize(df[POSITION_COLUMN].values), 'tab:green'),
        ]
        for name, yv, col in series:
            if scatter:
                ax.scatter(t, yv, s=8, alpha=0.5, color=col, label=name)
            else:
                ax.plot(t, yv, color=col, lw=1.2, label=name)
        ax.set_title(str(label))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized [0,1]')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9, loc='upper right', ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _draw_cumulative_power(cut_groups, out_path):
    """三组切片数据的功率累计曲线画在同一图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Fatigue: Cumulative Power (|Force| * |Velocity|)',
                 fontsize=14, fontweight='bold')
    for i, (label, df) in enumerate(cut_groups):
        elapsed, cumW = _cumulative_work(df)
        ax.plot(elapsed, cumW, lw=1.6,
                color=GROUP_COLORS[i % len(GROUP_COLORS)], label=str(label))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative |Force|*|Velocity|*dt  (approx. work, J)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _draw_total_work(cut_groups, out_path):
    """总功（累计功最终值）与 rep 数柱状对比图。"""
    labels, totals, reps = [], [], []
    for label, df in cut_groups:
        _, cumW = _cumulative_work(df)
        labels.append(str(label))
        totals.append(float(cumW[-1]) if len(cumW) else 0.0)
        reps.append(len(_get_rep_groups(df)))
    colors = [GROUP_COLORS[i % len(GROUP_COLORS)] for i in range(len(labels))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Fatigue: Total Work / Rep Count',
                 fontsize=14, fontweight='bold')
    # 总功
    bars0 = axes[0].bar(labels, totals, color=colors, alpha=0.85)
    for b, v in zip(bars0, totals):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}',
                     ha='center', va='bottom' if v >= 0 else 'top',
                     fontsize=10)
    axes[0].set_ylabel('Total |Force|*|Velocity|*dt  (approx. work, J)')
    axes[0].set_xlabel('Group')
    axes[0].set_title('Total Work')
    axes[0].grid(True, axis='y', alpha=0.3)
    # rep 数
    bars1 = axes[1].bar(labels, reps, color=colors, alpha=0.85)
    for b, v in zip(bars1, reps):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f'{int(v)}',
                     ha='center', va='bottom', fontsize=10)
    axes[1].set_ylabel('Number of reps')
    axes[1].set_xlabel('Group')
    axes[1].set_title('Rep Count')
    axes[1].grid(True, axis='y', alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _draw_mean_bars(cut_groups, out_path):
    """切片数据的平均交互力 / 平均速度柱状对比图（每组一根柱）。"""
    labels, force_means, vel_means = [], [], []
    for label, df in cut_groups:
        labels.append(str(label))
        # 平均交互力 = force_l + force_r 的均值
        force_means.append(float(np.mean(df['force_total'].values)))
        # 速度取绝对值后求平均（平均速率），避免上下行相互抵消
        vel_means.append(float(np.mean(np.abs(df[VELOCITY_COLUMN].values))))
    colors = [GROUP_COLORS[i % len(GROUP_COLORS)] for i in range(len(labels))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Fatigue (cut data): Mean Interaction Force / Mean Speed',
                 fontsize=14, fontweight='bold')
    # 平均交互力
    bars0 = axes[0].bar(labels, force_means, color=colors, alpha=0.85)
    for b, v in zip(bars0, force_means):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}',
                     ha='center', va='bottom' if v >= 0 else 'top',
                     fontsize=10)
    axes[0].set_ylabel('Mean interaction force (L+R) [N]')
    axes[0].set_xlabel('Group')
    axes[0].set_title('Mean Interaction Force')
    axes[0].grid(True, axis='y', alpha=0.3)
    # 平均速度（速率）
    bars1 = axes[1].bar(labels, vel_means, color=colors, alpha=0.85)
    for b, v in zip(bars1, vel_means):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=10)
    axes[1].set_ylabel('Mean speed |vel_l| [m/s]')
    axes[1].set_xlabel('Group')
    axes[1].set_title('Mean Speed')
    axes[1].grid(True, axis='y', alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _get_rep_groups(df):
    """返回 [(rep_index, rep_df), ...]，按时间排序。
    优先按 cycle_id 分组（一个完整周期=一个 rep）；
    若无 cycle_id，则把每个 upward segment 当作一个 rep。"""
    if 'cycle_id' in df.columns:
        ids = df['cycle_id'].unique()
        reps = [df[df['cycle_id'] == i] for i in ids]
    elif 'segment_id' in df.columns and 'movement_type' in df.columns:
        up = df[df['movement_type'] == 'upward']
        ids = up['segment_id'].unique()
        reps = [up[up['segment_id'] == i] for i in ids]
    else:
        return []
    reps = [r for r in reps if len(r) > 0]
    reps.sort(key=lambda r: float(np.min(r['time'].values)))
    return list(enumerate(reps))


def _rep_metrics(df):
    """按 rep 计算指标。返回 dict 或 None。
        time        : 每个 rep 的起始时间（相对全局起点，s）
        up_mean_vel : 每个 rep 上升阶段(upward)的平均速度（速率）
        rep_work    : 每个 rep 的做功 (|force|*|vel|*dt 之和)
        cum_work    : 做完该 rep 后的累积功
        vloss       : 速度丢失率 = (v_ref - v) / v_ref，v_ref 为该组最高平均速度
    """
    rep_groups = _get_rep_groups(df)
    if not rep_groups:
        return None
    t0 = float(np.min(df['time'].values))
    times, up_vels, works = [], [], []
    for _, rep in rep_groups:
        times.append(float(np.min(rep['time'].values)) - t0)
        # 上升阶段平均速度（速率）
        if 'movement_type' in rep.columns:
            up = rep[rep['movement_type'] == 'upward']
        else:
            up = rep
        if len(up) == 0:
            up = rep
        up_vels.append(float(np.mean(np.abs(up[VELOCITY_COLUMN].values))))
        # rep 做功 = |force|*|vel|*dt 之和
        rt = np.asarray(rep['time'].values, dtype=float)
        dt = np.diff(rt, prepend=rt[0])
        pos_dt = dt[dt > 0]
        med = np.median(pos_dt) if pos_dt.size else 0.01
        if not np.isfinite(med) or med <= 0:
            med = 0.01
        dt = np.where((dt <= 0) | (dt > 5 * med), med, dt)
        v_raw = np.asarray(rep[VELOCITY_COLUMN].values, dtype=float)
        f = np.abs(np.asarray(rep['force_total'].values, dtype=float))
        v = np.abs(v_raw)
        dW = f * v * dt
        # 只累计上升阶段（vel > 0）的功
        works.append(float(np.sum(np.where(v_raw > 0, dW, 0.0))))
    times = np.asarray(times)
    up_vels = np.asarray(up_vels)
    works = np.asarray(works)
    order = np.argsort(times)
    times, up_vels, works = times[order], up_vels[order], works[order]
    cum_work = np.cumsum(works)
    # 分母使用该组中上升阶段平均速度最高的 rep
    v_ref = float(np.max(up_vels)) if len(up_vels) else np.nan
    if np.isfinite(v_ref) and v_ref != 0:
        vloss = (v_ref - up_vels) / v_ref
    else:
        vloss = np.zeros_like(up_vels)
    return {'time': times, 'up_mean_vel': up_vels, 'rep_work': works,
            'cum_work': cum_work, 'vloss': vloss}


def _draw_rep_velocity(rep_list, out_path):
    """各 rep 上升阶段平均速度 vs rep 数（三组）。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Fatigue: Rep Ascending Mean Velocity vs Rep',
                 fontsize=14, fontweight='bold')
    for i, (label, m) in enumerate(rep_list):
        reps = np.arange(1, len(m['up_mean_vel']) + 1)
        ax.plot(reps, m['up_mean_vel'], marker='o', lw=1.6,
                color=GROUP_COLORS[i % len(GROUP_COLORS)], label=str(label))
    ax.set_xlabel('Rep number')
    ax.set_ylabel('Ascending-phase mean velocity [m/s]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _draw_velocity_loss(rep_list, out_path):
    """速度丢失率 vs rep 数（三组）。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Fatigue: Velocity Loss vs Rep',
                 fontsize=14, fontweight='bold')
    for i, (label, m) in enumerate(rep_list):
        reps = np.arange(1, len(m['vloss']) + 1)
        ax.plot(reps, m['vloss'] * 100.0, marker='o', lw=1.6,
                color=GROUP_COLORS[i % len(GROUP_COLORS)], label=str(label))
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Rep number')
    ax.set_ylabel('Velocity loss vs fastest rep [%]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _draw_work_vs_vloss(rep_list, out_path):
    """累积功 vs 速度丢失率（三组）。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Fatigue: Cumulative Work vs Velocity Loss',
                 fontsize=14, fontweight='bold')
    for i, (label, m) in enumerate(rep_list):
        ax.plot(m['vloss'] * 100.0, m['cum_work'], marker='o', lw=1.6,
                color=GROUP_COLORS[i % len(GROUP_COLORS)], label=str(label))
    ax.set_xlabel('Velocity loss vs fastest rep [%]')
    ax.set_ylabel('Cumulative work after rep  (approx., J)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _load_height_load_profiles(subject):
    """读取每组 load_file，返回 [(label, height, load), ...]。
    文件第一行=一半负载（需 *2），第二行=高度-0.7（需 +0.7）。"""
    fatigue_cfg = subject.config.get('fatigue_file', {})
    fatigue_data = fatigue_cfg.get('fatigue_data', {})
    robot_subfolder = fatigue_cfg.get('robot_folder', 'fatigue/robot')
    robot_folder = os.path.join(subject.folder, robot_subfolder)
    fatigue_root = os.path.join(
        subject.folder, os.path.dirname(robot_subfolder.rstrip('/')))
    profiles = []
    for label, info in fatigue_data.items():
        load_file = info.get('load_file', '')
        if not load_file:
            continue
        # 候选路径：robot 目录 / fatigue 根目录 / 受试者根目录
        candidates = [
            os.path.join(robot_folder, load_file),
            os.path.join(fatigue_root, load_file),
            os.path.join(subject.folder, load_file),
        ]
        path = next((p for p in candidates if os.path.exists(p)),
                    candidates[0])
        try:
            raw = pd.read_csv(path, header=None)
        except Exception as e:
            print(f'[{label}] load_file 读取失败({path}): {e}')
            continue
        arr = raw.apply(pd.to_numeric, errors='coerce').values
        if arr.shape[0] < 2:
            print(f'[{label}] load_file 行数不足，跳过。')
            continue
        load = np.asarray(arr[0], dtype=float) * 2.0     # 第一行：一半负载
        height = np.asarray(arr[1], dtype=float) + 0.7   # 第二行：高度-0.7
        profiles.append((label, height, load))
    return profiles


def _lighten(color, frac=0.4):
    """把颜色向白色混合，frac 越大越浅。"""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def _draw_height_load(profiles, out_path, real_points=None):
    """两个子图：左=高度-负载曲线（每组一条，无散点）；
    右=高度-交互力散点图（每组真实高度 vs force_total）。
    每组在两个子图中使用相同颜色。"""
    fig, (ax_curve, ax_scatter) = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle('Fatigue: Height-Load Profile / Height-Interaction Force',
                 fontsize=14, fontweight='bold')
    # 先清洗并按高度排序
    cleaned = []
    for label, height, load in profiles:
        mask = np.isfinite(height) & np.isfinite(load)
        h, l = height[mask], load[mask]
        if len(h) == 0:
            continue
        order = np.argsort(h)
        cleaned.append((label, h[order], l[order]))
    if not cleaned:
        print('无有效高度-负载数据，跳过。')
        plt.close(fig)
        return None
    color_map = {}
    for i, (label, h, l) in enumerate(cleaned):
        col = GROUP_COLORS[i % len(GROUP_COLORS)]
        color_map[str(label)] = col
        ax_curve.plot(h, l, lw=1.6, color=col, label=str(label))
    ax_curve.set_xlabel('Height [m]')
    ax_curve.set_ylabel('Load')
    ax_curve.set_title('Height - Load Profile')
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(fontsize=10, loc='best')

    # 右图：高度-交互力散点（每组与左图同色）
    if real_points:
        for label, (rh, rf) in real_points.items():
            label = str(label)
            if label not in color_map:
                color_map[label] = GROUP_COLORS[
                    len(color_map) % len(GROUP_COLORS)]
            col = color_map[label]
            rh = np.asarray(rh, dtype=float)
            rf = np.asarray(rf, dtype=float)
            m = np.isfinite(rh) & np.isfinite(rf)
            if np.any(m):
                ax_scatter.scatter(rh[m], rf[m], s=12, alpha=0.5,
                                   color=col, edgecolors='none', label=label)
                print(f'[{label}] 高度-交互力散点: 高度 '
                      f'{rh[m].min():.3f} ~ {rh[m].max():.3f} m，'
                      f'共 {int(np.sum(m))} 点')
    ax_scatter.set_xlabel('Height [m]')
    ax_scatter.set_ylabel('Interaction force (L+R) [N]')
    ax_scatter.set_title('Height - Interaction Force')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(fontsize=10, loc='best')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def main():
    subject = Subject(CONFIG_FILE)

    # 1. 读取三组机器人数据
    groups = _load_fatigue_groups(subject)
    if not groups:
        print('未能加载任何 fatigue 数据组，退出。')
        return

    # 2. 运动切片
    cut_groups = []
    for label, df in groups:
        cutted = _cut_group(df)
        if cutted is None or len(cutted) == 0:
            print(f'[{label}] 切片为空，跳过。')
            continue
        cut_groups.append((label, cutted))

    # 输出目录
    out_dir = os.path.join(subject.result_folder, 'fatigue')
    os.makedirs(out_dir, exist_ok=True)

    # 图1：原始信号（归一化）
    _draw_normalized_signals(
        groups,
        os.path.join(out_dir, 'fatigue_raw_signals.png'),
        'Fatigue: Force(L+R) / Velocity / Position (normalized)')

    # 图2-4：基于切片数据
    if cut_groups:
        _draw_normalized_signals(
            cut_groups,
            os.path.join(out_dir, 'fatigue_cut_signals.png'),
            'Fatigue (cut data): Force(L+R) / Velocity / Position (normalized)',
            scatter=True)
        _draw_cumulative_power(
            cut_groups,
            os.path.join(out_dir, 'fatigue_cumulative_power.png'))
        _draw_total_work(
            cut_groups,
            os.path.join(out_dir, 'fatigue_total_work.png'))
        _draw_mean_bars(
            cut_groups,
            os.path.join(out_dir, 'fatigue_mean_force_velocity.png'))
        # 逐 rep 指标（上升阶段平均速度、速度丢失率、累积功）
        rep_list = []
        for label, df in cut_groups:
            m = _rep_metrics(df)
            if m is not None and len(m['time']) > 0:
                rep_list.append((label, m))
        if rep_list:
            _draw_rep_velocity(
                rep_list,
                os.path.join(out_dir, 'fatigue_rep_velocity.png'))
            _draw_velocity_loss(
                rep_list,
                os.path.join(out_dir, 'fatigue_velocity_loss.png'))
            _draw_work_vs_vloss(
                rep_list,
                os.path.join(out_dir, 'fatigue_work_vs_vloss.png'))
        else:
            print('未能提取 rep 指标，跳过图6-8。')
    else:
        print('无有效切片数据，跳过图2-4。')

    # 图9：高度-负载图（来自各组 load_file）
    profiles = _load_height_load_profiles(subject)
    if profiles:
        # 各组真实(高度, 交互力)，仅上升阶段(vel>0)（切片数据优先，无则用原始）
        _src = cut_groups if cut_groups else groups
        real_points = {}
        for label, df in _src:
            up = df[df[VELOCITY_COLUMN] > 0]
            real_points[str(label)] = (
                np.asarray(up[POSITION_COLUMN].values, dtype=float),
                np.asarray(up['force_total'].values, dtype=float))
        _draw_height_load(
            profiles,
            os.path.join(out_dir, 'fatigue_height_load.png'),
            real_points=real_points)
    else:
        print('未能读取 load_file，跳过高度-负载图。')

    plt.show()


if __name__ == '__main__':
    main()