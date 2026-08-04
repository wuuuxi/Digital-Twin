"""
Isometric 数据：力变化阶段的 log(Force)-log(Activation) 图

流程：
  1. 从配置 modeling_file.isometric_data 读取每条等长(isometric)记录
     （每条含 robot_file + emg_file）。
  2. 加载机器人力/速度数据与 EMG，对齐并注入 EMG 特征。
  3. 按时间窗口截取点：TIME_WINDOW 手动给定 (start, end) 秒（相对记录起始时间）；
     设为 None 时自动检测“力开始变化 → 力停止变化”的阶段（平滑力导数阈值）。
  4. 计算 power = (force_l + force_r) * vel_l。
  5. 单独画一张 log10(Force) - log10(Activation) 散点大图，按肌肉分子图。

注意：横轴为总交互力 force_l+force_r；activation 采用归一化 EMG 包络
emg_<muscle>；force、activation 中非正值在取 log10 时会被丢弃。截取得到的
区间保留“力开始变化”到“力停止变化”之间的全部采样点。

用法：
    python example_isometric_logpower_logactivation.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.data.robot_processor import RobotProcessor
from digitaltwin.analysis.feature_injector import inject_emg_features


# ============================================================
#  选项
# ============================================================
CONFIG_FILE = '../config/20260513_squat_FTS09.json'
TARGET_MUSCLES = ['RVL', 'RVM', 'RGlutMax', 'RBF']

FORCE_COLUMNS = ['force_l', 'force_r']   # 相加得到总交互力 (N)
VELOCITY_COLUMN = 'vel_l'                # 速度列 (m/s)，power = force * 速度

# “力开始/停止变化”检测参数
SMOOTH_WINDOW_SEC = 0.10    # 力信号平滑窗口（秒）
DERIV_THRESH_FRAC = 0.05    # 导数阈值 = 平滑力导数绝对值最大值 * 该比例
PAD_SEC = 0.0               # 活动区间两侧额外保留（秒）

# 时间窗口（秒，相对每条记录的起始时间）。
# 设为 (start, end) 手动指定绘图区间，例如 (5.0, 10.0) 表示只画 5～10s 的点；
# 设为 None 时自动检测“力变化阶段”。
TIME_WINDOW = (10.0, 14.0)
# TIME_WINDOW = (10.0, 14.8)

POINT_SIZE = 8
ENTRY_COLORS = plt.cm.tab10.colors


def safe_log10(arr):
    """对非正元素返回 nan（不参与绘图），正元素取 log10。"""
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    pos = arr > 0
    out[pos] = np.log10(arr[pos])
    return out


def _moving_average(arr, win):
    arr = np.asarray(arr, dtype=float)
    win = max(1, int(win))
    if win <= 1 or win >= len(arr):
        return arr
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode='same')


def _detect_force_change_segment(force, time):
    """返回力“开始变化→停止变化”的索引区间 [start, end]。

    思路：平滑总力后取一阶导数，导数绝对值超过阈值的首末位置即为活动区间。
    """
    time = np.asarray(time, dtype=float)
    dt = np.median(np.diff(time)) if len(time) > 1 else 0.01
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.01
    win = int(round(SMOOTH_WINDOW_SEC / dt))

    f = _moving_average(force, win)
    dfdt = np.gradient(f, time)
    adf = np.abs(dfdt)
    if adf.size == 0 or np.nanmax(adf) <= 0:
        return 0, len(force) - 1

    thr = np.nanmax(adf) * DERIV_THRESH_FRAC
    active = np.where(adf > thr)[0]
    if active.size == 0:
        return 0, len(force) - 1

    pad = int(round(PAD_SEC / dt))
    start = max(0, int(active[0]) - pad)
    end = min(len(force) - 1, int(active[-1]) + pad)
    return start, end


def _process_isometric_entry(subject, emg_processor, aligner, label, info):
    """加载单条 isometric 记录 → 对齐 → 计算 power → 按时间窗口/力变化阶段截取。"""
    robot_file = info.get('robot_file', '')
    emg_file = info.get('emg_file', '')

    # 机器人数据（load_weight 在等长测试中无意义，传 0 仅占位）
    robot_data = RobotProcessor.process(
        robot_file, 0, subject.modeling_robot_folder, subject.folder,
        turn_position=subject.turn_position)
    if robot_data is None:
        print(f'[{label}] 机器人数据加载失败，跳过。')
        return None

    emg_data = emg_processor.process(
        emg_file, label, subject.modeling_emg_folder, subject.folder,
        motion_flag=subject.motion_flag,
        remove_leading_zeros=subject.remove_leading_zeros)
    if emg_data is None:
        print(f'[{label}] EMG 数据加载失败，跳过。')
        return None

    aligned = aligner.align_robot_emg(robot_data, emg_data)
    if aligned is None or 'time' not in aligned.columns:
        print(f'[{label}] 对齐失败，跳过。')
        return None
    aligned = inject_emg_features(aligned, emg_data, subject.emg_fs)

    avail = [c for c in FORCE_COLUMNS if c in aligned.columns]
    if not avail:
        print(f'[{label}] 缺少力列 {FORCE_COLUMNS}，跳过。')
        return None
    if VELOCITY_COLUMN not in aligned.columns:
        print(f'[{label}] 缺少速度列 {VELOCITY_COLUMN}，跳过。')
        return None

    aligned = aligned.copy()
    aligned['force_total'] = aligned[avail].sum(axis=1)
    aligned['power'] = aligned['force_total'] * aligned[VELOCITY_COLUMN]

    if TIME_WINDOW is not None:
        t = aligned['time'].values
        t_rel = t - t[0]  # 相对记录起始时间
        ws, we = TIME_WINDOW
        mask = (t_rel >= ws) & (t_rel <= we)
        seg = aligned.loc[mask].copy()
        print(f'[{label}] 手动时间窗口 {ws}~{we}s（相对起始），'
              f'{len(seg)} 个采样点')
    else:
        start, end = _detect_force_change_segment(
            aligned['force_total'].values, aligned['time'].values)
        seg = aligned.iloc[start:end + 1].copy()
        print(f'[{label}] 力变化阶段: 索引 {start}~{end}，'
              f'时长 {seg["time"].iloc[-1] - seg["time"].iloc[0]:.2f}s，'
              f'{len(seg)} 个采样点')
    return seg


def _fit_two_segments(x, y, n_break_grid=60):
    """连续两段线性拟合（单断点，在断点处连续）。

    模型: y = a + b*x + c*relu(x - xb)，断点前斜率为 b，断点后为 b+c。
    在 x 范围内网格搜索断点 xb，对每个 xb 用最小二乘解 (a,b,c)，
    取残差平方和最小者。返回 (xb, predict_fn) 或 None。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4 or np.ptp(x) <= 0:
        return None

    lo, hi = np.percentile(x, 10), np.percentile(x, 90)
    if hi <= lo:
        lo, hi = x.min(), x.max()
    candidates = np.linspace(lo, hi, n_break_grid)

    best = None
    for xb in candidates:
        hinge = np.maximum(0.0, x - xb)
        A = np.column_stack([np.ones_like(x), x, hinge])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        sse = float(resid @ resid)
        if best is None or sse < best[0]:
            best = (sse, xb, coef)
    if best is None:
        return None

    _, xb, coef = best
    a, b, c = coef

    def predict(xq):
        xq = np.asarray(xq, dtype=float)
        return a + b * xq + c * np.maximum(0.0, xq - xb)

    return xb, predict


def _draw_logpower_logact(segments, out_path):
    """画 log(Force)-log(Activation) 散点大图，按肌肉分子图，并叠加两段线性拟合。"""
    n = len(TARGET_MUSCLES)
    n_cols = min(3, n) if n > 0 else 1
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle('Isometric: log(Force) - log(Activation)',
                 fontsize=15, fontweight='bold')

    for k, musc in enumerate(TARGET_MUSCLES):
        r, c = divmod(k, n_cols)
        ax = axes[r][c]
        emg_col = f'emg_{musc}'

        all_x, all_y = [], []
        for label, color, seg in segments:
            if emg_col not in seg.columns:
                continue
            x = safe_log10(seg['force_total'].values)
            y = safe_log10(seg[emg_col].values)
            ax.scatter(x, y, s=POINT_SIZE, alpha=0.5,
                       color=color, label=label)
            all_x.append(x)
            all_y.append(y)

        # 用两段相连的线段拟合所有点（连续分段线性，单断点）
        if all_x:
            fx = np.concatenate(all_x)
            fy = np.concatenate(all_y)
            fit = _fit_two_segments(fx, fy)
            if fit is not None:
                xb, predict = fit
                m = np.isfinite(fx) & np.isfinite(fy)
                xs = np.unique(np.array([fx[m].min(), xb, fx[m].max()]))
                ax.plot(xs, predict(xs), color='k', lw=2, zorder=5,
                        label='两段拟合')

        ax.set_xlabel('log10(Force) (N)')
        ax.set_ylabel('log10(Activation)')
        ax.set_title(musc)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc='best')

    # 关闭多余的空子图
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _draw_time_activation(segments, out_path):
    """画 时间-力 + 时间-肌肉激活 散点大图（使用截取区间内全部点）。

    第一个子图为时间-力（force_l+force_r），其余子图为时间-各肌肉激活。
    """
    n = len(TARGET_MUSCLES) + 1  # +1 为时间-力子图
    n_cols = min(3, n) if n > 0 else 1
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle('Isometric: Time - Force / Activation',
                 fontsize=15, fontweight='bold')

    # 第一个子图：时间-力
    ax0 = axes[0][0]
    for label, color, seg in segments:
        if 'force_total' not in seg.columns:
            continue
        t = seg['time'].values - seg['time'].values[0]  # 从 0 开始
        ax0.scatter(t, seg['force_total'].values, s=POINT_SIZE, alpha=0.5,
                    color=color, label=label)
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Force (N)')
    ax0.set_title('Force')
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8, loc='best')

    # 其余子图：时间-肌肉激活
    for j, musc in enumerate(TARGET_MUSCLES):
        k = j + 1
        r, c = divmod(k, n_cols)
        ax = axes[r][c]
        emg_col = f'emg_{musc}'

        for label, color, seg in segments:
            if emg_col not in seg.columns:
                continue
            t = seg['time'].values - seg['time'].values[0]  # 从 0 开始
            ax.scatter(t, seg[emg_col].values, s=POINT_SIZE, alpha=0.5,
                       color=color, label=label)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Activation')
        ax.set_title(musc)
        ax.grid(True, alpha=0.3)

    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _draw_force_activation(segments, out_path):
    """画 力-肌肉激活 散点大图，按肌肉分子图（使用截取区间内全部点）。"""
    n = len(TARGET_MUSCLES)
    n_cols = min(3, n) if n > 0 else 1
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle('Isometric: Force - Activation',
                 fontsize=15, fontweight='bold')

    for k, musc in enumerate(TARGET_MUSCLES):
        r, c = divmod(k, n_cols)
        ax = axes[r][c]
        emg_col = f'emg_{musc}'

        for label, color, seg in segments:
            if emg_col not in seg.columns:
                continue
            ax.scatter(seg['force_total'].values, seg[emg_col].values,
                       s=POINT_SIZE, alpha=0.5, color=color, label=label)

        ax.set_xlabel('Force (N)')
        ax.set_ylabel('Activation')
        ax.set_title(musc)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc='best')

    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def main():
    subject = Subject(CONFIG_FILE)

    # isometric_data 未被 Subject 解析，直接从原始配置读取
    modeling = subject.config.get('modeling_file', {})
    iso_data = modeling.get('isometric_data', {})
    if not iso_data:
        print('配置中未找到 modeling_file.isometric_data，终止。')
        return

    # 复用 pipeline 内已构造好的 EMG 处理器与对齐器
    pipeline = MultiLoadPipeline(subject)
    emg_processor = pipeline.emg_processor
    aligner = pipeline.aligner

    segments = []
    for i, (label, info) in enumerate(iso_data.items()):
        seg = _process_isometric_entry(
            subject, emg_processor, aligner, label, info)
        if seg is not None and len(seg) > 0:
            color = ENTRY_COLORS[i % len(ENTRY_COLORS)]
            segments.append((label, color, seg))

    if not segments:
        print('没有可用的 isometric 数据，终止。')
        return

    save_dir = os.path.join(subject.result_folder, 'heatmap')
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir, 'isometric_logforce_logactivation.png')
    _draw_logpower_logact(segments, out_path)

    out_path_ta = os.path.join(
        save_dir, 'isometric_time_activation.png')
    _draw_time_activation(segments, out_path_ta)

    out_path_fa = os.path.join(
        save_dir, 'isometric_force_activation.png')
    _draw_force_activation(segments, out_path_fa)

    plt.show()


if __name__ == '__main__':
    main()