"""
log(功率) - log(激活) 双线段拟合（固定高度切片）

类似 example_load_activation_curve.py 中 X_AXIS='power' 的第四张图（log-log）：
  横轴 = log(功率)，纵轴 = log(RMS 激活)，按 肌肉 × 固定高度 排子图。
功率 = (force_l + force_r) * vel_l （W）。
散点配色等设置与 example_load_activation_curve.py 保持一致（按负载着色）。

在每个子图上用【连续分段线性】拟合这些散点（首尾严格相接、误差最小）：
  - 模型：连续折线 y = b0 + b1*x + Σ_j g_j * max(0, x - k_j)（铰链/截断线性基），
    由构造保证各段在节点 k_j 处严格相接（C0 连续，无跳变）；
  - 拟合：节点固定时关于系数是线性的，用最小二乘一次求解 -> 该节点下 RSS 全局最小；
  - 断点搜索：在候选节点位置上枚举 1 个/2 个节点（即两段/三段），每段点数有下限；
  - 段数选择：用 BIC = n*ln(RSS/n) + p*ln(n) 在 1/2/3 段之间择优，兼顾误差与过拟合；
  - 约束：最后一段（高 x）斜率限制为 >= 0（上升）；
  - 有多个断点时，各断点（节点）对应的 power/activation 都会打印。

另外作为对照，每个子图还用「直线(log-log)」与「对数曲线(原始)」各拟合一次（不画在图上），
与本方法一起把误差（RMSE/R²，log-log 与原始两个空间）列表打印对比。

用法：
    python example_logpower_logactivation_2seg.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline


# ============================================================
#  选项
# ============================================================
CONFIG_FILE = '../config/20250409_squat_NCMP001.json'
TARGET_MUSCLES = ["FibLon", "GL", "VL", "GlutMax", "SOL"]
MOVEMENT_TYPES = ['upward']

HEIGHT_FRACTIONS = [0.70, 0.75, 0.80]   # 在高度范围内取的相对位置
HEIGHT_WINDOW_FRAC = 0.05               # 散点高度窗口半宽（占总高度范围比例）
FORCE_COLUMNS = ['force_l', 'force_r']  # 相加得到总交互力 (N)
VELOCITY_COLUMN = 'vel_l'               # 速度列 (m/s)，power = force * 速度
LOAD_COLUMNS = ['load', 'load_value', 'load_weight']  # 负载列候选名
LOAD_COLORS = plt.cm.tab10.colors       # 不同负载使用不同颜色的散点

# 连续分段线性拟合参数（铰链基 + 断点搜索 + BIC 选段数）
FIT_MIN_SEG_PTS = 4   # 每段至少包含的点数
FIT_MAX_KNOTS = 2     # 最多节点数（2 节点 = 三段）
FIT_MAX_CAND = 60     # 每个节点的候选位置上限（大数据下做下采样）
FIT_LAST_RISING = True  # 约束“最后一段斜率 >= 0”（上升）


def safe_log(arr):
    """对非正元素返回 nan（不参与绘图），正元素取自然对数 ln（以 e 为底）。"""
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    pos = arr > 0
    out[pos] = np.log(arr[pos])
    return out


def _bic(n, rss, n_params):
    """高斯似然下的 BIC：n*ln(RSS/n) + p*ln(n)。越小越好。"""
    rss = max(float(rss), 1e-12)
    return n * np.log(rss / n) + n_params * np.log(n)


def _fit_pwl_continuous(xs, ys, knots, last_rising=FIT_LAST_RISING):
    """对给定节点用铰链基拟合【连续】分段线性，最小二乘最小化 RSS。
    模型：y = b0 + b1*x + Σ_j g_j * max(0, x - k_j)。
    last_rising=True 时约束“最后一段斜率 >= 0”（不满足则取边界 0 的约束最小二乘解）。
    返回 (segments, rss, n_params)，segments 在节点处严格相接。"""
    n = len(xs)
    K = len(knots)
    cols = [np.ones(n), xs]
    for k in knots:
        cols.append(np.maximum(0.0, xs - k))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)

    if last_rising:
        # 最后一段斜率 = b1 + Σ gammas = a·coef（a=[0,1,1,...,1]）；
        # 若 < 0，则在等式约束 a·coef = 0 下求约束最小二乘解（边界）。
        a = np.ones(coef.shape[0])
        a[0] = 0.0
        if float(a @ coef) < 0:
            M = np.linalg.pinv(A.T @ A)
            Ma = M @ a
            denom = float(a @ Ma)
            if denom > 0:
                coef = coef - Ma * (float(a @ coef) / denom)

    resid = ys - A @ coef
    rss = float(resid @ resid)

    b0, b1 = coef[0], coef[1]
    gammas = coef[2:]
    bounds = [xs[0]] + list(knots) + [xs[-1]]
    segments = []
    for i in range(K + 1):
        slope = b1 + float(np.sum(gammas[:i]))
        intercept = b0 - float(np.sum([gammas[j] * knots[j]
                                       for j in range(i)]))
        segments.append((slope, intercept, bounds[i], bounds[i + 1]))
    return segments, rss, 2 + K


def _candidate_indices(n, min_seg, max_cand):
    """候选节点索引（节点落在 xs[i-1] 与 xs[i] 之间），保证两侧各段点数 >= min_seg。"""
    lo, hi = min_seg, n - min_seg
    if hi < lo:
        return []
    idx = np.arange(lo, hi + 1)
    if len(idx) > max_cand:
        idx = np.unique(np.linspace(lo, hi, max_cand).astype(int))
    return [int(v) for v in idx]


def _fit_segments(x, y, min_seg_pts=FIT_MIN_SEG_PTS, max_knots=FIT_MAX_KNOTS,
                  max_cand=FIT_MAX_CAND):
    """连续分段线性拟合（铰链基 + 断点搜索 + BIC 选段数）：各段首尾严格相接，
    对给定节点 RSS 最小；在候选节点上枚举 1/2 个节点（两段/三段），用 BIC
    在 1/2/3 段之间择优，兼顾误差最小与防过拟合。

    返回 dict:
        {'segments': [seg, ...],   # seg=(slope, intercept, x_lo, x_hi)，x 从小到大
         'breaks': [knot, ...]}    # 节点(断点)，相邻段在此严格相接
    或 None（点数不足）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    if n < 2:
        return None

    # K=0：单段基准（也是点太少时的退化结果）
    segs0, rss0, p0 = _fit_pwl_continuous(xs, ys, [])
    best = (_bic(n, rss0, p0), segs0, [])
    if n < 2 * min_seg_pts:
        return {'segments': best[1], 'breaks': best[2]}

    cand = _candidate_indices(n, min_seg_pts, max_cand)

    # K=1：两段
    for i in cand:
        knots = [0.5 * (xs[i - 1] + xs[i])]
        segs, rss, p = _fit_pwl_continuous(xs, ys, knots)
        bic = _bic(n, rss, p)
        if bic < best[0]:
            best = (bic, segs, knots)

    # K=2：三段
    if max_knots >= 2:
        for a_i in range(len(cand)):
            i = cand[a_i]
            for b_i in range(a_i + 1, len(cand)):
                j = cand[b_i]
                if j - i < min_seg_pts:
                    continue
                k1 = 0.5 * (xs[i - 1] + xs[i])
                k2 = 0.5 * (xs[j - 1] + xs[j])
                if k2 - k1 <= 0:
                    continue
                segs, rss, p = _fit_pwl_continuous(xs, ys, [k1, k2])
                bic = _bic(n, rss, p)
                if bic < best[0]:
                    best = (bic, segs, [k1, k2])

    return {'segments': best[1], 'breaks': best[2]}


def _pwl_predict(res, xq):
    """连续分段线性在 log-x 空间的预测值（用于计算拟合误差）。"""
    segs = res['segments']
    breaks = res['breaks']
    xq = np.asarray(xq, dtype=float)
    idx = np.zeros(len(xq), dtype=int)
    for b in breaks:
        idx += (xq > b).astype(int)
    idx = np.clip(idx, 0, len(segs) - 1)
    out = np.empty(len(xq))
    for k, seg in enumerate(segs):
        sel = idx == k
        out[sel] = seg[0] * xq[sel] + seg[1]
    return out


def _ols_line(x, y):
    """最小二乘拟合 y = a + b*x，返回 (a, b)。"""
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _err_metrics(y_true, y_pred):
    """返回 (RMSE, R2)，忽略非有限值。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float('nan'), float('nan')
    e = y_true[m] - y_pred[m]
    rmse = float(np.sqrt(np.mean(e ** 2)))
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y_true[m] - np.mean(y_true[m])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return rmse, r2


def _print_fit_comparison(label, X, Y, res=None):
    """比较三种拟合在 log-log 与原始空间的误差并打印成表（不画图）：
      1) 之前的方法：连续分段 + BIC（在 log-log 空间拟合）；
      2) 直线 @ log-log：log(act) = a + b*log(power)；
      3) 对数曲线 @ 原始：act = a + b*ln(power)。
    X = log(power)，Y = log(activation)。各方法的预测都换算到两个空间后计误差。"""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    m = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[m], Y[m]
    n = len(X)
    if n < 2:
        print(f'[{label}] 有效点不足，跳过误差对比。')
        return

    act = np.exp(Y)            # 原始激活
    rows = []                  # (方法名, RMSE_loglog, R2_loglog, RMSE_raw, R2_raw)

    # 1) 之前的方法：连续分段 + BIC（log-log 空间）
    if res is None:
        res = _fit_segments(X, Y)
    if res is not None:
        Yhat = _pwl_predict(res, X)
        e_ll = _err_metrics(Y, Yhat)
        e_rw = _err_metrics(act, np.exp(Yhat))
        rows.append((f'连续分段({len(res["segments"])}段,BIC)',
                     e_ll[0], e_ll[1], e_rw[0], e_rw[1]))

    # 2) 直线 @ log-log：Y = a + b*X
    a, b = _ols_line(X, Y)
    Yhat = a + b * X
    e_ll = _err_metrics(Y, Yhat)
    e_rw = _err_metrics(act, np.exp(Yhat))
    rows.append(('直线@log-log', e_ll[0], e_ll[1], e_rw[0], e_rw[1]))

    # 3) 对数曲线 @ 原始：act = a + b*ln(power) = a + b*X
    a, b = _ols_line(X, act)
    act_hat = a + b * X
    with np.errstate(invalid='ignore'):
        Yhat = np.log(act_hat)     # 非正预测 -> nan，不计入 log-log 误差
    e_ll = _err_metrics(Y, Yhat)
    e_rw = _err_metrics(act, act_hat)
    rows.append(('对数曲线@原始', e_ll[0], e_ll[1], e_rw[0], e_rw[1]))

    # 打印表格
    print(f'\n[{label}] 拟合误差对比 (n={n})')
    print(f"  {'方法':<16}{'RMSE_loglog':>14}{'R2_loglog':>12}"
          f"{'RMSE_raw':>14}{'R2_raw':>10}")
    for name, e1, r1, e2, r2 in rows:
        print(f"  {name:<16}{e1:>14.4f}{r1:>12.4f}{e2:>14.5f}{r2:>10.4f}")


def _segment_draw_range(res, i):
    """第 i 段在 log-x 空间的绘制范围 (lo, hi)，向相邻交点延长。"""
    segs = res['segments']
    breaks = res['breaks']
    seg = segs[i]
    left_bound = seg[2] if i == 0 else breaks[i - 1]
    right_bound = seg[3] if i == len(segs) - 1 else breaks[i]
    return min(seg[2], left_bound), max(seg[3], right_bound)


def _plot_segments(ax, res, log_axes, label_on):
    """在 ax 上画各线段（均虚线），延长至相邻交点。
    log_axes=True 直接画 log 值；否则反变换为线性 (power, activation)。"""
    segs = res['segments']
    if len(segs) == 2:
        colors = ['C0', 'C3']
        names = ['Seg 1 (low x)', 'Seg 2 (high x)']
    else:
        colors = ['C0', 'C2', 'C3']
        names = ['Seg 1 (low x)', 'Seg 2 (mid)', 'Seg 3 (high x)']
    for i, seg in enumerate(segs):
        lo, hi = _segment_draw_range(res, i)
        gx = np.linspace(lo, hi, 100)
        gy = seg[0] * gx + seg[1]
        px, py = (gx, gy) if log_axes else (np.exp(gx), np.exp(gy))
        ax.plot(px, py, color=colors[i], lw=2, ls='--',
                label=(names[i] if label_on else None))


def _draw_loglog_twoseg(cutted, heights, h_window, x_label, suptitle, out_path):
    """画 肌肉 × 高度 的 log-log 大图，每个子图叠加双线段拟合。"""
    n_rows = len(TARGET_MUSCLES)
    n_cols = len(heights)

    # 检测负载列，为每个负载分配一种颜色（与 example_load_activation_curve.py 一致）
    load_col = next((c for c in LOAD_COLUMNS if c in cutted.columns), None)
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
        if emg_col not in cutted.columns:
            for c in range(n_cols):
                axes[r][c].set_axis_off()
                axes[r][c].set_title(f'{musc}: 无 {emg_col} 列')
            continue

        for c, h in enumerate(heights):
            ax = axes[r][c]
            frac = HEIGHT_FRACTIONS[c]
            mask = ((cutted['pos_l'] >= h - h_window) &
                    (cutted['pos_l'] <= h + h_window) &
                    (cutted['xval'] > 0) &
                    (cutted[emg_col] > 0))
            near = cutted[mask]

            # 散点（按负载着色，与 example_load_activation_curve.py 一致）
            all_lx, all_ly = [], []
            if load_col is not None:
                for lv in load_values:
                    sub = near[near[load_col] == lv]
                    if len(sub) == 0:
                        continue
                    lx = np.log(sub['xval'])
                    ly = safe_log(sub[emg_col])
                    ax.scatter(lx, ly, s=12, alpha=0.5,
                               color=load_color_map[lv],
                               label=(f'{lv:g} kg'
                                      if (r == 0 and c == 0) else None))
                    all_lx.append(np.asarray(lx, dtype=float))
                    all_ly.append(np.asarray(ly, dtype=float))
            else:
                lx = np.log(near['xval'])
                ly = safe_log(near[emg_col])
                ax.scatter(lx, ly, s=12, alpha=0.4, color='gray',
                           label='Raw data')
                all_lx.append(np.asarray(lx, dtype=float))
                all_ly.append(np.asarray(ly, dtype=float))

            # 线段拟合（两段或三段）
            if all_lx:
                X = np.concatenate(all_lx)
                Y = np.concatenate(all_ly)
                res = _fit_segments(X, Y)
                if res is not None:
                    _plot_segments(ax, res, log_axes=True,
                                   label_on=(r == 0 and c == 0))
                    segs = res['segments']
                    breaks = res['breaks']
                    info = (f'[{musc} @ h={h:.3f}] '
                            + ', '.join(f'seg{j + 1} slope={segs[j][0]:.3f}'
                                        for j in range(len(segs))))
                    for kk, xb in enumerate(breaks):
                        yb = 0.5 * ((segs[kk][0] * xb + segs[kk][1])
                                    + (segs[kk + 1][0] * xb + segs[kk + 1][1]))
                        info += (f'; 交点{kk + 1}: log_x={xb:.3f}, '
                                 f'log_y={yb:.3f} -> power={np.exp(xb):.3f} W, '
                                 f'activation={np.exp(yb):.5f}')
                    print(info)

                    # 与「直线@log-log」「对数曲线@原始」对比误差（不画图）
                    _print_fit_comparison(f'{musc} @ h={h:.3f}', X, Y, res=res)

            ax.set_xlabel(x_label)
            ax.set_ylabel('log(RMS activation)')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{musc} @ h={h:.3f} m ({int(frac * 100)}% range)')
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc='best')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def _draw_power_activation_twoseg(cutted, heights, h_window,
                                 x_label, suptitle, out_path):
    """power-activation 线性图（去除 log），叠加由 log-log 双线段反变换
    得到的幂律曲线（activation = exp(b) * power**m）。两段均用虚线。"""
    n_rows = len(TARGET_MUSCLES)
    n_cols = len(heights)

    load_col = next((c for c in LOAD_COLUMNS if c in cutted.columns), None)
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
        if emg_col not in cutted.columns:
            for c in range(n_cols):
                axes[r][c].set_axis_off()
                axes[r][c].set_title(f'{musc}: 无 {emg_col} 列')
            continue

        for c, h in enumerate(heights):
            ax = axes[r][c]
            frac = HEIGHT_FRACTIONS[c]
            mask = ((cutted['pos_l'] >= h - h_window) &
                    (cutted['pos_l'] <= h + h_window) &
                    (cutted['xval'] > 0) &
                    (cutted[emg_col] > 0))
            near = cutted[mask]

            # 散点（线性轴，按负载着色）；拟合仍在 log 空间进行
            all_lx, all_ly = [], []
            if load_col is not None:
                for lv in load_values:
                    sub = near[near[load_col] == lv]
                    if len(sub) == 0:
                        continue
                    ax.scatter(sub['xval'], sub[emg_col], s=12, alpha=0.5,
                               color=load_color_map[lv],
                               label=(f'{lv:g} kg'
                                      if (r == 0 and c == 0) else None))
                    all_lx.append(np.log(np.asarray(sub['xval'], dtype=float)))
                    all_ly.append(safe_log(sub[emg_col]))
            else:
                ax.scatter(near['xval'], near[emg_col], s=12, alpha=0.4,
                           color='gray', label='Raw data')
                all_lx.append(np.log(np.asarray(near['xval'], dtype=float)))
                all_ly.append(safe_log(near[emg_col]))

            # 在 log 空间拟合，再反变换为幂律曲线画到线性轴
            if all_lx:
                X = np.concatenate(all_lx)
                Y = np.concatenate(all_ly)
                res = _fit_segments(X, Y)
                if res is not None:
                    _plot_segments(ax, res, log_axes=False,
                                   label_on=(r == 0 and c == 0))

            ax.set_xlabel(x_label)
            ax.set_ylabel('RMS activation')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{musc} @ h={h:.3f} m ({int(frac * 100)}% range)')
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc='best')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def main():
    subject = Subject(CONFIG_FILE)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # ---- 收集切片数据 ----
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

    # ---- 构造功率列 xval = (force_l + force_r) * vel_l ----
    avail = [c for c in FORCE_COLUMNS if c in cutted.columns]
    if not avail:
        print(f'切片数据缺少力列 {FORCE_COLUMNS}，终止。')
        return
    if VELOCITY_COLUMN not in cutted.columns:
        print(f'切片数据缺少速度列 {VELOCITY_COLUMN}，终止。')
        return
    cutted['xval'] = cutted[avail].sum(axis=1) * cutted[VELOCITY_COLUMN]

    # ---- 固定高度 ----
    heights = [h_min + frac * (h_max - h_min) for frac in HEIGHT_FRACTIONS]
    h_window = (h_max - h_min) * HEIGHT_WINDOW_FRAC

    save_dir = os.path.join(subject.result_folder, 'heatmap')
    os.makedirs(save_dir, exist_ok=True)

    _draw_loglog_twoseg(
        cutted, heights, h_window,
        x_label='log(Power)  [Power in W]',
        suptitle='log(Power) — log(RMS) with two-segment fit',
        out_path=os.path.join(save_dir, 'logpower_logact_twoseg.png'))

    # 去 log 的 power-activation 图，叠加两段反变换后的幂律曲线
    _draw_power_activation_twoseg(
        cutted, heights, h_window,
        x_label='Power (W)',
        suptitle='Power — Activation with two-segment fit (transformed)',
        out_path=os.path.join(save_dir, 'power_activation_twoseg.png'))

    plt.show()


if __name__ == '__main__':
    main()