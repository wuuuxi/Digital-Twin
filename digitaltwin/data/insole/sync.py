"""
digitaltwin/data/insole/sync.py

鞋垫 / 机器人时间同步（互相关标定）。

为什么需要它
------------
鞋垫与机器人是两套独立采集系统，各自的起始时刻没有硬件同步。任何基于
文件元数据（info.csv 的 measurement_date）的对齐都只有分钟级可信度，而
一个下蹲相只有 1-2 s，秒级残差画出来就是“鞋垫力与机器人力整体错位”。

做法
----
两路信号各自去趋势 + 标准化，只在【深蹲段】上扫滞后做互相关，取相关
系数最大的滞后作为时间差，再用抛物线插值细化到亚采样点。

深蹲段怎么认：深蹲是双侧对称动作，左右鞋垫力应当同涨同落。所以用左右力的
【滑窗相关系数】做门控，r 低的片段（走动、调整站位、单脚卸力、上下杠）
一律不参与拟合。这一条比“力大于阈值”可靠得多：静止站立时力也很大，
但它没有波形，对互相关只有稀释作用。
"""
import numpy as np

from digitaltwin.utils.logger import beauty_print


SYNC_DT = 0.01            # 统一重采样步长 (s)
SYNC_MAX_LAG = 30.0       # 滞后搜索范围 ±(s)
SYNC_CORR_WIN = 1.0       # 左右一致性滑窗长度 (s)
SYNC_CORR_THR = 0.5       # 左右滑窗相关系数门槛
SYNC_MIN_SEG = 1.0        # 有效深蹲片段最短时长 (s)
SYNC_FORCE_FRAC = 0.10    # 着地门槛，占该侧峰值的比例
SYNC_DETREND_WIN = 5.0    # 去趋势滑窗长度 (s)
SYNC_MIN_CORR = 0.5       # 判定标定可信的最低峰值相关系数


def _moving_mean(x, n):
    """长度 n 的滑动均值，按有效样本数归一化。

    不能直接 convolve/n：两端会被零填充拉低，去趋势后就会在首尾凭空多出
    两个大峰，互相关会被这两个伪峰带跑。
    """
    x = np.asarray(x, dtype=float)
    n = max(1, int(n))
    if n <= 1:
        return x.copy()
    ker = np.ones(n, dtype=float)
    ok = np.isfinite(x)
    num = np.convolve(np.where(ok, x, 0.0), ker, mode='same')
    den = np.convolve(ok.astype(float), ker, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den < 1.0] = np.nan
    return out


def _detrend(x, dt, win_s=None):
    """减滑动均值。

    两路信号的直流完全不同（鞋垫含体重+配重，机器人只含杠力），不去直流
    的话相关系数会被两个常数主导，在所有滞后上都接近 1，峰值被碾平，
    定位不出时间差。
    """
    win_s = SYNC_DETREND_WIN if win_s is None else win_s
    n = int(round(win_s / dt))
    return np.asarray(x, dtype=float) - _moving_mean(x, n)


def _rolling_corr(a, b, n):
    """长度 n 的滑窗 Pearson 相关系数。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ma = _moving_mean(a, n)
    mb = _moving_mean(b, n)
    va = _moving_mean(a * a, n) - ma ** 2
    vb = _moving_mean(b * b, n) - mb ** 2
    cov = _moving_mean(a * b, n) - ma * mb
    with np.errstate(invalid='ignore', divide='ignore'):
        r = cov / np.sqrt(np.clip(va, 1e-12, None)
                          * np.clip(vb, 1e-12, None))
    return np.clip(np.nan_to_num(r, nan=0.0), -1.0, 1.0)


def _drop_short_runs(mask, min_len):
    """把长度不足 min_len 个样本的 True 段置假，并返回保留的区间。

    碎片段可能只是两个噪声峰偶然同相，它们可以在任意滞后上制造虚假的
    局部最优。
    """
    m = np.asarray(mask, dtype=bool).copy()
    if m.size == 0:
        return m, []
    d = np.diff(np.concatenate(([0], m.astype(int), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    kept = []
    for s, e in zip(starts, ends):
        if e - s < int(min_len):
            m[s:e] = False
        else:
            kept.append((int(s), int(e)))
    return m, kept


def squat_phase_mask(grid, force_l, force_r, corr_win=None, corr_thr=None,
                     force_frac=None, min_seg_s=None, detrend_win=None):
    """在等间隔网格上标出“左右同步发力”的深蹲段。

    判据：
      1. 左右去趋势力的滑窗相关系数 r >= corr_thr；
      2. 两侧都着地（各自 >= force_frac x 该侧峰值）；
      3. 连续时长 >= min_seg_s。

    Returns
    -------
    mask : np.ndarray[bool]
    info : dict -- r 序列、保留区间 (秒)、总时长
    """
    grid = np.asarray(grid, dtype=float)
    dt = float(np.median(np.diff(grid))) if grid.size > 1 else 0.01
    corr_win = SYNC_CORR_WIN if corr_win is None else corr_win
    corr_thr = SYNC_CORR_THR if corr_thr is None else corr_thr
    force_frac = SYNC_FORCE_FRAC if force_frac is None else force_frac
    min_seg_s = SYNC_MIN_SEG if min_seg_s is None else min_seg_s

    fl = np.asarray(force_l, dtype=float)
    fr = np.asarray(force_r, dtype=float)

    dl = _detrend(fl, dt, detrend_win)
    dr = _detrend(fr, dt, detrend_win)
    r = _rolling_corr(np.nan_to_num(dl), np.nan_to_num(dr),
                      max(3, int(round(corr_win / dt))))

    pl = np.nanmax(fl) if np.isfinite(fl).any() else 0.0
    pr = np.nanmax(fr) if np.isfinite(fr).any() else 0.0
    contact = (fl >= force_frac * pl) & (fr >= force_frac * pr)

    mask = (r >= corr_thr) & contact
    mask, runs = _drop_short_runs(mask, round(min_seg_s / dt))

    segments = [(float(grid[s]), float(grid[min(e, grid.size - 1)]))
                for s, e in runs]
    info = {
        'r_lr': r,
        'segments': segments,
        'duration_s': float(mask.sum()) * dt,
        'dt': dt,
    }
    return mask, info


def estimate_time_offset(time_l, force_l, time_r, force_r,
                         robot_time, robot_force,
                         dt=None, max_lag=None, corr_win=None,
                         corr_thr=None, force_frac=None,
                         min_seg_s=None, detrend_win=None,
                         allow_negative=False, verbose=True):
    """用互相关标定鞋垫相对机器人的时间差。

    约定：
        corrected_insole_time = raw_insole_time + offset

    即 offset > 0 表示鞋垫信号需要往后挪（鞋垫启动早于机器人）。

    Parameters
    ----------
    time_l, force_l : array-like -- 左鞋垫【未经任何对齐】的原始时间与力
    time_r, force_r : array-like -- 右鞋垫
    robot_time      : array-like -- 机器人相对时间 (s, 从 0 起)
    robot_force     : array-like -- 机器人参考力，建议 force_l + force_r
    allow_negative  : bool -- 若反相相关明显更强，是否接受它。默认 False：
                      只告警，不静静地把反相当成对齐。

    Returns
    -------
    dict or None
        offset / corr / polarity / reliable / at_edge /
        lags / corrs / grid / mask / insole_detrended / insole_total /
        segments / fit_duration_s / dt / fallback
    """
    dt = SYNC_DT if dt is None else float(dt)
    max_lag = SYNC_MAX_LAG if max_lag is None else float(max_lag)
    min_seg_s = SYNC_MIN_SEG if min_seg_s is None else float(min_seg_s)

    tl = np.asarray(time_l, dtype=float)
    fl = np.asarray(force_l, dtype=float)
    tr = np.asarray(time_r, dtype=float)
    fr = np.asarray(force_r, dtype=float)
    rt = np.asarray(robot_time, dtype=float)
    rf = np.asarray(robot_force, dtype=float)

    for name, arr in (('左鞋垫', tl), ('右鞋垫', tr), ('机器人', rt)):
        if arr.size < 2 or not np.isfinite(arr).any():
            beauty_print('  [Sync] {}数据不足，无法标定'.format(name),
                         type='warning')
            return None

    # 1) 统一网格 = 左右鞋垫时间轴的交集
    t0 = max(float(np.nanmin(tl)), float(np.nanmin(tr)))
    t1 = min(float(np.nanmax(tl)), float(np.nanmax(tr)))
    if not np.isfinite(t0) or not np.isfinite(t1) or (t1 - t0) < min_seg_s:
        beauty_print('  [Sync] 左右鞋垫时间轴重叠不足，无法标定',
                     type='warning')
        return None

    grid = np.arange(t0, t1 + 0.5 * dt, dt)
    gl = np.interp(grid, tl, fl)
    gr = np.interp(grid, tr, fr)

    # 2) 只保留左右趋势一致的深蹲段
    mask, mask_info = squat_phase_mask(
        grid, gl, gr, corr_win=corr_win, corr_thr=corr_thr,
        force_frac=force_frac, min_seg_s=min_seg_s, detrend_win=detrend_win)

    if mask.sum() * dt < min_seg_s:
        beauty_print(
            '  [Sync] 未找到足够长的左右同步段 (共 {:.1f}s)，'
            '退回全段拟合，结果可信度下降。请检查是否有一侧鞋垫未采到数据。'
            .format(mask.sum() * dt),
            type='warning')
        mask = np.isfinite(gl) & np.isfinite(gr)
        mask_info['segments'] = [(float(grid[0]), float(grid[-1]))]
        mask_info['fallback'] = True

    # 3) 去趋势后只在掩膜内参与相关
    ins = _detrend(gl + gr, dt, detrend_win)
    base = np.where(mask, ins, np.nan)

    # 4) 机器人铺到同一步长、两端各富余 max_lag 的网格上，
    #    之后“移位”就只是切片，不用在每个滞后上重新插值。
    M = int(round(max_lag / dt))
    K = grid.size
    rgrid = t0 + (np.arange(K + 2 * M) - M) * dt
    rob = np.interp(rgrid, rt, rf, left=np.nan, right=np.nan)
    rob = _detrend(rob, dt, detrend_win)

    lags = (np.arange(2 * M + 1) - M) * dt
    corrs = np.full(lags.size, np.nan)
    min_n = max(3, int(round(min_seg_s / dt)))

    for m in range(lags.size):
        seg = rob[m:m + K]
        ok = np.isfinite(base) & np.isfinite(seg)
        if int(ok.sum()) < min_n:
            continue
        a = base[ok]
        b = seg[ok]
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if denom <= 0:
            continue
        corrs[m] = float((a * b).sum() / denom)

    if not np.isfinite(corrs).any():
        beauty_print('  [Sync] 所有滞后上都无有效重叠，无法标定。'
                     '真实时间差可能超出 ±{:.0f}s 搜索范围。'.format(max_lag),
                     type='warning')
        return None

    # 5) 峰值 + 抛物线插值细化到亚采样点
    i_pos = int(np.nanargmax(corrs))
    i_abs = int(np.nanargmax(np.abs(corrs)))
    polarity = 1
    i = i_pos
    if i_abs != i_pos and abs(corrs[i_abs]) > corrs[i_pos] + 0.15:
        beauty_print(
            '  [Sync] 反相相关 ({:.2f} @ {:+.3f}s) 明显强于同相相关 '
            '({:.2f} @ {:+.3f}s)。鞋垫 GRF 与机器人杠力在深蹲中应当同相，'
            '请先确认机器人力的符号约定。'.format(
                corrs[i_abs], lags[i_abs], corrs[i_pos], lags[i_pos]),
            type='warning')
        if allow_negative:
            i = i_abs
            polarity = -1

    offset = float(lags[i])
    peak = float(corrs[i])
    if 0 < i < corrs.size - 1:
        y0, y1, y2 = corrs[i - 1], corrs[i], corrs[i + 1]
        if np.isfinite([y0, y1, y2]).all():
            den = y0 - 2.0 * y1 + y2
            if den != 0:
                delta = 0.5 * (y0 - y2) / den
                if abs(delta) <= 1.0:
                    offset += float(delta) * dt
                    peak = float(y1 - 0.25 * (y0 - y2) * delta)

    at_edge = (i <= 1) or (i >= corrs.size - 2)
    reliable = bool(abs(peak) >= SYNC_MIN_CORR and not at_edge)

    if at_edge:
        beauty_print(
            '  [Sync] 峰值落在搜索边界 (±{:.0f}s)，真实时间差可能更大，'
            '请加大 max_lag 重跑。'.format(max_lag), type='warning')
    if abs(peak) < SYNC_MIN_CORR:
        beauty_print(
            '  [Sync] 峰值相关系数只有 {:.2f}，低于门槛 {:.2f}，标定不可信。'
            '常见原因：该组深蹲次数太少、鞋垫中途掉帧、'
            '或两个文件根本不是同一次采集。'.format(peak, SYNC_MIN_CORR),
            type='warning')

    if verbose:
        print('  [Sync] offset={:+.3f}s, corr={:.3f}, 拟合时长={:.1f}s, '
              '深蹲段={} 个, reliable={}'.format(
                  offset, peak, mask.sum() * dt,
                  len(mask_info.get('segments', [])), reliable))

    return {
        'offset': offset,
        'corr': peak,
        'polarity': polarity,
        'reliable': reliable,
        'at_edge': bool(at_edge),
        'lags': lags,
        'corrs': corrs,
        'grid': grid,
        'mask': mask,
        'insole_detrended': ins,
        'insole_total': gl + gr,
        'segments': mask_info.get('segments', []),
        'fit_duration_s': float(mask.sum()) * dt,
        'dt': dt,
        'fallback': bool(mask_info.get('fallback', False)),
    }