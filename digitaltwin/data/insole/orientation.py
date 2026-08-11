"""
digitaltwin/data/insole/orientation.py

鞋垫方向诊断（足趾端在哪一行 / 内外侧是否镜像）。

为什么不能靠看图：热图是用同一段代码、同一个 origin 画的，左右子图没有
任何镜像处理。“看起来像一双脚印”只能说明两个矩阵在列方向上互为镜像，
它对【行】方向（足趾在哪一端）一无所知，而行方向才是决定膝力矩力臂的那一个。
所以这里用数据判，不用眼判。
"""
import numpy as np

from digitaltwin.utils.logger import beauty_print
from .io import MAP_MIN_FORCE_N


# 上一版把 ORIENT_CONTACT_FRAC=0.20 同时用于“着地”和“量宽度”，那是错的：
# 足跟压强高，0.20 x 全局峰值是个绝对高门槛，低压的前足会被整排判成没着地，
# 宽度判据于是退化成峰值判据的同义反复。现在把两件事分开：
# 几何足印用逐格 max over time + 很低的门槛。
ORIENT_CONTACT_FRAC = 0.20      # 仅用于峰值旁证的“高压区”比例
ORIENT_FOOTPRINT_FRAC = 0.05    # 二值足印门槛：全局峰值的比例
ORIENT_FOOTPRINT_ABS = 0.5      # 二值足印门槛的绝对下限 (N/cm2)
ORIENT_END_FRAC = 0.25          # 两端各取多少比例的行做对比
ORIENT_MIDLINE_GUARD_CM = 3.0   # 左右交叉检验的中点保护带 (cm)


def mean_contact_pressure(result, min_force=None):
    """只对着地帧求平均压强 (n_rows, n_cols)。

    悬空帧全是零，计进去会把分布整体稀释，而且不同组的悬空比例不同，
    会让组与组之间无法比较。
    """
    pressure = result.get('pressure') if result else None
    if pressure is None:
        return None

    pressure = np.asarray(pressure, dtype=float)
    force = np.asarray(result.get('force'), dtype=float)
    thr = MAP_MIN_FORCE_N if min_force is None else min_force

    use = np.isfinite(force) & (force >= thr)
    if use.sum() < 1:
        use = np.ones(len(pressure), dtype=bool)

    return np.nanmean(pressure[use], axis=0)


def peak_contact_pressure(result, min_force=None):
    """逐格 max over time (n_rows, n_cols)。

    量【几何足印】只能用逐格最大值，不能用均值：一个格子只要在整段里被
    踩到过，它就属于足印。用均值再配高门槛，等于要求前足的平均压强达到
    足跟量级，前足会被整排判成没着地。
    """
    pressure = result.get('pressure') if result else None
    if pressure is None:
        return None

    pressure = np.asarray(pressure, dtype=float)
    force = np.asarray(result.get('force'), dtype=float)
    thr = MAP_MIN_FORCE_N if min_force is None else min_force

    use = np.isfinite(force) & (force >= thr)
    if use.sum() < 1:
        use = np.ones(len(pressure), dtype=bool)

    return np.nanmax(pressure[use], axis=0)


def _footprint_mask(peak_map, frac=None, abs_floor=None):
    """二值足印。门槛 = max(frac x 全局峰值, 绝对下限)，并封顶在半峰值，
    免得某侧峰值特别低时绝对下限反而把整只脚都抹掉。
    """
    if peak_map is None:
        return None
    frac = ORIENT_FOOTPRINT_FRAC if frac is None else frac
    abs_floor = ORIENT_FOOTPRINT_ABS if abs_floor is None else abs_floor
    peak = float(np.nanmax(peak_map))
    if not np.isfinite(peak) or peak <= 0:
        return None
    thr = max(frac * peak, abs_floor)
    if thr > 0.5 * peak:
        thr = frac * peak
    return peak_map >= max(thr, 1e-9)


def _contact_width_profile(peak_map, frac=None, abs_floor=None):
    """每一行的足印宽度（格数），在二值足印上数。

    足印宽度沿长度是单峰的：最宽处是跖球，约在从足跟量起 70-75% 足长的
    位置，绝不在中点。所以“最宽行靠近哪一端”是纯几何判据，与本次重心
    前后分布无关，比拿两端互比稳健得多。
    """
    mask = _footprint_mask(peak_map, frac=frac, abs_floor=abs_floor)
    if mask is None:
        return None
    return mask.sum(axis=1).astype(float)


def diagnose_side_orientation(result, side='', min_force=None):
    """判断单侧鞋垫的行 0 是否在足趾端，并找出足弓在哪一侧。

    三个判据，按可靠性从高到低：
      1.【最宽行位置】主判据。最宽处 = 跖球，位于从足跟量起约 70-75% 足长处。
      2.【两端宽度对比】副判据，在二值足印上量，仅作交叉印证。
      3.【峰值压强端】旁证。足跟通常压强最高，但深蹲重心前移时前足完全可以
         反超，所以它不能单独作判据。

    Returns
    -------
    dict or None
    """
    mean_p = mean_contact_pressure(result, min_force=min_force)
    peak_map = peak_contact_pressure(result, min_force=min_force)
    width = _contact_width_profile(peak_map)
    if mean_p is None or peak_map is None or width is None:
        return None

    n_rows, n_cols = peak_map.shape
    k = max(1, int(round(n_rows * ORIENT_END_FRAC)))

    w_head = float(np.nanmean(width[:k]))
    w_tail = float(np.nanmean(width[-k:]))
    p_head = float(np.nanmax(mean_p[:k]))
    p_tail = float(np.nanmax(mean_p[-k:]))

    # 主判据：最宽行在【实际着地行范围】内的位置。垫比脚长，两端总有几行
    # 从不受力，把它们算进去会把归一化位置往中间拉。
    rows = np.flatnonzero(width > 0)
    if rows.size >= 3:
        r0, r1 = int(rows[0]), int(rows[-1])
    else:
        r0, r1 = 0, n_rows - 1
    sub = width[r0:r1 + 1]
    widest_row = int(r0 + int(np.argmax(sub)))
    span_rows = max(r1 - r0, 1)
    widest_frac = float((widest_row - r0) / span_rows)
    toe_first_suggest = bool(widest_frac < 0.5)

    # 副判据 / 旁证
    toe_first_by_width = w_head > w_tail
    toe_first_by_peak = p_tail > p_head

    # 足弓：中足段压强低的那半边是内侧。这是区分左右镜像的唯一解剖学
    # 锚点——外侧缘总是连续承重，内侧中段悬空。
    lo = int(round(n_rows / 3.0))
    hi = int(round(n_rows * 2.0 / 3.0))
    mid = mean_p[lo:hi] if hi > lo else mean_p
    half = n_cols // 2
    c_low = float(np.nanmean(mid[:, :half]))
    c_high = float(np.nanmean(mid[:, half:]))
    arch_col_side = 'low' if c_low < c_high else 'high'

    return {
        'side': side,
        'n_rows': n_rows,
        'n_cols': n_cols,
        'width_head': w_head,
        'width_tail': w_tail,
        'peak_head': p_head,
        'peak_tail': p_tail,
        'widest_row': widest_row,
        'widest_row_frac': widest_frac,
        'contact_row_first': r0,
        'contact_row_last': r1,
        'toe_first_suggest': toe_first_suggest,
        'toe_first_by_width': bool(toe_first_by_width),
        'toe_first_by_peak': bool(toe_first_by_peak),
        'agree': bool(toe_first_by_width == toe_first_suggest),
        'midfoot_low_cols': c_low,
        'midfoot_high_cols': c_high,
        'arch_col_side': arch_col_side,
    }


def diagnose_orientation(side_results, toe_first_used=True, min_force=None,
                         verbose=True):
    """交叉比对左右鞋垫的行/列方向，给出应该用什么 toe_first。

    Parameters
    ----------
    side_results : dict -- {'l': result, 'r': result}
    toe_first_used : bool or dict -- 读取时实际传给 load_pressure_map 的
        toe_first。cop_ant 已经受它影响，不告诉这个函数就无法反推。

    Returns
    -------
    dict -- 包含逐侧结果与两项交叉检验
    """
    out = {'sides': {}, 'issues': []}

    if isinstance(toe_first_used, dict):
        used = dict(toe_first_used)
    else:
        used = {'l': bool(toe_first_used), 'r': bool(toe_first_used)}

    for s in ('l', 'r'):
        res = (side_results or {}).get(s)
        if res is None:
            continue
        info = diagnose_side_orientation(res, side=s, min_force=min_force)
        if info is None:
            continue
        info['toe_first_used'] = used.get(s, True)
        # 与【主判据】比，不是与副判据比
        info['match'] = (info['toe_first_suggest'] == info['toe_first_used'])
        out['sides'][s] = info

    if verbose:
        print('  [Orient] 主判据 = 最宽行位置 (跖球在足长 70-75% 处)，'
              'wide_frac < 0.5 -> 行 0 在足趾端')
        print('    {:<5}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}'
              .format('side', 'wide_row', 'wide_frac', 'w_head',
                      'w_tail', 'p_ratio', 'suggest', 'used'))
        for s, i in out['sides'].items():
            p_ratio = (i['peak_tail'] / i['peak_head']
                       if i['peak_head'] > 1e-9 else float('nan'))
            print('    {:<5}{:>10d}{:>10.2f}{:>10.2f}{:>10.2f}'
                  '{:>10.2f}{:>10}{:>10}'.format(
                      s.upper(), int(i['widest_row']),
                      i['widest_row_frac'], i['width_head'],
                      i['width_tail'], p_ratio,
                      str(i['toe_first_suggest']),
                      str(i['toe_first_used'])))

    for s, i in out['sides'].items():
        # 副判据/旁证与主判据不一致【不】算错误：深蹲重心前移时前足压强
        # 反超足跟很常见，两端宽度也受门槛影响。只打印，不告警。
        if verbose and not i['agree']:
            print('    [Orient] {} 侧副判据(两端宽度)与主判据不一致，'
                  '以主判据(最宽行位置)为准。'.format(s.upper()))
        if verbose and (i['toe_first_by_peak'] != i['toe_first_suggest']):
            print('    [Orient] {} 侧峰值旁证与主判据不一致，属常见现象'
                  '(深蹲重心前移使前足压强反超足跟)。'.format(s.upper()))
        if not i['match']:
            out['issues'].append('{}: toe_first 与主判据不符'.format(s))
            beauty_print(
                '  [Orient] {} 侧 toe_first 传的是 {}，但最宽行位置'
                '(wide_frac={:.2f}) 指示应该是 {}。COP 前后方向会整个翻转，'
                '膝力矩结论会反过来，请核对热图后再继续。'.format(
                    s.upper(), i['toe_first_used'],
                    i['widest_row_frac'], i['toe_first_suggest']),
                type='warning')

    # 交叉检验：左右运动学已确认对称，两脚的 COP 应当落在相似位置。
    # 若两侧之和接近全长/全宽，就是典型的“有一侧被翻转了”。
    res_l = (side_results or {}).get('l')
    res_r = (side_results or {}).get('r')
    if res_l is not None and res_r is not None:
        meta = res_l['meta']
        for field, span, name in (
                ('cop_ant', meta['length_cm'], '前后 (行)'),
                ('cop_lat', meta['width_cm'], '内外 (列)')):
            a = np.asarray(res_l.get(field), dtype=float) * 100.0
            b = np.asarray(res_r.get(field), dtype=float) * 100.0
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if a.size == 0 or b.size == 0:
                continue
            ma, mb = float(a.mean()), float(b.mean())
            d_same = abs(ma - mb)
            d_flip = abs(ma + mb - float(span))
            # 中点保护：这个检验的前提是 COP 远离垫子中点。若两侧 COP 都
            # 贴着 span/2，L+R≈span 是代数必然，与是否翻转无关。
            guard = ORIENT_MIDLINE_GUARD_CM
            mid = float(span) / 2.0
            near_mid = (abs(ma - mid) < guard) and (abs(mb - mid) < guard)
            if near_mid:
                verdict = 'inconclusive'
            else:
                verdict = 'same' if d_same <= d_flip else 'mirrored'
            out[field] = {
                'mean_l': ma, 'mean_r': mb, 'span': float(span),
                'diff_same': d_same, 'diff_flip': d_flip,
                'near_midline': bool(near_mid),
                'verdict': verdict,
            }
            if verbose:
                note = ('（两侧 COP 都在中点 ±{:.0f}cm 内，此检验无判别力）'
                        .format(guard)) if near_mid else ''
                print('  [Orient] {} 方向: L={:.2f} R={:.2f} cm, '
                      '全长={:.2f} | |L-R|={:.2f} vs |L+R-全长|={:.2f} '
                      '-> {}{}'.format(name, ma, mb, span,
                                       d_same, d_flip, verdict, note))

        if out.get('cop_ant', {}).get('verdict') == 'mirrored':
            out['issues'].append('前后方向左右不一致')
            beauty_print(
                '  [Orient] 两侧前后 COP 之和接近鞋垫全长，说明有一侧的行'
                '顺序是反的。运动学已确认左右对称，两脚 COP 不应该差这么多。'
                '必须先修正再做逐帧 COP。',
                type='warning')

        sl = out['sides'].get('l', {}).get('arch_col_side')
        sr = out['sides'].get('r', {}).get('arch_col_side')
        if sl and sr:
            out['column_frame'] = ('world' if sl != sr else 'anatomical')
            out['column_frame_by_side'] = (sl, sr)
            out.setdefault('notes', []).append(
                '列坐标为 {} 坐标系 (L 足弓在列号{}侧, R 在列号{}侧)'.format(
                    out['column_frame'], sl, sr))
            if verbose:
                print('  [Orient] 足弓位置: L 在列号{}侧, R 在列号{}侧 -> '
                      '列坐标为 {} 坐标系'.format(sl, sr, out['column_frame']))
            if out['column_frame'] == 'anatomical':
                beauty_print(
                    '  [Orient] 两侧足弓在同一侧列号，说明文件按每只脚自己的'
                    '解剖方向存储。换算到模型的左右轴时需要把其中一侧列翻转，'
                    '否则额状面力矩（高内收）会错。矢状面膝力矩不受影响。',
                    type='warning')

    if verbose and not out['issues']:
        print('  [Orient] 未发现方向问题。')

    return out