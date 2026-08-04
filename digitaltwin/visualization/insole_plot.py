'''
insole_plot.py

鞋垫压力分布的可视化。只负责画图，不做任何校验或判定。

主入口 plot_load_pressure_cop()：一个配重一张图，左右脚各一个子图。
底图是该次采集的平均压强（网格与文件一致，通常 20 行 x 12 列），
上面叠 COP 轨迹，按该帧的总力着色。

这样一张图同时给出三件事：
  1. 压力分布的左右差异（同一张图里两脚共用色标，可直接比大小）
  2. COP 在深蹲过程中的摆动范围
  3. COP 均值相对模型常数接触点的偏移

图上文字全用英文。matplotlib 默认字体 DejaVu Sans 不包含汉字，
中文会退化成方块（并刷出大量 missing glyph 告警）。用英文比去碰
本地中文字体是否存在更可靠。

传入的 result 应该已经被调用方裁到深蹲窗口。本模块不做时间筛选，
给什么就画什么。
'''
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

SIDE_LABEL = {'l': 'Left foot (L)', 'r': 'Right foot (R)'}


def _safe_name(load_key):
    '''组名转文件名。组名可能带 '/'（如 IK-0.3m/s），不能直接落盘。'''
    text = str(load_key).strip().replace('.', 'p')
    for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|', ' '):
        text = text.replace(ch, '_')
    return text or 'unnamed'


def _mean_pressure(result, min_force=None):
    '''
    全时段平均压强 (n_rows, n_cols)。

    只对着地帧求平均。悬空帧全是零，计进去会把整体压强压低，
    而且不同配重的悬空比例不同，会让配重之间无法比较。
    '''
    pressure = result.get('pressure')
    if pressure is None:
        return None

    pressure = np.asarray(pressure, dtype=float)
    force = np.asarray(result.get('force'), dtype=float)

    if min_force is None:
        min_force = 20.0

    use = np.isfinite(force) & (force >= min_force)
    if use.sum() < 1:
        use = np.ones(len(pressure), dtype=bool)

    return np.nanmean(pressure[use], axis=0)


def _force_range(results):
    '''COP 配色用的总力区间 (N)。'''
    lo, hi = [], []
    for res in results:
        if res is None:
            continue
        f = np.asarray(res.get('force'), dtype=float)
        f = f[np.isfinite(f)]
        if f.size:
            lo.append(float(f.min()))
            hi.append(float(f.max()))
    if not lo:
        return None
    return (min(lo), max(hi))


def global_pressure_vmax(all_results, min_force=None):
    '''
    所有组、所有侧的平均压强最大值。

    把它传给 plot_load_pressure_cop(vmax=...)，各组热图的颜色才代表
    同一个压强值。若每组各自归一化，两张图看起来深浅差不多，
    实际可能差好几倍，跳组比较全是错的。

    Parameters
    ----------
    all_results : dict -- {load_key: {'l': result, 'r': result}}
    '''
    peaks = []
    for sides in (all_results or {}).values():
        for res in (sides or {}).values():
            if res is None:
                continue
            mean_p = _mean_pressure(res, min_force=min_force)
            if mean_p is not None and np.isfinite(mean_p).any():
                peaks.append(float(np.nanmax(mean_p)))
    return max(peaks) if peaks else None


def global_force_range(all_results):
    '''所有组、所有侧的总力区间，用于统一 COP 轨迹的配色。'''
    flat = []
    for sides in (all_results or {}).values():
        for res in (sides or {}).values():
            flat.append(res)
    return _force_range(flat)


def _cop_xy(result, meta):
    '''
    把 COP 换算成绘图坐标 (cm)。

    load_pressure_map 给出的 cop_ant 是「距足跟端的前向距离」，
    而图像的行 0 在足趾端，所以纵坐标要翻过来。
    '''
    length_cm = float(meta['length_cm'])

    cop_ant = np.asarray(result.get('cop_ant'), dtype=float) * 100.0
    cop_lat = np.asarray(result.get('cop_lat'), dtype=float) * 100.0

    y = length_cm - cop_ant
    x = cop_lat

    ok = np.isfinite(x) & np.isfinite(y)
    return x, y, ok


def _draw_one_side(ax, result, side, vmax, min_force=None,
                   contact_point_m=None, force_range=None):
    meta = result['meta']
    n_rows = int(meta['n_rows'])
    n_cols = int(meta['n_cols'])
    width_cm = float(meta['width_cm'])
    length_cm = float(meta['length_cm'])

    mean_p = _mean_pressure(result, min_force=min_force)
    if mean_p is None:
        ax.text(0.5, 0.5, 'no pressure-map data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_axis_off()
        return None, None

    # 行 0 在足趾端，origin=upper 让足趾朝上
    im = ax.imshow(mean_p, extent=(0, width_cm, length_cm, 0),
                   origin='upper', aspect='equal',
                   cmap='inferno', vmin=0, vmax=vmax,
                   interpolation='nearest')

    # 网格线，便于数格子
    for c in range(n_cols + 1):
        ax.axvline(c * width_cm / n_cols, color='w', lw=0.25, alpha=0.25)
    for r in range(n_rows + 1):
        ax.axhline(r * length_cm / n_rows, color='w', lw=0.25, alpha=0.25)

    # COP 轨迹：按该帧的总力着色。
    # 按时间着色只能看出行进方向，而真正要回答的问题是
    # 「压心随受力增大往哪里走」，所以颜色改成总力 (N)。
    lc = None
    x, y, ok = _cop_xy(result, meta)
    if ok.sum() > 2:
        xs, ys = x[ok], y[ok]
        force = np.asarray(result.get('force'), dtype=float)[ok]

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='cool', linewidths=0.9,
                            alpha=0.8)
        # 每一小段的颜色取两端点总力的均值
        lc.set_array(0.5 * (force[:-1] + force[1:]))
        if force_range is not None:
            lc.set_clim(float(force_range[0]), float(force_range[1]))
        ax.add_collection(lc)

        ax.plot(xs.mean(), ys.mean(), marker='X', ms=11,
                mfc='white', mec='black', mew=1.4, ls='none',
                label='COP mean', zorder=5)

    # 模型里用的常数接触点，画成横线便于对照
    if contact_point_m is not None:
        y_const = length_cm - float(contact_point_m) * 100.0
        ax.axhline(y_const, color='lime', ls='--', lw=1.4,
                   label='model constant contact point', zorder=4)

    ax.set_xlim(0, width_cm)
    ax.set_ylim(length_cm, 0)
    ax.set_xlabel('medio-lateral (cm)')
    ax.set_ylabel('distance from toe end (cm)')

    title = SIDE_LABEL.get(side, side.upper())
    if ok.sum() > 2:
        ant = np.asarray(result['cop_ant'], dtype=float)[ok] * 100.0
        title += '\nanterior COP {:.1f} +- {:.1f} cm, range {:.1f} cm'.format(
            ant.mean(), ant.std(), ant.max() - ant.min())
    ax.set_title(title, fontsize=10)

    return im, lc


def plot_load_pressure_cop(side_results, load_key='', contact_point_m=None,
                           min_force=None, describe=None, save_dir=None,
                           show=False, share_scale=True, dpi=140,
                           vmax=None, force_range=None):
    '''
    一个配重一张图，左右脚各一个子图。

    Parameters
    ----------
    side_results : dict -- {'l': result, 'r': result}，
                   result 为 InsoleProcessor.load_pressure_map 的返回值
                   （需要 return_matrix=True）。值为 None 的一侧会被跳过。
    load_key : str -- 组名，用于标题与文件名
    contact_point_m : float, optional -- 模型里 insole_contact_point 的
                      前向分量 (m)，画成参考线
    min_force : float, optional -- 参与平均的最小总力 (N)
    describe : str, optional -- 标题里附加的说明
    save_dir : str, optional -- 给出则保存 png
    show : bool -- 是否立即 plt.show()
    share_scale : bool -- 左右脚是否共用色标。共用才能直接比大小
    vmax : float, optional -- 压强色标上限。由调用方统一传入同一个值（见
           global_pressure_vmax），各组颜色才代表同一个压强，可以跳组
           比较；留空则每组自己算，颜色只在组内可比
    force_range : (float, float), optional -- COP 轨迹配色对应的总力区间 (N)，
           同样建议由调用方统一给出（见 global_force_range）

    Returns
    -------
    matplotlib.figure.Figure or None
    '''
    sides = [s for s in ('l', 'r')
             if side_results.get(s) is not None
             and side_results[s].get('pressure') is not None]
    if not sides:
        return None

    # 色标上限：调用方传入 vmax 时一律优先，这样所有组共用同一个刻度
    if vmax is None and share_scale:
        peaks = []
        for s in sides:
            mean_p = _mean_pressure(side_results[s], min_force=min_force)
            if mean_p is not None:
                peaks.append(float(np.nanmax(mean_p)))
        vmax = max(peaks) if peaks else None

    # COP 配色的力区间：未指定时按本组两脚的实际范围
    if force_range is None:
        force_range = _force_range([side_results[s] for s in sides])

    fig, axes = plt.subplots(1, len(sides), figsize=(4.2 * len(sides), 7.2))
    if len(sides) == 1:
        axes = [axes]

    im = None
    lc = None
    for ax, side in zip(axes, sides):
        this_vmax = vmax
        if vmax is None and not share_scale:
            mean_p = _mean_pressure(side_results[side], min_force=min_force)
            this_vmax = None if mean_p is None else float(np.nanmax(mean_p))
        drawn, drawn_lc = _draw_one_side(ax, side_results[side], side,
                                         this_vmax, min_force=min_force,
                                         contact_point_m=contact_point_m,
                                         force_range=force_range)
        if drawn is not None:
            im = drawn
        if drawn_lc is not None:
            lc = drawn_lc

    header = describe if describe else 'load = {}'.format(load_key)
    fig.suptitle('Mean plantar pressure with COP trajectory    {}'.format(
        header), fontsize=12)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.04)
        cbar.set_label('mean pressure (N/cm2)')
    if lc is not None:
        cbar2 = fig.colorbar(lc, ax=axes, fraction=0.045, pad=0.02)
        cbar2.set_label('COP colour: total force (N)')

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc='lower right', fontsize=8,
                       framealpha=0.85)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe = _safe_name(load_key)
        path = os.path.join(save_dir, 'insole_cop_{}.png'.format(safe))
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print('  [PLOT] 已保存 {}'.format(path))

    if show:
        plt.show()

    return fig


def plot_cop_across_loads(all_results, contact_point_m=None, save_dir=None,
                          show=False, dpi=140):
    '''
    把各组的前向 COP 均值画在一起，看是否随负载前移。

    Parameters
    ----------
    all_results : dict -- {load_key: {'l': result, 'r': result}}
    '''
    loads, mean_l, mean_r, sd_l, sd_r = [], [], [], [], []

    for load_key, sides in all_results.items():
        row = {}
        for s in ('l', 'r'):
            res = sides.get(s)
            if res is None:
                row[s] = (np.nan, np.nan)
                continue
            ant = np.asarray(res.get('cop_ant'), dtype=float) * 100.0
            ant = ant[np.isfinite(ant)]
            row[s] = (ant.mean(), ant.std()) if ant.size else (np.nan, np.nan)

        loads.append(str(load_key))
        mean_l.append(row['l'][0])
        sd_l.append(row['l'][1])
        mean_r.append(row['r'][0])
        sd_r.append(row['r'][1])

    if not loads:
        return None

    idx = np.arange(len(loads))
    fig, ax = plt.subplots(figsize=(1.1 * len(loads) + 3.5, 4.6))

    ax.errorbar(idx - 0.08, mean_l, yerr=sd_l, fmt='o-', capsize=3,
                label='Left (L)')
    ax.errorbar(idx + 0.08, mean_r, yerr=sd_r, fmt='s-', capsize=3,
                label='Right (R)')

    if contact_point_m is not None:
        ax.axhline(float(contact_point_m) * 100.0, color='k', ls='--', lw=1.2,
                   label='model constant contact point')

    ax.set_xticks(idx)
    ax.set_xticklabels(loads, rotation=20, ha='right')
    ax.set_xlabel('trial group')
    ax.set_ylabel('anterior COP from heel (cm)')
    ax.set_title('Mean anterior COP by group '
                 '(error bars = within-group SD)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'insole_cop_across_loads.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print('  [PLOT] 已保存 {}'.format(path))

    if show:
        plt.show()

    return fig