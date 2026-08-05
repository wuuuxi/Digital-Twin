'''
symmetry_plot.py

左右对称性诊断的四张图，供 example_symmetry_check.py 调用。

为什么是这四张：对称性不是一个数，而是四个不同层面的问题。
  ① SI 热图        —— 哪个通道偏？偏多少？随负载怎么变？
  ② 传递图          —— 偏差是从外力就有，还是沿运动链被放大出来的？
  ③ 左-右散点      —— 偏差是恒定偏置（截距）还是随幅值放大（斜率）？
  ④ 蝶形图          —— 偏差发生在动作周期的哪个相位？
四者回答的是四个不可互相替代的问题，所以都画。

数据契约（由 example_symmetry_check.collect_side_data 提供）：
  data = {load_key: {
      'load_value': float or nan,      # 等长/等速组为 nan
      'mode':       'isotonic' | 'isokinetic' | 'isometric',
      'force_l':    np.ndarray,        # 逐帧，优先是鞋垫 grf_l
      'force_r':    np.ndarray,
      'moments':    {base: {'l': float, 'r': float}},   # 均绝对值
      'angles':     {base: {'l': float, 'r': float}},   # 峰值
      'angle_curves': {base: {'grid': (101,),
                              'l': (n_cycle, 101),
                              'r': (n_cycle, 101)}},
  }}

全部标签用英文：matplotlib 默认字体 DejaVu Sans 没有 CJK 字形，
中文会成排渲染成方块（之前在鞋垫热图上已经踩过一次）。
'''
import os

import numpy as np
import matplotlib.pyplot as plt

from digitaltwin.utils.logger import beauty_print


# 关节名 -> 图上的短标签
PRETTY = {
    'knee_angle': 'Knee',
    'hip_flexion': 'Hip',
    'ankle_angle': 'Ankle',
    'subtalar_angle': 'Subtalar',
}

# 传递图的解剖顺序：地面 -> 踝 -> 膝 -> 髋
CHAIN_ORDER = ('__force__', 'ankle_angle', 'knee_angle', 'hip_flexion')

SIDE_COLOR = {'l': '#1f77b4', 'r': '#d62728'}

# 逐组配色：必须用【高对比的离散色】，不能用 viridis 这类连续色标。
# 9 组时 viridis 中段全是蓝绿，散点叠在一起根本分不出是哪一组。
GROUP_PALETTE = (
    '#e6194b',  # red
    '#3cb44b',  # green
    '#4363d8',  # blue
    '#f58231',  # orange
    '#911eb4',  # purple
    '#000000',  # black
    '#f032e6',  # magenta
    '#808000',  # olive
    '#00bcd4',  # cyan
    '#8b4513',  # brown
)


def _group_color(i):
    return GROUP_PALETTE[i % len(GROUP_PALETTE)]


def _linfit(x, y):
    '''最小二乘直线拟合；有效样本不足时返回 (nan, nan)。'''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan, np.nan
    k, b = np.polyfit(x[m], y[m], 1)
    return float(k), float(b)


def _thin(n, max_points):
    '''等间隔抽稀索引。散点太密时后画的组会把先画的完全盖住。'''
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


def symmetry_index(left, right):
    '''SI = (R - L) / (R + L) x 100%。正 = 偏右。

    用归一化差而不是直接用 R - L，是为了让力（上百 N）、
    力矩（上百 N·m）、关节角（十几度）能画在同一张图上。
    '''
    left = float(left)
    right = float(right)
    tot = left + right
    if not np.isfinite(tot) or abs(tot) < 1e-9:
        return np.nan
    return 100.0 * (right - left) / tot


def _safe_mean(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else np.nan


def sort_items(data):
    '''排序：定负载组按数值升序在前，等长/等速组按名字排在后。

    直接用 load_value 会在等长/等速组上拿到 nan，
    而 nan 参与比较的结果是未定义的，每次跑出来的顺序可能不一样。
    '''
    def key(item):
        k, rec = item
        v = rec.get('load_value', np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        if not np.isfinite(v):
            return (1, 0.0, str(k))
        return (0, v, str(k))

    return sorted(data.items(), key=key)


def _si_of(rec, spec):
    '''spec 是 ('force', None) / ('moment', base) / ('angle', base)。'''
    kind, base = spec
    if kind == 'force':
        return symmetry_index(_safe_mean(rec.get('force_l')),
                              _safe_mean(rec.get('force_r')))
    pair = rec.get('moments' if kind == 'moment' else 'angles', {}).get(base)
    if not pair:
        return np.nan
    return symmetry_index(pair['l'], pair['r'])


def build_si_table(data, moment_bases, angle_bases):
    '''组装 SI 矩阵。

    Returns
    -------
    row_labels : list[str]
    load_keys  : list[str]
    matrix     : (n_row, n_load) ndarray，单位 %
    '''
    specs = [('GRF share', ('force', None))]
    specs += [('{} moment'.format(PRETTY.get(b, b)), ('moment', b))
              for b in moment_bases]
    specs += [('{} angle'.format(PRETTY.get(b, b)), ('angle', b))
              for b in angle_bases]

    items = sort_items(data)
    keys = [k for k, _ in items]
    mat = np.full((len(specs), len(keys)), np.nan)
    for j, (_, rec) in enumerate(items):
        for i, (_, spec) in enumerate(specs):
            mat[i, j] = _si_of(rec, spec)
    return [s[0] for s in specs], keys, mat


# ============================================================
#  ① SI 热图
# ============================================================

def plot_si_heatmap(data, moment_bases, angle_bases, save_path=None,
                    vmax=None):
    '''通道 x 负载 的对称指数热图。

    发散色标以 0 为中心（蓝 = 偏左，红 = 偏右）。以 0 为中心是关键：
    用顺序色标会把 “偏左 5%” 和 “偏右 5%” 画成两种完全不同的颜色，
    而它们在物理上是同等程度的不对称。
    '''
    labels, keys, mat = build_si_table(data, moment_bases, angle_bases)
    if not keys:
        beauty_print('SI 热图：没有任何组的数据，跳过。', type='warning')
        return None

    finite = mat[np.isfinite(mat)]
    if vmax is None:
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmax = max(vmax, 5.0)

    fig, ax = plt.subplots(figsize=(1.1 * len(keys) + 4.5,
                                    0.55 * len(labels) + 2.6))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto')

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Load group')
    ax.set_title('Symmetry index  SI = (R - L)/(R + L)   [%]   '
                 'red = right-dominant')

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, 'n/a', ha='center', va='center',
                        fontsize=7, color='0.5')
                continue
            ax.text(j, i, '{:+.1f}'.format(v), ha='center', va='center',
                    fontsize=8,
                    color='white' if abs(v) > 0.6 * vmax else 'black')

    fig.colorbar(im, ax=ax, label='SI (%)')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print('[fig] SI 热图 -> {}'.format(save_path))
    return fig


# ============================================================
#  ② 不对称沿运动链的传递
# ============================================================

def plot_chain_transfer(data, save_path=None):
    '''横轴按解剖顺序 GRF -> Ankle -> Knee -> Hip，纵轴 SI。

    诊断逻辑：外力是一切力矩的源头。如果 GRF 的 SI 很小，
    而某一级关节的 SI 突然张开，那多出来的不对称就不可能来自力，
    只能来自力臂（COP 位置）或运动学。膝的力臂最短，对 COP 最敏感，
    所以 “纺锤形”（两端收拢、膝处张开）就是恒定 COP 的指纹。
    '''
    items = sort_items(data)
    if not items:
        return None

    xs = np.arange(len(CHAIN_ORDER))
    labels = ['GRF'] + [PRETTY.get(b, b) for b in CHAIN_ORDER[1:]]
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (key, rec) in enumerate(items):
        ys = []
        for name in CHAIN_ORDER:
            if name == '__force__':
                ys.append(_si_of(rec, ('force', None)))
            else:
                ys.append(_si_of(rec, ('moment', name)))
        color = cmap(i / max(len(items) - 1, 1))
        ax.plot(xs, ys, marker='o', color=color, label=str(key), linewidth=1.8)

    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Along the kinetic chain')
    ax.set_ylabel('SI (%)   positive = right-dominant')
    ax.set_title('Asymmetry transfer: ground reaction force -> joint moments')
    ax.grid(alpha=0.3)
    ax.legend(title='Load', fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print('[fig] 传递图 -> {}'.format(save_path))
    return fig


# ============================================================
#  ③ 左-右散点 + 恒等线
# ============================================================

def plot_lr_scatter(data, save_path=None, max_points_per_load=3000):
    '''逐帧左右力散点，x = 左侧，y = 右侧，叠 y = x 恒等线。

    这张图的唯一目的是把两种不对称分开：
      截距偏离 0（点云整体平移）-> 恒定偏置，典型于零点/站姿
      斜率偏离 1（点云旋转）  -> 增益不对称，典型于传感器饱和或真实重心转移
    S6 里 “静态截距正常但增量斜率 0.355” 说的就是后者，这里把它画出来。
    '''
    items = sort_items(data)
    if not items:
        return None

    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    lo, hi = np.inf, -np.inf
    for i, (key, rec) in enumerate(items):
        l = np.asarray(rec.get('force_l'), dtype=float)
        r = np.asarray(rec.get('force_r'), dtype=float)
        n = min(len(l), len(r))
        if n < 10:
            continue
        l, r = l[:n], r[:n]
        m = np.isfinite(l) & np.isfinite(r)
        l, r = l[m], r[m]
        if l.size < 10:
            continue
        if l.size > max_points_per_load:
            idx = np.linspace(0, l.size - 1, max_points_per_load).astype(int)
            l, r = l[idx], r[idx]

        color = cmap(i / max(len(items) - 1, 1))
        slope, intercept = np.polyfit(l, r, 1)
        ax.scatter(l, r, s=5, alpha=0.25, color=color)
        xx = np.array([l.min(), l.max()])
        ax.plot(xx, slope * xx + intercept, color=color, linewidth=1.8,
                label='{}  k={:.2f}  b={:+.0f}N'.format(key, slope, intercept))
        lo = min(lo, l.min(), r.min())
        hi = max(hi, l.max(), r.max())

    if not np.isfinite(lo):
        plt.close(fig)
        return None

    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1.2,
            label='perfect symmetry (y = x)')
    ax.set_xlabel('Left force (N)')
    ax.set_ylabel('Right force (N)')
    ax.set_title('Left vs right, frame by frame\n'
                 'offset from y=x -> constant bias;  slope != 1 -> gain asymmetry')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print('[fig] 左-右散点 -> {}'.format(save_path))
    return fig


# ============================================================
#  ④ 蝶形图
# ============================================================

def plot_butterfly(data, angle_bases, save_path=None, overlay_right=True):
    '''左侧向上、右侧镜像向下，横轴为归一化动作周期 0-100%。

    【为什么右侧全是负的】这是画法，不是数据：右侧画的是 -rm。
    屈膝/屈髋角本来就恒为正，所以左侧全在上、右侧全在下是预期行为。

    【怎么看对称性】镜像对称靠肉眼判断其实很不可靠（人对“上下两条曲线
    是否等高”的分辨能力远低于“两条线是否重合”）。所以 overlay_right=True
    时额外把右侧【不翻转】的虚线叠在左侧曲线上：完美对称 = 两线重合，
    中间的灰色填充就是 L-R 的差。下半部分保留镜像，用来看波形形状。

    阴影带是跨 cycle 的 ±1 SD，它是判定的【尺子】：标题里的
    max|L-R| / SD 小于 1 就说明左右差异还没同侧的 cycle 间波动大，
    那“不对称”只是噪声；大于 2 才值得当真。
    '''
    items = [(k, rec) for k, rec in sort_items(data)
             if rec.get('angle_curves')]
    if not items:
        beauty_print('蝶形图：没有任何组提供了逐 cycle 的关节角曲线，跳过。\n'
                     '常见原因：该组没有对应的 .mot 文件，或 mot 里缺左右成对的列。',
                     type='warning')
        return None

    bases = [b for b in angle_bases
             if any(b in rec['angle_curves'] for _, rec in items)]
    if not bases:
        return None

    n_row, n_col = len(bases), len(items)
    fig, axes = plt.subplots(n_row, n_col, sharex=True,
                             figsize=(2.6 * n_col + 1.5, 2.6 * n_row + 1.2),
                             squeeze=False)

    for i, base in enumerate(bases):
        for j, (key, rec) in enumerate(items):
            ax = axes[i][j]
            cur = rec['angle_curves'].get(base)
            if not cur:
                ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                        transform=ax.transAxes, color='0.5')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x = np.asarray(cur['grid'], dtype=float)
            L = np.asarray(cur['l'], dtype=float)
            R = np.asarray(cur['r'], dtype=float)
            lm, ls = np.nanmean(L, axis=0), np.nanstd(L, axis=0)
            rm, rs = np.nanmean(R, axis=0), np.nanstd(R, axis=0)

            ax.plot(x, lm, color=SIDE_COLOR['l'], linewidth=1.6, label='Left')
            ax.fill_between(x, lm - ls, lm + ls, color=SIDE_COLOR['l'],
                            alpha=0.20)
            # 镜像：右侧取负向下画
            ax.plot(x, -rm, color=SIDE_COLOR['r'], linewidth=1.6,
                    label='Right (mirrored)')
            ax.fill_between(x, -rm - rs, -rm + rs, color=SIDE_COLOR['r'],
                            alpha=0.20)
            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.6)

            # 右侧不翻转地叠在左侧上：判断“两线是否重合”比
            # 判断“上下是否等高”容易得多，灰色填充即 L-R。
            if overlay_right:
                ax.plot(x, rm, color=SIDE_COLOR['r'], linewidth=1.2,
                        linestyle='--', alpha=0.9, label='Right (overlaid)')
                ax.fill_between(x, lm, rm, color='0.35', alpha=0.30,
                                linewidth=0, label='L - R')

            diff = np.nanmax(np.abs(lm - rm)) if lm.size else np.nan
            # 用同侧 cycle 间波动做尺子：不归一化的话，1.5 度的差到底算大
            # 还是小根本无法回答——要看它相对于同一条腿自己的重复性。
            sd_ref = float(np.nanmean(np.concatenate([ls, rs]))) \
                if ls.size and rs.size else np.nan
            if np.isfinite(sd_ref) and sd_ref > 1e-6:
                ratio_txt = '{:.1f}x SD'.format(diff / sd_ref)
            else:
                ratio_txt = 'SD~0'
            ax.set_title('{}  {}\nmax |L-R| = {:.1f} deg = {}  (n={})'.format(
                key, PRETTY.get(base, base), diff, ratio_txt, L.shape[0]),
                fontsize=8)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel('{} (deg)'.format(PRETTY.get(base, base)))
            if i == n_row - 1:
                ax.set_xlabel('Cycle (%)')

    axes[0][0].legend(fontsize=7, loc='upper right')
    fig.suptitle('Butterfly plot: left up, right mirrored down; '
                 'dashed = right overlaid on left (gray fill = L - R). '
                 'Bands = +/-1 SD across cycles', y=0.995)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print('[fig] 蝶形图 -> {}'.format(save_path))
    return fig


# ============================================================
#  ⑤ 不对称性随合力 / 杆高的变化
# ============================================================

def _mode_marker(mode):
    '''等长 -> 方块，等速 -> 三角，定负载 -> 圆点。'''
    m = str(mode or '').lower()
    if m.startswith('isometric'):
        return 's'
    if m.startswith('isokinetic'):
        return '^'
    return 'o'


def _pair_arrays(rec):
    '''取出等长的左右力数组（同一切片，长度理论相同，仍做保护性截断）。'''
    l = np.asarray(rec.get('force_l'), dtype=float).ravel()
    r = np.asarray(rec.get('force_r'), dtype=float).ravel()
    n = min(l.size, r.size)
    if n < 1:
        return np.array([]), np.array([])
    return l[:n], r[:n]


def _group_force_mean(rec):
    l, r = _pair_arrays(rec)
    if l.size == 0:
        return np.nan
    return _safe_mean(l + r)


def plot_si_trend(data, moment_bases, save_path=None,
                  max_points_per_load=2500):
    '''不对称性的【趋势】图，三联。

    为什么要单独一张：热图给的是每组一个数，看不出规律；传递图看的是
    同一组沿运动链的走向。而“偏侧到底是恒定存在，还是被负载/深度
    催出来的”只有把 SI 放在连续横轴上才能回答，它也直接决定后面
    肌肉激活分析要不要按负载分别处理。

    左图 / 中图用【逐帧原始散点】，每组一种颜色 + 一条组内拟合直线。
    用组均值只有 9 个点，既看不到组内分布宽度，也无法区分
    “整组整体偏” 与 “只在大力/低位时偏”——而后者正是要判的东西。
    抽稀只为避免后画的组盖住先画的，不改变分布形状。

      左图 x = 逐帧合力 (N)，拟合斜率标注 %/kN，与
        example_symmetry_check.py 的 S7 组内斜率同一量纲，可直接对照。
      中图 x = 逐帧杆高 pos_l (m)。全程平移 = 恒定偏置；低位（图左侧）
        张开 = 偏侧发生在力最大的姿势上，指向高压下鞋垫欠读或重心转移。

    右图只能是组级的：ID 力矩的 SI 由整段均绝对值算出，一组一个数，
    没有逐帧版本。它回答另一个问题：若 GRF 的斜率≈0 而膝力矩的斜率
    明显为正，偏侧就不是外力带来的，而是随载荷放大的力臂（COP）误差。
    它的横轴用合力而不是标称配重：等长/等速组的 load_kg 是 nan，
    按 nan 做横轴会把这三组静默丢掉（在 plot_load_estimation 上已踩过一次）。
    '''
    items = sort_items(data)
    if not items:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8))
    ax_f, ax_h, ax_m = axes[0], axes[1], axes[2]

    n_height = 0
    for i, (key, rec) in enumerate(items):
        color = _group_color(i)
        marker = _mode_marker(rec.get('mode'))
        l, r = _pair_arrays(rec)
        if l.size < 30:
            continue
        tot = l + r
        m = np.isfinite(tot) & np.isfinite(l) & np.isfinite(r) \
            & (np.abs(tot) > 1.0)
        if int(m.sum()) < 30:
            continue
        f = tot[m]
        si = 100.0 * (r[m] - l[m]) / f

        # ---- 左图：SI vs 逐帧合力 ----
        idx = _thin(f.size, max_points_per_load)
        ax_f.scatter(f[idx], si[idx], s=6, alpha=0.22, color=color,
                     marker=marker, linewidths=0)
        k, b = _linfit(f, si)
        lab = str(key)
        if np.isfinite(k):
            xx = np.array([float(np.nanmin(f)), float(np.nanmax(f))])
            ax_f.plot(xx, k * xx + b, color=color, linewidth=2.2)
            lab = '{}  {:+.1f} %/kN'.format(key, k * 1000.0)
        ax_f.plot([], [], color=color, marker=marker, linewidth=2.2,
                  label=lab)

        # ---- 中图：SI vs 逐帧杆高 ----
        seg = rec.get('segment')
        if seg is None or not hasattr(seg, 'columns') \
                or 'pos_l' not in seg.columns:
            continue
        h_all = np.asarray(seg['pos_l'].values, dtype=float).ravel()
        n = min(h_all.size, l.size)
        if n < 30:
            continue
        hh, ll, rr = h_all[:n], l[:n], r[:n]
        tt = ll + rr
        mh = np.isfinite(hh) & np.isfinite(tt) & np.isfinite(ll) \
            & np.isfinite(rr) & (np.abs(tt) > 1.0)
        if int(mh.sum()) < 30:
            continue
        h = hh[mh]
        si_h = 100.0 * (rr[mh] - ll[mh]) / tt[mh]
        idx = _thin(h.size, max_points_per_load)
        ax_h.scatter(h[idx], si_h[idx], s=6, alpha=0.22, color=color,
                     marker=marker, linewidths=0)
        kh, bh = _linfit(h, si_h)
        lab_h = str(key)
        if np.isfinite(kh):
            xx = np.array([float(np.nanmin(h)), float(np.nanmax(h))])
            ax_h.plot(xx, kh * xx + bh, color=color, linewidth=2.2)
            lab_h = '{}  {:+.1f} %/m'.format(key, kh)
        ax_h.plot([], [], color=color, marker=marker, linewidth=2.2,
                  label=lab_h)
        n_height += 1

    ax_f.axhline(0.0, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
    ax_f.set_xlabel('Instantaneous total external force (N)')
    ax_f.set_ylabel('Frame-wise SI (%)   positive = right-dominant')
    ax_f.set_title('Asymmetry vs total force, frame by frame\n'
                   'circle = isotonic, triangle = isokinetic, '
                   'square = isometric')
    ax_f.grid(alpha=0.3)
    ax_f.legend(title='Load', fontsize=8, ncol=2)

    ax_h.axhline(0.0, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
    ax_h.set_xlabel('Bar height pos_l (m)   left = deep squat')
    ax_h.set_ylabel('Frame-wise SI (%)')
    ax_h.set_title('Asymmetry vs bar height, frame by frame\n'
                   'flat = constant bias;  tilted = depth-driven')
    ax_h.grid(alpha=0.3)
    if n_height:
        ax_h.legend(title='Load', fontsize=8, ncol=2)
    else:
        ax_h.text(0.5, 0.5, 'no pos_l in slices', ha='center', va='center',
                  transform=ax_h.transAxes, color='0.5')
        beauty_print('趋势图中间一联（SI vs 杆高）没画出来：切片数据里'
                     '没有 pos_l 列。\n请确认 collect_side_data 把 segment '
                     '一起放进了 data。', type='warning')

    # ---- 右图：组级的 ID 力矩 SI vs 组平均合力 ----
    specs = [('GRF', ('force', None))]
    specs += [(PRETTY.get(b, b) + ' moment', ('moment', b))
              for b in moment_bases]
    cmap = plt.cm.Dark2
    for i, (label, spec) in enumerate(specs):
        xs, ys, modes = [], [], []
        for key, rec in items:
            x = _group_force_mean(rec)
            y = _si_of(rec, spec)
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
                modes.append(rec.get('mode'))
        if len(xs) < 2:
            continue
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        order = np.argsort(xs)
        color = cmap(i % 8)

        slope_txt = ''
        k, b = _linfit(xs, ys)
        if np.isfinite(k):
            slope_txt = '   {:+.1f} %/kN'.format(k * 1000.0)
            xx = np.array([xs.min(), xs.max()])
            ax_m.plot(xx, k * xx + b, color=color, linewidth=1.0,
                      linestyle=':', alpha=0.8)

        ax_m.plot(xs[order], ys[order], color=color, linewidth=1.8,
                  alpha=0.9, label=label + slope_txt)
        for x, y, md in zip(xs, ys, modes):
            ax_m.plot([x], [y], marker=_mode_marker(md), color=color,
                      markersize=7, linestyle='none')

    ax_m.axhline(0.0, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
    ax_m.set_xlabel('Group mean total external force (N)')
    ax_m.set_ylabel('Group-level SI (%)')
    ax_m.set_title('Joint moment asymmetry vs load (group level)\n'
                   'GRF flat but knee rising -> moment-arm (COP) error')
    ax_m.grid(alpha=0.3)
    ax_m.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print('[fig] 不对称性趋势图 -> {}'.format(save_path))
    return fig


# ============================================================
#  一次画完五张
# ============================================================

def plot_symmetry_figures(data, moment_bases, angle_bases, out_dir=None,
                          show=True):
    '''依次画五张图。out_dir 为 None 时只显示不保存。'''
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _p(name):
        return os.path.join(out_dir, name) if out_dir else None

    figs = [
        plot_si_heatmap(data, moment_bases, angle_bases,
                        save_path=_p('symmetry_si_heatmap.png')),
        plot_chain_transfer(data, save_path=_p('symmetry_chain.png')),
        plot_lr_scatter(data, save_path=_p('symmetry_lr_scatter.png')),
        plot_butterfly(data, angle_bases,
                       save_path=_p('symmetry_butterfly.png')),
        plot_si_trend(data, moment_bases,
                      save_path=_p('symmetry_si_trend.png')),
    ]
    if show:
        plt.show()
    return [f for f in figs if f is not None]