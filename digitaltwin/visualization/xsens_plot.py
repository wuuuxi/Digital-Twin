"""
Xsens 关节角通用可视化模块。

提供 5 类通用绘图函数，同时支持固定负载和固定+变负载场景：
  1. plot_alignment           — 对齐可视化（归一化曲线 + 关节角）
  2. plot_movement_segments   — 运动切片（vel/pos/force/emg/关节角）
  3. plot_position_scatter    — 位置散点（Vel/EMG/关节角，向心/离心分色）
  4. plot_joint_scatter_lr    — 多关节角左右散点图（不同负载不同颜色）
  5. plot_joint_bar_lr        — 关节角均值柱状图 + 左右差异

每个函数接收统一的 data_groups 参数：
  data_groups = [
      {'key': '20', 'data': result_dict, 'is_vload': False},
      {'key': 'VL_label', 'data': vload_result_dict, 'is_vload': True},
  ]
从而同时支持仅固定负载和固定+变负载的场景。
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOAD_COLORS = plt.cm.tab10.colors
VLOAD_COLORS = plt.cm.Set2.colors


def _get_motion_defaults(target_motion):
    """根据运动类型返回默认参数"""
    if target_motion == 'benchpress':
        return {
            'xsens_joint': 'elbow_flex_r',
            'target_emg': ['emg_TriLat', 'emg_PMSte', 'emg_DelAnt', 'emg_Bic'],
            'joint_bases': ['arm_flex', 'arm_add', 'arm_rot', 'elbow_flex'],
        }
    return {
        'xsens_joint': 'knee_angle_r',
        'target_emg': ['emg_FibLon', 'emg_VL', 'emg_RF'],
        'joint_bases': ['hip_flexion', 'hip_adduction', 'hip_rotation',
                        'knee_angle', 'ankle_angle', 'subtalar_angle'],
    }


def build_data_groups(fixed_results, vload_results=None):
    """构建统一的 data_groups 列表"""
    groups = []
    for k in sorted(fixed_results.keys(), key=lambda x: float(x)):
        groups.append({'key': k, 'data': fixed_results[k], 'is_vload': False})
    if vload_results:
        for k in vload_results.keys():
            groups.append({'key': k, 'data': vload_results[k], 'is_vload': True})
    return groups


def _title(g):
    return f'VLoad: {g["key"]}' if g['is_vload'] else f'{g["key"]}kg'


def _title_color(g):
    return 'green' if g['is_vload'] else 'black'


# ==============================================================
#  图 1: 对齐可视化
# ==============================================================

def plot_alignment(data_groups, target_emg, xsens_joint):
    """每个负载一张子图，所有归一化曲线 + 关节角"""
    n = len(data_groups)
    if n == 0: return
    xsens_col = f'xsens_{xsens_joint}'

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1: axes = [axes]

    for i, g in enumerate(data_groups):
        ax = axes[i]
        res = g['data']
        rd = res.get('robot_data')
        if rd is not None:
            if 'force_l' in rd.columns:
                ax.plot(rd['time'], rd['force_l'] / (abs(rd['force_l']).max() + 1e-10),
                        label='Robot Force', alpha=0.7)
            if 'pos_l' in rd.columns:
                ax.plot(rd['time'], rd['pos_l'] - rd['pos_l'].iloc[-1],
                        label='Robot Position', alpha=0.7)
            if 'vel_l' in rd.columns:
                ax.plot(rd['time'], rd['vel_l'], label='Robot Velocity', alpha=0.7)

        ad = res.get('aligned_data')
        if ad is not None:
            for mc in target_emg:
                if mc in ad.columns:
                    ax.plot(ad['time'], ad[mc], '--',
                            label=f'EMG {mc.replace("emg_", "")}', alpha=0.7)
            if xsens_col in ad.columns:
                xv = ad[xsens_col].values
                xmax = np.nanmax(np.abs(xv)) + 1e-10
                ax.plot(ad['time'], xv / xmax, ':',
                        label=f'{xsens_joint} (norm)', alpha=0.8,
                        linewidth=1.5, color='purple')

        ax.set_title(_title(g), color=_title_color(g))
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalized Signal')
        ax.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================
#  图 2: 运动切片
# ==============================================================

def plot_movement_segments(data_groups, muscle_col, xsens_joint):
    """运动切片 5行 × N列"""
    n = len(data_groups)
    if n == 0: return
    xsens_col = f'xsens_{xsens_joint}'
    cols = ['vel_l', 'pos_l', 'force_l', muscle_col, xsens_col]
    labels = ['Velocity', 'Position', 'Force',
              muscle_col.replace('emg_', '') + ' Activation',
              f'{xsens_joint} (deg)']
    nr = len(cols)

    # 全局 Y 范围
    gy = {c: [float('inf'), float('-inf')] for c in cols}
    for g in data_groups:
        cd = g['data'].get('cutted_data')
        if cd is None: continue
        for c in cols:
            if c in cd.columns:
                v = cd[c].dropna().values
                if len(v): gy[c] = [min(gy[c][0], np.min(v)), max(gy[c][1], np.max(v))]
    for c in gy:
        if gy[c][0] != float('inf'):
            r = (gy[c][1] - gy[c][0]) * 0.1; gy[c][0] -= r; gy[c][1] += r

    fig, axes = plt.subplots(nr, n, figsize=(17, 8))
    if n == 1: axes = axes.reshape(nr, 1)
    cc = plt.cm.tab10.colors

    for ci, g in enumerate(data_groups):
        ad, cd = g['data'].get('aligned_data'), g['data'].get('cutted_data')
        for ri, (col, lab) in enumerate(zip(cols, labels)):
            ax = axes[ri, ci]
            if ad is not None and col in ad.columns and 'time' in ad.columns:
                ax.plot(ad['time'], ad[col], 'k-', alpha=0.3)
            if cd is not None and col in cd.columns and 'time' in cd.columns and 'cycle_id' in cd.columns:
                for cyi in range(int(cd['cycle_id'].max()) + 1):
                    seg = cd[cd['cycle_id'] == cyi]; clr = cc[cyi % len(cc)]
                    su = seg[seg['movement_type'] == 'upward'] if 'movement_type' in seg.columns else seg
                    sd = seg[seg['movement_type'] == 'downward'] if 'movement_type' in seg.columns else pd.DataFrame()
                    if col in su.columns and len(su): ax.plot(su['time'], su[col], '-', color=clr, linewidth=2)
                    if len(sd) and col in sd.columns: ax.plot(sd['time'], sd[col], '--', color=clr, linewidth=2, alpha=0.6)
            if col in gy and gy[col][0] != float('inf'): ax.set_ylim(gy[col])
            ax.set_ylabel(lab); ax.set_xlabel('Time (s)'); ax.grid(True, alpha=0.3)
            if cd is not None and 'time' in cd.columns and len(cd):
                ax.set_xlim(cd['time'].min() - 1, cd['time'].max() + 1)
            if ri == 0:
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.set_title(_title(g), fontsize=9, color=_title_color(g))
    fig.tight_layout()
    return fig


# ==============================================================
#  图 3: 位置散点 (Vel / EMG / 关节角)
# ==============================================================

def plot_position_scatter(data_groups, muscle_col, xsens_joint):
    """3行 × N列：Pos-Vel / Pos-EMG / Pos-JointAngle，向心/离心分色"""
    n = len(data_groups)
    if n == 0: return
    xsens_col = f'xsens_{xsens_joint}'

    gv, ge, gx = ([float('inf'), float('-inf')] for _ in range(3))
    for g in data_groups:
        cd = g['data'].get('cutted_data')
        if cd is None: continue
        if 'vel_l' in cd.columns:
            v = np.abs(cd['vel_l'].values); gv = [min(gv[0], np.min(v)), max(gv[1], np.max(v))]
        if muscle_col in cd.columns:
            e = np.abs(cd[muscle_col].values); ge = [min(ge[0], np.min(e)), max(ge[1], np.max(e))]
        if xsens_col in cd.columns:
            xv = cd[xsens_col].dropna().values
            if len(xv): gx = [min(gx[0], np.min(xv)), max(gx[1], np.max(xv))]
    for g_ in [gv, ge]:
        if g_[0] != float('inf'): r = (g_[1]-g_[0])*0.1; g_[0] = max(0, g_[0]-r); g_[1] += r
        else: g_[0], g_[1] = 0, 1
    if gx[0] != float('inf'): r = (gx[1]-gx[0])*0.1; gx[0] -= r; gx[1] += r
    else: gx[0], gx[1] = 0, 1

    fig, axes = plt.subplots(3, n, figsize=(15, 6))
    if n == 1: axes = axes.reshape(3, 1)

    for ci, g in enumerate(data_groups):
        cd = g['data'].get('cutted_data')
        if cd is None or 'pos_l' not in cd.columns or 'vel_l' not in cd.columns:
            for r in range(3): axes[r, ci].set_visible(False); continue
        pos, vel = cd['pos_l'].values, cd['vel_l'].values
        pm, nm = vel > 0, vel <= 0

        ax1 = axes[0, ci]
        ax1.scatter(pos[pm], np.abs(vel[pm]), alpha=0.6, s=10, label='Con')
        ax1.scatter(pos[nm], np.abs(vel[nm]), alpha=0.6, s=10, label='Ecc')
        ax1.set_ylim(gv); ax1.set_title(_title(g), fontsize=9, color=_title_color(g))
        ax1.set_xlabel('Position')
        if ci == 0: ax1.set_ylabel('Velocity')
        if ci == n-1: ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1, ci]
        if muscle_col in cd.columns:
            emg = cd[muscle_col].values
            ax2.scatter(pos[pm], np.abs(emg[pm]), alpha=0.6, s=10, label='Con')
            ax2.scatter(pos[nm], np.abs(emg[nm]), alpha=0.6, s=10, label='Ecc')
        ax2.set_ylim(ge); ax2.set_xlabel('Position')
        if ci == 0: ax2.set_ylabel('EMG Activation')
        if ci == n-1: ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2, ci]
        if xsens_col in cd.columns:
            xv = cd[xsens_col].values; valid = ~np.isnan(xv)
            ax3.scatter(pos[pm & valid], xv[pm & valid], alpha=0.6, s=10, label='Con')
            ax3.scatter(pos[nm & valid], xv[nm & valid], alpha=0.6, s=10, label='Ecc')
        ax3.set_ylim(gx); ax3.set_xlabel('Position')
        if ci == 0: ax3.set_ylabel(f'{xsens_joint} (deg)')
        if ci == n-1: ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================
#  图 4: 多关节角左右散点图
# ==============================================================

def plot_joint_scatter_lr(fixed_results, joint_bases, vload_results=None):
    """2行 × N列：上行 Right，下行 Left。不同负载不同颜色。"""
    nj = len(joint_bases)
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys()) if vload_results else []

    fig, axes = plt.subplots(2, nj, figsize=(3 * nj, 5), squeeze=False)
    fig.suptitle('Position vs Joint Angles (Top: Right, Bottom: Left)',
                 fontsize=14, fontweight='bold')

    for ji, jbase in enumerate(joint_bases):
        for ri, side in enumerate(['_r', '_l']):
            ax = axes[ri, ji]; col = f'xsens_{jbase}{side}'
            for j, vk in enumerate(vload_keys):
                cd = vload_results[vk].get('cutted_data')
                if cd is None or 'pos_l' not in cd.columns or col not in cd.columns: continue
                p, v = cd['pos_l'].values, cd[col].values; m = ~np.isnan(v)
                if np.any(m): ax.scatter(p[m], v[m], color=VLOAD_COLORS[j % len(VLOAD_COLORS)],
                    alpha=0.1, s=3, marker='^', label=f'VL: {vk}' if ji == 0 else None)
            for i, lw in enumerate(fixed_keys):
                cd = fixed_results[lw].get('cutted_data')
                if cd is None or 'pos_l' not in cd.columns or col not in cd.columns: continue
                p, v = cd['pos_l'].values, cd[col].values; m = ~np.isnan(v)
                if np.any(m): ax.scatter(p[m], v[m], color=LOAD_COLORS[i % len(LOAD_COLORS)],
                    alpha=0.5, s=10, label=f'{lw} kg' if ji == 0 else None)
            if ri == 0: ax.set_title(jbase, fontsize=10)
            ax.set_xlabel('Position (m)', fontsize=8); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
            if ji == 0:
                ax.set_ylabel(('Right' if side == '_r' else 'Left') + ' (deg)', fontsize=9)
                ax.legend(fontsize=6, loc='best')
    fig.tight_layout()
    return fig


# ==============================================================
#  图 5: 关节角均值柱状图 + 左右差异
# ==============================================================

def plot_joint_bar_lr(fixed_results, joint_bases, vload_results=None):
    """3行 × N列：Row1 Right，Row2 Left，Row3 |R-L| 差异。柱状图 ± std。"""
    nj = len(joint_bases)
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys()) if vload_results else []

    fig, axes = plt.subplots(3, nj, figsize=(3 * nj, 6), squeeze=False)
    fig.suptitle('Mean Joint Angle by Load', fontsize=14, fontweight='bold')

    def _collect(cd, col):
        if cd is None or col not in cd.columns: return None
        return cd[col].dropna().values

    # Row 0 & 1: Right / Left
    for ji, jbase in enumerate(joint_bases):
        for ri, side in enumerate(['_r', '_l']):
            ax = axes[ri, ji]; col = f'xsens_{jbase}{side}'
            bm, bs, bl, bc, bh = [], [], [], [], []
            for i, lw in enumerate(fixed_keys):
                v = _collect(fixed_results[lw].get('cutted_data'), col)
                if v is None or len(v) == 0: continue
                bm.append(np.mean(v)); bs.append(np.std(v)); bl.append(f'{lw} kg')
                bc.append(LOAD_COLORS[i % len(LOAD_COLORS)]); bh.append('')
            for j, vk in enumerate(vload_keys):
                v = _collect(vload_results[vk].get('cutted_data'), col)
                if v is None or len(v) == 0: continue
                bm.append(np.mean(v)); bs.append(np.std(v)); bl.append(f'VL:{vk}')
                bc.append(VLOAD_COLORS[j % len(VLOAD_COLORS)]); bh.append('//')
            _draw_bar(ax, bm, bs, bl, bc, bh)
            if ri == 0: ax.set_title(jbase, fontsize=10)
            if ji == 0: ax.set_ylabel(('Right' if side == '_r' else 'Left') + ' (deg)', fontsize=9)

    # Row 2: |R - L| difference
    for ji, jbase in enumerate(joint_bases):
        ax = axes[2, ji]; col_r, col_l = f'xsens_{jbase}_r', f'xsens_{jbase}_l'
        bm, bs, bl, bc, bh = [], [], [], [], []
        for i, lw in enumerate(fixed_keys):
            cd = fixed_results[lw].get('cutted_data')
            if cd is None or col_r not in cd.columns or col_l not in cd.columns: continue
            rv, lv = cd[col_r].values, cd[col_l].values; m = ~np.isnan(rv) & ~np.isnan(lv)
            if not np.any(m): continue
            d = np.abs(rv[m] - lv[m])
            bm.append(np.mean(d)); bs.append(np.std(d)); bl.append(f'{lw} kg')
            bc.append(LOAD_COLORS[i % len(LOAD_COLORS)]); bh.append('')
        for j, vk in enumerate(vload_keys):
            cd = vload_results[vk].get('cutted_data')
            if cd is None or col_r not in cd.columns or col_l not in cd.columns: continue
            rv, lv = cd[col_r].values, cd[col_l].values; m = ~np.isnan(rv) & ~np.isnan(lv)
            if not np.any(m): continue
            d = np.abs(rv[m] - lv[m])
            bm.append(np.mean(d)); bs.append(np.std(d)); bl.append(f'VL:{vk}')
            bc.append(VLOAD_COLORS[j % len(VLOAD_COLORS)]); bh.append('//')
        _draw_bar(ax, bm, bs, bl, bc, bh)
        if ji == 0: ax.set_ylabel('|R - L| (deg)', fontsize=9)
    fig.tight_layout()
    return fig


# ==============================================================
#  图 6: 关节角速度左右散点图
# ==============================================================

def plot_joint_vel_scatter_lr(fixed_results, joint_bases, vload_results=None):
    """2行 × N列：上行 Right vel，下行 Left vel。不同负载不同颜色。"""
    nj = len(joint_bases)
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys()) if vload_results else []

    fig, axes = plt.subplots(2, nj, figsize=(3 * nj, 5), squeeze=False)
    fig.suptitle('Position vs Joint Angular Velocity (Top: Right, Bottom: Left)',
                 fontsize=14, fontweight='bold')

    for ji, jbase in enumerate(joint_bases):
        for ri, side in enumerate(['_r', '_l']):
            ax = axes[ri, ji]; col = f'xsens_vel_{jbase}{side}'
            for j, vk in enumerate(vload_keys):
                cd = vload_results[vk].get('cutted_data')
                if cd is None or 'pos_l' not in cd.columns or col not in cd.columns: continue
                p, v = cd['pos_l'].values, cd[col].values; m = ~np.isnan(v)
                if np.any(m): ax.scatter(p[m], v[m], color=VLOAD_COLORS[j % len(VLOAD_COLORS)],
                    alpha=0.1, s=3, marker='^', label=f'VL: {vk}' if ji == 0 else None)
            for i, lw in enumerate(fixed_keys):
                cd = fixed_results[lw].get('cutted_data')
                if cd is None or 'pos_l' not in cd.columns or col not in cd.columns: continue
                p, v = cd['pos_l'].values, cd[col].values; m = ~np.isnan(v)
                if np.any(m): ax.scatter(p[m], v[m], color=LOAD_COLORS[i % len(LOAD_COLORS)],
                    alpha=0.5, s=10, label=f'{lw} kg' if ji == 0 else None)
            if ri == 0: ax.set_title(jbase, fontsize=10)
            ax.set_xlabel('Position (m)', fontsize=8); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
            if ji == 0:
                ax.set_ylabel(('Right' if side == '_r' else 'Left') + ' vel (deg/s)', fontsize=9)
                ax.legend(fontsize=6, loc='best')
    fig.tight_layout()
    return fig


# ==============================================================
#  图 7: 关节角速度均值柱状图 + 左右差异
# ==============================================================

def plot_joint_vel_bar_lr(fixed_results, joint_bases, vload_results=None):
    """3行 × N列：Row1 Right vel，Row2 Left vel，Row3 |R-L| vel 差异。"""
    nj = len(joint_bases)
    fixed_keys = sorted(fixed_results.keys(), key=lambda x: float(x))
    vload_keys = list(vload_results.keys()) if vload_results else []

    fig, axes = plt.subplots(3, nj, figsize=(3 * nj, 6), squeeze=False)
    fig.suptitle('Mean Joint Angular Velocity by Load\n'
                 '(Row 1: Right, Row 2: Left, Row 3: |R-L| Difference)',
                 fontsize=14, fontweight='bold')

    def _collect(cd, col):
        if cd is None or col not in cd.columns: return None
        return cd[col].dropna().values

    for ji, jbase in enumerate(joint_bases):
        for ri, side in enumerate(['_r', '_l']):
            ax = axes[ri, ji]; col = f'xsens_vel_{jbase}{side}'
            bm, bs, bl, bc, bh = [], [], [], [], []
            for i, lw in enumerate(fixed_keys):
                v = _collect(fixed_results[lw].get('cutted_data'), col)
                if v is None or len(v) == 0: continue
                bm.append(np.mean(np.abs(v))); bs.append(np.std(np.abs(v))); bl.append(f'{lw} kg')
                bc.append(LOAD_COLORS[i % len(LOAD_COLORS)]); bh.append('')
            for j, vk in enumerate(vload_keys):
                v = _collect(vload_results[vk].get('cutted_data'), col)
                if v is None or len(v) == 0: continue
                bm.append(np.mean(np.abs(v))); bs.append(np.std(np.abs(v))); bl.append(f'VL:{vk}')
                bc.append(VLOAD_COLORS[j % len(VLOAD_COLORS)]); bh.append('//')
            _draw_bar(ax, bm, bs, bl, bc, bh)
            if ri == 0: ax.set_title(jbase, fontsize=10)
            if ji == 0: ax.set_ylabel(('Right' if side == '_r' else 'Left') + ' |vel| (deg/s)', fontsize=9)

    for ji, jbase in enumerate(joint_bases):
        ax = axes[2, ji]; col_r, col_l = f'xsens_vel_{jbase}_r', f'xsens_vel_{jbase}_l'
        bm, bs, bl, bc, bh = [], [], [], [], []
        for i, lw in enumerate(fixed_keys):
            cd = fixed_results[lw].get('cutted_data')
            if cd is None or col_r not in cd.columns or col_l not in cd.columns: continue
            rv, lv = cd[col_r].values, cd[col_l].values; m = ~np.isnan(rv) & ~np.isnan(lv)
            if not np.any(m): continue
            d = np.abs(np.abs(rv[m]) - np.abs(lv[m]))
            bm.append(np.mean(d)); bs.append(np.std(d)); bl.append(f'{lw} kg')
            bc.append(LOAD_COLORS[i % len(LOAD_COLORS)]); bh.append('')
        for j, vk in enumerate(vload_keys):
            cd = vload_results[vk].get('cutted_data')
            if cd is None or col_r not in cd.columns or col_l not in cd.columns: continue
            rv, lv = cd[col_r].values, cd[col_l].values; m = ~np.isnan(rv) & ~np.isnan(lv)
            if not np.any(m): continue
            d = np.abs(np.abs(rv[m]) - np.abs(lv[m]))
            bm.append(np.mean(d)); bs.append(np.std(d)); bl.append(f'VL:{vk}')
            bc.append(VLOAD_COLORS[j % len(VLOAD_COLORS)]); bh.append('//')
        _draw_bar(ax, bm, bs, bl, bc, bh)
        if ji == 0: ax.set_ylabel('|R-L| vel diff (deg/s)', fontsize=9)
    fig.tight_layout()
    return fig


# ==============================================================
#  内部工具
# ==============================================================

def _draw_bar(ax, bm, bs, bl, bc, bh):
    """柱状图内部工具"""
    if not bm:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    x = np.arange(len(bm))
    bars = ax.bar(x, bm, yerr=bs, capsize=4, color=bc, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bi, h in enumerate(bh): bars[bi].set_hatch(h)
    for xi, m, s in zip(x, bm, bs):
        ax.text(xi, m + s + 0.3, f'{m:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(bl, fontsize=6, rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y'); ax.tick_params(labelsize=7)