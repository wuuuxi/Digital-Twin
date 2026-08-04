"""
固定负载 vs 变负载 —— 切片数据散点对比（横轴均为高度 position）

一张 1x3 大图，三个子图横轴均为高度 pos_l，纵轴分别为：
  子图 1：速度 vel_l (m/s)
  子图 2：力 force = force_l + force_r (N)
  子图 3：功率 power = (force_l + force_r) * vel_l (W)
三个子图均为散点图：
  - 固定负载：每个负载一种较浅的颜色（Pastel1），圆点
  - 变负载：  每组实验一种稍深的颜色（Dark2），三角点

数据来源：
  - 固定负载：pipeline.run()      -> pipeline.results
  - 变负载：  pipeline.run_vload() -> {label: {'cutted_data', ...}}

用法：
    python example_fixed_vs_vload_scatter.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline


# ============================================================
#  选项
# ============================================================
CONFIG_FILE = '../config/20250409_squat_NCMP001.json'
MOVEMENT_TYPES = ['upward']          # 只画上升阶段；None = 不过滤

POSITION_COLUMN = 'pos_l'            # 横轴：高度 (m)
VELOCITY_COLUMN = 'vel_l'            # 速度列 (m/s)
FORCE_COLUMNS = ['force_l', 'force_r']  # 相加得到总交互力 (N)

POINT_SIZE = 4                       # 散点大小（更小）
FIXED_COLORS = plt.cm.Pastel1.colors  # 固定负载：较浅，逐负载不同
VLOAD_COLORS = plt.cm.Dark2.colors    # 变负载：稍深，逐组不同


def _filter_movement(df, movement_types):
    """按运动阶段过滤（缺少 movement_type 列时原样返回）。"""
    if movement_types is not None and 'movement_type' in df.columns:
        df = df[df['movement_type'].isin(movement_types)]
    return df


def _prepare(cd, movement_types):
    """整理单组切片数据：拼接 / 过滤 / 补充 force_total 与 power。"""
    if cd is None:
        return None
    if isinstance(cd, list):
        cd = pd.concat(cd, ignore_index=True)
    if len(cd) == 0:
        return None
    df = _filter_movement(cd.copy(), movement_types)
    if len(df) == 0:
        return None
    avail = [c for c in FORCE_COLUMNS if c in df.columns]
    if avail:
        df['force_total'] = df[avail].sum(axis=1)
        df['power'] = (df['force_total'] * df[VELOCITY_COLUMN]
                       if VELOCITY_COLUMN in df.columns else np.nan)
    else:
        df['force_total'] = np.nan
        df['power'] = np.nan
    return df


def _scatter_group(axes, df, color, label, marker):
    """在三个子图上画一组数据：pos-vel / pos-force / pos-power。"""
    pos = df[POSITION_COLUMN].values
    panels = [VELOCITY_COLUMN, 'force_total', 'power']
    for ax, col in zip(axes, panels):
        if col in df.columns:
            ax.scatter(pos, df[col].values, s=POINT_SIZE, alpha=0.6,
                       color=color, marker=marker, label=label)


def main():
    subject = Subject(CONFIG_FILE)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # ---- 固定负载 ----
    pipeline.run(include_xsens=False)
    fixed_results = pipeline.results

    # ---- 变负载 ----
    vload_results = pipeline.run_vload()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 固定负载：每个负载一种较浅的颜色，圆点
    for li, lw in enumerate(fixed_results.keys()):
        df = _prepare(fixed_results[lw].get('cutted_data'), MOVEMENT_TYPES)
        if df is not None:
            _scatter_group(axes, df, FIXED_COLORS[li % len(FIXED_COLORS)],
                           f'{lw}kg', 'o')

    # 变负载：每组一种稍深的颜色，三角点
    if vload_results:
        for vi, (label, res) in enumerate(vload_results.items()):
            df = _prepare(res.get('cutted_data'), MOVEMENT_TYPES)
            if df is not None:
                _scatter_group(axes, df, VLOAD_COLORS[vi % len(VLOAD_COLORS)],
                               f'VL: {label}', '^')

    titles = [
        ('Velocity (m/s)', 'Position - Velocity'),
        ('Force (N)',      'Position - Force'),
        ('Power (W)',      'Position - Power'),
    ]
    for ax, (ylabel, title) in zip(axes, titles):
        ax.set_xlabel('Position (m)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle('Fixed vs Variable load \u2014 scatter by position',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_dir = os.path.join(subject.result_folder, 'heatmap')
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'fixed_vs_vload_pos_scatter.png')
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')

    plt.show()


if __name__ == '__main__':
    main()