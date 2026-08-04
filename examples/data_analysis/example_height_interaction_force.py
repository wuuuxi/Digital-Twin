"""
高度 - 交互力 散点图（疲劳实验机器人数据）

只画 example_fatigue_analysis.py 图9 右侧的「高度 - 交互力」散点图：
  - 横轴 = 真实高度 pos_l (m)，纵轴 = 交互力 force_total = force_l + force_r (N)；
  - 数据来自 fatigue_file.fatigue_data 的各组机器人文件（vl / cl / el）；
  - 每组用不同颜色；只画上升阶段（vel_l > 0）的点。

数据处理流程：
  1. 读取各组 robot_file：RobotProcessor.process(..., turn_position=False)，
     得到含 force_l/force_r/vel_l/pos_l/time 的 DataFrame，并加 force_total 列；
  2. 运动切片：DataAligner.cut_aligned_data 只保留有效运动片段（散点优先
     用切片数据，若某组切片为空则回退用该组原始数据）；
  3. 取上升阶段：vel_l > 0 的样本；
  4. 散点：每组 (pos_l, force_total)。

用法：
    python example_height_interaction_force.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from digitaltwin import Subject
from digitaltwin.data.robot_processor import RobotProcessor
from digitaltwin.analysis.alignment import DataAligner

# ============================================================
#  选项
# ============================================================
CONFIG_FILE = '../config/20260513_squat_FTS09.json'
FORCE_COLUMNS = ['force_l', 'force_r']   # 相加得到总交互力 (N)
VELOCITY_COLUMN = 'vel_l'                # 速度列 (m/s)
POSITION_COLUMN = 'pos_l'                # 位置（高度）列 (m)
ASCENDING_ONLY = True                    # 只画上升阶段 (vel_l > 0)
USE_CUT_DATA = True                      # 优先用切片数据（无则用原始）
GROUP_COLORS = plt.cm.tab10.colors


def _add_force_total(df):
    """添加 force_total = force_l + force_r 列。"""
    avail = [c for c in FORCE_COLUMNS if c in df.columns]
    if avail:
        df = df.copy()
        df['force_total'] = df[avail].sum(axis=1)
    return df


def _load_fatigue_groups(subject):
    """读取 fatigue_file 各组机器人数据，返回 [(label, robot_df), ...]。"""
    fatigue_cfg = subject.config.get('fatigue_file', {})
    fatigue_data = fatigue_cfg.get('fatigue_data', {})
    robot_subfolder = fatigue_cfg.get('robot_folder', 'fatigue/robot')
    robot_folder = os.path.join(subject.folder, robot_subfolder)

    groups = []
    for label, info in fatigue_data.items():
        robot_file = info.get('robot_file', '')
        # 疲劳数据固定 turn_position=False；load_weight 无意义，传 0 占位
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
    """对单组机器人数据做运动切片，返回切片 DataFrame 或 None。"""
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


def _draw_height_interaction_force(groups, out_path):
    """高度 - 交互力散点图：每组 (pos_l, force_total)，每组一种颜色。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Fatigue: Height - Interaction Force',
                 fontsize=14, fontweight='bold')
    any_point = False
    for i, (label, df) in enumerate(groups):
        if (POSITION_COLUMN not in df.columns
                or 'force_total' not in df.columns):
            continue
        sub = df
        if ASCENDING_ONLY and VELOCITY_COLUMN in df.columns:
            sub = df[df[VELOCITY_COLUMN] > 0]
        rh = np.asarray(sub[POSITION_COLUMN].values, dtype=float)
        rf = np.asarray(sub['force_total'].values, dtype=float)
        m = np.isfinite(rh) & np.isfinite(rf)
        if not np.any(m):
            continue
        ax.scatter(rh[m], rf[m], s=12, alpha=0.5,
                   color=GROUP_COLORS[i % len(GROUP_COLORS)],
                   edgecolors='none', label=str(label))
        any_point = True
        print(f'[{label}] 高度-交互力散点: 高度 '
              f'{rh[m].min():.3f} ~ {rh[m].max():.3f} m，'
              f'共 {int(np.sum(m))} 点')
    if not any_point:
        print('无有效高度-交互力数据。')
        plt.close(fig)
        return None
    ax.set_xlabel('Height [m]')
    ax.set_ylabel('Interaction force (L+R) [N]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')
    return fig


def main():
    subject = Subject(CONFIG_FILE)

    groups = _load_fatigue_groups(subject)
    if not groups:
        print('未能加载任何 fatigue 数据组，退出。')
        return

    # 散点优先用切片数据；某组切片为空则回退用该组原始数据
    draw_groups = []
    for label, df in groups:
        chosen = _cut_group(df) if USE_CUT_DATA else None
        if chosen is None or len(chosen) == 0:
            chosen = df
        draw_groups.append((label, chosen))

    out_dir = os.path.join(subject.result_folder, 'fatigue')
    os.makedirs(out_dir, exist_ok=True)

    _draw_height_interaction_force(
        draw_groups,
        os.path.join(out_dir, 'fatigue_height_interaction_force.png'))

    plt.show()


if __name__ == '__main__':
    main()