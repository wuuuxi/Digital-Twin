"""
变负载结果对比示例

对比 variable_load_file 中的变负载实际结果（高度、EMG）与预期结果
（vload_file 中的 Height / Load / Activation 规划值）。

可选：再叠加 heatmap 拟合的曲面对该工况下激活的预测值
（在每个规划点 (Height, Load) 上用 RBF / 平滑单调投影曲面预测）。

除了画出各预测曲线与实测散点的对比之外，还会在每个实测数据点上
计算三种预测方式（Expected / RBF / Monotonic）与实测 EMG 的 RMSE。
由于实测数据未记录瞬时负载，这里使用规划负载在该高度上的插值
作为实测点的负载，RBF / Monotonic 预测也在 (h_actual, l_planned) 上评估。

选项：
  - MOVEMENT_TYPES: 只用上升 ['upward']、只用下降 ['downward']、或两者
    ['upward', 'downward']；None 表示不过滤。
  - HEATMAP_MODE: 'both' | 'rbf' | 'monotonic' | 'none'，控制叠加哪种
    heatmap 预测曲线。

用法：
    python example_vload_result.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from digitaltwin import Subject, MultiLoadPipeline
from digitaltwin.analysis.rbf_fitting import predict_at


# ---- 选项 ----
# 与 example_heatmap.py 一致：选取实测数据使用的运动阶段
MOVEMENT_TYPES = ['upward']

# heatmap 预测叠加模式：'both' / 'rbf' / 'monotonic' / 'none'
HEATMAP_MODE = 'both'


# =========================================================
#  工具函数
# =========================================================

def filter_movement_types(df, movement_types):
    """按 movement_type 列过滤切片数据。"""
    if df is None or movement_types is None:
        return df
    if 'movement_type' not in df.columns:
        return df
    return df[df['movement_type'].isin(movement_types)]


def load_planned_vload(subject, vload_file):
    """从 subject.vload_load_folder 加载规划好的 vload csv。"""
    if not vload_file:
        return None
    path = os.path.join(subject.vload_load_folder, vload_file)
    if not os.path.exists(path):
        print(f'  规划文件不存在: {path}')
        return None
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]
    return df


def _heatmap_param_dir(subject):
    """返回 heatmap 拟合参数所在目录。"""
    return (subject.muscle_folder if subject.load_previous_data
            else os.path.join(subject.result_folder, 'heatmap/params'))


def load_rbf_params(subject, muscle):
    """加载原始 RBF 参数 ({muscle}_rbf_params.pkl)。"""
    path = os.path.join(_heatmap_param_dir(subject),
                        f'{muscle}_rbf_params.pkl')
    if not os.path.exists(path):
        print(f'  未找到 {muscle} 的 RBF 参数: {path}')
        return None
    with open(path, 'rb') as f:
        p = pickle.load(f)
    p = dict(p)
    p['monotonic_load'] = False
    return p


def load_monotonic_params(subject, muscle):
    """加载平滑单调投影后的完整 params ({muscle}_monotonic_params.pkl)。"""
    path = os.path.join(_heatmap_param_dir(subject),
                        f'{muscle}_monotonic_params.pkl')
    if not os.path.exists(path):
        print(f'  未找到 {muscle} 的 monotonic 参数: {path}')
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_heatmap_params_by_mode(subject, muscle, mode):
    """
    根据 HEATMAP_MODE 加载需要的预测参数。

    Returns
    -------
    list of (key, label, params, color, linestyle)
        key  -- 'rbf' 或 'monotonic'，用于 RMSE 表中的识别。
        label, params, color, linestyle -- 画图用。
    """
    if mode == 'none' or muscle is None:
        return []

    overlays = []
    if mode in ('rbf', 'both'):
        p = load_rbf_params(subject, muscle)
        if p is not None:
            overlays.append(('rbf', 'Heatmap (RBF)', p, 'C3', '--'))
    if mode in ('monotonic', 'both'):
        p = load_monotonic_params(subject, muscle)
        if p is not None:
            overlays.append(
                ('monotonic', 'Heatmap (monotonic)', p, 'C2', '-.'))
    return overlays


def _interp_sorted(x_query, x_known, y_known):
    """对 (x_known, y_known) 按 x_known 排序后做线性插值。"""
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    order = np.argsort(x_known)
    return np.interp(x_query, x_known[order], y_known[order])


def _rmse(pred, actual):
    """志错函数安全版本：志 nan 后计算 RMSE，无有效点返回 NaN。"""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() == 0:
        return float('nan'), 0
    diff = pred[mask] - actual[mask]
    return float(np.sqrt(np.mean(diff ** 2))), int(mask.sum())


def compute_rmse_at_actual_points(cutted, planned_df, heatmap_overlays,
                                  target_muscle):
    """
    在实测数据点上计算三种预测与实测 EMG 的 RMSE。

    实测数据不记录瞬时负载，这里使用规划负载 Load(Height) 在实测
    高度上的插值作为该点的负载；Expected 预测为 Activation(Height) 插值。

    Returns
    -------
    dict
        {key: (rmse, n_points)}。key 取值 'expected', 'rbf', 'monotonic'。
    """
    emg_col = f'emg_{target_muscle}'
    rmse_dict = {}

    if (cutted is None
            or emg_col not in cutted.columns
            or 'pos_l' not in cutted.columns
            or planned_df is None):
        return rmse_dict

    h_actual = cutted['pos_l'].values
    emg_actual = cutted[emg_col].values

    # Expected: 插值规划 Activation(Height)
    if {'Height', 'Activation'}.issubset(planned_df.columns):
        a_expected = _interp_sorted(
            h_actual, planned_df['Height'].values,
            planned_df['Activation'].values)
        rmse_dict['expected'] = _rmse(a_expected, emg_actual)

    # 在实测高度上插值出规划负载（供 heatmap 预测使用）
    if {'Height', 'Load'}.issubset(planned_df.columns) and heatmap_overlays:
        l_at_actual = _interp_sorted(
            h_actual, planned_df['Height'].values,
            planned_df['Load'].values)
        for key, _, params, _, _ in heatmap_overlays:
            try:
                pred = predict_at(params, h_actual, l_at_actual)
                rmse_dict[key] = _rmse(pred, emg_actual)
            except Exception as e:
                print(f'  {key} 预测失败: {e}')

    return rmse_dict


def _format_rmse(rmse_dict, key):
    """为 legend 生成 'RMSE=0.045' 样的后缀字符串。"""
    if key not in rmse_dict:
        return ''
    rmse, n = rmse_dict[key]
    if not np.isfinite(rmse):
        return ''
    return f' [RMSE={rmse:.4f}, n={n}]'


# =========================================================
#  绘图
# =========================================================

def plot_one_vload_entry(label, vload_result, planned_df, heatmap_overlays,
                         target_muscle, target_activation,
                         movement_types=None):
    """
    对单组变负载实验绘图。

    1×2 子图：
      - 左：Activation vs Height。三种预测（expected / rbf / monotonic）会在
        legend 中同时展示 RMSE。
      - 右：Planned Load vs Height（规划负载曲线参考）。

    Returns
    -------
    (fig, rmse_dict)
    """
    cutted = filter_movement_types(
        vload_result.get('cutted_data'), movement_types)
    emg_col = f'emg_{target_muscle}'

    # 先计算 RMSE，后面拼接到 legend 里
    rmse_dict = compute_rmse_at_actual_points(
        cutted, planned_df, heatmap_overlays, target_muscle)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    mt_label = ('+'.join(movement_types) if movement_types else 'all')
    fig.suptitle(
        f'{label}  (target muscle: {target_muscle}, '
        f'goal: {target_activation}, movement: {mt_label})',
        fontsize=12, fontweight='bold')

    # -------- 左：Activation vs Height --------
    ax = axes[0]

    # 实测 EMG（按高度散点）
    if (cutted is not None
            and emg_col in cutted.columns
            and 'pos_l' in cutted.columns):
        ax.scatter(cutted['pos_l'], cutted[emg_col], s=8, alpha=0.35,
                   color='gray', label='Actual EMG')

    # 规划激活曲线
    if (planned_df is not None
            and 'Height' in planned_df.columns
            and 'Activation' in planned_df.columns):
        h = planned_df['Height'].values
        a = planned_df['Activation'].values
        order = np.argsort(h)
        ax.plot(h[order], a[order], color='C0', linewidth=2,
                label=f'Expected (vload_file)' + _format_rmse(
                    rmse_dict, 'expected'))

    # 目标激活水平线
    if target_activation is not None:
        ax.axhline(target_activation, color='C1', linestyle=':',
                   linewidth=1.5, label=f'Goal = {target_activation}')

    # 可选：heatmap 预测（在每个规划点 (Height, Load) 上预测画曲线）
    if (heatmap_overlays
            and planned_df is not None
            and 'Height' in planned_df.columns
            and 'Load' in planned_df.columns):
        h = planned_df['Height'].values
        l = planned_df['Load'].values
        order = np.argsort(h)
        for key, ov_label, ov_params, ov_color, ov_ls in heatmap_overlays:
            try:
                pred = predict_at(ov_params, h, l)
                ax.plot(h[order], pred[order], color=ov_color, linewidth=2,
                        linestyle=ov_ls,
                        label=ov_label + _format_rmse(rmse_dict, key))
            except Exception as e:
                print(f'  {ov_label} 画图预测失败: {e}')

    ax.set_xlabel('Height (m)')
    ax.set_ylabel(f'{target_muscle} activation')
    ax.set_title('Activation vs Height')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # -------- 右：Planned Load vs Height --------
    ax = axes[1]
    if (planned_df is not None
            and 'Height' in planned_df.columns
            and 'Load' in planned_df.columns):
        h = planned_df['Height'].values
        l = planned_df['Load'].values
        order = np.argsort(h)
        ax.plot(h[order], l[order], color='C0', linewidth=2,
                label='Planned load')

    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Load (kg)')
    ax.set_title('Planned Load vs Height')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, rmse_dict


# =========================================================
#  主函数
# =========================================================

def _print_rmse_summary(rmse_summary):
    """打印所有实验、所有预测方式的 RMSE 总表。"""
    if not rmse_summary:
        return
    keys = ['expected', 'rbf', 'monotonic']
    headers = {'expected': 'Expected',
               'rbf': 'Heatmap (RBF)',
               'monotonic': 'Heatmap (monotonic)'}
    col_w = max(20, max(len(label) for label in rmse_summary) + 2)
    print('\n========== RMSE summary (actual EMG vs predictions) ==========')
    head = 'Label'.ljust(col_w) + ''.join(
        headers[k].rjust(22) for k in keys) + '   n'
    print(head)
    print('-' * len(head))
    for label, d in rmse_summary.items():
        row = label.ljust(col_w)
        n_print = ''
        for k in keys:
            if k in d and np.isfinite(d[k][0]):
                row += f'{d[k][0]:>22.4f}'
                n_print = d[k][1]
            else:
                row += '                     -'
        row += f'   {n_print}'
        print(row)
    print('=' * len(head))


def main():
    subject = Subject('../config/20250409_squat_NCMP001.json')
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # 加载变负载实测数据（aligned + cutted）
    vload_results = pipeline.run_vload()
    if not vload_results:
        print('未加载到变负载实际数据，终止。')
        return

    rmse_summary = {}

    # 遍历配置中的每组变负载实验
    for label, params in subject.vload_data.items():
        if label not in vload_results:
            print(f'\n>>> 跳过 {label}：未成功处理')
            continue

        vload_file = params.get('vload_file')
        target_muscle = params.get('target_muscle')
        target_activation = params.get('target_activation')

        print(f'\n>>> 对比 {label} '
              f'(muscle={target_muscle}, goal={target_activation})')

        planned_df = load_planned_vload(subject, vload_file)
        if planned_df is None:
            print('  跳过：未找到 vload_file 规划数据')
            continue

        heatmap_overlays = load_heatmap_params_by_mode(
            subject, target_muscle, HEATMAP_MODE)

        _, rmse_dict = plot_one_vload_entry(
            label=label,
            vload_result=vload_results[label],
            planned_df=planned_df,
            heatmap_overlays=heatmap_overlays,
            target_muscle=target_muscle,
            target_activation=target_activation,
            movement_types=MOVEMENT_TYPES,
        )

        # 控制台详细输出
        for k in ('expected', 'rbf', 'monotonic'):
            if k in rmse_dict:
                rmse, n = rmse_dict[k]
                if np.isfinite(rmse):
                    print(f'  RMSE [{k:>9}] = {rmse:.4f}  (n={n})')
        rmse_summary[label] = rmse_dict

    _print_rmse_summary(rmse_summary)

    plt.show()


if __name__ == '__main__':
    main()