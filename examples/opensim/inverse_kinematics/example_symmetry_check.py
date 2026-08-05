"""
example_symmetry_check.py

校验【外力信息】与【关节信息】是否相互吻合。

为什么需要它：ID 力矩是「运动学 + 外力」两路信息合成的结果。
example_validate_mot.py 已经把运动学那一路判定为可信，但外力那一路
至今没有独立校验。标定漂移、左右接反、单侧通道失效都不会报错，
只会静默地把 ID 力矩算错。

【重要：两种力不是一回事，不能混用】
  force_l / force_r  = 机器人（杠）两侧致动器的力。杠是刚体，两侧分担
                       几乎恒等于 50%，与【哪条腿承重多】无关。
                       它的总和 ≈ 配重 × g，不包含体重。
  grf_l / grf_r      = 鞋垫地面反力。它才是 ID 贴在 calcn_l / calcn_r 上
                       的外力，总和 ≈ (体重 + 配重) × g，分担才反映腿的负荷。
首版本误用 force_l / force_r 当鞋垫力，导致 S1 截距算出体重 ≈ 0 kg，
S3 拿一个恒为 50% 的量去卡 ID 力矩分担。现已改为优先用 grf_l / grf_r，
只有在鞋垫列缺失时才退回机器人力，并相应降级判据。

核心思路：同一个物理事实【哪一侧承担更多】有三路相互独立的测量：
  (a) 鞋垫 GRF 分担    grf_l / grf_r
  (b) ID 关节力矩分担   knee/hip/ankle 的 _l / _r moment
  (c) 运动学不对称      .mot 里左右关节角的峰值差
三者的逻辑关系并不对等，必须分开对待：
  (a) -> (b) 是【硬约束】，但仅当 (a) 真的是鞋垫 GRF 时才成立。
            ID 的外力就是 grf 贴到两只脚上的，所以两者必须指向同一侧。
            方向相反 = 接线错（grf_l/grf_r 接反或贴错 body），必须报错。
            注意分担差值【不】是硬约束：力矩还取决于力臂，而力臂由 COP
            与关节中心位置决定，两侧未必相同。
  (a) -> (c) 是【软关联】。受力多的一侧不一定蹲得更深，所以只提示，
            不当错误。只有当两者都强烈且方向相反时才值得怀疑。

检查项：
  [S1] 外力总量标定：总力对配重回归，斜率应等于 g = 9.81 N/kg。
       斜率不需要知道体重（体重只进截距）。截距的期望值取决于数据源：
         鞋垫 GRF -> 截距/g ≈ 体重；
         机器人力   -> 截距 ≈ 0（本来就不含体重）。
  [S2] 左右力分担：share_r = mean(R) / (mean(L) + mean(R))。
  [S3] 力分担 vs ID 力矩分担：只把【方向相反】判为错误；
       差值大只作为 COP 可疑的线索输出，不计 FAIL。
  [S4] 力分担 vs 运动学不对称（软关联，仅提示）。
  [S5] 单侧通道失效：某侧恒为常数 / 几乎为零 / 出现负值。
       鞋垫 GRF 不可能为负；机器人力在轻载下因钢缆松弛可以为负，
       因此负值判据会根据数据源自动切换严格/宽松。
  [S6] 【每侧增量增益】用实测杆力而非名义配重做闭合检验。
       物理上 GRF_total = 体重 + 杆力，所以把 GRF 对杆力回归：
         总斜率应 = 1.0（每多压 1 N 到杠上，地面就多受 1 N）
         单侧斜率应 ≈ 0.5（新增载荷应该两腿平分）
         截距/g 应 = 体重（且与 S1 的估计一致）
       这比 S1 严格：S1 用名义配重，假定杆力就等于配重×g；
       S6 用实测杆力，把杆侧的误差完全排除在外。
       单侧斜率偏离 0.5 -> 新增载荷没有两腿平分，要么是受试者真的
       向一侧转移，要么是那侧鞋垫在高压下欠读。两者由 S7 区分。
  [S7] 【饱和 vs 策略】区分器。把所有试次的逐帧数据汇到一起，
       看左侧占比随【瞬时总力】如何变化。
         若各试次塌缩在同一条曲线上（组间偏移小）-> 占比只由力的大小
           决定，与哪个试次无关，这是传感器压缩非线性的指纹；
         若每个试次各自一条线（组间偏移大）-> 是姿势/策略差异。
       仅输出证据，不判 FAIL——最终要靠鞋垫逐点数据看有无单元格顶值。

运行时机：example_validate_mot.py 通过之后、相信 ID 结果之前。
"""
import os
import json

import numpy as np
import pandas as pd

from digitaltwin.utils.logger import beauty_print
from digitaltwin.osim.mot_pipeline import get_mot_files
from digitaltwin.analysis.result_analysis import (
    read_opensim_table,
    get_load_keys,
    load_or_create_cutted_pipeline_results,
    get_segment_from_results,
    get_inverse_dynamics_path,
    find_id_moment_column,
    interpolate_column_to_segment,
)
from digitaltwin.config_manager import filter_load_keys, get_load_mode
from digitaltwin.visualization.symmetry_plot import plot_symmetry_figures


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../../config/20260513_squat_FTS09_xsens.json'

LOAD_KEYS = None

# 按负载模式筛选，而不是硬写组名。
# 本脚本的 S1 / S6 都要把外力对【配重】回归，只有定负载（isotonic）
# 组才有明确的配重；等速/等长组的“等效负载”是从受力反推的，
# 拿去做同一个回归等于用结果验证结果，没有意义。
# 之前写的 EXCLUDE_LOAD_KEYS=['0.15','0.3'] 在组名改成 IK-0.15/IK-0.3 之后
# 已经静默失效（不报错，但一个都没排除掉），所以改用模式筛选，
# 以后新增等长/等速组不用再改这里。
# 现在三种模式全部参与。但必须分清哪些检查吃得下全模式：
#   S2/S3/S4/S5 与四张图 —— 全模式。它们比的是“同一时刻左右两侧”，
#       不需要知道标称负载是多少。
#   S1/S6/S7    —— 只能用定负载组。它们把外力对【配重】回归，
#       而等速/等长组的“等效负载”本身就是从受力反推出来的，
#       拿去做同一个回归等于用结果验证结果。
# 所以不是“把筛选去掉”，而是把筛选从【取数】移到【具体检查】。
LOAD_MODES_FILTER = None
CALIBRATION_MODES = ('isotonic',)
EXCLUDE_LOAD_KEYS = []

# 同时用上升与下降。只用 upward 会系统性高估总力（加速度向上），
# 两个阶段合起来惯性项在一个完整循环内大致抵消，均值才能与
# 【体重 + 配重】直接比较。
# 等长组的段标的是 movement_type='isometric'，不在这两类里。
# 不需要在这里把 'isometric' 加进去：get_segment_from_results 已改为
# 当请求类型一个都没命中、而该组只有等长段时自动回退到等长段
# 并打印警告。把 'isometric' 硬加在这里反而会让定负载组也去混入
# 可能存在的静止段，把均值拉低。
MOVEMENT_TYPES = ('upward', 'downward')

G = 9.81

# [S1] 回归斜率允许偏离 g 的相对量
SLOPE_TOL = 0.15
# [S1] 截距推算出的体重合理区间（kg）
BODY_MASS_RANGE = (40.0, 150.0)

# [S2] 左右分担偏离 50% 多少算“明显偏侧”
SHARE_WARN = 0.10
# [S3] 力分担与力矩分担的允许差
SHARE_CONSISTENCY_TOL = 0.12
# [S4] 关节角峰值差超过多少度算“明显不对称”
PEAK_DIFF_WARN = 5.0

# 外力数据源优先级：(左列, 右列, 标记)
# 鞋垫 GRF 才是 ID 实际用的外力；机器人力只是降级退路。
FORCE_SOURCES = (
    ('grf_l', 'grf_r', 'insole'),
    ('force_l', 'force_r', 'robot'),
)
# [S1] 机器人力模式下，截距应接近 0（N）
ROBOT_INTERCEPT_TOL = 100.0
# 切片缓存名：必须与不含鞋垫的缓存分开，否则会读到没有 grf 列的旧缓存
CACHE_NAME = 'cutted_data_insole.csv'

# [S6] GRF 总力对实测杆力的回归斜率，理论值 1.0
TOTAL_GAIN_IDEAL = 1.0
TOTAL_GAIN_TOL = 0.08
# [S6] 单侧增量斜率，理论值 0.5
SIDE_GAIN_IDEAL = 0.5
SIDE_GAIN_TOL = 0.08
# [S6] 两条路径（S1 名义配重 / S6 实测杆力）推算体重的允许差（kg）
BODY_MASS_AGREE_TOL = 5.0
# [S7] 饱和判别：按瞬时总力分箱的箱数
SATURATION_BINS = 8

# [S3] 用哪些关节的 ID 力矩算左右分担
MOMENT_BASES = ('knee_angle', 'hip_flexion', 'ankle_angle')
# [S4] 用哪些关节角看运动学不对称
ANGLE_BASES = ('knee_angle', 'hip_flexion', 'ankle_angle')

# 五张对称性图（绘图实现在 digitaltwin/visualization/symmetry_plot.py）
# 第五张是趋势图：SI 随【合力】与【杆高】的变化。横轴用合力而不是标称
# 配重，等长/等速组的 load_kg 是 nan，按 nan 做横轴会把它们静默丢掉。
PLOT_FIGURES = True
SAVE_FIGURES = True
# 蝴形图需要逐 cycle 的左右关节角曲线，重采样到这么多个点
CYCLE_GRID_POINTS = 101


def get_base_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


def pick_force_source(seg):
    """选定外力列。优先鞋垫 GRF，否则退回机器人力。"""
    for col_l, col_r, tag in FORCE_SOURCES:
        if col_l in seg.columns and col_r in seg.columns:
            return col_l, col_r, tag
    return None, None, None


def _load_sort_key(item):
    '''排序键：定负载按数值升序在前，等长/等速按名字排在后。

    原来写的是 key=lambda kv: kv[1]['load_value']。引入等长/等速组后，
    load_value 是 nan，而 nan 参与比较的结果是未定义的，
    表现为表格行序每次不同、跟图里的颜色顺序对不上。
    '''
    key, rec = item
    v = rec.get('load_value', np.nan)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = np.nan
    if not np.isfinite(v):
        return (1, 0.0, str(key))
    return (0, v, str(key))


def _canon_load_key(value):
    try:
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f'{f:g}'
    except Exception:
        return str(value)


class Verdicts:
    """汇总 PASS/FAIL。与 example_validate_mot.py 保持同一种输出风格。"""

    def __init__(self):
        self.items = []

    def add(self, check, load_key, ok, detail=''):
        self.items.append((check, str(load_key), bool(ok), detail))

    def report(self):
        print('\n' + '=' * 80)
        print('[汇总] 鞋垫 ↔ 关节 一致性判定')
        print('=' * 80)
        fails = [i for i in self.items if not i[2]]
        if not fails:
            print('[PASS] 全部检查通过；外力与运动学互相吻合。')
            return True
        lines = [f'{len(fails)} 项校验未通过：']
        for check, load_key, _, detail in fails:
            lines.append(f'  - {check:<24} load={load_key:<6} {detail}')
        lines.append('鞋垫信息与关节信息不吻合，在修好之前 ID 力矩的绝对值不可用。')
        beauty_print('\n'.join(lines), type="warning")
        return False


# ============================================================
#  数据收集
# ============================================================

def collect_side_data(config, base_dir, pipeline_results, load_keys):
    """每个负载汇总三路信息。

    Returns
    -------
    dict {load_key: {...}}
    """
    mot_by_key = {_canon_load_key(k): v
                  for k, v in get_mot_files(config, base_dir).items()}
    out = {}

    for load_key in load_keys:
        seg = get_segment_from_results(pipeline_results, load_key,
                                       movement_types=MOVEMENT_TYPES)
        if seg is None or len(seg) == 0:
            print(f'[MISS] load={load_key}: 无切片数据，跳过')
            continue
        col_l, col_r, tag = pick_force_source(seg)
        if col_l is None:
            print(f'[MISS] load={load_key}: 切片数据既无 grf_l/grf_r '
                  f'也无 force_l/force_r，跳过')
            continue

        rec = {'segment': seg, 'source': tag,
               'source_cols': (col_l, col_r),
               'mode': get_load_mode(config, load_key, warn=False)}

        fl = seg[col_l].values.astype(float)
        fr = seg[col_r].values.astype(float)
        rec['force_l'] = fl
        rec['force_r'] = fr
        rec['force_total_mean'] = float(np.nanmean(fl + fr))

        # 实测杆力（机器人致动器）——S6 用它做闭合检验。
        # 即使当前外力源是鞋垫，这两列也仍然存在且含义不同。
        if 'force_l' in seg.columns and 'force_r' in seg.columns:
            rec['bar_total_mean'] = float(np.nanmean(
                seg['force_l'].values.astype(float)
                + seg['force_r'].values.astype(float)))
        else:
            rec['bar_total_mean'] = np.nan

        # 配重：优先用切片里的 load_value，否则用 load_key 本身
        if 'load_value' in seg.columns:
            rec['load_value'] = float(np.nanmedian(
                seg['load_value'].values.astype(float)))
        else:
            try:
                rec['load_value'] = float(load_key)
            except Exception:
                rec['load_value'] = np.nan

        # ID 左右力矩
        id_df = read_opensim_table(
            get_inverse_dynamics_path(config, base_dir, load_key))
        rec['moments'] = {}
        if id_df is not None and 'time' in id_df.columns:
            for base in MOMENT_BASES:
                pair = {}
                for side in ('l', 'r'):
                    col = find_id_moment_column(id_df, f'{base}_{side}')
                    if col is None:
                        continue
                    v = interpolate_column_to_segment(id_df, seg, col)
                    if v is None:
                        continue
                    pair[side] = float(np.nanmean(np.abs(v)))
                if len(pair) == 2:
                    rec['moments'][base] = pair

        # 运动学左右峰值
        rec['angles'] = {}
        mot_path = mot_by_key.get(_canon_load_key(load_key))
        if mot_path:
            mot_df = read_opensim_table(mot_path)
            if mot_df is not None and 'time' in mot_df.columns:
                t = mot_df['time'].values.astype(float)
                t0 = float(seg['time'].min())
                t1 = float(seg['time'].max())
                w = mot_df[(t >= t0) & (t <= t1)]
                for base in ANGLE_BASES:
                    cl, cr = f'{base}_l', f'{base}_r'
                    if cl in w.columns and cr in w.columns and len(w) > 20:
                        rec['angles'][base] = {
                            'l': float(np.nanmax(np.abs(
                                w[cl].values.astype(float)))),
                            'r': float(np.nanmax(np.abs(
                                w[cr].values.astype(float)))),
                        }

                # 蝴形图需要的逐 cycle 曲线：把每一段在时间上归一化到
                # 0-100%，再把 mot 里的左右关节角插值上去。
                # 必须按段归一化而不是直接拼时间轴：各次深蹲时长不同，
                # 不归一化就会把峰值错开平均掉，把真实差异抹成噪声。
                rec['angle_curves'] = {}
                grid = np.linspace(0.0, 100.0, CYCLE_GRID_POINTS)
                mot_t = mot_df['time'].values.astype(float)
                if 'segment_id' in seg.columns:
                    seg_ids = list(seg['segment_id'].unique())
                else:
                    seg_ids = [None]

                for base in ANGLE_BASES:
                    cl, cr = f'{base}_l', f'{base}_r'
                    if cl not in mot_df.columns or cr not in mot_df.columns:
                        continue
                    vl = mot_df[cl].values.astype(float)
                    vr = mot_df[cr].values.astype(float)
                    curves_l, curves_r = [], []
                    for sid in seg_ids:
                        sub = seg if sid is None else seg[seg['segment_id'] == sid]
                        if len(sub) < 10:
                            continue
                        st0 = float(sub['time'].min())
                        st1 = float(sub['time'].max())
                        if not np.isfinite(st0) or not np.isfinite(st1) \
                                or st1 <= st0:
                            continue
                        tt = st0 + (st1 - st0) * grid / 100.0
                        curves_l.append(np.interp(tt, mot_t, vl))
                        curves_r.append(np.interp(tt, mot_t, vr))
                    if curves_l and curves_r:
                        rec['angle_curves'][base] = {
                            'grid': grid,
                            'l': np.vstack(curves_l),
                            'r': np.vstack(curves_r),
                        }

        out[_canon_load_key(load_key)] = rec

    return out


# ============================================================
#  [S1] 鞋垫总量标定
# ============================================================

def source_of(data):
    """返回本次实际用的外力数据源标记。"""
    tags = {r.get('source') for r in data.values()}
    return tags.pop() if len(tags) == 1 else 'mixed'


def check_force_calibration(data, verdicts):
    src = source_of(data)
    label = '鞋垫 GRF' if src == 'insole' else '机器人力'
    print('\n' + '=' * 80)
    print(f'[S1] 外力总量标定：{label} 总力 vs 配重（斜率应 = g = 9.81 N/kg）')
    print('=' * 80)
    cols = {r.get('source_cols') for r in data.values()}
    print(f'  数据源: {src}  列: {sorted(cols)[0] if len(cols) == 1 else cols}')
    print(f'{"load(kg)":>10}{"总力均值(N)":>16}{"拟合值(N)":>14}{"残差(N)":>12}'
          f'{"残差占比":>12}')

    pts = [(r['load_value'], r['force_total_mean']) for r in data.values()
           if np.isfinite(r['load_value']) and np.isfinite(r['force_total_mean'])]
    if len(pts) < 3:
        print('  有效负载不足 3 个，无法回归，跳过。')
        return

    loads = np.array([p[0] for p in pts], dtype=float)
    totals = np.array([p[1] for p in pts], dtype=float)
    slope, intercept = np.polyfit(loads, totals, 1)

    for load_key, r in sorted(data.items(), key=_load_sort_key):
        fit = slope * r['load_value'] + intercept
        resid = r['force_total_mean'] - fit
        print(f'{r["load_value"]:>10.1f}{r["force_total_mean"]:>16.1f}'
              f'{fit:>14.1f}{resid:>12.1f}'
              f'{resid / max(abs(fit), 1e-9):>12.1%}')

    dev = abs(slope - G) / G
    print(f'\n  拟合斜率 = {slope:.3f} N/kg（理论 {G:.2f}，偏离 {dev:.1%}）')
    print(f'  拟合截距 = {intercept:.1f} N')

    ok_slope = dev <= SLOPE_TOL
    verdicts.add('S1 外力增益', 'all', ok_slope,
                 f'总力对配重的斜率 {slope:.2f} N/kg 偏离 g 达 {dev:.0%}；'
                 f'斜率偏低意味着重负时外力被系统性低估，'
                 f'ID 力矩会随负载“涨不动”')

    if src == 'insole':
        body_mass = intercept / G
        print(f'  截距/g -> 推算体重 ≈ {body_mass:.1f} kg')
        ok_mass = BODY_MASS_RANGE[0] <= body_mass <= BODY_MASS_RANGE[1]
        verdicts.add('S1 截距合理', 'all', ok_mass,
                     f'鞋垫 GRF 截距推算体重 {body_mass:.1f} kg 不在 '
                     f'{BODY_MASS_RANGE[0]:.0f}-{BODY_MASS_RANGE[1]:.0f} kg 内；'
                     f'GRF 必须包含体重，偏离说明鞋垫被去基线或标定有误')
    else:
        print(f'  机器人力模式：截距应接近 0（杆力不包含体重）')
        ok_zero = abs(intercept) <= ROBOT_INTERCEPT_TOL
        verdicts.add('S1 截距合理', 'all', ok_zero,
                     f'机器人力截距 {intercept:.1f} N 偏离 0 超过 '
                     f'{ROBOT_INTERCEPT_TOL:.0f} N，提示零点漂移')

    print('\n判读: 斜率不需要知道体重——体重只影响截距。')
    print('      斜率显著小于 9.81 -> 传感器在大载荷下欠线性或增益偏小。')
    print('      鞋垫 GRF 的截距必须≈体重；若≈ 0，说明拿到的根本不是 GRF，')
    print('      而是只含配重的杆力，此时 S3 的分担对比没有意义。')


# ============================================================
#  [S2][S3][S4] 分担一致性
# ============================================================

def _share(l, r):
    tot = l + r
    if not np.isfinite(tot) or abs(tot) < 1e-9:
        return np.nan
    return r / tot


def check_share_consistency(data, verdicts):
    src = source_of(data)
    label = '鞋垫 GRF' if src == 'insole' else '机器人力(仅参考)'
    print('\n' + '=' * 80)
    print(f'[S2][S3] 左右分担：{label} vs ID 力矩（均为右侧占比）')
    print('=' * 80)
    if src != 'insole':
        beauty_print(
            '当前用的是机器人力 force_l/force_r，不是鞋垫 GRF。\n'
            '杠是刚体，两侧致动器力几乎恒为 50:50，与哪条腿承重无关，\n'
            '因此本表【不】能用来判定左右接反。请先让切片数据包含 grf_l/grf_r\n'
            '（load_or_create_cutted_pipeline_results(include_insole=True)）再重跑。',
            type="warning")
    print(f'{"load":<8}{"力分担R":>10}' +
          ''.join(f'{b.replace("_angle", "").replace("_flexion", ""):>12}'
                  for b in MOMENT_BASES) + '   判定')

    for load_key, r in sorted(data.items(), key=_load_sort_key):
        f_share = _share(float(np.nanmean(r['force_l'])),
                         float(np.nanmean(r['force_r'])))
        row = f'{load_key:<8}{f_share:>10.1%}'

        notes = []
        for base in MOMENT_BASES:
            pair = r['moments'].get(base)
            if not pair:
                row += f'{"N/A":>12}'
                continue
            m_share = _share(pair['l'], pair['r'])
            row += f'{m_share:>12.1%}'

            if not np.isfinite(f_share) or not np.isfinite(m_share):
                continue
            gap = abs(m_share - f_share)
            # 方向相反（一个 > 50%、另一个 < 50%）且两边都不是微小偏离，
            # 是左右接反的典型指纹。
            flipped = ((f_share - 0.5) * (m_share - 0.5) < 0
                       and abs(f_share - 0.5) > 0.03
                       and abs(m_share - 0.5) > 0.03)
            # 只有鞋垫 GRF 才能当硬约束；机器人力模式下只打印不判错。
            if flipped:
                notes.append(f'{base}:方向相反')
                if src == 'insole':
                    verdicts.add('S3 力/力矩同侧', load_key, False,
                                 f'{base} GRF分担R={f_share:.0%} 但力矩分担R={m_share:.0%}，'
                                 f'两者指向相反 -> 怀疑 grf_l/grf_r 接反或贴错 body')
            elif gap > SHARE_CONSISTENCY_TOL:
                notes.append(f'{base}:差{gap:.0%}')

        print(row + '   ' + (', '.join(notes) if notes else 'OK'))

    print('\n判读: 只有【方向相反】才是错误，而且只在数据源是鞋垫 GRF 时成立。')
    print('      分担差值大不算错：力矩 = 力 × 力臂，而力臂由 COP 与关节中心')
    print('      决定，两侧本来就可以不同。但若差值随负载单调变大，就是')
    print('      COP（目前是常数）跟不上真实压心前移的典型信号。')


def check_kinematic_side(data, verdicts):
    print('\n' + '=' * 80)
    print('[S4] 力分担 vs 运动学不对称（软关联，仅提示）')
    print('=' * 80)
    print(f'{"load":<8}{"力分担R":>10}{"joint":>14}{"peak_l":>10}{"peak_r":>10}'
          f'{"峰值差":>10}   提示')

    hints = []
    for load_key, r in sorted(data.items(), key=_load_sort_key):
        f_share = _share(float(np.nanmean(r['force_l'])),
                         float(np.nanmean(r['force_r'])))
        for base, pair in r['angles'].items():
            diff = pair['r'] - pair['l']
            note = ''
            if abs(diff) > PEAK_DIFF_WARN and abs(f_share - 0.5) > SHARE_WARN:
                same_side = (diff > 0) == (f_share > 0.5)
                note = '同侧' if same_side else '异侧（值得看）'
                if not same_side:
                    hints.append(
                        f'load={load_key} {base}: 右侧承力 {f_share:.0%} '
                        f'但右侧关节角反而小 {abs(diff):.1f}°')
            print(f'{load_key:<8}{f_share:>10.1%}{base:>14}'
                  f'{pair["l"]:>10.1f}{pair["r"]:>10.1f}{diff:>+10.1f}   {note}')

    if hints:
        beauty_print('\n'.join(
            ['力分担与运动学不对称方向不一致（不一定是错，但值得核实）：']
            + [f'  - {h}' for h in hints]
            + ['可能原因：受试者用单侧代偿、鞋垫左右文件写反、'
               '或 Xsens 某侧传感器佩戴偏移。']),
            type="warning")
    else:
        print('\n  未发现力与运动学明显矛盾的情况。')

    print('\n判读: 这一项【不】计入 FAIL。承力多的一侧未必蹲得更深，')
    print('      反之亦然；只有两者都很强烈且方向相反时才需要排查。')


# ============================================================
#  [S5] 单侧通道失效
# ============================================================

def check_channel_health(data, verdicts):
    src = source_of(data)
    print('\n' + '=' * 80)
    print(f'[S5] 外力通道健康度（数据源: {src}）')
    print('=' * 80)
    print(f'{"load":<8}{"side":>6}{"mean(N)":>12}{"std(N)":>12}'
          f'{"min(N)":>12}{"负值帧比":>12}   判定')

    for load_key, r in sorted(data.items(), key=_load_sort_key):
        for side in ('l', 'r'):
            v = r[f'force_{side}']
            v = v[np.isfinite(v)]
            if v.size < 20:
                continue
            mean, std = float(np.mean(v)), float(np.std(v))
            vmin = float(np.min(v))
            neg_frac = float(np.mean(v < 0))

            problems = []
            if std < 1e-6:
                problems.append('恒定')
            if abs(mean) < 1.0:
                problems.append('几乎为零')
            # 鞋垫 GRF 不可能为负；机器人杆力在轻载下因钢缆松弛/振荡
            # 出现短暂负值是正常的，不应判错。
            if neg_frac > 0.01:
                if src == 'insole':
                    problems.append(f'负值{neg_frac:.0%}')
                else:
                    print(f'{"":<8}{"":>6}  （机器人力负值 {neg_frac:.0%}，'
                          f'轻载下钢缆松弛属正常，不判错）')

            print(f'{load_key:<8}{side:>6}{mean:>12.1f}{std:>12.1f}'
                  f'{vmin:>12.1f}{neg_frac:>12.1%}   '
                  f'{(", ".join(problems) if problems else "OK")}')
            verdicts.add('S5 通道健康', load_key, not problems,
                         f'{r["source_cols"][0 if side == "l" else 1]}: '
                         + ', '.join(problems))

    print('\n判读: 地面反力不可能为负。出现负值 -> 零点漂移或去基线错误；')
    print('      恒定 -> 该侧通道失效或数据被插值填充。')


# ============================================================
#  [S6] 每侧增量增益（对实测杆力回归）
# ============================================================

def _linfit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x[m], y[m], 1)
    return float(slope), float(intercept)


def check_side_gain(data, verdicts):
    """GRF_total = 体重 + 杆力。把 GRF 对【实测杆力】回归，
    总斜率应 = 1.0，单侧斜率应 ≈ 0.5。

    这比 S1（对名义配重回归）严格：它不依赖“杆力 = 配重×g”这个假设，
    因此能把杆侧误差与鞋垫侧误差分开。
    """
    src = source_of(data)
    if src != 'insole':
        print('\n[S6] 跳过：当前外力源不是鞋垫 GRF，无法与杆力做闭合检验。')
        return

    print('\n' + '=' * 80)
    print('[S6] 每侧增量增益：GRF vs 实测杆力（总斜率应 1.0，单侧应 0.5）')
    print('=' * 80)

    rows = sorted(data.values(), key=lambda r: r['load_value'])
    bar = np.array([r.get('bar_total_mean', np.nan) for r in rows], dtype=float)
    if not np.isfinite(bar).sum() >= 3:
        print('  缺少实测杆力 force_l/force_r，跳过。')
        return

    gl = np.array([float(np.nanmean(r['force_l'])) for r in rows])
    gr = np.array([float(np.nanmean(r['force_r'])) for r in rows])
    gt = gl + gr

    print(f'{"load(kg)":>10}{"杆力(N)":>12}{"GRF_L(N)":>12}{"GRF_R(N)":>12}'
          f'{"GRF总(N)":>12}{"体重+杆力":>14}{"偏差":>10}')

    s_tot, i_tot = _linfit(bar, gt)
    body_n = i_tot
    for r, b, l, rr, t in zip(rows, bar, gl, gr, gt):
        exp = body_n + b
        print(f'{r["load_value"]:>10.1f}{b:>12.1f}{l:>12.1f}{rr:>12.1f}'
              f'{t:>12.1f}{exp:>14.1f}{(t - exp) / max(exp, 1e-9):>10.1%}')

    s_l, i_l = _linfit(bar, gl)
    s_r, i_r = _linfit(bar, gr)
    body_mass = i_tot / G

    print(f'\n  总斜率   = {s_tot:.4f}（理论 {TOTAL_GAIN_IDEAL:.2f}）'
          f'   截距 = {i_tot:.1f} N -> 体重 ≈ {body_mass:.1f} kg')
    print(f'  左侧斜率 = {s_l:.4f}（理论 {SIDE_GAIN_IDEAL:.2f}）'
          f'   静态截距 = {i_l:.1f} N')
    print(f'  右侧斜率 = {s_r:.4f}（理论 {SIDE_GAIN_IDEAL:.2f}）'
          f'   静态截距 = {i_r:.1f} N')
    if np.isfinite(i_l + i_r) and abs(i_l + i_r) > 1e-6:
        print(f'  空载站立时左侧占比 = {i_l / (i_l + i_r):.1%}，'
              f'而新增载荷的左侧占比 = {s_l / max(s_l + s_r, 1e-9):.1%}')

    ok_tot = np.isfinite(s_tot) and abs(s_tot - TOTAL_GAIN_IDEAL) <= TOTAL_GAIN_TOL
    verdicts.add('S6 总量闭合', 'all', ok_tot,
                 f'GRF 总力对杆力的斜率 {s_tot:.3f}，偏离 1.0 超过 '
                 f'{TOTAL_GAIN_TOL:.0%}；每向杠上施 1 N，鞋垫只多读到 '
                 f'{s_tot:.2f} N -> 重负时外力被系统性低估')

    for side, s in (('L', s_l), ('R', s_r)):
        ok = np.isfinite(s) and abs(s - SIDE_GAIN_IDEAL) <= SIDE_GAIN_TOL
        verdicts.add(f'S6 单侧增益{side}', 'all', ok,
                     f'{side} 侧增量斜率 {s:.3f}，偏离 0.5 超过 '
                     f'{SIDE_GAIN_TOL:.2f}；新增载荷没有两腿平分，'
                     f'该侧外力比实际偏{"小" if s < SIDE_GAIN_IDEAL else "大"}')

    print('\n判读: 总斜率 < 1.0 -> 鞋垫整体欠读，ID 外力偏小。')
    print('      单侧斜率偏离 0.5 -> 新增载荷没有两腿平分。这会直接把')
    print('      不对称写进 ID 外力，而膝力矩是小残差，会被放大好几倍。')
    print('      注意：静态截距与增量斜率是两回事。单纯的增益错误会同时')
    print('      缩小截距和斜率；若截距正常而斜率偏低，则是压缩型非线性')
    print('      （高压下饱和）或真实的重心转移——由 S7 区分。')


# ============================================================
#  [S7] 饱和 vs 策略 区分器
# ============================================================

def check_saturation(data, verdicts):
    """判断左右分担随载荷变化，到底是传感器压缩还是受试者策略。

    关键区别：饱和是【瞬时力】的函数，与是哪个试次无关；
    策略是【试次】的属性。所以把所有试次的逐帧点汇到一起：
      — 各试次塌缩到同一条 share(F) 曲线上  -> 传感器特性
      — 各试次各自一条、组间偏移大        -> 姿势/策略
    本项不判 FAIL，只输出证据。
    """
    src = source_of(data)
    if src != 'insole':
        return

    print('\n' + '=' * 80)
    print('[S7] 左侧占比 vs 瞬时总力：传感器压缩 还是 重心转移？')
    print('=' * 80)

    pooled_f, pooled_s = [], []
    per_trial = []
    for load_key, r in sorted(data.items(), key=_load_sort_key):
        l = np.asarray(r['force_l'], dtype=float)
        rr = np.asarray(r['force_r'], dtype=float)
        tot = l + rr
        m = np.isfinite(tot) & (tot > 1.0) & np.isfinite(l)
        if int(m.sum()) < 20:
            continue
        f = tot[m]
        s = l[m] / f
        pooled_f.append(f)
        pooled_s.append(s)
        slope, _ = _linfit(f, s)
        per_trial.append((load_key, float(np.mean(f)), float(np.mean(s)),
                          slope * 1000.0))

    if len(per_trial) < 3:
        print('  有效试次不足，跳过。')
        return

    print(f'{"load":<8}{"均总力(N)":>12}{"均左占比":>12}'
          f'{"组内斜率(%/kN)":>18}')
    for load_key, fm, sm, sl in per_trial:
        print(f'{load_key:<8}{fm:>12.1f}{sm:>12.1%}{sl * 100:>18.2f}')

    f_all = np.concatenate(pooled_f)
    s_all = np.concatenate(pooled_s)
    g_slope, _ = _linfit(f_all, s_all)

    # 按瞬时总力分箱，看各试次在同一力水平上是否一致
    edges = np.quantile(f_all, np.linspace(0, 1, SATURATION_BINS + 1))
    print(f'\n{"总力区间(N)":>20}{"帧数":>10}{"左占比":>10}'
          f'{"试次间标准差":>14}')
    between = []
    for k in range(SATURATION_BINS):
        lo, hi = edges[k], edges[k + 1]
        per_load = []
        n = 0
        for f, s in zip(pooled_f, pooled_s):
            mm = (f >= lo) & (f < hi)
            if int(mm.sum()) >= 10:
                per_load.append(float(np.mean(s[mm])))
                n += int(mm.sum())
        if len(per_load) < 2:
            continue
        sd = float(np.std(per_load))
        between.append(sd)
        print(f'{f"{lo:.0f}-{hi:.0f}":>20}{n:>10}'
              f'{np.mean(per_load):>10.1%}{sd:>14.1%}')

    mean_between = float(np.mean(between)) if between else np.nan
    print(f'\n  汇总斜率 = {g_slope * 1000 * 100:.2f} %/kN'
          f'（左侧占比每增 1 kN 总力的变化）')
    print(f'  同一力水平上的试次间标准差均值 = {mean_between:.1%}')
    print('\n判读: 试次间标准差小（≪汇总斜率带来的变化）-> 占比只由力的')
    print('      大小决定，与试次无关，指向传感器压缩非线性；')
    print('      试次间标准差大 -> 每组各自一个工作点，指向姿势/策略差异。')
    print('      最终确认需要鞋垫逐点数据：看单元格是否顶在量程上限。')


# ============================================================
#  主程序
# ============================================================

def main():
    base_dir = get_base_dir()
    config_path = get_config_path()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    load_keys = filter_load_keys(config, load_keys=LOAD_KEYS,
                                 modes=LOAD_MODES_FILTER,
                                 exclude=EXCLUDE_LOAD_KEYS)
    print(f'参与负载: {load_keys}（模式筛选: {LOAD_MODES_FILTER}）')
    print(f'其中只有 {CALIBRATION_MODES} 组会进入 S1/S6/S7 的标定类回归。')

    # 必须 include_insole=True，否则切片里没有 grf_l / grf_r，
    # 只能退回到机器人力，而机器人力无法回答“哪条腿承重多”。
    _subject, _pipeline, pipeline_results = \
        load_or_create_cutted_pipeline_results(
            config_path, include_xsens=False, include_insole=True,
            debug=False, cache_name=CACHE_NAME)

    data = collect_side_data(config, base_dir, pipeline_results, load_keys)
    if not data:
        beauty_print('没有任何负载收集到有效数据，无法校验。', type="warning")
        return

    # 标定类检查只能吃定负载组（见顶部 CALIBRATION_MODES 的说明）；
    # 对称性检查与四张图吃全部模式。
    calib = {k: v for k, v in data.items()
             if v.get('mode') in CALIBRATION_MODES}
    skipped = sorted(set(data.keys()) - set(calib.keys()))
    if skipped:
        print(f'\n[S1/S6/S7] 仅用定负载组 {sorted(calib.keys())}；'
              f'跳过 {skipped}（其等效负载本身就是从受力反推的，'
              f'再拿去对受力回归是循环论证）。')

    verdicts = Verdicts()
    if calib:
        check_force_calibration(calib, verdicts)
    else:
        beauty_print('一个定负载组都没有，S1/S6/S7 全部跳过；'
                     '外力总量标定本次无法验证。', type="warning")

    check_share_consistency(data, verdicts)
    check_kinematic_side(data, verdicts)
    check_channel_health(data, verdicts)

    if calib:
        check_side_gain(calib, verdicts)
        check_saturation(calib, verdicts)

    verdicts.report()

    if PLOT_FIGURES:
        out_dir = None
        if SAVE_FIGURES and _subject is not None:
            out_dir = os.path.join(_subject.result_folder, 'symmetry')
        print('\n' + '=' * 80)
        print('[图] 对称性五图：SI 热图 / 运动链传递 / 左-右散点 / 蝴形图 / '
              'SI 趋势（vs 合力、vs 杆高）')
        print('=' * 80)
        plot_symmetry_figures(data, MOMENT_BASES, ANGLE_BASES,
                              out_dir=out_dir, show=True)


if __name__ == '__main__':
    main()