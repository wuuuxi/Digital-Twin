"""
symmetry.py — 左右对称性校验的【算法层】（纯分析）。

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
三者判断为独立、不等价，逻辑关系已标注在各项内部：

检查项：
  [S1] 外力总量标定：总力对配重回归，斜率应等于 g = 9.81 N/kg。
       截距的期望值取决于数据源：鞋垫 GRF -> 截距/g ≈ 体重；机器人力 -> ≈0。
  [S2] 左右力分担：share_r = mean(R) / (mean(L) + mean(R))。
  [S3] 力分担 vs ID 力矩分担：只把【方向相反】判为错误。
  [S4] 力分担 vs 运动学不对称（软关联，仅提示）。
  [S5] 单侧通道失效：某侧恒为常数 / 几乎为零 / 出现负值。
  [S6] 每侧增量增益：GRF 对【实测杆力】回归，总斜率=1.0、单侧≈0.5。
  [S7] 饱和 vs 策略区分器：各试次是否塌缩到同一条 share(F) 曲线。

对外接口：
  SymmetryCheckOptions() — 所有判据阈值 / 坐标集合的配置对象，
                           由 example 按采集实际情况填写
  collect_side_data()    — 单步取数（含 movement_types 过滤）
  Verdicts / check_*()   — 汇总与各项检查

注意：主流程入口 run_symmetry_check()（加载流水线结果 + 触发绘图）在
digitaltwin/pipelines/symmetry_check.py，本模块只提供纯计算部分。
"""
import os

import numpy as np

from digitaltwin.utils.data_io import canonical_load_key
from digitaltwin.utils.logger import beauty_print
from digitaltwin.osim.mot_pipeline import get_mot_files
from digitaltwin.analysis.result_analysis import (
    read_opensim_table,
    get_segment_from_results,
    get_inverse_dynamics_path,
    find_id_moment_column,
    interpolate_column_to_segment,
)
from digitaltwin.config_manager import get_load_mode


G = 9.81

# 外力数据源优先级：(左列, 右列, 标记)。
# 鞋垫 GRF 才是 ID 实际用的外力；机器人力只是降级退路。
DEFAULT_FORCE_SOURCES = (
    ('grf_l', 'grf_r', 'insole'),
    ('force_l', 'force_r', 'robot'),
)
# 默认用哪些关节的 ID 力矩算左右分担 / 看运动学不对称
DEFAULT_MOMENT_BASES = ('knee_angle', 'hip_flexion', 'ankle_angle')
DEFAULT_ANGLE_BASES = ('knee_angle', 'hip_flexion', 'ankle_angle')
# 切片缓存名：必须与不含鞋垫的缓存分开，否则会读到没有 grf 列的旧缓存
DEFAULT_CACHE_NAME = 'cutted_data_insole.csv'


class SymmetryCheckOptions:
    """
    校验的全部判据阈值与参与检查的坐标集合。

    example 只负责按采集实际情况填这份配置，判断逻辑全在本模块。
    所有字段都有与旧 example 常量一一对应的默认值，不传即为原行为。
    """

    def __init__(self, *,
                 slope_tol=0.15,
                 body_mass_range=(40.0, 150.0),
                 share_warn=0.10,
                 share_consistency_tol=0.12,
                 peak_diff_warn=5.0,
                 force_sources=DEFAULT_FORCE_SOURCES,
                 robot_intercept_tol=100.0,
                 cache_name=DEFAULT_CACHE_NAME,
                 total_gain_ideal=1.0,
                 total_gain_tol=0.08,
                 side_gain_ideal=0.5,
                 side_gain_tol=0.08,
                 body_mass_agree_tol=5.0,
                 saturation_bins=8,
                 moment_bases=DEFAULT_MOMENT_BASES,
                 angle_bases=DEFAULT_ANGLE_BASES,
                 cycle_grid_points=101,
                 calibration_modes=('isotonic',),
                 movement_types=('upward', 'downward')):
        # [S1] 回归斜率允许偏离 g 的相对量
        self.slope_tol = slope_tol
        # [S1] 截距推算出的体重合理区间（kg）
        self.body_mass_range = body_mass_range
        # [S2] 左右分担偏离 50% 多少算“明显偏侧”
        self.share_warn = share_warn
        # [S3] 力分担与力矩分担的允许差
        self.share_consistency_tol = share_consistency_tol
        # [S4] 关节角峰值差超过多少度算“明显不对称”
        self.peak_diff_warn = peak_diff_warn
        self.force_sources = tuple(force_sources)
        # [S1] 机器人力模式下，截距应接近 0（N）
        self.robot_intercept_tol = robot_intercept_tol
        self.cache_name = cache_name
        # [S6] GRF 总力对实测杆力的回归斜率，理论值 1.0
        self.total_gain_ideal = total_gain_ideal
        self.total_gain_tol = total_gain_tol
        # [S6] 单侧增量斜率，理论值 0.5
        self.side_gain_ideal = side_gain_ideal
        self.side_gain_tol = side_gain_tol
        # [S6] 两条路径（S1 名义配重 / S6 实测杆力）推算体重的允许差（kg）
        self.body_mass_agree_tol = body_mass_agree_tol
        # [S7] 饱和判别：按瞬时总力分箱的箱数
        self.saturation_bins = saturation_bins
        # [S3] 用哪些关节的 ID 力矩算左右分担
        self.moment_bases = tuple(moment_bases)
        # [S4] 用哪些关节角看运动学不对称
        self.angle_bases = tuple(angle_bases)
        # 蝴形图需要逐 cycle 的左右关节角曲线，重采样到这么多个点
        self.cycle_grid_points = int(cycle_grid_points)
        # 标定类回归（S1/S6/S7）只吃这些模式的负载组
        self.calibration_modes = tuple(calibration_modes)
        # collect_side_data 默认取的段类型（上升+下降）
        self.movement_types = tuple(movement_types)


def _find_base_dir():
    '''向上找到含 digitaltwin 包的项目根目录。'''
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, 'digitaltwin')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.abspath(__file__))


def pick_force_source(seg, force_sources):
    """选定外力列。优先鞋垫 GRF，否则退回机器人力。"""
    for col_l, col_r, tag in force_sources:
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


_canon_load_key = canonical_load_key


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

def collect_side_data(config, base_dir, pipeline_results, load_keys,
                      options=None, movement_types=None):
    """每个负载汇总三路信息。

    movement_types 默认用 options.movement_types（上升+下降）。
    传 ('upward',) 就得到只含上升阶段的同构数据，供“仅上升”趋势图使用。
    这里必须重新走一遍整个函数而不能事后过滤：ID 力矩与关节角峰值
    都是在取数时就按 seg 窗口聚合成标量的，拿到结果再滤已经没有逐帧信息。
    等长组没有 upward 段，get_segment_from_results 会自动回退到等长段，
    因此 IM 组在“仅上升”的那份里是整组保留的，不需要在这里特殊处理。

    Parameters
    ----------
    config : dict -- 配置内容
    base_dir : str -- 项目根目录（result/ 的上一级）
    pipeline_results : dict -- load_or_create_cutted_pipeline_results 的返回值
    load_keys : list[str]
    options : SymmetryCheckOptions, optional
    movement_types : tuple, optional -- 覆盖 options.movement_types

    Returns
    -------
    dict {load_key: {...}}
    """
    if options is None:
        options = SymmetryCheckOptions()
    mts = tuple(movement_types) if movement_types else options.movement_types
    mot_by_key = {_canon_load_key(k): v
                  for k, v in get_mot_files(config, base_dir).items()}
    out = {}

    for load_key in load_keys:
        seg = get_segment_from_results(pipeline_results, load_key,
                                       movement_types=mts)
        if seg is None or len(seg) == 0:
            print(f'[MISS] load={load_key}: 无切片数据，跳过')
            continue
        col_l, col_r, tag = pick_force_source(seg, options.force_sources)
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
            for base in options.moment_bases:
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
                # 逐段取并集，而不是用 [min, max] 这个大包络。
                # 包络会把两次深蹲【之间】的站立、调整站位、上下杠全算进来，
                # 而 rec['angles'] 取的是峰值（nanmax|·|），只要有一帧越界的姿势
                # 比深蹲中的峰值大，它就会直接改写这个峰值，S4 的左右峰值差
                # 就不再是深蹲的峰值差。其余各路（力、ID 力矩、逐 cycle 曲线）
                # 本来就已按段取，这里是唯一一处没对齐的口径。
                if 'segment_id' in seg.columns:
                    win_mask = np.zeros(t.shape, dtype=bool)
                    for sid in seg['segment_id'].unique():
                        sub = seg[seg['segment_id'] == sid]
                        if len(sub) < 10:
                            continue
                        win_mask |= ((t >= float(sub['time'].min()))
                                     & (t <= float(sub['time'].max())))
                else:
                    win_mask = ((t >= float(seg['time'].min()))
                                & (t <= float(seg['time'].max())))
                w = mot_df[win_mask]
                for base in options.angle_bases:
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
                grid = np.linspace(0.0, 100.0, options.cycle_grid_points)
                mot_t = mot_df['time'].values.astype(float)
                if 'segment_id' in seg.columns:
                    seg_ids = list(seg['segment_id'].unique())
                else:
                    seg_ids = [None]

                for base in options.angle_bases:
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


def check_force_calibration(data, verdicts, options):
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

    ok_slope = dev <= options.slope_tol
    verdicts.add('S1 外力增益', 'all', ok_slope,
                 f'总力对配重的斜率 {slope:.2f} N/kg 偏离 g 达 {dev:.0%}；'
                 f'斜率偏低意味着重负时外力被系统性低估，'
                 f'ID 力矩会随负载“涨不动”')

    if src == 'insole':
        body_mass = intercept / G
        print(f'  截距/g -> 推算体重 ≈ {body_mass:.1f} kg')
        lo, hi = options.body_mass_range
        ok_mass = lo <= body_mass <= hi
        verdicts.add('S1 截距合理', 'all', ok_mass,
                     f'鞋垫 GRF 截距推算体重 {body_mass:.1f} kg 不在 '
                     f'{lo:.0f}-{hi:.0f} kg 内；'
                     f'GRF 必须包含体重，偏离说明鞋垫被去基线或标定有误')
    else:
        print(f'  机器人力模式：截距应接近 0（杆力不包含体重）')
        ok_zero = abs(intercept) <= options.robot_intercept_tol
        verdicts.add('S1 截距合理', 'all', ok_zero,
                     f'机器人力截距 {intercept:.1f} N 偏离 0 超过 '
                     f'{options.robot_intercept_tol:.0f} N，提示零点漂移')

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


def check_share_consistency(data, verdicts, options):
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
                  for b in options.moment_bases) + '   判定')

    for load_key, r in sorted(data.items(), key=_load_sort_key):
        f_share = _share(float(np.nanmean(r['force_l'])),
                         float(np.nanmean(r['force_r'])))
        row = f'{load_key:<8}{f_share:>10.1%}'

        notes = []
        for base in options.moment_bases:
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
            elif gap > options.share_consistency_tol:
                notes.append(f'{base}:差{gap:.0%}')

        print(row + '   ' + (', '.join(notes) if notes else 'OK'))

    print('\n判读: 只有【方向相反】才是错误，而且只在数据源是鞋垫 GRF 时成立。')
    print('      分担差值大不算错：力矩 = 力 × 力臂，而力臂由 COP 与关节中心')
    print('      决定，两侧本来就可以不同。但若差值随负载单调变大，就是')
    print('      COP（目前是常数）跟不上真实压心前移的典型信号。')


def check_kinematic_side(data, verdicts, options):
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
            if abs(diff) > options.peak_diff_warn \
                    and abs(f_share - 0.5) > options.share_warn:
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

def check_channel_health(data, verdicts, options):
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


def check_side_gain(data, verdicts, options):
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

    print(f'\n  总斜率   = {s_tot:.4f}（理论 {options.total_gain_ideal:.2f}）'
          f'   截距 = {i_tot:.1f} N -> 体重 ≈ {body_mass:.1f} kg')
    print(f'  左侧斜率 = {s_l:.4f}（理论 {options.side_gain_ideal:.2f}）'
          f'   静态截距 = {i_l:.1f} N')
    print(f'  右侧斜率 = {s_r:.4f}（理论 {options.side_gain_ideal:.2f}）'
          f'   静态截距 = {i_r:.1f} N')
    if np.isfinite(i_l + i_r) and abs(i_l + i_r) > 1e-6:
        print(f'  空载站立时左侧占比 = {i_l / (i_l + i_r):.1%}，'
              f'而新增载荷的左侧占比 = {s_l / max(s_l + s_r, 1e-9):.1%}')

    ok_tot = np.isfinite(s_tot) and \
        abs(s_tot - options.total_gain_ideal) <= options.total_gain_tol
    verdicts.add('S6 总量闭合', 'all', ok_tot,
                 f'GRF 总力对杆力的斜率 {s_tot:.3f}，偏离 1.0 超过 '
                 f'{options.total_gain_tol:.0%}；每向杠上施 1 N，鞋垫只多读到 '
                 f'{s_tot:.2f} N -> 重负时外力被系统性低估')

    for side, s in (('L', s_l), ('R', s_r)):
        ok = np.isfinite(s) and \
            abs(s - options.side_gain_ideal) <= options.side_gain_tol
        verdicts.add(f'S6 单侧增益{side}', 'all', ok,
                     f'{side} 侧增量斜率 {s:.3f}，偏离 0.5 超过 '
                     f'{options.side_gain_tol:.2f}；新增载荷没有两腿平分，'
                     f'该侧外力比实际偏{"小" if s < options.side_gain_ideal else "大"}')

    print('\n判读: 总斜率 < 1.0 -> 鞋垫整体欠读，ID 外力偏小。')
    print('      单侧斜率偏离 0.5 -> 新增载荷没有两腿平分。这会直接把')
    print('      不对称写进 ID 外力，而膝力矩是小残差，会被放大好几倍。')
    print('      注意：静态截距与增量斜率是两回事。单纯的增益错误会同时')
    print('      缩小截距和斜率；若截距正常而斜率偏低，则是压缩型非线性')
    print('      （高压下饱和）或真实的重心转移——由 S7 区分。')


# ============================================================
#  [S7] 饱和 vs 策略 区分器
# ============================================================

def check_saturation(data, verdicts, options):
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
    edges = np.quantile(f_all, np.linspace(0, 1, options.saturation_bins + 1))
    print(f'\n{"总力区间(N)":>20}{"帧数":>10}{"左占比":>10}'
          f'{"试次间标准差":>14}')
    between = []
    for k in range(options.saturation_bins):
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
#  主流程入口已移至 digitaltwin/pipelines/symmetry_check.py
#  （它加载切片流水线结果并触发绘图；本模块只剩纯计算部分）
# ============================================================
