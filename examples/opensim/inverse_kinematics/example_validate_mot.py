"""
example_validate_mot.py

独立校验 Step 1 生成的 .mot 关节角是否可信。

为什么需要它：ID 力矩是关节角的二阶导数 + 外力的函数，
运动学一旦错了（绕接/符号/丢帧/标定），ID 不会报错，只会静默地给出
看起来很“正常”的错数字。因此必须在进入 ID 之前先把 .mot 判定为可信。

每项检查都是一个可否定的判据（falsifiable），而不是“看图感觉对”：

  [C0] 新版 Step 1 指纹：.mot 旁路文件 .dropouts.csv 是否存在，且 .mot 比 Excel 新。
       → 如果缺失，说明 .mot 是旧版代码生成的，unwrap 与丢帧检测都没生效。
  [C1] 基本信息：帧数/时长/实际采样率/时间单调递增/header 的 inDegrees。
  [C2] 幅值合理性：各坐标 |最大值| 是否在生理范围内（用绝对值，不依赖符号约定）。
  [C3] 连续性：相邻帧角速度上限，并单独识别 |Δ| ≈ 180°/360° 的绕接指纹。
  [C4] 跳负载一致性：同一坐标在 6 个负载间的均值不应出现离群点；
       若某负载均值 ≈ −中位数，则直接定位为符号/绕接翻转（106 kg 的 pelvis_tilt 就是这类）。
  [C5] 丢帧区间：现算 detect_frozen_intervals，列出区间与占比。
  [C6] 左右对称性：双腿深蹲下 knee_angle_l/r 应高度相关且峰值接近。
  [C2b] 限幅（饱和）检测：若某列有很多帧恰好等于其极值，说明波形被削平，
       其角速度/角加速度已不可信（卸到 ID 里就是错的惯性项）。
  [C7] 量纲互验（只在深蹲切片窗口内比较）：pelvis_ty 行程 / 机器人 pos_l 行程。
       不要求 1:1（杆在肩上，前倾使肩的行程大于骶盆），而是要求该比例
       在各负载间稳定；比例稳定即证明 Xsens 段长标定与单位正确。
  [C8] 原始 Xsens vs mot 的左右对称性对照：直接读 Xsens 'Joint Angles' 表，
       回答 C6 无法回答的问题——不对称是受试本来就有的，还是我们的转换引入的。

除 C0 / C1 / C5 外，所有判定都只在深蹲切片窗口内进行（RESTRICT_TO_SQUAT）。
试次前后受试者会走动、转身、调整杠铃，那些帧是真实运动却不属于被评估的动作，
算进来会同时抬高幅值、制造假的“非物理跳变”、并压低左右相关性。
C0/C1 是文件级属性，C5 故意跑整文件（窗口外的丢帧同样反映采集质量）。

运行时机：每次重跑 example_xsens_to_mot.py 之后、进入 Step 3 之前。
"""
import os
import json

import numpy as np
import pandas as pd

from digitaltwin.utils.logger import beauty_print
from digitaltwin.osim.mot_pipeline import (
    get_mot_files,
    detect_frozen_intervals,
    dropout_sidecar_path,
    DROPOUT_MONITOR_COLUMNS,
)
from digitaltwin.analysis.result_analysis import (
    read_opensim_table,
    get_load_keys,
    load_or_create_cutted_pipeline_results,
    get_segment_from_results,
    get_action_windows,
)
from digitaltwin.config_manager import filter_load_keys


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../../config/20260513_squat_FTS09_xsens.json'

LOAD_KEYS = None
EXCLUDE_LOAD_KEYS = []
# 组名已改成 '20kg' / 'IK-0.3m/s' / 'IM-1m'，硬写组名的排除表会全部失效，
# 而且失效时不会报错，只会把等速 / 等长组静静地拉进跳负载一致性统计。
# 改为按负载模式筛选：只校验定负载组，以后新增组不用再改脚本。
# 要一并校验等速 / 等长组时，写 LOAD_MODES_FILTER = None。
LOAD_MODES_FILTER = ('isotonic',)

# 只在深蹲切片窗口内做幅值 / 连续性 / 对称性判定。
# 窗口直接复用 pipeline 已有的动作切片结果（get_segment_from_results），
# 与 C7 用的是同一套，不重新实现切分逻辑。
RESTRICT_TO_SQUAT = True
SQUAT_MOVEMENT_TYPES = ('upward', 'downward')

# [C2] 生理上限（度）。用 |值| 比较，因此不依赖屈/伸符号约定。
ABS_MAX_DEG = {
    'pelvis_tilt': 60.0,
    'pelvis_list': 45.0,
    # pelvis_rotation 是绕竖直轴的朝向（yaw），不改变重力力臂，
    # 对关节力矩无影响。只查绕接，不设生理上限。
    'pelvis_rotation': 180.0,
    'hip_flexion_l': 140.0,
    'hip_flexion_r': 140.0,
    'hip_adduction_l': 45.0,
    'hip_adduction_r': 45.0,
    'knee_angle_l': 150.0,
    'knee_angle_r': 150.0,
    'ankle_angle_l': 50.0,
    'ankle_angle_r': 50.0,
    'lumbar_extension': 60.0,
}

# [C3] 角速度上限（度/秒）。深蹲是慢动作，居中节奏不超过 300 °/s。
MAX_ANG_VEL = 500.0
# 绕接指纹：单帧跳变接近 180° 或 360°
WRAP_BANDS = ((150.0, 210.0), (330.0, 390.0))

# [C2b] 限幅检测：超过 SAT_FRAC 的帧与极值的差小于 SAT_TOL 就判为被削平。
SAT_TOL = 1e-3       # 度
SAT_FRAC = 0.005     # 0.5% 的帧

# [C4] 跳负载一致性要监视的坐标与均值允许偏差（度）
# 不监视 pelvis_rotation：受试站位朝向本来就会逐次不同，不是数据错误。
CROSS_LOAD_COORDS = ('pelvis_tilt', 'pelvis_list',
                     'lumbar_extension', 'hip_flexion_l', 'knee_angle_l',
                     'ankle_angle_l')
CROSS_LOAD_TOL = 20.0

# [C6] 左右对称性
SYMMETRY_PAIRS = (('knee_angle_l', 'knee_angle_r'),
                  ('hip_flexion_l', 'hip_flexion_r'),
                  ('ankle_angle_l', 'ankle_angle_r'))
SYMMETRY_MIN_CORR = 0.90
SYMMETRY_MAX_PEAK_DIFF = 15.0

# [C8] 原始 Xsens vs mot 对照。需要重新读 Excel，比其他检查慢，
# 不需要时可关掉。
CHECK_RAW_XSENS = True
RAW_SHEET = 'Joint Angles ZXY'
# mot 坐标名 -> Xsens 'Joint Angles' 表的列名。
# 必须与 mot_pipeline.read_xsens_excel_for_opensim 里的 jmap 逐行一致，
# 否则本检查对比的根本不是同一个量。
RAW_COLUMN_MAP = {
    'hip_flexion_r':   'Right Hip Flexion/Extension',
    'hip_flexion_l':   'Left Hip Flexion/Extension',
    'hip_adduction_r': 'Right Hip Abduction/Adduction',
    'hip_adduction_l': 'Left Hip Abduction/Adduction',
    'hip_rotation_r':  'Right Hip Internal/External Rotation',
    'hip_rotation_l':  'Left Hip Internal/External Rotation',
    'knee_angle_r':    'Right Knee Flexion/Extension',
    'knee_angle_l':    'Left Knee Flexion/Extension',
    'ankle_angle_r':   'Right Ankle Dorsiflexion/Plantarflexion',
    'ankle_angle_l':   'Left Ankle Dorsiflexion/Plantarflexion',
}

# [C7] 量纲互验
CHECK_ROBOT_STROKE = True
# 不要求 1:1：杆在肩上，躯干前倾使肩的升降行程大于骶盆行程，
# 比例约 0.7 是生理上合理的。真正可否定的判据是：该比例在各负载间
# 应该稳定。某个负载的比例单独偏离，才意味着那一次的 Xsens 幅值有问题。
STROKE_RATIO_TOL = 0.15      # 比例偏离中位数的上限
ROBOT_SIGNAL = 'pos_l'
ROBOT_UNIT_TO_M = 1.0        # pos_l 若为 mm 则改为 0.001


def get_base_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


def _canon_load_key(value):
    try:
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f'{f:g}'
    except Exception:
        return str(value)


def _read_frame_rate(xlsx, default=60.0):
    """从 General Information 表读采样率；读不到则退回默认值。"""
    try:
        info = pd.read_excel(xlsx, sheet_name='General Information', header=None)
        for _, row in info.iterrows():
            if str(row[0]).strip() == 'Frame Rate':
                fs = float(row[1])
                if np.isfinite(fs) and fs > 0:
                    return fs
    except Exception:
        pass
    return default


def get_squat_windows(config_path, load_keys):
    """{load_key: (t0, t1)}：每个负载的深蹲切片时间窗。

    注意这是【首次到末次】深蹲的包络窗口，内部仍含组间站立，
    但已排除试次前后的走动与转身——后者才是污染对称性统计的主因。
    """
    if not RESTRICT_TO_SQUAT:
        return {}
    try:
        info = get_action_windows(config_path, load_keys,
                                  movement_types=SQUAT_MOVEMENT_TYPES,
                                  debug=False)
    except Exception as exc:
        beauty_print(f'无法加载动作切片（{exc}）；本次退回整文件统计，'
                     'C2/C3/C6/C8 会把试次前后的走动与转身也算进去，结论偏严。',
                     type="warning")
        return {}

    windows = {}
    for load_key, item in info.items():
        if item.get('window') is None:
            beauty_print(f'组 {load_key} 未取到动作窗口（'
                         f'{item.get("detail", "")}），该组按整文件统计。',
                         type="warning")
            continue
        windows[_canon_load_key(load_key)] = item['window']
    return windows


class Verdicts:
    """汇总 PASS/FAIL，便于最后一眼判定 .mot 能不能用。"""

    def __init__(self):
        self.items = []

    def add(self, check, load_key, ok, detail=''):
        self.items.append((check, str(load_key), bool(ok), detail))

    def report(self):
        print('\n' + '=' * 80)
        print('[汇总] .mot 可信度判定')
        print('=' * 80)
        fails = [i for i in self.items if not i[2]]
        if not fails:
            print('[PASS] 全部检查通过；可以进入 Step 3 / 敏感性分析。')
            return True
        lines = [f'{len(fails)} 项校验未通过：']
        for check, load_key, _, detail in fails:
            lines.append(f'  - {check:<28} load={load_key:<6} {detail}')
        lines.append('在修好这些项之前，ID 力矩的单调性结论无效。')
        beauty_print('\n'.join(lines), type="warning")
        return False


# ============================================================
#  [C0] 新版 Step 1 指纹
# ============================================================

def check_step1_freshness(config, base_dir, mot_by_key, verdicts):
    print('\n' + '=' * 80)
    print('[C0] Step 1 新鲜度（.mot 是否由包含 unwrap + 丢帧检测的新版生成）')
    print('=' * 80)
    print(f'{"load":<8}{"旁路文件":>12}{"mot mtime":>22}{"比Excel新":>12}')

    # config['folder'] 是绝对路径，xsens_folder 是它下面的子目录；
    # 之前用 base_dir 拼接会找不到 Excel，导致“比Excel新”一列全是 N/A。
    xsens_folder = os.path.join(config.get('folder', ''),
                                config.get('modeling_file', {})
                                .get('xsens_folder', 'xsens'))
    data = config.get('modeling_file', {}).get('data', {})

    for load_key, mot_path in mot_by_key.items():
        if not os.path.exists(mot_path):
            print(f'{load_key:<8}  .mot 不存在: {mot_path}')
            verdicts.add('C0 mot存在', load_key, False, 'mot 文件缺失')
            continue

        sidecar = dropout_sidecar_path(mot_path)
        has_sidecar = os.path.exists(sidecar)
        # 同时提示 raw 版本是否存在：example_xsens_to_mot.py 会先写 *_raw.mot，
        # 再把 clip 后的结果覆盖成标准名。对比两者可分离“Xsens 本身的问题”
        # 与“clip 造成的削平”。
        raw_path = mot_path.replace('_opensim.mot', '_opensim_raw.mot')
        if os.path.exists(raw_path):
            print(f'         (raw 版本存在，可对比: {os.path.basename(raw_path)})')
        mot_mtime = os.path.getmtime(mot_path)

        newer = 'N/A'
        entry = data.get(load_key) or data.get(str(load_key)) or {}
        xsens_file = entry.get('xsens_file')
        if xsens_file:
            xlsx = os.path.join(xsens_folder, xsens_file) \
                if xsens_folder else xsens_file
            if os.path.exists(xlsx):
                newer = 'yes' if mot_mtime > os.path.getmtime(xlsx) else 'no'

        print(f'{load_key:<8}{("yes" if has_sidecar else "no"):>12}'
              f'{pd.to_datetime(mot_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"):>22}'
              f'{newer:>12}')
        verdicts.add('C0 新版Step1生成', load_key, has_sidecar,
                     '缺 .dropouts.csv → .mot 仍为旧版产物，请重跑 example_xsens_to_mot.py')

    print('\n判读: 只要有一行旁路文件 = no，就说明 unwrap 与丢帧检测尚未作用于该 .mot，')
    print('      此时 example_inverse_dynamics.py 只是在重算旧运动学，结果与修改前逐位相同。')


# ============================================================
#  [C1][C2][C3][C5] 逐文件检查
# ============================================================

def check_single_mot(load_key, mot_path, verdicts, window=None):
    df_full = read_opensim_table(mot_path)
    if df_full is None or 'time' not in df_full.columns:
        verdicts.add('C1 可读', load_key, False, '无法读取或缺 time 列')
        return None

    t = df_full['time'].values.astype(float)
    dt = np.diff(t)

    # ---- C1 基本信息 ----
    fs = 1.0 / np.median(dt) if len(dt) else float('nan')
    mono = bool(np.all(dt > 0))
    jitter = (float(np.max(np.abs(dt - np.median(dt))) / np.median(dt))
              if len(dt) else float('inf'))
    with open(mot_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = ''.join(f.readline() for _ in range(12))
    in_degrees = 'inDegrees' in header

    print(f'\n--- load={load_key}  {os.path.basename(mot_path)} ---')
    print(f'  帧数={len(t)}  时长={t[-1] - t[0]:.2f}s  实际fs={fs:.2f}Hz  '
          f'步长抖动={jitter:.2%}  时间单调={mono}  header含inDegrees={in_degrees}')
    verdicts.add('C1 时间单调', load_key, mono, '时间列非严格递增')
    # 不假定具体帧率（Awinda 60 Hz / MVN Link 240 Hz 都可能），
    # 只要求采样均匀：步长抖动大才意味着跳帧或帧索引不连续。
    verdicts.add('C1 采样均匀', load_key, jitter < 0.05,
                 f'时间步长抖动 {jitter:.1%} > 5%（实际 fs≈{fs:.2f}Hz），'
                 f'可能有跳帧或帧索引不连续')
    verdicts.add('C1 inDegrees', load_key, in_degrees,
                 'header 缺 inDegrees=yes，OpenSim 可能按弧度解释（日志里的 '
                 '“assuming rotations in Degrees” 只是因为文件被当成旧版格式）')

    # ---- 截取深蹲窗口 ----
    # 下面的 C2/C2b/C3/C6 均只在此窗口内判定。试次前后的走动、转身、
    # 弯腰上下杠都是真实运动，但不属于被评估的动作；把它们算进来会
    # 同时抬高幅值、制造假的非物理跳变，并压低左右相关性。
    if window is not None:
        t0, t1 = window
        m = (t >= t0) & (t <= t1)
        df = df_full[m].reset_index(drop=True)
        print(f'  [窗口] 深蹲切片 {t0:.2f}-{t1:.2f}s  '
              f'{int(m.sum())}/{len(t)} 帧 ({m.mean():.0%})')
    else:
        df = df_full
        print('  [窗口] 未启用深蹲切片，按整文件统计')
    tw = df['time'].values.astype(float)
    dtw = np.diff(tw)
    if len(tw) < 20:
        verdicts.add('C2 窗口样本', load_key, False,
                     f'深蹲窗口内只有 {len(tw)} 帧，无法判定')
        return df

    # ---- C2 幅值合理性 ----
    print(f'  {"coord":<20}{"min":>10}{"max":>10}{"|max|":>10}{"限值":>8}{"判定":>8}')
    for coord, limit in ABS_MAX_DEG.items():
        if coord not in df.columns:
            continue
        v = df[coord].values.astype(float)
        amax = float(np.nanmax(np.abs(v)))
        ok = amax <= limit
        print(f'  {coord:<20}{np.nanmin(v):>10.2f}{np.nanmax(v):>10.2f}'
              f'{amax:>10.2f}{limit:>8.0f}{("OK" if ok else "FAIL"):>8}')
        verdicts.add('C2 幅值范围', load_key, ok,
                     f'{coord} |max|={amax:.1f}° > {limit:.0f}°')

    # ---- C2b 限幅（饱和）检测 ----
    sat_hits = []
    for coord in df.columns:
        if coord == 'time' or coord.startswith('pelvis_t'):
            continue
        v = df[coord].values.astype(float)
        v = v[np.isfinite(v)]
        if v.size < 20:
            continue
        # 恒定列不是“被削平”，而是被设计成常数：heading_free 模式下
        # pelvis_rotation 恒为 0。峰峰值接近 0 时必须跳过，否则每一帧
        # 都等于“极值”，会 100% 命中而产生假阳性。
        if float(np.max(v) - np.min(v)) < 1.0:
            continue
        for edge in (float(np.max(v)), float(np.min(v))):
            frac = float(np.mean(np.abs(v - edge) <= SAT_TOL))
            if frac >= SAT_FRAC:
                sat_hits.append((coord, edge, frac))
    if sat_hits:
        sat_hits.sort(key=lambda x: -x[2])
        print('  [C2b] 疑似限幅（被削平）的列: ' + ', '.join(
            f'{c}@{e:.2f}°({f:.1%}帧)' for c, e, f in sat_hits[:6]))
    else:
        print('  [C2b] 未检测到限幅')
    verdicts.add('C2b 无限幅', load_key, not sat_hits,
                 '数值被削平: ' + ', '.join(
                     f'{c}@{e:.2f}°' for c, e, _ in sat_hits[:6]))

    # ---- C3 连续性 / 绕接指纹 ----
    angle_cols = [c for c in df.columns
                  if c != 'time' and not c.startswith('pelvis_t')]
    worst = []
    wrap_hits = []
    for c in angle_cols:
        v = df[c].values.astype(float)
        d = np.abs(np.diff(v))
        with np.errstate(invalid='ignore', divide='ignore'):
            vel = d / dtw
        if len(vel) and np.nanmax(vel) > MAX_ANG_VEL:
            worst.append((c, float(np.nanmax(vel))))
        n_wrap = int(sum(int(np.nansum((d >= lo) & (d <= hi)))
                         for lo, hi in WRAP_BANDS))
        if n_wrap:
            wrap_hits.append((c, n_wrap))

    if worst:
        worst.sort(key=lambda x: -x[1])
        print(f'  [C3] 角速度超 {MAX_ANG_VEL:.0f}°/s 的列: '
              + ', '.join(f'{c}({v:.0f})' for c, v in worst[:6]))
    else:
        print(f'  [C3] 无列超过 {MAX_ANG_VEL:.0f}°/s')
    verdicts.add('C3 连续性', load_key, not worst,
                 '存在非物理跳变: ' + ', '.join(c for c, _ in worst[:6]))

    if wrap_hits:
        print('  [C3] 绕接指纹（单帧跳变≈180°/360°）: '
              + ', '.join(f'{c}×{n}' for c, n in wrap_hits[:6]))
    verdicts.add('C3 无绕接', load_key, not wrap_hits,
                 '仍有 180°/360° 跳变: ' + ', '.join(c for c, _ in wrap_hits[:6]))

    # ---- C5 丢帧区间 ----
    # C5 有意跑整文件：窗口外的丢帧同样反映这次采集的质量。
    cols = [c for c in DROPOUT_MONITOR_COLUMNS if c in df_full.columns]
    if cols:
        intervals = detect_frozen_intervals(t, df_full[cols].values.astype(float))
        if intervals:
            total = sum(t1 - t0 for t0, t1 in intervals)
            print(f'  [C5] 丢帧 {len(intervals)} 段 / {total:.2f}s '
                  f'({100.0 * total / (t[-1] - t[0]):.1f}%): '
                  + ', '.join(f'{a:.2f}-{b:.2f}s' for a, b in intervals[:6]))
        else:
            print('  [C5] 未检测到冻结/插值区间')

    # ---- C6 左右对称性 ----
    for left, right in SYMMETRY_PAIRS:
        if left not in df.columns or right not in df.columns:
            continue
        a = df[left].values.astype(float)
        b = df[right].values.astype(float)
        ok_mask = np.isfinite(a) & np.isfinite(b)
        if ok_mask.sum() < 20 or np.std(a[ok_mask]) < 1e-9:
            continue
        corr = float(np.corrcoef(a[ok_mask], b[ok_mask])[0, 1])
        peak_diff = abs(float(np.nanmax(np.abs(a)) - np.nanmax(np.abs(b))))
        ok = (corr >= SYMMETRY_MIN_CORR) and (peak_diff <= SYMMETRY_MAX_PEAK_DIFF)
        print(f'  [C6] {left} vs {right}: corr={corr:+.3f}  '
              f'峰值差={peak_diff:.1f}°  {"OK" if ok else "FAIL"}')
        verdicts.add('C6 左右对称', load_key, ok,
                     f'{left}/{right} corr={corr:+.2f}, 峰值差={peak_diff:.1f}°')

    return df


# ============================================================
#  [C4] 跳负载一致性
# ============================================================

def check_cross_load(frames, verdicts, coords=CROSS_LOAD_COORDS,
                     tol=CROSS_LOAD_TOL):
    print('\n' + '=' * 80)
    print('[C4] 跳负载一致性（均值应缓变，不应出现离群点）')
    print('=' * 80)

    load_keys = list(frames.keys())
    header = f'{"coord":<20}' + ''.join(f'{k:>12}' for k in load_keys) \
             + f'{"中位数":>10}{"判定":>26}'
    print(header)
    print('-' * len(header))

    for coord in coords:
        means, present = [], []
        for k in load_keys:
            df = frames[k]
            if df is None or coord not in df.columns:
                means.append(np.nan)
                continue
            m = float(np.nanmean(df[coord].values.astype(float)))
            means.append(m)
            present.append((k, m))
        if len(present) < 3:
            continue

        med = float(np.median([m for _, m in present]))
        flags = []
        for k, m in present:
            if abs(m - med) > tol:
                # 均值 ≈ −中位数 → 符号/绕接翻转，而不是真实姿态差异
                if abs(med) > 5.0 and abs(m + med) < 0.30 * abs(med):
                    flags.append(f'{k}:翻转')
                else:
                    flags.append(f'{k}:偏{m - med:+.0f}°')

        row = f'{coord:<20}' + ''.join(
            (f'{m:>12.2f}' if np.isfinite(m) else f'{"N/A":>12}') for m in means)
        row += f'{med:>10.2f}' + f'{(", ".join(flags) if flags else "OK"):>26}'
        print(row)

        for f in flags:
            k = f.split(':')[0]
            verdicts.add('C4 跳负载一致', k, False, f'{coord} {f}（中位数 {med:+.1f}°）')

    print('\n判读: “翻转”标记意为该负载的该坐标均值几乎恰好是其他负载的相反数。')
    print('      典型原因不是帧内绕接（那会被 C3 的 180°/360° 指纹抓到），')
    print('      而是服位朝向 heading 泄漏：受试朝向与全局 X 轴夹角近 180° 时，')
    print('      前后倾会整段变号而帧内没有任何跳变。Step 1 的 heading_free')
    print('      模式已在源头消除这一项。')


# ============================================================
#  [C7] 与机器人行程的量纲互验
# ============================================================

def check_robot_stroke(config_path, frames, verdicts):
    print('\n' + '=' * 80)
    print('[C7] 量纲互验: Xsens pelvis_ty 行程 vs 机器人 pos_l 行程')
    print('=' * 80)

    try:
        subject, _, pipeline_results = load_or_create_cutted_pipeline_results(
            config_path, include_xsens=False, debug=False)
    except Exception as exc:
        print(f'  无法加载 subject，跳过：{exc}')
        return

    path = os.path.join(subject.result_folder, 'aligned_data.csv')
    if not os.path.exists(path):
        print(f'  未找到 {path}，跳过。')
        return

    robot = pd.read_csv(path)
    load_col = next((c for c in ('load_weight', 'load', 'load_value')
                     if c in robot.columns), None)
    groups = ({_canon_load_key(k): g for k, g in robot.groupby(load_col)}
              if load_col else {'all': robot})

    print(f'{"load":<8}{"pelvis_ty行程(m)":>18}{"pos_l行程(m)":>16}{"比例":>10}')
    ratios = []

    for load_key, df in frames.items():
        rdf = groups.get(load_key, groups.get('all'))
        if df is None or rdf is None or 'pelvis_ty' not in df.columns \
                or ROBOT_SIGNAL not in rdf.columns:
            print(f'{load_key:<8}  缺 pelvis_ty 或 {ROBOT_SIGNAL}，跳过')
            continue

        # 只在深蹲切片窗口内比较。整文件比较是错的：试次前后的走动/
        # 坐下休息也会进入 pelvis_ty 的 5-95 百分位，把行程虚假放大。
        seg = get_segment_from_results(pipeline_results, load_key,
                                      movement_types=('upward', 'downward'))
        if seg is None or 'time' not in getattr(seg, 'columns', []) or len(seg) == 0:
            print(f'{load_key:<8}  无切片窗口，跳过')
            continue
        t0, t1 = float(seg['time'].min()), float(seg['time'].max())

        mt = df['time'].values.astype(float)
        ty = df['pelvis_ty'].values.astype(float)[(mt >= t0) & (mt <= t1)]
        pos_all = rdf[ROBOT_SIGNAL].values.astype(float) * ROBOT_UNIT_TO_M
        if 'time' in rdf.columns:
            rt = rdf['time'].values.astype(float)
            pos = pos_all[(rt >= t0) & (rt <= t1)]
        else:
            pos = pos_all
        if len(ty) < 10 or len(pos) < 10:
            print(f'{load_key:<8}  窗口内样本不足，跳过')
            continue
        # 用 5-95 百分位行程，避开单帧尖尖
        s_ty = float(np.nanpercentile(ty, 95) - np.nanpercentile(ty, 5))
        s_pos = float(np.nanpercentile(pos, 95) - np.nanpercentile(pos, 5))
        ratio = s_ty / max(abs(s_pos), 1e-9)
        ratios.append((load_key, s_ty, s_pos, ratio))
        print(f'{load_key:<8}{s_ty:>18.3f}{s_pos:>16.3f}{ratio:>10.3f}')

    if len(ratios) >= 3:
        med = float(np.median([r for *_, r in ratios]))
        print(f'\n  比例中位数 = {med:.3f}（骶盆行程 / 杆行程）')
        for load_key, s_ty, s_pos, ratio in ratios:
            dev = abs(ratio - med) / max(med, 1e-9)
            ok = dev <= STROKE_RATIO_TOL
            print(f'  {load_key:<8}比例={ratio:.3f}  偏离中位数 {dev:>6.1%}  '
                  f'{"OK" if ok else "FAIL"}')
            verdicts.add('C7 比例一致', load_key, ok,
                         f'骶盆/杆 行程比 {ratio:.3f}，偏离中位数 {med:.3f} 达 {dev:.0%}')

    print('\n判读: 比例本身不必等于 1——杆在肩上，躯干前倾使肩的升降大于骶盆，')
    print('      因此 0.6~0.8 是生理上合理的。可否定的判据是比例的跳负载稳定性：')
    print('      全部负载比例一致 -> Xsens 段长标定与 pos_l 单位都可信；')
    print('      某一负载单独偏离 -> 那一次的 Xsens 幅值可疑。')
    print('      这一项与相位一致性不同: 它检验的是幅值标定，而非时间对齐。')


# ============================================================
#  [C8] 原始 Xsens vs mot 的左右对称性对照
# ============================================================

def _lr_metrics(a, b):
    """返回 (corr, peak_diff)；样本不足或常量列返回 None。

    两个指标都对【两侧同时】变号免疫（corr 不变，peak 取了绝对值），
    所以 mot_pipeline 里的 sign_flip（如 hip_adduction 两侧同时乘 -1）
    不会影响原始与 mot 的可比性。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return None
    corr = float(np.corrcoef(a[m], b[m])[0, 1])
    peak_diff = abs(float(np.nanmax(np.abs(a)) - np.nanmax(np.abs(b))))
    return corr, peak_diff


def check_raw_xsens_symmetry(config, frames, verdicts, windows=None):
    """把 C6 的“有没有不对称”升级为“不对称是谁造成的”。

    C6 只能说明 .mot 里存在左右不对称，却无法回答这个不对称是
    受试者本来就有的，还是我们的 Xsens -> mot 转换引入的。
    本检查直接读 Xsens 'Joint Angles' 表（相邻 segment 的相对角，
    不经过任何我们写的坐标变换），用同一套判据再算一遍，再与 .mot 对照：

      原始 FAIL + mot FAIL -> 真实不对称，转换是忠实的，不该改程序；
      原始 OK   + mot FAIL -> 不对称是转换引入的，必须查 jmap / 符号 / 坐标变换；
      原始 FAIL + mot OK   -> 转换把真实差异抹平了（典型是被 clip 削顶）。
    """
    print('\n' + '=' * 80)
    print('[C8] 原始 Xsens 关节角 vs mot 的左右对称性对照')
    print('=' * 80)

    xsens_folder = os.path.join(config.get('folder', ''),
                                config.get('modeling_file', {})
                                .get('xsens_folder', 'xsens'))
    data = config.get('modeling_file', {}).get('data', {})

    print(f'{"joint":<14}{"load":>6}{"raw corr":>10}{"mot corr":>10}'
          f'{"raw diff":>10}{"mot diff":>10}   结论')
    print('-' * 80)

    for load_key, df in frames.items():
        if df is None:
            continue
        entry = next((v for k, v in data.items()
                      if _canon_load_key(k) == _canon_load_key(load_key)), None)
        xsens_file = (entry or {}).get('xsens_file')
        if not xsens_file:
            continue
        xlsx = os.path.join(xsens_folder, xsens_file) if xsens_folder else xsens_file
        if not os.path.exists(xlsx):
            print(f'{"":<14}{load_key:>6}   找不到 Excel，跳过: {xlsx}')
            continue
        try:
            raw = pd.read_excel(xlsx, sheet_name=RAW_SHEET)
        except Exception as exc:
            print(f'{"":<14}{load_key:>6}   读取 {RAW_SHEET} 失败，跳过: {exc}')
            continue

        # frames[load_key] 已经被截到深蹲窗口，原始表必须截同一个窗口，
        # 否则两边算的不是同一段时间，会凭空冒出“不一致”。
        win = (windows or {}).get(load_key)
        if win is not None:
            fs = _read_frame_rate(xlsx)
            fr = (raw['Frame'].values.astype(float) if 'Frame' in raw.columns
                  else np.arange(len(raw), dtype=float))
            raw_t = fr / fs
            raw = raw[(raw_t >= win[0]) & (raw_t <= win[1])]
            if len(raw) < 20:
                print(f'{"":<14}{load_key:>6}   窗口内原始帧不足，跳过')
                continue

        for left, right in SYMMETRY_PAIRS:
            cl, cr = RAW_COLUMN_MAP.get(left), RAW_COLUMN_MAP.get(right)
            if not cl or not cr:
                continue
            if cl not in raw.columns or cr not in raw.columns:
                continue
            if left not in df.columns or right not in df.columns:
                continue
            m_raw = _lr_metrics(raw[cl].values, raw[cr].values)
            m_mot = _lr_metrics(df[left].values, df[right].values)
            if m_raw is None or m_mot is None:
                continue

            raw_ok = (m_raw[0] >= SYMMETRY_MIN_CORR) and \
                     (m_raw[1] <= SYMMETRY_MAX_PEAK_DIFF)
            mot_ok = (m_mot[0] >= SYMMETRY_MIN_CORR) and \
                     (m_mot[1] <= SYMMETRY_MAX_PEAK_DIFF)

            if raw_ok == mot_ok:
                agree = True
                note = '一致（转换忠实）' if mot_ok else '一致（真实不对称）'
            elif mot_ok:
                agree = False
                note = '转换抹掉了真实不对称'
            else:
                agree = False
                note = '转换引入了不对称'

            base = left[:-2]
            print(f'{base:<14}{load_key:>6}{m_raw[0]:>10.2f}{m_mot[0]:>10.2f}'
                  f'{m_raw[1]:>10.1f}{m_mot[1]:>10.1f}   {note}')
            verdicts.add('C8 原始vs转换', load_key, agree,
                         f'{base}: {note}（raw corr={m_raw[0]:+.2f}/峰值差={m_raw[1]:.1f}°, '
                         f'mot corr={m_mot[0]:+.2f}/峰值差={m_mot[1]:.1f}°）')

    print('\n判读: 本项不再重复判定“对不对称”（那是 C6），而是判定两者是否吐口一致。')
    print('      一致 -> 转换忠实；C6 的 FAIL 属于受试者本身的动作特征，改程序无效。')
    print('      不一致 -> 问题在我们的代码里，先修代码再谈生理解释。')


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
    print(f'参与负载: {load_keys}')

    all_mot = {_canon_load_key(k): v
               for k, v in get_mot_files(config, base_dir).items()}
    mot_by_key = {_canon_load_key(k): all_mot[_canon_load_key(k)]
                  for k in load_keys if _canon_load_key(k) in all_mot}

    verdicts = Verdicts()

    check_step1_freshness(config, base_dir, mot_by_key, verdicts)

    print('\n' + '=' * 80)
    print('[C1][C2][C3][C5][C6] 逐文件检查')
    print('=' * 80)
    windows = get_squat_windows(config_path, list(mot_by_key.keys()))
    if RESTRICT_TO_SQUAT and not windows:
        beauty_print('未取到任何深蹲切片窗口，本次按整文件统计；'
                     'C2/C3/C6/C8 的结论会偏严。', type="warning")
    frames = {k: check_single_mot(k, p, verdicts, windows.get(k))
              for k, p in mot_by_key.items()}

    check_cross_load(frames, verdicts)

    if CHECK_RAW_XSENS:
        check_raw_xsens_symmetry(config, frames, verdicts, windows)

    if CHECK_ROBOT_STROKE:
        check_robot_stroke(config_path, frames, verdicts)

    verdicts.report()


if __name__ == '__main__':
    main()