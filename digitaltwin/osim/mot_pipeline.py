"""
mot_pipeline.py

Step 1: Xsens 数据 -> OpenSim .mot 关节角度文件。

包含：
  read_xsens_excel_for_opensim()  -- 解析 Xsens Excel 并生成 .mot
  run_step1_mot_conversion()      -- 批量转换流水线入口
  get_mot_files()                 -- 共享工具，供 muscle_analysis / inverse_dynamics 导入
  get_scaled_model()              -- 共享工具
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from digitaltwin.utils.logger import beauty_print


# ============================================================
#  共享工具函数（供 muscle_analysis.py / inverse_dynamics.py 导入）
# ============================================================

def get_mot_files(config, base_dir):
    """
    扫描 mot/ 目录，返回 {load_key: mot_file_path} 字典。
    仅返回已存在的文件。
    """
    experiment_label = config['experiment_label']
    mot_dir = os.path.join(base_dir, 'result', experiment_label, 'opensim', 'mot')
    mot_files = {}
    for load_key, file_info in config['modeling_file']['data'].items():
        xsens_file = file_info.get('xsens_file')
        if xsens_file is None:
            continue
        mot_name = Path(xsens_file).stem + '_opensim.mot'
        mot_path = os.path.join(mot_dir, mot_name)
        if os.path.exists(mot_path):
            mot_files[load_key] = mot_path
    return mot_files


def get_scaled_model(config, base_dir):
    """返回缩放后模型的完整路径（不检查文件是否存在）。"""
    experiment_label = config['experiment_label']
    opensim_dir = os.path.join(base_dir, 'result', experiment_label, 'opensim')
    return os.path.join(opensim_dir, f'whole body model_{experiment_label}.osim')


# ============================================================
#  数据质量工具（绕接展开 + 丢帧检测）
# ============================================================

# 丢帧检测监视的列：Xsens 掉线时全身列会同时“死掉”，
# 因此用多列同时线性作为判据，避免把单个关节真实的匀速运动误判为丢帧。
DROPOUT_MONITOR_COLUMNS = ('pelvis_tilt', 'hip_flexion_l',
                          'knee_angle_l', 'ankle_angle_l')


def unwrap_degrees(values):
    """
    消除角度序列的 ±180°/±360° 绕接（wrap）跳变，单位为度。

    四元数 -> 欧拉角 的转换把角度限制在主值区间内，当某个试次的姿态
    恰好跨过区间边界时，整段曲线会发生整体符号翻转或阶跃。这必须在
    生成 .mot 时修掉，否则下游 ID 的躯干/髋/膝力矩全部错误。

    np.unwrap 不接受 NaN，因此先线性填补、展开、再把 NaN 放回原位。
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return v
    finite = np.isfinite(v)
    if not finite.any():
        return v
    if finite.all():
        return np.degrees(np.unwrap(np.radians(v)))

    idx = np.arange(v.size)
    filled = v.copy()
    filled[~finite] = np.interp(idx[~finite], idx[finite], v[finite])
    out = np.degrees(np.unwrap(np.radians(filled)))
    out[~finite] = np.nan
    return out


def detect_frozen_intervals(time, matrix, min_duration=0.20, tol=1e-6,
                            ratio=0.05, window=0.15):
    """
    检测“丢帧后被保持或被线性插值填补”的时间区间。

    判据：二阶差分 ≈ 0。
      - 保持段（数值不变）：一阶、二阶差分都为 0；
      - 线性插值段（图上呈一条直线）：一阶差分恒定，二阶差分为 0；
      - 真实运动的二阶差分远高于 0。
    但 Xsens 自己对丢帧的填补并不是精确线性（图上看起来是直线，
    数值上却带微小波动），因此除了精确线性判据外，还加一个相对判据：
    滑窗 |d²| 低于全文件中位数的 ratio 倍。相对判据与采样率无关（240 Hz 下
    真实 d² 比 60 Hz 小约 16 倍，绝对阈值会失效）。
    只有当所有被监视列同时线性时才判为丢帧。

    Parameters
    ----------
    time         : (n,)   时间序列，秒
    matrix       : (n, k)  被监视的角度列
    min_duration : float  区间最短时长（秒），短于此值不上报
    tol          : float  精确线性判据的相对容差，阈值 = tol * 该列幅值
    ratio        : float  近似线性判据：滑窗 |d²| / 全文件 |d²| 中位数 的上限
    window       : float  近似线性判据的滑窗长度（秒）

    Returns
    -------
    list[(t_start, t_end)]
    """
    time = np.asarray(time, dtype=float)
    m = np.atleast_2d(np.asarray(matrix, dtype=float))
    if m.shape[0] != time.size and m.shape[1] == time.size:
        m = m.T
    n = time.size
    if n < 5 or m.shape[0] != n:
        return []

    d2 = np.abs(np.diff(m, n=2, axis=0))              # (n-2, k)
    scale = np.nanmax(np.abs(m), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    # 判据 a：精确线性（保持段或线性插值段，d² 几乎精确为 0）
    flat_exact = np.all(d2 <= tol * scale, axis=1)     # (n-2,)

    # 判据 b：近似线性。与该文件自身的典型 d² 相比，因此自适应采样率。
    dt_med = float(np.median(np.diff(time))) if n > 1 else 0.0
    w = max(3, int(round(window / dt_med))) if dt_med > 0 else 3
    ref = np.nanmedian(d2, axis=0)
    ref = np.where(np.isfinite(ref) & (ref > 0), ref, np.inf)
    kernel = np.ones(w) / w
    smooth = np.vstack([
        np.convolve(np.nan_to_num(d2[:, j], nan=np.inf), kernel, mode='same')
        for j in range(d2.shape[1])
    ]).T
    flat_approx = np.all(smooth <= ratio * ref, axis=1)

    frozen = np.zeros(n, dtype=bool)
    frozen[1:-1] = flat_exact | flat_approx
    frozen |= ~np.all(np.isfinite(m), axis=1)         # NaN 同样不可用

    intervals = []
    i = 0
    while i < n:
        if not frozen[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and frozen[j + 1]:
            j += 1
        if time[j] - time[i] >= min_duration:
            intervals.append((float(time[i]), float(time[j])))
        i = j + 1
    return intervals


def dropout_sidecar_path(mot_path):
    """与 .mot 同名的丢帧区间清单路径，例如 xxx_opensim.dropouts.csv。"""
    return str(Path(mot_path).with_suffix('.dropouts.csv'))


def save_dropout_intervals(mot_path, intervals):
    """把丢帧区间写到 .mot 旁边，供下游剔除对应时间段（不删除任何数据）。"""
    path = dropout_sidecar_path(mot_path)
    with open(path, 'w') as f:
        f.write('start,end\n')
        for t0, t1 in intervals:
            f.write(f'{t0:.6f},{t1:.6f}\n')
    return path


def load_dropout_intervals(mot_path, mot_df=None,
                           columns=DROPOUT_MONITOR_COLUMNS,
                           min_duration=0.20, tol=1e-6):
    """
    取某个 .mot 的丢帧区间。

    优先读取 Step1 生成的 sidecar；若不存在（例如 .mot 是旧版本生成的），
    就直接从 mot_df 现算，因此无需为了做诊断而重跑 Step1。
    """
    path = dropout_sidecar_path(mot_path)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) == 0:
            return []
        return [(float(a), float(b)) for a, b in zip(df['start'], df['end'])]

    if mot_df is None or 'time' not in getattr(mot_df, 'columns', []):
        return []
    cols = [c for c in columns if c in mot_df.columns]
    if not cols:
        return []
    return detect_frozen_intervals(mot_df['time'].values,
                                   mot_df[cols].values,
                                   min_duration=min_duration, tol=tol)


def in_intervals(times, intervals, margin=0.0):
    """布尔数组：times 是否落在任一区间内。margin 为额外的安全边界（秒）。"""
    t = np.asarray(times, dtype=float)
    mask = np.zeros(t.shape, dtype=bool)
    for t0, t1 in intervals or []:
        mask |= (t >= t0 - margin) & (t <= t1 + margin)
    return mask


# ============================================================
#  骶盆姿态：heading（服位朝向）处理
# ============================================================

# Xsens 全局坐标系: X 前, Y 左, Z 上（右手系）。
# 关键事实: Segment Orientation 给的是每个 segment 相对该【全局】坐标系
# 的姿态，而不是相对受试自身的解剖平面。因此绕全局 Y 的 pitch
# 只在受试正对全局 X 方向时才等于解剖学上的骶盆前后倾。
# 若受试的服位朝向（heading = 绕全局 Z 的 yaw）为 psi，则近似有
#     tilt_表观 =  tilt_真 * cos(psi) + list_真 * sin(psi)
#     list_表观 = -tilt_真 * sin(psi) + list_真 * cos(psi)
# 于是: psi ≈ 90° 时前后倾几乎全部泄露成侧倾；psi ≈ 180° 时前后倾【整段变号】。
# 这正是 106 kg 那一次 pelvis_tilt 整段翻转、而帧内没有任何 ±180° 跳变
# （所以 unwrap 完全无法察觉）的原因。
# heading 本身又是整个 Xsens 数据里最不可靠的量（依赖磁力计，可发生
# 漂移与 HDR 重置），因此正确做法是在提取欧拉角之前就把它转掉。
#
# 注意: Joint Angles 表里的髀/膝/踝等关节角是相邻 segment 之间的【相对】
# 角度，heading 在其中自动消去，因此不受该问题影响。只有骶盆是从
# 绝对姿态推导的，所以也只有骶盆需要修。
PELVIS_HEADING_MODES = ('heading_free', 'reference', 'absolute')


def segment_rotation_from_quat(df_quat, n_samples, segment='Pelvis'):
    """从 Segment Orientation - Quat 取某个 segment 的旋转。

    已用实测数据核实: Pelvis q0 = 0.998891 是标量项 w，q1..q3 是矢量项，
    而 scipy 的 from_quat 需要 (x, y, z, w)，所以顺序是 [q1, q2, q3, q0]。
    """
    pq = np.column_stack([
        df_quat[f'{segment} q1'].values[:n_samples],
        df_quat[f'{segment} q2'].values[:n_samples],
        df_quat[f'{segment} q3'].values[:n_samples],
        df_quat[f'{segment} q0'].values[:n_samples],
    ]).astype(float)
    return R.from_quat(pq)


def heading_degrees(rot):
    """逐帧 heading（度）= 体轴 x 在全局水平面内的方位角。"""
    m = rot.as_matrix()
    return np.degrees(np.arctan2(m[:, 1, 0], m[:, 0, 0]))


def _circular_mean_deg(values):
    """角度的圆周均值（直接取算术均值在 ±180° 附近会得到错的结果）。"""
    a = np.radians(np.asarray(values, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))


def find_static_window(time, vertical, window=1.0, speed_tol=0.05):
    """找试次开头的静立窗口（返回布尔掩码）。

    判据: 骶盆垂直速度的滑窗均值低于 speed_tol。找不到则退化为开头一段。
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(vertical, dtype=float)
    n = t.size
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    w = max(3, int(round(window / dt))) if dt > 0 else min(n, 10)
    w = min(w, max(3, n))
    mask = np.zeros(n, dtype=bool)
    if n < w + 2:
        mask[:w] = True
        return mask
    speed = np.abs(np.gradient(np.nan_to_num(v), t))
    smooth = np.convolve(speed, np.ones(w) / w, mode='valid')
    ok = np.flatnonzero(smooth < speed_tol)
    start = int(ok[0]) if ok.size else 0
    mask[start:start + w] = True
    return mask


def compute_pelvis_orientation(rot, time=None, vertical=None,
                               mode='heading_free', ref_window=1.0,
                               verbose=True):
    """
    从骶盆绝对姿态算出 OpenSim 的 pelvis_tilt / list / rotation。

    mode
    ----
    'heading_free' (默认、推荐)
        逐帧转掉瞬时 heading：tilt/list 始终相对受试当前的朝向，
        即真正的矢状面/额状面。对服位朝向、heading 漂移、受试中途
        转身全部免疫。副作用是 pelvis_rotation 恒为 0，但 yaw 不改变
        重力力臂，对 ID 力矩零贡献，所以不丢信息。
    'reference'
        只转掉静立参考窗口的常数 heading，保留试次内的真实转身。
    'absolute'
        旧行为（不推荐），只用于复现之前的结果。

    Returns
    -------
    (tilt, list_, rotation, psi_used, info)
    """
    if mode not in PELVIS_HEADING_MODES:
        raise ValueError(f'mode 必须是 {PELVIS_HEADING_MODES} 之一，得到 {mode!r}')

    psi = heading_degrees(rot)
    psi_span = float(np.nanmax(psi) - np.nanmin(psi)) if psi.size else 0.0

    if mode == 'absolute':
        psi_used = np.zeros_like(psi)
    elif mode == 'reference':
        if time is not None and vertical is not None:
            mask = find_static_window(time, vertical, window=ref_window)
        else:
            mask = np.zeros_like(psi, dtype=bool)
            mask[:max(3, psi.size // 20)] = True
        psi_used = np.full_like(psi, _circular_mean_deg(psi[mask]))
    else:
        psi_used = psi

    # Rz(-psi) · R(t)：先把全局坐标系绕竖直轴转到与受试朝向对齐。
    # 单轴序列时 scipy 要求 angles 的最后一维等于轴数（=1），
    # 直接传 (n,) 会被当成“一个旋转、n 个轴角”而报错，所以要 reshape 成 (n, 1)。
    psi_col = np.asarray(psi_used, dtype=float).reshape(-1, 1)
    rot_corrected = R.from_euler('z', -psi_col, degrees=True) * rot
    euler = rot_corrected.as_euler('zxy', degrees=True)
    rotation, list_, tilt = euler[:, 0], euler[:, 1], euler[:, 2]

    spans = {'tilt': float(np.nanmax(tilt) - np.nanmin(tilt)),
             'list': float(np.nanmax(list_) - np.nanmin(list_)),
             'rotation': float(np.nanmax(rotation) - np.nanmin(rotation))}
    info = {'mode': mode, 'heading_span_deg': psi_span,
            'heading_ref_deg': float(psi_used[0]) if psi_used.size else 0.0,
            'spans_deg': spans}

    if verbose:
        print(f'  [pelvis] heading 模式={mode}  原始 heading 跨度={psi_span:.1f}°'
              f'  参考 heading={info["heading_ref_deg"]:+.1f}°')
        print(f'  [pelvis] 变化幅度: tilt={spans["tilt"]:.1f}°  '
              f'list={spans["list"]:.1f}°  rotation={spans["rotation"]:.1f}°')
        # 可否定的自检: 深蹲是矢状面动作，tilt 必须是变化最大的那一个。
        if spans['tilt'] < max(spans['list'], spans['rotation']):
            beauty_print('tilt 不是变化最大的分量，轴映射可能仍有问题；'
                         '深蹲是矢状面动作，tilt 应远大于 list / rotation',
                         type="warning")
        if mode != 'heading_free' and psi_span > 30.0:
            beauty_print(f'heading 在该试次内变化 {psi_span:.0f}°，'
                         '常数扰动不足以修正，建议改用 heading_free',
                         type="warning")

    return tilt, list_, rotation, psi_used, info


def identify_euler_sequence(rot, df_euler, n_samples=None, segment='Pelvis',
                            candidates=('XYZ', 'ZXY', 'ZYX', 'YXZ',
                                        'xyz', 'zxy', 'zyx', 'yxz'),
                            tol=1.0, verbose=True):
    """
    用 Xsens 自带的 Segment Orientation - Euler 表反推它用的欧拉序列。

    这是一个自检，而不是假定: 如果四元数列序或轴定义理解错了，
    没有任何候选序列能与该表对上，RMS 会明显偏大。
    """
    cols = [f'{segment} {a}' for a in ('x', 'y', 'z')]
    if any(c not in df_euler.columns for c in cols):
        return None
    target = df_euler[cols].values[:n_samples].astype(float)
    scored = []
    for seq in candidates:
        ang = rot.as_euler(seq, degrees=True)
        order = [seq.lower().index(a) for a in 'xyz']
        diff = (ang[:, order] - target + 180.0) % 360.0 - 180.0
        scored.append((float(np.sqrt(np.nanmean(diff ** 2))), seq))
    scored.sort()
    rms, best = scored[0]
    if verbose:
        print(f'  [euler-check] 与 Xsens Euler 表最匹配的序列: {best}  '
              f'RMS={rms:.3f}°')
        if rms > tol:
            beauty_print(
                f'没有候选欧拉序列能与 Xsens Euler 表对上（最优 {best} '
                f'RMS={rms:.3f}° > {tol}°），四元数列序或轴定义可能有误；'
                f'次优: {scored[1][1]} RMS={scored[1][0]:.3f}°',
                type="warning")
    return best, rms


# ============================================================
#  Xsens Excel -> .mot 转换核心函数
# ============================================================

def read_xsens_excel_for_opensim(excel_path, output_mot_path=None,
                                 pelvis_heading_mode='heading_free',
                                 verify_euler_sheet=True):
    """
    从 Xsens 导出的 Excel 文件读取数据，生成 OpenSim 可用的 .mot 文件。
    arm_flex / arm_add 左右共四个关节角仍需调整。

    Parameters
    ----------
    excel_path      : str -- Xsens 导出的 .xlsx 文件路径
    output_mot_path : str, optional -- 输出路径
    pelvis_heading_mode : str -- 骶盆服位朝向的处理方式，见
                      compute_pelvis_orientation。默认 'heading_free'。
    verify_euler_sheet : bool -- 是否用 Segment Orientation - Euler 表
                      反推欧拉序列作为自检

    Returns
    -------
    (data_with_time : np.ndarray, joint_names_order : list)
    """
    # 1. 读取帧率
    df_info = pd.read_excel(excel_path, sheet_name='General Information', header=None)
    frame_rate = 60
    for _, row in df_info.iterrows():
        if row[0] == 'Frame Rate':
            frame_rate = float(row[1])
            break
    print(f'采样率: {frame_rate} Hz')

    # 2. 读取帧索引并计算时间
    df_pos = pd.read_excel(excel_path, sheet_name='Segment Position')
    frame_indices = df_pos['Frame'].values
    n_samples = len(frame_indices)
    time = frame_indices / frame_rate
    print(f'帧数: {n_samples},  时间范围: {time[0]:.3f} - {time[-1]:.3f} 秒')

    # 3. 骨盆角度（四元数 -> ZXY 欧拉角）
    df_quat = pd.read_excel(excel_path, sheet_name='Segment Orientation - Quat')
    rot_pelvis = segment_rotation_from_quat(df_quat, n_samples, 'Pelvis')

    # 自检：用 Xsens 自带的 Euler 表反推序列，验证四元数列序理解无误
    if verify_euler_sheet:
        try:
            df_euler = pd.read_excel(excel_path,
                                     sheet_name='Segment Orientation - Euler')
            identify_euler_sequence(rot_pelvis, df_euler, n_samples=n_samples)
        except Exception as exc:
            print(f'  [euler-check] 跳过（{exc}）')

    pelvis_vertical = df_pos['Pelvis z'].values[:n_samples].astype(float)
    pelvis_tilt, pelvis_list, pelvis_rotation, psi_used, _ = \
        compute_pelvis_orientation(rot_pelvis, time=time,
                                   vertical=pelvis_vertical,
                                   mode=pelvis_heading_mode)

    # 4. 骨盆位置 (Xsens x,y,z -> OpenSim x,y,z)
    #    水平分量必须与姿态处于同一参考系：同样绕竖直轴转 -psi。
    #    先减去第一帧，避免把全局原点也一起转（绝对平移对 ID 无影响）。
    px = df_pos['Pelvis x'].values[:n_samples].astype(float)
    py = df_pos['Pelvis y'].values[:n_samples].astype(float)
    c = np.cos(np.radians(-psi_used))
    s = np.sin(np.radians(-psi_used))
    dx, dy = px - px[0], py - py[0]
    pelvis_tx = c * dx - s * dy
    pelvis_ty = pelvis_vertical                        # Xsens z(上) -> OpenSim y(上)
    pelvis_tz = s * dx + c * dy                        # Xsens y(左) -> OpenSim z

    # 5. 关节角度
    df_j  = pd.read_excel(excel_path, sheet_name='Joint Angles ZXY')
    jmap  = {
        'hip_flexion_r':    df_j['Right Hip Flexion/Extension'],
        'hip_adduction_r':  df_j['Right Hip Abduction/Adduction'],
        'hip_rotation_r':   df_j['Right Hip Internal/External Rotation'],
        'knee_angle_r':     df_j['Right Knee Flexion/Extension'],
        'ankle_angle_r':    df_j['Right Ankle Dorsiflexion/Plantarflexion'],
        'subtalar_angle_r': df_j['Right Ankle Internal/External Rotation'],
        'mtp_angle_r':      df_j['Right Ball Foot Flexion/Extension'],
        'hip_flexion_l':    df_j['Left Hip Flexion/Extension'],
        'hip_adduction_l':  df_j['Left Hip Abduction/Adduction'],
        'hip_rotation_l':   df_j['Left Hip Internal/External Rotation'],
        'knee_angle_l':     df_j['Left Knee Flexion/Extension'],
        'ankle_angle_l':    df_j['Left Ankle Dorsiflexion/Plantarflexion'],
        'subtalar_angle_l': df_j['Left Ankle Internal/External Rotation'],
        'mtp_angle_l':      df_j['Left Ball Foot Flexion/Extension'],
        'lumbar_extension': df_j['L5S1 Flexion/Extension'],
        'lumbar_bending':   df_j['L5S1 Lateral Bending'],
        'lumbar_rotation':  df_j['L5S1 Axial Bending'],
        'arm_flex_r':       df_j['Right Shoulder Flexion/Extension'],   ## 需要调整
        'arm_add_r':        df_j['Right Shoulder Abduction/Adduction'],  ## 需要调整
        'arm_rot_r':        df_j['Right Shoulder Internal/External Rotation'],
        'elbow_flex_r':     df_j['Right Elbow Flexion/Extension'],
        'pro_sup_r':        df_j['Right Elbow Pronation/Supination'],
        'wrist_flex_r':     df_j['Right Wrist Flexion/Extension'],
        'wrist_dev_r':      df_j['Right Wrist Ulnar Deviation/Radial Deviation'],
        'arm_flex_l':       df_j['Left Shoulder Flexion/Extension'],    ## 需要调整
        'arm_add_l':        df_j['Left Shoulder Abduction/Adduction'],   ## 需要调整
        'arm_rot_l':        df_j['Left Shoulder Internal/External Rotation'],
        'elbow_flex_l':     df_j['Left Elbow Flexion/Extension'],
        'pro_sup_l':        df_j['Left Elbow Pronation/Supination'],
        'wrist_flex_l':     df_j['Left Wrist Flexion/Extension'],
        'wrist_dev_l':      df_j['Left Wrist Ulnar Deviation/Radial Deviation'],
        'SC_y':   df_j['Right T4 Shoulder Flexion/Extension'],
        'SC_x':   df_j['Right T4 Shoulder Abduction/Adduction'],
        'SC_z':   df_j['Right T4 Shoulder Internal/External Rotation'],
        'SC_y_l': df_j['Left T4 Shoulder Flexion/Extension'],
        'SC_x_l': df_j['Left T4 Shoulder Abduction/Adduction'],
        'SC_z_l': df_j['Left T4 Shoulder Internal/External Rotation'],
    }

    joint_names_order = [
        'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
        'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
        'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
        'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
        'arm_flex_r', 'arm_add_r', 'arm_rot_r',
        'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r',
        'arm_flex_l', 'arm_add_l', 'arm_rot_l',
        'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l',
        'SC_y', 'SC_x', 'SC_z', 'SC_y_l', 'SC_x_l', 'SC_z_l',
    ]

    # 6. 构建数据矩阵
    pelvis_vals = {
        'pelvis_tilt': pelvis_tilt, 'pelvis_list': pelvis_list,
        'pelvis_rotation': pelvis_rotation,
        'pelvis_tx': pelvis_tx, 'pelvis_ty': pelvis_ty, 'pelvis_tz': pelvis_tz,
    }
    angles_matrix = np.zeros((n_samples, len(joint_names_order)))
    for i, jname in enumerate(joint_names_order):
        if jname in pelvis_vals:
            angles_matrix[:, i] = pelvis_vals[jname]
        elif jname in jmap:
            angles_matrix[:, i] = jmap[jname].values[:n_samples]

    # 7. 消除欧拉角绕接跳变（在符号调整之前做）
    angle_cols = [j for j in joint_names_order
                  if j not in ('pelvis_tx', 'pelvis_ty', 'pelvis_tz')]
    unwrap_report = []
    for jname in angle_cols:
        ci = joint_names_order.index(jname)
        before = angles_matrix[:, ci]
        after = unwrap_degrees(before)
        shift = float(np.nanmax(np.abs(after - before))) if after.size else 0.0
        if shift > 1.0:
            unwrap_report.append((jname, shift))
        angles_matrix[:, ci] = after
    if unwrap_report:
        print('  [unwrap] 已消除绕接跳变的列:')
        for jname, shift in sorted(unwrap_report, key=lambda x: -x[1]):
            print(f'    {jname:<20} 最大修正 {shift:8.2f}°')
    else:
        print('  [unwrap] 未检测到绕接跳变')

    # 8. 符号调整
    sign_flip = {
        'pelvis_tilt': -1, 'pelvis_tz': -1,
        'hip_adduction_r': -1, 'hip_adduction_l': -1,
        'lumbar_extension': -1, 'arm_flex_r': -1, 'arm_flex_l': -1,
    }
    for jname, sign in sign_flip.items():
        if jname in joint_names_order:
            angles_matrix[:, joint_names_order.index(jname)] *= sign

    data_with_time = np.column_stack([time, angles_matrix])

    # 9. 丢帧检测：只标记，不删除数据
    monitor_idx = [joint_names_order.index(c) for c in DROPOUT_MONITOR_COLUMNS
                   if c in joint_names_order]
    dropouts = []
    if monitor_idx:
        dropouts = detect_frozen_intervals(time, angles_matrix[:, monitor_idx])
    if dropouts:
        total = sum(t1 - t0 for t0, t1 in dropouts)
        print(f'  [dropout] 检测到 {len(dropouts)} 段丢帧/插值区间，'
              f'合计 {total:.2f} s（占 {total / (time[-1] - time[0]) * 100:.1f}%）:')
        for t0, t1 in dropouts:
            print(f'    {t0:8.3f} - {t1:8.3f} s   ({t1 - t0:.2f} s)')
        beauty_print(f'检测到 {len(dropouts)} 段丢帧/插补区间（合计 {total:.2f} s），'
                     '这些区间的角速度/角加速度不可信；'
                     '下游会按时间段剔除，整组数据仍然保留。',
                     type="warning")
    else:
        print('  [dropout] 未检测到丢帧区间')

    # 10. 保存 .mot
    if output_mot_path is None:
        output_mot_path = Path(excel_path).stem + '_opensim.mot'
    with open(output_mot_path, 'w') as f:
        f.write('first trial\n')
        f.write(f'nRows={n_samples}\n')
        f.write(f'nColumns={len(joint_names_order) + 1}\n\n')
        f.write('# SIMM Motion File Header:\n')
        f.write(f'name {Path(excel_path).stem}\n')
        f.write(f'datacolumns {len(joint_names_order) + 1}\n')
        f.write(f'datarows {n_samples}\n')
        f.write('otherdata 1\n')
        # 显式声明角度单位。缺这一行时 OpenSim 只是按旧格式默认猜成度，
        # 一旦默认行为变了，所有力矩会错 57.3 倍而不报错。
        f.write('inDegrees=yes\n')
        data_min = np.min(data_with_time[:, 1:])
        data_max = np.max(data_with_time[:, 1:])
        f.write(f'range {data_min:.4f} {data_max:.4f}\n')
        f.write('endheader\n')
        f.write('time\t' + '\t'.join(joint_names_order) + '\n')
        for row in data_with_time:
            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')
    print(f'已保存: {output_mot_path}  {data_with_time.shape}')

    # 11. 丢帧清单写到 .mot 旁边，供 ID / 统计层剔除对应时间段
    sidecar = save_dropout_intervals(output_mot_path, dropouts)
    print(f'已保存丢帧清单: {sidecar}  ({len(dropouts)} 段)')

    return data_with_time, joint_names_order


# ============================================================
#  流水线接口
# ============================================================

def run_mot_conversion(config, base_dir, verbose=True):
    """
    将 modeling_file.data 中所有 xsens_file 转换为 OpenSim .mot 文件。

    Returns
    -------
    dict  {load_key: mot_file_path}
    """
    def log(msg):
        if verbose:
            print(msg)

    experiment_label = config['experiment_label']
    folder   = config['folder']
    modeling = config['modeling_file']
    xsens_dir = os.path.join(folder, modeling.get('xsens_folder', 'xsens'))

    output_dir = os.path.join(base_dir, 'result', experiment_label, 'opensim', 'mot')
    os.makedirs(output_dir, exist_ok=True)
    log(f'[mot] 输出目录: {output_dir}')

    mot_files = {}
    for load_key, file_info in modeling['data'].items():
        xsens_file = file_info.get('xsens_file')
        if xsens_file is None:
            log(f'  [{load_key}] 无 xsens_file，跳过')
            continue
        log(f'\n  [{load_key}] {xsens_file}')
        xsens_path = os.path.join(xsens_dir, xsens_file)
        mot_name   = Path(xsens_file).stem + '_opensim.mot'
        mot_path   = os.path.join(output_dir, mot_name)
        try:
            read_xsens_excel_for_opensim(xsens_path, mot_path)
            mot_files[load_key] = mot_path
            log(f'    已保存: {mot_path}')
        except Exception as e:
            log(f'    转换失败: {e}')

    log(f'\n[mot] 共转换 {len(mot_files)} 个文件')
    return mot_files


def run_step1_mot_conversion(config, base_dir, verbose=True):
    """步骤 1: Xsens Excel -> OpenSim .mot，输出到 result/{label}/opensim/mot/"""
    return run_mot_conversion(config, base_dir, verbose=verbose)