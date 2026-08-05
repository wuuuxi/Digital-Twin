"""
external_forces.py

生成 OpenSim ExternalLoads 文件，包含：
  1. 杆件对人体的作用力（由机器人传感器数据计算）
  2. 足底地面反力 GRF（由鞋垫传感器数据读取）

生成文件保存在共享目录：
  result/{experiment_label}/opensim/external_forces/{load_key}/
    bar_force_{load_key}.sto   -- 时序外力数据
    bar_loads_{load_key}.xml   -- ExternalLoads XML

主要接口：
  get_ext_forces_dir(config, base_dir, load_key) -> str
  generate_external_loads(config, base_dir, load_key, mot_path,
                          Mb=20.0, verbose=True) -> xml_path or None
"""
import os
import numpy as np
import opensim as osim

from digitaltwin.data.insole_processor import InsoleProcessor
from digitaltwin.utils.logger import beauty_print


def get_ext_forces_dir(config, base_dir, load_key):
    """
    返回外力文件的共享目录路径。
    muscle_analysis 和 inverse_dynamics 共用，避免重复生成。
    """
    experiment_label = config['experiment_label']
    return os.path.join(
        base_dir, 'result', experiment_label,
        'opensim', 'external_forces', str(load_key)
    )


def generate_external_loads(config, base_dir, load_key, mot_path,
                             Mb=20.0, verbose=True,
                             use_insole_info_timestamp=True):
    """
    从机器人数据和鞋垫数据生成 OpenSim ExternalLoads 文件。

    力学模型：
      杆件力（施加到 torso，-Y 向下）：
        F_bar = force_l + force_r + Mb*g + Mb * avg(acc_l, acc_r)

      足底 GRF（施加到左右 calcn，+Y 向上）：
        大小从 insole_file_l / insole_file_r 读取；
        作用点默认从 insole_map_l / insole_map_r 的逐帧压心算出，
        得不到时才退回 insole_contact_point 恒定值

    opensim_settings 可选字段：
      bar_mass              (float, kg,  默认 20.0)
      bar_contact_body      (str,        默认 'torso')
      bar_contact_point     ([x,y,z], m, 默认 [-0.07, 0.30, 0.0])
      insole_contact_body_l (str,        默认 'calcn_l')
      insole_contact_body_r (str,        默认 'calcn_r')
      insole_contact_point  ([x,y,z], m, 默认 [0.0, 0.0, 0.0])

    Parameters
    ----------
    config   : dict
    base_dir : str
    load_key : str
    mot_path : str   -- 用于获取时间轴
    Mb       : float -- 杆质量默认值（会被 opensim_settings.bar_mass 覆盖）
    verbose  : bool
    use_insole_info_timestamp : bool, default True
        是否使用鞋垫文件同目录 info.csv 中的 measurement_date，
        结合 robot_file 第一帧时间修正鞋垫时间轴。默认开启；
        如需退回鞋垫文件原始相对时间，可置为 False。

    Returns
    -------
    str or None  -- ExternalLoads XML 路径（失败返回 None）
    """
    def log(msg):
        if verbose:
            print(msg)

    ext_dir = get_ext_forces_dir(config, base_dir, load_key)
    os.makedirs(ext_dir, exist_ok=True)

    osim_cfg  = config.get('opensim_settings', {})
    modeling  = config['modeling_file']
    folder    = config['folder']
    file_info = modeling['data'].get(str(load_key))

    if file_info is None:
        log(f'  [EXT] load_key={load_key} 不在 modeling_file.data 中')
        return None

    # ---- 杆件力配置 ----
    bar_body  = osim_cfg.get('bar_contact_body',  'torso')
    bar_point = osim_cfg.get('bar_contact_point', [-0.07, 0.30, 0.0])
    Mb        = float(osim_cfg.get('bar_mass', Mb))
    g         = 9.81

    # ---- 读取机器人数据 ----
    robot_file = file_info.get('robot_file')
    if not robot_file:
        log('  [EXT] 无 robot_file，跳过外力生成')
        return None

    from digitaltwin.data.robot_processor import RobotProcessor
    log(f'  [EXT] 读取机器人数据: {robot_file}')
    robot_df = RobotProcessor.process(
        robot_file=robot_file,
        load_weight=str(load_key),
        robot_folder=folder,
        folder=folder,
    )
    if robot_df is None:
        log('  [EXT] 机器人数据加载失败')
        return None

    robot_time = robot_df['time'].values.astype(float)
    force_l = robot_df['force_l'].values.astype(float) if 'force_l' in robot_df else np.zeros(len(robot_df))
    force_r = robot_df['force_r'].values.astype(float) if 'force_r' in robot_df else np.zeros(len(robot_df))
    acc_l   = robot_df['acc_l'].values.astype(float)   if 'acc_l'   in robot_df else np.zeros(len(robot_df))
    acc_r   = robot_df['acc_r'].values.astype(float)   if 'acc_r'   in robot_df else np.zeros(len(robot_df))

    avg_acc = (acc_l + acc_r) / 2.0
    F_mag   = force_l + force_r + Mb * g + Mb * avg_acc
    F_bar_y = -F_mag   # OpenSim y 轴向上，杆力向下为负

    log(f'  [EXT] 杆质量={Mb:.1f}kg, g={g}m/s²')
    log(f'  [EXT] 杆力范围: [{F_mag.min():.1f}, {F_mag.max():.1f}] N')

    # ---- 对齐杆力到 .mot 时间轴 ----
    mot_table = osim.TimeSeriesTable(mot_path)
    mot_times = np.array(mot_table.getIndependentColumn())
    F_bar_resampled = np.interp(mot_times, robot_time, F_bar_y,
                                left=F_bar_y[0], right=F_bar_y[-1])

    # ---- 读取鞋垫 GRF ----
    insole_body_l = osim_cfg.get('insole_contact_body_l', 'calcn_l')
    insole_body_r = osim_cfg.get('insole_contact_body_r', 'calcn_r')
    insole_pt     = osim_cfg.get('insole_contact_point',  [0.0, 0.0, 0.0])
    ipx, ipy, ipz = float(insole_pt[0]), float(insole_pt[1]), float(insole_pt[2])

    # ---- 逐帧 COP 配置 ----
    # insole_use_frame_cop : 是否用鞋垫压力图算出的逐帧压心作为作用点
    # insole_heel_offset_x : 鞋垫足跟端在 calcn 局部坐标系里的 x 偏移 (m)
    # insole_column_frame  : 鞋垫列坐标是 world 还是每只脚自己的 anatomical
    use_frame_cop = bool(osim_cfg.get('insole_use_frame_cop', True))
    heel_x        = float(osim_cfg.get('insole_heel_offset_x', 0.0))
    col_frame     = str(osim_cfg.get('insole_column_frame', 'world')).lower()
    cop_x_min     = float(osim_cfg.get('insole_cop_x_min', 0.02))
    cop_x_max     = float(osim_cfg.get('insole_cop_x_max', 0.25))

    insole_folder = modeling.get('insole_folder', 'Sorted')
    insole_base   = os.path.join(folder, insole_folder)

    grf_l_resampled = np.zeros(len(mot_times))
    grf_r_resampled = np.zeros(len(mot_times))
    has_insole = False

    for side, key in [('L', 'insole_file_l'), ('R', 'insole_file_r')]:
        insole_rel = file_info.get(key)
        if insole_rel:
            t_s, f_s = InsoleProcessor.load(
                os.path.join(insole_base, insole_rel),
                verbose=verbose,
                use_info_timestamp=use_insole_info_timestamp,
                robot_file=robot_file,
                robot_folder=folder,
                folder=folder)
            if t_s is not None:
                resampled = InsoleProcessor.resample(t_s, f_s, mot_times)
                log(f'  [EXT] {side} 足底 GRF 范围: [{f_s.min():.1f}, {f_s.max():.1f}] N')
                has_insole = True
                if side == 'L':
                    grf_l_resampled = resampled
                else:
                    grf_r_resampled = resampled

    if not has_insole:
        log('  [EXT] 未找到鞋垫文件，仅包含杆件力')

    # ---- 逐帧 COP：把恒定作用点换成实测压心 ----
    #
    # 恒定作用点等于假设力臂全程不变。敏感性测试给出膝力矩对 COP 前后
    # 位置的斑度是 -491.6 N·m/m，也就是 1 cm 的力臂误差 = 4.9 N·m，
    # 正好是膝力矩单调性违反量（1.5-10.9 N·m）的量级。所以这一步既是修复，
    # 也是对“COP 假说”的可证伪检验：若真实逐帧 COP 的波动不足 1 cm，
    # 那么 COP 就不可能是单调性问题的原因。
    cop_px = {'l': np.full(len(mot_times), ipx),
              'r': np.full(len(mot_times), ipx)}
    cop_pz = {'l': np.full(len(mot_times), ipz),
              'r': np.full(len(mot_times), ipz)}
    cop_src = {'l': 'constant', 'r': 'constant'}

    if use_frame_cop and has_insole:
        for side, map_key in (('l', 'insole_map_l'), ('r', 'insole_map_r')):
            rel = file_info.get(map_key)
            if not rel:
                beauty_print(
                    '  [EXT] {} 侧未配置 {}，该侧仍用恒定作用点，'
                    '膝力矩会继续带着力臂误差。'.format(
                        side.upper(), map_key),
                    type='warning')
                continue

            res = InsoleProcessor.load_pressure_map(
                os.path.join(insole_base, rel),
                verbose=verbose,
                use_info_timestamp=use_insole_info_timestamp,
                robot_file=robot_file,
                robot_folder=folder,
                folder=folder,
                return_matrix=False)   # 只要 COP，不留矩阵，省内存
            if res is None:
                beauty_print(
                    '  [EXT] {} 侧压力图读取失败: {}，退回恒定作用点。'.format(
                        side.upper(), rel),
                    type='warning')
                continue

            t_map = np.asarray(res['time'], dtype=float)
            # 悬空帧的 COP 是 nan，必须用 nan-safe 插值：
            # np.interp 不认识 nan，一个 nan 会把两侧邻域一起污染。
            ant = InsoleProcessor.resample_nan_safe(
                t_map, res['cop_ant'], mot_times, max_gap_s=0.2)
            lat = InsoleProcessor.resample_nan_safe(
                t_map, res['cop_lat'], mot_times, max_gap_s=0.2)
            if ant is None or lat is None:
                beauty_print(
                    '  [EXT] {} 侧 COP 全为无效值（可能整段总力低于阈值），'
                    '退回恒定作用点。'.format(side.upper()),
                    type='warning')
                continue

            width_m = float(res['meta']['width_cm']) / 100.0
            # 前后：cop_ant 是距足跟端的距离，calcn 局部 x 轴向前
            px_side = ant + heel_x
            # 内外：先换成相对鞋垫中线的偏移
            dz_side = lat - width_m / 2.0
            if col_frame == 'anatomical' and side == 'l':
                # 按每只脚自己的解剖方向存储时，左脚的列需要翻转才能
                # 对齐到模型的左右轴。只影响额状面，不影响矢状面膝力矩。
                dz_side = -dz_side

            cov = float(np.mean(np.isfinite(px_side)))
            # 覆盖不到的帧（悬空/丢帧/鞋垫已停录）回填该侧均值。
            # 不能留 nan（OpenSim 会拒读），也不能填 0（那等于把作用点
            # 放到足跟原点）。反正这些帧的 GRF 本身也接近零，力矩贡献很小。
            fill_x = float(np.nanmean(px_side)) if np.any(np.isfinite(px_side)) else ipx
            fill_z = float(np.nanmean(dz_side)) if np.any(np.isfinite(dz_side)) else ipz
            px_side = np.where(np.isfinite(px_side), px_side, fill_x)
            dz_side = np.where(np.isfinite(dz_side), dz_side, fill_z)

            cop_px[side] = px_side
            cop_pz[side] = dz_side
            cop_src[side] = os.path.basename(rel)

            mean_x = float(np.mean(px_side))
            log('  [EXT] {} 侧逐帧 COP: x 均值 {:.3f} m '
                '(范围 {:.3f}~{:.3f}), z 均值 {:+.3f} m, '
                '有效覆盖 {:.0%}'.format(
                    side.upper(), mean_x, float(np.min(px_side)),
                    float(np.max(px_side)), float(np.mean(dz_side)), cov))

            # 自检：作用点落在 calcn 局部 x 的合理区间内。足长约 25-30 cm，
            # COP 应在足跟到跖球之间；跑出区间通常意味着 heel_offset 或
            # toe_first 设错，而不是受试者真的踩在那里。
            if not (cop_x_min <= mean_x <= cop_x_max):
                beauty_print(
                    '  [EXT] {} 侧 COP 的 x 均值 {:.3f} m 超出合理区间 '
                    '[{:.2f}, {:.2f}] m。请检查 toe_first 与 '
                    'insole_heel_offset_x，否则膝力矩力臂会整体偏。'.format(
                        side.upper(), mean_x, cop_x_min, cop_x_max),
                    type='warning')
            if cov < 0.5:
                beauty_print(
                    '  [EXT] {} 侧只有 {:.0%} 的帧有效 COP，其余帧用均值回填。'
                    '若这些帧落在深蹲窗口内，力臂会失真。'.format(
                        side.upper(), cov),
                    type='warning')
    elif not use_frame_cop:
        log('  [EXT] insole_use_frame_cop=False，作用点使用恒定值')

    # ---- 写 .sto 文件 ----
    # OpenSim 4.x 列名前缀规则：
    #   {prefix}_v{x/y/z}       -> 力分量
    #   {prefix}_p{x/y/z}       -> 作用点坐标
    #   {prefix}_torque_{x/y/z} -> 力矩分量
    px, py, pz = float(bar_point[0]), float(bar_point[1]), float(bar_point[2])
    cols = [
        'time',
        'bar_force_vx', 'bar_force_vy', 'bar_force_vz',
        'bar_force_px', 'bar_force_py', 'bar_force_pz',
        'bar_torque_x', 'bar_torque_y', 'bar_torque_z',
        'grf_l_vx', 'grf_l_vy', 'grf_l_vz',
        'grf_l_px', 'grf_l_py', 'grf_l_pz',
        'grf_l_torque_x', 'grf_l_torque_y', 'grf_l_torque_z',
        'grf_r_vx', 'grf_r_vy', 'grf_r_vz',
        'grf_r_px', 'grf_r_py', 'grf_r_pz',
        'grf_r_torque_x', 'grf_r_torque_y', 'grf_r_torque_z',
    ]

    sto_path = os.path.join(ext_dir, f'bar_force_{load_key}.sto')
    with open(sto_path, 'w') as fh:
        fh.write('external_forces\n')
        fh.write(f'nRows={len(mot_times)}\n')
        fh.write(f'nColumns={len(cols)}\n')
        fh.write('inDegrees=no\n')
        fh.write('endheader\n')
        fh.write('\t'.join(cols) + '\n')
        for i, t in enumerate(mot_times):
            row = [
                t,
                0.0, F_bar_resampled[i], 0.0,
                px, py, pz, 0.0, 0.0, 0.0,
                0.0, grf_l_resampled[i], 0.0,
                cop_px['l'][i], ipy, cop_pz['l'][i], 0.0, 0.0, 0.0,
                0.0, grf_r_resampled[i], 0.0,
                cop_px['r'][i], ipy, cop_pz['r'][i], 0.0, 0.0, 0.0,
            ]
            fh.write('\t'.join(f'{v:.6f}' for v in row) + '\n')

    log(f'  [EXT] 力文件: {sto_path}  ({len(mot_times)} frames)')

    # ---- 写 ExternalLoads XML ----
    xml_grf_l = (
        '\t\t\t<ExternalForce name="grf_l">\n'
        f'\t\t\t\t<applied_to_body>{insole_body_l}</applied_to_body>\n'
        '\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>\n'
        f'\t\t\t\t<point_expressed_in_body>{insole_body_l}</point_expressed_in_body>\n'
        '\t\t\t\t<force_identifier>grf_l_v</force_identifier>\n'
        '\t\t\t\t<point_identifier>grf_l_p</point_identifier>\n'
        '\t\t\t\t<torque_identifier>grf_l_torque_</torque_identifier>\n'
        '\t\t\t</ExternalForce>\n'
    ) if has_insole else ''

    xml_grf_r = (
        '\t\t\t<ExternalForce name="grf_r">\n'
        f'\t\t\t\t<applied_to_body>{insole_body_r}</applied_to_body>\n'
        '\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>\n'
        f'\t\t\t\t<point_expressed_in_body>{insole_body_r}</point_expressed_in_body>\n'
        '\t\t\t\t<force_identifier>grf_r_v</force_identifier>\n'
        '\t\t\t\t<point_identifier>grf_r_p</point_identifier>\n'
        '\t\t\t\t<torque_identifier>grf_r_torque_</torque_identifier>\n'
        '\t\t\t</ExternalForce>\n'
    ) if has_insole else ''

    xml_path = os.path.join(ext_dir, f'bar_loads_{load_key}.xml')
    xml_content = (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<OpenSimDocument Version="40000">\n'
        '\t<ExternalLoads name="bar_loads">\n'
        '\t\t<objects>\n'
        '\t\t\t<ExternalForce name="bar_force">\n'
        f'\t\t\t\t<applied_to_body>{bar_body}</applied_to_body>\n'
        '\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>\n'
        f'\t\t\t\t<point_expressed_in_body>{bar_body}</point_expressed_in_body>\n'
        '\t\t\t\t<force_identifier>bar_force_v</force_identifier>\n'
        '\t\t\t\t<point_identifier>bar_force_p</point_identifier>\n'
        '\t\t\t\t<torque_identifier>bar_torque_</torque_identifier>\n'
        '\t\t\t</ExternalForce>\n'
        + xml_grf_l + xml_grf_r +
        '\t\t</objects>\n'
        f'\t\t<datafile>{os.path.basename(sto_path)}</datafile>\n'
        '\t</ExternalLoads>\n'
        '</OpenSimDocument>\n'
    )
    with open(xml_path, 'w', encoding='utf-8') as fh:
        fh.write(xml_content)

    log(f'  [EXT] XML  : {xml_path}')
    log(f'  [EXT] 杆作用体: {bar_body},  作用点(local): {bar_point}')
    if has_insole:
        log(f'  [EXT] GRF 左脚: {insole_body_l},  右脚: {insole_body_r}')
        log(f"  [EXT] GRF 作用点来源: L={cop_src['l']}, R={cop_src['r']}")
    return xml_path