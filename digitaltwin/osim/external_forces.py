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
                             Mb=20.0, verbose=True):
    """
    从机器人数据和鞋垫数据生成 OpenSim ExternalLoads 文件。

    力学模型：
      杆件力（施加到 torso，-Y 向下）：
        F_bar = force_l + force_r + Mb*g + Mb * avg(acc_l, acc_r)

      足底 GRF（施加到左右 calcn，+Y 向上）：
        从 insole_file_l / insole_file_r 读取

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

    insole_folder = modeling.get('insole_folder', 'Sorted')
    insole_base   = os.path.join(folder, insole_folder)

    grf_l_resampled = np.zeros(len(mot_times))
    grf_r_resampled = np.zeros(len(mot_times))
    has_insole = False

    for side, key in [('L', 'insole_file_l'), ('R', 'insole_file_r')]:
        insole_rel = file_info.get(key)
        if insole_rel:
            t_s, f_s = InsoleProcessor.load(
                os.path.join(insole_base, insole_rel), verbose=verbose)
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
                0.0, F_bar_resampled[i], 0.0,   px,  py,  pz,  0.0, 0.0, 0.0,
                0.0, grf_l_resampled[i], 0.0,  ipx, ipy, ipz,  0.0, 0.0, 0.0,
                0.0, grf_r_resampled[i], 0.0,  ipx, ipy, ipz,  0.0, 0.0, 0.0,
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
    return xml_path