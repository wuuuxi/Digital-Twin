"""
opensim_pipeline.py

OpenSim 分析流水线：Xsens .mot 转换、肌肉分析、逆向动力学。

主要接口:
    run_mot_conversion(config, base_dir, verbose=True)  -> dict
    run_muscle_analysis(model_path, mot_path, output_dir,
                        coordinates=None, label=None, verbose=True)
    run_inverse_dynamics(model_path, mot_path, output_dir,
                         external_load_file=None, label=None, verbose=True)
    run_opensim_pipeline(config, base_dir, verbose=True)

输出目录结构:
    result/{experiment_label}/opensim/
        mot/           -- .mot 关节角度文件
        muscle_analysis/{load_key}/   -- 肌肉力臂等分析结果
        inverse_dynamics/{load_key}/  -- 逆向动力学结果（可选）
"""
import os
import opensim as osim


# ============================================================
#  .mot 转换
# ============================================================

def run_mot_conversion(config, base_dir, verbose=True):
    """
    将 modeling_file.data 中所有 xsens_file 转换为 OpenSim .mot 文件。

    Parameters
    ----------
    config : dict
        解析后的 JSON 配置。
    base_dir : str
        包含 workspace/ 和 result/ 的基准目录。
    verbose : bool

    Returns
    -------
    dict
        {load_key: mot_file_path}
    """
    from digitaltwin.data.xsens_processor import XsensProcessor
    from pathlib import Path

    def log(msg):
        if verbose:
            print(msg)

    experiment_label = config['experiment_label']
    folder = config['folder']
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

        xsens_path = os.path.join(xsens_dir, xsens_file)
        log(f'\n  [{load_key}] {xsens_file}')

        xsens_data = XsensProcessor.process(
            xsens_file, load_key, folder,
            xsens_folder=xsens_dir)

        if xsens_data is None:
            log(f'    加载失败，跳过')
            continue

        mot_name = Path(xsens_file).stem + '_opensim.mot'
        mot_path = os.path.join(output_dir, mot_name)
        XsensProcessor.save_mot(xsens_data, mot_path)
        mot_files[load_key] = mot_path
        log(f'    已保存: {mot_path}')

    log(f'\n[mot] 共转换 {len(mot_files)} 个文件')
    return mot_files


# ============================================================
#  肌肉分析
# ============================================================

def run_muscle_analysis(model_path, mot_path, output_dir,
                        coordinates=None, label='muscle_analysis',
                        external_load_file=None, verbose=True):
    """
    对指定 .mot 文件运行 OpenSim MuscleAnalysis。

    Parameters
    ----------
    model_path : str
        缩放后的 .osim 模型文件路径。
    mot_path : str
        输入的 .mot 关节角度文件路径。
    output_dir : str
        分析结果输出目录。
    coordinates : list of str, optional
        计算力矩臂的坐标名称列表。若为 None 则使用下肢+腰椎默认集。
    label : str
        AnalyzeTool 名称标识。
    verbose : bool

    Returns
    -------
    bool  -- True 表示成功
    """
    def log(msg):
        if verbose:
            print(msg)

    if coordinates is None:
        coordinates = [
            'hip_flexion_r', 'hip_flexion_l',
            'knee_angle_r',  'knee_angle_l',
            'ankle_angle_r', 'ankle_angle_l',
            'flex_extension', 'lat_bending', 'axial_rotation',
        ]

    os.makedirs(output_dir, exist_ok=True)
    log(f'  [MA] 模型  : {model_path}')
    log(f'  [MA] 运动  : {mot_path}')
    log(f'  [MA] 输出  : {output_dir}')
    log(f'  [MA] 坐标  : {coordinates}')

    try:
        table = osim.TimeSeriesTable(mot_path)
        t = table.getIndependentColumn()
        t_start, t_end = t[0], t[len(t) - 1]

        ma = osim.MuscleAnalysis()
        ma.setStartTime(t_start)
        ma.setEndTime(t_end)
        ma.setOn(True)
        ma.setStepInterval(1)
        ma.setInDegrees(True)
        ma.setComputeMoments(True)

        muscles = osim.ArrayStr()
        muscles.append('all')
        ma.setMuscles(muscles)

        coords = osim.ArrayStr()
        for c in coordinates:
            coords.append(c)
        ma.setCoordinates(coords)

        tool = osim.AnalyzeTool()
        tool.setName(label)
        tool.setModelFilename(model_path)
        tool.setCoordinatesFileName(mot_path)
        tool.updAnalysisSet().cloneAndAppend(ma)
        tool.setResultsDir(output_dir)
        tool.setReplaceForceSet(False)
        if external_load_file and os.path.exists(external_load_file):
            tool.setExternalLoadsFileName(external_load_file)
            log(f'  [MA] 外力  : {external_load_file}')
        tool.setSolveForEquilibrium(True)
        tool.setStartTime(t_start)
        tool.setFinalTime(t_end)

        setup_xml = os.path.join(output_dir, f'Setup_MuscleAnalysis_{label}.xml')
        tool.printToXML(setup_xml)
        log(f'  [MA] Setup XML: {setup_xml}')

        tool2 = osim.AnalyzeTool(setup_xml, True)
        result = tool2.run()
        log(f'  [MA] 完成: {result}')
        return True

    except Exception as e:
        log(f'  [MA] 错误: {e}')
        import traceback
        traceback.print_exc()
        return False


# ============================================================
#  逆向动力学
# ============================================================

def run_inverse_dynamics(model_path, mot_path, output_dir,
                         external_load_file=None, label='id',
                         verbose=True):
    """
    对指定 .mot 文件运行 OpenSim InverseDynamicsTool。

    Parameters
    ----------
    model_path : str
    mot_path : str
    output_dir : str
    external_load_file : str, optional
        外部载荷 XML 文件路径（无则不使用外力）。
    label : str
    verbose : bool

    Returns
    -------
    bool
    """
    def log(msg):
        if verbose:
            print(msg)

    os.makedirs(output_dir, exist_ok=True)
    log(f'  [ID] 模型  : {model_path}')
    log(f'  [ID] 运动  : {mot_path}')
    log(f'  [ID] 输出  : {output_dir}')

    try:
        table = osim.TimeSeriesTable(mot_path)
        t = table.getIndependentColumn()
        t_start, t_end = t[0], t[len(t) - 1]

        id_tool = osim.InverseDynamicsTool()
        id_tool.setName(label)
        id_tool.setModelFileName(model_path)
        id_tool.setCoordinatesFileName(mot_path)
        id_tool.setResultsDir(output_dir)
        id_tool.setStartTime(t_start)
        id_tool.setEndTime(t_end)

        excluded = osim.ArrayStr()
        excluded.append('Muscles')
        id_tool.setExcludedForces(excluded)

        if external_load_file and os.path.exists(external_load_file):
            id_tool.setExternalLoadsFileName(external_load_file)
            log(f'  [ID] 外力  : {external_load_file}')

        setup_xml = os.path.join(output_dir, f'Setup_InverseDynamics_{label}.xml')
        id_tool.printToXML(setup_xml)
        id_tool2 = osim.InverseDynamicsTool(setup_xml)
        result = id_tool2.run()
        log(f'  [ID] 完成: {result}')
        return True

    except Exception as e:
        log(f'  [ID] 错误: {e}')
        import traceback
        traceback.print_exc()
        return False


# ============================================================
#  外力生成：杆件对人体的作用力（深蹲）
# ============================================================

def _get_ext_forces_dir(config, base_dir, load_key):
    """
    返回外力文件的共享目录。
    muscle_analysis 和 inverse_dynamics 共用此目录，避免重复生成。
    """
    experiment_label = config['experiment_label']
    return os.path.join(
        base_dir, 'result', experiment_label,
        'opensim', 'external_forces', str(load_key)
    )

def generate_bar_external_loads(config, base_dir, load_key, mot_path,
                                output_dir, Mb=20.0, verbose=True):
    """
    从机器人数据计算深蹲杆件外力，生成 OpenSim ExternalLoads 所需文件。

    力学模型（牛顿第二定律，以杆为研究对象）：
        F_on_person = force_l + force_r + Mb*g + Mb * avg(acc_l, acc_r)
    方向：竖直向下，施加到 bar_contact_body（默认 torso）。

    关于作用点选择（深蹲背部杆接触点）：
        深蹲时杆架在斜方肌上，对应模型中 torso 体的上部后侧。
        默认作用点 bar_contact_point = [0.0, 0.30, -0.07] (torso 局部坐标, 单位 m)。
        其中 y≈0.30m 为 torso 体坐标系内大致肩胛高度，z=-0.07m 为向后偏移。
        可通过 opensim_settings.bar_contact_point 覆盖此默认值。
        若需精确校准，可用 opensim_settings.py 中的
        list_model_joints() 查看 Torsojnt 所在高度作为参考。

    生成文件：
        {output_dir}/bar_force_{load_key}.sto   -- 时序力数据
        {output_dir}/bar_loads_{load_key}.xml   -- ExternalLoads XML

    Parameters
    ----------
    config : dict
        opensim_settings 可选字段：
            bar_mass           (float, kg,  默认 20.0)
            bar_contact_body   (str,        默认 'torso')
            bar_contact_point  ([x,y,z], m, 默认 [0.0, 0.30, -0.07])
    base_dir : str
    load_key : str
    mot_path : str  -- 用于获取时间轴
    output_dir : str
    Mb : float  -- 杆质量默认值（会被 opensim_settings.bar_mass 覆盖）
    verbose : bool

    Returns
    -------
    str or None  -- ExternalLoads XML 路径（失败返回 None）
    """
    import numpy as np
    import pandas as pd

    def log(msg):
        if verbose:
            print(msg)

    # 外力文件写入共享目录（muscle_analysis 和 inverse_dynamics 共用，避免重复生成）
    ext_dir = _get_ext_forces_dir(config, base_dir, load_key)
    os.makedirs(ext_dir, exist_ok=True)

    # ---- 读取配置 ----
    osim_cfg    = config.get('opensim_settings', {})
    bar_body    = osim_cfg.get('bar_contact_body',  'torso')
    bar_point   = osim_cfg.get('bar_contact_point', [-0.07, 0.30, 0.0])
    Mb          = float(osim_cfg.get('bar_mass', Mb))
    g           = 9.81

    modeling  = config['modeling_file']
    folder    = config['folder']
    file_info = modeling['data'].get(str(load_key))
    if file_info is None:
        log(f'  [EXT] load_key={load_key} 不在 modeling_file.data 中')
        return None

    robot_file = file_info.get('robot_file')
    if not robot_file:
        log(f'  [EXT] load_key={load_key} 无 robot_file，跳过外力生成')
        return None

    # ---- 加载机器人数据 ----
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

    # ---- 计算外力 ----
    robot_time = robot_df['time'].values.astype(float)
    force_l = robot_df['force_l'].values.astype(float) if 'force_l' in robot_df else np.zeros(len(robot_df))
    force_r = robot_df['force_r'].values.astype(float) if 'force_r' in robot_df else np.zeros(len(robot_df))
    acc_l   = robot_df['acc_l'].values.astype(float)   if 'acc_l'   in robot_df else np.zeros(len(robot_df))
    acc_r   = robot_df['acc_r'].values.astype(float)   if 'acc_r'   in robot_df else np.zeros(len(robot_df))

    avg_acc = (acc_l + acc_r) / 2.0
    F_mag   = force_l + force_r + Mb * g + Mb * avg_acc   # N, 向下作用于人
    F_y     = -F_mag   # OpenSim y 轴向上，故向下为负

    log(f'  [EXT] 杆质量={Mb:.1f}kg, g={g}m/s²')
    log(f'  [EXT] F_mag 范围: [{F_mag.min():.1f}, {F_mag.max():.1f}] N')

    # ---- 对齐到 .mot 时间轴（numpy 线性插值，无 scipy 依赖）----
    mot_table = osim.TimeSeriesTable(mot_path)
    mot_times = np.array(mot_table.getIndependentColumn())
    F_resampled = np.interp(mot_times, robot_time, F_y,
                            left=F_y[0], right=F_y[-1])

    # ---- 写 .sto 力文件 ----
    #  ExternalLoads 列名规则: {data_source_name}_force_v{x/y/z} 为力，
    #                           {data_source_name}_force_p{x/y/z} 为作用点，
    #                           {data_source_name}_torque_{x/y/z} 为力矩。
    px, py, pz = float(bar_point[0]), float(bar_point[1]), float(bar_point[2])
    cols = ['time',
            'bar_force_vx', 'bar_force_vy', 'bar_force_vz',
            'bar_force_px', 'bar_force_py', 'bar_force_pz',
            'bar_torque_x', 'bar_torque_y', 'bar_torque_z']

    sto_path = os.path.join(ext_dir, f'bar_force_{load_key}.sto')
    with open(sto_path, 'w') as fh:
        fh.write('bar_external_force\n')
        fh.write(f'nRows={len(mot_times)}\n')
        fh.write(f'nColumns={len(cols)}\n')
        fh.write('inDegrees=no\n')
        fh.write('endheader\n')
        fh.write('\t'.join(cols) + '\n')
        for i, t in enumerate(mot_times):
            row = [t, 0.0, F_resampled[i], 0.0, px, py, pz, 0.0, 0.0, 0.0]
            fh.write('\t'.join(f'{v:.6f}' for v in row) + '\n')

    log(f'  [EXT] 力文件: {sto_path}  ({len(mot_times)} frames)')

    # ---- 写 ExternalLoads XML ----
    #  force_expressed_in_body=ground: 力在大地坐标系下表达（y 向下）
    #  point_expressed_in_body=torso:  作用点在 torso 局部坐标系下表达
    xml_path = os.path.join(ext_dir, f'bar_loads_{load_key}.xml')
    # OpenSim 4.x ExternalLoads XML 格式：
    #   force_identifier  指定 .sto 中力列的前缀（后缀自动加 x/y/z）
    #   point_identifier  指定作用点列的前缀
    #   torque_identifier 指定力矩列的前缀
    #   对应 .sto 列名： bar_force_vx/vy/vz, bar_force_px/py/pz, bar_torque_x/y/z
    xml_content = (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<OpenSimDocument Version="40000">\n'
        '	<ExternalLoads name="bar_loads">\n'
        '		<objects>\n'
        '			<ExternalForce name="bar_force">\n'
        f'				<applied_to_body>{bar_body}</applied_to_body>\n'
        '				<force_expressed_in_body>ground</force_expressed_in_body>\n'
        f'				<point_expressed_in_body>{bar_body}</point_expressed_in_body>\n'
        '				<force_identifier>bar_force_v</force_identifier>\n'
        '				<point_identifier>bar_force_p</point_identifier>\n'
        '				<torque_identifier>bar_torque_</torque_identifier>\n'
        '			</ExternalForce>\n'
        '		</objects>\n'
        f'		<datafile>{os.path.basename(sto_path)}</datafile>\n'
        '	</ExternalLoads>\n'
        '</OpenSimDocument>\n'
    )
    with open(xml_path, 'w', encoding='utf-8') as fh:
        fh.write(xml_content)

    log(f'  [EXT] XML  : {xml_path}')
    log(f'  [EXT] 作用体: {bar_body},  作用点(local): {bar_point}')
    return xml_path


# ============================================================
#  工具：从输出目录扫描已有 .mot 文件
# ============================================================

def _get_mot_files(config, base_dir):
    """
    根据 config 推导出各 load_key 对应的 .mot 文件路径字典。
    文件必须已由 run_step1_mot_conversion 生成。
    """
    from pathlib import Path
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


def _get_scaled_model(config, base_dir):
    experiment_label = config['experiment_label']
    opensim_dir = os.path.join(base_dir, 'result', experiment_label, 'opensim')
    return os.path.join(opensim_dir, f'whole body model_{experiment_label}.osim')


# ============================================================
#  独立步骤函数（可单独调用或在 example 中按需注释）
# ============================================================

def run_step1_mot_conversion(config, base_dir, verbose=True):
    """
    Step 1: Xsens Excel -> OpenSim .mot 关节角度文件。

    输出到 result/{experiment_label}/opensim/mot/

    Returns
    -------
    dict  {load_key: mot_file_path}
    """
    return run_mot_conversion(config, base_dir, verbose=verbose)


def run_step2_muscle_analysis(config, base_dir, coordinates=None,
                              use_bar_force=False, Mb=20.0, verbose=True):
    """
    Step 2: 对所有 load 运行肌肉分析 (MuscleAnalysis)。

    前提: Step 1 已完成（mot 文件存在）。
    输出到 result/{experiment_label}/opensim/muscle_analysis/{load_key}/
    外力文件共享到 result/{experiment_label}/opensim/external_forces/{load_key}/

    Parameters
    ----------
    coordinates : list of str, optional
        计算力矩臂的坐标列表。None 则优先读 opensim_settings，
        再退回到函数内置默认值。
    use_bar_force : bool
        是否从机器人数据自动计算杆件力并纳入肌肉分析。
        外力文件保存到共享目录，inverse_dynamics 可直接复用。
    Mb : float
        杆件质量默认值（会被 opensim_settings.bar_mass 覆盖）。
    """
    def log(msg):
        if verbose:
            print(msg)

    experiment_label = config['experiment_label']
    opensim_dir = os.path.join(base_dir, 'result', experiment_label, 'opensim')
    scaled_model = _get_scaled_model(config, base_dir)

    if not os.path.exists(scaled_model):
        log(f'[ERROR] 找不到缩放模型: {scaled_model}')
        log('请先运行 example_scaling.py')
        return

    # coordinates 优先级：调用参数 > JSON opensim_settings > 函数默认
    if coordinates is None:
        coordinates = config.get('opensim_settings', {}).get(
            'muscle_analysis_coordinates', None)

    mot_files = _get_mot_files(config, base_dir)
    if not mot_files:
        log('[ERROR] 未找到 .mot 文件，请先运行 run_step1_mot_conversion()')
        return

    log('\n' + '=' * 60)
    log('[Step 2] 肌肉分析 (MuscleAnalysis)')
    log('=' * 60)
    log(f'模型: {scaled_model}')

    if use_bar_force:
        log(f'外力模式: 自动计算杆件力 (Mb={Mb:.1f}kg)')
    else:
        log('外力模式: 无外力')

    for load_key, mot_path in mot_files.items():
        log(f'\n  负载 {load_key}:')
        ma_dir = os.path.join(opensim_dir, 'muscle_analysis', str(load_key))

        ext_file = None
        if use_bar_force:
            xml_candidate = os.path.join(
                _get_ext_forces_dir(config, base_dir, load_key),
                f'bar_loads_{load_key}.xml')
            if os.path.exists(xml_candidate):
                log(f'  [EXT] 外力文件已存在，直接复用: {xml_candidate}')
                ext_file = xml_candidate
            else:
                ext_file = generate_bar_external_loads(
                    config=config,
                    base_dir=base_dir,
                    load_key=load_key,
                    mot_path=mot_path,
                    output_dir=None,   # 内部会自动使用共享目录
                    Mb=Mb,
                    verbose=verbose,
                )

        run_muscle_analysis(
            model_path=scaled_model,
            mot_path=mot_path,
            output_dir=ma_dir,
            coordinates=coordinates,
            external_load_file=ext_file,
            label=f'{experiment_label}_{load_key}',
            verbose=verbose,
        )

    log('\n[Step 2] 肌肉分析完成')
    log(f'输出目录: {opensim_dir}/muscle_analysis/')


def run_step3_inverse_dynamics(config, base_dir,
                               use_bar_force=False, Mb=20.0,
                               external_load_file=None, verbose=True):
    """
    Step 3: 对所有 load 运行逆向动力学 (InverseDynamics)。

    前提: Step 1 已完成（mot 文件存在）。
    输出到 result/{experiment_label}/opensim/inverse_dynamics/{load_key}/

    Parameters
    ----------
    use_bar_force : bool
        是否从机器人数据自动计算杆件外力并传入 ID。
        True 时会调用 generate_bar_external_loads() 逐个负载生成外力文件。
        外力参数可在 opensim_settings 中配置：
            bar_mass           (float, kg,  默认 20.0)
            bar_contact_body   (str,        默认 'torso')
            bar_contact_point  ([x,y,z], m, 默认 [0.0, 0.30, -0.07])
    Mb : float
        杆件质量默认值（会被 opensim_settings.bar_mass 覆盖）。
    external_load_file : str, optional
        手动指定的外力 XML 文件（对所有 load 共用）。
        与 use_bar_force 互斜：若 use_bar_force=True 则忽略此参数。
    """
    def log(msg):
        if verbose:
            print(msg)

    experiment_label = config['experiment_label']
    opensim_dir = os.path.join(base_dir, 'result', experiment_label, 'opensim')
    scaled_model = _get_scaled_model(config, base_dir)

    if not os.path.exists(scaled_model):
        log(f'[ERROR] 找不到缩放模型: {scaled_model}')
        log('请先运行 example_scaling.py')
        return

    mot_files = _get_mot_files(config, base_dir)
    if not mot_files:
        log('[ERROR] 未找到 .mot 文件，请先运行 run_step1_mot_conversion()')
        return

    log('\n' + '=' * 60)
    log('[Step 3] 逆向动力学 (InverseDynamics)')
    log('=' * 60)
    log(f'模型: {scaled_model}')
    if use_bar_force:
        log(f'外力模式: 自动计算杆件力 (Mb={Mb:.1f}kg)')
    elif external_load_file:
        log(f'外力模式: 手动指定 ({external_load_file})')
    else:
        log('外力模式: 无外力')

    for load_key, mot_path in mot_files.items():
        log(f'\n  负载 {load_key}:')
        id_dir = os.path.join(opensim_dir, 'inverse_dynamics', str(load_key))

        # 负载适用的外力文件
        ext_file = external_load_file
        if use_bar_force:
            # 优先复用 Step 2 已生成的共享外力文件
            xml_candidate = os.path.join(
                _get_ext_forces_dir(config, base_dir, load_key),
                f'bar_loads_{load_key}.xml')
            if os.path.exists(xml_candidate):
                log(f'  [EXT] 外力文件已存在，直接复用: {xml_candidate}')
                ext_file = xml_candidate
            else:
                ext_file = generate_bar_external_loads(
                    config=config,
                    base_dir=base_dir,
                    load_key=load_key,
                    mot_path=mot_path,
                    output_dir=None,   # 内部自动使用共享目录
                    Mb=Mb,
                    verbose=verbose,
                )
            if ext_file is None:
                log(f'  [WARN] 外力生成失败，此负载将不包含外力运行 ID')

        run_inverse_dynamics(
            model_path=scaled_model,
            mot_path=mot_path,
            output_dir=id_dir,
            external_load_file=ext_file,
            label=f'{experiment_label}_{load_key}',
            verbose=verbose,
        )

    log('\n[Step 3] 逆向动力学完成')
    log(f'输出目录: {opensim_dir}/inverse_dynamics/')