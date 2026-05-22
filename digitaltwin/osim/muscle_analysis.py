"""
muscle_analysis.py

OpenSim 肌肉分析（MuscleAnalysis）流水线。

接口：
    run_muscle_analysis(model_path, mot_path, output_dir, ...) -> bool
    run_step2_muscle_analysis(config, base_dir, ...) -> None
"""
import os
import opensim as osim

from digitaltwin.osim.mot_pipeline import get_mot_files, get_scaled_model
from digitaltwin.osim.external_forces import get_ext_forces_dir, generate_external_loads


def _flatten_muscle_list(muscles):
    """
    将 JSON 中的 muscle_analysis_muscles 展平成 OpenSim MuscleAnalysis 可用列表。

    支持：
      ["vas_lat_r", "rect_fem_r"]
      [["glut_max1_r", "glut_max2_r"], "rect_fem_r"]
    """
    if muscles is None:
        return None
    out = []
    for item in muscles:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            for x in item:
                if x:
                    out.append(str(x))
        else:
            out.append(str(item))
    return out


def run_muscle_analysis(model_path, mot_path, output_dir,
                        coordinates=None, muscles_to_analyze=None,
                        label='muscle_analysis',
                        external_load_file=None, verbose=True):
    """
    对指定 .mot 文件运行 OpenSim MuscleAnalysis。

    Parameters
    ----------
    model_path          : str
    mot_path            : str
    output_dir          : str
    coordinates         : list of str, optional
    muscles_to_analyze  : list[str], optional
        若为 None，则分析 all；若提供，则只分析这些 OpenSim 肌肉。
    label               : str
    external_load_file  : str, optional
    verbose             : bool

    Returns
    -------
    bool
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

    muscles_to_analyze = _flatten_muscle_list(muscles_to_analyze)

    os.makedirs(output_dir, exist_ok=True)
    log(f'  [MA] 模型  : {model_path}')
    log(f'  [MA] 运动  : {mot_path}')
    log(f'  [MA] 输出  : {output_dir}')
    log(f'  [MA] 坐标  : {coordinates}')
    log(f'  [MA] 肌肉  : {muscles_to_analyze if muscles_to_analyze else "all"}')

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
        if muscles_to_analyze:
            for m in muscles_to_analyze:
                muscles.append(m)
        else:
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
        result = osim.AnalyzeTool(setup_xml, True).run()
        log(f'  [MA] 完成: {result}')
        return True

    except Exception as e:
        log(f'  [MA] 错误: {e}')
        import traceback
        traceback.print_exc()
        return False


def run_step2_muscle_analysis(config, base_dir, coordinates=None,
                              use_external_forces=False, Mb=20.0, verbose=True):
    """
    Step 2: 对所有 load 运行肌肉分析 (MuscleAnalysis)。

    前提: Step 1 已完成（mot 文件存在）。
    输出到 result/{experiment_label}/opensim/muscle_analysis/{load_key}/
    外力文件共享到 result/{experiment_label}/opensim/external_forces/{load_key}/

    JSON 支持：
      opensim_settings.muscle_analysis_coordinates
      opensim_settings.muscle_analysis_muscles

    为兼容，也会读取顶层 config["muscle_analysis_muscles"]。
    """
    def log(msg):
        if verbose:
            print(msg)

    experiment_label = config['experiment_label']
    opensim_dir  = os.path.join(base_dir, 'result', experiment_label, 'opensim')
    scaled_model = get_scaled_model(config, base_dir)

    if not os.path.exists(scaled_model):
        log(f'[ERROR] 找不到缩放模型: {scaled_model}')
        return

    osim_cfg = config.get('opensim_settings', {})

    if coordinates is None:
        coordinates = osim_cfg.get('muscle_analysis_coordinates', None)

    muscles_to_analyze = (
        osim_cfg.get('muscle_analysis_muscles')
        or config.get('muscle_analysis_muscles')
    )
    muscles_to_analyze = _flatten_muscle_list(muscles_to_analyze)

    # 如果给了 musc_label，一并检查长度，方便发现 JSON 配置不一致
    musc_label = config.get('emg_settings', {}).get('musc_label', [])
    if muscles_to_analyze and musc_label and len(muscles_to_analyze) != len(musc_label):
        log(f'[WARN] muscle_analysis_muscles 数量 ({len(muscles_to_analyze)}) '
            f'与 emg_settings.musc_label 数量 ({len(musc_label)}) 不一致。')
        log('       若某个 EMG 标签对应多个 OpenSim 肌肉，可用嵌套 list；程序会展开后用于 MuscleAnalysis。')

    mot_files = get_mot_files(config, base_dir)
    if not mot_files:
        log('[ERROR] 未找到 .mot 文件，请先运行 run_step1_mot_conversion()')
        return

    log('\n' + '=' * 60)
    log('[Step 2] 肌肉分析 (MuscleAnalysis)')
    log('=' * 60)
    log(f'模型: {scaled_model}')
    log(f'外力: {"杆件力 + 足底 GRF" if use_external_forces else "无"}')
    log(f'肌肉: {muscles_to_analyze if muscles_to_analyze else "all"}')

    for load_key, mot_path in mot_files.items():
        log(f'\n  负载 {load_key}:')
        ma_dir   = os.path.join(opensim_dir, 'muscle_analysis', str(load_key))
        ext_file = None

        if use_external_forces:
            xml_candidate = os.path.join(
                get_ext_forces_dir(config, base_dir, load_key),
                f'bar_loads_{load_key}.xml')
            if os.path.exists(xml_candidate):
                log(f'  [EXT] 复用已有外力文件: {xml_candidate}')
                ext_file = xml_candidate
            else:
                ext_file = generate_external_loads(
                    config=config, base_dir=base_dir,
                    load_key=load_key, mot_path=mot_path,
                    Mb=Mb, verbose=verbose)

        run_muscle_analysis(
            model_path=scaled_model, mot_path=mot_path,
            output_dir=ma_dir, coordinates=coordinates,
            muscles_to_analyze=muscles_to_analyze,
            external_load_file=ext_file,
            label=f'{experiment_label}_{load_key}',
            verbose=verbose)

    log('\n[Step 2] 肌肉分析完成')
    log(f'输出目录: {opensim_dir}/muscle_analysis/')