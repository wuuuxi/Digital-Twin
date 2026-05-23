"""
muscle_analysis.py

OpenSim 肌肉分析（MuscleAnalysis）流水线。

接口：
    run_muscle_analysis(model_path, mot_path, output_dir, ...) -> bool
    run_step2_muscle_analysis(config, base_dir, ...) -> None

EMG 驱动模式：
    当 use_emg_controls=True 时，会从 emg_processor 读取 MVC 归一化后的肌肉激活，
    生成 OpenSim controls .sto 文件，传给 AnalyzeTool，
    使 MuscleAnalysis 使用真实激活而非默认 0.01。

    JSON 中的 emg_settings.musc_label 与 opensim_settings.muscle_analysis_muscles
    一一对应，程序通过该映射关系将 EMG 通道信号写入 controls 文件。
    没有 EMG 覆盖的肌肉（如 use_only_configured_muscles=False 时分析 all）
    将使用 DEFAULT_ACTIVATION 填充。
"""
import os
import numpy as np
import opensim as osim

from digitaltwin.osim.mot_pipeline import get_mot_files, get_scaled_model
from digitaltwin.osim.external_forces import get_ext_forces_dir, generate_external_loads

# 没有 EMG 对应的肌肉使用的默认激活系数
DEFAULT_ACTIVATION = 0.02


# ============================================================
#  辅助函数
# ============================================================

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


def _build_emg_label_to_muscles_map(config):
    """
    返回 {emg_label: [opensim_muscle_name, ...]} 的映射。

    基于 JSON 中的：
      emg_settings.musc_label          = ["LVL", "LRF", "LBF", ...]
      opensim_settings.muscle_analysis_muscles = ["vas_lat_l", "rect_fem_l", ...]
    两者一一对应。
    """
    musc_labels = config.get('emg_settings', {}).get('musc_label', [])
    osim_cfg = config.get('opensim_settings', {})
    ma_muscles = (
        osim_cfg.get('muscle_analysis_muscles')
        or config.get('muscle_analysis_muscles')
    )
    if not musc_labels or not ma_muscles:
        return {}
    mapping = {}
    for label, item in zip(musc_labels, ma_muscles):
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            muscles = [str(x) for x in item if x]
        else:
            muscles = [str(item)]
        if muscles:
            mapping[label] = muscles
    return mapping


def build_emg_controls_file(config, base_dir, load_key, mot_path,
                             out_dir, muscles_to_write, verbose=True):
    """
    读取当前 load 的 EMG 数据，将归一化激活信号写入 OpenSim controls .sto 文件。

    Parameters
    ----------
    config            : dict
    base_dir          : str
    load_key          : str
    mot_path          : str  -- 用于读取标准时间轴
    out_dir           : str  -- 输出目录
    muscles_to_write  : list[str] or None
        需要写入 controls 的 OpenSim 肌肉名列表。
        None 表示仅写入有 EMG 对应的肌肉。
    verbose           : bool

    Returns
    -------
    str or None
        已生成的 controls 文件路径，失败则返回 None。
    """
    def log(msg):
        if verbose:
            print(msg)

    try:
        from digitaltwin.data.emg_processor import EMGProcessor
    except ImportError:
        log('  [EMG] 无法导入 EMGProcessor，跳过 EMG controls 生成')
        return None

    file_info = config.get('modeling_file', {}).get('data', {}).get(str(load_key))
    if not file_info:
        log(f'  [EMG] 找不到 load={load_key} 的文件配置')
        return None

    emg_file = file_info.get('emg_file')
    if not emg_file:
        # 尝试公共 emg_file 字段
        emg_file = config.get('emg_file', {}).get(str(load_key))
    if not emg_file:
        log(f'  [EMG] load={load_key} 无 emg_file 字段，跳过 EMG controls')
        return None

    emg_settings = config.get('emg_settings', {})
    folder = config.get('folder', base_dir)

    # modeling_file 下的 emg_file 对应的文件夹优先用 modeling_file.emg_folder，
    # 并拼上项目根目录 folder 变成绝对路径。
    modeling_emg_folder = config.get('modeling_file', {}).get('emg_folder', None)
    if modeling_emg_folder:
        emg_folder = os.path.join(folder, modeling_emg_folder)
    else:
        raw_emg_folder = emg_settings.get('emg_folder', '')
        emg_folder = os.path.join(folder, raw_emg_folder) if raw_emg_folder else folder

    fs = int(emg_settings.get('fs', 1000))
    musc_label = emg_settings.get('musc_label', [])
    musc_mvc = emg_settings.get('musc_mvc', [])
    remove_leading_zeros = config.get('motion_settings', {}).get(
        'remove_leading_zeros', emg_settings.get('remove_leading_zeros', False))
    motion_flag = config.get('motion_settings', {}).get(
        'motion_flag', emg_settings.get('motion_flag', 'all'))

    processor = EMGProcessor(
        fs=fs,
        musc_mvc=musc_mvc,
        musc_label=musc_label,
    )
    emg_data = processor.process(
        emg_file=emg_file,
        load_weight=str(load_key),
        emg_folder=emg_folder,
        folder=folder,
        motion_flag=motion_flag,
        remove_leading_zeros=remove_leading_zeros,
    )
    if emg_data is None:
        log(f'  [EMG] load={load_key} EMG 数据处理失败')
        return None

    emg_time = np.asarray(emg_data['time'], dtype=float)
    norm_signals = emg_data.get('norm_signals', {})

    # EMG label -> OpenSim muscles 映射
    label_to_muscles = _build_emg_label_to_muscles_map(config)

    # 建立 OpenSim 肌肉 -> 激活串列的字典
    # 优先级：如果一个 OpenSim 肌肉对应多个 EMG label，取均値
    muscle_activation_map = {}   # {opensim_muscle_name: np.ndarray on emg_time}
    for label, muscles in label_to_muscles.items():
        if label not in norm_signals:
            continue
        sig = np.clip(np.asarray(norm_signals[label], dtype=float), 0.0, 1.0)
        for m in muscles:
            if m not in muscle_activation_map:
                muscle_activation_map[m] = []
            muscle_activation_map[m].append(sig)

    # 平均
    for m in muscle_activation_map:
        stacked = np.stack(muscle_activation_map[m], axis=0)
        muscle_activation_map[m] = np.nanmean(stacked, axis=0)

    # 读取 .mot 时间轴
    try:
        mot_table = osim.TimeSeriesTable(mot_path)
        col_time = mot_table.getIndependentColumn()
        t_mot = np.array([col_time[i] for i in range(len(col_time))])
    except Exception as e:
        log(f'  [EMG] 读取 mot 时间轴失败: {e}')
        return None

    # 决定要写入的肌肉列表
    if muscles_to_write is None:
        # 仅写有 EMG 对应的肌肉
        write_cols = sorted(muscle_activation_map.keys())
    else:
        write_cols = list(muscles_to_write)

    if not write_cols:
        log('  [EMG] 无需要写入的肌肉，跳过')
        return None

    # 构建 .sto 内容
    n_rows = len(t_mot)
    label_str = f"{config.get('experiment_label', 'exp')}_{load_key}"
    lines = [
        f'{label_str}_emg_controls',
        'version=1',
        f'nRows={n_rows}',
        f'nColumns={len(write_cols) + 1}',
        'inDegrees=no',
        'endheader',
        '\t'.join(['time'] + write_cols),
    ]
    for t in t_mot:
        row_vals = []
        for m in write_cols:
            if m in muscle_activation_map:
                # 插尺到 .mot 时间轴
                sig = muscle_activation_map[m]
                v = float(np.interp(t, emg_time, sig,
                                    left=float(sig[0]),
                                    right=float(sig[-1])))
                v = max(0.0, min(1.0, v))
            else:
                v = DEFAULT_ACTIVATION
            row_vals.append(f'{v:.6f}')
        lines.append('\t'.join([f'{t:.6f}'] + row_vals))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{label_str}_emg_controls.sto')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    log(f'  [EMG] controls 文件已保存: {out_path}')
    log(f'  [EMG] 包含 {len(write_cols)} 块肌肉'
        f'，其中 {sum(1 for m in write_cols if m in muscle_activation_map)} 块有 EMG 数据'
        f'，其余使用默认激活 {DEFAULT_ACTIVATION}')
    return out_path


# ============================================================
#  模型辅助
# ============================================================

def _get_leg_muscles_from_model(model_path,
                                group_names=('left_leg', 'right_leg'),
                                verbose=True):
    """
    使用 OpenSim Python API 从 ForceSet 的 ObjectGroup 中获取肌肉列表。

    关键点：ObjectGroup 不继承自 Set，没有 getSize()/get()。
    正确迭代接口是 ``group.getMembers()``，
    返回 ``ArrayPtrs<Object>``，该对象才有 getSize()/get()。

    Parameters
    ----------
    model_path   : str  (.osim 文件路径)
    group_names  : tuple[str]
        目标组名称，默认 ('left_leg', 'right_leg')。
    verbose      : bool

    Returns
    -------
    list[str] or None
    """
    def log(msg):
        if verbose:
            print(msg)

    try:
        model = osim.Model(model_path)
        model.initSystem()
        force_set = model.getForceSet()

        leg_muscles = []
        for gname in group_names:
            try:
                group   = force_set.getGroup(gname)    # 返回 ObjectGroup
                members = group.getMembers()            # 返回 ArrayPtrs<Object>
                count   = 0
                for i in range(members.getSize()):
                    name = members.get(i).getName()
                    if name not in leg_muscles:
                        leg_muscles.append(name)
                    count += 1
                log(f'  [MA] 肌肉组 "{gname}": {count} 块')
            except Exception as eg:
                log(f'  [MA] 警告: 无法读取肌肉组 "{gname}": {eg}')

        if leg_muscles:
            log(f'  [MA] 腿部肌肉共 {len(leg_muscles)} 块')
            return leg_muscles

        log('  [MA] 警告: 未从任何肌肉组获取到肌肉，将使用 all')
        return None
    except Exception as e:
        log(f'  [MA] 无法读取模型肌肉组: {e}')
        import traceback
        traceback.print_exc()
        return None


# ============================================================
#  主分析函数
# ============================================================

def run_muscle_analysis(model_path, mot_path, output_dir,
                        coordinates=None, muscles_to_analyze=None,
                        controls_file=None,
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
        若为 None 则分析 all；若提供，则只分析这些 OpenSim 肌肉。
    controls_file       : str, optional
        OpenSim controls .sto 文件路径。提供时会覆盖默认激活系数，
        实现 EMG 驱动。
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
    if controls_file:
        log(f'  [MA] EMG控制: {controls_file}')

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

        # EMG 驱动模式：
        #   setSolveForEquilibrium(False) 跳过肌纤维平衡迭代，直接使用
        #   controls 文件中的 EMG 激活，避免平衡失败时覆盖激活。
        #   对于 fiber length 不准的模型更合适。
        solve_eq = (controls_file is None)  # 无 EMG controls 时保持默认行为
        tool.setSolveForEquilibrium(solve_eq)
        log(f'  [MA] 平衡迭代: {solve_eq}')

        tool.setStartTime(t_start)
        tool.setFinalTime(t_end)

        setup_xml = os.path.join(output_dir, f'Setup_MuscleAnalysis_{label}.xml')
        tool.printToXML(setup_xml)

        # 重载 XML 后再设置 controls，避免 XML 中路径失效导致 controls 被静默忽略
        final_tool = osim.AnalyzeTool(setup_xml, True)
        if controls_file and os.path.exists(controls_file):
            final_tool.setControlsFileName(os.path.abspath(controls_file))
            log(f'  [MA] controls 已加载到最终工具')
        result = final_tool.run()
        log(f'  [MA] 完成: {result}')
        return True

    except Exception as e:
        log(f'  [MA] 错误: {e}')
        import traceback
        traceback.print_exc()
        return False


# ============================================================
#  流水线入口
# ============================================================

def run_step2_muscle_analysis(config, base_dir, coordinates=None,
                              use_external_forces=False, Mb=20.0,
                              use_emg_controls=False,
                              use_only_configured_muscles=True,
                              leg_muscles_only=False,
                              load_keys=None,
                              verbose=True):
    """
    Step 2: 对指定 load 运行肌肉分析 (MuscleAnalysis)。

    前提: Step 1 已完成（mot 文件存在）。
    输出到 result/{experiment_label}/opensim/muscle_analysis/{load_key}/

    Parameters
    ----------
    config                       : dict
    base_dir                     : str
    coordinates                  : list[str], optional
        若为 None 从 JSON opensim_settings.muscle_analysis_coordinates 读取。
    use_external_forces          : bool
        是否加入外力（杆件力 + GRF）。
    Mb                           : float
        杆件质量（kg），仅当 use_external_forces=True 时有效。
    use_emg_controls             : bool
        是否使用 EMG 归一化激活作为 OpenSim controls。
        需要 JSON 中配置 emg_settings 且每个 load 指定 emg_file。
    use_only_configured_muscles  : bool
        True  = 只分析 JSON opensim_settings.muscle_analysis_muscles 中配置的肌肉。
        False = 分析模型中所有肌肉 (all)，其中有 EMG 覆盖的使用
                EMG 激活，其余使用默认激活。
    leg_muscles_only             : bool
        True = 仅分析 OpenSim 模型 ForceSet 中 left_leg 和 right_leg
               两个肌肉组所包含的肌肉，其余肌肉不参与计算（减少计算量）。
        与 muscle_analysis_muscles / use_only_configured_muscles 独立：
          - use_only_configured_muscles=True  → 取 configured_muscles ∩ leg_muscles
          - use_only_configured_muscles=False → 直接使用 leg_muscles
    load_keys                    : list[str] or None
        指定要处理的负载列表（如 ["20", "38", "56"]）。
        None = 处理所有 load。
    verbose                      : bool
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

    # 肌肉列表决策
    configured_muscles = _flatten_muscle_list(
        osim_cfg.get('muscle_analysis_muscles')
        or config.get('muscle_analysis_muscles')
    )

    # 从模型 left_leg / right_leg 组获取腿部肌肉（与 configured_muscles 无关）
    leg_muscles = None
    if leg_muscles_only:
        leg_muscles = _get_leg_muscles_from_model(scaled_model, verbose=verbose)
        if leg_muscles is None:
            log('[WARN] 未能从模型读取腿部肌肉组，将回退到默认行为')

    if use_only_configured_muscles and configured_muscles:
        if leg_muscles is not None:
            # 取 configured_muscles 与 leg_muscles 的交集，保持顺序
            leg_set = set(leg_muscles)
            muscles_to_analyze = [m for m in configured_muscles if m in leg_set]
            log(f'  [MA] 配置肌肉 x 腿部肌肉 交集: {len(muscles_to_analyze)} 块')
        else:
            muscles_to_analyze = configured_muscles
    elif leg_muscles is not None:
        # use_only_configured_muscles=False 且有腿部肌肉组
        muscles_to_analyze = leg_muscles
    else:
        muscles_to_analyze = None  # = all

    # 对 EMG controls，始终只写 configured_muscles 中有 EMG 映射的肌肉
    emg_muscles_to_write = configured_muscles

    # 检查 musc_label 与 configured_muscles 数量是否匹配
    musc_label = config.get('emg_settings', {}).get('musc_label', [])
    if configured_muscles and musc_label and len(configured_muscles) != len(musc_label):
        log(f'[WARN] muscle_analysis_muscles 数量 ({len(configured_muscles)}) '
            f'与 emg_settings.musc_label 数量 ({len(musc_label)}) 不一致。')
        log('       若某个 EMG 标签对应多个 OpenSim 肌肉，可用嵌套 list。')

    # 决定要处理的 load
    all_mot_files = get_mot_files(config, base_dir)
    if not all_mot_files:
        log('[ERROR] 未找到 .mot 文件，请先运行 run_step1_mot_conversion()')
        return

    if load_keys is not None:
        load_keys_set = set(str(k) for k in load_keys)
        mot_files = {k: v for k, v in all_mot_files.items() if str(k) in load_keys_set}
        if not mot_files:
            log(f'[WARN] 指定的 load_keys={load_keys} 与现有 mot 文件无交集')
            return
    else:
        mot_files = all_mot_files

    log('\n' + '=' * 60)
    log('[Step 2] 肌肉分析 (MuscleAnalysis)')
    log('=' * 60)
    log(f'模型: {scaled_model}')
    log(f'外力: {"杆件力 + 足底 GRF" if use_external_forces else "无"}')
    log(f'肌肉: {muscles_to_analyze if muscles_to_analyze else "all"}')
    log(f'EMG 控制: {use_emg_controls}')
    log(f'负载: {list(mot_files.keys())}')

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
            if ext_file is None:
                log('  [WARN] 外力生成失败，此负载将不包含外力')

        controls_file = None
        if use_emg_controls:
            controls_file = build_emg_controls_file(
                config=config,
                base_dir=base_dir,
                load_key=load_key,
                mot_path=mot_path,
                out_dir=ma_dir,
                muscles_to_write=emg_muscles_to_write,
                verbose=verbose,
            )
            if controls_file is None:
                log(f'  [WARN] load={load_key} EMG controls 生成失败，将使用默认激活')

        run_muscle_analysis(
            model_path=scaled_model,
            mot_path=mot_path,
            output_dir=ma_dir,
            coordinates=coordinates,
            muscles_to_analyze=muscles_to_analyze,
            controls_file=controls_file,
            external_load_file=ext_file,
            label=f'{experiment_label}_{load_key}',
            verbose=verbose,
        )

    log('\n[Step 2] 肌肉分析完成')
    log(f'输出目录: {opensim_dir}/muscle_analysis/')