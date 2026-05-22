"""
inverse_dynamics.py

OpenSim 逆向动力学流水线。

主要接口:
    run_inverse_dynamics(model_path, mot_path, output_dir,
                         external_load_file=None, label='id',
                         verbose=True) -> bool
    run_step3_inverse_dynamics(config, base_dir,
                               use_external_forces=False, Mb=20.0,
                               external_load_file=None, verbose=True) -> None
"""
import os
import opensim as osim

from digitaltwin.osim.external_forces import get_ext_forces_dir, generate_external_loads


def _get_mot_files(config, base_dir):
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
#  底层函数
# ============================================================

def run_inverse_dynamics(model_path, mot_path, output_dir,
                         external_load_file=None, label='id',
                         verbose=True):
    """
    对指定 .mot 文件运行 OpenSim InverseDynamicsTool。

    Parameters
    ----------
    model_path         : str
    mot_path           : str
    output_dir         : str
    external_load_file : str, optional -- 外力 XML 路径
    label              : str
    verbose            : bool

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
#  Step 3 流水线
# ============================================================

def run_step3_inverse_dynamics(config, base_dir,
                               use_external_forces=False, Mb=20.0,
                               external_load_file=None, verbose=True):
    """
    Step 3: 对所有 load 运行逆向动力学 (InverseDynamics)。

    前提: Step 1 已完成（mot 文件存在）。
    输出到 result/{experiment_label}/opensim/inverse_dynamics/{load_key}/

    Parameters
    ----------
    use_external_forces : bool
        True 时自动计算杆件力 + 足底 GRF。
        若 Step 2 已生成外力文件，直接复用，不重新生成。
    Mb                  : float -- 杆质量默认值
    external_load_file  : str, optional
        手动指定外力 XML（对所有 load 共用）。
        与 use_external_forces 互斥。
    """
    def log(msg):
        if verbose:
            print(msg)

    experiment_label = config['experiment_label']
    opensim_dir  = os.path.join(base_dir, 'result', experiment_label, 'opensim')
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
    if use_external_forces:
        log(f'外力: 杆件力 + 足底 GRF (Mb={Mb:.1f}kg)')
    elif external_load_file:
        log(f'外力: 手动指定 ({external_load_file})')
    else:
        log('外力: 无')

    for load_key, mot_path in mot_files.items():
        log(f'\n  负载 {load_key}:')
        id_dir   = os.path.join(opensim_dir, 'inverse_dynamics', str(load_key))
        ext_file = external_load_file

        if use_external_forces:
            xml_candidate = os.path.join(
                get_ext_forces_dir(config, base_dir, load_key),
                f'bar_loads_{load_key}.xml')
            if os.path.exists(xml_candidate):
                log(f'  [EXT] 外力文件已存在，直接复用: {xml_candidate}')
                ext_file = xml_candidate
            else:
                ext_file = generate_external_loads(
                    config=config, base_dir=base_dir,
                    load_key=load_key, mot_path=mot_path,
                    Mb=Mb, verbose=verbose,
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