"""
inverse_dynamics.py

OpenSim 逆向动力学（InverseDynamics）流水线。

接口：
    run_inverse_dynamics(model_path, mot_path, output_dir, ...) -> bool
    run_step3_inverse_dynamics(config, base_dir, ...) -> None
"""
import os
import opensim as osim

from digitaltwin.osim.mot_pipeline import get_mot_files, get_scaled_model
from digitaltwin.osim.external_forces import get_ext_forces_dir, generate_external_loads


def run_inverse_dynamics(model_path, mot_path, output_dir,
                         external_load_file=None, label='id',
                         output_body_forces=False,
                         verbose=True):
    """
    对指定 .mot 文件运行 OpenSim InverseDynamicsTool。

    Parameters
    ----------
    model_path          : str
    mot_path            : str
    output_dir          : str
    external_load_file  : str, optional
    label               : str
    output_body_forces  : bool
        若为 True，同时输出各关节处的体力/力矩
        （body_forces_at_joints），文件名为
        {label}_body_forces_at_joints.sto。
        默认 False。
    verbose             : bool

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

        # output_body_forces 在 Python API 中没有直接的 setter，
        # 通过修改 XML 文件实现。
        if output_body_forces:
            with open(setup_xml, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            # 1. 开启 body forces 输出
            xml_content = xml_content.replace(
                '<output_body_forces>false</output_body_forces>',
                '<output_body_forces>true</output_body_forces>'
            )
            # 2. joints_to_report_body_forces 默认为自闭合标签（空），
            #    必须设为 "All" 才会实际输出。
            import re
            xml_content = re.sub(
                r'<joints_to_report_body_forces\s*/>',
                '<joints_to_report_body_forces>All</joints_to_report_body_forces>',
                xml_content
            )
            # 兼容有内容的形式
            xml_content = re.sub(
                r'<joints_to_report_body_forces>(?!All).*?</joints_to_report_body_forces>',
                '<joints_to_report_body_forces>All</joints_to_report_body_forces>',
                xml_content, flags=re.DOTALL
            )
            # 如果该标签不存在，在 </output_body_forces> 后插入
            if 'joints_to_report_body_forces' not in xml_content:
                xml_content = xml_content.replace(
                    '<output_body_forces>true</output_body_forces>',
                    '<output_body_forces>true</output_body_forces>\n\t\t<joints_to_report_body_forces>All</joints_to_report_body_forces>'
                )
            with open(setup_xml, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            log('  [ID] 输出体力: 已启用 (body_forces_at_joints，All 关节)')

        result = osim.InverseDynamicsTool(setup_xml).run()
        log(f'  [ID] 完成: {result}')
        return True

    except Exception as e:
        log(f'  [ID] 错误: {e}')
        import traceback
        traceback.print_exc()
        return False


def run_step3_inverse_dynamics(config, base_dir,
                               use_external_forces=False, Mb=20.0,
                               external_load_file=None,
                               output_body_forces=False,
                               verbose=True):
    """
    Step 3: 对所有 load 运行逆向动力学 (InverseDynamics)。

    前提: Step 1 已完成（mot 文件存在）。
    输出到 result/{experiment_label}/opensim/inverse_dynamics/{load_key}/
    Step 2 已生成的外力文件会被直接复用。

    Parameters
    ----------
    output_body_forces : bool
        若为 True，同时输出各关节处的体力/力矩
        ({label}_body_forces_at_joints.sto)。默认 False。
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

    mot_files = get_mot_files(config, base_dir)
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
                log(f'  [EXT] 复用已有外力文件: {xml_candidate}')
                ext_file = xml_candidate
            else:
                ext_file = generate_external_loads(
                    config=config, base_dir=base_dir,
                    load_key=load_key, mot_path=mot_path,
                    Mb=Mb, verbose=verbose)
            if ext_file is None:
                log('  [WARN] 外力生成失败，此负载将不包含外力')

        run_inverse_dynamics(
            model_path=scaled_model, mot_path=mot_path,
            output_dir=id_dir, external_load_file=ext_file,
            label=f'{experiment_label}_{load_key}',
            output_body_forces=output_body_forces,
            verbose=verbose)

    log('\n[Step 3] 逆向动力学完成')
    log(f'输出目录: {opensim_dir}/inverse_dynamics/')