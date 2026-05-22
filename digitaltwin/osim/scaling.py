"""
opensim_scaling.py

OpenSim 模型全身缩放工具。

主要接口：
    scale_full_body_model(model_path, target_lengths, output_path,
                          joint_map=None, adjust_mass=True, verbose=True)

设计原则：
  - DEFAULT_JOINT_MAP 定义可直接测量的体段（具有明确连接关节对）。
  - DEFAULT_INHERIT_MAP 定义所有需要继承缩放因子的体段。
  - ulna / talus / patella / clavicle / scapula / lumbar 等均继承相关体段，
    不单独测量（避免因关节名不匹配而产生错误缩放因子）。
"""
import os
import shutil
import numpy as np
import opensim as osim


# ============================================================
#  默认关节映射表
#  格式：  body_name: ([近端关节候选名], [远端关节候选名])
#  僅包含可直接测量的体段，其余体段内置到 DEFAULT_INHERIT_MAP。
# ============================================================
DEFAULT_JOINT_MAP = {
    # --- 下肢 ---
    'femur_r':   (['hip_r'],                    ['knee_r', 'walker_knee_r']),
    'femur_l':   (['hip_l'],                    ['knee_l', 'walker_knee_l']),
    'tibia_r':   (['knee_r', 'walker_knee_r'],  ['ankle_r', 'subtalar_r']),
    'tibia_l':   (['knee_l', 'walker_knee_l'],  ['ankle_l', 'subtalar_l']),
    'calcn_r':   (['subtalar_r', 'ankle_r'],    ['mtp_r']),
    'calcn_l':   (['subtalar_l', 'ankle_l'],    ['mtp_l']),
    # --- 骨盆 (宽度) ---
    'pelvis':    (['hip_r'],                    ['hip_l']),
    # --- 躯干
    #  候选名覆盖多种常见模型命名规范。
    #  如果均失败，将自动扫描模型中所有关节并打印提示。
    # torso：近端 = 躯干底部关节，远端 = 肩关节（模型无颈/头关节时用肩关节代替）
    # 已加入该模型的实际关节名 Torsojnt / L1_L2_IVD_jnt 以及常见备选名
    'torso':     (['Torsojnt', 'L1_L2_IVD_jnt', 'L5_S1_IVDjnt',
                   'back', 'lumbar', 'lumbosacral', 'spine', 'back_body',
                   'thoracolumbar', 'L5_S1', 'vertebra_lumbar_body',
                   'lumbar5_torso', 'L5', 'spinal'],
                  ['acromial_r', 'acromial_l', 'sc1',
                   'neck', 'neck_body', 'cervical', 'thorax_neck',
                   'neck1', 'head', 'skull']),
    # --- 上肢
    #  注意： ulna 不单独测量，始终继承 radius 因子。
    'humerus_r': (['shoulder_r', 'acromial_r', 'unrothum_r'],  ['elbow_r']),
    'humerus_l': (['shoulder_l', 'acromial_l', 'unrothum_l'],  ['elbow_l']),
    'radius_r':  (['elbow_r'],  ['wrist_r', 'radius_hand_r']),
    'radius_l':  (['elbow_l'],  ['wrist_l', 'radius_hand_l']),
}


# ============================================================
#  默认继承映射表
#  格式：  body_name: donor_body
#  body_name 体段的缩放因子继承自 donor_body。
# ============================================================
DEFAULT_INHERIT_MAP = {
    # 下肢
    'talus_r':      'tibia_r',     # 距骨 → 小腿
    'talus_l':      'tibia_l',
    'toes_r':       'calcn_r',     # 足趧 → 跟骨
    'toes_l':       'calcn_l',
    'patella_r':    'femur_r',     # 膌盖骨 → 大腿
    'patella_l':    'femur_l',
    # 骨盆复合
    'sacrum':       'pelvis',      # 骨盆 → 骨盆
    # 躯干 / 腾點
    'Abdomen':      'torso',
    'LB_wrap':      'torso',
    'lumbar1':      'torso',
    'lumbar2':      'torso',
    'lumbar3':      'torso',
    'lumbar4':      'torso',
    'lumbar5':      'torso',
    # 上肢： ulna 始终与 radius 共用同一缩放因子
    'ulna_r':       'radius_r',
    'ulna_l':       'radius_l',
    'hand_r':       'radius_r',
    'hand_l':       'radius_l',
    # 肩胛 / 肩胛胛 (缩放因子同上臂)
    'clavicle_r':   'humerus_r',
    'clavicle_l':   'humerus_l',
    'clavicle_2':   'humerus_r',
    'clavicle_l_1': 'humerus_l',
    'clavicle_l_2': 'humerus_l',
    'scapula_r':    'humerus_r',
    'scapula_l':    'humerus_l',
    'scapula_1':    'humerus_r',
    'scapula_2':    'humerus_r',
    'scapula_l_1':  'humerus_l',
    'scapula_l_2':  'humerus_l',
}


# ============================================================
#  底层工具
# ============================================================

def vec3_to_np(v):
    """OpenSim Vec3 → numpy"""
    return np.array([v.get(i) for i in range(3)], dtype=float)


def _try_get_joint(model, names):
    """\u4ece候选名列表中查找第一个存在的关节。"""
    js = model.getJointSet()
    for name in names:
        try:
            j = js.get(name)
            if j is not None:
                return j, name
        except Exception:
            pass
    return None, None


def get_joint_position(model, state, joint_names):
    """
    获取关节 parent frame 在 ground 中的位置。
    joint_names 可以是字符串或字符串列表。
    """
    if isinstance(joint_names, str):
        joint_names = [joint_names]
    joint, found_name = _try_get_joint(model, joint_names)
    if joint is None:
        return None, None
    pos = vec3_to_np(
        joint.getParentFrame().getTransformInGround(state).p())
    return pos, found_name


def compute_segment_length(model, state, proximal_names, distal_names):
    """
    计算两个关节位置之间的距离。
    返回 (length, prox_joint_name, dist_joint_name)。
    """
    prox_pos, prox_name = get_joint_position(model, state, proximal_names)
    dist_pos, dist_name = get_joint_position(model, state, distal_names)
    if prox_pos is None or dist_pos is None:
        return None, prox_name, dist_name
    return float(np.linalg.norm(dist_pos - prox_pos)), prox_name, dist_name


def scan_torso_joints(model, state, verbose=True):
    """
    自动扫描模型关节，找到与 torso 体相连的关节对，
    返回最大距离的两个关节名称。

    策略：
      1. 尝试通过 getParentBodyName() / getChildBodyName() 找 torso 关节（更可靠）。
      2. 退回到 getParentFrame().getBody().getName() / getChildFrame().getBody()。
      3. 若仍 < 2 个，打印所有关节名并返回 (None, None)。
    """
    js = model.getJointSet()
    torso_joints = []  # (joint_name, pos_in_ground)

    for i in range(js.getSize()):
        try:
            j = js.get(i)
            pb, cb = '', ''
            # 优先用 getParentBodyName / getChildBodyName（OpenSim 4.x API）
            try:
                pb = j.getParentBodyName()
            except Exception:
                pass
            try:
                cb = j.getChildBodyName()
            except Exception:
                pass
            # 退回方案
            if not pb:
                try:
                    pb = j.getParentFrame().getBody().getName()
                except Exception:
                    pass
            if not cb:
                try:
                    cb = j.getChildFrame().getBody().getName()
                except Exception:
                    pass

            if 'torso' in pb.lower() or 'torso' in cb.lower():
                pos = vec3_to_np(
                    j.getParentFrame().getTransformInGround(state).p())
                torso_joints.append((j.getName(), pos))
        except Exception:
            pass

    if len(torso_joints) < 2:
        if verbose:
            print('  [WARN] 未找到足够的 torso 关节，模型中所有关节如下:')
            for i in range(js.getSize()):
                try:
                    jname = js.get(i).getName()
                    print(f'    {jname}')
                except Exception:
                    pass
        return None, None

    # 取空间中最远的两个关节
    best_dist = -1
    best_pair = (torso_joints[0][0], torso_joints[-1][0])
    for i in range(len(torso_joints)):
        for k in range(i + 1, len(torso_joints)):
            d = float(np.linalg.norm(
                torso_joints[k][1] - torso_joints[i][1]))
            if d > best_dist:
                best_dist = d
                best_pair = (torso_joints[i][0], torso_joints[k][0])

    if verbose:
        print(f'  [AUTO] torso 关节扫描: '
              f'[{best_pair[0]}] → [{best_pair[1]}] '
              f'= {best_dist:.4f}m')
    return best_pair


# ============================================================
#  全身缩放
# ============================================================

def scale_full_body_model(model_path, target_lengths, output_path,
                          joint_map=None, inherit_map=None,
                          subject_mass=None, verbose=True):
    """
    以目标节段长度对 OpenSim 全身模型进行缩放。

    Parameters
    ----------
    model_path : str
        原始 .osim 模型文件路径。
    target_lengths : dict
        {body_name: target_length_in_meters}
        支持的 body_name 见 DEFAULT_JOINT_MAP 键名。
    output_path : str
        缩放后模型保存路径。
    joint_map : dict, optional
        自定义关节映射，覆盖默认表中对应条目。
    inherit_map : dict, optional
        自定义继承映射，覆盖默认表中对应条目。
    subject_mass : float, optional
        受试者实测体重（kg）。若提供，则在几何缩放完成后，
        按比例将模型总质量缩放到该值（每个 Body 的质量乘以
        subject_mass / 模型总质量）。若为 None，则不调整质量。
    verbose : bool
        是否打印详细日志。

    Returns
    -------
    model 或 None（失败时）
    """
    def log(msg):
        if verbose:
            print(msg)

    # 合并映射表
    full_jmap    = dict(DEFAULT_JOINT_MAP)
    full_inhmap  = dict(DEFAULT_INHERIT_MAP)
    if joint_map:
        full_jmap.update(joint_map)
    if inherit_map:
        full_inhmap.update(inherit_map)

    # ---- 1. 加载模型 ----
    log('\n[1/6] 加载模型...')
    model = osim.Model(model_path)
    state = model.initSystem()
    log(f'  模型: {model_path}')

    # ---- 2. 计算当前长度 & 缩放因子 ----
    log('\n[2/6] 计算当前节段长度并生成缩放因子...')
    segment_scales = {}   # body_name -> scale_factor
    length_info    = {}   # body_name -> (cur, tgt, scale, p_jnt, d_jnt)
    failed_bodies  = []

    for body_name, target_len in target_lengths.items():
        if body_name not in full_jmap:
            log(f'  [SKIP] {body_name}: 不在关节映射表中')
            failed_bodies.append(body_name)
            continue

        prox_names, dist_names = full_jmap[body_name]

        # torso 特殊处理：先尝试预设关节名，失败则自动扫描
        if body_name == 'torso' and not dist_names:
            # dist_names 为空列表 — 不可能发生，但保安
            pass

        cur_len, p_jnt, d_jnt = compute_segment_length(
            model, state, prox_names, dist_names)

        # torso 备用方案：自动扫描
        if (cur_len is None or cur_len < 1e-6) and body_name == 'torso':
            auto_p, auto_d = scan_torso_joints(model, state, verbose=verbose)
            if auto_p and auto_d:
                cur_len, p_jnt, d_jnt = compute_segment_length(
                    model, state, [auto_p], [auto_d])

        if cur_len is None or cur_len < 1e-6:
            log(f'  [WARN] {body_name}: 无法计算当前长度 '
                f'(尝试关节: {prox_names} → {dist_names})')
            failed_bodies.append(body_name)
            continue

        scale = target_len / cur_len
        segment_scales[body_name] = scale
        length_info[body_name]    = (cur_len, target_len, scale, p_jnt, d_jnt)
        log(f'  {body_name:22s}: {cur_len:.4f}m → {target_len:.4f}m '
            f'(scale={scale:.4f}, [{p_jnt}] -> [{d_jnt}])')

    # ---- 3. 处理继承体段 ----
    log('\n[3/6] 处理继承体段...')
    # 外部传入的 target_lengths 中可能包含继承体段，优先用
    for body_name, donor_body in full_inhmap.items():
        if body_name in segment_scales:
            continue  # 已经直接计算了
        # 扫描 donor 链得到因子
        visited = set()
        donor = donor_body
        while donor and donor not in visited:
            if donor in segment_scales:
                segment_scales[body_name] = segment_scales[donor]
                log(f'  {body_name:22s}: 继承 [{donor}] 因子 = {segment_scales[donor]:.4f}')
                break
            visited.add(donor)
            donor = full_inhmap.get(donor)

    if not segment_scales:
        log('[ERROR] 没有任何可用缩放因子，终止。')
        return None

    # ---- 4. 构建 ScaleSet 并应用缩放 ----
    log('\n[4/6] 应用缩放...')
    scale_set = osim.ScaleSet()
    body_set  = model.getBodySet()
    for body_name, factor in segment_scales.items():
        if not body_set.contains(body_name):
            log(f'  [SKIP] {body_name}: 模型中不存在此体段')
            continue
        seg = osim.Scale()
        seg.setSegmentName(body_name)
        seg.setScaleFactors(osim.Vec3(factor, factor, factor))
        scale_set.adoptAndAppend(seg)
        log(f'  {body_name:22s}: scale = {factor:.6f}')

    try:
        model.scale(state, scale_set, False)
        log('  ScaleSet 缩放成功!')
    except Exception as e:
        log(f'  [ERROR] ScaleSet 缩放失败: {e}')
        return None

    # ---- 5. 调整质量（按受试者实测体重整体缩放）----
    if subject_mass is not None:
        log(f'\n[5/6] 按受试者体重 {subject_mass:.2f} kg 调整各体段质量...')
        # 计算几何缩放后的模型总质量
        state2 = model.initSystem()
        total_model_mass = 0.0
        for i in range(body_set.getSize()):
            try:
                total_model_mass += body_set.get(i).getMass()
            except Exception:
                pass
        if total_model_mass < 1e-6:
            log('  [WARN] 模型总质量为 0，跳过质量调整')
        else:
            mass_ratio = subject_mass / total_model_mass
            log(f'  缩放前模型总质量: {total_model_mass:.2f} kg')
            log(f'  质量缩放比例    : {mass_ratio:.6f}')
            for i in range(body_set.getSize()):
                try:
                    body = body_set.get(i)
                    orig = body.getMass()
                    new_mass = orig * mass_ratio
                    body.setMass(new_mass)
                    log(f'  {body.getName():22s}: {orig:.4f} -> {new_mass:.4f} kg')
                except Exception as e:
                    log(f'  [WARN] 体段质量调整失败: {e}')
            log(f'  调整后模型总质量: {subject_mass:.2f} kg')
    else:
        log('\n[5/6] 未提供受试者体重，跳过质量调整')

    # ---- 6. 保存 ----
    log('\n[6/6] 保存缩放后模型...')
    model.finalizeConnections()
    state_final = model.initSystem()
    model.printToXML(output_path)
    log(f'  已保存: {output_path}')

    # ---- 7. 验证 + 打印身高体重 ----
    log('\n[7/6] 验证缩放结果...')
    log(f'  {"Body":24s}  {"Current":>8}  {"Target":>8}  {"Error%":>8}')
    log('  ' + '-' * 56)
    all_ok = True
    for body_name, (cur, tgt, sc, pj, dj) in sorted(length_info.items()):
        prox_names, dist_names = full_jmap[body_name]
        new_len, _, _ = compute_segment_length(
            model, state_final, prox_names, dist_names)
        if new_len is None:
            log(f'  {body_name:24s}  无法验证')
            continue
        err = abs(new_len - tgt) / tgt * 100
        flag = '' if err < 2.0 else ' !!'
        log(f'  {body_name:24s}  {new_len:8.4f}  {tgt:8.4f}  {err:7.2f}%{flag}')
        if err >= 2.0:
            all_ok = False

    log('\n  缩放全部成功！' if all_ok
        else '\n  部分体段误差较大，请检查关节映射表。')
    if failed_bodies:
        log(f'  未处理的体段: {failed_bodies}')

    # ---- 打印整体身高与总体重 ----
    log('')
    _print_height_mass(model, state_final, log)

    return model


# ============================================================
#  高层接口：从 JSON 配置文件驱动缩放
# ============================================================

def scale_from_config(config, base_dir, verbose=True):
    """
    从 JSON 配置字典驱动全身缩放。

    流程：
      1. 取 modeling_file.data 中第一个 xsens 文件 → 提取节段长度 → 缩放模型。
      2. 取第二个 xsens 文件（若存在）进行交叉验证。

    Parameters
    ----------
    config : dict
        解析后的 JSON 内容。需包含字段：
            experiment_label, folder, modeling_file
        可选字段：
            opensim_settings.subject_mass  (受试者实测体重 kg)
    base_dir : str
        包含 workspace/ 和 result/ 的基准目录，通常为 example 脚本位置向上两级。
    verbose : bool
        是否打印详细日志。

    Returns
    -------
    model 或 None
    """
    from digitaltwin.data.xsens_processor import XsensProcessor

    def log(msg):
        if verbose:
            print(msg)

    # --- 读取配置 ---
    experiment_label = config['experiment_label']
    folder           = config['folder']   # Xsens Excel 和 Robot 文件根目录
    osim_cfg         = config.get('opensim_settings', {})
    subject_mass     = osim_cfg.get('subject_mass', None)

    modeling  = config['modeling_file']
    xsens_dir = os.path.join(folder, modeling.get('xsens_folder', 'xsens'))
    data      = modeling['data']

    # 按 JSON 键顺序收集所有 xsens 文件路径
    xsens_files = [
        os.path.join(xsens_dir, v['xsens_file'])
        for v in data.values()
        if 'xsens_file' in v
    ]
    if not xsens_files:
        log('[ERROR] modeling_file.data 中未找到任何 xsens_file。')
        return None

    xsens_file_1 = xsens_files[0]                               # 用于缩放
    xsens_file_2 = xsens_files[1] if len(xsens_files) > 1 else None  # 用于交叉验证

    # --- 路径 ---
    input_model = os.path.join(base_dir, 'workspace', 'whole body model.osim')
    output_dir  = os.path.join(base_dir, 'result', experiment_label, 'opensim')
    os.makedirs(output_dir, exist_ok=True)
    output_model = os.path.join(
        output_dir, f'whole body model_{experiment_label}.osim')

    # 将 workspace/Geometry 文件夹复制到输出目录，
    # OpenSim 在加载模型时需要该文件夹中的网格文件。
    geometry_src = os.path.join(base_dir, 'workspace', 'Geometry')
    geometry_dst = os.path.join(output_dir, 'Geometry')
    if os.path.isdir(geometry_src):
        if os.path.isdir(geometry_dst):
            log(f'  Geometry 已存在，跳过复制: {geometry_dst}')
        else:
            shutil.copytree(geometry_src, geometry_dst)
            log(f'  已复制 Geometry: {geometry_dst}')
    else:
        log(f'  [WARN] 未找到 workspace/Geometry 文件夹，如模型加载失败请手动复制')

    log('\n' + '=' * 60)
    log(f'实验标签 : {experiment_label}')
    log(f'输入模型 : {input_model}')
    log(f'输出模型 : {output_model}')
    log(f'缩放来源 : {os.path.basename(xsens_file_1)}')
    if xsens_file_2:
        log(f'交叉验证 : {os.path.basename(xsens_file_2)}')
    if subject_mass is not None:
        log(f'受试者体重: {subject_mass} kg')
    log('=' * 60)

    # --- Step 1: 提取 Xsens 节段长度 ---
    log('\n[节段提取] 读取缩放用 Xsens 文件...')
    measurements_1   = XsensProcessor.extract_segment_measurements(xsens_file_1, verbose=verbose)
    target_lengths   = XsensProcessor.segment_measurements_to_opensim_targets(measurements_1)
    XsensProcessor.print_segment_measurements(measurements_1)
    XsensProcessor.print_opensim_targets_table(measurements_1)

    # --- Step 2: 缩放模型 ---
    model = scale_full_body_model(
        model_path=input_model,
        target_lengths=target_lengths,
        output_path=output_model,
        subject_mass=subject_mass,
        verbose=verbose,
    )

    # --- Step 3: 交叉验证 ---
    if model is not None and xsens_file_2:
        log(f'\n[交叉验证] 使用第二个 Xsens 文件验证...')
        measurements_2 = XsensProcessor.extract_segment_measurements(xsens_file_2, verbose=verbose)
        targets_2      = XsensProcessor.segment_measurements_to_opensim_targets(measurements_2)

        full_jmap = dict(DEFAULT_JOINT_MAP)
        state_v   = model.initSystem()

        log(f'  {"Body":24s}  {"Model (scaled)":>14}  '
            f'{"Xsens file 2":>14}  {"Error%":>8}')
        log('  ' + '-' * 68)
        all_ok = True
        for body_name, tgt in sorted(targets_2.items()):
            if body_name not in full_jmap:
                continue
            prox_names, dist_names = full_jmap[body_name]
            cur_len, _, _ = compute_segment_length(
                model, state_v, prox_names, dist_names)
            if cur_len is None:
                continue
            err  = abs(cur_len - tgt) / tgt * 100
            flag = '' if err < 5.0 else ' !!'
            log(f'  {body_name:24s}  {cur_len:14.4f}  {tgt:14.4f}  {err:7.2f}%{flag}')
            if err >= 5.0:
                all_ok = False
        log('  交叉验证通过！' if all_ok
            else '  部分体段差异较大（可能来自不同文件姿态差异）')

    # ---- 计算深蹲杆接触点（torso 局部坐标）----
    log('\n[杆接触点] 计算深蹲杆作用点...')
    state_for_contact = model.initSystem()
    bar_contact_point = compute_bar_contact_point(model, state_for_contact, verbose=verbose)
    log(f'  bar_contact_point (torso 局部坐标): {bar_contact_point}')

    # ---- 计算鞋垫接触点（calcn 局部坐标）----
    log('\n[鞋垫接触点] 计算足底接触点...')
    insole_contact_point_r = compute_insole_contact_point(model, state_for_contact, side='r', verbose=verbose)
    insole_contact_point_l = compute_insole_contact_point(model, state_for_contact, side='l', verbose=verbose)
    # 左右脚取平均（两脚通常对称）
    insole_contact_point = [
        round((insole_contact_point_r[i] + insole_contact_point_l[i]) / 2, 4)
        for i in range(3)
    ]
    log(f'  insole_contact_point (calcn 局部坐标，左右平均): {insole_contact_point}')

    return model, bar_contact_point, insole_contact_point


# ============================================================
#  深蹲杆接触点计算
# ============================================================

def compute_bar_contact_point(model, state, verbose=True):
    """
    从缩放后的模型估算深蹲背部杆接触点（torso 局部坐标）。

    算法：
      1. 将肩关节 acromial_r 在 ground 坐标系中的位置转换到 torso 局部坐标系。
      2. 接触点高度 ≈ 肩关节在 torso 局部坐标系中的 y 坐标。
      3. 前后偏移取肩关节 torso 局部 z 偏移再向后一小段（-0.04 m）。
      4. 左右居中：取 x=0。

    Parameters
    ----------
    model  : osim.Model  -- 已 initSystem 的缩放后模型
    state  : osim.State  -- 模型对应状态
    verbose : bool

    Returns
    -------
    list [x, y, z]  单位 m， torso 局部坐标系
    """
    def log(msg):
        if verbose:
            print(msg)

    try:
        ground     = model.getGround()
        torso_body = model.getBodySet().get('torso')

        # 尝试获取肩关节位置（ground 坐标系）
        shoulder_pos_ground = None
        for jname in ['acromial_r', 'acromial_l', 'sc1', 'sc2']:
            pos, _ = get_joint_position(model, state, [jname])
            if pos is not None:
                shoulder_pos_ground = pos
                log(f'  [BAR] 参考关节: {jname}  ground位置 = {pos}')
                break

        if shoulder_pos_ground is None:
            log('  [BAR] 未找到肩关节，将使用默认接触点 [0.0, 0.30, -0.07]')
            return [0.0, 0.30, -0.07]

        # 将肩关节 ground 点转换到 torso 局部坐标系
        p_g = osim.Vec3(*shoulder_pos_ground)
        p_local = ground.findStationLocationInAnotherFrame(state, p_g, torso_body)
        lx = p_local.get(0)
        ly = p_local.get(1)
        lz = p_local.get(2)
        log(f'  [BAR] 肩关节 torso局部坐标 = [{lx:.4f}, {ly:.4f}, {lz:.4f}]')

        # OpenSim 全局坐标系：X 前方，Y 上方，Z 右方
        # torso 局部坐标系直立时通常与全局坐标系对齐
        # 深蹲杆在斜方肌上：
        #   x ≈ 肩关节局部 x - 0.04 m（杆在肩胛后方，封X为前方故减小）
        #   y ≈ 肩关节局部 y（杆弄在肖部高度）
        #   z = 0（Z 为左右方向，居中为 0）
        contact = [round(float(lx) - 0.04, 4),
                   round(float(ly), 4),
                   round(0.0, 4)]
        log(f'  [BAR] 计算接触点 (torso局部) = {contact}')
        log('  [BAR]   x = 肩关节局部x - 0.04（向后偏移，X前方故减小）')
        log('  [BAR]   y = 肩关节局部y（肖部高度）')
        log('  [BAR]   z = 0（Z 为左右方向，杆居中）')
        return contact

    except Exception as e:
        log(f'  [BAR] 接触点计算失败: {e}，使用默认值')
        return [0.0, 0.30, -0.07]


# ============================================================
#  身高 / 体重 统计工具
# ============================================================

def _print_height_mass(model, state, log_fn):
    """
    估算模型整体身高并打印总质量。

    身高估算策略：
      1. 取足底最低 Y 坐标（mtp 或 calcn 关节位置），
         作为地面参考零点。
      2. 取肩关节 (acromial) 位置的最高 Y 坐标，
         加上固定头部估算高度 (0.24m) 作为顶部。
      3. 若找不到以上关节，退回到骨盆 ×2 的粗估。
    总质量：遍历所有 Body 求和（与 model.getTotalMass() 等价）。
    """
    try:
        # --- 总质量 ---
        bs = model.getBodySet()
        total_mass = 0.0
        for i in range(bs.getSize()):
            try:
                total_mass += bs.get(i).getMass()
            except Exception:
                pass

        # --- 足底 Y（最低点） ---
        foot_y = None
        for jname in ['mtp_r', 'mtp_l', 'subtalar_r', 'subtalar_l',
                      'ankle_r', 'ankle_l']:
            pos, _ = get_joint_position(model, state, [jname])
            if pos is not None:
                y = float(pos[1])
                if foot_y is None or y < foot_y:
                    foot_y = y
        if foot_y is None:
            foot_y = 0.0

        # --- 肩关节 Y（最高测量点） ---
        shoulder_y = None
        for jname in ['acromial_r', 'acromial_l', 'sc1', 'sc2',
                      'sc1_l', 'sc2_l']:
            pos, _ = get_joint_position(model, state, [jname])
            if pos is not None:
                y = float(pos[1])
                if shoulder_y is None or y > shoulder_y:
                    shoulder_y = y

        HEAD_NECK_EST = 0.24   # 肩→头顶估算（米）

        if shoulder_y is not None:
            height = (shoulder_y - foot_y) + HEAD_NECK_EST
            log_fn(f'  模型估算身高 : {height:.3f} m  '
                   f'(肩高={shoulder_y:.3f}m, 足底={foot_y:.3f}m, '
                   f'+头颈估算={HEAD_NECK_EST}m)')
        else:
            # 退回方案：骨盆高度 × 2
            pelvis_pos, _ = get_joint_position(model, state, ['ground_pelvis', 'hip_r'])
            if pelvis_pos is not None:
                height = float(pelvis_pos[1]) * 2.0
                log_fn(f'  模型估算身高 : {height:.3f} m  (骨盆高度×2 粗估)')
            else:
                log_fn('  模型估算身高 : 无法计算（未找到参考关节）')

        log_fn(f'  模型总质量   : {total_mass:.2f} kg')
    except Exception as e:
        log_fn(f'  [WARN] 身高/体重计算失败: {e}')


# ============================================================
#  Xsens 测量转换工具
# ============================================================

def xsens_to_opensim_targets(measurements):
    """
    将 Xsens 节段长度字典转换为 OpenSim target_lengths 字典。

    Parameters
    ----------
    measurements : dict
        键名如 'Right Thigh Length'，值为长度列表或标量（米）。

    Returns
    -------
    dict  {opensim_body_name: target_length_m}
    """
    def mean(key):
        v = measurements.get(key, [])
        if hasattr(v, '__len__') and len(v) > 0:
            return float(np.mean(v))
        if isinstance(v, (int, float)):
            return float(v)
        return None

    targets = {}

    def _set(key, val):
        if val is not None and np.isfinite(val):
            targets[key] = val

    # 下肢
    _set('femur_r',  mean('Right Thigh Length'))
    _set('femur_l',  mean('Left Thigh Length'))
    _set('tibia_r',  mean('Right Shank Length'))
    _set('tibia_l',  mean('Left Shank Length'))
    _set('calcn_r',  mean('Right Foot Length'))
    _set('calcn_l',  mean('Left Foot Length'))
    # 骨盆
    _set('pelvis',   mean('Pelvis Width'))
    # 躯干
    spine_keys = ['Pelvis to L5 Length', 'L5 to T8 Length', 'T8 to Neck Length']
    spine_vals = [mean(k) for k in spine_keys]
    if all(v is not None for v in spine_vals):
        _set('torso', sum(spine_vals))
    # 上肢
    _set('humerus_r', mean('Right Upper Arm Length'))
    _set('humerus_l', mean('Left Upper Arm Length'))
    _set('radius_r',  mean('Right Forearm Length'))
    _set('radius_l',  mean('Left Forearm Length'))

    return targets


# ============================================================
#  小工具：列出模型中所有关节名称（调试用）
# ============================================================

def list_model_joints(model_path):
    """加载模型并打印所有关节名称，用于调试 joint_map。"""
    model = osim.Model(model_path)
    model.initSystem()
    js = model.getJointSet()
    print(f'\n模型中共 {js.getSize()} 个关节:')
    for i in range(js.getSize()):
        try:
            j = js.get(i)
            try:
                pb = j.getParentFrame().getBody().getName()
            except Exception:
                pb = '?'
            try:
                cb = j.getChildFrame().getBody().getName()
            except Exception:
                cb = '?'
            print(f'  [{i:2d}] {j.getName():35s}  '
                  f'parent_body={pb:15s}  child_body={cb}')
        except Exception:
            pass


def list_model_bodies(model_path):
    """加载模型并打印所有体段名称。"""
    model = osim.Model(model_path)
    bs = model.getBodySet()
    print(f'\n模型中共 {bs.getSize()} 个体段:')
    for i in range(bs.getSize()):
        try:
            print(f'  {bs.get(i).getName()}')
        except Exception:
            pass


# ============================================================
#  鞋垫接触点计算
# ============================================================

def compute_insole_contact_point(model, state, side='r', verbose=True):
    """
    从缩放后的模型估算鞋垫接触点（calcn 局部坐标）。

    算法：
      取 mtp 关节在 calcn 局部坐标系中的位置，
      接触点 = mtp 局部坐标 × 0.5（即足弓中部）。

    Parameters
    ----------
    model  : osim.Model
    state  : osim.State
    side   : 'r' or 'l'
    verbose : bool

    Returns
    -------
    list [x, y, z]  单位 m，calcn 局部坐标系
    """
    def log(msg):
        if verbose:
            print(msg)

    try:
        ground     = model.getGround()
        calcn_body = model.getBodySet().get(f'calcn_{side}')

        mtp_pos_ground, mtp_name = get_joint_position(
            model, state, [f'mtp_{side}'])
        if mtp_pos_ground is None:
            log(f'  [INSOLE] 未找到 mtp_{side}，使用默认接触点 [0.09, 0.0, 0.0]')
            return [0.09, 0.0, 0.0]

        # 将 mtp 关节转换到 calcn 局部坐标系
        p_g   = osim.Vec3(*mtp_pos_ground)
        p_loc = ground.findStationLocationInAnotherFrame(state, p_g, calcn_body)
        lx = round(float(p_loc.get(0)) * 0.5, 4)
        ly = round(float(p_loc.get(1)) * 0.5, 4)
        lz = round(float(p_loc.get(2)) * 0.5, 4)

        log(f'  [INSOLE] mtp_{side} calcn局部坐标 = '
            f'[{p_loc.get(0):.4f}, {p_loc.get(1):.4f}, {p_loc.get(2):.4f}]')
        log(f'  [INSOLE] insole_contact_point (calcn_{side} 局部×0.5) = [{lx}, {ly}, {lz}]')
        return [lx, ly, lz]
    except Exception as e:
        log(f'  [INSOLE] 接触点计算失败: {e}，使用默认值')
        return [0.09, 0.0, 0.0]


# ============================================================
#  旧版兼容
# ============================================================

def compute_limb_lengths(model, state):
    """[旧版兑容]"""
    pos = {}
    for jname in ['acromial_r', 'elbow_r', 'radius_hand_r']:
        p, _ = get_joint_position(model, state, [jname])
        pos[jname] = p
    sp = pos.get('acromial_r')
    ep = pos.get('elbow_r')
    wp = pos.get('radius_hand_r')
    if None in (sp, ep, wp):
        return None, None, (sp, ep, wp)
    return (float(np.linalg.norm(ep - sp)),
            float(np.linalg.norm(wp - ep)),
            (sp, ep, wp))


def scale_opensim_model(model_path, target_uarm_length, target_farm_length,
                        output_path):
    """[旧版兑容] 仅缩放上臂+前臂。"""
    return scale_full_body_model(
        model_path=model_path,
        target_lengths={
            'humerus_r': target_uarm_length,
            'humerus_l': target_uarm_length,
            'radius_r':  target_farm_length,
            'radius_l':  target_farm_length,
        },
        output_path=output_path,
    )


if __name__ == '__main__':
    # 调试输出模型中所有关节 (如需查看 torso 关节名称可取注失败这行)
    MODEL = 'E:/VSCode/Projects/OpenSimRealtimeDisplay/Model/whole body model_DisplayOnly/model.osim'
    list_model_joints(MODEL)
    list_model_bodies(MODEL)