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
#  Xsens Excel -> .mot 转换核心函数
# ============================================================

def read_xsens_excel_for_opensim(excel_path, output_mot_path=None):
    """
    从 Xsens 导出的 Excel 文件读取数据，生成 OpenSim 可用的 .mot 文件。
    arm_flex / arm_add 左右共四个关节角仍需调整。

    Parameters
    ----------
    excel_path      : str -- Xsens 导出的 .xlsx 文件路径
    output_mot_path : str, optional -- 输出路径

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
    pq = np.column_stack([
        df_quat['Pelvis q1'].values[:n_samples],
        df_quat['Pelvis q2'].values[:n_samples],
        df_quat['Pelvis q3'].values[:n_samples],
        df_quat['Pelvis q0'].values[:n_samples],
    ])
    euler_zxy    = R.from_quat(pq).as_euler('zxy', degrees=True)
    pelvis_rotation = euler_zxy[:, 0]
    pelvis_list     = euler_zxy[:, 1]
    pelvis_tilt     = euler_zxy[:, 2]

    # 4. 骨盆位置 (Xsens x,y,z -> OpenSim x,y,z)
    pelvis_tx = df_pos['Pelvis x'].values[:n_samples]
    pelvis_ty = df_pos['Pelvis z'].values[:n_samples]   # Xsens z(上) -> OpenSim y(上)
    pelvis_tz = df_pos['Pelvis y'].values[:n_samples]   # Xsens y(右) -> OpenSim z(右)

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

    # 7. 符号调整
    sign_flip = {
        'pelvis_tilt': -1, 'pelvis_tz': -1,
        'hip_adduction_r': -1, 'hip_adduction_l': -1,
        'lumbar_extension': -1, 'arm_flex_r': -1, 'arm_flex_l': -1,
    }
    for jname, sign in sign_flip.items():
        if jname in joint_names_order:
            angles_matrix[:, joint_names_order.index(jname)] *= sign

    data_with_time = np.column_stack([time, angles_matrix])

    # 8. 保存 .mot
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
        data_min = np.min(data_with_time[:, 1:])
        data_max = np.max(data_with_time[:, 1:])
        f.write(f'range {data_min:.4f} {data_max:.4f}\n')
        f.write('endheader\n')
        f.write('time\t' + '\t'.join(joint_names_order) + '\n')
        for row in data_with_time:
            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')
    print(f'已保存: {output_mot_path}  {data_with_time.shape}')
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