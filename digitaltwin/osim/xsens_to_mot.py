import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def read_xsens_excel_for_opensim(excel_path: str, output_mot_path: str = None):
    """
    从 Xsens 导出的 Excel 文件读取数据，生成 OpenSim 可用的 .mot 文件
    arm_flex, arm_add 左右共四个关节角仍需调整
    """

    # ==================== 1. 读取元数据获取帧率 ====================
    df_info = pd.read_excel(excel_path, sheet_name='General Information', header=None)
    frame_rate = None
    for i, row in df_info.iterrows():
        if row[0] == 'Frame Rate':
            frame_rate = float(row[1])
            break

    if frame_rate is None:
        print("警告: 未找到 Frame Rate，使用默认值 60 Hz")
        frame_rate = 60

    print(f"采样率: {frame_rate} Hz")

    # ==================== 2. 读取数据并计算时间 ====================
    df_pos = pd.read_excel(excel_path, sheet_name='Segment Position')
    frame_indices = df_pos['Frame'].values
    n_samples = len(frame_indices)

    # 正确的时间计算: 帧索引 / 帧率
    time = frame_indices / frame_rate

    print(f"帧数: {n_samples}")
    print(f"时间范围: {time[0]:.3f} - {time[-1]:.3f} 秒")
    print(f"持续时间: {time[-1] - time[0]:.3f} 秒")

    # ==================== 3. 骨盆角度 (从四元数计算) ====================
    df_quat = pd.read_excel(excel_path, sheet_name='Segment Orientation - Quat')

    pelvis_q0 = df_quat['Pelvis q0'].values[:n_samples]
    pelvis_q1 = df_quat['Pelvis q1'].values[:n_samples]
    pelvis_q2 = df_quat['Pelvis q2'].values[:n_samples]
    pelvis_q3 = df_quat['Pelvis q3'].values[:n_samples]

    # 四元数转欧拉角 (ZXY 顺序)
    r = R.from_quat(np.column_stack([pelvis_q1, pelvis_q2, pelvis_q3, pelvis_q0]))
    euler_zxy = r.as_euler('zxy', degrees=True)

    pelvis_rotation = euler_zxy[:, 0]  # Z: 轴向旋转
    pelvis_list = euler_zxy[:, 1]  # X: 侧弯/倾斜
    pelvis_tilt = euler_zxy[:, 2]  # Y: 前后倾

    # ==================== 4. 骨盆位置 (坐标系转换) ====================
    # Xsens: x=前, y=右, z=上
    # OpenSim: x=前, y=上, z=右
    pelvis_tx = df_pos['Pelvis x'].values[:n_samples]
    pelvis_ty = df_pos['Pelvis z'].values[:n_samples]  # z(上) -> y(上)
    pelvis_tz = df_pos['Pelvis y'].values[:n_samples]  # y(右) -> z(右)

    # ==================== 5. 读取关节角度 (Joint Angles ZXY) ====================
    df_joints = pd.read_excel(excel_path, sheet_name='Joint Angles ZXY')
    df_joints_xzy = pd.read_excel(excel_path, sheet_name='Joint Angles XZY')

    joint_mapping = {
        # 右侧下肢
        'hip_flexion_r': df_joints['Right Hip Flexion/Extension'],
        'hip_adduction_r': df_joints['Right Hip Abduction/Adduction'],
        'hip_rotation_r': df_joints['Right Hip Internal/External Rotation'],
        'knee_angle_r': df_joints['Right Knee Flexion/Extension'],
        'ankle_angle_r': df_joints['Right Ankle Dorsiflexion/Plantarflexion'],
        'subtalar_angle_r': df_joints['Right Ankle Internal/External Rotation'],
        'mtp_angle_r': df_joints['Right Ball Foot Flexion/Extension'],

        # 左侧下肢
        'hip_flexion_l': df_joints['Left Hip Flexion/Extension'],
        'hip_adduction_l': df_joints['Left Hip Abduction/Adduction'],
        'hip_rotation_l': df_joints['Left Hip Internal/External Rotation'],
        'knee_angle_l': df_joints['Left Knee Flexion/Extension'],
        'ankle_angle_l': df_joints['Left Ankle Dorsiflexion/Plantarflexion'],
        'subtalar_angle_l': df_joints['Left Ankle Internal/External Rotation'],
        'mtp_angle_l': df_joints['Left Ball Foot Flexion/Extension'],

        # 躯干/腰椎
        'lumbar_extension': df_joints['L5S1 Flexion/Extension'],
        'lumbar_bending': df_joints['L5S1 Lateral Bending'],
        'lumbar_rotation': df_joints['L5S1 Axial Bending'],

        # 右上肢
        'arm_flex_r': df_joints['Right Shoulder Flexion/Extension'],  ## 需要调整
        'arm_add_r': df_joints['Right Shoulder Abduction/Adduction'],  ## 需要调整
        'arm_rot_r': df_joints['Right Shoulder Internal/External Rotation'],
        'elbow_flex_r': df_joints['Right Elbow Flexion/Extension'],
        'pro_sup_r': df_joints['Right Elbow Pronation/Supination'],
        'wrist_flex_r': df_joints['Right Wrist Flexion/Extension'],
        'wrist_dev_r': df_joints['Right Wrist Ulnar Deviation/Radial Deviation'],

        # 左上肢
        'arm_flex_l': df_joints['Left Shoulder Flexion/Extension'],  ## 需要调整
        'arm_add_l': df_joints['Left Shoulder Abduction/Adduction'],  ## 需要调整
        'arm_rot_l': df_joints['Left Shoulder Internal/External Rotation'],
        'elbow_flex_l': df_joints['Left Elbow Flexion/Extension'],
        'pro_sup_l': df_joints['Left Elbow Pronation/Supination'],
        'wrist_flex_l': df_joints['Left Wrist Flexion/Extension'],
        'wrist_dev_l': df_joints['Left Wrist Ulnar Deviation/Radial Deviation'],

        # 肩胛
        'SC_y': df_joints['Right T4 Shoulder Flexion/Extension'],
        'SC_x': df_joints['Right T4 Shoulder Abduction/Adduction'],
        'SC_z': df_joints['Right T4 Shoulder Internal/External Rotation'],
        'SC_y_l': df_joints['Left T4 Shoulder Flexion/Extension'],
        'SC_x_l': df_joints['Left T4 Shoulder Abduction/Adduction'],
        'SC_z_l': df_joints['Left T4 Shoulder Internal/External Rotation'],
    }

    # ==================== 6. 构建数据矩阵 ====================
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
        'SC_y', 'SC_x', 'SC_z', 'SC_y_l', 'SC_x_l', 'SC_z_l'
    ]

    angles_matrix = np.zeros((n_samples, len(joint_names_order)))

    for i, joint_name in enumerate(joint_names_order):
        if joint_name in ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']:
            if joint_name == 'pelvis_tilt':
                angles_matrix[:, i] = pelvis_tilt
            elif joint_name == 'pelvis_list':
                angles_matrix[:, i] = pelvis_list
            else:
                angles_matrix[:, i] = pelvis_rotation
        elif joint_name in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
            if joint_name == 'pelvis_tx':
                angles_matrix[:, i] = pelvis_tx
            elif joint_name == 'pelvis_ty':
                angles_matrix[:, i] = pelvis_ty
            else:
                angles_matrix[:, i] = pelvis_tz
        elif joint_name in joint_mapping:
            angles_matrix[:, i] = joint_mapping[joint_name].values[:n_samples]
        else:
            angles_matrix[:, i] = np.zeros(n_samples)
            print(f"警告: 未找到关节 {joint_name}")

    # ==================== 7. 应用符号调整 ====================
    signs = np.ones(len(joint_names_order))
    signs[0] = -1  # pelvis_tilt
    signs[5] = -1  # pelvis_tz
    signs[7] = -1  # hip_flexion_r (根据原代码)
    signs[14] = -1  # hip_adduction_l
    signs[20] = -1  # lumbar_extension
    signs[24] = -1  # arm_flex_r
    signs[31] = -1  # arm_flex_l

    angles_matrix = angles_matrix * signs

    # ==================== 8. 保存 .mot 文件 ====================
    data_with_time = np.column_stack([time, angles_matrix])

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

        data_min = np.min(data_with_time[:, 1:])  # 排除时间列
        data_max = np.max(data_with_time[:, 1:])
        f.write(f'range {data_min:.4f} {data_max:.4f}\n')
        f.write('endheader\n')

        col_names = 'time\t' + '\t'.join(joint_names_order)
        f.write(col_names + '\n')

        for row in data_with_time:
            f.write('\t'.join([f'{val:.6f}' for val in row]) + '\n')

    print(f"\n已保存到: {output_mot_path}")
    print(f"数据形状: {data_with_time.shape}")

    # 验证输出
    print("\n前5行数据预览:")
    print(data_with_time[:5, :5])  # 显示时间 + 前4个关节角度

    return data_with_time, joint_names_order


if __name__ == "__main__":
    excel_file = "MVN-005.xlsx"
    data, joint_names = read_xsens_excel_for_opensim(excel_file)