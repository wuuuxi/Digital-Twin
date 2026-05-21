"""
Xsens 运动捕捉数据处理器。
支持 MVNX 文件和 Xsens 导出的 Excel 文件（含关节角度、段位置等）。
读取时不自动保存 .mot 文件；如需转换请调用 save_mot() 或运行 example_xsens_to_mot.py。
"""
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None


class XsensProcessor:
    """
    Xsens 运动捕捉数据处理器。
    负责加载、解析和缓存 Xsens 数据（MVNX 或 Excel 格式）。
    """

    # OpenSim 关节名顺序（与 .mot 文件一致）
    JOINT_NAMES = [
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

    SIGN_FLIP_INDICES = [0, 5, 7, 14, 20, 24, 31]

    # ----------------------------------------------------------------
    #  节段长度测量定义（用于模型缩放）
    #  每对 (测量名, 开始节段, 结束节段) 中，
    #  Xsens 各节段坐标原点 = 该节段近端关节中心，
    #  故两节段原点间距 = 近端关节中心间距。
    # ----------------------------------------------------------------
    SEGMENT_PAIRS = [
        # 下肢
        ('Right Thigh Length',     'Right Upper Leg',  'Right Lower Leg'),
        ('Left Thigh Length',      'Left Upper Leg',   'Left Lower Leg'),
        ('Right Shank Length',     'Right Lower Leg',  'Right Foot'),
        ('Left Shank Length',      'Left Lower Leg',   'Left Foot'),
        ('Right Foot Length',      'Right Foot',       'Right Toe'),
        ('Left Foot Length',       'Left Foot',        'Left Toe'),
        # 骨盆
        ('Pelvis Width',           'Right Upper Leg',  'Left Upper Leg'),
        # 躯干
        ('Pelvis to L5 Length',    'Pelvis',           'L5'),
        ('L5 to T8 Length',        'L5',               'T8'),
        ('T8 to Neck Length',      'T8',               'Neck'),
        ('Neck to Head Length',    'Neck',             'Head'),
        # 上肢
        # Right Upper Arm 原点 = 肩关节，Right Forearm 原点 = 肘关节
        ('Right Upper Arm Length', 'Right Upper Arm',  'Right Forearm'),
        ('Left Upper Arm Length',  'Left Upper Arm',   'Left Forearm'),
        ('Right Forearm Length',   'Right Forearm',    'Right Hand'),
        ('Left Forearm Length',    'Left Forearm',     'Left Hand'),
        # 其他
        ('Shoulder Width',         'Right Shoulder',   'Left Shoulder'),
    ]

    @staticmethod
    def process(xsens_file, load_weight, folder, xsens_folder=None):
        """
        处理 Xsens 数据（自动检测 Excel 或 MVNX 格式）。
        不自动保存 .mot 文件。

        Parameters
        ----------
        xsens_file : str or None
        load_weight : str
        folder : str
        xsens_folder : str, optional

        Returns
        -------
        dict or None
            {time, joint_angles (DataFrame), segment_positions (DataFrame), metadata}
        """
        if xsens_file is None:
            return None

        file_path = XsensProcessor._resolve_file_path(
            xsens_file, xsens_folder, folder)
        if file_path is None:
            print(f"负载 {load_weight}: Xsens 文件未找到: {xsens_file}")
            return None

        try:
            ext = Path(file_path).suffix.lower()
            if ext in ('.xlsx', '.xls'):
                xsens_data = XsensProcessor._process_excel(file_path)
            elif ext == '.mvnx':
                xsens_data = XsensProcessor._process_mvnx(file_path)
            else:
                print(f"不支持的 Xsens 文件格式: {ext}")
                return None

            if xsens_data:
                xsens_data['metadata']['load_weight'] = load_weight
            return xsens_data

        except Exception as e:
            print(f"Xsens 数据处理错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def save_mot(xsens_data, output_path):
        """
        将已加载的 Xsens 数据保存为 OpenSim .mot 文件。

        Parameters
        ----------
        xsens_data : dict
            process() 返回的数据字典
        output_path : str
            输出 .mot 文件路径
        """
        joint_df = xsens_data.get('joint_angles')
        if joint_df is None:
            print("无关节角度数据，无法保存 .mot"); return

        time = joint_df['time'].values
        joint_names = [c for c in joint_df.columns if c != 'time']
        angles = joint_df[joint_names].values
        n_samples = len(time)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        data = np.column_stack([time, angles])
        with open(output_path, 'w') as f:
            f.write('first trial\n')
            f.write(f'nRows={n_samples}\n')
            f.write(f'nColumns={len(joint_names) + 1}\n\n')
            f.write('# SIMM Motion File Header:\n')
            f.write(f'name xsens_data\n')
            f.write(f'datacolumns {len(joint_names) + 1}\n')
            f.write(f'datarows {n_samples}\n')
            f.write('otherdata 1\n')
            d_min, d_max = np.min(angles), np.max(angles)
            f.write(f'range {d_min:.4f} {d_max:.4f}\n')
            f.write('endheader\n')
            f.write('time\t' + '\t'.join(joint_names) + '\n')
            for row in data:
                f.write('\t'.join([f'{v:.6f}' for v in row]) + '\n')

    # ================================================================
    #  节段长度提取（用于模型缩放）
    # ================================================================

    @staticmethod
    def extract_segment_measurements(xsens_file, verbose=True):
        """
        从 Xsens Excel 文件读取 Segment Position 工作表，
        计算 SEGMENT_PAIRS 中所有节段对的逐帧距离。

        Parameters
        ----------
        xsens_file : str
            Xsens 导出的 Excel (.xlsx) 文件路径。
        verbose : bool

        Returns
        -------
        dict
            {measurement_name: [length_per_frame, ...]}
        """
        if verbose:
            print(f'  读取 Xsens 文件: {xsens_file}')
        df_pos = pd.read_excel(xsens_file, sheet_name='Segment Position')
        n_frames = len(df_pos)
        if verbose:
            print(f'  数据总帧数: {n_frames}')

        measurements = {name: [] for name, _, _ in XsensProcessor.SEGMENT_PAIRS}
        for frame in range(n_frames):
            for name, seg1, seg2 in XsensProcessor.SEGMENT_PAIRS:
                p1 = XsensProcessor._seg_pos(df_pos, frame, seg1)
                p2 = XsensProcessor._seg_pos(df_pos, frame, seg2)
                if p1 is not None and p2 is not None:
                    measurements[name].append(float(np.linalg.norm(p2 - p1)))

        if verbose:
            valid = max((len(v) for v in measurements.values()), default=0)
            print(f'  有效帧数: {valid} / {n_frames}')
        return measurements

    @staticmethod
    def _seg_pos(df_pos, frame, segment):
        """获取指定帧和节段的 (x, y, z) 坐标，失败返回 None。"""
        try:
            return np.array([
                df_pos.loc[frame, f'{segment} x'],
                df_pos.loc[frame, f'{segment} y'],
                df_pos.loc[frame, f'{segment} z'],
            ], dtype=float)
        except Exception:
            return None

    @staticmethod
    def segment_measurements_to_opensim_targets(measurements):
        """
        将 extract_segment_measurements() 返回的字典转换为
        OpenSim scale_full_body_model() 所需的 target_lengths 字典（取均值）。

        Returns
        -------
        dict  {opensim_body_name: target_length_m}
        """
        def mean(key):
            v = measurements.get(key, [])
            return float(np.mean(v)) if len(v) > 0 else None

        targets = {}

        def _set(k, val):
            if val is not None and np.isfinite(val):
                targets[k] = val

        _set('femur_r',   mean('Right Thigh Length'))
        _set('femur_l',   mean('Left Thigh Length'))
        _set('tibia_r',   mean('Right Shank Length'))
        _set('tibia_l',   mean('Left Shank Length'))
        _set('calcn_r',   mean('Right Foot Length'))
        _set('calcn_l',   mean('Left Foot Length'))
        _set('pelvis',    mean('Pelvis Width'))
        spine = [mean(k) for k in ['Pelvis to L5 Length', 'L5 to T8 Length', 'T8 to Neck Length']]
        if all(v is not None for v in spine):
            _set('torso', sum(spine))
        _set('humerus_r', mean('Right Upper Arm Length'))
        _set('humerus_l', mean('Left Upper Arm Length'))
        _set('radius_r',  mean('Right Forearm Length'))
        _set('radius_l',  mean('Left Forearm Length'))
        return targets

    @staticmethod
    def print_segment_measurements(measurements):
        """打印各节段长度统计（均值 ± 标准差）。"""
        print('\n' + '=' * 72)
        print('Xsens 身体节段长度计算结果')
        print('=' * 72)
        groups = [
            ('下肢', ['Right Thigh Length', 'Left Thigh Length',
                      'Right Shank Length', 'Left Shank Length',
                      'Right Foot Length',  'Left Foot Length']),
            ('骨盆', ['Pelvis Width']),
            ('躯干', ['Pelvis to L5 Length', 'L5 to T8 Length',
                      'T8 to Neck Length',   'Neck to Head Length']),
            ('上肢', ['Right Upper Arm Length', 'Left Upper Arm Length',
                      'Right Forearm Length',   'Left Forearm Length',
                      'Shoulder Width']),
        ]
        for group_name, keys in groups:
            print(f'\n[{group_name}]')
            print('-' * 72)
            for key in keys:
                vals = measurements.get(key, [])
                if vals:
                    m, s = float(np.mean(vals)), float(np.std(vals))
                    cv = s / m * 100 if m > 1e-9 else 0
                    print(f'  {key:30s}: {m:.4f} ± {s:.4f} m  (CV={cv:.2f}%)')
                else:
                    print(f'  {key:30s}: 数据缺失')

    @staticmethod
    def print_opensim_targets_table(measurements):
        """打印 Xsens → OpenSim 节段缩放对照表。"""
        targets = XsensProcessor.segment_measurements_to_opensim_targets(measurements)
        print('\n' + '=' * 60)
        print('【OpenSim 缩放对照表】')
        print('=' * 60)
        print(f'  {"OpenSim 体段":22s}  {"Xsens 测量 (m)":>16}')
        print('  ' + '-' * 42)
        for body, length in sorted(targets.items()):
            print(f'  {body:22s}  {length:.4f}')
        spine_parts = ['Pelvis to L5 Length', 'L5 to T8 Length', 'T8 to Neck Length']
        spine_vals = [np.mean(measurements.get(k, [float('nan')])) for k in spine_parts]
        if all(np.isfinite(v) for v in spine_vals):
            print(f'\n  torso = {spine_vals[0]:.4f}(Pelvis-L5)'
                  f' + {spine_vals[1]:.4f}(L5-T8)'
                  f' + {spine_vals[2]:.4f}(T8-Neck)'
                  f' = {sum(spine_vals):.4f} m')

    @staticmethod
    def _resolve_file_path(xsens_file, xsens_folder, folder):
        if os.path.isabs(xsens_file):
            return xsens_file if os.path.exists(xsens_file) else None
        if xsens_folder:
            path = os.path.join(xsens_folder, xsens_file)
            if os.path.exists(path):
                return path
        path = os.path.join(folder, xsens_file)
        if os.path.exists(path):
            return path
        return None

    # ==============================================================
    #  Excel 格式处理
    # ==============================================================

    @staticmethod
    def _process_excel(excel_path):
        if R is None:
            raise ImportError("需要 scipy.spatial.transform.Rotation")

        df_info = pd.read_excel(excel_path, sheet_name='General Information', header=None)
        frame_rate = 60.0
        for _, row in df_info.iterrows():
            if row[0] == 'Frame Rate':
                frame_rate = float(row[1]); break

        df_pos = pd.read_excel(excel_path, sheet_name='Segment Position')
        n_samples = len(df_pos)
        time = df_pos['Frame'].values / frame_rate

        df_quat = pd.read_excel(excel_path, sheet_name='Segment Orientation - Quat')
        pelvis_q = np.column_stack([
            df_quat['Pelvis q1'].values[:n_samples],
            df_quat['Pelvis q2'].values[:n_samples],
            df_quat['Pelvis q3'].values[:n_samples],
            df_quat['Pelvis q0'].values[:n_samples],
        ])
        euler_zxy = R.from_quat(pelvis_q).as_euler('zxy', degrees=True)
        pelvis_rotation, pelvis_list, pelvis_tilt = euler_zxy[:, 0], euler_zxy[:, 1], euler_zxy[:, 2]

        pelvis_tx = df_pos['Pelvis x'].values[:n_samples]
        pelvis_ty = df_pos['Pelvis z'].values[:n_samples]
        pelvis_tz = df_pos['Pelvis y'].values[:n_samples]

        df_j = pd.read_excel(excel_path, sheet_name='Joint Angles ZXY')
        jmap = {
            'hip_flexion_r': 'Right Hip Flexion/Extension',
            'hip_adduction_r': 'Right Hip Abduction/Adduction',
            'hip_rotation_r': 'Right Hip Internal/External Rotation',
            'knee_angle_r': 'Right Knee Flexion/Extension',
            'ankle_angle_r': 'Right Ankle Dorsiflexion/Plantarflexion',
            'subtalar_angle_r': 'Right Ankle Internal/External Rotation',
            'mtp_angle_r': 'Right Ball Foot Flexion/Extension',
            'hip_flexion_l': 'Left Hip Flexion/Extension',
            'hip_adduction_l': 'Left Hip Abduction/Adduction',
            'hip_rotation_l': 'Left Hip Internal/External Rotation',
            'knee_angle_l': 'Left Knee Flexion/Extension',
            'ankle_angle_l': 'Left Ankle Dorsiflexion/Plantarflexion',
            'subtalar_angle_l': 'Left Ankle Internal/External Rotation',
            'mtp_angle_l': 'Left Ball Foot Flexion/Extension',
            'lumbar_extension': 'L5S1 Flexion/Extension',
            'lumbar_bending': 'L5S1 Lateral Bending',
            'lumbar_rotation': 'L5S1 Axial Bending',
            'arm_flex_r': 'Right Shoulder Flexion/Extension',
            'arm_add_r': 'Right Shoulder Abduction/Adduction',
            'arm_rot_r': 'Right Shoulder Internal/External Rotation',
            'elbow_flex_r': 'Right Elbow Flexion/Extension',
            'pro_sup_r': 'Right Elbow Pronation/Supination',
            'wrist_flex_r': 'Right Wrist Flexion/Extension',
            'wrist_dev_r': 'Right Wrist Ulnar Deviation/Radial Deviation',
            'arm_flex_l': 'Left Shoulder Flexion/Extension',
            'arm_add_l': 'Left Shoulder Abduction/Adduction',
            'arm_rot_l': 'Left Shoulder Internal/External Rotation',
            'elbow_flex_l': 'Left Elbow Flexion/Extension',
            'pro_sup_l': 'Left Elbow Pronation/Supination',
            'wrist_flex_l': 'Left Wrist Flexion/Extension',
            'wrist_dev_l': 'Left Wrist Ulnar Deviation/Radial Deviation',
            'SC_y': 'Right T4 Shoulder Flexion/Extension',
            'SC_x': 'Right T4 Shoulder Abduction/Adduction',
            'SC_z': 'Right T4 Shoulder Internal/External Rotation',
            'SC_y_l': 'Left T4 Shoulder Flexion/Extension',
            'SC_x_l': 'Left T4 Shoulder Abduction/Adduction',
            'SC_z_l': 'Left T4 Shoulder Internal/External Rotation',
        }

        pelvis_locals = {
            'pelvis_tilt': pelvis_tilt, 'pelvis_list': pelvis_list,
            'pelvis_rotation': pelvis_rotation,
            'pelvis_tx': pelvis_tx, 'pelvis_ty': pelvis_ty, 'pelvis_tz': pelvis_tz,
        }

        joint_names = XsensProcessor.JOINT_NAMES
        angles = np.zeros((n_samples, len(joint_names)))
        for i, jn in enumerate(joint_names):
            if jn in pelvis_locals:
                angles[:, i] = pelvis_locals[jn]
            elif jn in jmap and jmap[jn] in df_j.columns:
                angles[:, i] = df_j[jmap[jn]].values[:n_samples]

        signs = np.ones(len(joint_names))
        for idx in XsensProcessor.SIGN_FLIP_INDICES:
            signs[idx] = -1
        angles *= signs

        joint_df = pd.DataFrame(angles, columns=joint_names)
        joint_df.insert(0, 'time', time)

        pos_df = pd.DataFrame({
            'time': time, 'pelvis_tx': pelvis_tx,
            'pelvis_ty': pelvis_ty, 'pelvis_tz': pelvis_tz,
        })
        if 'Left Hand x' in df_pos.columns:
            pos_df['left_hand_x'] = df_pos['Left Hand x'].values[:n_samples]
            pos_df['left_hand_y'] = df_pos['Left Hand z'].values[:n_samples]
            pos_df['left_hand_z'] = df_pos['Left Hand y'].values[:n_samples]

        return {
            'time': time,
            'joint_angles': joint_df,
            'segment_positions': pos_df,
            'metadata': {
                'fs': frame_rate, 'n_samples': n_samples,
                'file_path': excel_path, 'joint_names': joint_names,
            }
        }

    # ==============================================================
    #  MVNX 格式处理（简化版）
    # ==============================================================

    @staticmethod
    def _process_mvnx(file_path):
        decoded_file = file_path.replace('.mvnx', '_decoded.pkl')
        if os.path.exists(decoded_file):
            with open(decoded_file, 'rb') as f:
                return pickle.load(f)
        try:
            n_samples, fs = 1000, 100
            xsens_data = {
                'time': np.arange(n_samples) / fs,
                'joint_angles': None, 'segment_positions': None,
                'metadata': {'fs': fs, 'n_samples': n_samples, 'file_path': file_path},
            }
            with open(decoded_file, 'wb') as f:
                pickle.dump(xsens_data, f)
            return xsens_data
        except Exception:
            return None