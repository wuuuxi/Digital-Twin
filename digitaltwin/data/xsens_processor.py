"""
Xsens/MVNX数据处理模块
处理运动捕捉数据
"""
# import numpy as np
# import pandas as pd
# import os
# from digitaltwin.data.utils import find_nearest_idx, resample_data, print_debug_info, save_pickle, load_pickle
# from digitaltwin.data.utils import extract_continuous_segments, select_longest_segment

import os
import numpy as np
import pickle
import glob


class XsensProcessor:
    """
    Xsens 运动捕捉数据处理器。
    负责加载、解析和缓存 MVNX 数据。
    """

    @staticmethod
    def process(xsens_file, load_weight, folder):
        """
        处理 Xsens 数据。

        Parameters
        ----------
        xsens_file : str or None
            Xsens 文件路径，None 则自动搜索
        load_weight : str
            负载重量标识
        folder : str
            实验根目录

        Returns
        -------
        dict or None
            包含时间、关节角度、段位置等的数据字典
        """
        if xsens_file is None:
            xsens_file = XsensProcessor._auto_find(load_weight, folder)
            if xsens_file is None:
                print(f"负载 {load_weight}kg: 未找到 Xsens 文件")
                return None

        # 确保绝对路径
        if not os.path.isabs(xsens_file):
            xsens_file = os.path.join(folder, xsens_file)

        if not os.path.exists(xsens_file):
            print(f"Xsens 文件不存在: {xsens_file}")
            return None

        try:
            # 尝试加载缓存
            decoded_file = xsens_file.replace('.mvnx', '_decoded.pkl')
            if os.path.exists(decoded_file):
                with open(decoded_file, 'rb') as f:
                    xsens_data = pickle.load(f)
                print(f"负载 {load_weight}kg: 加载缓存 Xsens 数据")
            else:
                # 解析 MVNX 文件
                xsens_data = XsensProcessor._parse_mvnx(xsens_file)
                if xsens_data:
                    with open(decoded_file, 'wb') as f:
                        pickle.dump(xsens_data, f)
                    print(f"负载 {load_weight}kg: 解析并缓存 Xsens 数据")

            if xsens_data:
                xsens_data['metadata']['load_weight'] = load_weight
            return xsens_data

        except Exception as e:
            print(f"Xsens 数据处理错误: {e}")
            return None

    @staticmethod
    def _auto_find(load_weight, folder):
        """自动搜索匹配的 Xsens 文件"""
        pattern = f"*{load_weight}*mvnx"
        files = glob.glob(os.path.join(folder, pattern))
        if not files:
            files = glob.glob(
                os.path.join(folder, '**', pattern), recursive=True)
        return files[0] if files else None

    @staticmethod
    def _parse_mvnx(file_path):
        """
        解析 MVNX 文件。
        注意：这是简化版本，实际使用时需根据 MVNX 格式完善解析逻辑。
        """
        try:
            n_samples = 1000
            fs = 100
            xsens_data = {
                'time': np.arange(n_samples) / fs,
                'segment_positions': np.zeros((n_samples, 3, 10)),
                'joint_angles': np.zeros((n_samples, 3, 20)),
                'metadata': {
                    'fs': fs,
                    'n_segments': 10,
                    'n_joints': 20,
                    'file_path': file_path
                }
            }
            return xsens_data
        except:
            return None


# class XsensProcessor:
#     """Xsens运动数据处理器"""
#
#     def __init__(self, motion_label='benchpress'):
#         """
#         初始化Xsens处理器
#
#         Parameters:
#         -----------
#         motion_label : str
#             运动类型，'benchpress'或'squat'
#         """
#         self.motion_label = motion_label
#         self.data = None
#         self.lift_data = None
#         self.lower_data = None
#         self.ArmLength = 0.37
#
#     def load_from_excel(self, file_path, sheet_name='Segment Position'):
#         """
#         从Excel文件加载Xsens数据
#
#         Parameters:
#         -----------
#         file_path : str
#             Excel文件路径
#         sheet_name : str
#             工作表名称
#
#         Returns:
#         --------
#         pd.DataFrame
#             Xsens数据
#         """
#         try:
#             print_debug_info(f"加载Xsens文件: {os.path.basename(file_path)}")
#             data = pd.read_excel(file_path, sheet_name=sheet_name)
#             print_debug_info(f"Xsens数据加载成功: {len(data)}行")
#             return data
#         except Exception as e:
#             print_debug_info(f"Xsens加载失败: {e}")
#             return None
#
#     def extract_hand_data(self, position_data, velocity_data=None, fs=60):
#         """
#         提取手部位置和速度数据
#
#         Parameters:
#         -----------
#         position_data : pd.DataFrame
#             位置数据
#         velocity_data : pd.DataFrame, optional
#             速度数据
#         fs : int
#             采样率
#
#         Returns:
#         --------
#         pd.DataFrame
#             处理后的手部数据
#         """
#         # 提取左手Z轴位置
#         hand_position = position_data[['Left Hand z']].rename(columns={'Left Hand z': 'pos_l'})
#
#         # 创建时间列
#         time = np.linspace(0, hand_position.shape[0] - 1, hand_position.shape[0]) / fs
#         xsens_data = pd.DataFrame({'time': time})
#         xsens_data['pos_l'] = hand_position['pos_l'].values
#
#         # 如果提供了速度数据，提取速度
#         if velocity_data is not None:
#             hand_velocity = velocity_data[['Left Hand z']].rename(columns={'Left Hand z': 'vel_l'})
#             xsens_data['vel_l'] = hand_velocity['vel_l'].values
#         else:
#             # 计算速度
#             xsens_data['vel_l'] = np.gradient(xsens_data['pos_l'].values, xsens_data['time'].values)
#
#         return xsens_data
#
#     def crop_by_time(self, data, start_time, end_time, time_col='time'):
#         """
#         根据时间裁剪数据
#
#         Parameters:
#         -----------
#         data : pd.DataFrame
#             原始数据
#         start_time : float
#             开始时间
#         end_time : float
#             结束时间
#         time_col : str
#             时间列名
#
#         Returns:
#         --------
#         pd.DataFrame
#             裁剪后的数据
#         """
#         time = data[time_col].values
#         start_idx = find_nearest_idx(time, start_time)
#         end_idx = find_nearest_idx(time, end_time)
#         return data.iloc[start_idx:end_idx].copy()
#
#     def extract_phases(self, data, pos_col='pos_l', vel_col='vel_l'):
#         """
#         分离上升和下降阶段
#
#         Parameters:
#         -----------
#         data : pd.DataFrame
#             原始数据
#         pos_col : str
#             位置列名
#         vel_col : str
#             速度列名
#
#         Returns:
#         --------
#         tuple
#             (lift_data, lower_data)
#         """
#         pos = data[pos_col].values
#         vel = data[vel_col].values
#
#         # 使用速度符号判断运动方向
#         lift_indices = np.where(vel > 0.01)[0]
#         lower_indices = np.where(vel < -0.01)[0]
#
#         # 如果速度判断不充分，使用位置变化判断
#         if len(lift_indices) == 0 or len(lower_indices) == 0:
#             lift_indices = []
#             lower_indices = []
#             for i in range(1, len(pos)):
#                 if pos[i] > pos[i - 1]:
#                     lift_indices.append(i)
#                 elif pos[i] < pos[i - 1]:
#                     lower_indices.append(i)
#
#         # 提取连续的运动段
#         lift_segments = extract_continuous_segments(lift_indices)
#         lower_segments = extract_continuous_segments(lower_indices)
#
#         # 选择最长的连续段
#         lift_data = select_longest_segment(data, lift_segments)
#         lower_data = select_longest_segment(data, lower_segments)
#
#         # 重置时间
#         if lift_data is not None:
#             lift_duration = lift_data['time'].iloc[-1] - lift_data['time'].iloc[0]
#             lift_data = lift_data.copy()
#             lift_data['time'] = np.linspace(0, lift_duration, len(lift_data))
#
#         if lower_data is not None:
#             lower_duration = lower_data['time'].iloc[-1] - lower_data['time'].iloc[0]
#             lower_data = lower_data.copy()
#             lower_data['time'] = np.linspace(0, lower_duration, len(lower_data))
#
#         self.lift_data = lift_data
#         self.lower_data = lower_data
#
#         return lift_data, lower_data
#
#     def apply_position_constraints(self, data, pos_col='pos_l'):
#         """
#         应用位置约束
#
#         Parameters:
#         -----------
#         data : pd.DataFrame
#             原始数据
#         pos_col : str
#             位置列名
#
#         Returns:
#         --------
#         pd.DataFrame
#             约束后的数据
#         """
#         if pos_col not in data.columns:
#             return data
#
#         data = data.copy()
#
#         if self.motion_label == 'benchpress':
#             data[pos_col] = data[pos_col].clip(0.68, 0.68 + self.ArmLength)
#         elif self.motion_label == 'squat':
#             data[pos_col] = data[pos_col].clip(0.95, 1.4)
#
#         return data
#
#     def synchronize_phases(self, lift_data, lower_data):
#         """
#         同步上升和下降阶段的时间长度
#
#         Parameters:
#         -----------
#         lift_data : pd.DataFrame
#             上升阶段数据
#         lower_data : pd.DataFrame
#             下降阶段数据
#
#         Returns:
#         --------
#         tuple
#             (lift_data, lower_data)
#         """
#         if lift_data is None or lower_data is None:
#             return lift_data, lower_data
#
#         lift_len = len(lift_data)
#         lower_len = len(lower_data)
#
#         # 确定目标长度（取较短的那个）
#         target_len = min(lift_len, lower_len)
#
#         # 重采样到相同长度
#         if lift_len != target_len:
#             lift_data = resample_data(lift_data, target_len)
#
#         if lower_len != target_len:
#             lower_data = resample_data(lower_data, target_len)
#
#         return lift_data, lower_data
#
#     def combine_phases(self, lift_data, lower_data):
#         """
#         合并上升和下降阶段，添加阶段标记
#
#         Parameters:
#         -----------
#         lift_data : pd.DataFrame
#             上升阶段数据
#         lower_data : pd.DataFrame
#             下降阶段数据
#
#         Returns:
#         --------
#         pd.DataFrame
#             合并后的数据
#         """
#         combined_data = pd.DataFrame()
#
#         if lift_data is not None:
#             lift_with_phase = lift_data.copy()
#             lift_with_phase['phase'] = 'lift'
#             combined_data = pd.concat([combined_data, lift_with_phase], ignore_index=True)
#
#         if lower_data is not None:
#             lower_with_phase = lower_data.copy()
#             lower_with_phase['phase'] = 'lower'
#             combined_data = pd.concat([combined_data, lower_with_phase], ignore_index=True)
#
#         if len(combined_data) > 0:
#             total_duration = combined_data['time'].iloc[-1] - combined_data['time'].iloc[0]
#             combined_data['time'] = np.linspace(0, total_duration, len(combined_data))
#
#         return combined_data
#
#     def save_processed_data(self, save_path):
#         """保存处理后的Xsens数据"""
#         data_to_save = {
#             'data': self.data,
#             'lift_data': self.lift_data,
#             'lower_data': self.lower_data
#         }
#         save_pickle(data_to_save, save_path)
#         print_debug_info(f"Xsens数据已保存到: {save_path}")
#
#     def load_processed_data(self, load_path):
#         """加载处理后的Xsens数据"""
#         data = load_pickle(load_path)
#         if data:
#             self.data = data.get('data')
#             self.lift_data = data.get('lift_data')
#             self.lower_data = data.get('lower_data')
#             print_debug_info(f"Xsens数据已加载: {load_path}")
#         return data