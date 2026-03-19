"""
机器人数据处理模块
处理机器人传感器数据
"""
# import numpy as np
# import pandas as pd
# import os
# from digitaltwin.data.utils import print_debug_info, save_pickle, load_pickle

import os
import numpy as np
import pandas as pd


class RobotProcessor:
    """
    机器人数据加载和预处理。
    负责读取机器人 CSV/Excel 文件，统一列名，处理时间列，调整位置数据。
    """

    # 机器人数据列名映射（原始列名 → 标准列名）
    COLUMN_MAP = {
        'axis4_force': 'pos_l',
        'axis4_vel': 'force_l',
        'axis4_pos': 'vel_l',
        'axis4_accel': 'acc_l',
        'axis3_force': 'pos_r',
        'axis3_vel': 'force_r',
        'axis3_pos': 'vel_r',
        'axis3_accel': 'acc_r',
        'Timestamp': 'time',
        'Time': 'time',
    }

    @staticmethod
    def process(robot_file, load_weight, robot_folder, folder,
                turn_position=False):
        """
        处理单个负载的机器人数据文件。

        Parameters
        ----------
        robot_file : str
            机器人数据文件名或路径
        load_weight : str
            负载重量标识
        robot_folder : str
            机器人数据目录
        folder : str
            实验根目录
        turn_position : bool
            是否反转位置数据方向

        Returns
        -------
        pd.DataFrame or None
            处理后的机器人数据
        """
        try:
            # 1. 解析文件路径
            file_path = RobotProcessor._resolve_file_path(
                robot_file, robot_folder, folder)
            if file_path is None:
                return None

            # 2. 读取数据
            robot_df = RobotProcessor._read_file(file_path)
            if robot_df is None:
                return None

            # 3. 统一列名
            robot_df = RobotProcessor._rename_columns(robot_df)

            # 4. 处理时间列
            robot_df = RobotProcessor._process_time_column(robot_df)

            # 5. 添加负载信息
            robot_df['load'] = float(load_weight)
            robot_df['load_weight'] = load_weight

            # 6. 调整位置数据
            if 'pos_l' in robot_df.columns:
                if turn_position:
                    robot_df['pos_l'] = 0.7 + robot_df['pos_l']
                else:
                    robot_df['pos_l'] = 0.7 - robot_df['pos_l']

            return robot_df

        except Exception as e:
            print(f"机器人数据处理错误: {e}")
            return None

    @staticmethod
    def _resolve_file_path(robot_file, robot_folder, folder):
        """解析机器人数据文件的完整路径"""
        if os.path.isabs(robot_file):
            if os.path.exists(robot_file):
                return robot_file
            return None
        file_path = os.path.join(robot_folder, robot_file)
        if os.path.exists(file_path):
            return file_path
        file_path = os.path.join(folder, robot_file)
        if os.path.exists(file_path):
            return file_path
        print(f"机器人文件不存在: {robot_file}")
        return None

    @staticmethod
    def _read_file(file_path):
        """根据文件格式读取数据"""
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            else:
                try:
                    return pd.read_csv(file_path, sep='\t')
                except:
                    return pd.read_csv(file_path, sep=',')
        except Exception as e:
            print(f"读取文件失败: {e}")
            return None

    @staticmethod
    def _rename_columns(df):
        """统一列名映射"""
        mapping = {}
        for orig, target in RobotProcessor.COLUMN_MAP.items():
            if orig in df.columns:
                mapping[orig] = target
        return df.rename(columns=mapping)

    @staticmethod
    def _process_time_column(df):
        """处理时间列，确保从 0 开始的相对时间（秒）"""
        if 'time' not in df.columns:
            if 'TimeStamp' in df.columns:
                df['time'] = pd.to_datetime(df['TimeStamp'])
            else:
                fs = 100  # 假设 100Hz 采样率
                df['time'] = np.arange(len(df)) / fs

        if (df['time'].dtype == 'object'
                or 'datetime' in str(df['time'].dtype)):
            df['time_abs'] = pd.to_datetime(df['time'])
            df['time'] = (
                (df['time_abs'] - df['time_abs'].iloc[0])
                .dt.total_seconds()
            )
        else:
            df['time'] = df['time'] - df['time'].iloc[0]

        return df


# class RobotProcessor:
#     """机器人数据处理器"""
#
#     def __init__(self, turn_position=False):
#         """
#         初始化机器人处理器
#
#         Parameters:
#         -----------
#         turn_position : bool
#             是否翻转位置数据
#         """
#         self.turn_position = turn_position
#         self.data = None
#
#     def load_from_csv(self, file_path):
#         """
#         从CSV文件加载机器人数据
#
#         Parameters:
#         -----------
#         file_path : str
#             CSV文件路径
#
#         Returns:
#         --------
#         pd.DataFrame
#             机器人数据
#         """
#         try:
#             print_debug_info(f"加载机器人文件: {os.path.basename(file_path)}")
#
#             if file_path.endswith('.csv'):
#                 df = pd.read_csv(file_path)
#             elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
#                 df = pd.read_excel(file_path)
#             else:
#                 try:
#                     df = pd.read_csv(file_path, sep='\t')
#                 except:
#                     df = pd.read_csv(file_path, sep=',')
#
#             print_debug_info(f"机器人数据加载成功: {len(df)}行")
#             return df
#
#         except Exception as e:
#             print_debug_info(f"机器人数据加载失败: {e}")
#             return None
#
#     def standardize_columns(self, df):
#         """
#         标准化列名
#
#         Parameters:
#         -----------
#         df : pd.DataFrame
#             原始数据
#
#         Returns:
#         --------
#         pd.DataFrame
#             标准化后的数据
#         """
#         column_mapping = {
#             'axis4_force': 'force_l',
#             'axis4_vel': 'vel_l',
#             'axis4_pos': 'pos_l',
#             'axis4_accel': 'acc_l',
#             'axis3_force': 'force_r',
#             'axis3_vel': 'vel_r',
#             'axis3_pos': 'pos_r',
#             'axis3_accel': 'acc_r',
#             'Timestamp': 'time',
#             'Time': 'time'
#         }
#
#         # 只映射存在的列
#         rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
#
#         if rename_dict:
#             df = df.rename(columns=rename_dict)
#
#         return df
#
#     def create_time_column(self, df, fs=100):
#         """
#         创建时间列
#
#         Parameters:
#         -----------
#         df : pd.DataFrame
#             数据
#         fs : int
#             采样率
#
#         Returns:
#         --------
#         pd.DataFrame
#             添加时间列后的数据
#         """
#         if 'time' in df.columns:
#             return df
#
#         if 'TimeStamp' in df.columns:
#             df['time'] = pd.to_datetime(df['TimeStamp'])
#         else:
#             df['time'] = np.arange(len(df)) / fs
#
#         return df
#
#     def normalize_time(self, df, time_col='time'):
#         """
#         归一化时间（从0开始）
#
#         Parameters:
#         -----------
#         df : pd.DataFrame
#             数据
#         time_col : str
#             时间列名
#
#         Returns:
#         --------
#         pd.DataFrame
#             处理后的数据
#         """
#         if time_col not in df.columns:
#             return df
#
#         df = df.copy()
#
#         if df[time_col].dtype == 'object' or 'datetime' in str(df[time_col].dtype):
#             df['time_abs'] = pd.to_datetime(df[time_col])
#             df[time_col] = (df['time_abs'] - df['time_abs'].iloc[0]).dt.total_seconds()
#         else:
#             df[time_col] = df[time_col] - df[time_col].iloc[0]
#
#         return df
#
#     def apply_position_transform(self, df, pos_col='pos_l'):
#         """
#         应用位置变换
#
#         Parameters:
#         -----------
#         df : pd.DataFrame
#             数据
#         pos_col : str
#             位置列名
#
#         Returns:
#         --------
#         pd.DataFrame
#             变换后的数据
#         """
#         if pos_col not in df.columns:
#             return df
#
#         df = df.copy()
#
#         if self.turn_position:
#             df[pos_col] = 0.7 + df[pos_col]
#         else:
#             df[pos_col] = 0.7 - df[pos_col]
#
#         return df
#
#     def add_load_info(self, df, load_weight):
#         """
#         添加负载信息
#
#         Parameters:
#         -----------
#         df : pd.DataFrame
#             数据
#         load_weight : str or float
#             负载重量
#
#         Returns:
#         --------
#         pd.DataFrame
#             添加负载信息后的数据
#         """
#         df = df.copy()
#         df['load'] = float(load_weight)
#         df['load_weight'] = str(load_weight)
#         return df
#
#     def process_robot_data(self, file_path, load_weight, fs=100):
#         """
#         完整的机器人数据处理流程
#
#         Parameters:
#         -----------
#         file_path : str
#             文件路径
#         load_weight : str or float
#             负载重量
#         fs : int
#             采样率
#
#         Returns:
#         --------
#         pd.DataFrame
#             处理后的机器人数据
#         """
#         # 加载数据
#         df = self.load_from_csv(file_path)
#         if df is None:
#             return None
#
#         # 标准化列名
#         df = self.standardize_columns(df)
#
#         # 创建时间列
#         df = self.create_time_column(df, fs)
#
#         # 归一化时间
#         df = self.normalize_time(df)
#
#         # 应用位置变换
#         df = self.apply_position_transform(df)
#
#         # 添加负载信息
#         df = self.add_load_info(df, load_weight)
#
#         self.data = df
#         return df
#
#     def save_processed_data(self, save_path):
#         """保存处理后的机器人数据"""
#         if self.data is not None:
#             self.data.to_csv(save_path, index=False)
#             print_debug_info(f"机器人数据已保存到: {save_path}")
#
#     def load_processed_data(self, load_path):
#         """加载处理后的机器人数据"""
#         if os.path.exists(load_path):
#             self.data = pd.read_csv(load_path)
#             print_debug_info(f"机器人数据已加载: {load_path}")
#         return self.data