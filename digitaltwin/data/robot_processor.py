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


class RobotOriginProcessor:
    """
    机器人原始数据处理器。
    处理机器人直接输出的原始 CSV 文件（如变负载实验数据），
    该格式前 17 行为设备头部信息，实际数据从第 18 行开始。

    与 RobotProcessor 的区别：
    - RobotProcessor: 读取经过软件处理后的标准格式（Timestamp + axis4_*）
    - RobotOriginProcessor: 读取机器人原始输出（skiprows=17, ' time' + ' axis3_*'）
    """

    # 原始格式列名映射（列名前有空格）
    COLUMN_MAP = {
        ' time': 'time',
        ' axis3_force(N)': 'force_l',
        ' axis3_pos(m)': 'pos_l',
        ' axis3_vel(m/s)': 'vel_l',
    }

    HEADER_ROWS = 17  # 设备头部信息行数

    @staticmethod
    def process(robot_file, robot_folder, folder,
                turn_position=False, start_time=0, end_time=-1):
        """
        处理机器人原始数据文件。

        Parameters
        ----------
        robot_file : str
            机器人数据文件名
        robot_folder : str
            主搜索目录
        folder : str
            实验根目录（回退搜索）
        turn_position : bool
            是否反转位置数据方向
        start_time : float
            起始时间截取（秒），0 表示不截取
        end_time : float
            结束时间截取（秒），-1 表示不截取

        Returns
        -------
        pd.DataFrame or None
            处理后的机器人数据，包含 time, pos_l, vel_l, force_l 列
        """
        try:
            # 1. 解析文件路径
            file_path = RobotOriginProcessor._resolve_file_path(
                robot_file, robot_folder, folder)
            if file_path is None:
                return None

            # 2. 读取数据（跳过头部信息）
            robot_df = pd.read_csv(
                file_path,
                skiprows=RobotOriginProcessor.HEADER_ROWS)

            # 3. 提取并重命名列
            available = [c for c in RobotOriginProcessor.COLUMN_MAP
                         if c in robot_df.columns]
            if not available:
                print(f"原始机器人数据中未找到预期列: "
                      f"{robot_df.columns.tolist()}")
                return None
            robot_data = robot_df[available].rename(
                columns=RobotOriginProcessor.COLUMN_MAP)

            # 4. 确保 time 列为数值型
            robot_data['time'] = pd.to_numeric(
                robot_data['time'], errors='coerce')
            robot_data = robot_data.dropna(
                subset=['time']).reset_index(drop=True)

            # 5. 位置调整
            if 'pos_l' in robot_data.columns:
                if turn_position:
                    robot_data['pos_l'] = 0.7 + robot_data['pos_l']
                else:
                    robot_data['pos_l'] = 0.7 - robot_data['pos_l']

            # 6. 时间截取
            if start_time and start_time > 0:
                from digitaltwin.utils.array_tools import (
                    find_nearest_idx)
                start_idx = find_nearest_idx(
                    robot_data['time'].values, start_time)
                robot_data = robot_data.iloc[
                    start_idx:].reset_index(drop=True)

            if end_time != -1 and end_time > 0:
                from digitaltwin.utils.array_tools import (
                    find_nearest_idx)
                end_idx = find_nearest_idx(
                    robot_data['time'].values, end_time)
                robot_data = robot_data.iloc[
                    :end_idx].reset_index(drop=True)

            return robot_data

        except Exception as e:
            print(f"原始机器人数据处理错误: {e}")
            return None

    @staticmethod
    def _resolve_file_path(robot_file, robot_folder, folder):
        """解析原始机器人数据文件路径"""
        if os.path.isabs(robot_file):
            if os.path.exists(robot_file):
                return robot_file
            return None
        for search_dir in [robot_folder, folder]:
            file_path = os.path.join(search_dir, robot_file)
            if os.path.exists(file_path):
                return file_path
        print(f"原始机器人文件不存在: {robot_file}")
        return None