"""
insole_processor.py

读取足底压力鞋垫数据。

文件格式：CSV，跳过前 3 行，
  第 1 列: time (s)
  第 2 列: value (N, 正方向向上，代表地面对人的支撑力)

默认时间处理：
  - 先读取鞋垫文件同目录下的 info.csv；
  - 从 I2 或 measurement_date 字段读取测量起始时间；
  - 读取 robot_file 第一帧时间作为对齐零点；
  - 由于电脑系统日期 / 小时可能设置错误，对齐时只比较
    “分钟:秒.毫秒”，忽略日期和小时；
  - 将鞋垫 time 修正为相对 robot 第一帧的时间。
"""
import os
import numpy as np
import pandas as pd


class InsoleProcessor:
    @staticmethod
    def _log(msg, verbose=True):
        if verbose:
            print(msg)

    @staticmethod
    def _read_csv_or_excel(path):
        """读取 csv / xlsx，用于解析 robot 第一帧时间。"""
        if path is None:
            return None
        if path.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(path)
        return pd.read_csv(path)

    @staticmethod
    def _resolve_robot_path(robot_file, robot_folder=None, folder=None):
        """解析 robot_file 的完整路径。"""
        if not robot_file:
            return None

        try:
            from digitaltwin.data.robot_processor import RobotProcessor
            return RobotProcessor._resolve_file_path(
                robot_file, robot_folder or '', folder or '')
        except Exception:
            pass

        if os.path.isabs(robot_file):
            return robot_file if os.path.exists(robot_file) else None

        candidates = []
        if robot_folder:
            candidates.append(os.path.join(robot_folder, robot_file))
        if folder:
            candidates.append(os.path.join(folder, robot_file))
        candidates.append(robot_file)

        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    @staticmethod
    def get_robot_first_timestamp(robot_file, robot_folder=None, folder=None,
                                  verbose=True):
        """
        读取 robot_file 第一帧时间。

        优先使用 Timestamp / Time / TimeStamp / time 列。
        后续对齐只使用“分钟:秒.毫秒”，忽略日期和小时。
        """
        robot_path = InsoleProcessor._resolve_robot_path(
            robot_file, robot_folder=robot_folder, folder=folder)
        if robot_path is None:
            InsoleProcessor._log(
                f'  [Insole] 找不到 robot_file: {robot_file}', verbose)
            return None

        try:
            df = InsoleProcessor._read_csv_or_excel(robot_path)
        except Exception as e:
            InsoleProcessor._log(
                f'  [Insole] 读取 robot_file 失败: {e}', verbose)
            return None

        if df is None:
            return None

        for col in ['Timestamp', 'Time', 'TimeStamp', 'time']:
            if col not in df.columns:
                continue

            vals = df[col].dropna()
            if len(vals) == 0:
                continue

            ts = pd.to_datetime(vals.iloc[0], utc=True, errors='coerce')
            if pd.notna(ts):
                return ts

        InsoleProcessor._log(
            '  [Insole] robot_file 未找到可解析的时间戳', verbose)
        return None

    @staticmethod
    def read_measurement_date_from_info(insole_path, verbose=True):
        """
        从鞋垫文件同文件夹的 info.csv 中读取 measurement_date。

        兼容：
          1. I2：第 2 行、第 9 列；
          2. 第一行为表头且存在 measurement_date 列；
          3. 单元格为 measurement_date，取其下方或右侧单元格。
        """
        info_path = os.path.join(os.path.dirname(insole_path), 'info.csv')
        if not os.path.exists(info_path):
            InsoleProcessor._log(
                f'  [Insole] 未找到 info.csv: {info_path}', verbose)
            return None

        try:
            raw = pd.read_csv(info_path, header=None, dtype=str)
        except Exception as e:
            InsoleProcessor._log(
                f'  [Insole] 读取 info.csv 失败: {e}', verbose)
            return None

        candidates = []

        # I2 = row index 1, col index 8
        if raw.shape[0] >= 2 and raw.shape[1] >= 9:
            candidates.append(raw.iat[1, 8])

        # 扫描 measurement_date 单元格，优先取下方，其次取右侧
        for r in range(raw.shape[0]):
            for c in range(raw.shape[1]):
                cell = str(raw.iat[r, c]).strip()
                if cell != 'measurement_date':
                    continue
                if r + 1 < raw.shape[0]:
                    candidates.append(raw.iat[r + 1, c])
                if c + 1 < raw.shape[1]:
                    candidates.append(raw.iat[r, c + 1])

        # 按第一行为表头读取
        try:
            table = pd.read_csv(info_path, dtype=str)
            if 'measurement_date' in table.columns and len(table) > 0:
                candidates.append(table.loc[0, 'measurement_date'])
        except Exception:
            pass

        for value in candidates:
            if value is None:
                continue
            value = str(value).strip()
            if value == '' or value.lower() == 'nan':
                continue

            ts = pd.to_datetime(value, utc=True, errors='coerce')
            if pd.notna(ts):
                return ts

        InsoleProcessor._log(
            f'  [Insole] info.csv 中未解析到 measurement_date: {info_path}',
            verbose)
        return None

    @staticmethod
    def _minute_second_of_hour(ts):
        """
        只取时间戳中的“分钟:秒.毫秒”，忽略日期和小时。
        """
        return (
            ts.minute * 60.0
            + ts.second
            + ts.microsecond / 1e6
            + ts.nanosecond / 1e9
        )

    @staticmethod
    def _offset_by_minute_second(measurement_ts, robot_start_ts):
        """
        计算 measurement_date 相对 robot 第一帧的偏移，只比较分钟及之后。

        如果刚好跨过小时边界，例如 robot=10:59:58、insole=11:00:02，
        直接相减会得到 -3596s。这里将偏移折回到 [-1800, 1800]，
        得到更合理的 +4s。
        """
        offset_s = (
            InsoleProcessor._minute_second_of_hour(measurement_ts)
            - InsoleProcessor._minute_second_of_hour(robot_start_ts)
        )

        if offset_s > 1800.0:
            offset_s -= 3600.0
        elif offset_s < -1800.0:
            offset_s += 3600.0

        return float(offset_s)

    @staticmethod
    def align_time_with_info(time, insole_path, robot_file=None,
                             robot_folder=None, folder=None,
                             use_info_timestamp=True, verbose=True):
        """
        按 info.csv measurement_date + robot_file 第一帧时间修正鞋垫时间轴。

        默认 use_info_timestamp=True。若置为 False，直接返回原始 time。
        对齐时忽略日期和小时，只比较“分钟:秒.毫秒”。
        """
        if not use_info_timestamp:
            return time

        measurement_ts = InsoleProcessor.read_measurement_date_from_info(
            insole_path, verbose=verbose)
        robot_start_ts = InsoleProcessor.get_robot_first_timestamp(
            robot_file, robot_folder=robot_folder, folder=folder,
            verbose=verbose)

        if measurement_ts is None or robot_start_ts is None:
            InsoleProcessor._log(
                '  [Insole] measurement_date 或 robot 第一帧时间缺失，'
                '退回使用鞋垫文件原始相对时间',
                verbose)
            return time

        offset_s = InsoleProcessor._offset_by_minute_second(
            measurement_ts, robot_start_ts)
        aligned_time = np.asarray(time, dtype=float) + offset_s

        InsoleProcessor._log(
            f'  [Insole] 时间修正: measurement_date={measurement_ts.isoformat()}, '
            f'robot_start={robot_start_ts.isoformat()}, '
            f'offset_min_sec={offset_s:.3f}s (忽略日期和小时)',
            verbose)

        return aligned_time

    @staticmethod
    def load(file_path, verbose=True, use_info_timestamp=True,
             robot_file=None, robot_folder=None, folder=None):
        """
        读取单个鞋垫 CSV 文件。

        Parameters
        ----------
        file_path : str
            CSV 文件完整路径。
        verbose : bool
        use_info_timestamp : bool, default True
            是否按 info.csv measurement_date + robot_file 第一帧时间
            修正鞋垫时间轴。可随时置为 False 退回原始时间。
        robot_file, robot_folder, folder : str, optional
            用于定位并读取 robot_file 第一帧时间。若缺失，则自动退回原始时间。

        Returns
        -------
        time  : np.ndarray or None
        force : np.ndarray or None  -- +Y 向上的地面支撑力 (N)
        """
        def log(msg):
            if verbose:
                print(msg)

        if not os.path.exists(file_path):
            log(f'  [Insole] 文件不存在: {file_path}')
            return None, None

        import io

        # 逐一尝试常见编码；鞋垫软件可能输出 UTF-8 (with/without BOM)、
        # Latin-1 或 CP1252，Windows 中文环境还可能是 GBK
        encodings = ('utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'gbk')
        raw_lines = None
        used_enc  = None
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as fh:
                    raw_lines = fh.readlines()
                used_enc = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if raw_lines is None:
            log(f'  [Insole] 无法以任何已知编码读取: {file_path}')
            return None, None

        try:
            # 跳过前 3 行表头
            content = ''.join(raw_lines[3:])
            data = np.genfromtxt(io.StringIO(content), delimiter=',',
                                 usecols=(0, 1), invalid_raise=False)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            data = data[~np.isnan(data).any(axis=1)]
            if len(data) == 0:
                log(f'  [Insole] 文件无有效数据: {file_path}')
                return None, None
            time = data[:, 0]
            force = data[:, 1]
            time = InsoleProcessor.align_time_with_info(
                time, file_path,
                robot_file=robot_file,
                robot_folder=robot_folder,
                folder=folder,
                use_info_timestamp=use_info_timestamp,
                verbose=verbose)

            log(f'  [Insole] 已加载 ({used_enc}): {os.path.basename(file_path)}  ({len(data)} frames)')
            return time, force
        except Exception as e:
            log(f'  [Insole] 读取失败: {e}')
            return None, None

    @staticmethod
    def resample(time, force, target_times):
        """
        将鞋垫力信号线性插值到目标时间轴。

        Parameters
        ----------
        time         : np.ndarray -- 原始时间轴
        force        : np.ndarray -- 对应力值
        target_times : np.ndarray -- 目标时间轴

        Returns
        -------
        np.ndarray  -- 与 target_times 等长
        """
        return np.interp(target_times, time, force,
                         left=force[0], right=force[-1])