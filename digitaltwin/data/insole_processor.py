"""
insole_processor.py

读取足底压力鞋垫数据。

文件格式：CSV，跳过前 3 行，
  第 1 列: time (s)
  第 2 列: value (N, 正方向向上，代表地面对人的支撑力)
"""
import os
import numpy as np


class InsoleProcessor:

    @staticmethod
    def load(file_path, verbose=True):
        """
        读取单个鞋垫 CSV 文件。

        Parameters
        ----------
        file_path : str  -- CSV 文件完整路径
        verbose   : bool

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
            log(f'  [Insole] 已加载 ({used_enc}): {os.path.basename(file_path)}  ({len(data)} frames)')
            return data[:, 0], data[:, 1]
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