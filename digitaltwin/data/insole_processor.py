"""
insole_processor.py

读取足底压力鞋垫数据。

支持两类文件：

1. 汇总力文件 `Medilogic Insoles-XT Force.csv`
   CSV，跳过前 3 行，
     第 1 列: time (s)
     第 2 列: value (N, 正方向向上，代表地面对人的支撑力)
   由 load() 读取。

2. 逐点压力图 `Medilogic Insoles-XT Medilogic Insoles G2. Foot.csv`
   signal_matrix 格式，由 load_pressure_map() 读取。
     第 1 行: 元数据字段名
     第 2 行: 元数据值 (frequency / count / units / cell_count_x /
              cell_count_y / cell_size_mm_x / cell_size_mm_y)
     第 3 行: 空行
     第 4 行: 列头 time,x1..xN
     之后每一帧 = cell_count_y 行 x cell_count_x 列压强，
     仅每帧第一行带时间戳，帧与帧之间以空行分隔。
   压强单位通常为 N/cm2；乘单元面积后求和即该脚总力，
   同时可由压强分布求逐帧压心 COP。

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

from digitaltwin.utils.logger import beauty_print


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

    # ------------------------------------------------------------------
    # 逐点压力图分支 (Medilogic Insoles G2. Foot.csv)
    # ------------------------------------------------------------------

    MAP_HEADER_ROWS = 4      # 元数据名 / 元数据值 / 空行 / 列头
    MAP_MIN_FORCE_N = 20.0   # 求 COP 的最小总力，低于此值视为悬空

    @staticmethod
    def _read_lines_any_encoding(file_path, verbose=True):
        '''按常见编码逐一尝试读取文本行，返回 (lines, encoding)。'''
        encodings = ('utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'gbk')
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as fh:
                    return fh.readlines(), enc
            except (UnicodeDecodeError, LookupError):
                continue
        InsoleProcessor._log(
            '  [InsoleMap] 无法以任何已知编码读取: ' + str(file_path), verbose)
        return None, None

    @staticmethod
    def read_pressure_map_header(file_path, verbose=True):
        '''
        读取逐点压力图的元数据行。

        Returns
        -------
        dict or None
            name / frequency / count / units / begin_time /
            n_cols / n_rows / cell_dx_cm / cell_dy_cm /
            cell_area_cm2 / width_cm / length_cm
        '''
        import csv

        lines, _ = InsoleProcessor._read_lines_any_encoding(
            file_path, verbose=verbose)
        if lines is None or len(lines) < 2:
            return None

        try:
            keys = next(csv.reader([lines[0].rstrip()]))
            vals = next(csv.reader([lines[1].rstrip()]))
        except Exception as e:
            InsoleProcessor._log(
                '  [InsoleMap] 元数据行解析失败: ' + str(e), verbose)
            return None

        meta_raw = {}
        for k, v in zip(keys, vals):
            meta_raw[str(k).strip()] = str(v).strip()

        def as_float(key, default=None):
            try:
                return float(meta_raw.get(key, ''))
            except (TypeError, ValueError):
                return default

        def as_int(key, default=None):
            value = as_float(key, None)
            return default if value is None else int(round(value))

        n_cols = as_int('cell_count_x')
        n_rows = as_int('cell_count_y')
        dx_mm = as_float('cell_size_mm_x')
        dy_mm = as_float('cell_size_mm_y')

        if not n_cols or not n_rows or not dx_mm or not dy_mm:
            InsoleProcessor._log(
                '  [InsoleMap] 元数据缺少网格尺寸字段: ' + str(file_path),
                verbose)
            return None

        dx_cm = dx_mm / 10.0
        dy_cm = dy_mm / 10.0

        return {
            'type': meta_raw.get('type', ''),
            'name': meta_raw.get('name', ''),
            'frequency': as_float('frequency'),
            'count': as_int('count'),
            'units': meta_raw.get('units', ''),
            'begin_time': as_float('begin_time', 0.0),
            'n_cols': n_cols,
            'n_rows': n_rows,
            'cell_dx_cm': dx_cm,
            'cell_dy_cm': dy_cm,
            'cell_area_cm2': dx_cm * dy_cm,
            'width_cm': n_cols * dx_cm,
            'length_cm': n_rows * dy_cm,
        }

    @staticmethod
    def _parse_pressure_frames(lines, n_rows, n_cols, verbose=True):
        '''
        解析帧块，返回 (time, matrix)，matrix 形状 (n_frames, n_rows, n_cols)。

        行数不完整的帧会被丢弃并计数告警。
        '''
        import csv

        times = []
        frames = []
        state = {'rows': None, 'time': None, 'bad': 0}

        def close_frame():
            rows = state['rows']
            if rows is None:
                return
            if len(rows) == n_rows and state['time'] is not None:
                times.append(state['time'])
                frames.append(rows)
            else:
                state['bad'] += 1
            state['rows'] = None
            state['time'] = None

        for line in lines:
            stripped = line.strip()
            if stripped == '' or set(stripped) <= {','}:
                close_frame()
                continue

            try:
                parts = next(csv.reader([line.rstrip()]))
            except Exception:
                continue
            if not parts:
                continue

            head = parts[0].strip()

            if head != '':
                close_frame()
                try:
                    state['time'] = float(head)
                except ValueError:
                    state['time'] = None
                    continue
                state['rows'] = []

            if state['rows'] is None:
                continue

            row = []
            for v in parts[1:1 + n_cols]:
                v = v.strip()
                try:
                    row.append(float(v) if v != '' else 0.0)
                except ValueError:
                    row.append(0.0)
            while len(row) < n_cols:
                row.append(0.0)

            state['rows'].append(row)

        close_frame()

        if state['bad']:
            beauty_print(
                '  [InsoleMap] 丢弃行数不完整的帧: {} 帧'.format(state['bad']),
                type='warning')

        if not frames:
            return None, None

        return np.asarray(times, dtype=float), np.asarray(frames, dtype=float)

    @staticmethod
    def load_pressure_map(file_path, verbose=True, use_info_timestamp=True,
                          robot_file=None, robot_folder=None, folder=None,
                          toe_first=True, min_force=None,
                          return_matrix=True):
        '''
        读取逐点压力图文件，返回总力与逐帧压心。

        Parameters
        ----------
        file_path : str
            `... Medilogic Insoles G2. Foot.csv` 完整路径。
        toe_first : bool, default True
            网格行号 0 是否位于足趾端。判定依据：最宽的行是跖骨头，
            压强最高且略窄的行是足跟。若某侧鞋垫方向相反，置为 False。
        min_force : float, optional
            求 COP 的最小总力阈值，默认 MAP_MIN_FORCE_N。
            低于阈值的帧 COP 记为 nan。
        return_matrix : bool, default True
            是否保留完整压强矩阵（较占内存）。

        Returns
        -------
        dict or None
            time     : (n,)  修正后的时间轴 (s)
            force    : (n,)  由压强积分得到的总力 (N)
            cop_ant  : (n,)  压心距足跟端的前向距离 (m)
            cop_lat  : (n,)  压心距网格第 0 列的横向距离 (m)
            pressure : (n, n_rows, n_cols) 压强，return_matrix=True 时给出
            meta     : dict，见 read_pressure_map_header
        '''
        if not os.path.exists(file_path):
            beauty_print('  [InsoleMap] 文件不存在: ' + str(file_path),
                         type='warning')
            return None

        meta = InsoleProcessor.read_pressure_map_header(
            file_path, verbose=verbose)
        if meta is None:
            beauty_print('  [InsoleMap] 元数据解析失败: ' + str(file_path),
                         type='warning')
            return None

        lines, used_enc = InsoleProcessor._read_lines_any_encoding(
            file_path, verbose=verbose)
        if lines is None:
            return None

        n_rows = meta['n_rows']
        n_cols = meta['n_cols']

        time, matrix = InsoleProcessor._parse_pressure_frames(
            lines[InsoleProcessor.MAP_HEADER_ROWS:],
            n_rows, n_cols, verbose=verbose)

        if matrix is None:
            beauty_print('  [InsoleMap] 未解析到任何完整帧: ' + str(file_path),
                         type='warning')
            return None

        declared = meta['count']
        if declared and len(time) != declared:
            beauty_print(
                '  [InsoleMap] 帧数与元数据不一致: 解析 {} / 声明 {}'.format(
                    len(time), declared),
                type='warning')

        # 压强 -> 力。units 含 cm2 时需乘单元面积
        units = str(meta.get('units', '')).lower().replace(' ', '')
        if 'cm2' in units or 'cm^2' in units:
            scale = meta['cell_area_cm2']
        elif 'mm2' in units or 'mm^2' in units:
            scale = meta['cell_dx_cm'] * meta['cell_dy_cm'] * 100.0
        else:
            scale = 1.0
            beauty_print(
                '  [InsoleMap] 未识别的单位 {}，按已是力处理，'
                '不乘单元面积'.format(meta.get('units')),
                type='warning')

        force = matrix.sum(axis=(1, 2)) * scale

        # 逐帧压心
        thr = (InsoleProcessor.MAP_MIN_FORCE_N
               if min_force is None else min_force)
        xs = (np.arange(n_cols) + 0.5) * meta['cell_dx_cm']
        ys = (np.arange(n_rows) + 0.5) * meta['cell_dy_cm']

        tot = matrix.sum(axis=(1, 2))
        with np.errstate(invalid='ignore', divide='ignore'):
            cop_row = (matrix.sum(axis=2) @ ys) / tot
            cop_col = (matrix.sum(axis=1) @ xs) / tot

        invalid = ~(force >= thr)
        cop_row[invalid] = np.nan
        cop_col[invalid] = np.nan

        # 行号 0 在足趾端时，距足跟端的前向距离 = 全长 - 行向坐标
        cop_ant_cm = (meta['length_cm'] - cop_row) if toe_first else cop_row

        # 边缘列若有明显压强，说明脚踩到垫边，力有溢出丢失
        col_mean = matrix.mean(axis=(0, 1))
        edge = max(col_mean[0], col_mean[-1])
        if col_mean.max() > 0 and edge > 0.05 * col_mean.max():
            beauty_print(
                '  [InsoleMap] 边缘列存在明显压强 (边缘 {:.3f} vs 峰值 {:.3f})，'
                '脚可能踩出感应区，总力与 COP 均会偏低'.format(
                    edge, col_mean.max()),
                type='warning')

        time = InsoleProcessor.align_time_with_info(
            time, file_path,
            robot_file=robot_file,
            robot_folder=robot_folder,
            folder=folder,
            use_info_timestamp=use_info_timestamp,
            verbose=verbose)

        n_valid = int((~invalid).sum())
        InsoleProcessor._log(
            '  [InsoleMap] 已加载 ({}): {}  ({} frames, {}x{} cells, '
            '有效 COP {} 帧)'.format(
                used_enc, os.path.basename(file_path), len(time),
                n_rows, n_cols, n_valid),
            verbose)

        result = {
            'time': time,
            'force': force,
            'cop_ant': cop_ant_cm / 100.0,
            'cop_lat': cop_col / 100.0,
            'meta': meta,
        }
        if return_matrix:
            result['pressure'] = matrix

        return result

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

    @staticmethod
    def resample_nan_safe(time, values, target_times, max_gap_s=None):
        """把含 nan 的信号（例如悬空帧的 COP）插值到目标时间轴。

        为什么不能直接用 resample：np.interp 不认识 nan，一个 nan 会沿着
        两侧的线性段污染邻域，把 COP 拉到任意位置。这里只用【有效样本】
        建插值，并且：
          - 目标点落在有效样本覆盖范围之外时返回 nan，不做外推；
          - 若某目标点两侧最近有效样本的间隔超过 max_gap_s，也返回 nan，
            避免跨过长段悬空/丢帧硬连一条直线。

        Returns
        -------
        np.ndarray or None -- 与 target_times 等长；全无有效样本时返回 None
        """
        t = np.asarray(time, dtype=float)
        v = np.asarray(values, dtype=float)
        tgt = np.asarray(target_times, dtype=float)
        n = min(t.size, v.size)
        if n < 2:
            return None
        t, v = t[:n], v[:n]

        ok = np.isfinite(t) & np.isfinite(v)
        if int(ok.sum()) < 2:
            return None
        t_ok, v_ok = t[ok], v[ok]
        order = np.argsort(t_ok)
        t_ok, v_ok = t_ok[order], v_ok[order]

        out = np.interp(tgt, t_ok, v_ok)
        out[(tgt < t_ok[0]) | (tgt > t_ok[-1])] = np.nan

        if max_gap_s is not None and max_gap_s > 0:
            idx = np.searchsorted(t_ok, tgt).clip(1, t_ok.size - 1)
            gap = t_ok[idx] - t_ok[idx - 1]
            out[gap > max_gap_s] = np.nan

        return out

    # ------------------------------------------------------------------
    # 方向诊断（足趾端 / 内外侧镜像）
    # ------------------------------------------------------------------
    #
    # 为什么不能靠看图：热图是用同一段代码、同一个 origin 画的，
    # 左右子图没有任何镜像处理。“看起来像一双脚印”只能说明两个
    # 矩阵在列方向上互为镜像，它对【行】方向（足趾在哪一端）一无所知，
    # 而行方向才是决定膝力矩力臂的那一个。所以这里用数据判，不用眼判。

    # 上一版把 ORIENT_CONTACT_FRAC=0.20 同时用于“着地”和“量宽度”，那是错的：
    # 足跟压强高，0.20 x 全局峰值是个绝对高门槛，低压的前足会被整排判成
    # 没着地，宽度判据于是退化成峰值判据的同义反复，9 组 x 2 侧全部判反。
    # 现在把两件事分开：几何足印用逐格 max over time + 很低的门槛。
    ORIENT_CONTACT_FRAC = 0.20      # 仅用于峰值旁证的“高压区”比例
    ORIENT_FOOTPRINT_FRAC = 0.05    # 二值足印门槛：全局峰值的比例
    ORIENT_FOOTPRINT_ABS = 0.5      # 二值足印门槛的绝对下限 (N/cm2)
    ORIENT_END_FRAC = 0.25          # 两端各取多少比例的行做对比
    ORIENT_MIDLINE_GUARD_CM = 3.0   # 左右交叉检验的中点保护带 (cm)

    @staticmethod
    def mean_contact_pressure(result, min_force=None):
        '''只对着地帧求平均压强 (n_rows, n_cols)。

        悬空帧全是零，计进去会把分布整体稀释，而且不同组的悬空比例
        不同，会让组与组之间无法比较。
        '''
        pressure = result.get('pressure') if result else None
        if pressure is None:
            return None

        pressure = np.asarray(pressure, dtype=float)
        force = np.asarray(result.get('force'), dtype=float)
        thr = (InsoleProcessor.MAP_MIN_FORCE_N
               if min_force is None else min_force)

        use = np.isfinite(force) & (force >= thr)
        if use.sum() < 1:
            use = np.ones(len(pressure), dtype=bool)

        return np.nanmean(pressure[use], axis=0)

    @staticmethod
    def peak_contact_pressure(result, min_force=None):
        '''逐格 max over time (n_rows, n_cols)。

        量【几何足印】只能用逐格最大值，不能用均值：一个格子只要在整段里
        被踩到过，它就属于足印。用均值再配高门槛，等于要求前足的平均压强
        达到足跟量级，前足会被整排判成没着地——这正是上一版宽度判据失效的
        机制。
        '''
        pressure = result.get('pressure') if result else None
        if pressure is None:
            return None

        pressure = np.asarray(pressure, dtype=float)
        force = np.asarray(result.get('force'), dtype=float)
        thr = (InsoleProcessor.MAP_MIN_FORCE_N
               if min_force is None else min_force)

        use = np.isfinite(force) & (force >= thr)
        if use.sum() < 1:
            use = np.ones(len(pressure), dtype=bool)

        return np.nanmax(pressure[use], axis=0)

    @staticmethod
    def _footprint_mask(peak_map, frac=None, abs_floor=None):
        '''二值足印。门槛 = max(frac x 全局峰值, 绝对下限)，并封顶在半峰值，
        免得某侧峰值特别低时绝对下限反而把整只脚都抹掉。
        '''
        if peak_map is None:
            return None
        frac = (InsoleProcessor.ORIENT_FOOTPRINT_FRAC
                if frac is None else frac)
        abs_floor = (InsoleProcessor.ORIENT_FOOTPRINT_ABS
                     if abs_floor is None else abs_floor)
        peak = float(np.nanmax(peak_map))
        if not np.isfinite(peak) or peak <= 0:
            return None
        thr = max(frac * peak, abs_floor)
        if thr > 0.5 * peak:
            thr = frac * peak
        return peak_map >= max(thr, 1e-9)

    @staticmethod
    def _contact_width_profile(peak_map, frac=None, abs_floor=None):
        '''每一行的足印宽度（格数），在二值足印上数。

        足印宽度沿长度是单峰的：最宽处是跖球，约在从足跟量起 70-75% 足长
        的位置，绝不在中点。所以“最宽行靠近哪一端”是纯几何判据，与本次
        重心前后分布无关，比拿两端互比稳健得多。
        '''
        mask = InsoleProcessor._footprint_mask(
            peak_map, frac=frac, abs_floor=abs_floor)
        if mask is None:
            return None
        return mask.sum(axis=1).astype(float)

    @staticmethod
    def diagnose_side_orientation(result, side='', min_force=None):
        '''判断单侧鞋垫的行 0 是否在足趾端，并找出足弓在哪一侧。

        三个判据，按可靠性从高到低：
          1.【最宽行位置】主判据。最宽处 = 跖球，位于从足跟量起约 70-75%
            足长处，所以最宽行靠近哪一端，那一端就是前足。纯几何。
          2.【两端宽度对比】副判据，在二值足印上量，仅作交叉印证。
          3.【峰值压强端】旁证。足跟通常压强最高，但深蹲重心前移时前足
            完全可以反超，所以它不能单独作判据。

        Returns
        -------
        dict or None
            toe_first_suggest  : bool  -- 主判据结论
            widest_row         : int
            widest_row_frac    : float -- 最宽行在着地范围内的归一化位置
            toe_first_by_width : bool  -- 副判据
            toe_first_by_peak  : bool  -- 旁证
            agree              : bool  -- 主副判据是否一致
            contact_row_first / contact_row_last : int -- 实际着地行范围
            arch_col_side      : str   -- 'low' / 'high'，足弓偏向列号小还是列号大的一侧
        '''
        mean_p = InsoleProcessor.mean_contact_pressure(
            result, min_force=min_force)
        peak_map = InsoleProcessor.peak_contact_pressure(
            result, min_force=min_force)
        width = InsoleProcessor._contact_width_profile(peak_map)
        if mean_p is None or peak_map is None or width is None:
            return None

        n_rows, n_cols = peak_map.shape
        k = max(1, int(round(n_rows * InsoleProcessor.ORIENT_END_FRAC)))

        w_head = float(np.nanmean(width[:k]))
        w_tail = float(np.nanmean(width[-k:]))
        p_head = float(np.nanmax(mean_p[:k]))
        p_tail = float(np.nanmax(mean_p[-k:]))

        # 主判据：最宽行在【实际着地行范围】内的位置。垫比脚长，两端总有
        # 几行从不受力，把它们算进去会把归一化位置往中间拉。
        rows = np.flatnonzero(width > 0)
        if rows.size >= 3:
            r0, r1 = int(rows[0]), int(rows[-1])
        else:
            r0, r1 = 0, n_rows - 1
        sub = width[r0:r1 + 1]
        widest_row = int(r0 + int(np.argmax(sub)))
        span_rows = max(r1 - r0, 1)
        widest_frac = float((widest_row - r0) / span_rows)
        toe_first_suggest = bool(widest_frac < 0.5)

        # 副判据 / 旁证
        toe_first_by_width = w_head > w_tail
        toe_first_by_peak = p_tail > p_head

        # 足弓：中足段压强低的那半边是内侧。这是区分左右镜像的唯一
        # 解剖学锚点——外侧缘总是连续承重，内侧中段悬空。
        lo = int(round(n_rows / 3.0))
        hi = int(round(n_rows * 2.0 / 3.0))
        mid = mean_p[lo:hi] if hi > lo else mean_p
        half = n_cols // 2
        c_low = float(np.nanmean(mid[:, :half]))
        c_high = float(np.nanmean(mid[:, half:]))
        arch_col_side = 'low' if c_low < c_high else 'high'

        return {
            'side': side,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'width_head': w_head,
            'width_tail': w_tail,
            'peak_head': p_head,
            'peak_tail': p_tail,
            'widest_row': widest_row,
            'widest_row_frac': widest_frac,
            'contact_row_first': r0,
            'contact_row_last': r1,
            'toe_first_suggest': toe_first_suggest,
            'toe_first_by_width': bool(toe_first_by_width),
            'toe_first_by_peak': bool(toe_first_by_peak),
            'agree': bool(toe_first_by_width == toe_first_suggest),
            'midfoot_low_cols': c_low,
            'midfoot_high_cols': c_high,
            'arch_col_side': arch_col_side,
        }

    @staticmethod
    def diagnose_orientation(side_results, toe_first_used=True,
                             min_force=None, verbose=True):
        '''交叉比对左右鞋垫的行/列方向，给出应该用什么 toe_first。

        Parameters
        ----------
        side_results : dict -- {'l': result, 'r': result}
        toe_first_used : bool or dict -- 读取时实际传给 load_pressure_map
            的 toe_first。cop_ant 已经受它影响，不告诉这个函数就无法反推。

        Returns
        -------
        dict -- 包含逐侧结果与两项交叉检验
        '''
        out = {'sides': {}, 'issues': []}

        if isinstance(toe_first_used, dict):
            used = dict(toe_first_used)
        else:
            used = {'l': bool(toe_first_used), 'r': bool(toe_first_used)}

        for s in ('l', 'r'):
            res = (side_results or {}).get(s)
            if res is None:
                continue
            info = InsoleProcessor.diagnose_side_orientation(
                res, side=s, min_force=min_force)
            if info is None:
                continue
            info['toe_first_used'] = used.get(s, True)
            # 与【主判据】比，不是与副判据比
            info['match'] = (info['toe_first_suggest']
                             == info['toe_first_used'])
            out['sides'][s] = info

        if verbose:
            print('  [Orient] 主判据 = 最宽行位置 (跖球在足长 70-75% 处)，'
                  'wide_frac < 0.5 -> 行 0 在足趾端')
            print('    {:<5}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}'
                  .format('side', 'wide_row', 'wide_frac', 'w_head',
                          'w_tail', 'p_ratio', 'suggest', 'used'))
            for s, i in out['sides'].items():
                p_ratio = (i['peak_tail'] / i['peak_head']
                           if i['peak_head'] > 1e-9 else float('nan'))
                print('    {:<5}{:>10d}{:>10.2f}{:>10.2f}{:>10.2f}'
                      '{:>10.2f}{:>10}{:>10}'.format(
                          s.upper(), int(i['widest_row']),
                          i['widest_row_frac'], i['width_head'],
                          i['width_tail'], p_ratio,
                          str(i['toe_first_suggest']),
                          str(i['toe_first_used'])))

        for s, i in out['sides'].items():
            # 副判据/旁证与主判据不一致【不】算错误：深蹲重心前移时前足压强
            # 反超足跟很常见，两端宽度也受门槛影响。只打印，不告警。
            if verbose and not i['agree']:
                print('    [Orient] {} 侧副判据(两端宽度)与主判据不一致，'
                      '以主判据(最宽行位置)为准。'.format(s.upper()))
            if verbose and (i['toe_first_by_peak'] != i['toe_first_suggest']):
                print('    [Orient] {} 侧峰值旁证与主判据不一致，属常见现象'
                      '(深蹲重心前移使前足压强反超足跟)。'.format(s.upper()))
            if not i['match']:
                out['issues'].append('{}: toe_first 与主判据不符'.format(s))
                beauty_print(
                    '  [Orient] {} 侧 toe_first 传的是 {}，但最宽行位置'
                    '(wide_frac={:.2f}) 指示应该是 {}。COP 前后方向会整个翻转，'
                    '膝力矩结论会反过来，请核对热图后再继续。'.format(
                        s.upper(), i['toe_first_used'],
                        i['widest_row_frac'], i['toe_first_suggest']),
                    type='warning')

        # 交叉检验：左右运动学已确认对称，两脚的 COP 应当落在相似位置。
        # 若两侧之和接近全长/全宽，就是典型的“有一侧被翻转了”。
        res_l = (side_results or {}).get('l')
        res_r = (side_results or {}).get('r')
        if res_l is not None and res_r is not None:
            meta = res_l['meta']
            for field, span, name in (
                    ('cop_ant', meta['length_cm'], '前后 (行)'),
                    ('cop_lat', meta['width_cm'], '内外 (列)')):
                a = np.asarray(res_l.get(field), dtype=float) * 100.0
                b = np.asarray(res_r.get(field), dtype=float) * 100.0
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if a.size == 0 or b.size == 0:
                    continue
                ma, mb = float(a.mean()), float(b.mean())
                d_same = abs(ma - mb)
                d_flip = abs(ma + mb - float(span))
                # 中点保护：这个检验的前提是 COP 远离垫子中点。若两侧 COP
                # 都贴着 span/2，L+R≈span 是代数必然，与是否翻转无关，判成
                # mirrored 纯属假阳性（上一版 56/75 组就是这么误报的）。
                guard = InsoleProcessor.ORIENT_MIDLINE_GUARD_CM
                mid = float(span) / 2.0
                near_mid = (abs(ma - mid) < guard) and (abs(mb - mid) < guard)
                if near_mid:
                    verdict = 'inconclusive'
                else:
                    verdict = 'same' if d_same <= d_flip else 'mirrored'
                out[field] = {
                    'mean_l': ma, 'mean_r': mb, 'span': float(span),
                    'diff_same': d_same, 'diff_flip': d_flip,
                    'near_midline': bool(near_mid),
                    'verdict': verdict,
                }
                if verbose:
                    note = ('（两侧 COP 都在中点 ±{:.0f}cm 内，此检验无判别力）'
                            .format(guard)) if near_mid else ''
                    print('  [Orient] {} 方向: L={:.2f} R={:.2f} cm, '
                          '全长={:.2f} | |L-R|={:.2f} vs |L+R-全长|={:.2f} '
                          '-> {}{}'.format(name, ma, mb, span,
                                           d_same, d_flip, verdict, note))

            if out.get('cop_ant', {}).get('verdict') == 'mirrored':
                out['issues'].append('前后方向左右不一致')
                beauty_print(
                    '  [Orient] 两侧前后 COP 之和接近鞋垫全长，说明有一侧的行'
                    '顺序是反的。运动学已确认左右对称，两脚 COP 不应该差这么多。'
                    '必须先修正再做逐帧 COP。',
                    type='warning')

            sl = out['sides'].get('l', {}).get('arch_col_side')
            sr = out['sides'].get('r', {}).get('arch_col_side')
            if sl and sr:
                out['column_frame'] = ('world' if sl != sr else 'anatomical')
                out['column_frame_by_side'] = (sl, sr)
                # 汇总必须写实际判定值：上一版汇总说 anatomical、逐组打印说
                # world，就是因为这里只在 anatomical 时才留下痕迹。
                out.setdefault('notes', []).append(
                    '列坐标为 {} 坐标系 (L 足弓在列号{}侧, R 在列号{}侧)'.format(
                        out['column_frame'], sl, sr))
                if verbose:
                    print('  [Orient] 足弓位置: L 在列号{}侧, R 在列号{}侧 -> '
                          '列坐标为 {} 坐标系'.format(
                              sl, sr, out['column_frame']))
                if out['column_frame'] == 'anatomical':
                    beauty_print(
                        '  [Orient] 两侧足弓在同一侧列号，说明文件按每只脚自己的'
                        '解剖方向存储。换算到模型的左右轴时需要把其中一侧列翻转，'
                        '否则额状面力矩（高内收）会错。矢状面膝力矩不受影响。',
                        type='warning')

        if verbose and not out['issues']:
            print('  [Orient] 未发现方向问题。')

        return out