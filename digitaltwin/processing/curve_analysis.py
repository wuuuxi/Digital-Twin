import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import curve_fit


class CurveAnalyzer:
    """
    曲线分析器
    负责处理对齐后的数据，计算位置-肌肉激活平均曲线
    支持插值法和分箱法两种计算方式
    """

    def __init__(self, debug_label=False):
        self.debug_label = debug_label

    def print_debug_info(self, message, print_label=True):
        if print_label:
            print(f"[CurveAnalyzer] {message}")

    def process_for_curves(self, data_df, print_label=False):
        """
        处理对齐后的数据，计算每个负载下、每个运动阶段的位置-肌肉激活平均曲线

        Parameters:
        -----------
        data_df : pd.DataFrame
            经过 DataAligner.cut_aligned_data 处理后的数据
        print_label : bool
            是否打印调试信息

        Returns:
        -----------
        dict
            {load: {phase: {'emg_curves': {...}, 'vel_curves': {...}, 'n_segments': int}}}
        """
        required_columns = ['pos_l', 'movement_phase']
        for col in required_columns:
            if col not in data_df.columns:
                raise ValueError(f"数据框中缺少必要列: {col}")

        emg_columns = [col for col in data_df.columns if col.startswith('EMG') or col.startswith('emg')]
        if not emg_columns:
            raise ValueError("数据框中未找到EMG数据列")

        if print_label:
            print(f"找到 {len(emg_columns)} 个EMG通道: {emg_columns}")

        if 'load' not in data_df.columns:
            print("警告: 数据框中没有负载信息，将使用单一负载处理")
            data_df = data_df.copy()
            data_df['load'] = 'default_load'

        unique_loads = data_df['load_weight'].unique()
        unique_phases = data_df['movement_phase'].unique()

        if print_label:
            print(f"找到 {len(unique_loads)} 个负载: {unique_loads}")
            print(f"找到 {len(unique_phases)} 个运动阶段: {unique_phases}")

        results_dict = {}

        for load in unique_loads:
            load_data = data_df[data_df['load_weight'] == load]
            load_results = {}

            for phase in unique_phases:
                phase_data = load_data[load_data['movement_phase'] == phase]

                if len(phase_data) == 0:
                    print(f"负载 {load} 的运动阶段 {phase} 没有数据")
                    continue

                segment_ids = phase_data['segment_id'].unique()
                print(f"负载 {load}, 阶段 {phase}: 找到 {len(segment_ids)} 个片段")

                all_segments_pos = []
                all_segments_vel = []

                for seg_id in segment_ids:
                    segment_data = phase_data[phase_data['segment_id'] == seg_id]
                    if len(segment_data) < 5:
                        continue
                    all_segments_pos.append(segment_data['pos_l'].values)
                    all_segments_vel.append(segment_data['vel_l'].values)

                if len(all_segments_pos) < 2:
                    print(f"负载 {load}, 阶段 {phase}: 有效片段太少 ({len(all_segments_pos)})")
                    continue

                # 对每个EMG通道进行处理
                emg_results = {}
                for emg_channel in emg_columns:
                    all_segments_emg = []
                    for seg_id in segment_ids:
                        segment_data = phase_data[phase_data['segment_id'] == seg_id]
                        if len(segment_data) < 5:
                            continue
                        all_segments_emg.append(segment_data[emg_channel].values)

                    avg_curve_data = self.calculate_average_curve_binning(
                        all_segments_pos, all_segments_emg,
                        n_bins=15, binning_method='percentile', min_samples_per_bin=3
                    )

                    if avg_curve_data is not None:
                        emg_results[emg_channel] = avg_curve_data

                vel_results = self.calculate_average_curve_binning(
                    all_segments_pos, all_segments_vel,
                    n_bins=15, binning_method='percentile', min_samples_per_bin=3
                )

                if emg_results:
                    load_results[phase] = {
                        'emg_curves': emg_results,
                        'vel_curves': vel_results,
                        'n_segments': len(segment_ids),
                        'segment_ids': segment_ids.tolist()
                    }

            if load_results:
                results_dict[load] = load_results

        return results_dict

    def normalize_position_range(self, position_array, phase='concentric'):
        """
        将位置数据归一化到0-100%范围

        Parameters:
        -----------
        position_array : np.array
            原始位置数组
        phase : str
            运动阶段 ('concentric' 或 'eccentric')

        Returns:
        -----------
        np.array
            归一化后的位置数组（0-100%）
        """
        min_pos = np.min(position_array)
        max_pos = np.max(position_array)

        if max_pos == min_pos:
            return np.zeros_like(position_array)

        normalized = 100 * (position_array - min_pos) / (max_pos - min_pos)

        if phase.lower() == 'concentric':
            if normalized[0] > normalized[-1]:
                normalized = 100 - normalized
        elif phase.lower() == 'eccentric':
            if normalized[0] < normalized[-1]:
                normalized = 100 - normalized

        return normalized

    def calculate_average_curve_interp(self, positions_list, emg_list, n_grid_points=20,
                                       curve_model='polynomial', polynomial_degree=4):
        """
        使用插值法计算多个片段数据的平均曲线

        Parameters:
        -----------
        positions_list : list of np.array
            多个片段的位置数据列表
        emg_list : list of np.array
            多个片段的EMG数据列表
        n_grid_points : int
            插值网格点数
        curve_model : str
            曲线模型类型: 'polynomial', 'gaussian', 'spline'
        polynomial_degree : int
            多项式拟合阶数

        Returns:
        -----------
        dict or None
            包含平均曲线数据的字典
        """
        all_min_pos = []
        all_max_pos = []

        for pos in positions_list:
            if len(pos) > 0:
                all_min_pos.append(np.min(pos))
                all_max_pos.append(np.max(pos))

        if not all_min_pos:
            return None

        common_min = np.min(all_min_pos)
        common_max = np.max(all_max_pos)
        standard_grid = np.linspace(common_min, common_max, n_grid_points)

        interpolated_emg = []
        valid_segments = []

        for i, (pos, emg) in enumerate(zip(positions_list, emg_list)):
            if len(pos) < 2 or len(emg) < 2:
                continue

            unique_mask = np.concatenate(([True], np.diff(pos) != 0))
            pos_unique = pos[unique_mask]
            emg_unique = emg[unique_mask]

            if len(pos_unique) < 2:
                continue

            try:
                f_interp = interp1d(pos_unique, emg_unique, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
                emg_interp = f_interp(standard_grid)
                interpolated_emg.append(emg_interp)
                valid_segments.append(i)
            except Exception as e:
                print(f"片段 {i} 插值失败: {e}")
                continue

        if len(interpolated_emg) < 2:
            print(f"有效片段太少 ({len(interpolated_emg)})，无法计算平均曲线")
            return None

        interpolated_array = np.array(interpolated_emg)
        mean_emg = np.nanmean(interpolated_array, axis=0)
        std_emg = np.nanstd(interpolated_array, axis=0, ddof=1)
        sem_emg = std_emg / np.sqrt(len(interpolated_emg))

        fitted_curve = None
        fit_params = None
        r_squared = None
        fit_success = False

        valid_mask = ~np.isnan(mean_emg)
        x_valid = standard_grid[valid_mask]
        y_valid = mean_emg[valid_mask]

        if len(x_valid) < 5:
            print("有效数据点太少，跳过曲线拟合")
            fitted_curve = mean_emg
        else:
            try:
                if curve_model == 'polynomial':
                    coeffs = np.polyfit(x_valid, y_valid, polynomial_degree)
                    fitted_curve = np.polyval(coeffs, standard_grid)
                    fit_params = coeffs.tolist()
                    fit_success = True

                elif curve_model == 'gaussian':
                    def gaussian_func(x, a, b, c, d):
                        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + d

                    initial_guess = [
                        np.max(y_valid) - np.min(y_valid),
                        x_valid[np.argmax(y_valid)],
                        20,
                        np.min(y_valid)
                    ]
                    popt, _ = curve_fit(gaussian_func, x_valid, y_valid, p0=initial_guess, maxfev=5000)
                    fitted_curve = gaussian_func(standard_grid, *popt)
                    fit_params = popt.tolist()
                    fit_success = True

                elif curve_model == 'spline':
                    from scipy.interpolate import UnivariateSpline
                    spline = UnivariateSpline(x_valid, y_valid, s=len(x_valid) * 0.5)
                    fitted_curve = spline(standard_grid)
                    fit_params = spline.get_coeffs().tolist()
                    fit_success = True

                if fit_success and fitted_curve is not None:
                    y_pred = fitted_curve[valid_mask]
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    if ss_tot > 0:
                        r_squared = 1 - (ss_res / ss_tot)

            except Exception as e:
                print(f"曲线拟合失败: {e}")
                fitted_curve = mean_emg

        return {
            'centers': standard_grid.tolist(),
            'means': mean_emg.tolist(),
            'stds': std_emg.tolist(),
            'sems': sem_emg.tolist(),
            'fitted_curve': fitted_curve.tolist() if fitted_curve is not None else None,
            'fit_parameters': fit_params,
            'r_squared': r_squared,
            'n_segments': len(interpolated_emg),
            'valid_segment_indices': valid_segments,
            'curve_model': curve_model
        }

    def calculate_average_curve_binning(self, positions_list, emg_list, n_bins=20,
                                        binning_method='percentile', min_samples_per_bin=2):
        """
        使用分箱方法计算平均曲线

        Parameters:
        -----------
        positions_list : list of np.array
            多个片段的位置数据列表
        emg_list : list of np.array
            多个片段的EMG数据列表
        n_bins : int
            位置区间的数量
        binning_method : str
            分箱方法：'uniform' 或 'percentile'
        min_samples_per_bin : int
            每个区间最少需要的样本数

        Returns:
        -----------
        dict or None
            包含平均曲线数据的字典
        """
        all_positions = []
        all_emg = []
        segment_ids = []

        for i, (pos, emg) in enumerate(zip(positions_list, emg_list)):
            if len(pos) == 0 or len(emg) == 0:
                continue
            min_len = min(len(pos), len(emg))
            all_positions.extend(pos[:min_len])
            all_emg.extend(emg[:min_len])
            segment_ids.extend([i] * min_len)

        if len(all_positions) < n_bins * 2:
            print(f"数据点太少 ({len(all_positions)})，无法进行分箱")
            return None

        all_positions = np.array(all_positions)
        all_emg = np.array(all_emg)
        segment_ids = np.array(segment_ids)

        # 确定位置区间边界
        if binning_method == 'uniform':
            pos_min, pos_max = np.min(all_positions), np.max(all_positions)
            bin_edges = np.linspace(pos_min, pos_max, n_bins + 1)
        elif binning_method == 'percentile':
            bin_edges = np.percentile(all_positions, np.linspace(0, 100, n_bins + 1))
        else:
            raise ValueError(f"未知的分箱方法: {binning_method}")

        # 对每个区间计算统计量
        bin_centers = []
        bin_means = []
        bin_stds = []
        bin_sems = []
        bin_counts = []
        bin_segments = []

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (all_positions >= bin_edges[i]) & (all_positions <= bin_edges[i + 1])
            else:
                mask = (all_positions >= bin_edges[i]) & (all_positions < bin_edges[i + 1])

            bin_emg = all_emg[mask]
            bin_positions = all_positions[mask]

            if len(bin_positions) > 0:
                bin_center = np.mean(bin_positions)
            else:
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2

            emg_mean = np.nanmean(bin_emg)
            emg_std = np.nanstd(bin_emg, ddof=1) if len(bin_emg) > 1 else 0
            emg_sem = emg_std / np.sqrt(len(bin_emg)) if len(bin_emg) > 1 else 0
            unique_segs = np.unique(segment_ids[mask])

            bin_centers.append(bin_center)
            bin_means.append(emg_mean)
            bin_stds.append(emg_std)
            bin_sems.append(emg_sem)
            bin_counts.append(len(bin_emg))
            bin_segments.append(len(unique_segs))

        # 删除两端样本数不足的区间
        if len(bin_counts) > 0:
            left_trim = 0
            for i in range(len(bin_counts)):
                if bin_counts[i] < min_samples_per_bin:
                    left_trim = i + 1
                else:
                    break

            right_trim = len(bin_counts)
            for i in range(len(bin_counts) - 1, -1, -1):
                if bin_counts[i] < min_samples_per_bin:
                    right_trim = i
                else:
                    break

            if left_trim < right_trim:
                bin_centers = bin_centers[left_trim:right_trim]
                bin_means = bin_means[left_trim:right_trim]
                bin_stds = bin_stds[left_trim:right_trim]
                bin_sems = bin_sems[left_trim:right_trim]
                bin_counts = bin_counts[left_trim:right_trim]
                bin_segments = bin_segments[left_trim:right_trim]
            else:
                print("没有足够的有效区间")
                return None

        if len(bin_centers) < 3:
            print(f"有效区间太少 ({len(bin_centers)})，无法形成曲线")
            return None

        # 平滑插值
        try:
            spline_mean = CubicSpline(bin_centers, bin_means)
            smooth_positions = np.linspace(min(bin_centers), max(bin_centers), 100)
            smooth_means = spline_mean(smooth_positions)

            degree = min(3, len(bin_centers) - 1)
            coeffs = np.polyfit(bin_centers, bin_means, degree)
            fitted_curve = np.polyval(coeffs, smooth_positions)

            y_pred = np.polyval(coeffs, bin_centers)
            ss_res = np.sum((np.array(bin_means) - y_pred) ** 2)
            ss_tot = np.sum((np.array(bin_means) - np.mean(bin_means)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else None

        except Exception as e:
            print(f"曲线平滑失败: {e}")
            smooth_positions = bin_centers
            smooth_means = bin_means
            fitted_curve = None
            r_squared = None
            coeffs = None

        return {
            'centers': bin_centers,
            'means': bin_means,
            'stds': bin_stds,
            'sems': bin_sems,
            'counts': bin_counts,
            'segments': bin_segments,
            'smooth_positions': (smooth_positions.tolist()
                                 if isinstance(smooth_positions, np.ndarray) else smooth_positions),
            'smooth_means': (smooth_means.tolist()
                             if isinstance(smooth_means, np.ndarray) else smooth_means),
            'fitted_curve': fitted_curve.tolist() if fitted_curve is not None else None,
            'fit_parameters': coeffs.tolist() if coeffs is not None else None,
            'r_squared': r_squared,
            'n_total_points': len(all_positions),
            'n_bins': len(bin_centers),
            'binning_method': binning_method
        }