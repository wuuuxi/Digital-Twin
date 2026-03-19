import numpy as np
import pandas as pd


class DataAligner:
    """
    数据对齐器
    负责机器人数据与EMG数据的时间对齐、运动切片和位置标准化
    """

    def __init__(self, debug_label=False):
        self.debug_label = debug_label

    def print_debug_info(self, message, print_label=True):
        if print_label:
            print(f"[DataAligner] {message}")

    def align_robot_emg(self, robot_data, emg_data):
        """
        对齐机器人数据和EMG数据（最近邻时间匹配）

        Parameters:
        -----------
        robot_data : pd.DataFrame
            机器人数据
        emg_data : dict
            EMG数据字典，包含 'time' 和 'norm_signals'

        Returns:
        -----------
        pd.DataFrame
            对齐后的DataFrame
        """
        try:
            robot_time = robot_data['time'].values
            emg_time = emg_data['time']

            aligned_emg = {}
            for muscle_name, norm_signal in emg_data['norm_signals'].items():
                aligned_signal = []
                for t in robot_time:
                    idx = np.argmin(np.abs(emg_time - t))
                    aligned_signal.append(norm_signal[idx])
                aligned_emg[muscle_name] = np.array(aligned_signal)

            aligned_df = robot_data.copy()
            for muscle_name, signal in aligned_emg.items():
                aligned_df[f'emg_{muscle_name}'] = signal

            aligned_df['time_robot'] = robot_time
            aligned_df['time_aligned'] = True

            return aligned_df

        except Exception as e:
            self.print_debug_info(f"数据对齐错误: {str(e)}")
            return None

    def cut_aligned_data(self, data):
        """
        根据速度信息切片对齐数据，识别运动阶段

        Parameters:
        -----------
        data : pd.DataFrame
            对齐后的数据，包含速度(vel_l)列

        Returns:
        -----------
        pd.DataFrame
            切片后的数据，包含 movement_type, movement_phase, segment_id 等列
        """
        if data is None or 'vel_l' not in data.columns:
            self.print_debug_info("没有速度数据可用于切片")
            return []

        velocity = data['vel_l'].values
        position = data['pos_l'].values
        time = data['time'].values

        # 找到速度过零点
        zero_crossings = []
        for i in range(len(velocity) - 1):
            if velocity[i] == 0 or velocity[i] * velocity[i + 1] <= 0:
                zero_crossings.append(i)

        if not zero_crossings:
            self.print_debug_info("未找到速度过零点")
            return [data]

        # 根据速度变化方向识别运动阶段
        movement_segments = []
        for i in range(len(zero_crossings) - 1):
            start_idx = zero_crossings[i]
            end_idx = zero_crossings[i + 1]

            if self._is_valid_movement_segment(start_idx, end_idx, velocity, time):
                segment_data = data.iloc[start_idx:end_idx + 1].copy()

                seg_position = position[start_idx:end_idx + 1]
                seg_velocity = velocity[start_idx:end_idx + 1]
                avg_velocity = np.mean(seg_velocity)

                if avg_velocity > 0:
                    movement_type = "upward"
                    movement_phase = "concentric"
                else:
                    movement_type = "downward"
                    movement_phase = "eccentric"

                segment_data['movement_type'] = movement_type
                segment_data['movement_phase'] = movement_phase
                segment_data['segment_id'] = len(movement_segments)

                seg_duration = time[end_idx] - time[start_idx]
                segment_data['movement_duration'] = seg_duration
                segment_data['max_velocity'] = np.max(np.abs(seg_velocity))
                segment_data['avg_velocity'] = avg_velocity
                segment_data['start_time'] = time[start_idx]
                segment_data['end_time'] = time[end_idx]

                movement_segments.append(segment_data)

        # 过滤无效片段
        filtered_segments = self._filter_movement_segments(movement_segments, velocity, time)

        # 识别完整运动周期
        complete_movements = self._identify_complete_movements(filtered_segments)

        if not complete_movements:
            self.print_debug_info(f"找到 {len(filtered_segments)} 个运动片段")
            result_df = pd.concat(filtered_segments, ignore_index=True)
        else:
            result_df = pd.concat(complete_movements, ignore_index=True)

        result_df = self._postprocess_position_standardization(result_df, print_label=False)

        return result_df

    def _is_valid_movement_segment(self, start_idx, end_idx, velocity, time):
        """检查是否是一个有效的运动片段"""
        if end_idx <= start_idx or end_idx - start_idx < 10:
            return False

        segment_duration = time[end_idx] - time[start_idx]
        if segment_duration < 0.2:
            return False

        segment_velocity = velocity[start_idx:end_idx + 1]
        max_speed = np.max(np.abs(segment_velocity))
        if max_speed < 0.03:
            return False

        positive_count = np.sum(segment_velocity > 0.01)
        negative_count = np.sum(segment_velocity < -0.01)
        total_significant = positive_count + negative_count

        if total_significant > 0:
            dominant_ratio = max(positive_count, negative_count) / total_significant
            if dominant_ratio < 0.7:
                return False

        return True

    def _filter_movement_segments(self, segments, velocity, time):
        """过滤无效的运动片段"""
        filtered = []
        position_ranges = np.asarray([np.max(s['pos_l']) - np.min(s['pos_l']) for s in segments])
        max_range = np.max(position_ranges)

        for i, segment in enumerate(segments):
            start_idx = segment.index[0]
            end_idx = segment.index[-1]

            seg_velocity = segment['vel_l'].values
            avg_speed = np.mean(np.abs(seg_velocity))
            max_speed = np.max(np.abs(seg_velocity))
            duration = time[end_idx] - time[start_idx]
            position_range = position_ranges[i]

            if (duration >= 0.3 and
                    max_speed >= 0.05 and
                    avg_speed >= 0.02 and
                    abs(position_range - max_range) < 0.1):
                filtered.append(segment)
            else:
                if self.debug_label:
                    self.print_debug_info(
                        f"过滤掉片段 {i}: 持续时间={duration:.2f}s, "
                        f"最大速度={max_speed:.3f}, 平均速度={avg_speed:.3f}, "
                        f"运动范围={position_range:.3f}")

        return filtered

    def _merge_movement_segments(self, segments):
        """合并相邻的同向运动阶段"""
        if not segments:
            return []

        merged = []
        current_segment = segments[0].copy()

        for i in range(1, len(segments)):
            current_type = current_segment['movement_type'].iloc[0]
            next_segment = segments[i]
            next_type = next_segment['movement_type'].iloc[0]

            current_end = current_segment['time'].iloc[-1]
            next_start = next_segment['time'].iloc[0]
            time_gap = next_start - current_end

            if (current_type == next_type and
                    time_gap < 0.5 and
                    len(current_segment) + len(next_segment) < 1000):
                current_segment = pd.concat([current_segment, next_segment], ignore_index=True)
                current_segment['segment_id'] = current_segment['segment_id'].iloc[0]
            else:
                merged.append(current_segment)
                current_segment = next_segment.copy()

        merged.append(current_segment)
        return merged

    def _identify_complete_movements(self, segments):
        """
        识别完整的运动周期（包含向上和向下阶段）
        """
        if len(segments) < 2:
            return []

        complete_movements = []
        current_movement = []

        for i in range(len(segments)):
            segment = segments[i]
            movement_type = segment['movement_type'].iloc[0]

            if not current_movement:
                current_movement.append(segment)
            else:
                prev_type = current_movement[-1]['movement_type'].iloc[0]

                if prev_type != movement_type:
                    current_movement.append(segment)

                    if len(current_movement) >= 2:
                        types = [seg['movement_type'].iloc[0] for seg in current_movement]
                        is_complete = True
                        for j in range(len(types) - 1):
                            if types[j] == types[j + 1]:
                                is_complete = False
                                break

                        if is_complete and len(current_movement) % 2 == 0:
                            complete_cycle = pd.concat(current_movement, ignore_index=True)
                            complete_cycle['cycle_id'] = len(complete_movements)
                            complete_cycle['phase'] = ''

                            phase_count = 0
                            for seg in current_movement:
                                seg_len = len(seg)
                                if phase_count % 2 == 0:
                                    phase_name = "concentric" if seg['movement_type'].iloc[0] == "upward" else "eccentric"
                                else:
                                    phase_name = "eccentric" if seg['movement_type'].iloc[0] == "upward" else "concentric"
                                complete_cycle.iloc[-seg_len:, complete_cycle.columns.get_loc('phase')] = phase_name
                                phase_count += 1

                            complete_movements.append(complete_cycle)
                            current_movement = []
                else:
                    if len(current_movement) >= 2:
                        incomplete_cycle = pd.concat(current_movement, ignore_index=True)
                        incomplete_cycle['cycle_id'] = len(complete_movements)
                        incomplete_cycle['is_complete'] = False
                        complete_movements.append(incomplete_cycle)
                    current_movement = [segment]

        if current_movement and len(current_movement) >= 2:
            last_cycle = pd.concat(current_movement, ignore_index=True)
            last_cycle['cycle_id'] = len(complete_movements)
            types = [seg['movement_type'].iloc[0] for seg in current_movement]
            is_complete = all(types[j] != types[j + 1] for j in range(len(types) - 1))
            last_cycle['is_complete'] = is_complete
            complete_movements.append(last_cycle)

        return complete_movements

    def _postprocess_position_standardization(self, df, print_label=True):
        """
        后处理：用中间周期的最大值截断第一个和最后一个周期的边界
        """
        if 'cycle_id' not in df.columns or 'pos_l' not in df.columns:
            self.print_debug_info("缺少必要的列，跳过位置截断", print_label=print_label)
            return df

        unique_cycles = np.sort(df['cycle_id'].unique())
        cycle_count = len(unique_cycles)

        if cycle_count <= 2:
            self.print_debug_info(f"只有 {cycle_count} 个周期，跳过位置截断", print_label=print_label)
            return df

        self.print_debug_info(f"开始位置截断，共有 {cycle_count} 个周期", print_label=print_label)

        cycle_stats = {}
        for cycle_id in unique_cycles:
            cycle_data = df[df['cycle_id'] == cycle_id]
            positions = cycle_data['pos_l'].values
            cycle_stats[cycle_id] = {
                'min_pos': np.min(positions),
                'max_pos': np.max(positions),
                'range': np.max(positions) - np.min(positions),
                'data': cycle_data,
                'original_length': len(cycle_data)
            }
            self.print_debug_info(
                f"周期 {cycle_id}: 位置范围=[{cycle_stats[cycle_id]['min_pos']:.3f}, "
                f"{cycle_stats[cycle_id]['max_pos']:.3f}]", print_label=print_label)

        middle_cycle_ids = unique_cycles[1:-1]
        middle_max_positions = [cycle_stats[cid]['max_pos'] for cid in middle_cycle_ids]
        middle_min_positions = [cycle_stats[cid]['min_pos'] for cid in middle_cycle_ids]

        if not middle_max_positions:
            self.print_debug_info("没有中间周期，跳过截断", print_label=print_label)
            return df

        reference_max_pos = np.max(middle_max_positions)
        reference_min_pos = np.min(middle_min_positions)

        self.print_debug_info(f"参考范围: [{reference_min_pos:.3f}, {reference_max_pos:.3f}]",
                              print_label=print_label)

        # 处理第一个周期
        standardized_dfs = []
        first_cycle_id = unique_cycles[0]
        first_stats = cycle_stats[first_cycle_id]

        if first_stats['min_pos'] < reference_min_pos:
            self.print_debug_info(
                f"第一个周期开始位置 {first_stats['min_pos']:.3f} < 参考值 {reference_min_pos:.3f}，从开始截断",
                print_label=print_label)
            cropped_first = self._crop_cycle_start_to_minimum(first_stats['data'], reference_min_pos)
            if cropped_first is not None and len(cropped_first) >= 10:
                standardized_dfs.append(cropped_first)
            else:
                standardized_dfs.append(first_stats['data'])
        else:
            standardized_dfs.append(first_stats['data'])

        # 中间周期不变
        for cycle_id in middle_cycle_ids:
            standardized_dfs.append(cycle_stats[cycle_id]['data'])

        # 处理最后一个周期
        last_cycle_id = unique_cycles[-1]
        last_stats = cycle_stats[last_cycle_id]

        if last_stats['max_pos'] > reference_max_pos:
            self.print_debug_info(
                f"最后一个周期结束位置 {last_stats['max_pos']:.3f} > 参考值 {reference_max_pos:.3f}，从结束截断",
                print_label=print_label)
            cropped_last = self._crop_cycle_end_to_maximum(last_stats['data'], reference_max_pos)
            if cropped_last is not None and len(cropped_last) >= 10:
                standardized_dfs.append(cropped_last)
            else:
                standardized_dfs.append(last_stats['data'])
        else:
            standardized_dfs.append(last_stats['data'])

        result_df = pd.concat(standardized_dfs, ignore_index=True)
        self._validate_truncation_result(result_df, reference_min_pos, reference_max_pos, print_label=print_label)

        return result_df

    def _crop_cycle_start_to_minimum(self, cycle_df, min_threshold):
        """从开始截断周期，删除位置过低的部分"""
        positions = cycle_df['pos_l'].values

        for i in range(len(positions)):
            if positions[i] >= min_threshold * 0.98:
                check_window = min(5, i)
                window_positions = positions[max(0, i - check_window):i + 1]
                if np.min(window_positions) >= min_threshold * 0.98:
                    return cycle_df.iloc[i:]

        mean_pos = np.mean(positions)
        for i in range(len(positions)):
            if positions[i] >= mean_pos * 0.9:
                return cycle_df.iloc[i:]

        crop_idx = int(len(cycle_df) * 0.2)
        return cycle_df.iloc[crop_idx:]

    def _crop_cycle_end_to_maximum(self, cycle_df, max_threshold):
        """从结束截断周期，删除位置过高的部分"""
        positions = cycle_df['pos_l'].values

        for i in range(len(positions) - 1, -1, -1):
            if positions[i] <= max_threshold * 1.02:
                check_window = min(5, len(positions) - i - 1)
                window_positions = positions[i:min(len(positions), i + check_window + 1)]
                if np.max(window_positions) <= max_threshold * 1.02:
                    return cycle_df.iloc[:i + 1]

        for i in range(len(positions) - 5):
            if positions[i] > positions[i + 1] and positions[i] > max_threshold:
                return cycle_df.iloc[:i + 1]

        crop_idx = int(len(cycle_df) * 0.8)
        return cycle_df.iloc[:crop_idx]

    def _validate_truncation_result(self, df, ref_min, ref_max, print_label=True):
        """验证截断结果"""
        if 'cycle_id' not in df.columns:
            return

        unique_cycles = np.sort(df['cycle_id'].unique())
        self.print_debug_info("=== 截断后验证 ===", print_label=print_label)

        for cycle_id in unique_cycles:
            cycle_data = df[df['cycle_id'] == cycle_id]
            positions = cycle_data['pos_l'].values
            cycle_min = np.min(positions)
            cycle_max = np.max(positions)

            min_status = "✓" if cycle_min >= ref_min * 0.95 else "✗"
            max_status = "✓" if cycle_max <= ref_max * 1.05 else "✗"

            self.print_debug_info(
                f"周期 {cycle_id}: "
                f"min={cycle_min:.3f}{min_status}(参考{ref_min:.3f}), "
                f"max={cycle_max:.3f}{max_status}(参考{ref_max:.3f})",
                print_label=print_label)