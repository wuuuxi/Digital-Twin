import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


class CurvePlotter:
    """
    曲线可视化器
    负责绘制位置-肌肉激活平均曲线、负载比较图，以及保存曲线数据
    """

    def __init__(self, subject=None, debug_label=False):
        self.subject = subject
        self.results = {}
        self.debug_label = debug_label

    def set_results(self, results):
        """设置处理结果数据"""
        self.results = results

    def _log(self, msg):
        if self.debug_label:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _get_default_muscles(self, subject=None):
        """根据运动类型返回默认目标肌肉列表"""
        s = subject or self.subject
        if s and s.target_motion == 'benchpress':
            return ['emg_TriLat', 'emg_PMSte', 'emg_DelAnt', 'emg_Bic']
        return ['emg_FibLon', 'emg_VL', 'emg_RF']

    def plot_average_curves(self, results_dict, save_path=None, figsize=(15, 10),
                            emg_channels=None):
        """
        可视化平均曲线结果

        Parameters:
        -----------
        results_dict : dict
            CurveAnalyzer.process_for_curves 的输出
        save_path : str, optional
            保存图像的路径
        figsize : tuple
            图像大小
        emg_channels : list, optional
            指定要绘制的EMG通道，默认自动检测
        """
        if not results_dict:
            print("没有可绘制的数据")
            return

        loads = list(results_dict.keys())
        phases = set()
        for load_data in results_dict.values():
            phases.update(load_data.keys())
        phases = list(phases)

        # 自动检测或使用指定的EMG通道
        if emg_channels is None:
            emg_channels_set = set()
            for load_data in results_dict.values():
                for phase_data in load_data.values():
                    if 'emg_curves' in phase_data:
                        emg_channels_set.update(phase_data['emg_curves'].keys())
            emg_channels = sorted(list(emg_channels_set))

        print(f"绘图: {len(loads)}个负载, {len(phases)}个阶段, {len(emg_channels)}个EMG通道")

        n_rows = len(emg_channels)
        n_cols = len(phases)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        colors = plt.cm.Set1(np.linspace(0, 1, len(loads)))
        load_colors = dict(zip(loads, colors))

        for row_idx, emg_channel in enumerate(emg_channels):
            for col_idx, phase in enumerate(phases):
                ax = axes[row_idx, col_idx]

                for load in loads:
                    if load in results_dict and phase in results_dict[load]:
                        load_data = results_dict[load][phase]

                        if 'emg_curves' in load_data and emg_channel in load_data['emg_curves']:
                            curve_data = load_data['emg_curves'][emg_channel]

                            x = np.array(curve_data['centers'])
                            y_mean = np.array(curve_data['means'])
                            y_sem = np.array(curve_data['sems'])

                            color = load_colors[load]
                            ax.plot(x, y_mean, color=color, linewidth=2,
                                    label=f"{load} (n={load_data.get('n_segments', '?')})")
                            ax.fill_between(x, y_mean - y_sem, y_mean + y_sem,
                                            color=color, alpha=0.2)

                            if curve_data.get('fitted_curve') is not None:
                                x_fit = np.array(curve_data.get('smooth_positions', curve_data['centers']))
                                y_fit = np.array(curve_data['fitted_curve'])
                                ax.plot(x_fit, y_fit, color=color, linestyle='--',
                                        linewidth=1.5, alpha=0.8)

                if row_idx == 0:
                    ax.set_title(f"{phase.capitalize()} Phase", fontsize=12, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f"{emg_channel}\nActivation (mV)", fontsize=10)
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Position (%)", fontsize=10)

                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

        plt.suptitle('Position-Muscle Activation Average Curves by Load and Movement Phase',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")

        self._plot_load_comparison(results_dict, emg_channels, phases)

    def _plot_load_comparison(self, results_dict, emg_channels, phases):
        """
        绘制负载比较图

        Parameters:
        -----------
        results_dict : dict
            曲线分析结果
        emg_channels : list
            EMG通道列表
        phases : list
            运动阶段列表
        """
        for phase in phases:
            n_rows = len(emg_channels)
            fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), squeeze=False)

            for row_idx, emg_channel in enumerate(emg_channels):
                ax = axes[row_idx, 0]

                for load in results_dict.keys():
                    if (phase in results_dict[load] and
                            emg_channel in results_dict[load][phase].get('emg_curves', {})):
                        curve_data = results_dict[load][phase]['emg_curves'][emg_channel]

                        x = np.array(curve_data['centers'])
                        y_mean = np.array(curve_data['means'])
                        y_sem = np.array(curve_data['sems'])

                        n_seg = results_dict[load][phase].get('n_segments', '?')
                        ax.plot(x, y_mean, linewidth=2,
                                label=f"{load} (n={n_seg})")
                        ax.fill_between(x, y_mean - y_sem, y_mean + y_sem, alpha=0.2)

                ax.set_title(f"{emg_channel} - {phase.capitalize()} Phase", fontsize=12)
                ax.set_xlabel("Position (%)")
                ax.set_ylabel("Activation (mV)")
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.suptitle(f'Load Comparison: {phase.capitalize()} Phase',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

    def save_curve_data_to_csv(self, results_dict, output_dir='./curve_data'):
        """
        将平均曲线数据保存到CSV文件

        Parameters:
        -----------
        results_dict : dict
            CurveAnalyzer.process_for_curves 的输出
        output_dir : str
            输出目录路径
        """
        os.makedirs(output_dir, exist_ok=True)

        for load, load_data in results_dict.items():
            for phase, phase_data in load_data.items():
                if 'emg_curves' in phase_data:
                    for emg_channel, curve_data in phase_data['emg_curves'].items():
                        fitted = curve_data.get('fitted_curve')
                        if fitted and len(fitted) != len(curve_data['centers']):
                            fitted = None  # smooth 曲线长度不同，不放入按 bin 的表
                        df = pd.DataFrame({
                            'centers': curve_data['centers'],
                            'means': curve_data['means'],
                            'stds': curve_data['stds'],
                            'sems': curve_data['sems'],
                            'fitted_curve': (fitted if fitted
                                             else [np.nan] * len(curve_data['centers']))
                        })

                        metadata = {
                            'load': load,
                            'phase': phase,
                            'emg_channel': emg_channel,
                            'n_segments': phase_data.get('n_segments', curve_data.get('n_segments', 'N/A')),
                            'r_squared': curve_data.get('r_squared', 'N/A'),
                            'binning_method': curve_data.get('binning_method', 'N/A')
                        }

                        filename = f"curve_{load}_{phase}_{emg_channel}.csv"
                        filepath = os.path.join(output_dir, filename)
                        df.to_csv(filepath, index=False)

                        metadata_file = os.path.join(output_dir, f"metadata_{filename}")
                        pd.DataFrame([metadata]).to_csv(metadata_file, index=False)

        print(f"曲线数据已保存到: {output_dir}")

    # ==================== 对齐可视化 ====================

    def visualize_alignment(self, results=None, subject=None, load_weights=None,
                            save_fig=True, target_cols=None):
        """可视化数据对齐：力、位置、速度、EMG信号"""
        results = results or self.results
        subject = subject or self.subject
        if not results:
            print("没有处理结果可供可视化"); return
        if load_weights is None:
            load_weights = list(results.keys())
        n_loads = len(load_weights)
        if n_loads == 0: return

        fig, axes = plt.subplots(n_loads, 1, figsize=(10, 3 * n_loads))
        if n_loads == 1: axes = [axes]

        for i, lw in enumerate(load_weights):
            if lw not in results: continue
            result = results[lw]
            ax = axes[i]

            if result.get('robot_data') is not None:
                rd = result['robot_data']
                if 'force_l' in rd.columns:
                    ax.plot(rd['time'], rd['force_l'] / (abs(rd['force_l']).max() + 1e-10),
                            label='Robot Force', alpha=0.7)
                if 'pos_l' in rd.columns:
                    ax.plot(rd['time'], rd['pos_l'] - rd['pos_l'].iloc[-1],
                            label='Robot Position', alpha=0.7)
                if 'vel_l' in rd.columns:
                    ax.plot(rd['time'], rd['vel_l'], label='Robot Velocity', alpha=0.7)
                if 'acc_l' in rd.columns:
                    ax.plot(rd['time'], rd['acc_l'] / (abs(rd['acc_l']).max() + 1e-10) / 2,
                            label='Robot Acceleration', alpha=0.7)

            if result.get('aligned_data') is not None:
                ad = result['aligned_data']
                emg_cols = [c for c in ad.columns if c.startswith('emg_')]
                if emg_cols:
                    tc = target_cols
                    if tc is None:
                        tc = (['emg_TriLat'] if subject and subject.target_motion == 'benchpress'
                              else ['emg_FibLon', 'emg_VL', 'emg_RF'])
                    for mc in tc:
                        if mc in ad.columns:
                            ax.plot(ad['time'], ad[mc], '--',
                                    label=f'EMG {mc.replace("emg_", "")}', alpha=0.7)

            ax.set_title(f'Load: {lw}kg')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized Signal')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_fig and subject:
            fp = os.path.join(subject.result_folder, 'data_alignment.png')
            plt.savefig(fp, dpi=300)
            self._log(f"对齐可视化已保存到: {fp}")

    # ==================== 运动切片可视化 ====================

    def visualize_movement_segments(self, results=None, subject=None, load_weights=None,
                                     save_fig=True, target_muscles=None, legend_label=False):
        """可视化运动切片结果：速度、位置、力、EMG分行显示"""
        results = results or self.results
        subject = subject or self.subject
        if not results:
            print("没有处理结果可供可视化"); return
        if load_weights is None:
            load_weights = list(results.keys())
        load_num = len(load_weights)
        if load_num == 0: return

        if target_muscles is None:
            target_muscles = self._get_default_muscles(subject)[:1]
        muscle_col = target_muscles[0]

        fig, axes = plt.subplots(4, load_num, figsize=(17, 8))
        if load_num == 1: axes = axes.reshape(4, 1)

        # 计算全局Y轴范围
        gy = {k: [float('inf'), float('-inf')] for k in ['vel', 'pos', 'force', 'emg']}
        col_map = {'vel': 'vel_l', 'pos': 'pos_l', 'force': 'force_l', 'emg': muscle_col}
        for lw in load_weights:
            if lw not in results: continue
            cd = results[lw].get('cutted_data')
            if cd is None: continue
            for k, c in col_map.items():
                if c in cd.columns:
                    gy[k][0] = min(gy[k][0], np.min(cd[c]))
                    gy[k][1] = max(gy[k][1], np.max(cd[c]))
        for k in gy:
            if gy[k][0] != float('inf'):
                r = (gy[k][1] - gy[k][0]) * 0.1
                gy[k][0] -= r; gy[k][1] += r

        for li, lw in enumerate(load_weights):
            if lw not in results: continue
            res = results[lw]
            ad, cd = res.get('aligned_data'), res.get('cutted_data')
            if ad is None or cd is None: continue
            seg_num = int(max(cd['cycle_id']) + 1)

            axs = [axes[j, li] for j in range(4)]
            cols = ['vel_l', 'pos_l', 'force_l', muscle_col]
            labels = ['Velocity', 'Position', 'Force', 'EMG Activation']
            gy_keys = ['vel', 'pos', 'force', 'emg']

            for ax, col in zip(axs, cols):
                if col in ad.columns:
                    ax.plot(ad['time'], ad[col], 'k-', alpha=0.3)

            colors = plt.cm.tab10.colors
            for i in range(seg_num):
                clr = colors[i % len(colors)]
                seg = cd[cd['cycle_id'] == i]
                su = seg[seg['movement_type'] == 'upward']
                sd_seg = seg[seg['movement_type'] == 'downward']
                for ax, col in zip(axs, cols):
                    if col in su.columns:
                        ax.plot(su['time'], su[col], '-', color=clr, linewidth=2)
                        ax.plot(sd_seg['time'], sd_seg[col], '--', color=clr, linewidth=2, alpha=0.6)

            for ax, gk, lbl in zip(axs, gy_keys, labels):
                ax.set_ylim(gy[gk][0], gy[gk][1])
                ax.set_ylabel(lbl)
                ax.set_xlabel('Time (s)')
                st, et = np.min(cd['time']), np.max(cd['time'])
                ax.set_xlim(st - 1, et + 1)
                ax.grid(True, alpha=0.3)
                if legend_label: ax.legend(loc='upper right')

            axs[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axs[0].set_title(f'Load {lw}kg')

        plt.tight_layout()
        if save_fig and subject:
            fp = os.path.join(subject.result_folder, 'movement_segmentation.png')
            plt.savefig(fp, dpi=300)
            self._log(f"运动切片可视化已保存到: {fp}")

    # ==================== 3D散点图 ====================

    def visualize_test_3d_scatter(self, results=None, subject=None, load_weights=None,
                                   save_fig=True, target_muscles=None):
        """3D散点图（位置-速度-负载）和2D散点图（力-激活等）"""
        results = results or self.results
        subject = subject or self.subject
        if not results:
            print("没有处理结果可供可视化"); return
        if load_weights is None:
            load_weights = list(results.keys())
        if not load_weights: return

        if target_muscles is None:
            target_muscles = (['emg_PMSte'] if subject and subject.target_motion == 'benchpress'
                              else self._get_default_muscles(subject)[:1])
        muscle_col = target_muscles[0]
        muscle_name = muscle_col.replace('emg_', '')

        fig1, ax1 = plt.subplots(figsize=(7, 8), subplot_kw={'projection': '3d'})
        fig2, ax2 = plt.subplots(figsize=(7, 8))
        fig3 = plt.figure(figsize=(17, 7))
        ax31 = fig3.add_subplot(131)
        ax32 = fig3.add_subplot(132)
        ax33 = fig3.add_subplot(133)
        fig5 = plt.figure(figsize=(17, 6))

        # 全局范围
        gv = [float('inf'), float('-inf')]
        ge = [float('inf'), float('-inf')]
        for lw in load_weights:
            if lw not in results: continue
            ad = results[lw].get('aligned_data')
            if ad is None or muscle_col not in ad.columns: continue
            v, e = np.abs(ad['vel_l'].values), np.abs(ad[muscle_col].values)
            gv = [min(gv[0], np.min(v)), max(gv[1], np.max(v))]
            ge = [min(ge[0], np.min(e)), max(ge[1], np.max(e))]
        for g in [gv, ge]:
            if g[0] != float('inf'):
                r = (g[1] - g[0]) * 0.1
                g[0] = max(0, g[0] - r); g[1] += r
            else:
                g[0], g[1] = 0, 1

        cmap = plt.cm.viridis
        all_scatter = []

        for li, lw in enumerate(load_weights):
            if lw not in results: continue
            cd = results[lw].get('cutted_data')
            if cd is None or muscle_col not in cd.columns: continue

            pos = cd['pos_l'].values
            vel = cd['vel_l'].values
            force = cd['force_l'].values
            emg = cd[muscle_col].values
            load = cd['load'].values
            power = vel * force
            acc = cd['acc_l'].values
            all_scatter.extend(emg)

            ax1.scatter3D(pos, vel, load, c=emg, cmap=cmap, s=10, alpha=0.6, label=f'{lw}kg')
            ax2.scatter(force, emg, s=10, alpha=0.6, label=f'{lw}kg')
            ax31.scatter(pos, vel, s=10, alpha=0.6, label=f'{lw}kg')
            ax32.scatter(pos, force, s=10, alpha=0.6, label=f'{lw}kg')
            ax33.scatter(pos, power, s=10, alpha=0.6, label=f'{lw}kg')

            pmask, nmask = vel > 0, vel <= 0
            ax51 = fig5.add_subplot(2, len(load_weights), li + 1)
            ax52 = fig5.add_subplot(2, len(load_weights), li + 1 + len(load_weights))
            ax51.scatter(pos[pmask], np.abs(vel[pmask]), alpha=0.6, s=10, label='Concentric')
            ax51.scatter(pos[nmask], np.abs(vel[nmask]), alpha=0.6, s=10, label='Eccentric')
            ax51.set_ylim(gv[0], gv[1]); ax51.set_title(f'{lw}kg')
            ax52.scatter(pos[pmask], np.abs(emg[pmask]), alpha=0.6, s=10, label='Concentric')
            ax52.scatter(pos[nmask], np.abs(emg[nmask]), alpha=0.6, s=10, label='Eccentric')
            ax52.set_ylim(ge[0], ge[1])
            ax51.set_xlabel('Position'); ax52.set_xlabel('Position')
            if li == 0:
                ax51.set_ylabel('Velocity'); ax52.set_ylabel('EMG Activation')
            elif li == len(load_weights) - 1:
                ax51.legend(); ax52.legend()

            mv = np.mean(vel[pmask]) if np.any(pmask) else 0
            mpv_mask = (vel > 0) & (acc > 0)
            mpv = np.mean(vel[mpv_mask]) if np.any(mpv_mask) else 0
            pl1 = 11.4196 * mv**2 - 81.904 * mv + 114.03
            pl2 = 11.2988 * mpv**2 - 78.05 * mpv + 113.04
            print(f'{lw}kg: MPV={mpv:.3f}, MV={mv:.3f}, PL1={pl1:.3f}, PL2={pl2:.3f}')

        if all_scatter:
            norm = plt.Normalize(np.min(all_scatter), np.max(all_scatter))
            for coll in ax1.collections: coll.set_norm(norm)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
            fig1.colorbar(sm, ax=ax1, pad=0.1).set_label(
                f'{muscle_name} Activation', rotation=270, labelpad=15)

        ax1.set_xlabel('Position', labelpad=10)
        ax1.set_ylabel('Velocity', labelpad=10)
        ax1.set_zlabel('Load (kg)', labelpad=10)
        ax1.set_title('3D Scatter: Position-Velocity-Load\n(Colored by Muscle Activation)')
        ax2.set_xlabel('Force'); ax2.set_ylabel(f'{muscle_name} Activation')
        ax31.set_xlabel('Position'); ax31.set_ylabel('Velocity')
        ax32.set_xlabel('Position'); ax32.set_ylabel('Force')
        ax33.set_xlabel('Position'); ax33.set_ylabel('Power')
        for ax in [ax2, ax31, ax32, ax33]:
            ax.grid(True, alpha=0.3); ax.legend(loc='best')
        for fig in [fig2, fig3, fig5]: fig.tight_layout()

        if save_fig and subject:
            fp = os.path.join(subject.result_folder, 'scatter_plots.png')
            plt.savefig(fp, dpi=300)
            self._log(f"散点图已保存到: {fp}")

    # ==================== 肌肉分析 ====================

    def visualize_muscle_analysis(self, results=None, subject=None, load_weights=None,
                                   save_fig=True, target_muscles=None):
        """肌肉分析图：3D散点、力-激活散点、位置-速度/EMG散点"""
        results = results or self.results
        subject = subject or self.subject
        if not results:
            print("没有处理结果可供可视化"); return
        if load_weights is None:
            load_weights = list(results.keys())
        if not load_weights: return

        if target_muscles is None:
            target_muscles = self._get_default_muscles(subject)
        muscle_num = len(target_muscles)

        fig1, axes1 = plt.subplots(1, muscle_num, figsize=(5 * muscle_num, 8),
                                    subplot_kw={'projection': '3d'})
        fig2, axes2 = plt.subplots(1, muscle_num, figsize=(5 * muscle_num, 5))
        fig3, axes3 = plt.subplots(muscle_num + 1, len(load_weights),
                                    figsize=(3 * len(load_weights), 3 * muscle_num + 3))
        if muscle_num == 1: axes1 = [axes1]; axes2 = [axes2]
        if len(load_weights) == 1: axes3 = axes3.reshape(-1, 1)

        # 全局范围
        gv = [float('inf'), float('-inf')]
        gem_min = np.full(muscle_num, float('inf'))
        gem_max = np.full(muscle_num, float('-inf'))
        for lw in load_weights:
            if lw not in results: continue
            ad = results[lw].get('aligned_data')
            if ad is None: continue
            if 'vel_l' in ad.columns:
                av = np.abs(ad['vel_l'].values)
                gv = [min(gv[0], np.min(av)), max(gv[1], np.max(av))]
            for idx, mc in enumerate(target_muscles):
                if mc in ad.columns:
                    ae = np.abs(ad[mc].values)
                    gem_min[idx] = min(gem_min[idx], np.min(ae))
                    gem_max[idx] = max(gem_max[idx], np.max(ae))

        rv = (gv[1] - gv[0]) * 0.1 if gv[0] != float('inf') else 0
        gv = [gv[0] - rv, gv[1] + rv] if gv[0] != float('inf') else [0, 1]
        for i in range(muscle_num):
            re = (gem_max[i] - gem_min[i]) * 0.1 if gem_min[i] != float('inf') else 0
            gem_min[i] = gem_min[i] - re if gem_min[i] != float('inf') else 0
            gem_max[i] = gem_max[i] + re if gem_max[i] != float('-inf') else 1

        cmap = plt.cm.viridis
        all_muscle_data = [[] for _ in target_muscles]

        for li, lw in enumerate(load_weights):
            if lw not in results: continue
            res = results[lw]
            cd = res.get('cutted_data')
            avg = res.get('average_data', {}).get(lw, {})
            if cd is None or not avg: continue

            pos = cd['pos_l'].values
            vel = cd['vel_l'].values
            force = cd['force_l'].values
            load = cd['load'].values
            pmask, nmask = vel > 0, vel <= 0

            avg_ecc = avg.get('eccentric', {})
            avg_con = avg.get('concentric', {})
            vel_ecc = avg_ecc.get('vel_curves', {})
            vel_con = avg_con.get('vel_curves', {})

            # 速度子图
            ax3v = axes3[0, li]
            if vel_con:
                xc = np.array(vel_con['centers'])
                yc = np.array(vel_con['means'])
                ysc = np.array(vel_con['stds'])
                ax3v.scatter(pos[pmask], np.abs(vel[pmask]), alpha=0.6, s=10, color='#1f77b4', label='Con')
                ax3v.plot(xc, np.abs(yc), linewidth=2, color='#1f77b4')
                ax3v.fill_between(xc, np.abs(yc - ysc), np.abs(yc + ysc), alpha=0.2, color='#1f77b4')
            if vel_ecc:
                xe = np.array(vel_ecc['centers'])
                ye = np.array(vel_ecc['means'])
                yse = np.array(vel_ecc['stds'])
                ax3v.scatter(pos[nmask], np.abs(vel[nmask]), alpha=0.6, s=10, color='#ff7f0e', label='Ecc')
                ax3v.plot(xe, np.abs(ye), linewidth=2, color='#ff7f0e')
                ax3v.fill_between(xe, np.abs(ye - yse), np.abs(ye + yse), alpha=0.2, color='#ff7f0e')
            ax3v.set_ylim(gv[0], gv[1]); ax3v.set_xlabel('Position'); ax3v.set_title(f'{lw}kg')
            if li == 0: ax3v.set_ylabel('Velocity')
            elif li == len(load_weights) - 1: ax3v.legend()

            all_emg = [cd[mc].values if mc in cd.columns else np.zeros(len(cd))
                       for mc in target_muscles]

            for mi, (ax1, ax2, mc, emg) in enumerate(zip(axes1, axes2, target_muscles, all_emg)):
                all_muscle_data[mi].extend(emg)
                mn = mc.replace('emg_', '')
                emg_ecc_c = avg_ecc.get('emg_curves', {}).get(mc, {})
                emg_con_c = avg_con.get('emg_curves', {}).get(mc, {})

                ax1.scatter3D(pos, vel, load, c=emg, cmap=cmap, s=10, alpha=0.6, label=f'{lw}kg')
                ax1.set_xlabel('Position', labelpad=10)
                ax1.set_ylabel('Velocity', labelpad=10)
                ax1.set_zlabel('Load', labelpad=10)
                ax1.set_title(f'Muscle: {mn}', fontsize=12)

                ax2.scatter(force, emg, s=10, alpha=0.6, label=f'{lw}kg')
                ax2.set_xlabel('Force'); ax2.set_ylabel(f'{mn} Activation')
                ax2.grid(True, alpha=0.3); ax2.legend(loc='best')

                ax3e = axes3[mi + 1, li]
                ax3e.scatter(pos[pmask], np.abs(emg[pmask]), alpha=0.6, s=10, color='#1f77b4')
                ax3e.scatter(pos[nmask], np.abs(emg[nmask]), alpha=0.6, s=10, color='#ff7f0e')
                if emg_con_c:
                    xc2 = np.array(emg_con_c['centers'])
                    yc2 = np.array(emg_con_c['means'])
                    ysc2 = np.array(emg_con_c['stds'])
                    ax3e.plot(xc2, yc2, linewidth=2, color='#1f77b4')
                    ax3e.fill_between(xc2, yc2 - ysc2, yc2 + ysc2, alpha=0.2, color='#1f77b4')
                if emg_ecc_c:
                    xe2 = np.array(emg_ecc_c['centers'])
                    ye2 = np.array(emg_ecc_c['means'])
                    yse2 = np.array(emg_ecc_c['stds'])
                    ax3e.plot(xe2, ye2, linewidth=2, color='#ff7f0e')
                    ax3e.fill_between(xe2, ye2 - yse2, ye2 + yse2, alpha=0.2, color='#ff7f0e')
                ax3e.set_ylim(gem_min[mi], gem_max[mi]); ax3e.set_xlabel('Position')
                if li == 0: ax3e.set_ylabel(f'{mn} Activation')
                elif li == len(load_weights) - 1: ax3e.legend()

        for mi, (ax, mc) in enumerate(zip(axes1, target_muscles)):
            mn = mc.replace('emg_', '')
            if all_muscle_data[mi]:
                vmin, vmax = np.min(all_muscle_data[mi]), np.max(all_muscle_data[mi])
                norm = plt.Normalize(vmin, vmax)
                for coll in ax.collections: coll.set_norm(norm)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                fig1.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8).set_label(
                    f'{mn} Activation', labelpad=15)

        for fig in [fig2, fig3]: fig.tight_layout()
        if save_fig and subject:
            fp = os.path.join(subject.result_folder, 'muscle_analysis.png')
            plt.savefig(fp, dpi=300)
            self._log(f"肌肉分析图已保存到: {fp}")

    # ==================== 基于位置的误差分析 ====================

    def visualize_analyze_kinematic_emg_errors_by_position(self, results=None, subject=None,
                                                            load_weights=None, save_fig=True,
                                                            target_muscles=None):
        """基于位置分析运动学误差与EMG误差的关系"""
        results = results or self.results
        subject = subject or self.subject
        if not results:
            print("没有处理结果可供分析"); return None
        if load_weights is None:
            load_weights = list(results.keys())
        if not load_weights: return None

        if target_muscles is None:
            target_muscles = self._get_default_muscles(subject)
        muscle_num = len(target_muscles)
        muscle_names = [c.replace('emg_', '') for c in target_muscles]

        fw = 5 * muscle_num
        fig1, axes1 = plt.subplots(3, muscle_num, figsize=(fw, 12))
        fig2, axes2 = plt.subplots(3, muscle_num, figsize=(fw, 12))
        if muscle_num == 1:
            axes1 = axes1.reshape(3, 1); axes2 = axes2.reshape(3, 1)

        position_bins = 20
        all_pos = []
        for lw in load_weights:
            if lw not in results: continue
            cd = results[lw].get('cutted_data')
            if cd is not None and 'pos_l' in cd.columns:
                all_pos.extend(cd['pos_l'].values)
        if not all_pos:
            print("没有位置数据可供分析"); return None

        pmin, pmax = np.min(all_pos), np.max(all_pos)
        bin_edges = np.linspace(pmin, pmax, position_bins + 1)

        pos_binned = {}
        for mc in target_muscles:
            pos_binned[mc] = {}
            for i in range(position_bins):
                pos_binned[mc][i] = {
                    'pos_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                    'pos_errors': [], 'vel_errors': [], 'emg_errors': []}

        for lw in load_weights:
            if lw not in results: continue
            res = results[lw]
            cd = res.get('cutted_data')
            avg = res.get('average_data', {}).get(lw, {})
            if cd is None or not avg: continue
            avg_ecc = avg.get('eccentric', {})
            avg_con = avg.get('concentric', {})

            pos = cd['pos_l'].values
            vel = cd['vel_l'].values
            ecc_mask = vel < 0
            con_mask = vel >= 0

            for mi, mc in enumerate(target_muscles):
                if mc not in cd.columns: continue
                emg = cd[mc].values
                emg_ecc_c = avg_ecc.get('emg_curves', {}).get(mc)
                emg_con_c = avg_con.get('emg_curves', {}).get(mc)
                vel_ecc_c = avg_ecc.get('vel_curves')
                vel_con_c = avg_con.get('vel_curves')
                if not (emg_ecc_c and emg_con_c and vel_ecc_c and vel_con_c): continue

                pos_err = np.zeros_like(pos)
                vel_err = np.zeros_like(vel)
                emg_err = np.zeros_like(emg)

                if np.any(ecc_mask):
                    ep = pos[ecc_mask]
                    avg_ve = np.interp(np.abs(ep), np.abs(np.array(vel_ecc_c['centers'])),
                                       np.abs(np.array(vel_ecc_c['means'])))
                    avg_ee = np.interp(np.abs(ep), np.abs(np.array(emg_ecc_c['centers'])),
                                       np.abs(np.array(emg_ecc_c['means'])))
                    pos_err[ecc_mask] = np.abs(ep - ep.mean())
                    vel_err[ecc_mask] = np.abs(np.abs(vel[ecc_mask]) - avg_ve)
                    emg_err[ecc_mask] = np.abs(np.abs(emg[ecc_mask]) - avg_ee)

                if np.any(con_mask):
                    cp = pos[con_mask]
                    avg_vc = np.interp(cp, np.array(vel_con_c['centers']),
                                       np.abs(np.array(vel_con_c['means'])))
                    avg_ec = np.interp(cp, np.array(emg_con_c['centers']),
                                       np.abs(np.array(emg_con_c['means'])))
                    pos_err[con_mask] = np.abs(cp - cp.mean())
                    vel_err[con_mask] = np.abs(np.abs(vel[con_mask]) - avg_vc)
                    emg_err[con_mask] = np.abs(np.abs(emg[con_mask]) - avg_ec)

                for i in range(len(pos)):
                    bi = np.digitize(pos[i], bin_edges) - 1
                    if 0 <= bi < position_bins:
                        bd = pos_binned[mc][bi]
                        bd['pos_errors'].append(pos_err[i])
                        bd['vel_errors'].append(vel_err[i])
                        bd['emg_errors'].append(emg_err[i])

                valid = ~np.isnan(pos_err) & ~np.isnan(vel_err) & ~np.isnan(emg_err)
                pev, vev, eev = pos_err[valid], vel_err[valid], emg_err[valid]
                pe_corr = np.corrcoef(pev, eev)[0, 1] if len(pev) > 1 else 0
                ve_corr = np.corrcoef(vev, eev)[0, 1] if len(vev) > 1 else 0
                mn = muscle_names[mi]

                # 位置误差 vs EMG误差
                ax1t = axes1[0, mi]
                ax1t.scatter(pev, eev, s=15, alpha=0.6, label=f'{lw}kg (r={pe_corr:.2f})',
                             c=vel[valid], cmap='coolwarm', edgecolors='k', linewidths=0.5)
                if len(pev) > 2:
                    z = np.polyfit(pev, eev, 1); p = np.poly1d(z)
                    xr = np.linspace(np.min(pev), np.max(pev), 100)
                    ax1t.plot(xr, p(xr), 'k-', linewidth=2, alpha=0.8)
                ax1t.set_xlabel('Position Error (m)'); ax1t.set_ylabel(f'{mn} EMG Error')
                ax1t.set_title(f'{mn}: Pos Error vs EMG Error', fontweight='bold')
                ax1t.grid(True, alpha=0.3); ax1t.legend(fontsize=9)

                # 位置分组相关性
                ax1m = axes1[1, mi]
                bc, pe_c, pv_c = [], [], []
                for bi in range(position_bins):
                    bd = pos_binned[mc][bi]
                    if len(bd['pos_errors']) > 3 and len(bd['emg_errors']) > 3:
                        bc.append(bd['pos_center'])
                        pe_c.append(np.corrcoef(bd['pos_errors'], bd['emg_errors'])[0, 1])
                        if len(bd['vel_errors']) > 3:
                            pv_c.append(np.corrcoef(bd['pos_errors'], bd['vel_errors'])[0, 1])
                if bc:
                    ax1m.plot(bc, pe_c, 'o-', linewidth=2, label='Pos-EMG Corr', color='blue')
                    if pv_c:
                        ax1m.plot(bc[:len(pv_c)], pv_c, 's-', linewidth=2, label='Pos-Vel Corr', color='green')
                    ax1m.set_xlabel('Position (m)'); ax1m.set_ylabel('Correlation')
                    ax1m.set_title(f'{mn}: Position-wise Correlations')
                    ax1m.grid(True, alpha=0.3); ax1m.legend(fontsize=9); ax1m.set_ylim(-1, 1)

                ax1b = axes1[2, mi]
                ax1b.hist(pev, bins=30, alpha=0.7, label=f'Pos Err', color='blue', density=True)
                ax1b.hist(eev, bins=30, alpha=0.7, label=f'EMG Err', color='red', density=True)
                ax1b.set_xlabel('Error Value'); ax1b.set_ylabel('Density')
                ax1b.set_title(f'{mn}: Error Distributions'); ax1b.legend(fontsize=9); ax1b.grid(True, alpha=0.3)

                # 速度误差 vs EMG误差
                ax2t = axes2[0, mi]
                ax2t.scatter(vev, eev, s=15, alpha=0.6, label=f'{lw}kg (r={ve_corr:.2f})',
                             c=pos[valid], cmap='viridis', edgecolors='k', linewidths=0.5)
                if len(vev) > 2:
                    z = np.polyfit(vev, eev, 1); p = np.poly1d(z)
                    xr = np.linspace(np.min(vev), np.max(vev), 100)
                    ax2t.plot(xr, p(xr), 'k-', linewidth=2, alpha=0.8)
                ax2t.set_xlabel('Velocity Error (m/s)'); ax2t.set_ylabel(f'{mn} EMG Error')
                ax2t.set_title(f'{mn}: Vel Error vs EMG Error', fontweight='bold')
                ax2t.grid(True, alpha=0.3); ax2t.legend(fontsize=9)

                ax2m = axes2[1, mi]
                bc2, ve_c2 = [], []
                for bi in range(position_bins):
                    bd = pos_binned[mc][bi]
                    if len(bd['vel_errors']) > 3 and len(bd['emg_errors']) > 3:
                        bc2.append(bd['pos_center'])
                        ve_c2.append(np.corrcoef(bd['vel_errors'], bd['emg_errors'])[0, 1])
                if bc2:
                    ax2m.plot(bc2, ve_c2, 'o-', linewidth=2, label='Vel-EMG Corr', color='purple')
                    ax2m.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax2m.set_xlabel('Position (m)'); ax2m.set_ylabel('Correlation')
                    ax2m.set_title(f'{mn}: Vel-EMG Correlation by Position')
                    ax2m.grid(True, alpha=0.3); ax2m.legend(fontsize=9); ax2m.set_ylim(-1, 1)

                ax2b = axes2[2, mi]
                ax2b.hist(vev, bins=30, alpha=0.7, label=f'Vel Err', color='green', density=True)
                ax2b.hist(eev, bins=30, alpha=0.7, label=f'EMG Err', color='red', density=True)
                ax2b.set_xlabel('Error Value'); ax2b.set_ylabel('Density')
                ax2b.set_title(f'{mn}: Error Distributions'); ax2b.legend(fontsize=9); ax2b.grid(True, alpha=0.3)

        for mi in range(muscle_num):
            if axes1[0, mi].collections:
                fig1.colorbar(axes1[0, mi].collections[0], ax=axes1[0, mi], label='Velocity (m/s)')
            if axes2[0, mi].collections:
                fig2.colorbar(axes2[0, mi].collections[0], ax=axes2[0, mi], label='Position (m)')

        fig1.suptitle('Position-Based: Position Error vs EMG Error', fontsize=14, fontweight='bold', y=0.98)
        fig2.suptitle('Position-Based: Velocity Error vs EMG Error', fontsize=14, fontweight='bold', y=0.98)
        fig1.tight_layout(rect=[0, 0, 1, 0.96])
        fig2.tight_layout(rect=[0, 0, 1, 0.96])

        if save_fig and subject:
            f1p = os.path.join(subject.result_folder, 'pos_error_vs_emg_error_by_position.png')
            f2p = os.path.join(subject.result_folder, 'vel_error_vs_emg_error_by_position.png')
            fig1.savefig(f1p, dpi=300, bbox_inches='tight')
            fig2.savefig(f2p, dpi=300, bbox_inches='tight')
            self._log(f"误差分析图已保存到: {f1p}, {f2p}")

        return {'position_binned_errors': pos_binned, 'position_bins': position_bins}

    # ==================== 单肌肉误差分析 ====================

    def analyze_muscle_kinematic_errors_individual(self, results=None, subject=None,
                                                    load_weights=None, save_fig=True,
                                                    target_muscles=None):
        """每个肌肉单独一张图分析运动学误差与EMG误差的关系"""
        results = results or self.results
        subject = subject or self.subject
        if not results:
            print("没有处理结果可供分析"); return None
        if load_weights is None:
            load_weights = list(results.keys())
        if not load_weights: return None

        if target_muscles is None:
            if subject and subject.target_motion == 'benchpress':
                target_muscles = ['emg_TriLat', 'emg_PMSte']
            else:
                target_muscles = ['emg_FibLon', 'emg_VL', 'emg_RF']
        muscle_names = [c.replace('emg_', '') for c in target_muscles]

        analysis_results = {}

        for mi, (mc, mn) in enumerate(zip(target_muscles, muscle_names)):
            print(f"分析肌肉: {mn}")
            fig, axes = plt.subplots(2, len(load_weights), figsize=(5 * len(load_weights), 8))
            if len(load_weights) == 1: axes = axes.reshape(2, 1)
            fig.suptitle(f'{mn}: Kinematic Errors vs EMG Errors',
                         fontsize=16, fontweight='bold', y=0.98)

            all_pos_arr, all_vel_arr, all_emg_arr = [], [], []
            for lw in load_weights:
                if lw not in results: continue
                cd = results[lw].get('cutted_data')
                if cd is not None and mc in cd.columns:
                    all_pos_arr.append(cd['pos_l'].values)
                    all_vel_arr.append(cd['vel_l'].values)
                    all_emg_arr.append(cd[mc].values)

            avg_pos = avg_vel = avg_emg_val = None
            if all_pos_arr:
                lens = [len(p) for p in all_pos_arr]
                if len(set(lens)) == 1:
                    avg_pos = np.mean(all_pos_arr, axis=0)
                    avg_vel = np.mean(all_vel_arr, axis=0)
                    avg_emg_val = np.mean(all_emg_arr, axis=0)

            load_corrs = []
            for li, lw in enumerate(load_weights):
                if lw not in results:
                    axes[0, li].axis('off'); axes[1, li].axis('off'); continue
                cd = results[lw].get('cutted_data')
                if cd is None or mc not in cd.columns:
                    axes[0, li].axis('off'); axes[1, li].axis('off'); continue

                pos = cd['pos_l'].values
                vel = cd['vel_l'].values
                emg = cd[mc].values

                if avg_pos is not None and len(pos) == len(avg_pos):
                    pe, ve, ee = pos - avg_pos, vel - avg_vel, emg - avg_emg_val
                else:
                    pe = pos - np.mean(pos)
                    ve = vel - np.mean(vel)
                    ee = emg - np.mean(emg)

                pe_corr = np.corrcoef(pe, ee)[0, 1] if len(pe) > 1 else 0
                ve_corr = np.corrcoef(ve, ee)[0, 1] if len(ve) > 1 else 0
                load_corrs.append({'load': lw, 'pos_emg_corr': pe_corr, 'vel_emg_corr': ve_corr})

                axp = axes[0, li]
                axp.scatter(pe, ee, s=20, alpha=0.6, c=vel, cmap='coolwarm',
                            edgecolors='k', linewidth=0.5)
                if len(pe) > 1:
                    try:
                        z = np.polyfit(pe, ee, 1); p_fn = np.poly1d(z)
                        xr = np.linspace(np.min(pe), np.max(pe), 100)
                        axp.plot(xr, p_fn(xr), 'r-', linewidth=2, alpha=0.8)
                    except: pass
                axp.set_xlabel('Position Error (m)'); axp.set_ylabel('EMG Error (mV)')
                axp.set_title(f'{lw}kg\nr = {pe_corr:.3f}', fontweight='bold')
                axp.grid(True, alpha=0.3, linestyle='--')
                axp.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
                axp.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)

                axv = axes[1, li]
                axv.scatter(ve, ee, s=20, alpha=0.6, c=pos, cmap='viridis',
                            edgecolors='k', linewidth=0.5)
                if len(ve) > 1:
                    try:
                        z = np.polyfit(ve, ee, 1); p_fn = np.poly1d(z)
                        xr = np.linspace(np.min(ve), np.max(ve), 100)
                        axv.plot(xr, p_fn(xr), 'r-', linewidth=2, alpha=0.8)
                    except: pass
                axv.set_xlabel('Velocity Error (m/s)'); axv.set_ylabel('EMG Error (mV)')
                axv.set_title(f'{lw}kg\nr = {ve_corr:.3f}', fontweight='bold')
                axv.grid(True, alpha=0.3, linestyle='--')
                axv.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
                axv.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)

            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            if load_corrs:
                avg_pc = np.mean([c['pos_emg_corr'] for c in load_corrs])
                avg_vc = np.mean([c['vel_emg_corr'] for c in load_corrs])
                fig.text(0.02, 0.02,
                         f"Avg Pos-EMG Corr: {avg_pc:.3f}\nAvg Vel-EMG Corr: {avg_vc:.3f}",
                         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            if save_fig and subject:
                fp = os.path.join(subject.result_folder, f'{mn}_kinematic_errors.png')
                plt.savefig(fp, dpi=300, bbox_inches='tight')
                self._log(f"{mn} 误差分析图已保存到: {fp}")

            analysis_results[mn] = {'load_correlations': load_corrs}

        self._create_correlation_summary_plot(
            analysis_results, muscle_names, load_weights, save_fig, subject)
        return analysis_results

    def _create_correlation_summary_plot(self, analysis_results, muscle_names,
                                          load_weights, save_fig, subject=None):
        """创建相关性总结柱状图"""
        subject = subject or self.subject
        if not analysis_results: return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        pc, vc, labels = [], [], []
        for mn in muscle_names:
            if mn in analysis_results:
                for cd in analysis_results[mn].get('load_correlations', []):
                    pc.append(cd['pos_emg_corr'])
                    vc.append(cd['vel_emg_corr'])
                    labels.append(f"{mn}\n{cd['load']}kg")

        x = np.arange(len(pc))
        bars1 = axes[0].bar(x, pc, alpha=0.7, color='blue')
        for i, (b, v) in enumerate(zip(bars1, pc)):
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                         f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Correlation')
        axes[0].set_title('Position Error vs EMG Error', fontweight='bold')
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        axes[0].grid(True, alpha=0.3, axis='y')

        bars2 = axes[1].bar(x, vc, alpha=0.7, color='green')
        for i, (b, v) in enumerate(zip(bars2, vc)):
            axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                         f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('Correlation')
        axes[1].set_title('Velocity Error vs EMG Error', fontweight='bold')
        axes[1].axhline(y=0, color='k', linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_fig and subject:
            fp = os.path.join(subject.result_folder, 'correlation_summary.png')
            plt.savefig(fp, dpi=300, bbox_inches='tight')
            self._log(f"相关性总结图已保存到: {fp}")