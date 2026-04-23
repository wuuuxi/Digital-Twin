"""
MVC 计算与可视化示例

强制重新计算 MVC（即使 JSON 中已有 musc_mvc），并可视化中间结果。

图 1：大图（n_files 行 × n_muscles 列）
      每个子图：原始信号 + 带通滤波信号 + 包络信号
图 2：频谱图（n_files 行 × n_muscles 列）
      每个子图：原始信号的单侧幅值谱
图 3：功率谱密度（n_files 行 × n_muscles 列）
      每个子图：Welch 法估计的 PSD
图 4：运动伪影百分比柱状图（每个肌肉一个子图）
图 5：MVC 候选值柱状图（每个肌肉一个子图，最终 MVC 用红线标注）

用法：
    python example_compute_mvc.py
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from scipy.signal import welch
from digitaltwin import Subject
from digitaltwin.data.emg_processor import EMGProcessor


def main():
    # --- 配置 ---
    config_path = '../config/20250409_squat_NCMP001.json'
    subject = Subject(config_path)

    # --- 收集所有 EMG 文件 ---
    all_emg_files = list(subject.mvc_files)  # mvc_file 列表
    for load_key, file_info in subject.modeling_data.items():
        ef = file_info.get('emg_file')
        if ef and ef not in all_emg_files:
            all_emg_files.append(ef)

    print(f'所有 EMG 文件 ({len(all_emg_files)}): {all_emg_files}')

    # --- 确定 emg_folder ---
    emg_folder = subject.emg_emg_folder or subject.modeling_emg_folder

    # --- 确定 motion_flag ---
    motion_flag = subject.motion_flag
    remove_leading_zeros = subject.remove_leading_zeros

    # --- 强制重新计算 MVC ---
    print('\n强制重新计算 MVC...')
    result = EMGProcessor.compute_mvc_from_files(
        emg_files=all_emg_files,
        emg_folder=emg_folder,
        folder=subject.folder,
        fs=subject.emg_fs,
        musc_label=subject.musc_label,
        motion_flag=motion_flag,
        remove_leading_zeros=remove_leading_zeros,
    )

    musc_mvc = result['musc_mvc']
    per_file = result['per_file']
    print(f'\nMVC 结果: {musc_mvc[:6]}...')

    # --- 写回新 JSON 文件（原文件名 + _mvc） ---
    base, ext = os.path.splitext(config_path)
    mvc_config_path = f'{base}_mvc{ext}'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    config.setdefault('emg_settings', {})['musc_mvc'] = musc_mvc
    with open(mvc_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'MVC 已保存到新文件: {mvc_config_path}')

    # --- 可视化参数 ---
    muscles_to_plot = subject.musc_label[:6]  # 前 6 块肌肉
    file_names = [f for f in all_emg_files if f in per_file]
    n_files = len(file_names)
    n_muscles = len(muscles_to_plot)

    if n_files == 0 or n_muscles == 0:
        print('No data to plot'); return

    # ==================================================================
    #  图 1：原始信号 + 带通滤波 + 包络（n_files 行 × n_muscles 列）
    # ==================================================================
    fig1, axes1 = plt.subplots(n_files, n_muscles,
                               figsize=(3 * n_muscles, 2 * n_files),
                               squeeze=False)
    fig1.suptitle('EMG Signals: Raw / Bandpass / Envelope',
                  fontsize=14, fontweight='bold')

    for fi, fname in enumerate(file_names):
        fd = per_file[fname]
        for mi, musc in enumerate(muscles_to_plot):
            ax = axes1[fi, mi]
            if musc not in fd:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            d = fd[musc]
            raw = d['raw']
            filtered = d['filtered']
            envelope = d['envelope']
            t = np.arange(len(raw)) / subject.emg_fs

            ax.plot(t, raw - np.mean(raw), alpha=0.3, linewidth=0.5, color='gray', label='Raw (demeaned)')
            ax.plot(t, filtered, alpha=0.5, linewidth=0.5, color='blue', label='Bandpass')
            ax.plot(t, envelope, linewidth=1.0, color='red', label='Envelope')

            ax.tick_params(labelsize=6)
            if fi == 0:
                ax.set_title(musc, fontsize=9)
            if mi == 0:
                short_name = fname.split('/')[-1][-12:-4]
                ax.set_ylabel(short_name, fontsize=7)
            if fi == n_files - 1:
                ax.set_xlabel('Time (s)', fontsize=7)
            if fi == 0 and mi == n_muscles - 1:
                ax.legend(fontsize=5, loc='upper right')

    fig1.tight_layout()

    # ==================================================================
    #  图 2：频谱图（n_files 行 × n_muscles 列）
    # ==================================================================
    fig2, axes2 = plt.subplots(n_files, n_muscles,
                               figsize=(3 * n_muscles, 2 * n_files),
                               squeeze=False)
    fig2.suptitle('EMG Frequency Spectrum (Single-Sided Amplitude)',
                  fontsize=14, fontweight='bold')

    for fi, fname in enumerate(file_names):
        fd = per_file[fname]
        for mi, musc in enumerate(muscles_to_plot):
            ax = axes2[fi, mi]
            if musc not in fd:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            d = fd[musc]
            raw = d['raw']
            raw_demeaned = raw - np.mean(raw)
            N = len(raw_demeaned)
            fft_vals = np.fft.rfft(raw_demeaned)
            fft_mag = np.abs(fft_vals) * 2.0 / N
            freqs = np.fft.rfftfreq(N, d=1.0 / subject.emg_fs)

            ax.plot(freqs, fft_mag, linewidth=0.6, color='steelblue')
            ax.set_xlim(0, subject.emg_fs / 2)
            ax.tick_params(labelsize=6)
            if fi == 0:
                ax.set_title(musc, fontsize=9)
            if mi == 0:
                short_name = fname.split('/')[-1][-12:-4]
                ax.set_ylabel(short_name, fontsize=7)
            if fi == n_files - 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=7)

    fig2.tight_layout()

    # ==================================================================
    #  图 3：功率谱密度 PSD（n_files 行 × n_muscles 列）
    # ==================================================================
    fig3psd, axes3psd = plt.subplots(n_files, n_muscles,
                                     figsize=(3 * n_muscles, 2 * n_files),
                                     squeeze=False)
    fig3psd.suptitle('EMG Power Spectral Density (Welch)',
                     fontsize=14, fontweight='bold')

    for fi, fname in enumerate(file_names):
        fd = per_file[fname]
        for mi, musc in enumerate(muscles_to_plot):
            ax = axes3psd[fi, mi]
            if musc not in fd:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            d = fd[musc]
            raw = d['raw']
            raw_demeaned = raw - np.mean(raw)
            f_psd, psd = welch(raw_demeaned, fs=subject.emg_fs,
                               nperseg=min(1024, len(raw_demeaned)))

            ax.semilogy(f_psd, psd, linewidth=0.6, color='darkorange')
            ax.set_xlim(0, subject.emg_fs / 2)
            ax.tick_params(labelsize=6)
            if fi == 0:
                ax.set_title(musc, fontsize=9)
            if mi == 0:
                short_name = fname.split('/')[-1][-12:-4]
                ax.set_ylabel(short_name, fontsize=7)
            if fi == n_files - 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=7)

    fig3psd.tight_layout()

    # ==================================================================
    #  图 4：运动伪影百分比柱状图（每个肌肉一个子图）
    # ==================================================================
    n_cols_bar = min(3, n_muscles)
    n_rows_bar = (n_muscles + n_cols_bar - 1) // n_cols_bar

    fig4, axes4 = plt.subplots(n_rows_bar, n_cols_bar,
                               figsize=(5 * n_cols_bar, 3 * n_rows_bar),
                               squeeze=False)
    fig4.suptitle('Motion Artifact Percentage by File',
                  fontsize=14, fontweight='bold')
    colors_bar = plt.cm.tab10.colors

    for mi, musc in enumerate(muscles_to_plot):
        ax = axes4[mi // n_cols_bar][mi % n_cols_bar]
        artifact_pcts = []
        labels = []

        for fi, fname in enumerate(file_names):
            fd = per_file[fname]
            if musc in fd:
                artifact_pcts.append(fd[musc]['artifact_pct'])
                labels.append(fname.split('/')[-1][-12:-4])

        if not artifact_pcts:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(musc); continue

        x = np.arange(len(artifact_pcts))
        ax.bar(x, artifact_pcts,
               color=[colors_bar[i % len(colors_bar)] for i in range(len(x))],
               alpha=0.8, edgecolor='black', linewidth=0.5)

        for xi, pct in zip(x, artifact_pcts):
            ax.text(xi, pct + 0.5, f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.axhline(y=15, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.8, label='15% threshold')
        ax.set_title(musc)
        ax.set_ylabel('Artifact %')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=7)

    for idx in range(n_muscles, n_rows_bar * n_cols_bar):
        axes4[idx // n_cols_bar][idx % n_cols_bar].set_visible(False)
    fig4.tight_layout()

    # ==================================================================
    #  图 5：MVC 候选值柱状图（每个肌肉一个子图，红线=最终 MVC）
    # ==================================================================
    fig5, axes5 = plt.subplots(n_rows_bar, n_cols_bar,
                               figsize=(5 * n_cols_bar, 3 * n_rows_bar),
                               squeeze=False)
    fig5.suptitle('MVC Candidate Values by File (P95-P99 Avg)\n'
                  'Red line = Final MVC',
                  fontsize=14, fontweight='bold')

    for mi, musc in enumerate(muscles_to_plot):
        ax = axes5[mi // n_cols_bar][mi % n_cols_bar]
        candidates = []
        labels = []

        for fi, fname in enumerate(file_names):
            fd = per_file[fname]
            if musc in fd:
                candidates.append(fd[musc]['percentile_avg'])
                labels.append(fname.split('/')[-1][-12:-4])

        if not candidates:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(musc); continue

        x = np.arange(len(candidates))
        ax.bar(x, candidates,
               color=[colors_bar[i % len(colors_bar)] for i in range(len(x))],
               alpha=0.8, edgecolor='black', linewidth=0.5)

        for xi, val in zip(x, candidates):
            ax.text(xi, val + 0.001, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=7)

        # 红线标注最终 MVC
        musc_idx = subject.musc_label.index(musc) if musc in subject.musc_label else -1
        if musc_idx >= 0 and musc_idx < len(musc_mvc):
            final_mvc = musc_mvc[musc_idx]
            ax.axhline(y=final_mvc, color='red', linestyle='--',
                       linewidth=2, alpha=0.8, label=f'MVC={final_mvc:.4f}')
            ax.legend(fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_title(musc)
        ax.set_ylabel('P95-P99 Avg (mV)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=7)

    for idx in range(n_muscles, n_rows_bar * n_cols_bar):
        axes5[idx // n_cols_bar][idx % n_cols_bar].set_visible(False)
    fig5.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()