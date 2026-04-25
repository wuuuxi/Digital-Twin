"""
MVC 可视化工具

提供 MVC 计算过程中各阶段的可视化函数，供 example_compute_mvc.py 等脚本复用。

函数列表：
    plot_emg_signals_grid     - 原始信号 / 带通滤波 / 包络（n_files × n_muscles）
    plot_frequency_spectrum_grid - 单侧幅值谱（n_files × n_muscles）
    plot_psd_grid             - Welch 功率谱密度（n_files × n_muscles）
    plot_artifact_pct_bar     - 运动伪影百分比柱状图（每肌肉一子图）
    plot_mvc_candidates_bar   - MVC 候选值柱状图（每肌肉一子图，红线=最终 MVC）
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def _short_name(fname):
    """提取文件名最后 8 个字符作为标签。"""
    return fname.split('/')[-1][-12:-4]


# ==================================================================
#  图 1：原始信号 + 带通滤波 + 包络
# ==================================================================
def plot_emg_signals_grid(per_file, file_names, muscles, fs):
    """
    绘制 EMG 信号网格图（n_files 行 × n_muscles 列）。

    每个子图包含：去均值原始信号、带通滤波信号、包络信号。

    Parameters
    ----------
    per_file : dict
        compute_mvc_from_files 返回的 per_file 字典。
    file_names : list[str]
        要绘制的文件名列表。
    muscles : list[str]
        要绘制的肌肉名列表。
    fs : int
        采样频率。

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_files = len(file_names)
    n_muscles = len(muscles)
    fig, axes = plt.subplots(n_files, n_muscles,
                             figsize=(3 * n_muscles, 2 * n_files),
                             squeeze=False)
    fig.suptitle('EMG Signals: Raw / Bandpass / Envelope',
                 fontsize=14, fontweight='bold')

    for fi, fname in enumerate(file_names):
        fd = per_file[fname]
        for mi, musc in enumerate(muscles):
            ax = axes[fi, mi]
            if musc not in fd:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            d = fd[musc]
            raw = d['raw']
            filtered = d['filtered']
            envelope = d['envelope']
            t = np.arange(len(raw)) / fs

            ax.plot(t, raw - np.mean(raw), alpha=0.3, linewidth=0.5,
                    color='gray', label='Raw (demeaned)')
            ax.plot(t, filtered, alpha=0.5, linewidth=0.5,
                    color='blue', label='Bandpass')
            ax.plot(t, envelope, linewidth=1.0,
                    color='red', label='Envelope')

            ax.tick_params(labelsize=6)
            if fi == 0:
                ax.set_title(musc, fontsize=9)
            if mi == 0:
                ax.set_ylabel(_short_name(fname), fontsize=7)
            if fi == n_files - 1:
                ax.set_xlabel('Time (s)', fontsize=7)
            if fi == 0 and mi == n_muscles - 1:
                ax.legend(fontsize=5, loc='upper right')

    fig.tight_layout()
    return fig


# ==================================================================
#  图 2：单侧幅值谱
# ==================================================================
def plot_frequency_spectrum_grid(per_file, file_names, muscles, fs):
    """
    绘制 EMG 频谱网格图（n_files 行 × n_muscles 列）。

    每个子图：去均值原始信号的单侧幅值谱（FFT）。

    Parameters
    ----------
    per_file : dict
    file_names : list[str]
    muscles : list[str]
    fs : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_files = len(file_names)
    n_muscles = len(muscles)
    fig, axes = plt.subplots(n_files, n_muscles,
                             figsize=(3 * n_muscles, 2 * n_files),
                             squeeze=False)
    fig.suptitle('EMG Frequency Spectrum (Single-Sided Amplitude)',
                 fontsize=14, fontweight='bold')

    for fi, fname in enumerate(file_names):
        fd = per_file[fname]
        for mi, musc in enumerate(muscles):
            ax = axes[fi, mi]
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
            freqs = np.fft.rfftfreq(N, d=1.0 / fs)

            ax.plot(freqs, fft_mag, linewidth=0.6, color='steelblue')
            ax.set_xlim(0, fs / 2)
            ax.tick_params(labelsize=6)
            if fi == 0:
                ax.set_title(musc, fontsize=9)
            if mi == 0:
                ax.set_ylabel(_short_name(fname), fontsize=7)
            if fi == n_files - 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=7)

    fig.tight_layout()
    return fig


# ==================================================================
#  图 3：功率谱密度（Welch）
# ==================================================================
def plot_psd_grid(per_file, file_names, muscles, fs):
    """
    绘制 EMG 功率谱密度网格图（n_files 行 × n_muscles 列）。

    每个子图：Welch 法估计的 PSD（对数 y 轴）。

    Parameters
    ----------
    per_file : dict
    file_names : list[str]
    muscles : list[str]
    fs : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_files = len(file_names)
    n_muscles = len(muscles)
    fig, axes = plt.subplots(n_files, n_muscles,
                             figsize=(3 * n_muscles, 2 * n_files),
                             squeeze=False)
    fig.suptitle('EMG Power Spectral Density (Welch)',
                 fontsize=14, fontweight='bold')

    for fi, fname in enumerate(file_names):
        fd = per_file[fname]
        for mi, musc in enumerate(muscles):
            ax = axes[fi, mi]
            if musc not in fd:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            d = fd[musc]
            raw = d['raw']
            raw_demeaned = raw - np.mean(raw)
            f_psd, psd = welch(raw_demeaned, fs=fs,
                               nperseg=min(1024, len(raw_demeaned)))

            ax.semilogy(f_psd, psd, linewidth=0.6, color='darkorange')
            ax.set_xlim(0, fs / 2)
            ax.tick_params(labelsize=6)
            if fi == 0:
                ax.set_title(musc, fontsize=9)
            if mi == 0:
                ax.set_ylabel(_short_name(fname), fontsize=7)
            if fi == n_files - 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=7)

    fig.tight_layout()
    return fig


# ==================================================================
#  图 4：运动伪影百分比柱状图
# ==================================================================
def plot_artifact_pct_bar(per_file, file_names, muscles):
    """
    绘制运动伪影百分比柱状图（每个肌肉一个子图）。

    Parameters
    ----------
    per_file : dict
    file_names : list[str]
    muscles : list[str]

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3 * n_rows),
                             squeeze=False)
    fig.suptitle('Motion Artifact Percentage by File',
                 fontsize=14, fontweight='bold')

    for mi, musc in enumerate(muscles):
        ax = axes[mi // n_cols][mi % n_cols]
        artifact_pcts = []
        labels = []

        for fname in file_names:
            fd = per_file[fname]
            if musc in fd:
                artifact_pcts.append(fd[musc]['artifact_pct'])
                labels.append(_short_name(fname))

        if not artifact_pcts:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(musc); continue

        x = np.arange(len(artifact_pcts))
        ax.bar(x, artifact_pcts,
               color=[colors[i % len(colors)] for i in range(len(x))],
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

    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()
    return fig


# ==================================================================
#  图 5：MVC 候选值柱状图
# ==================================================================
def plot_mvc_candidates_bar(per_file, file_names, muscles,
                            musc_label, musc_mvc):
    """
    绘制 MVC 候选值柱状图（每个肌肉一个子图，红线=最终 MVC）。

    Parameters
    ----------
    per_file : dict
    file_names : list[str]
    muscles : list[str]
        要绘制的肌肉子集。
    musc_label : list[str]
        完整的肌肉标签列表（用于索引 musc_mvc）。
    musc_mvc : list[float]
        最终 MVC 值列表，与 musc_label 对应。

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_muscles = len(muscles)
    n_cols = min(3, n_muscles)
    n_rows = (n_muscles + n_cols - 1) // n_cols
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3 * n_rows),
                             squeeze=False)
    fig.suptitle('MVC Candidate Values by File (P95-P99 Avg)\n'
                 'Red line = Final MVC',
                 fontsize=14, fontweight='bold')

    for mi, musc in enumerate(muscles):
        ax = axes[mi // n_cols][mi % n_cols]
        candidates = []
        labels = []

        for fname in file_names:
            fd = per_file[fname]
            if musc in fd:
                candidates.append(fd[musc]['percentile_avg'])
                labels.append(_short_name(fname))

        if not candidates:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(musc); continue

        x = np.arange(len(candidates))
        ax.bar(x, candidates,
               color=[colors[i % len(colors)] for i in range(len(x))],
               alpha=0.8, edgecolor='black', linewidth=0.5)

        for xi, val in zip(x, candidates):
            ax.text(xi, val + 0.001, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=7)

        # 红线标注最终 MVC
        musc_idx = musc_label.index(musc) if musc in musc_label else -1
        if musc_idx >= 0 and musc_idx < len(musc_mvc):
            final_mvc = musc_mvc[musc_idx]
            ax.axhline(y=final_mvc, color='red', linestyle='--',
                       linewidth=2, alpha=0.8,
                       label=f'MVC={final_mvc:.4f}')
            ax.legend(fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax.set_title(musc)
        ax.set_ylabel('P95-P99 Avg (mV)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=7)

    for idx in range(n_muscles, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.tight_layout()
    return fig