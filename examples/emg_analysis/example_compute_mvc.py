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
import json
import os
from digitaltwin import Subject
from digitaltwin.data.emg_processor import EMGProcessor
from digitaltwin.visualization.mvc_plot import (
    plot_emg_signals_grid,
    plot_frequency_spectrum_grid,
    plot_psd_grid,
    plot_artifact_pct_bar,
    plot_mvc_candidates_bar,
)


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

    if not file_names or not muscles_to_plot:
        print('No data to plot'); return

    # --- 绘图 ---
    plot_emg_signals_grid(per_file, file_names, muscles_to_plot, subject.emg_fs)
    plot_frequency_spectrum_grid(per_file, file_names, muscles_to_plot, subject.emg_fs)
    plot_psd_grid(per_file, file_names, muscles_to_plot, subject.emg_fs)
    plot_artifact_pct_bar(per_file, file_names, muscles_to_plot)
    plot_mvc_candidates_bar(per_file, file_names, muscles_to_plot, subject.musc_label, musc_mvc)

    plt.show()


if __name__ == '__main__':
    main()