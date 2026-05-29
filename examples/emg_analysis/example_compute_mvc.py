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
import numpy as np
from digitaltwin import Subject
from digitaltwin.data.emg_processor import EMGProcessor
from digitaltwin.visualization.mvc_plot import (
    plot_emg_signals_grid,
    plot_frequency_spectrum_grid,
    plot_psd_grid,
    plot_artifact_pct_bar,
    plot_mvc_candidates_bar,
)


def compute_mvc_from_file_groups(file_groups, subject,
                                 motion_flag='all',
                                 remove_leading_zeros=False):
    """
    按不同 emg_folder 分组计算 MVC，再跨组取每块肌肉最大值。

    适用于：
      - emg_settings.mvc_file 使用 emg_settings.emg_folder
      - modeling_file.data[*].emg_file 使用 modeling_file.emg_folder

    file_groups : list[dict]
        [{'label': 'mvc', 'emg_folder': path, 'emg_files': [...]}, ...]
    """
    musc_max = None
    per_file = {}
    file_names = []

    for group in file_groups:
        label = group['label']
        emg_folder = group['emg_folder']
        emg_files = group['emg_files']
        if not emg_files:
            continue

        print(f'\n[{label}] EMG folder: {emg_folder}')
        print(f'[{label}] files ({len(emg_files)}): {emg_files}')

        result = EMGProcessor.compute_mvc_from_files(
            emg_files=emg_files,
            emg_folder=emg_folder,
            folder=subject.folder,
            fs=subject.emg_fs,
            musc_label=subject.musc_label,
            motion_flag=motion_flag,
            remove_leading_zeros=remove_leading_zeros,
        )

        group_mvc = np.asarray(result['musc_mvc'], dtype=float)
        if musc_max is None:
            musc_max = group_mvc
        else:
            musc_max = np.maximum(musc_max, group_mvc)

        # 给 per_file 加前缀，避免 mvc_file 和 modeling_file 中同名文件互相覆盖
        for fname, data in result['per_file'].items():
            display_name = f'{label}/{fname}'
            per_file[display_name] = data
            file_names.append(display_name)

    if musc_max is None:
        musc_max = np.zeros(len(subject.musc_label))

    return {
        'musc_mvc': [round(v, 4) for v in musc_max.tolist()],
        'per_file': per_file,
        'file_names': file_names,
    }


def main():
    # --- 配置 ---
    config_path = '../config/20260513_squat_FTS09_xsens.json'
    subject = Subject(config_path)

    # --- 按来源分别收集 EMG 文件 ---
    # emg_settings.mvc_file 应使用 emg_settings.emg_folder
    mvc_files = list(subject.mvc_files)

    # modeling_file.data[*].emg_file 应使用 modeling_file.emg_folder
    modeling_emg_files = []
    for load_key, file_info in subject.modeling_data.items():
        ef = file_info.get('emg_file')
        if ef and ef not in modeling_emg_files:
            modeling_emg_files.append(ef)

    file_groups = [
        {
            'label': 'mvc_file',
            'emg_folder': subject.emg_emg_folder,
            'emg_files': mvc_files,
        },
        {
            'label': 'modeling_file',
            'emg_folder': subject.modeling_emg_folder,
            'emg_files': modeling_emg_files,
        },
    ]

    total_files = sum(len(g['emg_files']) for g in file_groups)
    print(f'所有 EMG 文件 ({total_files})，按 emg_folder 分组计算 MVC')

    # --- 确定 motion_flag ---
    motion_flag = subject.motion_flag
    remove_leading_zeros = subject.remove_leading_zeros

    # --- 强制重新计算 MVC ---
    print('\n强制重新计算 MVC...')
    result = compute_mvc_from_file_groups(
        file_groups=file_groups,
        subject=subject,
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
    muscles_to_plot = subject.musc_label[:12]  # 前 6 块肌肉
    file_names = result['file_names']

    if not file_names or not muscles_to_plot:
        print('No data to plot'); return

    # --- 绘图 ---
    plot_emg_signals_grid(per_file, file_names, muscles_to_plot, subject.emg_fs)
    plot_frequency_spectrum_grid(per_file, file_names, muscles_to_plot, subject.emg_fs)
    plot_psd_grid(per_file, file_names, muscles_to_plot, subject.emg_fs)
    plot_artifact_pct_bar(per_file, file_names, muscles_to_plot)
    plot_mvc_candidates_bar(per_file, file_names, muscles_to_plot,
                            subject.musc_label, musc_mvc)

    plt.show()


if __name__ == '__main__':
    main()