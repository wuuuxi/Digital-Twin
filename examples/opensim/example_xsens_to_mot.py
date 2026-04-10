"""
Xsens Excel → OpenSim .mot 批量转换工具

读取 config 中所有负载的 xsens_file，转换为 OpenSim .mot 格式。
默认不自动转换，只有运行本脚本才会执行转换。

输出位置：folder/mot/<xsens_filename>_opensim.mot

用法：
    python example_xsens_to_mot.py
"""
import os
from pathlib import Path
from digitaltwin import Subject
from digitaltwin.data.xsens_processor import XsensProcessor


def main():
    # --- 配置 ---
    subject = Subject('../config/20250409_squat_NCMP001_xsens.json')

    # --- 输出目录 ---
    output_dir = os.path.join(subject.folder, 'mot')
    os.makedirs(output_dir, exist_ok=True)
    print(f'输出目录: {output_dir}')

    # --- 遍历所有负载，转换 xsens 文件 ---
    for load_key, file_info in subject.modeling_data.items():
        xsens_file = file_info.get('xsens_file')
        if xsens_file is None:
            print(f'负载 {load_key}kg: 无 xsens_file，跳过')
            continue

        print(f'\n处理负载 {load_key}kg: {xsens_file}')

        # 加载 Xsens 数据
        xsens_data = XsensProcessor.process(
            xsens_file, load_key, subject.folder,
            xsens_folder=subject.modeling_xsens_folder)

        if xsens_data is None:
            print(f'  加载失败，跳过')
            continue

        # 保存 .mot
        mot_filename = Path(xsens_file).stem + '_opensim.mot'
        mot_path = os.path.join(output_dir, mot_filename)
        XsensProcessor.save_mot(xsens_data, mot_path)

    print(f'\n转换完成！所有 .mot 文件保存在: {output_dir}')


if __name__ == '__main__':
    main()