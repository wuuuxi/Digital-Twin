"""
example_xsens_to_mot.py

从 JSON 配置文件读取实验参数，批量将 Xsens Excel 文件转换为 OpenSim .mot 文件。
输出目录： result/{experiment_label}/opensim/mot/
"""
import json
import os

from digitaltwin.osim.opensim_pipeline import run_mot_conversion


# ============================================================
#  ★ 配置
# ============================================================
CONFIG_FILE = '../config/20260513_squat_FTS09.json'
BASE_DIR    = '../..'


def main():
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    base_dir    = os.path.normpath(os.path.join(os.path.dirname(__file__), BASE_DIR))

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    mot_files = run_mot_conversion(config, base_dir, verbose=True)
    print(f'\n✅ 转换完成，共 {len(mot_files)} 个 .mot 文件。')


if __name__ == '__main__':
    main()