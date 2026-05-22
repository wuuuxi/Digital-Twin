"""
example_scaling.py

从 JSON 配置文件读取实验参数，对 OpenSim 全身模型进行缩放。

流程：
  1. 读取 JSON → 定位 xsens 文件、输入/输出模型路径
  2. 从第一个 xsens 文件提取节段长度 → 缩放模型
  3. 用第二个 xsens 文件进行交叉验证
  4. 打印模型整体身高与总质量
  5. 计算深蹲杆接触点（torso 局部坐标），打印并回写到 JSON
"""
import json
import os

from digitaltwin.osim.scaling import scale_from_config


CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'


def main():
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))
    base_dir    = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
    overwrite_bar_contact_point = False

    print(f'配置文件: {config_path}')
    print(f'基准目录: {base_dir}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    result = scale_from_config(config, base_dir, verbose=True)

    if result is None:
        print('\n❌ 缩放失败，请检查配置与日志。')
        return

    model, bar_contact_point, insole_contact_point = result

    print(f'\n✅ 缩放完成！')
    print(f'\n杆接触点     (torso 局部坐标): {bar_contact_point}')
    print(f'鞋垫接触点   (calcn 局部坐标): {insole_contact_point}')

    # 回写到 JSON
    if overwrite_bar_contact_point is True:
        if 'opensim_settings' not in config:
            config['opensim_settings'] = {}
        config['opensim_settings']['bar_contact_point']   = bar_contact_point
        config['opensim_settings']['insole_contact_point'] = insole_contact_point

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f'\n已将 bar_contact_point 和 insole_contact_point 回写到: {config_path}')


if __name__ == '__main__':
    main()