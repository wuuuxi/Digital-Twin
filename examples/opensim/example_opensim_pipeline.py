"""
example_opensim_pipeline.py

从 JSON 配置文件分步驱动 OpenSim 后处理：
  Step 1 — Xsens Excel -> .mot 关节角度文件
  Step 2 — 肌肉分析 (MuscleAnalysis)
  Step 3 — 逆向动力学 (InverseDynamics, 按需开启)

前提: 已运行 example_scaling.py 完成模型缩放。
如果某步骤已运行过，直接注释掉对应函数调用即可。
只需修改 CONFIG_FILE 和 BASE_DIR 即可运行。
"""
import json
import os

from digitaltwin.osim.opensim_pipeline import (
    run_step1_mot_conversion,
    run_step2_muscle_analysis,
    run_step3_inverse_dynamics,
    generate_bar_external_loads,
)


# ============================================================
#  配置
# ============================================================
CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'
BASE_DIR    = '../..'


def main():
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    base_dir    = os.path.normpath(os.path.join(os.path.dirname(__file__), BASE_DIR))

    print(f'配置文件: {config_path}')
    print(f'基准目录: {base_dir}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ---- Step 1: Xsens -> .mot ----------------------------------------
    # 输出到 result/{experiment_label}/opensim/mot/
    # 已运行过则可注释掉此行
    # run_step1_mot_conversion(config, base_dir)

    # ---- Step 2: 肌肉分析 -----------------------------------------------
    # 输出到 result/{experiment_label}/opensim/muscle_analysis/{load_key}/
    # 外力文件共享到 result/{experiment_label}/opensim/external_forces/{load_key}/
    # coordinates=None 时读取 JSON 中的 muscle_analysis_coordinates
    # 已运行过则可注释掉此行
    #
    # 不加外力:
    # run_step2_muscle_analysis(config, base_dir)
    #
    # 加入杆件力（推荐）：外力文件会自动生成到共享目录
    run_step2_muscle_analysis(config, base_dir, use_bar_force=True)

    # ---- Step 3: 逆向动力学（按需开启）----------------------------------
    # 输出到 result/{experiment_label}/opensim/inverse_dynamics/{load_key}/
    #
    # 方案A：不加外力
    # run_step3_inverse_dynamics(config, base_dir)
    #
    # 方案B：自动从机器人数据计算杆件力（推荐）
    # 力 = force_l + force_r + Mb*g + Mb*avg(acc_l, acc_r)
    # 作用体/作用点可在 JSON opensim_settings 中配置
    # run_step3_inverse_dynamics(config, base_dir, use_bar_force=True)
    #
    # 方案C：手动指定外力 XML
    # run_step3_inverse_dynamics(config, base_dir,
    #                            external_load_file='path/to/loads.xml')


if __name__ == '__main__':
    main()