"""
example_opensim_pipeline.py

OpenSim 完整流水线示例：
  Step 1 -- Xsens -> .mot
  Step 2 -- 肌肉分析
  Step 3 -- 逆向动力学

外力模块：digitaltwin/osim/external_forces.py
  包含杆件力（由机器人数据计算）+ 足底 GRF（由鞋垫数据读取）
  生成文件保存到 external_forces/ 共享目录，Step 2/3 共用。
"""
import os
import json

# ---- 配置 -------------------------------------------------------
# 本文件位于 examples/opensim/，项目根目录为向上两级。
# 这样不同电脑上只需要修改 JSON 中的 folder 等数据路径。
base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
config_path = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    '../config/20260513_squat_FTS09_xsens.json'
))

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# ---- 导入 ----------------------------------------------------------
from digitaltwin.osim.mot_pipeline import run_step1_mot_conversion
from digitaltwin.osim.muscle_analysis import run_step2_muscle_analysis
from digitaltwin.osim.inverse_dynamics import run_step3_inverse_dynamics

# ---- Step 1: Xsens -> .mot ------------------------------------------
# 输出到 result/{experiment_label}/opensim/mot/
# run_step1_mot_conversion(config, base_dir)

# ---- Step 2: 肌肉分析 (MuscleAnalysis) -----------------------------
# 输出到 result/{experiment_label}/opensim/muscle_analysis/{load_key}/
# 若 JSON 中设置 opensim_settings.muscle_analysis_muscles，
# 则只分析这些肌肉；否则分析 all。
#
# 不加外力:
# run_step2_muscle_analysis(config, base_dir)
#
# 加入杆件力 + 足底 GRF（推荐）:
run_step2_muscle_analysis(config, base_dir, use_external_forces=True)

# ---- Step 3: 逆向动力学 (InverseDynamics) ---------------------------
# 输出到 result/{experiment_label}/opensim/inverse_dynamics/{load_key}/
# 外力文件如已由 Step 2 生成，将直接复用
#
# 不加外力:
# run_step3_inverse_dynamics(config, base_dir)
#
# 加入杆件力 + 足底 GRF（推荐）:
# run_step3_inverse_dynamics(config, base_dir, use_external_forces=True)