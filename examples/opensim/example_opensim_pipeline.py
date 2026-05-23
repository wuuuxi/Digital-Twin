"""
example_opensim_pipeline.py

OpenSim 完整流水线示例：
  Step 1 -- Xsens -> .mot
  Step 2 -- 肌肉分析
  Step 3 -- 逆向动力学

外力模块：digitaltwin/osim/external_forces.py
  包含杆件力（由机器人数据计算）+ 足底 GRF（由鞋庞数据读取）
  生成文件保存到 external_forces/ 共享目录，Step 2/3 共用。
"""
import os
import json

# ---- 配置 -------------------------------------------------------
# 本文件位于 examples/opensim/，项目根目录为向上两级。
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

# ================================================================
# Step 2 配置选项
# ================================================================

# 选项 2：肌肉范围
# True  = 只分析 JSON opensim_settings.muscle_analysis_muscles 中配置的肌肉
# False = 分析模型中所有肌肉 (all)
ONLY_CONFIGURED_MUSCLES = False

# 选项 3：只保留腿部肌肉（left_leg + right_leg 肌肉组）
# True  = 只分析模型 ForceSet 中 left_leg / right_leg 两个组内的肌肉，
#         其他肌肉不参与计算，可显著减少运算量。
# False = 不做额外限制
LEG_MUSCLES_ONLY = True

# ================================================================

# ---- Step 1: Xsens -> .mot ------------------------------------------
# 输出到 result/{experiment_label}/opensim/mot/
# run_step1_mot_conversion(config, base_dir)

# ---- Step 2: 肌肉分析 (MuscleAnalysis) -----------------------------
# 输出到 result/{experiment_label}/opensim/muscle_analysis/{load_key}/
#
# 模式 A：默认激活（不使用 EMG）：
# run_step2_muscle_analysis(
#     config, base_dir,
#     use_only_configured_muscles=ONLY_CONFIGURED_MUSCLES,
#     load_keys=LOAD_KEYS,
# )
#
# 模式 B：EMG 驱动（推荐，需要 JSON 配置 emg_file / emg_settings）：
run_step2_muscle_analysis(
    config, base_dir,
    use_external_forces=True,
    use_emg_controls=True,  # EMG 驱动
    use_only_configured_muscles=False,
    leg_muscles_only=LEG_MUSCLES_ONLY,
    load_keys=None,
)

# ---- Step 3: 逆向动力学 (InverseDynamics) ---------------------------
# 输出到 result/{experiment_label}/opensim/inverse_dynamics/{load_key}/
# 外力文件如已由 Step 2 生成，将直接复用
# run_step3_inverse_dynamics(config, base_dir,
#                            use_external_forces=True,
#                            output_body_forces=True)