"""Unified example for the recommended offline workflow.

运行方式（建议从项目根目录运行）：
    python examples/unified_example.py

本文件是一个流程编排示例，不替代各目录下的专项 example。默认会运行：
1. 鞋垫时间偏移标定；
2. MVC 计算并生成 *_mvc.json；
3. 使用更新后的配置运行标准多负载分析；
4. 生成普通 RBF / P-spline 热图。

其他验证、变负载和 OpenSim 步骤保留为可选函数，并在 main() 中注释掉。
"""

import json
import os

import matplotlib.pyplot as plt

from digitaltwin import MultiLoadPipeline, Subject
from digitaltwin.data.mvc import (
    compute_mvc_from_file_groups,
    create_mvc_config,
)
from digitaltwin.data.insole import (
    calibrate_insole_offsets,
    check_insole_offsets,
    print_report,
)

# 使用含 Xsens、鞋垫和完整 EMG 配置的示例数据；可替换为其他 config。
CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "config", "20260513_squat_FTS09_xsens.json"
)

# 标准分析是否注入鞋垫 GRF / 是否读取 Xsens。
INCLUDE_INSOLE = True
INCLUDE_XSENS = False

# 热图使用的肌肉和动作阶段。
HEATMAP_MUSCLES = ["LGL", "LBF", "LVL", "LVM", "LGlutMax",
                   "RGL", "RBF", "RVL", "RVM", "RGlutMax"]
MOVEMENT_TYPES = ["upward"]

# 对齐验证图使用的代表性传感器列。
VALIDATION_EMG_COLUMN = "emg_RGL"
VALIDATION_XSENS_COLUMN = "xsens_knee_angle_r"


def get_base_dir():
    # 功能：返回项目根目录，用于 OpenSim 核心函数的输出路径。
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def calibrate_insole_time_offsets():
    # 功能：补齐或显式重算鞋垫与机器人时间偏移。
    subject = Subject(CONFIG_FILE)
    missing = check_insole_offsets(subject, verbose=False)
    report = calibrate_insole_offsets(
        subject,
        write_json=True,
        min_corr=0.5,
        max_lag=30.0,
        corr_thr=0.5,
        min_overlap_frac=0.4,
        plot=True,
        show=False,
    )
    print_report(report)
    return subject


def compute_mvc_and_create_config():
    # 功能：重新计算 MVC，并生成带 musc_mvc 的 *_mvc.json，不覆盖原 config。
    subject = Subject(CONFIG_FILE)

    modeling_emg_files = []
    for file_info in subject.modeling_data.values():
        emg_file = file_info.get("emg_file")
        if emg_file and emg_file not in modeling_emg_files:
            modeling_emg_files.append(emg_file)

    file_groups = [
        {
            "label": "mvc_file",
            "emg_folder": subject.emg_emg_folder,
            "emg_files": list(subject.mvc_files),
        },
        {
            "label": "modeling_file",
            "emg_folder": subject.modeling_emg_folder,
            "emg_files": modeling_emg_files,
        },
    ]

    result = compute_mvc_from_file_groups(
        file_groups,
        subject,
        motion_flag=subject.motion_flag,
        remove_leading_zeros=subject.remove_leading_zeros,
    )
    if not result["file_names"]:
        raise RuntimeError("没有找到可用于 MVC 计算的 EMG 文件。")

    # 维持 unified example 的现有行为，但把有副作用的写回显式表达出来。
    _, mvc_config_path = create_mvc_config(
        subject, result["musc_mvc"], write=True)
    print(f"MVC 配置已保存到: {mvc_config_path}")
    return mvc_config_path


def run_insole_analysis_validation(config_path):
    # 功能：按 CONFIG 中实际存在的传感器，验证机器人与鞋垫/EMG/Xsens
    # 是否共享同一时间轴。机器人数据是必选基准，其余传感器按组可选。
    subject = Subject(config_path)

    def has_file(key):
        return any(
            isinstance(info.get(key), str) and bool(info.get(key).strip())
            for info in subject.modeling_data.values()
        )

    has_insole = has_file("insole_file_l") or has_file("insole_file_r")
    has_emg = has_file("emg_file")
    has_xsens = has_file("xsens_file")

    if has_insole:
        check_insole_offsets(subject)

    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    # 这里不使用 unified example 的全局开关：验证图必须忠实反映当前
    # CONFIG，而不是因为另一个分析步骤关闭了 Xsens 就漏画 Xsens。
    results = pipeline.run(
        include_xsens=has_xsens,
        include_insole=has_insole,
        write=True,
    )

    aligned_cols = set()
    segmented_cols = set()
    for result in results.values():
        if result.aligned_data is not None:
            aligned_cols.update(result.aligned_data.columns)
        segments = result.segments
        if hasattr(segments, "columns"):
            segmented_cols.update(segments.columns)

    # 对齐验证固定选择一条具有明确含义的代表列，避免不同配置因列名排序
    # 而悄悄更换绘图对象。
    emg_cols = sorted(c for c in aligned_cols if c.startswith("emg_"))
    xsens_cols = sorted(c for c in aligned_cols if c.startswith("xsens_"))
    insole_cols = [c for c in ("grf_l", "grf_r") if c in aligned_cols]
    selected_emg = (
        [VALIDATION_EMG_COLUMN]
        if VALIDATION_EMG_COLUMN in aligned_cols else []
    )
    selected_xsens = (
        [VALIDATION_XSENS_COLUMN]
        if VALIDATION_XSENS_COLUMN in aligned_cols else []
    )
    if has_emg and not selected_emg:
        print(f"[Validation] 未找到代表性 EMG 列: {VALIDATION_EMG_COLUMN}")
    if has_xsens and not selected_xsens:
        print(f"[Validation] 未找到代表性 Xsens 列: {VALIDATION_XSENS_COLUMN}")
    alignment_cols = insole_cols + selected_emg + selected_xsens

    # 切片图使用一个代表性 EMG 通道，并把所有可用鞋垫列和少量 Xsens
    # 列作为额外行；没有 EMG 时传入空列表，由 CurvePlotter 省略 EMG 行。
    segment_emg = selected_emg
    segment_extra = [c for c in insole_cols if c in segmented_cols]
    segment_extra += [c for c in selected_xsens if c in segmented_cols]

    print(
        "[Validation] CONFIG sensors: robot=True, "
        f"insole={has_insole}, emg={has_emg}, xsens={has_xsens}")
    print(
        f"[Validation] aligned columns: EMG={len(emg_cols)}, "
        f"Xsens={len(xsens_cols)}, insole={insole_cols}, "
        f"selected_emg={selected_emg}, selected_xsens={selected_xsens}")
    pipeline.visualize_alignment(target_cols=alignment_cols)
    pipeline.visualize_movement_segments(
        target_muscles=segment_emg,
        extra_sensor_cols=segment_extra,
    )
    return subject, pipeline, results


def run_standard_analysis(config_path):
    # 功能：运行固定负载的标准对齐、特征注入和动作切片流程。
    subject = Subject(config_path)
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True
    results = pipeline.run(
        include_xsens=INCLUDE_XSENS,
        include_insole=INCLUDE_INSOLE,
        write=True,
    )
    return subject, pipeline, results


def generate_heatmaps(subject, pipeline):
    # 功能：从标准切片数据拟合 RBF 和 monotone P-spline 热图参数。
    return pipeline.generate_heatmaps(
        muscles=HEATMAP_MUSCLES,
        movement_types=MOVEMENT_TYPES,
        fit_3d=False,
    )


def optimize_variable_load(subject, pipeline):
    # 功能：使用已生成的热图参数，求解配置中目标对应的变负载方案。
    return pipeline.run_variable_load_optimization(
        variable_mode=subject.variable_mode,
        use_pspline=True,
    )


def optimize_variable_load_estimated_load(subject, pipeline):
    # 功能：先用逐样本估算负载生成 heatmap_estimated_load 参数，再求解
    # 配置中目标对应的变负载方案。该探索性分支默认不在 main 中运行。
    pipeline.generate_heatmaps_with_estimated_load(
        muscles=HEATMAP_MUSCLES,
        movement_types=MOVEMENT_TYPES,
    )
    return pipeline.run_variable_load_optimization(
        variable_mode=subject.variable_mode,
        use_pspline=True,
        load_source="estimated",
    )


def compare_variable_load(subject, pipeline):
    # 功能：将实际变负载结果与规划文件及热图预测进行比较。
    vload_results = pipeline.run_vload()
    if not vload_results:
        print("配置中没有可用的变负载结果，跳过比较。")
    return vload_results


def run_opensim_pipeline(config_path):
    # OpenSim is an optional extra; keep the data-only part of this unified
    # example importable in a lightweight installation.
    from digitaltwin.osim.mot_pipeline import run_step1_mot_conversion
    from digitaltwin.osim.muscle_analysis import run_step2_muscle_analysis
    from digitaltwin.osim.inverse_dynamics import run_step3_inverse_dynamics
    from digitaltwin.osim.scaling import scale_from_config

    # 功能：执行 OpenSim 的模型缩放、Xsens→MOT、MuscleAnalysis 和 ID 主流程。
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    base_dir = get_base_dir()

    scale_from_config(config, base_dir, verbose=True)
    run_step1_mot_conversion(config, base_dir)
    run_step2_muscle_analysis(
        config,
        base_dir,
        use_external_forces=True,
        use_emg_controls=True,
        leg_muscles_only=True,
    )
    run_step3_inverse_dynamics(
        config,
        base_dir,
        use_external_forces=True,
        output_body_forces=False,
    )


def main():
    # calibrate_insole_time_offsets()
    # mvc_config_path = compute_mvc_and_create_config()  # 重新计算 MVC
    mvc_config_path = CONFIG_FILE

    subject, pipeline, _ = run_standard_analysis(mvc_config_path)
    run_insole_analysis_validation(mvc_config_path)  # 带 GRF 的对齐/切片分析
    plt.show()

    generate_heatmaps(subject, pipeline)
    optimize_variable_load(subject, pipeline)  # 根据热图求解变负载方案

    # optimize_variable_load_estimated_load(subject, pipeline)
    # 需要时先生成 heatmap_estimated_load，再取消本行注释。

    # compare_variable_load(subject, pipeline)  # 比较已有的变负载实测结果。

    run_opensim_pipeline(mvc_config_path)  # OpenSim 分支

    plt.show()


if __name__ == "__main__":
    main()
