"""
example_symmetry_check.py

校验【外力信息】与【关节信息】是否相互吻合（S1-S7 的详细说明、
判据与绘图实现都在 digitaltwin/pipelines/symmetry_check.py 里，
本文件只负责填参数并调用）。

为什么需要它：ID 力矩是「运动学 + 外力」两路信息合成的结果。
example_validate_mot.py 已经把运动学那一路判定为可信，但外力那一路
至今没有独立校验。标定漂移、左右接反、单侧通道失效都不会报错，
只会静默地把 ID 力矩算错。

【重要：两种力不是一回事，不能混用】
  force_l / force_r  = 机器人（杠）两侧致动器的力。杠是刚体，两侧分担
                       几乎恒等于 50%，与【哪条腿承重多】无关。
                       它的总和 ≈ 配重 × g，不包含体重。
  grf_l / grf_r      = 鞋垫地面反力。它才是 ID 贴在 calcn_l / calcn_r 上
                       的外力，总和 ≈ (体重 + 配重) × g，分担才反映腿的负荷。
首版本误用 force_l / force_r 当鞋垫力，导致 S1 截距算出体重 ≈ 0 kg，
S3 拿一个恒为 50% 的量去卡 ID 力矩分担。现已改为优先用 grf_l / grf_r，
只有在鞋垫列缺失时才退回机器人力，并相应降级判据。

检查项一览（详细判据见 symmetry_check.py）：
  [S1] 外力总量标定：总力对配重回归，斜率应 ≈ g = 9.81 N/kg
  [S2] 左右力分担  [S3] 力分担 vs ID 力矩分担（方向相反才判错）
  [S4] 力分担 vs 运动学不对称（软关联，仅提示）  [S5] 单侧通道失效
  [S6] 每侧增量增益（GRF 对实测杆力回归，总斜率 1.0 / 单侧 0.5）
  [S7] 饱和 vs 策略 区分器（仅输出证据，不判 FAIL）

运行时机：example_validate_mot.py 通过之后、相信 ID 结果之前。
"""
import os

from digitaltwin.pipelines.symmetry_check import (
    run_symmetry_check,
    SymmetryCheckOptions,
)


# ============================================================
#  配置（参数输入，逻辑在 digitaltwin/pipelines/symmetry_check.py）
# ============================================================

CONFIG_FILE = '../../config/20260513_squat_FTS09_xsens.json'

LOAD_KEYS = None

# 按负载模式筛选，而不是硬写组名。
# 本检查的 S1 / S6 都要把外力对【配重】回归，只有定负载（isotonic）
# 组才有明确的配重；等速/等长组的“等效负载”是从受力反推的，
# 拿去做同一个回归等于用结果验证结果，没有意义。
# 现在三种模式全部参与。但必须分清哪些检查吃得下全模式：
#   S2/S3/S4/S5 与四张图 —— 全模式。它们比的是“同一时刻左右两侧”，
#       不需要知道标称负载是多少。
#   S1/S6/S7    —— 只能用定负载组。它们把外力对【配重】回归，
#       而等速/等长组的“等效负载”本身就是从受力反推出来的，
#       拿去做同一个回归等于用结果验证结果。
# 所以不是“把筛选去掉”，而是把筛选从【取数】移到【具体检查】。
LOAD_MODES_FILTER = None
CALIBRATION_MODES = ('isotonic',)
EXCLUDE_LOAD_KEYS = []

# 同时用上升与下降。只用 upward 会系统性高估总力（加速度向上），
# 两个阶段合起来惯性项在一个完整循环内大致抵消，均值才能与
# 【体重 + 配重】直接比较。
# 等长组的段标的是 movement_type='isometric'，不在这两类里，
# get_segment_from_results 会自动回退到等长段，不需要硬加在这里。
MOVEMENT_TYPES = ('upward', 'downward')

# [S1] 回归斜率允许偏离 g 的相对量
SLOPE_TOL = 0.15
# [S1] 截距推算出的体重合理区间（kg）
BODY_MASS_RANGE = (40.0, 150.0)

# [S2] 左右分担偏离 50% 多少算“明显偏侧”
SHARE_WARN = 0.10
# [S3] 力分担与力矩分担的允许差
SHARE_CONSISTENCY_TOL = 0.12
# [S4] 关节角峰值差超过多少度算“明显不对称”
PEAK_DIFF_WARN = 5.0

# 外力数据源优先级：(左列, 右列, 标记)
# 鞋垫 GRF 才是 ID 实际用的外力；机器人力只是降级退路。
FORCE_SOURCES = (
    ('grf_l', 'grf_r', 'insole'),
    ('force_l', 'force_r', 'robot'),
)
# [S1] 机器人力模式下，截距应接近 0（N）
ROBOT_INTERCEPT_TOL = 100.0
# 切片缓存名：必须与不含鞋垫的缓存分开，否则会读到没有 grf 列的旧缓存
CACHE_NAME = 'cutted_data_insole.csv'

# [S6] GRF 总力对实测杆力的回归斜率，理论值 1.0
TOTAL_GAIN_IDEAL = 1.0
TOTAL_GAIN_TOL = 0.08
# [S6] 单侧增量斜率，理论值 0.5
SIDE_GAIN_IDEAL = 0.5
SIDE_GAIN_TOL = 0.08
# [S6] 两条路径（S1 名义配重 / S6 实测杆力）推算体重的允许差（kg）
BODY_MASS_AGREE_TOL = 5.0
# [S7] 饱和判别：按瞬时总力分箱的箱数
SATURATION_BINS = 8

# [S3] 用哪些关节的 ID 力矩算左右分担
MOMENT_BASES = ('knee_angle', 'hip_flexion', 'ankle_angle')
# [S4] 用哪些关节角看运动学不对称
ANGLE_BASES = ('knee_angle', 'hip_flexion', 'ankle_angle')

# 五张对称性图（绘图实现在 digitaltwin/visualization/symmetry_plot.py）
PLOT_FIGURES = True
SAVE_FIGURES = True
# 第五张三联的纵轴是否统一（3 个子图画的都是 SI，单位都是 %）：
#   'all'    三联完全一致
#   'frames' 只统一左、中两联（都是逐帧 SI），右联组级 SI 自适应
#   'none'   各自独立
SI_TREND_SHARE_Y = 'all'
# 第六张：与第五张完全同构，但只用上升（向心）阶段。
UPWARD_ONLY_TREND = True
UPWARD_MOVEMENT_TYPES = ('upward',)
# 蝴形图需要逐 cycle 的左右关节角曲线，重采样到这么多个点
CYCLE_GRID_POINTS = 101

# 五张图全部画完（并已存盘）后先列清单，由使用者挑选展示哪几张。
# 无人值守运行时读不到输入，会自动退回 PLOT_SELECT_DEFAULT。
# 第三、第五张现在改用图内的组复选框（symmetry_plot.attach_group_selector），
# 若仍想在弹图前按编号挑“展示哪几张”，把 PLOT_SELECT 改回 True 即可。
PLOT_SELECT = False
PLOT_SELECT_DEFAULT = 'all'      # 非交互环境下的默认选择：'all' 或 'none'
PLOT_SELECT_FIGSIZE_SCALE = 1.5  # 被选中的图放大倍数


def main():
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), CONFIG_FILE))

    options = SymmetryCheckOptions(
        slope_tol=SLOPE_TOL,
        body_mass_range=BODY_MASS_RANGE,
        share_warn=SHARE_WARN,
        share_consistency_tol=SHARE_CONSISTENCY_TOL,
        peak_diff_warn=PEAK_DIFF_WARN,
        force_sources=FORCE_SOURCES,
        robot_intercept_tol=ROBOT_INTERCEPT_TOL,
        cache_name=CACHE_NAME,
        total_gain_ideal=TOTAL_GAIN_IDEAL,
        total_gain_tol=TOTAL_GAIN_TOL,
        side_gain_ideal=SIDE_GAIN_IDEAL,
        side_gain_tol=SIDE_GAIN_TOL,
        body_mass_agree_tol=BODY_MASS_AGREE_TOL,
        saturation_bins=SATURATION_BINS,
        moment_bases=MOMENT_BASES,
        angle_bases=ANGLE_BASES,
        cycle_grid_points=CYCLE_GRID_POINTS,
        calibration_modes=CALIBRATION_MODES,
        movement_types=MOVEMENT_TYPES,
    )

    ok = run_symmetry_check(
        config_path,
        options=options,
        load_keys=LOAD_KEYS,
        exclude_load_keys=EXCLUDE_LOAD_KEYS,
        load_modes_filter=LOAD_MODES_FILTER,
        plot_figures=PLOT_FIGURES,
        save_figures=SAVE_FIGURES,
        si_trend_share_y=SI_TREND_SHARE_Y,
        upward_only_trend=UPWARD_ONLY_TREND,
        upward_movement_types=UPWARD_MOVEMENT_TYPES,
        plot_select=PLOT_SELECT,
        plot_select_default=PLOT_SELECT_DEFAULT,
        plot_select_figsize_scale=PLOT_SELECT_FIGSIZE_SCALE,
        show=True,
    )


if __name__ == '__main__':
    main()
