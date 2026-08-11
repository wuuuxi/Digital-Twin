"""
symmetry_check.py — 左右对称性校验的【编排层】。

主流程入口 run_symmetry_check()：
  1. 读 config；
  2. 调 load_or_create_cutted_pipeline_results(include_insole=True)
     取得含鞋垫 GRF 的切片结果；
  3. 用 analysis/symmetry.py 的纯算法（collect_side_data / check_* /
     Verdicts）做 S1-S7 校验；
  4. 触发对称性图（plot_symmetry_figures）。

纯计算部分（SymmetryCheckOptions / collect_side_data / Verdicts /
check_*）已移至 digitaltwin/analysis/symmetry.py，此处保留重导出以便
旧路径（digitaltwin.pipelines.symmetry_check.*）继续可用。
"""
import json
import os

from digitaltwin.analysis.symmetry import (
    SymmetryCheckOptions,
    Verdicts,
    collect_side_data,
    check_force_calibration,
    check_share_consistency,
    check_kinematic_side,
    check_channel_health,
    check_side_gain,
    check_saturation,
    _find_base_dir,
)
from digitaltwin.config_manager import filter_load_keys
from digitaltwin.pipelines.standard_analysis import (
    load_or_create_cutted_pipeline_results,
)
from digitaltwin.utils.logger import beauty_print
from digitaltwin.visualization.symmetry_plot import plot_symmetry_figures

# 兼容重导出（其余见 analysis/symmetry.py）
__all__ = [
    'run_symmetry_check',
    'SymmetryCheckOptions',
    'collect_side_data',
    'Verdicts',
    'check_force_calibration',
    'check_share_consistency',
    'check_kinematic_side',
    'check_channel_health',
    'check_side_gain',
    'check_saturation',
]


def run_symmetry_check(config_path, options=None, *,
                       load_keys=None,
                       exclude_load_keys=None,
                       load_modes_filter=None,
                       plot_figures=True,
                       save_figures=True,
                       si_trend_share_y='all',
                       upward_only_trend=True,
                       upward_movement_types=('upward',),
                       plot_select=False,
                       plot_select_default='all',
                       plot_select_figsize_scale=1.5,
                       show=True):
    """
    外力信息 vs 关节信息 一致性校验（S1-S7）+ 对称性图。

    Parameters
    ----------
    config_path : str -- config json 的完整路径
    options : SymmetryCheckOptions, optional -- 判据阈值配置，None 用默认
    load_keys : list[str], optional -- 参与的负载组，None = 全部
    exclude_load_keys : list[str], optional -- 排除的负载组
    load_modes_filter : tuple[str], optional -- 按负载模式筛选
        （None = 全部；('isotonic',) = 只跑定负载组）
    plot_figures : bool -- 是否画五张对称性图
    save_figures : bool -- 是否把图存到 result/<subject>/symmetry/
    si_trend_share_y : {'all', 'frames', 'none'} -- 第五张三联的纵轴统一方式
    upward_only_trend : bool -- 是否另取一份仅上升阶段的数据画第六张图
    upward_movement_types : tuple -- 第六张图的段类型
    plot_select : bool -- 是否在弹图前按编号挑选展示哪几张
    plot_select_default : {'all', 'none'} -- 非交互环境的默认选择
    plot_select_figsize_scale : float -- 被选中图的放大倍数
    show : bool -- 结束时是否 plt.show()

    Returns
    -------
    bool -- True 全部检查通过
    """
    if options is None:
        options = SymmetryCheckOptions()

    if not os.path.exists(config_path):
        beauty_print(f'找不到配置文件: {config_path}', type='warning')
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    base_dir = _find_base_dir()

    load_keys = filter_load_keys(config, load_keys=load_keys,
                                 modes=load_modes_filter,
                                 exclude=exclude_load_keys)
    print(f'参与负载: {load_keys}（模式筛选: {load_modes_filter}）')
    print(f'其中只有 {options.calibration_modes} 组会进入 S1/S6/S7 的标定类回归。')

    # 必须 include_insole=True，否则切片里没有 grf_l / grf_r，
    # 只能退回到机器人力，而机器人力无法回答“哪条腿承重多”。
    _subject, _pipeline, pipeline_results = \
        load_or_create_cutted_pipeline_results(
            config_path, include_xsens=False, include_insole=True,
            debug=False, cache_name=options.cache_name)

    data = collect_side_data(config, base_dir, pipeline_results, load_keys,
                             options)
    if not data:
        beauty_print('没有任何负载收集到有效数据，无法校验。', type="warning")
        return False

    # 标定类检查只能吃定负载组；对称性检查与四张图吃全部模式。
    calib = {k: v for k, v in data.items()
             if v.get('mode') in options.calibration_modes}
    skipped = sorted(set(data.keys()) - set(calib.keys()))
    if skipped:
        print(f'\n[S1/S6/S7] 仅用定负载组 {sorted(calib.keys())}；'
              f'跳过 {skipped}（其等效负载本身就是从受力反推的，'
              f'再拿去对受力回归是循环论证）。')

    verdicts = Verdicts()
    if calib:
        check_force_calibration(calib, verdicts, options)
    else:
        beauty_print('一个定负载组都没有，S1/S6/S7 全部跳过；'
                     '外力总量标定本次无法验证。', type="warning")

    check_share_consistency(data, verdicts, options)
    check_kinematic_side(data, verdicts, options)
    check_channel_health(data, verdicts, options)

    if calib:
        check_side_gain(calib, verdicts, options)
        check_saturation(calib, verdicts, options)

    all_ok = verdicts.report()

    if plot_figures:
        out_dir = None
        if save_figures and _subject is not None:
            out_dir = os.path.join(_subject.result_folder, 'symmetry')
        print('\n' + '=' * 80)
        print('[图] 对称性：SI 热图 / 运动链传递 / 左-右散点 / 蝴形图 / '
              'SI 趋势（vs 合力、vs 杆高）' +
              (' / SI 趋势·仅上升阶段' if upward_only_trend else ''))
        print('=' * 80)

        data_up = None
        if upward_only_trend:
            print(f'[图6] 另取一份仅 {upward_movement_types} 的数据（等长组无'
                  f'上升/下降之分，会自动回退到等长段，整组保留）…')
            data_up = collect_side_data(config, base_dir, pipeline_results,
                                        load_keys, options,
                                        movement_types=upward_movement_types)

        plot_symmetry_figures(data, options.moment_bases, options.angle_bases,
                              out_dir=out_dir, show=show,
                              select=plot_select,
                              select_default=plot_select_default,
                              figsize_scale=plot_select_figsize_scale,
                              si_trend_share_y=si_trend_share_y,
                              data_upward=data_up)

    return all_ok
