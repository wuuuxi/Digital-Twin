"""
example_inverse_dynamics_diagnostics.py

用于排查 inverse_dynamics 输出是否随深蹲负载合理变化。

流程：
  1. 对指定 load 运行 Step 3 InverseDynamics；
  2. 立即复用 example_data_analysis.py / MultiLoadPipeline 的标准运动切片；
  3. 只取标准 upward 阶段；
  4. 将 inverse_dynamics.sto 中的关节力矩按 time 插值到 upward 切片时间点；
  5. 打印每个 load、每个左腿关节的：
       - signed mean：有符号平均力矩
       - mean abs：平均绝对力矩

为什么同时打印 mean 和 mean abs？
  - signed mean 可能因为正负号或相位混合而相互抵消；
  - mean abs 更适合检查“负载增加时力矩绝对值是否上升”。
"""
import os
import json

from digitaltwin.osim.inverse_dynamics import run_step3_inverse_dynamics
from digitaltwin.analysis.result_analysis import (
    load_or_create_cutted_pipeline_results,
    build_left_joint_coordinate_map,
    summarize_inverse_dynamics_moments,
    print_summary_table,
    get_load_keys,
)


# ============================================================
#  配置
# ============================================================

CONFIG_FILE = '../config/20260513_squat_FTS09_xsens.json'

# None = 全部；也可以指定，如 ['20', '38', '56']
LOAD_KEYS = None

# None = 从 opensim_settings.muscle_analysis_coordinates 中自动取所有左腿关节
# 也可以指定，如 ['hip_flexion', 'knee_angle', 'ankle_angle']
JOINT_BASES_TO_PRINT = None

# 只统计标准 upward 阶段；也可改为 ('downward',) 或 ('upward', 'downward')
MOVEMENT_TYPES = ('upward',)

# 切片缓存设置：
#   False = 优先读取 result_folder/cutted_data.csv；
#           如果没有，则尝试用 aligned_data.csv 快速重新切片；
#           如果 aligned_data.csv 也没有，才运行完整 MultiLoadPipeline。
#   True  = 强制重新生成 cutted_data.csv。
FORCE_REBUILD_CUTTED_CACHE = False
CUTTED_CACHE_NAME = 'cutted_data.csv'

# ID 外力设置
USE_EXTERNAL_FORCES = True
MB = 20.0
OUTPUT_BODY_FORCES = False


# ============================================================
#  路径
# ============================================================

def get_base_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))


def get_config_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), CONFIG_FILE))


# ============================================================
#  主程序
# ============================================================

def main():
    base_dir = get_base_dir()
    config_path = get_config_path()

    print(f'配置文件: {config_path}')
    print(f'基准目录: {base_dir}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    load_keys = get_load_keys(config, LOAD_KEYS)
    coord_map = build_left_joint_coordinate_map(
        config,
        joint_bases=JOINT_BASES_TO_PRINT,
    )

    if not coord_map:
        raise ValueError(
            '未找到可统计的左腿关节坐标；请检查 '
            'opensim_settings.muscle_analysis_coordinates'
        )

    print('\n将统计以下左腿关节坐标：')
    for joint_base, coord in coord_map.items():
        print(f'  {joint_base}: {coord}')

    # 1) 先运行 InverseDynamics
    # 注意：run_step3_inverse_dynamics 当前没有 load_keys 参数，
    # 因此这里运行配置中所有可用负载。后续统计表仍按 LOAD_KEYS 过滤打印。
    run_step3_inverse_dynamics(
        config=config,
        base_dir=base_dir,
        use_external_forces=USE_EXTERNAL_FORCES,
        Mb=MB,
        output_body_forces=OUTPUT_BODY_FORCES,
        verbose=True,
    )

    # 2) 再获得标准切片时间点
    #    优先读取 cutted_data.csv；若不存在，则用 aligned_data.csv 快速切片；
    #    两者都没有时，才运行完整 MultiLoadPipeline。
    subject, pipeline, pipeline_results = load_or_create_cutted_pipeline_results(
        config_path,
        include_xsens=False,
        debug=True,
        force_rebuild=FORCE_REBUILD_CUTTED_CACHE,
        cache_name=CUTTED_CACHE_NAME,
    )

    # 3) 打印 signed mean
    id_mean = summarize_inverse_dynamics_moments(
        config=subject.config,
        base_dir=base_dir,
        pipeline_results=pipeline_results,
        load_keys=load_keys,
        coordinates=coord_map,
        movement_types=MOVEMENT_TYPES,
        statistic='mean',
    )

    print_summary_table(
        title=f'ID 关节力矩 signed mean（标准切片: {MOVEMENT_TYPES}）',
        summary=id_mean,
        load_keys=load_keys,
        unit='N·m',
        note=('说明: 对 inverse_dynamics.sto 的 ID moment 按标准切片时间点插值，'
              '然后直接计算有符号平均值。')
    )

    # 4) 打印 mean abs，更适合检查负载增大时绝对力矩是否增大
    id_mean_abs = summarize_inverse_dynamics_moments(
        config=subject.config,
        base_dir=base_dir,
        pipeline_results=pipeline_results,
        load_keys=load_keys,
        coordinates=coord_map,
        movement_types=MOVEMENT_TYPES,
        statistic='mean_abs',
    )

    print_summary_table(
        title=f'ID 关节力矩 mean abs（标准切片: {MOVEMENT_TYPES}）',
        summary=id_mean_abs,
        load_keys=load_keys,
        unit='N·m',
        note=('说明: 对同一批 upward 时间点先取 |ID moment|，再计算平均值；'
              '该表更适合检查负载增加时关节力矩绝对值是否上升。')
    )


if __name__ == '__main__':
    main()