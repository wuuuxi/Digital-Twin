# Digital Twin 当前代码架构

## 1. 文档目的

本文档描述重构完成后的实际代码结构，而不是未来目标结构。项目遵循以下边界：

- `digitaltwin/` 保存可复用的数据处理、分析、OpenSim 和可视化能力；
- `examples/` 保存配置选择、API 调用、少量输出与研究探索代码；
- `examples/` 可以依赖 `digitaltwin/`，`digitaltwin/` 不依赖 `examples/`；
- OpenSim、Pyomo/IPOPT 和 pygame 是可选能力，基础数据分析不应在 import 时加载它们；
- 会写文件的 library API 默认尽量不写，example 需要写出时显式传入 `write=True` 等参数；
- pipeline 使用 `PipelineResults`、`TrialResult`、`TrialMetadata` 表达结果，新的调用方应优先使用字段访问。

本轮采用渐进迁移：`activation`、`processing`、`config` 等是面向新代码的 domain façade；部分成熟算法仍保留在原实现文件中，由 façade 转发，以避免重构改变数值行为。

## 2. 当前目录树

以下目录树省略 `__pycache__/`、实验结果、日志及具体 JSON 配置文件。

```text
Digital Twin/
├── pyproject.toml                                 # 包元数据、基础依赖和可选 extras
├── requirements.txt
├── doc/
│   └── ARCHITECTURE.md
├── digitaltwin/                                  # 可复用 library
│   ├── __init__.py
│   ├── subject.py                                # 实验配置上下文与路径解析
│   ├── config_manager.py                         # 现有配置和负载解析实现
│   ├── config/
│   │   ├── __init__.py
│   │   ├── experiment.py
│   │   ├── loads.py
│   │   └── realtime.py                           # realtime 配置兼容入口
│   ├── models/
│   │   ├── __init__.py
│   │   └── results.py                            # trial 与 pipeline 结构化结果
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_manager.py                       # realtime 数据回放管理
│   │   ├── robot.py                              # 机器人数据 façade
│   │   ├── robot_processor.py                    # 机器人文件读取与标准化
│   │   ├── emg.py                                # EMG 数据 façade
│   │   ├── emg_processor.py                      # EMG 滤波、特征和 MVC 候选
│   │   ├── mvc.py
│   │   ├── xsens.py                              # Xsens 数据 façade
│   │   ├── xsens_processor.py                    # Xsens 关节/节段数据解析
│   │   └── insole/
│   │       ├── __init__.py
│   │       ├── processor.py                      # pipeline 鞋垫处理门面
│   │       ├── io.py
│   │       ├── timebase.py                       # offset 解析和时间修正
│   │       ├── sync.py
│   │       ├── calibration.py
│   │       └── orientation.py
│   ├── processing/                               # 对齐、切片与特征准备入口
│   │   ├── __init__.py
│   │   ├── alignment.py                          # 时间对齐 façade
│   │   ├── segmentation.py                       # 动作切片 façade
│   │   └── features.py                           # EMG/Xsens 特征注入 façade
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── alignment.py                          # 对齐、动作切片与位置标准化实现
│   │   ├── curve_analysis.py
│   │   ├── curves.py                             # 曲线分析 façade
│   │   ├── feature_injector.py
│   │   ├── result_analysis.py                    # 结果表、动作区间和 ID 汇总
│   │   ├── symmetry.py
│   │   ├── tabular.py                            # 表格分析 façade
│   │   ├── activation/                           # 肌肉激活曲面推荐 API
│   │   │   ├── __init__.py
│   │   │   ├── data.py
│   │   │   ├── fitting.py                        # activation 拟合 façade
│   │   │   ├── rbf.py                            # RBF 拟合/预测 façade
│   │   │   ├── pspline.py                        # 单调 P-spline façade
│   │   │   ├── evaluation.py
│   │   │   ├── io.py
│   │   │   └── generator.py                      # 完整拟合流程 façade
│   │   ├── heatmap/                              # activation 历史实现/兼容层
│   │   │   ├── __init__.py
│   │   │   ├── heatmap_generator.py              # 曲面生成与保存编排
│   │   │   ├── heatmap_io.py
│   │   │   ├── rbf_fitting.py
│   │   │   └── monotone_pspline.py
│   │   └── vload/
│   │       ├── __init__.py
│   │       ├── variable_load.py                   # Pyomo/IPOPT 变负载优化
│   │       ├── vload_planning.py
│   │       └── vload_metrics.py
│   ├── pipelines/                                # 跨 domain 流程编排
│   │   ├── __init__.py
│   │   ├── multi_load.py                         # 固定多负载主 pipeline
│   │   ├── standard_analysis.py                  # 标准切片、缓存和动作窗口
│   │   ├── vload.py
│   │   └── symmetry_check.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plot_curves.py
│   │   ├── activation.py                         # activation 绘图 façade
│   │   ├── heatmap.py
│   │   ├── emg_feature_plot.py
│   │   ├── mvc.py                                # MVC 绘图 façade
│   │   ├── mvc_plot.py
│   │   ├── insole_plot.py
│   │   ├── insole_sync_plot.py
│   │   ├── xsens_plot.py
│   │   ├── symmetry_plot.py
│   │   ├── audio.py
│   │   ├── realtime.py
│   │   └── vload/
│   │       ├── __init__.py
│   │       ├── variable_load_plot.py
│   │       ├── vload_comparison_plot.py
│   │       └── vload_result_plot.py
│   ├── osim/                                     # 可选 OpenSim 功能
│   │   ├── __init__.py
│   │   ├── scaling.py
│   │   ├── mot_pipeline.py                       # Xsens→MOT 与质量处理
│   │   ├── external_forces.py
│   │   ├── muscle_analysis.py
│   │   ├── inverse_dynamics.py
│   │   └── realtime/
│   │       ├── __init__.py
│   │       ├── osim_model.py
│   │       └── muscle_state.py
│   └── utils/
│       ├── __init__.py
│       ├── array_tools.py
│       ├── data_io.py
│       ├── file_tools.py
│       └── logger/
│           ├── __init__.py
│           └── beauty_logger.py
└── examples/                                     # API 用法与研究演示
    ├── README.md
    ├── unified_example.py                        # 推荐离线总流程
    ├── config/
    │   └── *.json
    ├── data_analysis/
    │   ├── example_data_analysis.py
    │   ├── example_load_estimation.py
    │   ├── example_load_xsens.py
    │   ├── example_load_xsens_vload.py
    │   ├── example_fatigue_analysis.py
    │   ├── example_height_interaction_force.py
    │   └── insole/
    │       ├── example_data_analysis_insoles.py
    │       ├── example_insole_sync_offset.py
    │       └── example_insole_map_check.py
    ├── emg_analysis/
    │   ├── example_compute_mvc.py
    │   ├── example_emg_frequency.py
    │   ├── example_emg_frequency_vload.py
    │   ├── example_emg_rms.py
    │   └── example_emg_rms_vload.py
    ├── heatmap/                                  # activation 建模与研究探索
    │   ├── example_heatmap.py
    │   ├── example_heatmap_estimated_load.py
    │   ├── example_load_activation_curve.py
    │   ├── example_rbf_and_pspline.py
    │   ├── example_variable_load.py
    │   ├── example_logpower_logactivation_2seg.py # 分段 log-log 研究
    │   └── example_isometric_logpower_logactivation.py # 等长拟合研究
    ├── vload_analysis/
    │   ├── example_vload_comparison.py
    │   ├── example_vload_comparison_xsens.py
    │   ├── example_vload_result.py
    │   ├── example_vload_result_est_load.py
    │   ├── example_fixed_vs_vload_scatter.py
    │   └── example_emg_force_ratio.py
    ├── opensim/
    │   ├── example_opensim_pipeline.py
    │   ├── example_muscle_moment_arm_angle.py
    │   ├── example_muscle_moment_contribution.py
    │   ├── inverse_kinematics/
    │   │   ├── example_scaling.py
    │   │   ├── example_xsens_to_mot.py
    │   │   ├── example_validate_mot.py
    │   │   └── example_symmetry_check.py
    │   └── inverse_dynamics/
    │       ├── example_external_force.py
    │       ├── example_inverse_dynamics.py
    │       ├── example_plot_inverse_dynamics.py
    │       ├── example_cop_sensitivity.py        # COP 敏感性研究
    │       └── example_shear_reconstruction.py   # 剪切力重建研究
    └── metronome/
        ├── bp-metronome.py                       # 卧推节拍与回放
        ├── sq-metronome.py                       # 深蹲节拍与回放
        └── config/
            ├── bp_config.json
            └── sq_config.json
```

## 3. 根目录职责

| 文件或目录 | 职责 |
| --- | --- |
| `pyproject.toml` | 定义项目元数据、基础依赖及 `opensim`、`realtime`、`optimization` 可选依赖。 |
| `requirements.txt` | 保留完整开发/历史环境所需依赖列表。轻量安装优先使用 `pyproject.toml`。 |
| `doc/` | 保存稳定的项目设计与架构说明。 |
| `result/` | 默认实验结果输出目录，不属于 library 源码。 |
| `workspace/` | OpenSim 等流程的工作文件与日志目录，不属于公共 API。 |
| `backup/` | 历史或人工备份内容，不参与当前包导入。 |

## 4. `digitaltwin/` 文件职责

### 4.1 顶层与配置

| 文件 | 职责 |
| --- | --- |
| `digitaltwin/__init__.py` | 轻量公共入口，导出 `Subject`、`MultiLoadPipeline` 和结果 dataclass，不加载 OpenSim/Pyomo/pygame。 |
| `subject.py` | 读取实验配置、解析路径与常用参数、准备 MVC，并提供实验级上下文。 |
| `config_manager.py` | 现有配置与负载解析实现；同时保留 realtime `ConfigManager`。新代码优先经 `digitaltwin.config` 调用。 |
| `config/__init__.py` | 配置 domain 的稳定导出入口。 |
| `config/experiment.py` | 读取和显式写出实验 JSON 配置。 |
| `config/loads.py` | 负载 key、负载模式、筛选、排序和显示相关 façade。 |
| `config/realtime.py` | realtime 配置管理的兼容入口。 |

### 4.2 共享结果模型

| 文件 | 职责 |
| --- | --- |
| `models/__init__.py` | 导出共享结果模型。 |
| `models/results.py` | 定义 `TrialMetadata`、`TrialResult`、`PipelineResults`，并提供旧 `cutted_data` 名称的迁移期兼容访问。 |

推荐访问方式：

```python
results = pipeline.run(write=True)
trial = results["20"]
aligned = trial.aligned_data
segments = trial.segments
```

### 4.3 数据读取与传感器处理

| 文件 | 职责 |
| --- | --- |
| `data/__init__.py` | 导出机器人、EMG、Xsens、MVC 和鞋垫的常用入口。 |
| `data/data_manager.py` | realtime/metronome 的数据回放、动作阶段拆分与帧推进。 |
| `data/robot.py` | 机器人数据的 domain façade。 |
| `data/robot_processor.py` | 读取和标准化机器人数据，处理时间、列名、位置方向和负载 metadata；也支持原始机器人格式。 |
| `data/emg.py` | EMG 数据的 domain façade。 |
| `data/emg_processor.py` | EMG 文件读取、滤波、包络、归一化、RMS、MDF 和 MVC 候选计算。 |
| `data/mvc.py` | 跨文件组汇总 MVC，并在显式要求时生成新的 `*_mvc.json`。 |
| `data/xsens.py` | Xsens 数据的 domain façade。 |
| `data/xsens_processor.py` | 读取 Xsens Excel/MVNX、解析关节角和节段位置、提取缩放测量值。 |

`data/insole/`：

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 导出鞋垫读取、时间偏移、同步、方向和标定 API。 |
| `processor.py` | 面向 pipeline 的 `InsoleProcessor` façade。 |
| `io.py` | 读取力/压力图文件、解析 header/frame/COP，并提供安全重采样。 |
| `timebase.py` | 解析、检查和应用 `insole_time_offset`；统一时间修正规则。 |
| `sync.py` | 识别深蹲有效阶段并以互相关估计鞋垫与机器人时间偏移。 |
| `calibration.py` | 组合机器人参考和鞋垫同步，批量生成标定报告并可显式写回 JSON。 |
| `orientation.py` | 基于接触压力、足印宽度等信息诊断左右及 toe/heel 方向。 |

### 4.4 对齐、切片与特征准备

`digitaltwin.processing` 是新代码应使用的入口；历史算法实现暂时仍在 `analysis` 下。

| 文件 | 职责 |
| --- | --- |
| `processing/__init__.py` | 统一导出对齐、切片和特征注入 API。 |
| `processing/alignment.py` | 转发 `DataAligner` 和 movement-type filter。 |
| `processing/segmentation.py` | 动作切片职责的 façade；当前仍复用 `DataAligner`。 |
| `processing/features.py` | 转发 EMG MDF/RMS 与 Xsens 特征注入函数。 |
| `analysis/alignment.py` | 实现机器人–EMG 对齐、速度过零切片、等长力阈值切片、周期合并与位置标准化。 |
| `analysis/feature_injector.py` | 将 EMG MDF/RMS 和 Xsens 角度/角速度插值到机器人时间轴。 |

### 4.5 通用分析

| 文件 | 职责 |
| --- | --- |
| `analysis/__init__.py` | 导出轻量分析 API；部分旧高层名称通过延迟加载兼容。新代码不应从 analysis 反向调用 pipeline。 |
| `analysis/curves.py` | 曲线分析的 domain façade。 |
| `analysis/curve_analysis.py` | 实现动作阶段平均曲线、位置归一化、插值和分箱统计。 |
| `analysis/tabular.py` | 表格/结果分析的公共 façade。 |
| `analysis/result_analysis.py` | OpenSim 表读取、动作区间提取、插值、ID 力矩汇总等表格分析工具。 |
| `analysis/symmetry.py` | 左右受力、运动学、通道健康、增益、饱和和对称性判据。 |

### 4.6 肌肉激活曲面

`analysis.activation` 是推荐命名；`analysis.heatmap` 保留数值实现和兼容入口。

| 文件 | 职责 |
| --- | --- |
| `activation/__init__.py` | 导出 activation 数据、拟合、预测、评估和模型读取 API；生成器按需加载。 |
| `activation/data.py` | 从 `PipelineResults` 收集切片，并计算逐样本估算负载。 |
| `activation/fitting.py` | RBF/3D activation fitting 的稳定 façade。 |
| `activation/rbf.py` | RBF basis、拟合、预测和序列化 façade。 |
| `activation/pspline.py` | monotone P-spline 拟合与预测 façade。 |
| `activation/evaluation.py` | overall/by-load RMSE 评估 façade。 |
| `activation/io.py` | activation 参数目录与 RBF/P-spline 参数加载 façade。 |
| `activation/generator.py` | 完整 activation fitting workflow 的兼容 façade。 |
| `heatmap/__init__.py` | 历史 heatmap package 入口。 |
| `heatmap/heatmap_generator.py` | 收集切片、生成 nominal/estimated-load activation surface、保存参数并调用通用绘图。 |
| `heatmap/heatmap_io.py` | 读取当前 RBF/P-spline pickle 参数。 |
| `heatmap/rbf_fitting.py` | RBF 与通用 activation map 的数值拟合、预测和 RMSE 实现。 |
| `heatmap/monotone_pspline.py` | 二维单调 P-spline basis、约束拟合和预测实现。 |

分段 log-power/log-activation、isometric 两段拟合仍属于研究探索，保留在 `examples/heatmap/`。

### 4.7 变负载分析

| 文件 | 职责 |
| --- | --- |
| `analysis/vload/__init__.py` | 延迟导出优化、规划文件和评估 API；没有 Pyomo 时仍可使用非优化功能。 |
| `analysis/vload/variable_load.py` | RBF/P-spline 变负载优化、IPOPT 定位、单/多肌肉约束、CSV 输出和绘图编排。 |
| `analysis/vload/vload_planning.py` | 读取并标准化已规划的变负载文件。 |
| `analysis/vload/vload_metrics.py` | 计算实测结果与 RBF/P-spline 预测之间的 RMSE。 |

### 4.8 Pipeline 编排

| 文件 | 职责 |
| --- | --- |
| `pipelines/__init__.py` | 导出固定负载、变负载、标准缓存和对称性 pipeline。 |
| `pipelines/multi_load.py` | 编排机器人、可选 EMG/Xsens/鞋垫的读取、对齐、特征注入、切片、activation fitting 和变负载优化。 |
| `pipelines/standard_analysis.py` | 提供标准数据流程、结构化切片缓存和动作窗口加载/重建。 |
| `pipelines/vload.py` | 处理 variable-load 机器人、EMG、Xsens 和负载规划数据。 |
| `pipelines/symmetry_check.py` | 准备各数据源并运行左右对称性分析与绘图。 |

### 4.9 可视化

| 文件 | 职责 |
| --- | --- |
| `visualization/__init__.py` | 以延迟 import 导出常用图和 realtime 音频类。 |
| `visualization/plot_curves.py` | 标准对齐、动作切片、平均曲线、负载估算和肌肉/运动学诊断图。 |
| `visualization/activation.py` | activation surface 绘图的推荐 façade。 |
| `visualization/heatmap.py` | RBF/P-spline 3D、2D、load-sensitivity 和模型对比图的实现。 |
| `visualization/emg_feature_plot.py` | 固定/变负载 MDF、RMS 对时间、位置和负载的通用图。 |
| `visualization/mvc.py` | MVC 诊断图的推荐 façade。 |
| `visualization/mvc_plot.py` | EMG 信号、频谱、PSD、artifact 和 MVC candidate 图。 |
| `visualization/insole_plot.py` | 压力分布、COP、跨负载 COP 和接触点图。 |
| `visualization/insole_sync_plot.py` | 鞋垫–机器人时间标定诊断图。 |
| `visualization/xsens_plot.py` | Xsens 对齐、动作切片、左右关节和 fixed/vload 对比图。 |
| `visualization/symmetry_plot.py` | SI heatmap、传递链、左右散点、butterfly 和趋势图。 |
| `visualization/audio.py` | pygame 音频资源和动作阶段提示音。 |
| `visualization/realtime.py` | realtime 全局音频调度和播放速度控制。 |
| `visualization/vload/__init__.py` | 延迟导出变负载绘图 API。 |
| `visualization/vload/variable_load_plot.py` | 变负载优化轨迹、目标激活和危险区域图。 |
| `visualization/vload/vload_comparison_plot.py` | fixed/vload 机器人运动学和 EMG 柱状比较。 |
| `visualization/vload/vload_result_plot.py` | 计划、实测、RBF/P-spline/estimated-load overlay 与 RMSE 输出。 |

### 4.10 OpenSim 与 realtime

`digitaltwin.osim` 属于可选功能。基础安装不会从顶层加载 OpenSim。

| 文件 | 职责 |
| --- | --- |
| `osim/__init__.py` | 延迟导出 OpenSim realtime 类，并在缺依赖时给出清晰错误。 |
| `osim/scaling.py` | 全身模型缩放、Xsens 测量映射、bar/insole 接触点和模型测量。 |
| `osim/mot_pipeline.py` | Xsens→MOT、角度 unwrap、丢帧检测、骨盆方向校正和批量转换。 |
| `osim/external_forces.py` | 生成机器人杆力、鞋垫 GRF、COP 和 ExternalLoads 文件。 |
| `osim/muscle_analysis.py` | 构造 EMG controls、选择肌肉并运行单组/批量 MuscleAnalysis。 |
| `osim/inverse_dynamics.py` | 运行单组或批量 InverseDynamics 并保存结果。 |
| `osim/realtime/__init__.py` | 导出 realtime OpenSim model adapter。 |
| `osim/realtime/osim_model.py` | 加载 OpenSim visualizer、设置坐标/激活并更新模型状态。 |
| `osim/realtime/muscle_state.py` | 根据动作高度推算关节角和肌肉激活状态。 |

Shear reconstruction 仍是研究探索，完整实现保留在 example，不进入 `digitaltwin.osim`。

### 4.11 通用工具

| 文件 | 职责 |
| --- | --- |
| `utils/__init__.py` | 导出通用日志入口。 |
| `utils/array_tools.py` | 重采样、连续区间、排序插值和 RMSE 等数组工具。 |
| `utils/data_io.py` | 标准化负载 key，并读取 CSV 数据。 |
| `utils/file_tools.py` | pickle 读写、结果目录创建和兼容调试输出。 |
| `utils/logger/__init__.py` | 导出统一日志 API。 |
| `utils/logger/beauty_logger.py` | 实现 `BeautyLogger` 和 `beauty_print`。 |

## 5. `examples/` 文件职责

examples 只负责选择配置、设置参数、调用 library API 和展示结果。以下研究探索例外：分段 log-power/log-activation、isometric 拟合与 shear reconstruction 暂时保留完整实现。

### 5.1 顶层与标准数据分析

| 文件 | 职责 |
| --- | --- |
| `README.md` | example 导航、运行方式、前置条件和缓存兼容说明。 |
| `unified_example.py` | 推荐离线流程：鞋垫 offset、MVC、标准分析、传感器对齐验证、activation surface、变负载优化及 OpenSim 分支。 |
| `config/*.json` | 示例实验配置和数据路径。 |
| `data_analysis/example_data_analysis.py` | 标准多负载对齐、切片和诊断可视化。 |
| `data_analysis/example_load_estimation.py` | 由机器人力与加速度估计等效负载。 |
| `data_analysis/example_load_xsens.py` | 固定负载 Xsens/EMG/机器人联合分析。 |
| `data_analysis/example_load_xsens_vload.py` | fixed/vload 的 Xsens 和 EMG 联合比较。 |
| `data_analysis/example_fatigue_analysis.py` | 疲劳试次的工作量、速度损失和运动学分析。 |
| `data_analysis/example_height_interaction_force.py` | 按高度展示疲劳组交互力。 |

### 5.2 鞋垫与 EMG

| 文件 | 职责 |
| --- | --- |
| `data_analysis/insole/example_data_analysis_insoles.py` | 把左右鞋垫 GRF 注入标准对齐结果。 |
| `data_analysis/insole/example_insole_sync_offset.py` | 标定鞋垫–机器人 offset，并按开关显式写回配置。 |
| `data_analysis/insole/example_insole_map_check.py` | 检查压力图/力文件、方向、COP、饱和和数据质量。 |
| `emg_analysis/example_compute_mvc.py` | 从 MVC 和 modeling 文件组重新计算 MVC，生成诊断图和新配置。 |
| `emg_analysis/example_emg_frequency.py` | 固定负载 EMG MDF 分析。 |
| `emg_analysis/example_emg_frequency_vload.py` | fixed/vload EMG MDF 对比。 |
| `emg_analysis/example_emg_rms.py` | 固定负载 EMG RMS 分析。 |
| `emg_analysis/example_emg_rms_vload.py` | fixed/vload EMG RMS 对比。 |

### 5.3 Activation surface 与变负载

| 文件 | 职责 |
| --- | --- |
| `heatmap/example_heatmap.py` | 由 nominal load 数据拟合 RBF 和 monotone P-spline activation surface。 |
| `heatmap/example_heatmap_estimated_load.py` | 由逐样本估算负载拟合 activation surface。 |
| `heatmap/example_load_activation_curve.py` | 展示高度–负载–激活/功率关系曲线。 |
| `heatmap/example_rbf_and_pspline.py` | 比较 RBF/P-spline 对变负载实测数据的预测。 |
| `heatmap/example_variable_load.py` | 使用 activation 参数求解目标激活对应的变负载方案。 |
| `heatmap/example_logpower_logactivation_2seg.py` | 研究探索：分段 log-power/log-activation 拟合。 |
| `heatmap/example_isometric_logpower_logactivation.py` | 研究探索：等长试次的力/激活与时间/激活拟合。 |
| `vload_analysis/example_vload_comparison.py` | 无 Xsens 时比较 fixed/vload 机器人与 EMG 结果。 |
| `vload_analysis/example_vload_comparison_xsens.py` | 有 Xsens 时比较 fixed/vload 关节、机器人与 EMG 结果。 |
| `vload_analysis/example_vload_result.py` | 对比计划负载、实测负载和 nominal-load activation 预测。 |
| `vload_analysis/example_vload_result_est_load.py` | 对比计划/实测结果和 estimated-load activation 预测。 |
| `vload_analysis/example_fixed_vs_vload_scatter.py` | 按高度比较 fixed/vload 力与运动散点。 |
| `vload_analysis/example_emg_force_ratio.py` | 比较固定/变负载下肌肉力占比与 EMG 激活占比。 |

### 5.4 OpenSim 与 realtime examples

| 文件 | 职责 |
| --- | --- |
| `opensim/example_opensim_pipeline.py` | 编排 Xsens→MOT、MuscleAnalysis 和 InverseDynamics 步骤。 |
| `opensim/example_muscle_moment_arm_angle.py` | 对齐关节角与肌肉力臂并拟合关系。 |
| `opensim/example_muscle_moment_contribution.py` | 汇总肌肉力矩贡献并与 ID 净力矩比较。 |
| `opensim/inverse_kinematics/example_scaling.py` | 由 Xsens 测量缩放 OpenSim 模型。 |
| `opensim/inverse_kinematics/example_xsens_to_mot.py` | 将 Xsens 文件转换为 OpenSim MOT。 |
| `opensim/inverse_kinematics/example_validate_mot.py` | 检查 MOT 时间、幅值、连续性、丢帧和左右一致性。 |
| `opensim/inverse_kinematics/example_symmetry_check.py` | 运行机器人、鞋垫和 OpenSim 左右对称性检查。 |
| `opensim/inverse_dynamics/example_external_force.py` | 生成并检查 OpenSim 外力文件。 |
| `opensim/inverse_dynamics/example_inverse_dynamics.py` | 运行 OpenSim InverseDynamics。 |
| `opensim/inverse_dynamics/example_plot_inverse_dynamics.py` | 绘制 ID 力矩和力平衡诊断。 |
| `opensim/inverse_dynamics/example_cop_sensitivity.py` | 研究 COP 位置对 ID 力矩与负载单调性的影响。 |
| `opensim/inverse_dynamics/example_shear_reconstruction.py` | 研究探索：用 Xsens 与鞋垫重建剪切外力。 |
| `metronome/bp-metronome.py` | 卧推实时/回放节拍和 OpenSim 可视化。 |
| `metronome/sq-metronome.py` | 深蹲实时/回放节拍和 OpenSim 可视化。 |
| `metronome/config/*.json` | 两种动作的 realtime、audio、playback 和 visualization 配置。 |

## 6. 主要调用与依赖方向

推荐依赖方向为：

```text
config/models/utils
        ↓
       data
        ↓
    processing
        ↓
     analysis
        ↓
    pipelines ─────→ visualization
        ↓
     examples

data/analysis ─────→ osim（仅 OpenSim 分支）
data + osim + audio → realtime examples
```

约束：

1. `digitaltwin` 不 import `examples`。
2. 新 analysis 代码不 import pipelines；`analysis/__init__.py` 中的延迟导出只用于旧调用兼容。
3. 数值分析不依赖 visualization；完整生成流程由 pipeline/generator 负责编排绘图。
4. `models` 不依赖 data、analysis、pipeline 或 OpenSim。
5. 非 OpenSim 模块不得在 import time 强制加载 OpenSim。
6. Pyomo/IPOPT 仅在调用变负载优化时需要；IPOPT 是外部可执行程序。
7. examples 之间原则上不互相 import；研究脚本若仍有历史依赖，应在后续稳定后提取到 library。

## 7. 安装边界与结果兼容

```text
pip install -e .
pip install -e ".[optimization]"
pip install -e ".[opensim]"
pip install -e ".[realtime]"
pip install -e ".[all]"
```

- `optimization` 安装 Pyomo，但仍需单独提供 IPOPT；
- 基础安装支持机器人、EMG、鞋垫、Xsens、activation 和非求解型 vload 分析；
- `MultiLoadPipeline.run()` 默认 `write=False`，example 为保留输出行为会显式传 `write=True`；
- 本次结构化结果和 activation 参数迁移不保证兼容旧缓存。遇到旧 `aligned_data`/`cutted_data` cache 或旧模型参数读取问题时，应重新运行对应 example；
- library 不会主动删除旧结果。
