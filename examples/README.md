# Examples Guide

本文档按脚本实际调用的核心代码整理；`examples/config/` 中的 JSON 是数据集配置，不是可单独运行的 example。当前 library 边界和各文件职责见 [`doc/ARCHITECTURE.md`](../doc/ARCHITECTURE.md)。基础功能可用 `pip install -e .` 安装；OpenSim、realtime 和 optimization 功能分别使用对应 extra，完整历史环境仍可参考根目录 `requirements.txt`。

## Refactored library entry points

新的示例应把数据处理和可复用算法放在 `digitaltwin/`，脚本本身只保留配置、路径、API 调用和演示性绘图。标准固定负载流程现在返回结构化结果：

```python
from digitaltwin import MultiLoadPipeline, Subject

subject = Subject(CONFIG_FILE)
pipeline = MultiLoadPipeline(subject)
results = pipeline.run(include_xsens=False, write=True)  # 写文件必须显式声明
trial = results["50"]
aligned = trial.aligned_data
segments = trial.segments
combined = results.collect_segments(movement_types=["upward"])
```

`trial.segments` 是原 `cutted_data` 的命名字段；迁移期间仍可通过 `trial.get("cutted_data")` 读取旧名称。激活曲面相关的新入口是 `digitaltwin.analysis.activation`，旧的 `analysis.heatmap` 仅作为兼容层保留；现有 `examples/heatmap/` 文件暂不移动，后续再按 domain 重命名。默认安装不需要 OpenSim；只有运行 OpenSim 目录下的示例时才安装相应可选依赖。

变负载优化除了 Pyomo 还需要 IPOPT 可执行文件。conda 环境可使用 `conda install -c conda-forge ipopt`；Windows 下 library 会同时检查当前环境的 `Library/bin/ipopt.exe`，也可通过 `DIGITALTWIN_IPOPT_EXECUTABLE` 指定完整路径。

本轮结果 dataclass 和缓存结构发生变化，旧的 `aligned_data`/`cutted_data` 缓存或热图参数不保证可复用。若读取失败，请删除或忽略旧结果并重新运行生成。

## Quick Overview

| Example / group | Main function | Order / relation |
| --- | --- | --- |
| Unified offline workflow | `unified_example.py` 按配置串联鞋垫 offset、MVC、标准分析、传感器对齐验证、activation surface、变负载优化和 OpenSim 分支 | 推荐总入口；按需要注释或开启耗时步骤 |
| Standard multi-load analysis | 用 `MultiLoadPipeline.run()` 对固定负载的机器人、EMG、鞋垫数据做对齐、切片和诊断可视化 | 通用入口；鞋垫时间偏移正确时可运行 |
| Xsens analysis | 将 Xsens 关节运动加入固定负载结果，比较左右关节、EMG 与运动学 | 独立分支；需使用含 Xsens 的 config |
| Xsens + variable-load analysis | 同时比较固定负载和变负载的 Xsens/EMG/关节结果 | `example_load_xsens.py` 的扩展分支 |
| Load estimation | 根据机器人力与加速度估计等效负载并可视化 | 独立诊断；依赖标准 `run()` 结果 |
| Fatigue analysis | 读取 `fatigue_file` 的多组机器人数据，绘制疲劳过程、运动学/力和高度-负载图 | 独立功能；只适用于带疲劳配置的数据集 |
| Height–interaction force | 以高度为横轴比较疲劳组的交互力曲线 | 疲劳分析的另一种展示；与上者互相独立 |
| Insole analysis | 将左右鞋垫 GRF 注入标准对齐结果并检查对齐/动作切片 | 依赖 config 中的 `insole_time_offset` |
| Insole offset calibration | 估计鞋垫与机器人时间偏移，并可写回 JSON | 应在鞋垫分析前运行；会修改配置文件（`WRITE_JSON=True`） |
| Insole map check | 校验鞋垫压力图与力文件的文件、时间轴、相关性、增益、残差、饱和和 COP | 校准后的独立质量检查；可与标准分析并行 |
| MVC computation | 从 EMG 文件组计算每块肌肉 MVC，生成诊断图并写出 `*_mvc.json` | 若没有可用 MVC，应先运行；输入配置不会原地覆盖 |
| EMG RMS | 绘制固定负载和变负载的 EMG RMS 随时间/高度/负载变化 | 依赖 `run()` + `run_vload()`；与频域脚本平行 |
| EMG frequency | 绘制固定负载 EMG 频域特征（MDF） | 与 RMS 平行的另一种特征分析 |
| EMG RMS + vload | 明确把固定负载与变负载的 RMS 放在同一组图中比较 | RMS 的变负载专用实现 |
| EMG frequency + vload | 明确把固定负载与变负载的频域特征放在同一组图中比较 | Frequency 的变负载专用实现 |
| Heatmap fitting | 从切片数据拟合 RBF 与 monotone P-spline 的高度–负载–肌肉激活曲面，并比较预测 | 变负载建模的主要前置步骤 |
| Estimated-load heatmap | 用机器人力/加速度估计负载后拟合热图，并保存 estimated-load 参数 | 与普通 heatmap 替代；下游 `vload_result_est_load` 需要它 |
| Load–activation curves | 按高度和负载绘制激活/功率关系曲线及散点 | 热图的独立曲线展示 |
| Log power–log activation (2 segments) | 在上升/下降分段下绘制 log–log 功率–激活关系 | 与 load-activation 曲线平行的另一种拟合展示 |
| Isometric log power–activation | 专门处理等长组，绘制力/激活、时间/激活等图 | 独立于等速/等负载热图 |
| RBF/P-spline vload comparison | 将变负载实测结果与 RBF、P-spline 预测叠加比较 | 需要先有热图参数和变负载数据 |
| Variable-load optimization | 使用热图参数求解目标肌肉激活对应的变负载方案 | 通常在 `example_heatmap.py` 之后；两种拟合实现可选 |
| Realtime squat metronome | 读取实时配置、回放数据，以音频节拍驱动 OpenSim 深蹲可视化 | 独立实时应用；运行 `config/sq_config.json` |
| Realtime bench-press metronome | 与深蹲版本同一框架，用于卧推回放和节拍可视化 | 与深蹲版本互斥选择 |
| OpenSim scaling | 从 Xsens 身体测量缩放全身模型并计算接触点 | OpenSim 分支的模型准备步骤 |
| Xsens → MOT | 把 Xsens Excel 转为 `.mot`，检测/记录丢帧并检查坐标范围 | OpenSim Step 1；应先于 MOT 校验和后续分析 |
| MOT validation | 检查 MOT 的时间、幅值、连续性、绕接、丢帧、跨负载一致性和左右对称 | 每次重生成 MOT 后、进入 ID 前运行 |
| OpenSim symmetry check | 用标准 pipeline 检查动作及左右运动对称性并保存图 | 独立质量检查；可与 MOT validation 并行 |
| OpenSim full pipeline | 编排 Xsens→MOT、MuscleAnalysis、InverseDynamics 三步 | 顺序入口；当前脚本默认实际执行 Step 2，Step 1/3 为注释调用，需按需打开 |
| Inverse dynamics | 生成外力并运行 OpenSim 逆动力学，输出关节力矩 | 需要缩放模型、可用 MOT 和外力输入 |
| External-force inspection | 检查/绘制机器人杆力与鞋垫 GRF 外力文件 | ID 前的外力诊断 |
| Shear reconstruction | 从 Xsens 质心/姿态与鞋垫数据重建校正后的外力，再可选重跑 ID | ID 的高级校正分支；替代默认外力生成 |
| ID plotting / force balance | 读取切片、GRF、机器人力和 ID 力矩，绘制按负载的力矩及力平衡图 | 需要已有 ID 结果；可选择自动重跑 |
| COP sensitivity | 扫描鞋垫 COP 位置，量化其对 ID 力矩和负载单调性的影响 | ID 后的敏感性实验；不覆盖正式 ID 结果，但会重写共享外力中间文件 |
| Shear/ID plotting helpers | 将 MuscleAnalysis 肌肉力臂与关节角对齐并拟合 | 需要 Step 2 的 MuscleAnalysis 输出 |
| Muscle moment contribution | 将肌肉力矩贡献堆叠，并与 ID 净力矩比较 | 同时需要 Step 2 和 Step 3 输出 |
| EMG force ratio | 比较固定/变负载下单块肌肉的力占比与 EMG 激活占比 | 依赖标准和 vload 结果；独立指标分析 |
| Fixed vs vload scatter | 按高度比较固定负载与变负载的力/运动散点 | 依赖 `run()` + `run_vload()` |
| Vload comparison (no Xsens) | 在没有 Xsens 时比较固定/变负载机器人运动学和 EMG 特征 | 与 Xsens 版本二选一 |
| Vload comparison (Xsens) | 在有 Xsens 时额外比较关节角、角速度和 Xsens 运动学 | 与 no-Xsens 版本二选一 |
| Vload result | 将变负载实测高度/负载/EMG 与规划文件及热图预测比较 | 需要 `vload_file`、实测 variable-load 数据和热图参数 |
| Vload result + estimated load | 在上例基础上加入 estimated-load 热图预测 | 先运行 `example_heatmap_estimated_load.py` |

## How to Run

脚本大多通过相对路径读取 config，建议进入脚本所在目录后运行，例如：

```text
cd examples/data_analysis
python example_data_analysis.py
```

统一离线流程从项目根目录运行：

```text
python examples/unified_example.py
```

OpenSim 逆动力学目录中的脚本还会互相导入工具函数，因此应在该目录运行；实时脚本应在 `examples/metronome` 运行。运行前通常只需修改脚本顶部的 `CONFIG_FILE`、目标肌肉/负载筛选和绘图开关。`example_insole_sync_offset.py` 的 `WRITE_JSON=True` 会写回配置；MVC 脚本则默认新建 `*_mvc.json`。

## Recommended Workflow

项目没有覆盖所有功能的一条必经顺序。按目标选择下列分支：

```text
原始配置/数据
├─ 鞋垫同步：example_insole_sync_offset
│  ├─ example_insole_map_check（可选质量检查）
│  └─ example_data_analysis_insoles / example_data_analysis
├─ EMG 标定：example_compute_mvc（仅无可靠 MVC 时）
│  └─ data_analysis / emg_analysis
├─ 固定/变负载建模：标准 run → example_heatmap
│  ├─ example_variable_load（可选：求变负载方案）
│  ├─ example_vload_result / example_rbf_and_pspline（可选：验证实测 vload）
│  └─ example_heatmap_estimated_load → example_vload_result_est_load（可选替代预测）
└─ OpenSim：example_scaling → example_xsens_to_mot → example_validate_mot
   ├─ example_symmetry_check（可选质量检查）
   ├─ example_opensim_pipeline / Step 2 MuscleAnalysis
   └─ 外力检查 → Step 3 InverseDynamics → 绘图/力臂/肌肉贡献/COP sensitivity
```

`example_vload_comparison.py` 与 `example_vload_comparison_xsens.py` 是同一目的的两种实现：按数据是否含 Xsens 二选一。`example_emg_rms*.py` 和 `example_emg_frequency*.py` 是时域幅值与频域特征的平行方案，不要求先后。普通 heatmap 与 estimated-load heatmap 也是两条替代建模路径；只有 estimated-load 结果链要求先生成 estimated-load 参数。

## Example Details

上述总表已经按目录给出每个可运行 `.py` 的位置、主要功能和关系。快速定位时：

- 想看“数据是否对齐/切片是否正确”：`data_analysis/`。
- 想处理 EMG 特征或重新计算 MVC：`emg_analysis/`。
- 想建立高度–负载–激活模型：`heatmap/`。
- 想比较固定负载与变负载：`vload_analysis/`，以及带 Xsens 的 `data_analysis/example_load_xsens_vload.py`。
- 想做 OpenSim 运动学、肌肉分析或 ID：`opensim/`。
- 想做实时声音节拍和模型回放：`metronome/`。

## Unified Workflow

`unified_example.py` 已提供推荐离线主线，各步骤保留为独立函数，便于按实验需要启用：

1. 读取明确的 config，标定鞋垫时间偏移，并按需重新计算 MVC。
2. 运行 `MultiLoadPipeline`，得到结构化的固定负载对齐/切片结果；对齐验证会根据 config 自动加入鞋垫、`emg_RGL` 和右膝 Xsens 角度代表线。
3. 生成普通 activation surface（RBF + P-spline）并按需求解变负载；estimated-load surface/优化作为注释的替代分支。
4. 如果 config 含变负载实测数据，可运行 `run_vload` 和相应结果验证。
5. OpenSim 分支按 scaling → Xsens-to-MOT → MuscleAnalysis / InverseDynamics 执行，其他诊断作为 optional 后处理。

不要把“有 Xsens”和“无 Xsens”的 vload comparison 同时加入；两者是互斥替代方案。普通 heatmap 与 estimated-load heatmap也不必同时作为主路径，但可以作为对照分别运行。实时 metronome 不应纳入离线 unified workflow，它是独立应用。
