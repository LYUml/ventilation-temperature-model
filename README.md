# Corridor baseline-temperature models

本仓库区分两个不同的应用阶段：

- **设计阶段**：`RdfMz5r1cModel` 从 RDF 和气象构建多区 ISO 13790 5R1C 热模型，不接受目标建筑历史室温。
- **运行阶段**：`BaseTaModel` 使用已建建筑历史室温进行未来 24 小时预测，仅作为 forecast 对照。

## 目录

```text
├── base_ta.py                 公共导入入口
├── config.json                数据与目标空间配置
├── data/                      项目 RDF、气象和案例温度
├── model/
│   ├── mz5r1c.py              设计阶段多区 5R1C
│   ├── base_ta.py             运行阶段统一接口
│   ├── common/                数据 QA 与 RDF 解析
│   ├── kernel/                forecast 基线
│   ├── rc_narx_ridge/         forecast RC 模型
│   ├── rc_narx_ridge_optimization/ forecast 模型复用的 RC 工具与结构验证
│   ├── rdf_kernel_narx_ridge/ forecast RDF-Kernel 模型
│   └── rdf_rc_narx_ridge/     forecast RDF-RC 模型
├── docs/
│   ├── INPUT_PROVENANCE_AUDIT.md
│   ├── MODEL_AUDIT.md
│   └── manuscript.md
├── tests/                     回归测试
└── archive/                   已淘汰或非生产实验
```

## 设计阶段模型

```python
from base_ta import RdfMz5r1cModel

model = RdfMz5r1cModel()
temperature = model.simulate(
    "data/building.rdf",
    weather,
    ["2FCORRIDOR", "3FCORRIDOR", "4FCORRIDOR"],
)
```

设计阶段验证默认保留三层走廊并新增 `4F412`。`4F412` 作为显式基准空间时，
其人员、照明、设备得热和理想 HVAC 被关闭；实测温度只在仿真完成后用于评价，
不会进入模型：

```bash
python -m model.mz5r1c_validation
```

当前案例温度文件已按确认后的房间编号使用 `4F412`。锁定测试段上，
新增 412 零内热场景的 RMSE 为 `1.325 °C`；连同三层走廊的四目标平均
RMSE 为 `0.976 °C`。这些实测值只用于仿真后的评价。

## 412 敏感性分析

```bash
python -m model.design_sensitivity
```

该实验只用验证段判断敏感方向，锁定测试段不参与选择。单因素结果显示，
412 对热容量最敏感：采用当前热容量的 `2× / 4×` 情景时，锁定测试 RMSE
分别为 `0.771 / 0.512 °C`；这些数值是缺少真实材料层时的不确定性情景，
不能作为拟合后的正式项目参数。渗透率、外界 UA 和邻室 UA 的扫描没有产生
同等幅度的稳定改善。

已验证但收益有限的 5R2C 空气电容实验已移入 `archive/models/`。主模型继续
使用 MZ5R1C；下一步优先获取材料层并计算项目真实有效热容量。

模型默认采用严格项目数据模式。当前项目仍缺少真实材料热容量、已确认的太阳辐射语义以及完整 HVAC/空气交换边界，因此严格模式会直接报错，不会静默使用默认值。

显式设置 `allow_non_project_assumptions=True` 仅用于研究复现。此前按 RDF 原始日程运行时，锁定案例的三层 RMSE 为 `0.561 / 0.591 / 1.062 °C`，平均 `0.738 °C`。新增 `4F412` 无热扰基准场景后，412 的非零办公室设备日程会被清零，并通过多区耦合影响走廊，因此新场景必须独立报告，不能沿用旧成绩。这些结果均不是全项目数据合规成绩，也不能与需要目标建筑历史室温的 Kernel 直接比较。

输入来源、剩余缺口和 SHGC 粒度规则见 [输入来源审计](docs/INPUT_PROVENANCE_AUDIT.md)。阶段性实验及作废成绩位于 [archive](archive/README.md)。

## 运行阶段 forecast

```python
from base_ta import BaseTaModel

model = BaseTaModel(method="rdf_kernel_narx_ridge")
model.fit(
    "data/building.rdf",
    historical_weather,
    historical_indoor_temperature,
    ["2FCORRIDOR", "3FCORRIDOR", "4FCORRIDOR"],
)
temperature = model.predict(next_24h_weather)
```

该接口明确要求目标建筑历史室温，不适用于未建成建筑。

## 验证

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m model.compare
```

`results/` 为可再生成输出并由 Git 忽略。
