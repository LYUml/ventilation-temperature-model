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

模型默认采用严格项目数据模式。当前项目仍缺少真实材料热容量、已确认的太阳辐射语义以及完整 HVAC/空气交换边界，因此严格模式会直接报错，不会静默使用默认值。

显式设置 `allow_non_project_assumptions=True` 仅用于研究复现。接入 RDF 真实逐时日程和构件级 SHGC 后，该模式在锁定案例上的三层 RMSE 为 `0.561 / 0.591 / 1.062 °C`，平均 `0.738 °C`。这不是全项目数据合规成绩，不能与需要目标建筑历史室温的 Kernel 直接比较。

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
