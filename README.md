# Corridor Baseline Temperature Model

建筑走廊未来 24 小时基线温度建模仓库。范围只包括基线温度：从 RDF 提取建筑物理信息，利用逐小时室外、走廊和房间温度建立并验证预测模型。TPE、ELA、CONTAM 和后续参数标定由师兄负责，不在本仓库维护范围内。

## 目录

```text
ventilation-temperature-model/
├── data/                         原始研究数据
│   ├── NBuilding.rdf             建筑空间、界面、面积和 U 值
│   └── TEMPERATURE-rev.csv       421 小时室外/走廊/房间温度
├── baseline-temperature/         可运行的基线温度研究代码
│   ├── configs/                  可复现实验配置
│   ├── src/                      温度模型与验证
│   ├── tests/                    自动测试
│   └── outputs/                  本地生成结果（Git 忽略）
├── doc/route4recap/              方法路线与跨设备交接记录
├── manuscript.md                 研究文稿
└── README.md
```

## 数据现状

`TEMPERATURE-rev.csv` 覆盖 2026-05-09 00:00 至 2026-05-26 12:00，共 421 个连续小时，无缺失值。字段包括：

- 室外温度：`OUTDOOR`
- 走廊温度：`2FCORRIDOR`、`3FCORRIDOR`、`4FCORRIDOR`
- 房间温度：`2F215`、`3F308`–`3F310`、`4F408`–`4F411`、`5F510`

RDF 提供几何、空间邻接、围护结构面积、U 值、窗 SHGC 和表面朝向。它作为温度模型的建筑信息输入，不在本仓库中执行 CONTAM。

## 快速开始

```powershell
cd baseline-temperature
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

运行当前物理约束模型：

```powershell
python -m src.validate_rdf_2r2c --config configs/temperature_model.json
```

运行核滑动平均基线：

```powershell
python -m src.validate_24h_baseline --config configs/temperature_model.json
```

运行候选模型比较：

```powershell
python -m src.benchmark_baseline_models --config configs/temperature_model.json
python -m src.benchmark_hybrid_methods --config configs/temperature_model.json
```

所有运行结果写入 `baseline-temperature/outputs/`。该目录不提交到 Git，可由上述命令重新生成。

## 当前结论

- 核滑动平均的锁定测试集平均 24 小时 RMSE 约为 `0.344 °C`。
- RDF 约束 2R2C 当前约为 `0.541 °C`，尚未超过基线。
- RC 参数出现贴边，说明当前数据不足以独立辨识所有热阻、热容量和未知热增益。
- 下一步应先利用已有房间温度建立房间–走廊热耦合，再考虑增加太阳辐射、HVAC、风和门窗状态数据。

## 数据与复现约定

- `data/` 是原始输入，不由模型脚本覆盖。
- `configs/` 保存可提交、可复现的实验条件。
- `outputs/` 只保存派生结果，不进入版本控制。
- 新模型必须按时间划分训练、验证、测试集，并与相同测试窗口下的核基线比较。
- 标准默认值和人为边界必须在结果中标记，不得写成实测或已标定参数。
