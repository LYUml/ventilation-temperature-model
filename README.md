# Baseline corridor temperature models

极简、可复现的走廊未来24小时温度模型。最终工作树只使用真实温度、真实54399气象数据和真实建筑RDF。

## 文件结构

```text
ventilation-temperature-model/
├── README.md
├── base_ta.py                     对外统一导入入口
├── config.json                    唯一配置入口
├── requirements.txt               最小依赖
├── data/                           只读真实输入
│   ├── building.rdf
│   ├── temperature.csv
│   └── weather_54399.csv
├── model/
│   ├── common/
│   │   ├── data.py                数据QA、切分、指标
│   │   └── rdf.py                 RDF几何与共享界面提取
│   ├── base_ta.py                 两种模型的训练、预测与持久化
│   ├── kernel/
│   │   ├── README.md
│   │   └── run.py                 稳定核基线
│   ├── rc_narx_ridge/
│   │   ├── README.md
│   │   └── run.py                 RC轨迹＋正则残差校准
│   ├── rdf_rc_narx_ridge/
│   │   ├── README.md
│   │   └── run.py                 RC＋RDF历史特征校准
│   ├── rdf_kernel_narx_ridge/
│   │   ├── README.md
│   │   ├── ABLATION.md
│   │   └── run.py                 Kernel＋RDF残差递推
│   ├── rc_narx_ridge_optimization/
│   │   └── run.py                 RC结构与NARX-Ridge组合实验
│   └── compare.py                 统一比较入口
├── docs/
│   ├── MODEL_AUDIT.md             数据、参数与幻觉审计
│   └── manuscript.md              研究文稿
├── tests/                          数据、RDF、模型回归测试
└── results/                        可再生成，Git忽略
```

## 可复用预测接口

`base_ta.py`将最终的两个时序室温模型暴露为同一个接口。历史气象、历史室温和未来气象保持为三个独立的`pandas.DataFrame`；时间必须连续且按小时对齐。室温列名使用RDF中的空间名称。

```python
from base_ta import calculateBaseTa

temperature = calculateBaseTa(
    inputRdf="data/building.rdf",
    weatherHistory=historical_weather,
    indoorHistory=historical_indoor_temperature,
    forecastWeather=next_24h_weather,
    targetSpaces=["2FCORRIDOR", "3FCORRIDOR", "4FCORRIDOR"],
    method="rdf_kernel_narx_ridge",
)
```

`method="rdf_rc_narx_ridge"`选择RC灰箱混合模型；`method="rdf_kernel_narx_ridge"`选择默认的高性能数据驱动模型。返回值是`{空间名: 温度列表}`。

重复预测时应只训练一次，并保存模型：

```python
from base_ta import BaseTaModel

model = BaseTaModel(method="rdf_rc_narx_ridge")
model.fit("data/building.rdf", historical_weather, historical_indoor_temperature)
model.save("base_ta.pkl")

model = BaseTaModel.load("base_ta.pkl")  # 只加载可信来源的模型文件
temperature = model.predict(next_24h_weather)
```

气象表默认自动识别`Ta`、`TEM`或`OUTDOOR`为室外温度，并使用历史与预测表中全部同名数值气象列。时间可来自`Timestamp`、`DatetimeIndex`或`year/mon/day/hour`。若不传`targetSpaces`，接口返回RDF与室温历史中名称匹配的全部空间。新建筑必须提供自身的历史室温进行校准，并使用与当前解析器兼容的MOOSAS Turtle RDF模式。

可复用接口是面向真实预测的部署形式：未来邻室温度未知时，模型只使用历史邻室温度，并将所有有历史测量的RDF空间作为联合状态。原有`run.py`中的锁定RMSE属于旧评估流程，不能直接视为任意新建筑上的部署精度；新建筑应保留独立时段重新验证。

## 方法

### 1. Kernel baseline

仅使用室外温度慢状态。平均RMSE `0.344 °C`，作为锁定基线。

### 2. RDF graph state

RDF共享构件生成2F↔3F↔4F拓扑，RDF界面UA生成邻室边界。平均RMSE `0.397 °C`，用于证明“加入RDF”本身不会自动改善预测。

### 3. RDF-Kernel NARX-Ridge

Kernel生成慢趋势，RDF选择邻室边界和相连楼层，Ridge递推预测残差。平均RMSE `0.319 °C`，相对Kernel提升约`7.3%`。

### 4. RC NARX-Ridge

训练集辨识的等效RC轨迹加Ridge直接多步校准。当前平均RMSE `0.350 °C`；保留为不读取房间温度的独立走廊基线。

### 5. RDF-RC NARX-Ridge

一阶RC轨迹加历史RC误差与RDF选取的邻室/相邻楼层历史温度，由共享Ridge校准未来24小时。当前测试RMSE约`0.321 °C`，作为待外部验证的增强候选。

### 6. RC NARX-Ridge组合优化实验

比较1R1C/2R2C、一步/24小时参数标定、共享/分楼层Ridge，以及历史残差/RDF历史特征。完整结果位于`results/rc_narx_ridge_optimization/metrics.json`；当前仅作为消融实验，不根据测试集反向选型。

这里的`NARX-Ridge`表示使用历史状态和外部输入的Ridge校准结构，不等同于严格的Polynomial NARX；后者旧实验RMSE约`0.414 °C`。

## 运行

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m model.compare
```

生成结果写入`results/`。参数与删除的假设见[模型审计](docs/MODEL_AUDIT.md)。
