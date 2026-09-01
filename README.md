# Baseline corridor temperature models

极简、可复现的走廊未来24小时温度模型。最终工作树只使用真实温度、真实54399气象数据和真实建筑RDF。

## 文件结构

```text
ventilation-temperature-model/
├── README.md
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
