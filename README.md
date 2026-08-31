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
│   ├── rdf_msts/
│   │   ├── README.md
│   │   ├── ABLATION.md
│   │   └── run.py                 原创统一RDF-MSTS
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

### 3. RDF-MSTS（主方法）

RDF-constrained Multi-scale Spatial Thermal State model。Kernel慢趋势是统一状态方程内部的慢流形，图约束VARX只递推偏离慢流形的创新状态；没有多个完整模型的输出加权。平均RMSE `0.319 °C`，相对Kernel提升约`7.3%`。

## 运行

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m model.compare
```

生成结果写入`results/`。参数与删除的假设见[模型审计](docs/MODEL_AUDIT.md)。
