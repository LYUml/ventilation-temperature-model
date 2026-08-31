# 03 数据驱动方法

| 方法 | 平均RMSE/°C | 判断 |
|---|---:|---|
| 多输入ARX＋邻室＋朝向辐射 | 0.318 | 本类最佳，超过Kernel |
| NARX-Ridge | 0.355 | 接近Kernel |
| NARX-ExtraTrees | 0.395 | 4F较差 |
| MIMO-N4SID | 0.380 | 单独使用不够稳定 |
| SysIdentPy polynomial NARX | 0.414 | 未超过Kernel |
| 气象NARX-Ridge | 0.425 | 气象维度相对样本过高 |
| Direct MLP | 0.668 | 小数据过拟合 |
| MLP＋NSGA-II | 约0.786 | 不建议继续扩大搜索 |
| NARX-MLP | 3.814 | 严重不稳定 |

源码：

- `src/benchmark_baseline_models.py`
- `src/validate_mlp_nsga.py`
- `src/benchmark_hybrid_methods.py`（SysIdentPy）
- `src/benchmark_open_innovation.py`（ARX、N4SID）

输出：

- `outputs/baseline_model_benchmark/`
- `outputs/mlp_nsga_validation/`
- `outputs/open_innovation_benchmark/`

数据驱动方法用于方法比较和融合构件，不作为统一物理参数解释。
