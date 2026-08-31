# 04 混合与融合方法

| 方法 | 平均RMSE/°C | 定位 |
|---|---:|---|
| Adaptive Kernel–RC–NARX | 0.279 | 旧融合性能上界 |
| Hierarchical RC–NARX–Ridge | 0.350 | 未超过Kernel |
| KL-TIF | **0.270** | 当前数值最优 |

KL-TIF暂称 `Kernel Latent Thermal Innovation Fusion`，由慢趋势、边界ARX、多楼层热创新状态和受限N4SID修正组成。它是有热过程分工的受约束ensemble，不是单一物理方程。

源码：

- `src/benchmark_hybrid_methods.py`
- `src/benchmark_open_innovation.py`

输出：

- `outputs/hybrid_method_benchmark/`
- `outputs/open_innovation_benchmark/`

```powershell
python -m src.benchmark_hybrid_methods --config configs/temperature_model.json
python -m src.benchmark_open_innovation --config configs/temperature_model.json
```
