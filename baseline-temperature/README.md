# 可运行原型

这里仅包含基线温度建模的 Python 实现。命令均从本目录运行。

## 源码索引

### 基础模块

- `rdf_building.py`：从 RDF 提取空间、面积、U 值和 UA。
- `temperature_model.py`：核滑动平均基线及通用数据检查、指标函数。
- `temperature_rc_model.py`：早期 RC 温度模型。

### 当前验证与模型比较

- `validate_24h_baseline.py`：无未来室温泄漏的 24 小时滚动基线。
- `benchmark_baseline_models.py`：传统基线模型比较。
- `benchmark_hybrid_methods.py`：RC、NARX及混合方法比较。
- `validate_mlp_nsga.py`：MLP/多目标搜索实验。
- `validate_ceff_state_space.py`：固定 Ceff 状态空间实验。
- `validate_rdf_2r2c.py`：RDF 约束的统一 2R2C 模型及参数贴边诊断。

## 推荐命令

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.validate_24h_baseline --config configs/temperature_model.json
python -m src.validate_rdf_2r2c --config configs/temperature_model.json
```

结果自动写入 `outputs/`。该目录属于可再生成的本地文件，不提交到 Git。

## 重要限制

- 当前温度数据只有 421 小时，且集中在一个春季时段。
- RDF 的可开启窗属性不等于测量期间真实开启状态。
- 暂无风、太阳辐射、HVAC 功率和人员数据。
- `165 kJ/(m²·K)` 是中等热质量类别默认值，不是该建筑实测热容量。
- 参数达到上下界时只表示不可辨识或模型缺项，不能作为建筑物理结论。
