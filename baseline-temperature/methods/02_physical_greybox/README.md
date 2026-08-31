# 02 物理与灰箱方法

| 方法 | 平均RMSE/°C | 说明 |
|---|---:|---|
| 早期1R1C/邻室RC | 约0.47–0.76 | 连续rollout；证明邻室温度有用 |
| 简化2R2C | 0.783 | 结构过简，4F误差大 |
| 固定Ceff状态空间 | 0.551 | 只能识别H/C，参数不可辨识 |
| RDF约束2R2C | 0.541 | 多项参数贴边 |
| 气象＋邻室＋RDF-2R2C | 0.431 | 物理模型中当前最好 |
| DarkGreyBox风格结构搜索 | 0.742 | 1R1C/2R2C验证集选型 |

源码：

- `src/temperature_rc_model.py`
- `src/validate_ceff_state_space.py`
- `src/validate_rdf_2r2c.py`
- `src/rdf_building.py`

主要输出：

- `outputs/ceff_state_space_validation/`
- `outputs/rdf_2r2c_validation/`

```powershell
python -m src.validate_ceff_state_space --config configs/temperature_model.json
python -m src.validate_rdf_2r2c --config configs/temperature_model.json
```

注意：当前模型虽读取RDF，但有效UA、热容量和渗透参数仍不能解释成已识别的真实物性。
