# 01 稳定基线

## Kernel因果滑动平均

- 用途：当前最干净、输入最少的交付基线。
- 输入：室外温度。
- 平均RMSE：0.344 °C。
- 分层RMSE：2F 0.388、3F 0.256、4F 0.387 °C。
- 源码：`src/temperature_model.py`、`src/validate_24h_baseline.py`。
- 输出：`outputs/baseline_24h_validation/`。

```powershell
python -m src.validate_24h_baseline --config configs/temperature_model.json
```

`temperature_model.py` 中的早期70%/30%结果仅作历史参考。
