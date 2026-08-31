# 05 RDF-MSTS统一状态模型

## RDF-MSTS

`RDF-constrained Multi-scale Spatial Thermal State Model` 使用一个41维伴随式状态方程同时预测三层走廊：

```text
x(t+1) = A x(t) + B u(t+1)
T_corridor(t) = C x(t)
```

状态包含三层过去12小时温度、三层慢热状态、共享楼体状态和室外慢状态。输入包含邻室温度、室外温度、pvlib四向立面辐射、风、湿度和日周期。

- 平均RMSE：0.401 °C。
- 分层RMSE：2F 0.425、3F 0.186、4F 0.593 °C。
- 相对Kernel：下降16.6%，当前未超过基线。
- 谱半径：0.995。
- 源码：`src/validate_rdf_msts.py`。
- 输出：`outputs/rdf_msts_validation/`。

```powershell
python -m src.validate_rdf_msts --config configs/temperature_model.json
```

RDF现已实际用于生成房间—走廊边界、2F↔3F↔4F竖向拓扑，并提供体积、围护UA和窗面积缩放。RDF确认的4F实测边界为`4F408/409/410`与上方`5F510`，不包含此前误写的`4F411`。慢热状态仍是辨识状态，不是实测墙体温度。
