# RC NARX-Ridge

训练集辨识的一阶等效 RC 生成24小时轨迹，Ridge 对其残差进行 NARX 风格的直接多步校准。
RC 系数是等效辨识参数，不冒充实测热阻或热容量；模型会读取预测窗开始前一小时的走廊温度。

```powershell
python -m model.rc_narx_ridge.run
```
