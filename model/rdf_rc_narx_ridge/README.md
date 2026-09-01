# RDF-RC NARX-Ridge

一阶RC生成24小时基础轨迹；共享Ridge使用历史RC误差、RDF选取的邻室边界和相邻楼层历史温度校准24小时轨迹。

该名称表示NARX-Ridge风格的直接多步模型，不等同于严格的Polynomial NARX。

```powershell
python -m model.rdf_rc_narx_ridge.run
```
