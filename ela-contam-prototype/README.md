# ELA-CONTAM 最小原型

这个目录是独立实验区，不会修改 `archive-*` 中师兄提供的文件。

## 名词

- **ELA（Effective Leakage Area，有效漏风面积）**：把许多细小缝隙等效成一个孔洞后的面积。
- **CONTAM**：美国国家标准与技术研究院（NIST）开发的建筑空气流动计算软件。
- **ContamX**：CONTAM 的命令行计算程序，由 Python 自动调用。
- **烟囱效应**：室内外温差造成空气在低处进入、高处排出的现象。
- **ACH（Air Changes per Hour，每小时换气次数）**：一小时进入房间的室外空气量相当于几个房间体积。
- **RMSE（均方根误差）**和 **MAE（平均绝对误差）**：温度预测误差，单位为摄氏度，越小越好。

## 运行最小漏风案例

在本目录执行：

```powershell
python -m src.run_case --config configs/smoke_test.json
```

结果写入 `outputs/smoke_test/`：

- `generated.prj`：Python 生成的 CONTAM 项目文件；
- `generated.sim`：ContamX 的二进制结果；
- `generated.sqlite3`：Python 可直接读取的结果数据库；
- `path_flows.csv`：高、低漏风路线的压力与流量；
- `run_summary.json`：质量守恒、换气次数和运行状态；
- `contam_stdout.log`、`contam_xlog.log`：完整计算日志。

## 运行自动测试

```powershell
python -m unittest discover -s tests -v
```

测试包括质量守恒、等温零流量、近似零 ELA、ELA 加倍以及重复运行一致性。

## 运行楼层基准温度模型

```powershell
python -m src.temperature_model --config configs/temperature_model.json
```

第二版物理热惯性模型：

```powershell
python -m src.temperature_rc_model --config configs/temperature_rc_model.json
```

RC 是“热阻–热容”模型。结果同时给出使用上一小时实测温度的一步预测，以及不再读取后续室温的连续预测，避免只看一个偏乐观的误差。

三楼层假设尺寸流程测试：

```powershell
python -m src.run_three_floor_case --config configs/three_floor_smoke_test.json
```

该案例的尺寸和 ELA 全部是假设值，只用于确认三层模型自动生成、求解、读取和守恒检查能够完整运行。

程序只读取 `../data/TEMPERATURE-rev.csv`，以前 70% 时间数据建立模型，以后 30% 时间数据检查模型。结果写入 `outputs/temperature_model/`。

## 重要限制

最小案例使用的 ELA 是文献参考值，只用于验证程序，不代表实际办公楼。真实建筑需要气密性测试数据，或者明确标记为低、中、高漏风情景。
