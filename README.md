# Infiltration Modeling

基于 **ELA（等效漏风面积）**、**CONTAM** 和 **RC（热阻–热容）模型**的建筑渗风与楼层走廊温度建模原型。

## 文件结构

```text
infiltration/
├── BiCEM/                   BiCEM优化程序
├── data/                    实测温度数据
├── doc/                     技术路线与历史资料
│   ├── attempt/             前期尝试
│   ├── byxj/                师兄提供的资料
│   └── route/               当前技术路径
├── ela-contam-prototype/    ELA–CONTAM与温度模型原型
│   ├── configs/             JSON输入参数
│   ├── outputs/             模型运行结果
│   ├── src/                 Python程序
│   ├── templates/           CONTAM项目模板
│   └── tests/               自动测试
├── moosas/                  MoosasPy源码
├── manuscript.md            研究文稿
└── README.md
```

## 快速运行

运行三楼层假设案例：

```bash
cd ela-contam-prototype
python -m src.run_three_floor_case --config configs/three_floor_smoke_test.json
```

运行全部测试：

```bash
python -m unittest discover -s tests -v
```

运行 RC 楼层温度模型：

```bash
python -m src.temperature_rc_model --config configs/temperature_rc_model.json
```

详细方法见 [`doc/route/0824-技术路径.md`](doc/route/0824-技术路径.md)。
