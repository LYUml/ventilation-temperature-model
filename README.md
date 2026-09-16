# Ventilation temperature model

使用建筑 RDF 和逐小时气象数据模拟各空间的室内温度。

## Tree

```text
.
├── rdf_weather_temperature.py   RDF + weather 公共接口
├── base_ta.py                   运行阶段预测模型入口
├── model/
│   ├── mz5r1c.py                RDF 多空间 5R1C 热模型
│   ├── base_ta.py               历史数据预测模型
│   └── common/                  RDF 与数据解析
├── data/
│   ├── building.rdf
│   └── weather_54399.csv
├── tests/
└── requirements.txt
```

## 用法

默认计算 RDF 中全部可用空间：

```python
from rdf_weather_temperature import simulateIndoorTemperature

temperature = simulateIndoorTemperature(
    "data/building.rdf",
    "data/weather_54399.csv",
)
```

只计算一个空间：

```python
temperature = simulateIndoorTemperature(
    "data/building.rdf",
    "data/weather_54399.csv",
    targetSpace="4F412",
)
```

返回格式：

```python
{
    "4F412": [23.1, 23.0, 22.8],
    "2FCORRIDOR": [22.5, 22.4, 22.3],
}
```

标准气象 CSV 列：

```text
timestamp,outdoor_temperature_c,diffuse_solar_w_m2,direct_solar_w_m2
```

现有 `weather_54399.csv` 的 `year/mon/day/hour`、`TEM`、`diffuse`、`direct` 也可直接读取。

## 测试

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
