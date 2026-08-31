# Ablation

同一数据切分下：

| 方法 | RDF | 慢状态 | 平均24h RMSE |
|---|---|---|---:|
| Kernel | 否 | 室外EWMA | 0.344 °C |
| RDF graph state | 是 | 否 | 0.397 °C |
| RDF-MSTS | 是 | 是 | 0.319 °C |

这说明RDF拓扑本身不会自动提高精度；慢状态与图约束的结合才有效。
