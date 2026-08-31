# 方法目录

本目录按研究方法组织导航。实际可执行源码仍保留在 `src/`，以兼容既有命令和测试。

| 分类 | 当前定位 | 代表结果（平均24 h RMSE） |
|---|---|---:|
| `01_stable_baseline` | 稳定交付基线 | Kernel 0.344 °C |
| `02_physical_greybox` | RC、热容量、RDF灰箱 | 最佳约0.431 °C |
| `03_data_driven` | Ridge、NARX、MLP、N4SID | 最佳ARX 0.318 °C |
| `04_hybrid_fusion` | 受约束融合与性能上界 | KL-TIF 0.270 °C |
| `05_rdf_msts` | 单一RDF约束状态方程 | RDF-MSTS 0.401 °C |

所有结果使用前60%训练、中间20%验证、后20%锁定测试；每层62个重叠24小时窗口。早期70%/30%实验不参与这张排名。
