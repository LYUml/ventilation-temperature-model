# CONTAM–MATLAB 耦合算法的渗风模型扩展方案

## 1. 扩展目标与约束

本方案在现有 CONTAM–MATLAB 耦合算法上增加围护结构渗风计算，保持原有计算框架不变：

1. CONTAM 负责多区域压力与空气流量计算；
2. MATLAB 根据 CONTAM 输出的流量矩阵求解房间热平衡；
3. 更新后的室温回传 CONTAM；
4. 重复计算直至温度满足原有收敛判据。

渗风扩展不修改原有房间节点、门窗路径、流量矩阵结构和热平衡求解逻辑。新增内容限于：

- 在 CONTAM 中增加围护结构泄漏路径；
- 为泄漏路径增加必要的物理参数；
- 在结果解析时识别渗风流量；
- 在原有热平衡中计入渗风路径携带的显热；
- 增加渗风量、渗风 ACH 和渗风显热输出。

## 2. 推荐建模路线

建议采用 CONTAM 原生的压力网络渗风建模方法，而非在 MATLAB 端附加固定渗风换气次数。

对连接室外节点与房间 $i$ 的围护结构泄漏路径 $e$，采用压差幂律：

$$
Q_e=C_e\operatorname{sgn}(\Delta P_e)
\left|\Delta P_e\right|^{n_e},
\tag{1}
$$

式中：

- $Q_e$：路径有向体积流量，$\mathrm{m^3/s}$；
- $C_e$：体积流量系数，$\mathrm{m^3/(s\cdot Pa^{n_e})}$；
- $n_e$：流动指数；
- $\Delta P_e$：路径两端总压差，$\mathrm{Pa}$。

路径压差仍由 CONTAM 根据区域压力、风压和热压统一计算，可概念性表示为：

$$
\Delta P_e
=P_{\mathrm{out},e}-P_{i,e}
+\Delta P_{\mathrm{wind},e}
+\Delta P_{\mathrm{stack},e}.
\tag{2}
$$

MATLAB 不重复求解式（1）或式（2），只读取 CONTAM 求解后的路径流量。

## 3. 与原有模型的接口

### 3.1 CONTAM 侧

原模型中保留以下路径：

- 房间之间的门、洞口和通道；
- 房间与室外之间的开启门窗；
- 已有机械通风路径。

新增以下路径：

```text
OUTSIDE ── envelope leakage path ── Zone i
```

建议每条渗风路径至少包含以下字段：

| 字段 | 含义 | 必要性 |
|---|---|---:|
| `pathName` | 路径唯一标识 | 必需 |
| `zoneName` | 所属房间 | 必需 |
| `pathType` | `leakage` 或 `infiltration` | 必需 |
| `flowCoefficient` | $C_e$ | 必需 |
| `flowExponent` | $n_e$ | 必需 |
| `referencePressure` | 泄漏数据的参考压差 | 视输入类型而定 |
| `effectiveLeakageArea` | 有效漏风面积 | ELA 路线必需 |
| `pathHeight` | 路径高度 | 建议 |
| `facade` | 所属立面 | 建议 |
| `surfaceArea` | 路径代表的围护结构面积 | 参数分配时需要 |

优先使用 CONTAM 的 Leakage Area 或相应 Power-law airflow element。门窗大开口继续使用原有模型，不改为裂缝元件。

### 3.2 MATLAB 侧

现有完整流量矩阵保持为：

$$
\mathbf M_{\mathrm{flux}}
=\left[M_{ij}\right]_{m\times m},
\tag{3}
$$

其中包含室外节点和全部室内区域。新增渗风路径后，CONTAM 输出的 $\mathbf M_{\mathrm{flux}}$ 已经包含渗风引起的室外—房间流量，因此 MATLAB 不应再次向矩阵叠加经验渗风量。

为单独输出渗风指标，需要额外取得路径级结果：

$$
\mathcal F_{\mathrm{path}}
=\left\{e,u_e,v_e,Q_e,\mathrm{type}_e\right\}.
\tag{4}
$$

若当前接口只能输出区域间合计矩阵、不能输出路径编号和类型，则：

- 房间热平衡仍可包含渗风；
- 但无法在同一对室外—房间节点存在并行门窗和裂缝路径时，准确分离渗风量与有意通风量。

因此，是否能够读取 CONTAM 路径级流量是扩展实施前需要确认的关键接口条件。

## 4. 渗风参数化

### 4.1 优先级

参数来源建议采用以下优先级：

1. 压差—流量实测数据；
2. 整栋建筑气密性测试 $n_{50}$；
3. 有效漏风面积 ELA；
4. 同类型建筑或构造的文献数据；
5. 低、中、高气密性情景。

当缺少实测流动指数时，可暂取：

$$
n_e=0.65,
\tag{5}
$$

并将其标记为假设参数。原模型中大开口采用的 $n=0.5$ 不直接用于普通围护结构裂缝。

### 4.2 由参考流量换算流量系数

若路径在参考压差 $\Delta P_r$ 下的流量为 $Q_r$：

$$
C_e=\frac{Q_r}{(\Delta P_r)^{n_e}}.
\tag{6}
$$

### 4.3 由 $n_{50}$ 换算整栋建筑流量系数

$$
Q_{50}=\frac{n_{50}V_{\mathrm{bldg}}}{3600},
\tag{7}
$$

$$
C_{\mathrm{bldg}}=\frac{Q_{50}}{50^n}.
\tag{8}
$$

若采用外围护结构面积分配：

$$
C_e=C_{\mathrm{bldg}}
\frac{A_e}{\displaystyle\sum_{r\in\mathcal E_{\mathrm{leak}}}A_r}.
\tag{9}
$$

如果掌握窗框长度、构造接缝长度或不同构件的气密性指标，应以相应权重替代简单面积分配。

### 4.4 路径空间离散

建议分两级实现：

- 最小模型：每个房间的每个外立面设置一条等效泄漏路径；
- 精细模型：每个外立面的泄漏能力分配至低、中、高三个位置，以增强对热压的描述。

同一立面拆分后，应保持总泄漏能力不变：

$$
C_{e,\mathrm{low}}
+C_{e,\mathrm{mid}}
+C_{e,\mathrm{high}}
=C_{e,\mathrm{facade}}.
\tag{10}
$$

## 5. 与原有热平衡的关系

原有房间热平衡保持不变：

$$
\rho c_pV_i\frac{dT_i}{dt}
=Q_{\mathrm{heat},i}
+Q_{\mathrm{vent},i}
+Q_{\mathrm{envelope},i}.
\tag{11}
$$

新增渗风后，$Q_{\mathrm{vent},i}$ 仍由完整流量矩阵计算。为结果归因，可将其分解为：

$$
Q_{\mathrm{vent},i}
=Q_{\mathrm{original},i}
+Q_{\mathrm{inf},i},
\tag{12}
$$

其中室外渗风流入形成的显热项为：

$$
Q_{\mathrm{inf},i}
=\rho c_p
\sum_{e\in\mathcal E_{\mathrm{leak}},\;\mathrm{out}\rightarrow i}
Q_e(T_{\mathrm{out}}-T_i).
\tag{13}
$$

更一般的空气显热守恒仍为：

$$
Q_{\mathrm{air},i}
=\sum_j\dot m_{ji}c_pT_j
-\sum_j\dot m_{ij}c_pT_i.
\tag{14}
$$

式（13）是式（14）中由室外裂缝流入产生的组成部分。若现有 MATLAB 程序已根据完整流量矩阵计算式（14），则只需让 CONTAM 输出包含新增泄漏路径，不应在完整换热项之外再次增加式（13）。

## 6. 新增输出指标

房间 $i$ 的渗风换气次数定义为：

$$
\mathrm{ACH}_{\mathrm{inf},i}
=\frac{3600}{V_i}
\sum_{e\in\mathcal E_{\mathrm{leak}},\;\mathrm{out}\rightarrow i}Q_e.
\tag{15}
$$

建议增加以下输出：

| 输出 | 含义 |
|---|---|
| `infiltrationFlow` | 室外经泄漏路径流入房间的流量 |
| `exfiltrationFlow` | 房间经泄漏路径流向室外的流量 |
| `infiltrationACH` | 式（15）定义的渗风换气次数 |
| `infiltrationSensibleHeat` | 式（13）定义的渗风显热 |
| `totalOutdoorACH` | 渗风、开启门窗和机械新风的合计室外空气 ACH |

`infiltrationACH` 只统计室外经泄漏路径流入房间的流量，不包含：

- 房间间流动；
- 开启门窗的自然通风；
- 渗出流量的绝对值。

## 7. 扩展后的计算流程

扩展算法仍沿用原流程，只增加渗风路径生成与分类输出：

1. 读取原有建筑网络、房间参数、门窗路径和室内发热量；
2. 根据气密性输入生成室外—房间泄漏路径；
3. 将原有路径和新增泄漏路径共同写入 CONTAM；
4. 设置当前迭代的房间温度；
5. 调用 CONTAM 求解完整压力网络和流量矩阵；
6. 读取完整区域流量矩阵；
7. 根据路径标识提取渗风流入量、渗出量和渗风 ACH；
8. MATLAB 沿用原热平衡方法计算新室温；
9. 按原判据计算：

   $$
   \Delta T_{\max}
   =\max_i\left|T_i^{(k)}-T_i^{(k-1)}\right|;
   \tag{16}
   $$

10. 当 $\Delta T_{\max}<0.01\,^{\circ}\mathrm C$ 时输出结果，否则将新温度回传 CONTAM 并继续迭代。

## 8. 建议的实施顺序

### 阶段一：接口确认

确认现有程序是否支持：

1. 创建 CONTAM Leakage Area 或 Power-law airflow element；
2. 为同一室外—房间节点对保存多条并行路径；
3. 读取路径级有向流量及路径编号；
4. 明确 CONTAM 输出是 $\mathrm{m^3/s}$、$\mathrm{m^3/h}$ 还是 $\mathrm{kg/s}$；
5. 区分原矩阵中室外节点的行列方向。

### 阶段二：单房间验证

建立一个房间、一个室外节点和两条不同高度泄漏路径，完成：

- 幂律流量检查；
- 流动方向检查；
- 质量守恒检查；
- 渗风显热方向检查；
- MATLAB–CONTAM 温度迭代检查。

### 阶段三：多房间接入

在原建筑模型中按外立面和房间批量生成泄漏路径，测试门窗通风、区域间流动和渗风同时存在时的结果。

### 阶段四：参数与案例分析

使用实测气密性参数或低、中、高气密性情景，分析渗风对房间温度、自然通风效果和能耗的影响。

## 9. 验证与验收条件

### 9.1 退化一致性

当所有新增路径满足：

$$
C_e=0,
\tag{17}
$$

扩展模型应恢复原模型结果。

### 9.2 空气质量守恒

每个房间应满足：

$$
\left|
\sum_e\dot m_{e,\mathrm{in}}
-\sum_e\dot m_{e,\mathrm{out}}
\right|<\varepsilon_m.
\tag{18}
$$

### 9.3 单位一致性

若 CONTAM 输出体积流量，则进入热平衡前应转换为质量流量：

$$
\dot m=\rho Q.
\tag{19}
$$

若 CONTAM 已输出质量流量，不应重复乘空气密度。

### 9.4 物理趋势

- 增大流量系数后，统计期平均渗风量应总体增加；
- 增大风速或室内外温差后，渗风量应表现出相应变化；
- 室外冷空气渗入时，渗风显热应降低房间空气温度或增加供暖需求；
- 室外热空气渗入时，渗风显热应提高房间空气温度或增加制冷需求。

### 9.5 数值收敛

保留原温度收敛判据，并增加最大迭代次数和不收敛记录。如果新增热压反馈导致温度迭代振荡，可在不改变总体耦合框架的条件下对回传温度增加松弛：

$$
T_i^{(k+1),\mathrm{used}}
=(1-\alpha)T_i^{(k)}
+\alpha T_i^{(k+1),\mathrm{calculated}},
\qquad 0<\alpha\leq1.
\tag{20}
$$

松弛仅作为数值稳定措施，不改变 CONTAM 与 MATLAB 的物理分工。

## 10. 实施前需要确认的事项

| 编号 | 待确认事项 | 对实施的影响 |
|---:|---|---|
| 1 | 当前 CONTAM 项目使用何种 airflow element 表示门窗 | 决定裂缝元件如何与原路径共存 |
| 2 | 当前接口能否创建 Leakage Area/Power-law 元件 | 决定是否可自动生成渗风路径 |
| 3 | 当前输出能否取得路径级流量 | 决定能否单独统计渗风指标 |
| 4 | 流量矩阵的行列方向和单位 | 决定热平衡和 ACH 计算方式 |
| 5 | 当前热平衡是动态积分还是稳态矩阵求解 | 决定时间步和收敛处理 |
| 6 | 是否存在 $n_{50}$、ELA 或其他气密性数据 | 决定参数可信度和校准方式 |
| 7 | 是否需要考虑内隔墙泄漏 | 决定泄漏路径的空间范围 |
| 8 | 是否只考虑显热 | 决定是否需要增加湿度与潜热模块 |

## 11. 方案判断

该扩展方案与原有算法兼容，理由如下：

1. 原文已经采用 $Q=C(\Delta P)^n$ 的压力–流量关系，裂缝模型可沿用同一数学形式；
2. CONTAM 原生支持 Leakage Area 和 Power-law 泄漏元件；
3. 新增路径由 CONTAM 统一参与压力网络和质量守恒求解，无需改变 MATLAB 的压力求解逻辑；
4. 原 MATLAB 热平衡已使用空气流量矩阵，新增渗风可通过同一矩阵进入换热计算；
5. 当新增路径系数为零时，模型可以退化回原算法；
6. CONTAM 与热模型交换流量和温度的路线已有公开研究和验证案例。

建议采用“CONTAM 内建立泄漏路径、MATLAB 沿用完整流量矩阵计算热平衡、路径级结果单独统计渗风”的实现路线。首轮开发应先完成单房间最小案例，不建议直接在完整建筑中批量加入全部泄漏路径。

## 参考资料

1. W. S. Dols and B. J. Polidoro, *CONTAM User Guide and Program Documentation Version 3.2*, NIST Technical Note 1887, 2015. <https://doi.org/10.6028/NIST.TN.1887>
2. L. C. Ng, A. Musser, A. K. Persily, and S. J. Emmerich, “Multizone airflow models for calculating infiltration rates in commercial reference buildings,” *Energy and Buildings*, 58, 11–18, 2013. <https://doi.org/10.1016/j.enbuild.2012.11.035>
3. W. S. Dols, S. J. Emmerich, and B. J. Polidoro, “Coupling the Multizone Airflow and Contaminant Transport Software CONTAM with EnergyPlus Using Co-Simulation,” *Building Simulation*, 9, 469–479, 2016. <https://doi.org/10.1007/s12273-016-0279-2>
4. D. Qi, L. Wang, and R. Zmeureanu, “Verification of a Multizone Airflow and Energy Network Model by Analytical Solutions to Stack-driven Flows in Buildings,” eSim 2014.
5. EnergyPlus, `AirflowNetwork:MultiZone:Surface:Crack` documentation. <https://bigladdersoftware.com/epx/docs/22-1/input-output-reference/group-airflow-network.html>
6. LBL Modelica Buildings Library, `Buildings.Airflow.Multizone`. <https://simulationresearch.lbl.gov/modelica/releases/v12.1.0/help/Buildings_Airflow_Multizone.html>
