# LPG-Ventilation Paper Draft: Introduction and Method

## 1. Introduction

Natural ventilation is a high-impact passive strategy for reducing cooling demand, improving indoor environmental quality, and supporting low-carbon design in public buildings. Its importance is amplified in buildings with large volumes, deep plans, mixed functions, atria, corridors, halls, semi-open interfaces, and complex inter-room relationships, where ventilation performance depends not only on facade openings but also on whole-building spatial organization. In this context, cross ventilation should be treated as a building-scale network phenomenon rather than a single-room or single-window effect, because its effectiveness is governed by the connectivity among outdoor boundaries, indoor zones, doors, windows, voids, shafts, and circulation routes. Accordingly, early-stage natural-ventilation design must evaluate whether a layout can form effective inlet-outlet paths, whether interior zones can participate in through-flow, whether circulation spaces operate as distributors or bottlenecks, and whether opening strategies can reduce mechanical cooling demand while maintaining acceptable adaptive comfort.[^1] Through coupled airflow-thermal interactions, natural ventilation can simultaneously influence indoor comfort and energy use by modifying convective heat exchange, zone temperature trajectories, and the duration/intensity of mechanical conditioning demand.[^2]

To quantify these coupled airflow-thermal effects, Climate Fluid Dynamic (CFD) remains the standard method for resolving detailed local airflow distributions and is indispensable for validating environmental-field behavior, facade effects, local thermal comfort, and pollutant transport, especially in later design stages. This study does not treat node-network analysis as a replacement for CFD; rather, the two are complementary for early-stage decision-making. For the complex-building scenarios described above, the key early questions are often building-scale, including inlet-outlet organization, network continuity of airflow routes, and bottleneck/distribution effects across interconnected zones. In these cases, high-fidelity CFD can still provide valuable field-level insight, but it also depends on boundary conditions, meshing strategies, local opening geometries, interior obstructions, and facade details that are often undefined or unstable during conceptual development. As a result, early CFD analyses may rely on assumptions about details that have not yet become design commitments. Graph and node-network indicators therefore offer a practical companion perspective: they are typically more quantitative, direct, and fast for comparing alternatives at the building-scale (rather than only the space-scale) during iteration. Although ML surrogates can accelerate CFD prediction, they still depend strongly on CFD-generated datasets and discretization assumptions [^3]. At this stage, graph representation offers a more decision-relevant abstraction by preserving topological and relational structure while suppressing unnecessary geometric detail, thereby enabling performance reasoning directly on space and flow networks.[^4] Consistent with the geometry-to-graph paradigm based on convex graph encoding, graph-native representations provide an effective bridge between architectural form, simulation-ready structure, and performance-driven exploration in early design.[^5]

Building on this rationale, graph-based methods have shown strong capability for representing, generating, predicting, and optimizing architectural spatial organization and building performance [^4]. In the building ventilation and energy domain, graph representations can be broadly categorized into labelled property graphs (LPGs), typically used to encode CFD voxel topology, space topology, or HVAC topology, and knowledge graphs (KGs), typically used for Building Information Modeling (BIM) interpretation and generation. Table 1 summarizes representative applications and graph representations across building performance forecasting (BPF) as Graph based Prediction, building performance optimization (BPO) as Graph based Optimization, and design generation (DG) as Graph Generation. Among these, space and envelope topology are particularly relevant to early-stage design and have been adopted in BPF[^6], BPO[^7], and DG studies [^8] [^9] [^10]. Such topological representations are well suited to layout-level reasoning, because they support quantitative evaluation using graph metrics [^8] and graph-based performance assessment (energy[^11], thermal[^12], ventilation[^13]). However, due to the high computational cost of CFD and whole-building energy simulation, many studies rely on data-driven machine-learning (ML) surrogates trained on simulated or measured datasets rather than on directly interpretable physical formulations. This black-box dependence weakens interpretability of the relationship between design parameters and energy outcomes, limiting design guidance during iteration. In addition, some AI-agent-driven DG/BPO workflows [^11] couple optimization directly with high-fidelity engines such as EnergyPlus, leading to substantial computational time. For early-stage BPO and DG, simplified physical or semi-empirical models are therefore often a more practical and tractable option.

Table 1

| Topic | Graph representation | Study | Optimized or predicted target | Performance simulation method | Optimization method |
|---|---|---|---|---|---|
| Graph based Prediction | voxel topology LPG | [^14] [^15] [^16] |Indoor airflow | CNN/GNN CFD surrogate model | --- |
| Graph based Prediction | envelope, sensor, HVAC LPG | [^6] [^13] |Indoor thermal environment prediction | GNN-RNN model on measured data | --- |
| Graph based Optimization | voxel topology LPG | [^17] | Natural ventilation & cost | graph-based CFD surrogate model | VentHAN; NSGA-II |
| Graph based Optimization | envelope, sensor, HVAC LPG | [^7] [^18] [^12] | HVAC control, comfort and energy in open-plan spaces | PINN model on measured data | Physics-informed reinforcement learning |
| Graph Generation | BIM KG | [^19] [^11] | Space connectivity; energy performance | EnergyPlus | NSGA-like framework; AI-Agent workflow |
| Graph Generation | Space topology LPG | [^8] [^9] [^10] |Space layout generation on graph reliability | --- | NSGA-II;diffusion/denoising process |
| Graph Generation | envelope, sensor, HVAC LPG | [^20] [^21] | HVAC System deisgn; energy performance | EnergyPlus | reinforcement learning |


Multizone airflow network (AFN) models provide a more appropriate abstraction for early-stage ventilation design and a more natural simulation substrate for space- and HVAC-oriented graph representations.[^22] Relative to CFD, AFN is computationally lighter, less dependent on unresolved local geometry, and more consistent with topology-centered design questions.[^23] [^24] By representing rooms/zones as pressure nodes and doors, windows, shafts, cracks, and openings as flow paths, AFN shifts the analytical focus from fine-grained local distributions to inter-zone and indoor-outdoor connectivity, which is the primary decision layer during conceptual design.

Natural ventilation assessment, however, requires coupled airflow-thermal reasoning rather than airflow-only analysis. Through thermal-airflow co-simulation [^25] [^26], AFN-based workflows can jointly resolve flow exchange and thermal response, thereby producing reliable zone-level indicators such as air change rate (ACH) and indoor air temperature.[^23]  When further coupled with adaptive comfort and energy models,[^27] the same framework can propagate airflow and temperature states to thermal comfort and energy-use outcomes.[^25] [^26] This extended coupling improves physical fidelity, but it also increases computational cost; therefore, for graph-based optimization tasks, a practical balance must be maintained between simulation complexity and evaluation efficiency.[^24] [^28]

This paper addresses the above gap by proposing an LPG-Ventilation framework for early-stage natural ventilation assessment and graph-based design optimization. The research extends the CONTAM-based multizone airflow simulation from MOOSAS [^29], a early-design stage simulation software, coulping with a sensible heat-balance calculation to evaluate the adaptive comfort and the energy conservation with natrual ventilation strategies. The temperature result of this coulped simulation was calibrated by temperature record of multiple rooms in a natrual-ventilated office building in the spring period, ensuring the reliability of this semi-empirical simulation (R2=0.78). By adapting the model transformation function of MOOSAS [^5] and CUGER [^4], this framework allows flexible representation of KG and LPG and applys simultaneously to design model (*.obj, *.skp, *.ifc etc.). The same representation serves as the data foundation for GNN surrogate modeling and the grapth-based editable design optimization. Therefore, to explain its application potential in BPO and DG, this paper also show a optimization application with Bi-level Cross-Entropy Method (Bi-level CEM) targets on the structure and properties of space and outlet topology.[^30]

The key novelty of this study is its workflow-level integration with real design practice. By bridging B-rep, KG, and LPG representations, the framework enables direct graph-based intervention on original design models rather than on detached surrogate geometries. Within a unified pipeline, the same executable graph supports three tasks: rapid graph-based building performance forecasting (BPF) via GNN surrogates, graph-based building performance optimization (BPO) of spatial and outlet topology, and design generation with back-mapping to B-rep or BIM models. In addition, the semi-empirical simulation remains physically interpretable: outputs from CONTAM and the sensible heat-balance model preserve explicit links between design variables and performance responses, supporting traceable diagnosis and iterative refinement.

Academically, the framework contributes a unified representation that operationally links building-performance simulation, graph computation, and design optimization in one executable graph substrate. It provides a decision-oriented tool for evaluating and improving both natural and mechanically assisted ventilation strategies in complex public buildings, including opening layout, cross-ventilation organization, airflow-path configuration, and ventilation-enabled energy reduction.

## 2. Method

This section presents the proposed **LPG-Ventilation framework**, a graph-based workflow for evaluating and optimizing ventilation-driven performance in spatially complex buildings. The framework is built around a coupled simulation–optimization logic. First, a building design model is represented as a **ventilation-oriented labelled property graph** (LPG), where spaces are encoded as nodes and airflow paths are encoded as directed edges. This representation follows the broader development of graph-based building topology modeling and linked building data, while specializing the schema for executable ventilation simulation.[^31] [^4] The LPG can be derived from the knowledge graph generated by MOOSAS from B-rep building models, while the B-rep-to-KG conversion itself is treated as a previously established technical foundation rather than the novelty of this paper. Second, the LPG is translated into a multizone airflow network and linked to **CONTAM** to calculate pressure-driven and mechanically driven inter-zonal airflow. Third, the airflow matrix is coupled with a **sensible heat balance** model to calculate zone-level air temperatures. Fourth, the resulting hourly air-change rates and temperatures are passed to an adaptive comfort model and to the MOOSAS energy module to estimate comfort availability and energy demand under ventilation strategies. Finally, the same LPG serves as the decision space for a **Bi-level Cross-Entropy Method** (BiCEM), enabling joint optimization of graph structure and graph attributes.

The overall workflow can be summarized as follows. Given a building network dictionary or an LPG $G$, the ventilation module first generates a CONTAM project and computes an airflow network matrix $\mathbf{M}$. The airflow matrix is then used in a heat-balance solver to update the indoor temperature vector $\mathbf{T}$. Depending on the selected coupling mode, this process can be executed sequentially, iteratively in a ping-pong manner, or through an onion-like staged update. The coupled task returns four major outputs for each space: **air changes per hour** (ACH), **natural-ventilation indoor temperature**, **baseline energy demand**, and **energy demand under ventilation strategy**. These outputs are subsequently used as evaluation metrics for graph-based prediction or optimization. In the optimization workflow, BiCEM samples candidate LPGs, synchronizes them back to the ventilation and energy network dictionaries, evaluates each candidate through the coupled simulation, and updates its sampling distributions according to elite candidates.

| Component | Input Representation | Main Operation | Output |
|---|---|---|---|
| MOOSAS model conversion | B-rep building model | Previously established B-rep-to-KG conversion and graph extraction | Spatial KG / design graph |
| Ventilation LPG construction | KG or network dictionary | Zone-node and path-edge LPG generation | Executable ventilation LPG |
| Multizone airflow simulation | LPG-derived AFN | CONTAM simulation of inter-zonal and outdoor airflow | Hourly airflow matrix and ACH |
| Thermal coupling | AFN matrix, outdoor temperature, zone heat load | Sensible heat-balance solution with airflow advection | Zone air temperature |
| Comfort and energy assessment | ACH, temperature, weather, zone energy model | Adaptive comfort evaluation and MOOSAS energy calculation | Comfort state and energy demand |
| Graph optimization | LPG schema and variable bounds | BiCEM structure–attribute sampling and elite update | Optimized ventilation graph |

### 2.1 Coupled CONTAM–Sensible Heat Balance Model

The core simulation model couples a **multizone airflow network** with a **zone-level sensible heat balance**. Unlike CFD, which resolves local velocity, turbulence and detailed flow structures at high spatial resolution, the multizone airflow network represents a building as a set of well-mixed zones connected by airflow paths. This abstraction is suitable for early design because it focuses on spatial adjacency, openings, cross-ventilation paths and pressure-driven flow organization, rather than relying on fine geometric details that are often unavailable during schematic design.

Let the building contain $n$ indoor zones and one outdoor boundary node. The airflow network is represented by a directed mass-flow matrix

$$
\mathbf{F} = [F_{ij}] \in \mathbb{R}^{(n+1)\times(n+1)},
$$

where $F_{ij}$ denotes the airflow rate from node $i$ to node $j$. The additional $(n+1)$-th node represents the outdoor environment. In the implemented workflow, CONTAM is used to solve the multizone airflow network and return an airflow matrix for each simulation step. CONTAM is a widely used multizone airflow and contaminant transport simulation tool for building airflow-network analysis.[^32] The matrix records both indoor–indoor exchange and outdoor–indoor exchange, thereby preserving the building-scale ventilation organization.

The airflow result is converted into ACH for each zone by normalizing outdoor or total exchange flow by zone volume:

$$
\mathrm{ACH}_i = \frac{3600\,\dot{V}_i}{V_i},
$$

where $\dot{V}_i$ is the volumetric airflow rate associated with zone $i$, and $V_i$ is the zone volume. In the implementation, the CONTAM output matrix is parsed and mapped to user-defined zone names, allowing the subsequent thermal and energy modules to retain semantic consistency with the building model.

The thermal component solves a zone-level sensible heat balance. For zone $i$, the steady-state sensible balance can be written as

$$
Q_i + \sum_{j=1}^{n} \rho c_p \dot{V}_{ji}(T_j - T_i) + \rho c_p \dot{V}_{0i}(T_0 - T_i) = 0,
$$

where $Q_i$ is the internal sensible heat gain or heat load of zone $i$, $\rho$ is air density, $c_p$ is the specific heat capacity of air, $\dot{V}_{ji}$ is the airflow rate from zone $j$ to zone $i$, $\dot{V}_{0i}$ is the outdoor air inflow rate, $T_j$ and $T_i$ are indoor zone temperatures, and $T_0$ is the outdoor temperature. In the current implementation, $\rho$ is taken as approximately $1.2\,\mathrm{kg/m^3}$, and $c_p$ as approximately $1005\,\mathrm{J/(kg\cdot K)}$. The airflow rates are converted from hourly volumetric flow to heat-transfer coefficients through the factor $\rho c_p/3600$.

The implemented matrix form can be expressed as

$$
\mathbf{A}(\mathbf{F})\mathbf{T} = -\left(\mathbf{Q} + \mathbf{q}_{0}\right),
$$

where $\mathbf{T}=[T_1,\ldots,T_n]$ is the unknown indoor temperature vector, $\mathbf{Q}=[Q_1,\ldots,Q_n]$ is the zone heat-load vector, and $\mathbf{q}_{0}$ is the outdoor-air heat contribution. The coefficient matrix $\mathbf{A}$ is assembled from the airflow matrix after applying the heat-capacity conversion factor. The diagonal terms represent the net outflow balance of each zone, while off-diagonal terms represent advective heat exchange between connected zones. The outdoor boundary contribution is calculated as

$$
\mathbf{q}_{0}=\rho c_p\,\mathbf{F}_{0\rightarrow z}T_0/3600,
$$

where $\mathbf{F}_{0\rightarrow z}$ denotes the outdoor-to-zone airflow vector. The indoor temperature is then obtained by solving

$$
\mathbf{T} = -\left(\mathbf{Q}+\mathbf{q}_{0}\right)\mathbf{A}^{-1}.
$$

In practice, the solver returns zone temperatures in Kelvin and then converts them to Celsius for comfort and energy calculations. To avoid physically unrealistic values caused by singular or near-singular airflow matrices in early-stage exploratory cases, the implementation includes a bounded post-processing step for extreme results.

The airflow and thermal models are coupled because airflow depends on pressure and temperature conditions, while indoor temperature depends on airflow exchange. The framework therefore implements three coupling modes. In **sequence mode**, the airflow network is simulated first and then passed once to the heat-balance solver. In **ping-pong mode**, CONTAM simulation and heat-balance solution are alternated: CONTAM produces an airflow matrix, the heat-balance model updates zone temperatures, and the updated temperatures are written back into the CONTAM project for the next iteration. In **onion mode**, the coupled calculation is organized as staged updates that progressively propagate the influence of airflow and thermal states through the network. These three modes allow the workflow to balance numerical cost and coupling fidelity according to the analysis purpose.

The key methodological point is that the framework does not attempt to reproduce local CFD flow fields. Instead, it calculates the building-level organization of ventilation exchange and its thermal consequence at the spatial-network scale, which is more consistent with early-stage design decisions concerning room layout, opening placement and cross-ventilation routes.

### 2.2 Adaptive Comfort and Energy Models

Based on the coupled airflow–temperature results, the framework evaluates whether each zone can be regarded as comfortable under natural ventilation. The comfort module follows an **adaptive comfort evaluation** logic, where the acceptable indoor temperature range is determined by outdoor climatic conditions rather than by a fixed setpoint, consistent with the adaptive thermal comfort paradigm used for naturally ventilated buildings.[^33] [^1] For each hour $t$ and zone $i$, the coupled simulation returns indoor temperature $T_{i,t}$ and ACH $a_{i,t}$. The adaptive comfort state is evaluated as a binary or fractional indicator:

$$
C_{i,t}=\mathbb{I}\left(T_{i,t}^{\min}(T_{0,t}) \le T_{i,t} \le T_{i,t}^{\max}(T_{0,t})\right),
$$

where $T_{0,t}$ is the outdoor temperature, and $T_{i,t}^{\min}$ and $T_{i,t}^{\max}$ are outdoor-temperature-dependent comfort limits. In the implemented annual comfort module, the upper comfort boundary is formulated as a linear function of outdoor temperature. The code applies the following adaptive rule:

$$
T_{i,t} < 0.31T_{0,t}+20.3,
$$

with an additional lower-bound screening. This converts the continuous thermal output of the coupled model into a comfort availability signal that can be aggregated by zone, by hour, or across the whole building.

Because indoor temperature can fluctuate strongly under natural ventilation, the annual comfort calculation also includes a smoothing mechanism that reflects thermal inertia. The smoothed temperature $\widetilde{T}_{i,t}$ can be conceptually expressed as

$$
\widetilde{T}_{i,t}=kT_{i,t}+(1-k)\frac{1}{w}\sum_{\tau=t-w}^{t-1}\widetilde{T}_{i,\tau},
$$

where $k$ controls the weight of the current coupled simulation result and $w$ defines the historical smoothing window. This treatment is particularly useful when the model is applied to annual or long-period simulations, because it reduces unrealistic hour-to-hour discontinuities while preserving the main thermal trend driven by ventilation.

The energy evaluation is conducted through the MOOSAS energy module. For each zone, the baseline energy task calculates hourly or daily heating, cooling and lighting loads:

$$
E^{\mathrm{base}}_{i,t}=E^{\mathrm{heat}}_{i,t}+E^{\mathrm{cool}}_{i,t}+E^{\mathrm{light}}_{i,t}.
$$

The ventilation-aware energy demand is then estimated by linking comfort availability to the heating and cooling loads. When the naturally ventilated condition is comfortable, mechanical heating or cooling can be reduced or avoided for that hour. A simplified implementation-level expression is

$$
E^{\mathrm{vent}}_{i,t}=\left(E^{\mathrm{heat}}_{i,t}+E^{\mathrm{cool}}_{i,t}\right)(1-C_{i,t})+E^{\mathrm{light}}_{i,t}+E^{\mathrm{mech}}_{i,t},
$$

where $E^{\mathrm{mech}}_{i,t}$ is the energy required by mechanically assisted ventilation, if any. This formulation makes the effect of ventilation explicit: natural ventilation reduces conditioning energy only when the zone is predicted to be comfortable under the coupled airflow–temperature calculation, while lighting demand remains independent of the ventilation state.

Mechanical ventilation energy is calculated from airflow paths labelled as mechanical or forced ventilation routes. For a mechanical path $e$, the ventilation power can be represented as

$$
P_{e,t}=\frac{\Delta p_e\dot{V}_{e,t}}{\eta_e},
$$

where $\Delta p_e$ is the pressure rise associated with the path, $\dot{V}_{e,t}$ is the airflow rate, and $\eta_e$ is fan efficiency. Equivalently, the implementation can use a specific fan power coefficient $s_e$:

$$
E^{\mathrm{mech}}_{e,t}=s_e\dot{V}_{e,t}\Delta t.
$$

The ventilation module identifies relevant paths through path labels and path attributes, reads mass or volumetric airflow from the airflow network result, and combines this with pressure or fan-power assumptions to estimate the mechanical ventilation energy contribution. Therefore, the final ventilation-aware energy output contains both passive savings from comfort-driven natural ventilation and active penalties from mechanically driven airflow.

The coupled task ultimately produces four zone-level indicators:

$$
\left\{\mathrm{ACH}_{i,t},\;T_{i,t},\;E^{\mathrm{base}}_{i,t},\;E^{\mathrm{vent}}_{i,t}\right\}.
$$

These indicators serve two purposes. First, they provide interpretable performance results for architects and engineers, including cross-ventilation effectiveness, indoor temperature and energy reduction. Second, they form the objective and constraint signals for graph-based optimization and prediction.

### 2.3 Ventilation LPG Schema

The simulation framework uses a **labelled property graph** as the executable data representation for ventilation analysis. A ventilation LPG is defined as

$$
G=(V,E,\ell_V,\ell_E,\mathbf{X}_V,\mathbf{X}_E),
$$

where $V$ is the node set, $E$ is the directed edge set, $\ell_V$ and $\ell_E$ are node and edge labels, and $\mathbf{X}_V$, $\mathbf{X}_E$ are node and edge property maps. In this framework, nodes represent indoor zones and the outdoor boundary, while directed edges represent airflow paths. The LPG is implemented as a directed multigraph because multiple ventilation paths can exist between the same pair of spaces.

The node set is defined as

$$
V=V_z\cup\{v_0\},
$$

where $V_z=\{v_1,\ldots,v_n\}$ represents indoor spatial zones, and $v_0$ is the outside node. Each zone node stores a compact set of thermal-envelope attributes:

$$
\mathbf{x}_{v_i}=\{U^{\mathrm{wall}}_i,\;U^{\mathrm{win}}_i,\;\mathrm{SHGC}^{\mathrm{win}}_i\}.
$$

These correspond to `zone_wallU`, `zone_winU` and `zone_win_SHGC` in the implementation. They are selected because they directly affect heat gain, heat loss and the energy module, while remaining available or estimable during early design.

Each path edge $e=(u,v,k)$ stores airflow-path properties:

$$
\mathbf{x}_{e}=\{h_e,\;w_e,\;\Delta p_e\},
$$

where $h_e$ is path height, $w_e$ is path width, and $\Delta p_e$ is the pressure parameter. These correspond to `pathHeight`, `pathWidth` and `pressure`. The edge key $k$ preserves the identity of the path, allowing the graph to distinguish multiple doors, windows or mechanical ventilation links between the same zones.

| LPG element | Graph object | Main properties | Simulation meaning |
|---|---|---|---|
| Indoor space | Zone node | `zone_wallU`, `zone_winU`, `zone_win_SHGC` | Thermal envelope properties for energy and heat-balance calculation |
| Outdoor boundary | Outside node | Labelled as `OUTSIDE` | Boundary condition for outdoor airflow and temperature |
| Door/window/opening | Directed path edge | `pathHeight`, `pathWidth`, `pressure` | AFN path geometry and pressure-driving attribute |
| Mechanical ventilation path | Directed path edge, typically zone-to-outside or inter-zone | Path label, pressure, flow-related properties | Forced ventilation route and fan-energy calculation |

The LPG schema is intentionally compact. It does not attempt to encode all geometric details of a B-rep model, such as local corner geometry, duct shape or detailed opening profiles. Instead, it preserves the information required to compute spatial ventilation organization: which zones are connected, how large the paths are, whether they connect to outside, and what pressure or mechanical forcing is associated with them. This abstraction is critical for early-stage design because it supports rapid iteration without requiring high-fidelity geometric information that is usually defined only in later design phases.

The LPG is connected to MOOSAS through bidirectional conversion. On one side, MOOSAS provides the previously established B-rep-to-KG conversion, allowing design models to be transformed into a semantic spatial graph. On the other side, the ventilation module converts the network dictionary into an executable LPG and writes edited LPG attributes back into the network dictionary. The conversion from network dictionary to LPG maps zone user names to graph nodes and path user names to graph edges. The reverse conversion writes only whitelisted properties back to the simulation model, ensuring that the graph-editing process remains controlled and physically meaningful.

Formally, the conversion can be described as

$$
\Phi_{\mathrm{LPG}}:\mathcal{D}_{\mathrm{network}}\rightarrow G,
$$

$$
\Psi_{\mathrm{network}}:G\rightarrow\mathcal{D}_{\mathrm{network}},
$$

where $\mathcal{D}_{\mathrm{network}}$ is the MOOSAS energy and ventilation network dictionary. The forward mapping $\Phi_{\mathrm{LPG}}$ extracts zone nodes and path edges, while the backward mapping $\Psi_{\mathrm{network}}$ synchronizes edited node and edge properties back to the simulation input. During this synchronization, window area is recalculated from outdoor-linked paths, while newly added optimization paths are handled separately to avoid unintended changes to the baseline envelope model.

This LPG schema provides the methodological bridge between simulation and graph intelligence. For graph prediction, the same schema can be used as the input representation for GNN-based surrogate models, where node and edge embeddings predict ACH, temperature, comfort or energy outcomes. For graph optimization, the same schema defines the decision variables for modifying path existence, opening dimensions, envelope properties and mechanical ventilation pressure.

### 2.4 Bi-level Cross-Entropy Method for LPG Optimization

The proposed optimization module uses a **Bi-level Cross-Entropy Method** to optimize both the structure and attributes of the ventilation LPG. The method extends the general cross-entropy principle, which updates a sampling distribution from elite samples toward high-performing regions of the search space.[^34] The need for a bi-level formulation arises because ventilation design decisions contain two qualitatively different variable types. Structural decisions determine whether an airflow path exists between two zones or between a zone and the outside. Attribute decisions determine continuous properties such as opening size, envelope performance and mechanical pressure. These two variable types have different probability models and feasibility constraints; therefore, they are optimized with coupled but distinct CEM distributions.

Let the candidate ventilation LPG be represented as

$$
G(\mathbf{z},\mathbf{x}),
$$

where $\mathbf{z}\in\{0,1\}^{m}$ is a binary structure vector and $\mathbf{x}\in\mathbb{R}^{d}$ is a continuous attribute vector. Each binary variable $z_r$ corresponds to a candidate edge option $e_r$, including existing indoor paths, possible new indoor paths and possible mechanically driven zone-to-outside paths. Each continuous variable $x_j$ corresponds to an optimizable node or edge property within a predefined bound:

$$
L_j\le x_j\le U_j.
$$

The optimization problem is formulated as

$$
\min_{\mathbf{z},\mathbf{x}} J\left(G(\mathbf{z},\mathbf{x})\right),
$$

subject to

$$
\mathbf{z}\in\mathcal{Z},\quad \mathbf{x}\in\mathcal{X},\quad G(\mathbf{z},\mathbf{x})\in\mathcal{G}_{\mathrm{valid}},
$$

where $J$ is the objective function obtained from the coupled LPG-Ventilation simulation, $\mathcal{Z}$ is the feasible structure space, $\mathcal{X}$ is the bounded attribute space, and $\mathcal{G}_{\mathrm{valid}}$ is the set of valid ventilation graphs. In the current implementation, the default objective is to minimize the ventilation-aware energy demand:

$$
J\left(G\right)=\sum_{i\in V_z}\sum_{t\in\mathcal{T}}E^{\mathrm{vent}}_{i,t},
$$

although the same framework can also optimize comfort, ACH or temperature. For objectives that should be maximized, such as comfort or ACH, the implementation minimizes their negative values:

$$
J(G)=-\sum_{i,t}C_{i,t}\quad\text{or}\quad J(G)=-\sum_{i,t}\mathrm{ACH}_{i,t}.
$$

The structure-level CEM models each binary variable as a Bernoulli random variable:

$$
 z_r\sim \mathrm{Bernoulli}(p_r),\quad r=1,\ldots,m.
$$

The probability vector $\mathbf{p}$ is initialized according to the baseline graph: existing edges are assigned high initial probability, while non-existing candidate edges are assigned low initial probability. At each outer iteration, the optimizer samples multiple structure vectors, repairs or rejects infeasible structures, and then evaluates each valid structure through an inner attribute optimization process.

After evaluation, a subset of elite structure candidates $\mathcal{E}_s$ is selected according to the lowest objective values. The maximum-likelihood estimate of each Bernoulli parameter is

$$
\hat{p}^{\mathrm{MLE}}_r=\frac{1}{|\mathcal{E}_s|}\sum_{\mathbf{z}\in\mathcal{E}_s}z_r.
$$

The structure distribution is updated with exponential smoothing:

$$
 p_r^{(q+1)}=\mathrm{clip}\left((1-\alpha_s)p_r^{(q)}+\alpha_s\hat{p}^{\mathrm{MLE}}_r, p_{\min},p_{\max}\right),
$$

where $q$ is the outer iteration index, $\alpha_s$ is the structure learning rate, and $[p_{\min},p_{\max}]$ prevents premature convergence to deterministic edge selections.

The attribute-level CEM models continuous variables with a Gaussian distribution:

$$
 x_j\sim\mathcal{N}(\mu_j,\sigma_j^2),\quad j=1,\ldots,d,
$$

followed by clipping to the predefined bounds $[L_j,U_j]$. For each sampled structure, the inner loop samples multiple attribute vectors, assembles the corresponding LPG, runs the coupled simulation, and selects elite attribute candidates $\mathcal{E}_a$. The attribute distribution is updated as

$$
\hat{\mu}^{\mathrm{MLE}}_j=\frac{1}{|\mathcal{E}_a|}\sum_{\mathbf{x}\in\mathcal{E}_a}x_j,
$$

$$
\hat{\sigma}^{\mathrm{MLE}}_j=\sqrt{\frac{1}{|\mathcal{E}_a|}\sum_{\mathbf{x}\in\mathcal{E}_a}(x_j-\hat{\mu}^{\mathrm{MLE}}_j)^2},
$$

$$
\mu_j^{(r+1)}=(1-\alpha_a)\mu_j^{(r)}+\alpha_a\hat{\mu}^{\mathrm{MLE}}_j,
$$

$$
\sigma_j^{(r+1)}=\max\left((1-\alpha_a)\sigma_j^{(r)}+\alpha_a\hat{\sigma}^{\mathrm{MLE}}_j,\sigma_{j,\min}\right),
$$

where $r$ is the inner iteration index, $\alpha_a$ is the attribute learning rate, and $\sigma_{j,\min}$ is a lower bound that maintains exploration.

The bi-level logic can therefore be expressed as

$$
\mathbf{z}^{(q)}\sim P_s(\mathbf{z};\mathbf{p}^{(q)}),
$$

$$
\mathbf{x}^{(r)}\sim P_a(\mathbf{x};\boldsymbol{\mu}^{(r)},\boldsymbol{\sigma}^{(r)}\mid \mathbf{z}^{(q)}),
$$

$$
G^{(q,r)}=G(\mathbf{z}^{(q)},\mathbf{x}^{(r)}),
$$

$$
J^{(q,r)}=\mathcal{S}\left(G^{(q,r)}\right),
$$

where $\mathcal{S}(\cdot)$ denotes the complete LPG-Ventilation simulation chain: LPG synchronization, CONTAM airflow simulation, sensible heat-balance calculation, adaptive comfort evaluation and energy calculation.

The implementation applies several feasibility controls to ensure that sampled graphs remain meaningful. First, the outside boundary is locked: original outside-related edges must be preserved, and newly added outside edges are restricted to controlled mechanical ventilation links. Second, sampled indoor structures are constrained by a maximum number of indoor edges to avoid unrealistically dense airflow networks. Third, the graph is checked for valid node and edge organization before simulation. Invalid graphs are either repaired or assigned a failure penalty. These constraints can be written as

$$
\Omega_{\mathrm{outside}}(G)=1,
$$

$$
\sum_{r\in\mathcal{E}_{\mathrm{indoor}}}z_r\le M_{\max},
$$

$$
G\in\mathcal{G}_{\mathrm{connected/valid}},
$$

where $\Omega_{\mathrm{outside}}$ denotes the outside-edge locking rule and $M_{\max}$ is the maximum allowed number of indoor candidate edges.

Within the proposed framework, BiCEM is not merely a numerical optimizer; it is a graph-based design mechanism. Structural variables correspond to design operations such as adding, removing or retaining airflow paths. Attribute variables correspond to adjusting opening dimensions, envelope parameters or mechanical pressure. Because every sampled design is represented as an LPG and evaluated by the same coupled simulation pipeline, the optimization result remains traceable to spatial design actions. This makes the framework suitable for early-stage exploration of cross ventilation, hybrid ventilation and mechanical ventilation organization in complex public buildings.

| Optimization level | Variable | Probability model | Design meaning | Update criterion |
|---|---|---|---|---|
| Structure level | $\mathbf{z}\in\{0,1\}^{m}$ | Bernoulli distribution | Existence of airflow paths and mechanical links | Elite structures with lower objective values |
| Attribute level | $\mathbf{x}\in\mathbb{R}^{d}$ | Bounded Gaussian distribution | Opening size, pressure and envelope attributes | Elite attributes under each candidate structure |
| Simulation evaluation | $G(\mathbf{z},\mathbf{x})$ | Deterministic coupled workflow | Ventilation, comfort and energy performance | CONTAM + heat balance + comfort + energy calculation |

Consequently, the Method integrates three layers of contribution. The physical layer couples airflow and sensible heat balance to produce temperature and ACH. The performance layer translates these results into adaptive comfort and energy indicators. The graph layer provides a compact, editable and optimizable LPG representation, through which BiCEM can search both spatial ventilation organization and attribute settings. Together, these layers allow ventilation analysis to move from isolated simulation toward graph-based prediction and optimization for early-stage building design.

## References

[^1]: Center for the Built Environment, "Adaptive Comfort Model," University of California, Berkeley. https://cbe.berkeley.edu/research/adaptive-comfort-model/

[^2]: R. J. de Dear and G. S. Brager, "Thermal comfort in naturally ventilated buildings: revisions to ASHRAE Standard 55," *Energy and Buildings*, 34, 549-561, 2002. https://www.sciencedirect.com/science/article/abs/pii/S0378778802000051

[^3]: R. Mao, Y. Lan, L. Liang, T. Yu, M. Mu, W. Leng, and Z. Long, "Rapid CFD Prediction Based on Machine Learning Surrogate Model in Built Environment: A Review," *Fluids*, 10(8), 193, 2025. https://www.mdpi.com/2311-5521/10/8/193

[^4]: Y. Li, J. Xiao, H. Zhou, and B. Lin, "From geometry to graph: Automation of building performance modeling via convex graph encoding," *Automation in Construction*, 183, 106815, 2026. https://www.sciencedirect.com/science/article/pii/S0926580526000567

[^5]: J. Xiao, Q. Wang, Y. Li, Z. Yu, H. Zhou, and B. Lin, "A Fully Automated DM-BIM-BEM Pipeline Enabling Graph-Based Intelligence, Interoperability, and Performance-Driven Early Design," arXiv:2601.16813, 2026. https://arxiv.org/abs/2601.16813

[^6]: W. Wang and Z. Zhou, "Spatio-temporal Prediction of Indoor Thermal Environment Based on Graph Neural Network and Recurrent Neural Network," *Building Simulation*, 2025. https://link.springer.com/article/10.1007/s12273-025-1233-y

[^7]: S. Nagarathinam and A. Vasan, "PhyGICS - A Physics-informed Graph Neural Network-based Intelligent HVAC Controller for Open-plan Spaces," *ACM e-Energy*, 2024. https://dl.acm.org/doi/10.1145/3632775.3661960

[^8]: M. Keshavarzi and M. Rahmani-Asl, "GenFloor: Interactive generative space layout system via encoded tree graphs," *Frontiers of Architectural Research*, 2021. https://www.sciencedirect.com/science/article/pii/S2095263521000715

[^9]: P. Su et al., "Floor plan graph learning for generative design of residential buildings: a discrete denoising diffusion model," *Building Research & Information*, 2024. https://www.tandfonline.com/doi/full/10.1080/09613218.2023.2288097

[^10]: R. Hu et al., "Graph2Plan: learning floorplan generation from layout graphs," *ACM Transactions on Graphics (TOG)*, 4(39), 118:1-118:14, 2025. https://doi.org/10.1145/3386569.3392391

[^11]: C. Tang et al., "Archi-Agents Approximating the architectural design process with collaborative specialized multi-agent system llm and building information modeling," *Architectural Informatics - Proceedings of the 30th CAADRIA Conference*, 2025. https://doi.org/10.52842/conf.caadria.2025.3.141

[^12]: Y. He et al., "Enhancing Intelligent HVAC optimization with graph attention networks and stacking ensemble learning," *Scientific Reports*, 15, 5119, 2025. https://www.nature.com/articles/s41598-025-89776-6

[^13]: J. Onyejizu and R. F. Karlicek, "Poster Abstract: Dynamic Graph Learning for Multizone Indoor CO2 Prediction," *Proceedings of the 12th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation*, 2025. https://doi.org/10.1145/3736425.3772126

[^14]: J. Vandewiel et al., "Approximating CFD simulations of natural ventilation: A deep surrogate model with spatial attention mechanism," *Journal of Building Engineering*, 2025. https://www.sciencedirect.com/science/article/pii/S2352710225004427

[^15]: J. M. Han and A. Malkawi, "Airvox: efficient computational fluid dynamics prediction using 3D convolutional neural networks for building design," *Journal of Building Performance Simulation*, 18(1), 2025. https://www.tandfonline.com/doi/full/10.1080/19401493.2024.2410744

[^16]: W. Zhang, C. Zhang, Y. Zhao, Z. Wang, Y. Liu, C. Zhou, and Y. Hu, "Convolutional neural networks-based surrogate model for fast computational fluid dynamics simulations of indoor airflow distribution," *Energy and Buildings*, 326, 115020, 2025. https://www.sciencedirect.com/science/article/pii/S0378778824011368

[^17]: Z. Xu, W. Lu, and Z. Peng, "Enhancing natural ventilation in modular buildings: A reversible WindModule graph generative design approach," *Building and Environment*, 2025. https://www.sciencedirect.com/science/article/pii/S0360132325013666

[^18]: J. Zhang et al., "Graph neural network-based spatio-temporal indoor environment prediction and optimal control for central air-conditioning systems," *Building and Environment*, 2023. https://www.sciencedirect.com/science/article/pii/S0360132323006278

[^19]: V. J. L. Gan, "BIM-based graph data model for automatic generative design of modular buildings," *Automation in Construction*, 2022. https://www.sciencedirect.com/science/article/pii/S0926580521005136

[^20]: H. Wang, R. Jin, P. Xu, and J. Gu, "Generation Method for HVAC Systems Design Schemes in Office Buildings Based on Deep Graph Generative Models," *Buildings*, 14(11), 3405, 2024. https://www.mdpi.com/2075-5309/14/11/3405

[^21]: M. Wang, G. Lilis, D. Mavrokapnidis, K. Katsigarakis, I. Korolija, and D. Rovas, "A knowledge graph-based framework to automate the generation of building energy models using geometric relation checking and HVAC topology establishment," *Energy and Buildings*, 2024. https://doi.org/10.1016/j.enbuild.2024.115035

[^22]: J. Song, S. Yoon, and J. Nah, "Ontology-assisted GPT-based building performance simulation and assessment: Implementation of multizone airflow simulation," *Energy and Buildings*, 325, 114983, 2024. https://doi.org/10.1016/j.enbuild.2024.114983

[^23]: W. Axley, S. J. Emmerich, and G. N. Walton, "Modeling the Performance of a Naturally Ventilated Commercial Building with a Multizone Coupled Thermal/Airflow Simulation Tool," NIST/ASHRAE, 2002. https://www.nist.gov/publications/modeling-performance-naturally-ventilated-commercial-building-multizone-coupled-1

[^24]: G. A. Faggianelli, A. Brun, E. Wurtz, and M. Muselli, "Assessment of different airflow modeling approaches on a naturally ventilated Mediterranean building," *Energy and Buildings*, 107, 345-354, 2015. https://doi.org/10.1016/j.enbuild.2015.08.038

[^25]: P. Michalak, "Thermal-Airflow Coupling in Hourly Energy Simulation of a Building with Natural Stack Ventilation," *Energies*, 15(11), 4175, 2022. https://www.mdpi.com/1996-1073/15/11/4175

[^26]: Y. Chen, L. Gu, and J. Zhang, "EnergyPlus and CHAMPS-Multizone co-simulation for energy and indoor air quality analysis," *Building Simulation*, 8, 371-380, 2015. https://doi.org/10.1007/s12273-015-0211-1

[^27]: W. S. Dols, S. J. Emmerich, and B. J. Polidoro, "Coupling the Multizone Airflow and Contaminant Transport Software CONTAM with EnergyPlus Using Co-Simulation," *Building Simulation*, 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC4873778/

[^28]: M. Eydner, B. Toufek, T. Benzler, and K. Stergiaropoulos, "Investigation of a multizone building with HVAC system using a coupled thermal and airflow model," *E3S Web of Conferences*, 111, 04040, 2019. https://doi.org/10.1051/e3sconf/201911104040

[^29]: B. Lin, H. Chen, Q. Yu, X. Zhou, S. Lv, Q. He, and Z. Li, "MOOSAS - A systematic solution for multiple objective building performance optimization in the early design stage," *Building and Environment*, 200, 107929, 2021. https://doi.org/10.1016/j.buildenv.2021.107929

[^30]: B. Grueter, J. Diepolder, M. Bittner, F. Holzapfel, and J. Z. Ben-Asher, "Bi-level Cross Entropy Method and Optimal Control for Air Traffic Sequencing and Trajectory Optimization," *AIAA Scitech 2020 Forum*, 2020. https://doi.org/10.2514/6.2020-1590

[^31]: M. H. Rasmussen, M. Lefrancois, G. F. Schneider, and P. Pauwels, "BOT: The Building Topology Ontology of the W3C Linked Building Data Group," *Semantic Web*, 11(1), 143-161, 2020. https://doi.org/10.3233/SW-200385

[^32]: National Institute of Standards and Technology (NIST), "CONTAM Software," https://www.nist.gov/services-resources/software/contam

[^33]: R. J. de Dear and G. S. Brager, "Developing an adaptive model of thermal comfort and preference," *Energy and Buildings*, 104, 145-167, 2002. https://doi.org/10.1016/S0378-7788(02)00053-4

[^34]: R. Y. Rubinstein and D. P. Kroese, "The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation, and Machine Learning," Springer, 2004. https://doi.org/10.1007/978-1-4757-4321-0
