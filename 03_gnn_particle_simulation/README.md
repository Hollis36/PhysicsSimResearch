# GNN-Based Particle Simulation: 基于图神经网络的粒子仿真综合研究笔记

> 面向喷涂路径规划与物理仿真研究者的深度调研报告
> 调研日期: 2026-02-04 (更新: **2026-06-17**)

---

## 目录

0. [🔄 最新进展更新 (2026-02 → 2026-06)](#-最新进展更新-2026-02--2026-06)
1. [领域概述：从经典粒子方法到神经网络替代](#1-领域概述从经典粒子方法到神经网络替代)
2. [GNN 仿真核心原理](#2-gnn-仿真核心原理)
3. [主要方法详解与对比](#3-主要方法详解与对比)
4. [关键论文列表](#4-关键论文列表)
5. [与喷涂粒子动力学建模的对接方案](#5-与喷涂粒子动力学建模的对接方案)
6. [开源代码资源汇总](#6-开源代码资源汇总)
7. [推荐入门路线](#7-推荐入门路线)

---

## 🔄 最新进展更新 (2026-02 → 2026-06)

> 本节于 **2026-06-17** 增补,记录自原调研 (2026-02-04) 以来的前沿进展。每条均附可验证 arXiv 来源;明确区分 **[已验证]** 与 **[需谨慎]**,并诚实标注空白。

### 0.1 核心趋势: 走向"统一 Transformer 拉格朗日粒子仿真器"

**[已验证] WorldParticle: Unified World Simulation of Lagrangian Particle Dynamics via Transformer** (arXiv:[2605.15305](https://arxiv.org/abs/2605.15305), 2026-05-14, Wang, Guo, … Chenfanfu Jiang, Komura, Matusik, P.Y. Chen) —— 本窗口**最重要**的进展。单个 Transformer,在共享的拉格朗日粒子表示上做"预测-校正";一个模型覆盖 **布料、弹性固体、牛顿 + 非牛顿流体、颗粒、分子动力学 (共 6 类)**;粒子 tokenizer + 分层 super-token 编码器;可泛化到未见材料/边界/外力,支持**交互控制与逆向设计**。方向上正**取代单物理 GNS 式模型**。(摘要未给出与 GNS 的定量对比。)

### 0.2 新论文 (Feb–Jun 2026, 均已核验 arXiv)

| 论文 | 会议 | 日期 | arXiv | 一句话 (+ 与喷涂相关性) |
|------|------|------|-------|-------------------------|
| **Differentiable GNN Simulator for Back-Analysis of Post-Liquefaction Residual Strength** | arXiv (geotech) | 2026-02-12 | [2602.11621](https://arxiv.org/abs/2602.11621) | GNS (MPM 训练) + autodiff 逆优化,从流动破坏 runout 反推残余强度。**可微分 GNS 的逆问题范式**,与喷涂"从涂层反推参数"同构 |
| **GNN for Multitask Prediction of Rheological & Microstructural Behavior in Suspensions** | arXiv (soft matter) | 2026-02-10 | [2602.07296](https://arxiv.org/abs/2602.07296) | GNN 从颗粒构型同时预测流变 (黏度/剪应力) + 微结构。**与涂料/涂层最相关** —— 油漆即稠密颗粒悬浮液 |
| **EquiformerV3: Scaling Efficient SE(3)-Equivariant Graph Attention Transformers** | arXiv | 2026-04-10 | [2604.09130](https://arxiv.org/abs/2604.09130) | MD/材料的 SOTA SE(3)-等变力/能 GNN-Transformer;比 V2 小 5×、训练快 1.75–5.9×。原子尺度 (非宏观流体),但**等变架构前沿参考** |
| **LBM-Driven PINN for Droplet Wettability on Rough Surfaces** | arXiv | 2026-04-03 | [2604.03481](https://arxiv.org/abs/2604.03481) | LBM 驱动 PINN 建模液滴在粗糙表面的铺展/钉扎/毛细滞后;L2≈0.02,R²≈0.999,>10⁴ evals/s。**液滴相关** (是 PINN,非粒子 GNN) |

> **相关锚点 (窗口外但重要)**: **NeuralDEM** 已在 **Nature Communications Physics 8, 440 (2025-11-18)** 正式发表 (arXiv:[2411.09678](https://arxiv.org/abs/2411.09678)):实时 DEM 代理,25 万颗粒漏斗单 GPU **1.4 s** vs 16 核 CPU **3 h**。是 DEM 神经替代最强的同行评审锚点。

### 0.3 已有方法/代码库状态 (截至 2026-06-17)

- **LagrangeBench** (`tumaer/lagrangebench`): **窗口内无新发布**,最新 tag 仍为 **v0.2.0 (2024-07-08)**;仓库有 2026 维护提交但无新版本/数据集。
- **NeuralMPM** (arXiv:2408.15753): **无已确认的 2026 后续**;最新仍为 v2 (2025-02-24),3D 扩展仍是 future work。
- **Neural SPH (2402.06275) / JAX-SPH**: 窗口内无新版本或发布。
- **GNS (geoelements/gns 血统)**: 活动体现在**应用**而非核心发布 —— 如 §0.2 的液化逆分析 (2602.11621) 使用可微分 GNS 框架。

### 0.4 新基准与空白

- **新基准: 无。** 窗口内未发现新的拉格朗日/粒子基准套件;LagrangeBench (NeurIPS 2023) 仍是标准,无 2026 后继。
- **空白提示 (诚实标注)**: 未发现任何 2026 的 **GNN/粒子代理专门用于喷涂雾化、液滴撞击沉积、热喷涂或涂膜厚度**。最接近的 2026 工作仅为一个液滴润湿 PINN (2604.03481) 与一个悬浮液流变 GNN (2602.07296),均非喷涂代理。**学习式*粒子级*喷涂/涂层仿真在 2026 前沿文献中仍是空白** —— 对本研究是开放机会。

---

## 1. 领域概述：从经典粒子方法到神经网络替代

### 1.1 经典粒子仿真方法

在喷涂、流体、颗粒流等工程仿真领域，三类经典无网格粒子方法占据主导地位：

| 方法 | 全称 | 适用场景 | 核心思想 |
|------|------|----------|----------|
| **SPH** | Smoothed Particle Hydrodynamics (光滑粒子流体动力学) | 流体、自由表面流、喷涂液滴 | 用核函数加权邻域粒子来近似连续场的微分算子 |
| **MPM** | Material Point Method (物质点法) | 大变形固体、沙/泥/胶、流固耦合 | 混合拉格朗日-欧拉方法，粒子携带物质状态，网格求解动量方程 |
| **DEM** | Discrete Element Method (离散元法) | 颗粒流、粉末沉积、碰撞接触 | 显式追踪每个颗粒的受力与运动，求解牛顿方程 |

**核心痛点**：这些经典方法在精度上表现优异，但计算成本极高。以 MPM 模拟颗粒柱塌塌为例，56 核 CPU 集群需要约 2.5 小时，而工程优化/逆问题往往需要成千上万次前向仿真，传统方法完全无法满足实时性需求。

### 1.2 神经网络替代模型的兴起

2016 年，Battaglia 等人在 NeurIPS 发表了开创性的 **Interaction Networks** (交互网络) 论文，首次证明了神经网络可以学习物体间的物理交互并预测动力学轨迹。这篇论文奠定了"将粒子系统建模为图，用消息传递学习交互"的范式基础。

随后，这一领域经历了爆发式发展：

```
时间线:
2016  Interaction Networks (Battaglia et al., NeurIPS)         -- 开山之作
2019  DPI-Net (Li et al., ICLR)                               -- 动态粒子交互网络
2020  GNS (Sanchez-Gonzalez et al., ICML)                     -- DeepMind通用粒子仿真器
2021  MeshGraphNets (Pfaff et al., ICLR)                      -- 网格+图混合仿真
2022  SEGNN (Brandstetter et al., ICLR)                       -- E(3)等变消息传递
2022  DMCF (Prantl et al., NeurIPS)                           -- 动量守恒流体学习
2023  LagrangeBench (Toshev et al., NeurIPS)                  -- 拉格朗日流体基准
2024  NeuralMPM (Rochman-Sharabi et al., ICML→TMLR)           -- MPM启发的神经仿真
2024  Neural SPH (Toshev et al., ICML)                        -- SPH增强GNN
2025  Hybrid Neural-MPM (Xu et al.)                           -- 实时交互流体仿真
2025  Dynami-CAL GraphNet (Sharma & Fink, Nature Comms)       -- 守恒律GNN
2025  TGNNS (时序图网络仿真器)                                 -- 100x加速DEM/MPM
```

**核心驱动力**：GNN 替代模型相比经典方法可实现 **100x-5000x 加速**，同时保持 ~5% 以内的误差，使得实时仿真、逆问题求解和大规模优化成为可能。

---

## 2. GNN 仿真核心原理

### 2.1 粒子系统 → 图表示

GNN 粒子仿真的核心洞察：**粒子系统天然映射为图结构**。

```
粒子系统                          图结构
-----------                       ---------
每个粒子  ←——对应——→  节点 (Node)
粒子间相互作用  ←——对应——→  边 (Edge)
粒子状态 (位置/速度/材料) ←——对应——→  节点特征
相对位置/距离  ←——对应——→  边特征
```

**图构建策略**：
- **KNN图**：每个粒子连接到最近的 K 个邻居 (常用 K=10-20)
- **半径图**：连接半径 R 内的所有粒子 (类似SPH的支持域)
- **动态图**：每个时间步重新构建图 (因粒子运动导致拓扑变化)

### 2.2 Encode-Process-Decode 架构

几乎所有 GNN 粒子仿真器都遵循 DeepMind GNS 提出的 **Encode-Process-Decode** 三阶段架构：

```
                 ┌─────────────┐
    粒子状态 ──→ │  Encoder     │ ──→  潜在图 (Latent Graph)
    (位置,速度,  │  (MLP)       │      节点/边嵌入向量
     材料类型)   └─────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Processor   │ ──→  更新后的潜在图
                 │  (M轮消息    │      包含了交互信息
                 │   传递)      │
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Decoder     │ ──→  加速度预测
                 │  (MLP)       │
                 └─────────────┘
                        │
                        ▼
                 欧拉积分器更新位置/速度
```

### 2.3 消息传递机制 (Message Passing)

消息传递是 GNN 仿真的核心计算单元，每轮包含三步：

```python
# 伪代码: 单轮消息传递
for each edge (i, j) in graph:
    # 1. 消息计算: 基于两端节点和边特征生成消息
    message_ij = phi_e(node_i, node_j, edge_ij)

for each node i in graph:
    # 2. 消息聚合: 汇总所有邻居发来的消息
    aggregated_i = SUM(message_ij for j in neighbors(i))

    # 3. 节点更新: 用聚合消息更新节点状态
    node_i = phi_v(node_i, aggregated_i)
```

- `phi_e` 和 `phi_v` 是可学习的 MLP
- 通常堆叠 M=10-15 轮消息传递以传播长程信息
- 每轮感受野扩展一跳，M 轮后信息可传播 M 跳

### 2.4 训练策略

| 策略 | 说明 |
|------|------|
| **单步监督** | 模型预测下一步加速度，用欧拉积分更新位置，与真实位置计算MSE损失 |
| **噪声注入** | 训练时向输入位置添加噪声 (类似数据增强)，缓解长时回滚误差累积 |
| **多步回滚训练** | 展开多步预测计算损失，提升长期稳定性 (计算成本高) |
| **课程学习** | 从短回滚逐步增加到长回滚 |

### 2.5 物理归纳偏置 (Physics Inductive Biases)

现代方法越来越注重将物理先验嵌入网络架构中：

| 归纳偏置类型 | 代表方法 | 实现方式 |
|-------------|---------|---------|
| **平移/旋转等变性** | SEGNN | 使用 O(3) 不可约表示和 Clebsch-Gordan 张量积 |
| **动量守恒** | Dynami-CAL GraphNet, DMCF | 反对称核设计、边局部参考系 |
| **能量守恒** | Hamiltonian GNN, Lagrangian NN | 学习哈密顿量/拉格朗日量，通过梯度导出力 |
| **惯性参考系** | GNS (geoelements) | 引入简单物理偏置 (如重力加速度参考系) |
| **SPH物理约束** | Neural SPH | 嵌入压力项、粘性力项、外力项 |

---

## 3. 主要方法详解与对比

### 3.1 GNS (Graph Network-based Simulator)

- **论文**: "Learning to Simulate Complex Physics with Graph Networks" (Sanchez-Gonzalez et al., ICML 2020)
- **ArXiv**: https://arxiv.org/abs/2002.09405
- **代码**: https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate (TF) / https://github.com/geoelements/gns (PyTorch)
- **核心创新**:
  - 提出通用的 Encode-Process-Decode 架构，单一模型可仿真流体、沙、胶等多种材料
  - 证明了噪声注入训练策略对长程回滚的重要性
  - 训练时用数千粒子，测试时可泛化到 30 倍以上粒子数
- **精度**: 训练数据的回滚误差 ~5%，对 MPM/SPH 仿真结果的忠实度极高
- **速度**: PyTorch 实现 (geoelements/gns) 报告 **5000x 加速** (相对56核CPU的MPM)，单GPU 20秒 vs MPM 2.5小时
- **泛化性**: 可泛化到未见过的初始条件、32倍空间范围、30倍粒子数，保持5000步动力学合理性

### 3.2 DPI-Net (Dynamic Particle Interaction Networks)

- **论文**: "Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids" (Li et al., ICLR 2019)
- **ArXiv**: https://jiajunwu.com/papers/dpi_iclr.pdf
- **代码**: https://github.com/YunzhuLi/DPI-Net
- **项目页**: http://dpi.csail.mit.edu/
- **核心创新**:
  - 动态构建交互图 (不依赖固定拓扑)
  - 层次化图结构实现多尺度效应传播
  - 多步消息传递处理瞬时力传播
  - 支持从少量观测快速适应新环境
- **应用**: 刚体操纵、可变形物体、流体交互的学习与控制
- **改进版**: VGPL-Dynamics-Prior (2020)，长期预测更稳定

### 3.3 MeshGraphNets

- **论文**: "Learning Mesh-Based Simulation with Graph Networks" (Pfaff et al., ICLR 2021)
- **ArXiv**: https://arxiv.org/abs/2010.03409
- **代码**: https://github.com/echowve/meshGraphNets_pytorch (PyTorch复现)
- **核心创新**:
  - 将网格仿真问题映射到图上 (网格节点→图节点，网格边+世界空间边→图边)
  - 同时处理网格内和世界空间的远程交互
  - 支持自适应网格分辨率
- **精度/速度**: 比训练用的仿真器快 **1-2个数量级**
- **应用**: 空气动力学、结构力学、布料仿真

### 3.4 NeuralMPM

- **论文**: "A Neural Material Point Method for Particle-based Simulations" (Rochman-Sharabi et al., 2024)
- **ArXiv**: https://arxiv.org/abs/2408.15753
- **项目页**: https://neuralmpm.isach.be/
- **发表**: ICML 2024 AI4Science Workshop → TMLR 2025
- **核心创新**:
  - 受 MPM 启发，将粒子插值到固定网格上 → 用 U-Net 在网格上预测更新 → 插值回粒子
  - 完全绕过了 GNN 中昂贵的邻域搜索和动态图构建
  - 处理器对粒子数不变 (只处理体素化表示)
- **精度**: 与 GNS/DMCF 相当或更优的长程精度
- **速度**: 训练时间从 10天→15小时，内存消耗降 10x-100x，推理加速 5x-10x (相对GNS)
- **数据集**: WATERRAMPS, SANDRAMPS, GOOP, MULTIMATERIAL 等6个数据集

### 3.5 Hybrid Neural-MPM

- **论文**: "Hybrid Neural-MPM for Interactive Fluid Simulations in Real-Time" (Xu et al., 2025)
- **ArXiv**: https://arxiv.org/abs/2505.18926
- **核心创新**:
  - 混合策略：优先用 GNN 神经物理，复杂场景自动回退 MPM
  - 安全守卫机制 (safeguard condition) 抑制误差累积
  - 集成扩散模型控制器，支持用户手绘草图指定流体轨迹
  - 低时空分辨率构建 GNN，不显著降低精度
- **架构细节**: 10层GNN，128维潜空间，5步历史速度作节点特征
- **训练**: 1天 (单张4090 GPU)
- **应用前景**: 实时交互流体仿真，对喷涂路径规划有直接参考意义

### 3.6 SEGNN (Steerable E(3) Equivariant GNN)

- **论文**: "Geometric and Physical Quantities Improve E(3) Equivariant Message Passing" (Brandstetter et al., ICLR 2022 Spotlight)
- **ArXiv**: https://arxiv.org/abs/2110.02905
- **代码**: https://github.com/RobDHess/Steerable-E3-GNN
- **核心创新**:
  - 节点和边特征不再限于标量不变量，可以包含协变信息 (向量、张量)
  - 使用 O(3) 不可约表示构建可导向 MLP
  - Clebsch-Gordan 张量积替代标准线性变换
  - 门控非线性激活函数保持等变性
- **精度**: 在 QM9、Open Catalyst Project、N-body 系统上表现优异
- **意义**: 为粒子仿真提供了严格的旋转/平移等变性保证

### 3.7 Dynami-CAL GraphNet

- **论文**: "A Physics-Informed Graph Neural Network Conserving Linear and Angular Momentum for Dynamical Systems" (Sharma & Fink, Nature Communications 2025)
- **ArXiv**: https://arxiv.org/abs/2501.07373
- **Nature Comms**: https://www.nature.com/articles/s41467-025-67802-5
- **核心创新**:
  - 强制逐对守恒线动量和角动量 (不同于基于能量守恒的 Hamiltonian/Lagrangian 方法)
  - 边局部正交参考系 (edge-aligned orthonormal basis)，满足 SO(3) 等变性和 T(3) 不变性
  - 牛顿第三定律通过反对称设计硬编码到网络中
  - 同时预测内力和旋转力矩 (6自由度模型)
- **优势**: 即使在耗散/外力存在时仍能守恒动量 (能量方法在此场景失效)
- **精度**: 在 3D 非弹性碰撞颗粒系统上，单步和多步预测均优于 GMN/EGNN/ClofNet 等所有基线
- **数据效率**: 仅需 500 个训练样本即达到强劲性能

### 3.8 Neural SPH

- **论文**: "Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics" (Toshev et al., ICML 2024)
- **ArXiv**: https://arxiv.org/abs/2402.06275
- **代码**: https://github.com/tumaer/neuralsph
- **核心创新**:
  - 识别了 GNN 仿真器中的张力不稳定性导致的粒子聚集问题
  - 将标准 SPH 求解器的组件 (压力项、粘性项、外力项) 嵌入 GNN 训练和推理
  - "SPH增强"策略可应用于任意现有 GNN 仿真器
- **精度**: 比基线 GNN 提升数量级的回滚误差，显著延长稳定预测时间

### 3.9 DMCF (Conservation of Momentum for Fluid Learning)

- **论文**: "Guaranteed Conservation of Momentum for Learning Particle-based Fluid Dynamics" (Prantl et al., NeurIPS 2022)
- **代码**: https://github.com/tum-pbs/DMCF
- **核心创新**:
  - 反对称连续卷积核 (ASCC)，通过镜像+取反实现严格动量守恒
  - 层次化网络架构+精心设计的重采样方案
  - 时间一致性训练策略
- **泛化**: 可泛化到100万粒子的新场景

### 3.10 TGNNS (Temporal Graph Neural Network Simulator)

- **论文**: "A physical-information-flow-constrained temporal graph neural network-based simulator for granular materials" (CMAME, 2025)
- **核心创新**: 物理信息流约束的时序GNN，将DEM颗粒力学的物理先验编码到图的时序演化中
- **速度**: **~100x 加速** (相对最先进的GPU-based DEM/MPM)

---

## 3.11 综合对比表

| 方法 | 发表 | 表示方式 | 物理偏置 | 速度提升 | 核心优势 | 核心局限 |
|------|------|---------|---------|---------|---------|---------|
| **GNS** | ICML 2020 | 粒子→KNN图 | 噪声注入、惯性系 | 5000x vs MPM | 通用、成熟、易用 | 邻域搜索昂贵、长程误差累积 |
| **DPI-Net** | ICLR 2019 | 动态层次图 | 多步消息传递 | 10-100x | 多材料、可控制 | 规模有限 |
| **MeshGraphNets** | ICLR 2021 | 网格+世界边 | 自适应分辨率 | 10-100x | 网格精度 | 需要网格输入 |
| **NeuralMPM** | TMLR 2025 | 粒子→体素→粒子 | MPM的P2G/G2P | 5-10x vs GNS | 无需邻域搜索、训练快 | 分辨率受网格限制 |
| **Hybrid Neural-MPM** | 2025 | GNN+MPM混合 | MPM回退保证 | 实时 (>30fps) | 实时交互、误差可控 | 系统复杂度高 |
| **SEGNN** | ICLR 2022 | E(3)等变图 | 严格SO(3)等变 | 与GNS相当 | 物理对称性保证 | 计算开销略大 |
| **Dynami-CAL GraphNet** | Nat. Comms 2025 | 边局部参考系图 | 动量守恒硬约束 | 与GNS相当 | 守恒+耗散兼容 | 尚无公开代码 |
| **Neural SPH** | ICML 2024 | SPH增强GNN | SPH物理项 | 与GNS相当 | 长程稳定性大幅提升 | 需要SPH领域知识 |
| **DMCF** | NeurIPS 2022 | 连续卷积 | 反对称核守恒 | 与GNS相当 | 动量守恒 | 边界稳定性不足 |
| **TGNNS** | CMAME 2025 | 时序图 | DEM物理信息流 | 100x vs GPU-DEM | 颗粒系统高精度 | 限于颗粒流 |

---

## 4. 关键论文列表

### 4.1 奠基性论文

| # | 论文 | 作者 | 会议/期刊 | 年份 |
|---|------|------|----------|------|
| 1 | Interaction Networks for Learning about Objects, Relations and Physics | Battaglia et al. | NeurIPS | 2016 |
| 2 | Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids (DPI-Net) | Li et al. | ICLR | 2019 |
| 3 | Learning to Simulate Complex Physics with Graph Networks (GNS) | Sanchez-Gonzalez et al. | ICML | 2020 |
| 4 | Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets) | Pfaff et al. | ICLR | 2021 |

### 4.2 物理归纳偏置

| # | 论文 | 作者 | 会议/期刊 | 年份 |
|---|------|------|----------|------|
| 5 | Geometric and Physical Quantities Improve E(3) Equivariant Message Passing (SEGNN) | Brandstetter et al. | ICLR (Spotlight) | 2022 |
| 6 | Guaranteed Conservation of Momentum for Learning Particle-based Fluid Dynamics (DMCF) | Prantl et al. | NeurIPS | 2022 |
| 7 | Lagrangian Neural Networks | Cranmer et al. | ICLR Workshop | 2020 |
| 8 | Hamiltonian Graph Neural Networks | (多组) | Various | 2020-2024 |
| 9 | BroGNet: Momentum-Conserving GNNs for Brownian Dynamics | (Various) | ICLR | 2024 |
| 10 | Dynami-CAL GraphNet: Conservation of Linear and Angular Momentum | Sharma & Fink | Nature Communications | 2025 |

### 4.3 SPH/MPM 神经替代

| # | 论文 | 作者 | 会议/期刊 | 年份 |
|---|------|------|----------|------|
| 11 | Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics | Toshev et al. | ICML | 2024 |
| 12 | LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite | Toshev et al. | NeurIPS D&B | 2023 |
| 13 | A Neural Material Point Method for Particle-based Simulations (NeuralMPM) | Rochman-Sharabi et al. | TMLR | 2025 |
| 14 | Hybrid Neural-MPM for Interactive Fluid Simulations in Real-Time | Xu et al. | arXiv | 2025 |
| 15 | GNS: A Generalizable Graph Neural Network-based Simulator | Kumar et al. | JOSS / arXiv | 2022 |
| 16 | JAX-SPH: A Differentiable Smoothed Particle Hydrodynamics Framework | Toshev et al. | arXiv | 2024 |

### 4.4 层次化与可扩展

| # | 论文 | 作者 | 会议/期刊 | 年份 |
|---|------|------|----------|------|
| 17 | Scalable Graph Networks for Particle Simulations | Martinkus et al. | AAAI | 2021 |
| 18 | DHMP: Discovering Message Passing Hierarchies for Mesh-Based Physics | (Various) | ICLR submission | 2025 |

### 4.5 喷涂/热喷涂相关

| # | 论文 | 作者 | 期刊 | 年份 |
|---|------|------|------|------|
| 19 | Physics-Informed Neural Networks for Predicting Particle Properties in Plasma Spraying | (Various) | J. Thermal Spray Tech. | 2025 |
| 20 | Implementation of ANN for Forecasting HVOF Spray Process | (Various) | J. Thermal Spray Tech. | 2021 |

---

## 5. 与喷涂粒子动力学建模的对接方案

### 5.1 喷涂仿真的物理特征分析

喷涂过程涉及的物理现象：

```
喷枪 → 高速气流携带涂料液滴/粉末 → 飞行中的液滴动力学 → 撞击基材 → 铺展/固化

关键物理过程:
1. 液滴雾化 (Atomization): 液膜破碎为离散液滴
2. 液滴输运 (Transport): 气-液两相流中液滴的加速/减速/蒸发
3. 液滴碰撞 (Collision): 液滴间合并/分裂
4. 液滴-壁面撞击 (Impact): 铺展、飞溅、沉积
5. 涂膜形成 (Film Formation): 液滴叠加形成连续涂膜
```

### 5.2 GNN方法的对接策略

#### 策略一：基于 GNS 的端到端液滴动力学学习

**适用场景**: 液滴飞行轨迹预测、喷涂锥形区域内的粒子分布预测

```
实施路径:
1. 用经典 CFD/SPH 仿真生成液滴轨迹数据集
   - 工具: OpenFOAM (Euler-Lagrange模型) 或 DualSPHysics
   - 数据: 液滴位置、速度、直径随时间的变化
2. 将液滴转化为图节点 (含材料属性: 表面张力、粘度、密度)
3. 构建邻域图 (半径连接，反映液滴间的气动耦合)
4. 用 GNS 框架训练 → 预测液滴加速度
5. 回滚生成完整喷涂模式

推荐代码基础: geoelements/gns (PyTorch, MIT License)
预期加速: 100x-1000x
```

#### 策略二：Hybrid Neural-MPM 实时喷涂模拟

**适用场景**: 喷涂路径在线优化、交互式路径规划

```
实施路径:
1. 用 MPM 仿真生成涂料流体的训练数据
2. 训练 GNN 替代器 (低分辨率即可)
3. 设置安全守卫条件: 当液滴浓度/速度超出训练范围时回退到 MPM
4. 集成到路径规划循环中: 每条候选路径用神经仿真快速评估
5. 选出最优路径后再用高精度 MPM 验证

优势: 实时反馈、物理保证、支持人机交互
```

#### 策略三：NeuralMPM 大规模喷涂膜厚预测

**适用场景**: 全工件表面的涂膜厚度分布预测

```
实施路径:
1. NeuralMPM 的体素化表示天然适合涂膜厚度 (类似高度场)
2. 将喷涂液滴 P2G 到工件表面网格上
3. U-Net 预测涂膜厚度随时间的演化
4. 大规模预测不受粒子数限制

优势: 训练快 (15小时 vs 10天)、内存友好、推理快
```

#### 策略四：Dynami-CAL GraphNet 守恒喷涂粒子模型

**适用场景**: 高精度喷涂液滴碰撞动力学 (含耗散)

```
实施路径:
1. 喷涂过程中液滴碰撞是耗散过程 → 能量不守恒但动量守恒
2. Dynami-CAL GraphNet 的动量守恒架构完美匹配此物理特征
3. 可准确预测液滴合并/分裂后的速度分布
4. 6DOF 模型可捕捉液滴旋转

优势: 物理一致性强、小数据集即可训练、可泛化
```

### 5.3 推荐的技术路线图

```
Phase 1 (1-2个月): 基础验证
├── 安装 geoelements/gns
├── 用内置数据集 (WaterDrop/Sand) 训练和评估
├── 理解 Encode-Process-Decode 架构
└── 生成简单喷涂数据集 (2D, 1000粒子)

Phase 2 (2-4个月): 喷涂场景适配
├── 用 OpenFOAM 或 Taichi MPM 生成喷涂液滴数据
├── 设计喷涂特定的节点/边特征 (液滴直径、Weber数、Oh数)
├── 添加气流场作为外力条件
├── 训练并评估回滚精度
└── 对比经典仿真器的速度和精度

Phase 3 (4-6个月): 集成与优化
├── 接入路径规划模块 (逆问题/优化循环)
├── 实现 Hybrid Neural-MPM 混合策略
├── 添加涂膜厚度预测 (NeuralMPM 或后处理)
├── 考虑加入 Neural SPH 的物理增强
└── 3D扩展和大规模测试

Phase 4 (6-12个月): 高级应用
├── Dynami-CAL 式守恒律约束 (液滴碰撞场景)
├── 实时交互演示系统
├── 可微分仿真器用于梯度优化路径
└── 发表成果
```

---

## 6. 开源代码资源汇总

### 6.1 核心仿真框架

| 项目 | 语言/框架 | 许可证 | Stars | 链接 |
|------|----------|--------|-------|------|
| **geoelements/gns** | PyTorch + PyG | MIT | ~500 | https://github.com/geoelements/gns |
| **GNS-PyTorch** (DeepMind复现) | PyTorch | - | ~200 | https://github.com/zhouxian/GNS-PyTorch |
| **DeepMind原版** | TensorFlow | Apache 2.0 | - | https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate |
| **DPI-Net** | PyTorch | - | ~300 | https://github.com/YunzhuLi/DPI-Net |
| **MeshGraphNets (PyTorch)** | PyTorch + PyG | - | ~200 | https://github.com/echowve/meshGraphNets_pytorch |
| **SEGNN** | PyTorch + e3nn | - | ~200 | https://github.com/RobDHess/Steerable-E3-GNN |
| **DMCF** | PyTorch | - | ~100 | https://github.com/tum-pbs/DMCF |
| **Scalable GNNs** | PyTorch | - | ~50 | https://github.com/KarolisMart/scalable-gnns |

### 6.2 SPH/MPM 神经替代

| 项目 | 语言/框架 | 链接 |
|------|----------|------|
| **Neural SPH** | JAX + JAX-MD | https://github.com/tumaer/neuralsph |
| **LagrangeBench** | JAX + PyG | https://github.com/tumaer/lagrangebench |
| **JAX-SPH** | JAX | https://arxiv.org/abs/2403.04750 |
| **NeuralMPM** | PyTorch + PyG | https://neuralmpm.isach.be/ |

### 6.3 经典仿真器 (用于生成训练数据)

| 项目 | 方法 | 链接 |
|------|------|------|
| **Taichi MPM** | MPM (GPU加速) | https://github.com/taichi-dev/taichi |
| **CB-Geo MPM** | MPM (HPC) | https://github.com/cb-geo/mpm |
| **DualSPHysics** | SPH (GPU) | https://github.com/DualSPHysics/DualSPHysics |
| **SPlisHSPlasH** | SPH (多方法) | https://github.com/InteractiveComputerGraphics/SPlisHSPlasH |
| **LIGGGHTS** | DEM | https://www.cfdem.com/liggghts-open-source-discrete-element-method-particle-simulation-code |

### 6.4 辅助工具库

| 项目 | 用途 | 链接 |
|------|------|------|
| **PyTorch Geometric (PyG)** | 图神经网络库 | https://github.com/pyg-team/pytorch_geometric |
| **e3nn** | E(3)等变网络库 | https://github.com/e3nn/e3nn |
| **JAX-MD** | 可微分分子动力学 | https://github.com/jax-md/jax-md |
| **torch-cluster** | 高效邻域搜索 | (PyG子包) |

---

## 7. 推荐入门路线

### 7.1 零基础路线 (无GNN经验)

```
Week 1-2: 图神经网络基础
├── Stanford CS224W 课程 (免费): https://web.stanford.edu/class/cs224w/
├── 重点学习: 消息传递、GCN、GraphSAGE、GAT
├── PyTorch Geometric 官方教程
└── 动手: 用PyG实现简单的节点分类

Week 3-4: 物理仿真基础
├── 阅读SPH/MPM科普材料 (如Taichi文档中的MPM教程)
├── 运行Taichi的水滴下落示例
├── 理解粒子仿真的基本循环: 力计算→积分→位置更新
└── 生成简单的粒子轨迹数据

Week 5-6: GNN粒子仿真入门
├── 精读GNS论文 (ICML 2020)
├── 克隆 geoelements/gns，运行WaterDrop数据集
├── 逐行理解Encode-Process-Decode代码
├── 修改超参数观察效果 (消息传递轮数、隐藏维度)
└── 阅读Stanford CS224W的GNS教程博客

Week 7-8: 进阶方法
├── 阅读NeuralMPM和Neural SPH论文
├── 运行LagrangeBench基准 (含多种方法对比)
├── 理解物理归纳偏置 (等变性、守恒律)
└── 选择与自身研究最相关的方法深入
```

### 7.2 有GNN/ML经验的快速路线

```
Day 1-3: 核心论文精读
├── GNS (ICML 2020) - 架构和训练策略
├── NeuralMPM (TMLR 2025) - 替代GNN的体素化方案
└── Dynami-CAL GraphNet (Nat. Comms 2025) - 守恒律嵌入

Day 4-7: 代码实践
├── geoelements/gns: 训练+评估+可视化
├── LagrangeBench: 多方法横向对比
└── 选一个方法深入修改

Week 2-3: 自定义数据集
├── 用现有仿真器生成喷涂相关数据
├── 设计节点/边特征编码方案
├── 训练、调参、评估回滚性能
└── 与经典仿真器对比速度/精度
```

### 7.3 核心阅读材料优先级

| 优先级 | 材料 | 原因 |
|--------|------|------|
| P0 (必读) | GNS 论文 + geoelements/gns 代码 | 领域标准框架，代码质量高 |
| P0 (必读) | LagrangeBench 论文 + 代码 | 唯一的标准化基准，含多种方法对比 |
| P1 (强烈推荐) | NeuralMPM 论文 | 训练效率革命性提升，实用性强 |
| P1 (强烈推荐) | Neural SPH 论文 + 代码 | 揭示了GNN仿真器的关键问题和解法 |
| P2 (推荐) | Hybrid Neural-MPM | 实时仿真方案，对路径规划有直接价值 |
| P2 (推荐) | Dynami-CAL GraphNet | 最新的守恒律方法，Nature Comms背书 |
| P3 (参考) | SEGNN, DMCF, DPI-Net | 理解等变性和守恒的不同实现路径 |

---

## 附录：常见问题

### Q1: GNN仿真器能否处理喷涂中的多相流 (气-液)?

可以。GNS框架支持多种材料类型 (通过 particle_type 编码)。需要在节点特征中加入材料标识 (气体粒子 vs 液滴粒子 vs 壁面粒子)。NeuralMPM 的 MULTIMATERIAL 数据集已验证了多材料混合仿真能力。

### Q2: 如何处理喷涂中的边界条件 (喷嘴入口、工件壁面)?

GNS 中通过 particle_type 区分内部粒子和边界粒子。对于喷嘴入口可设计时变的源项 (每步注入新粒子)。工件壁面可以用固定的边界粒子表示。Hybrid Neural-MPM 支持用户交互式添加障碍物。

### Q3: 训练数据需要多少?

GNS 通常需要 20-100 条不同初始条件的轨迹，每条轨迹包含几百到几千个时间步。Dynami-CAL GraphNet 展示了仅需 500 个训练样本即达到强劲性能。NeuralMPM 的训练效率更高 (15小时 vs 10天)。

### Q4: 推荐的GPU配置?

- 入门/原型验证: 单张 NVIDIA RTX 4090 (24GB) 即可
- 大规模训练: NVIDIA A100 (40/80GB)
- geoelements/gns 支持多GPU分布式训练
- NeuralMPM 的内存需求比 GNS 低 10-100 倍

### Q5: 可微分性对路径规划的意义?

GNN仿真器天然可微分 (PyTorch autograd)。这意味着可以直接通过仿真器反向传播梯度到喷涂参数 (喷嘴位置、速度、角度)，实现基于梯度的路径优化，而非昂贵的无梯度优化 (如遗传算法)。JAX-SPH 和 LagrangeBench 已验证了可微分仿真的可行性。

---

> **引用说明**: 本研究笔记基于对以下来源的综合调研:
> - [Learning to Simulate Complex Physics with Graph Networks (ICML 2020)](https://arxiv.org/abs/2002.09405)
> - [GNS: A generalizable GNN-based simulator (geoelements)](https://github.com/geoelements/gns)
> - [DPI-Net (ICLR 2019)](https://github.com/YunzhuLi/DPI-Net)
> - [MeshGraphNets (ICLR 2021)](https://arxiv.org/abs/2010.03409)
> - [NeuralMPM (TMLR 2025)](https://arxiv.org/abs/2408.15753)
> - [Hybrid Neural-MPM (2025)](https://arxiv.org/abs/2505.18926)
> - [SEGNN (ICLR 2022)](https://arxiv.org/abs/2110.02905)
> - [Dynami-CAL GraphNet (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-67802-5)
> - [Neural SPH (ICML 2024)](https://arxiv.org/abs/2402.06275)
> - [DMCF (NeurIPS 2022)](https://github.com/tum-pbs/DMCF)
> - [LagrangeBench (NeurIPS 2023)](https://github.com/tumaer/lagrangebench)
> - [Interaction Networks (NeurIPS 2016)](https://arxiv.org/abs/1612.00222)
> - [JAX-SPH (2024)](https://arxiv.org/abs/2403.04750)
> - [PINNs for Plasma Spraying (J. Thermal Spray Tech. 2025)](https://link.springer.com/article/10.1007/s11666-025-01965-x)
> - [TGNNS for Granular Materials (CMAME 2025)](https://www.sciencedirect.com/science/article/pii/S0045782524007904)
