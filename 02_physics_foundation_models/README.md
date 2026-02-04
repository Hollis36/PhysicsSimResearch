# Physics Foundation Models -- 深度调研报告

> 调研日期: 2026-02-04
> 面向研究者: 喷涂路径规划 (MFARainbowNet / Rainbow DQN) + RGB-IR 多模态检测方向
> 目标: 全面梳理物理基础模型 (Physics Foundation Models, PFMs) 领域前沿, 评估与喷涂物理仿真对接的可行性

---

## 一、领域概述与发展脉络

### 1.1 什么是物理基础模型 (Physics Foundation Model)?

物理基础模型借鉴大语言模型 (LLM) 的 **"训练一次, 到处部署" (train once, deploy anywhere)** 范式, 旨在构建统一的神经网络架构, 能够:

1. 在多种物理系统 (流体、固体、热传导、电磁、等离子体等) 的仿真数据上进行大规模预训练
2. 通过微调 (fine-tuning) 或零样本推理 (zero-shot) 快速适配未见过的物理场景
3. 替代传统数值求解器 (CFD/FEM), 实现数个数量级的加速

核心思想: 不同物理系统的偏微分方程 (PDE) 虽然形式各异, 但底层存在共享的数学结构 (对流、扩散、非线性耦合). 基础模型可以学习这些跨域通用的物理表征.

### 1.2 发展脉络

```
2020-2021  Neural Operator 奠基
           - FNO (Fourier Neural Operator, Li et al. 2020)
           - DeepONet (Lu et al. 2021)
           - Physics-Informed Neural Networks (PINNs, Raissi et al.)
               |
2022-2023  天气/气候领域率先突破 -- 单一物理系统的大模型
           - FourCastNet (NVIDIA, 2022)
           - Pangu-Weather (Huawei, Nature 2023)
           - GraphCast (Google DeepMind, Science 2023)
           - ClimaX (Microsoft, ICML 2023)
               |
2024       多物理预训练与基准涌现
           - MPP (Polymathic AI, NeurIPS 2024) -- 首个跨PDE类型预训练
           - Poseidon (ETH Zurich, NeurIPS 2024) -- PDE基础模型
           - DPOT (Tsinghua, ICML 2024) -- 去噪算子Transformer
           - UPT (JKU, NeurIPS 2024) -- 统一物理Transformer
           - The Well (Polymathic AI, NeurIPS 2024) -- 15TB物理仿真基准
           - GenCast (Google DeepMind, Nature 2024) -- 概率天气模型
           - Aurora (Microsoft, Nature 2025) -- 大气基础模型
           - PDEformer (Peking Univ, ICLR 2024 Workshop)
           - VICON (UCLA, 2024) -- 视觉上下文算子网络
               |
2025-2026  走向统一物理基础模型
           - GPhyT (General Physics Transformer, 2025) -- 1.8TB, 零样本泛化
           - PhysiX (UCLA, 2025) -- 4.5B参数, 离散Tokenizer
           - Walrus (Polymathic AI, 2025) -- 1.3B跨域连续体动力学
           - AION-1 (Polymathic AI, NeurIPS 2025) -- 天文多模态
           - CompNO (2026) -- 组合式算子基础模型
```

### 1.3 与传统方法的关系

| 方法类别 | 代表 | 优势 | 局限 |
|---------|------|------|------|
| 传统数值方法 (CFD/FEM) | OpenFOAM, ANSYS | 物理精度高, 可解释 | 计算昂贵, 单次求解 |
| Physics-Informed NN (PINN) | 嵌入PDE约束的网络 | 无需标注数据 | 训练困难, 难以泛化 |
| Neural Operator (单任务) | FNO, DeepONet | 快速推理 | 一个方程训一个模型 |
| **Physics Foundation Model** | GPhyT, Walrus, PhysiX | 跨域迁移, 一次训练多次部署 | 仍处于研究早期, 精度差距 |

---

## 二、主要模型详细对比

### 2.1 综合对比表

| 模型 | 机构 | 年份 | 参数量 | 训练数据 | 核心创新 | 论文 | 代码 | 开源 |
|------|------|------|--------|----------|----------|------|------|------|
| **GPhyT** | 独立研究 | 2025.09 | 9.2M-796M | 1.8TB (The Well + 自定义) | Transformer + 数值积分器混合架构; 零样本泛化 | [arXiv:2509.13805](https://arxiv.org/abs/2509.13805) | [GitHub](https://github.com/FloWsnr/General-Physics-Transformer) | 是 |
| **PhysiX** | UCLA | 2025.06 | **4.5B** | The Well (8个2D数据集) + 视频预训练 | 离散Tokenizer + 自回归 + 精炼模块; 从视频迁移知识 | [arXiv:2506.17774](https://arxiv.org/abs/2506.17774) | [GitHub](https://github.com/ArshKA/PhysiX) | 是 |
| **Walrus** | Polymathic AI | 2025.11 | **1.3B** | 15TB, 19个场景, 63个物理场 | 随机压缩(Jittering)抑制误差; 自适应Patching; 2D/3D统一 | [arXiv:2511.15684](https://arxiv.org/abs/2511.15684) | [GitHub](https://github.com/PolymathicAI/walrus) | 是 (MIT) |
| **AION-1** | Polymathic AI | 2025 | - | 200M+观测, ~100TB | 天文多模态 (39种模态); 掩码建模 | [polymathic-ai.org](https://polymathic-ai.org/blog/aion-1/) | - | - |
| **Poseidon** | ETH Zurich | 2024.05 | 多尺度 (T/S/B/L) | 流体动力学PDE | 多尺度算子Transformer; 时间条件LayerNorm; 半群性质扩充数据 | [arXiv:2405.19101](https://arxiv.org/abs/2405.19101) | [GitHub](https://github.com/camlab-ethz/poseidon) | 是 |
| **MPP** | Polymathic AI | 2024 (NeurIPS) | 多尺度 | PDEBench (NS, SW, DR等) | 共享嵌入 + 轴注意力; 首个跨PDE预训练 | [arXiv:2310.02994](https://arxiv.org/abs/2310.02994) | [GitHub](https://github.com/PolymathicAI/multiple_physics_pretraining) | 是 |
| **DPOT** | 清华大学 | 2024 (ICML) | 7M-1B | PDEBench 多数据集 | 去噪预训练 + 傅里叶注意力 | [arXiv:2403.03542](https://arxiv.org/abs/2403.03542) | [GitHub](https://github.com/thu-ml/DPOT) | 是 |
| **UPT** | JKU Linz | 2024 (NeurIPS) | - | 网格/粒子模拟 | 无网格/粒子的压缩潜空间; 可扩展至4.2M输入点 | [arXiv:2402.12365](https://arxiv.org/abs/2402.12365) | [项目页](https://ml-jku.github.io/UPT/) | 是 |
| **VICON** | UCLA | 2024.11 | - | 流体动力学 | 视觉上下文算子; 2D PDE上下文学习; 灵活时间步 | [arXiv:2411.16063](https://arxiv.org/abs/2411.16063) | [GitHub](https://github.com/Eydcao/VICON) | 是 |
| **PDEformer** | 北京大学 | 2024 | - | 3M样本 (1D PDE) | 计算图编码PDE符号+数值; Graph Transformer + INR | [arXiv:2402.12652](https://arxiv.org/abs/2402.12652) | - | - |
| **CompNO** | - | 2026.01 | - | - | 组合式: Foundation Block + Adaptation Block; 模块化算子 | [MDPI](https://www.mdpi.com/2076-3417/16/2/972) | - | - |

### 2.2 天气/气候物理模型对比

| 模型 | 机构 | 年份 | 参数量 | 架构 | 分辨率 | 训练数据 | 推理速度 | 论文 | 代码 |
|------|------|------|--------|------|--------|----------|----------|------|------|
| **FourCastNet** | NVIDIA | 2022-2025 | - | AFNO/SFNO/球面卷积(V3) | 0.25deg | ERA5 (~5TB) | 2秒/周 | [Blog](https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/) | [GitHub](https://github.com/NVlabs/FourCastNet) |
| **Pangu-Weather** | 华为 | 2023 | - | 3D Earth-Specific Transformer | 0.25deg, 13层 | ERA5 (1979-2021) | 1.4秒/24h | [Nature 619:533](https://www.nature.com/articles/s41586-023-06185-3) | [GitHub](https://github.com/198808xc/Pangu-Weather) |
| **GraphCast** | Google DeepMind | 2023 | **37M** | 消息传递GNN (编码器-处理器-解码器) | 0.25deg, 37层 | ERA5 (1979-2017) | <1分钟/10天 | [Science 382:1416](https://www.science.org/doi/10.1126/science.adi2336) | [GitHub](https://github.com/google-deepmind/graphcast) |
| **GenCast** | Google DeepMind | 2024 | - | 扩散模型 + 图Transformer | 0.25deg, 13层 | ERA5 (1979-2018) | 8分钟/15天集合 | [Nature (2024)](https://www.nature.com/articles/s41586-024-08252-9) | [GitHub](https://github.com/google-deepmind/graphcast) |
| **ClimaX** | Microsoft | 2023 | ~数百万 | ViT + 变量Tokenization | 1.4-5.6deg | CMIP6 | - | [arXiv:2301.10343](https://arxiv.org/abs/2301.10343) | - |
| **Aurora** | Microsoft | 2025 | **1.3B** | 3D Swin Transformer U-Net + Perceiver | **0.1deg** | 1M+小时 | <1分钟/10天 | [Nature (2025)](https://www.nature.com/articles/s41586-025-09005-y) | [GitHub](https://github.com/microsoft/aurora) |

**性能排名 (2024评估, 东亚/西太平洋):** FengWu > FuXi > GraphCast > Pangu-Weather > FourCastNet V2

**关键发现:**
- 所有AI天气模型在极端事件上都存在 **"模糊化" (blurring)** 问题, 即倾向于平均化而低估极端值
- GenCast 在97.2%的变量/时间组合上超越ECMWF ENS集合预报
- Aurora 在0.1度分辨率下计算加速 ~5000x, 是目前综合最强的大气基础模型

---

## 三、关键技术深度解析

### 3.1 GPhyT -- 通用物理Transformer

**核心架构:**
```
输入: 4D物理场 (时间 x 高度 x 宽度 x 物理量)
  |
  v
[线性Tokenization] --> 非重叠时空管道(tubelet) patch + 绝对位置编码
  |
  v
[时空Transformer] --> 统一注意力 (非分解式, 保证对湍流/激波的表达力)
  |
  v
[神经微分器] --> 学习 dX/dt (时间导数)
  |
  v
[数值积分器] --> Forward Euler 外推未来状态
```

**关键设计决策:**
- 选择统一注意力而非分解注意力: 虽然计算代价更高, 但能捕获湍流、激波交互等复杂非可分现象
- Forward Euler 积分器: 消融实验显示 RK4 等高阶方法无显著精度提升
- 混合架构 (学习微分 + 数值积分): 相比纯端到端预测, 误差累积近似线性增长

**关键结果:**
- GPhyT-M vs UNet: MSE 降低 **5x**
- GPhyT-M vs FNO: MSE 降低 **29x**
- 零样本泛化: 可对完全未见的物理系统进行上下文学习推理

### 3.2 PhysiX -- 离散化物理Token

**核心思路: 将物理场当作"视频"处理**

```
物理仿真快照序列 --> [NVIDIA Cosmos 离散Tokenizer (DV8x16x16)]
                              |
                              v
                     离散Token序列 --> [自回归Transformer (4.5B)] --> 下一Token预测
                              |
                              v
                     [精炼模块 (Refinement)] --> 校正离散化舍入误差
```

**核心创新:**
1. **离散化 + 自回归**: 首次将 LLM 的 next-token prediction 成功迁移到物理仿真
2. **视频知识迁移**: 从自然视频预训练的Tokenizer迁移到物理仿真, 有效解决物理数据稀缺问题
3. **精炼模块**: 补偿从连续物理场到离散Token的精度损失
4. **多任务协同**: 8个不同物理任务联合训练, 性能优于单独训练

**在 The Well 基准上超越所有专用基线模型和先前SOTA.**

### 3.3 Walrus -- 跨域连续体动力学基础模型

**覆盖域:** 声学、经典流体、非牛顿流、等离子体、活性物质、天体物理 (中子星合并、超新星等)

**关键技术:**
1. **随机压缩 (Patch Jittering)**: 在下采样前对数据施加随机平移, 打破固定网格模式, 抑制长期误差累积
2. **计算自适应Patching (CSM)**: 对不同分辨率数据自动调节压缩级别 -- 低分辨率少压缩, 高分辨率多压缩
3. **2D/3D统一处理**: 将2D数据嵌入3D空间 (类似"把纸放入薄盒子"), 用张量律感知变换做数据增强
4. **非对称归一化**: 输入按RMS归一化, 预测的增量 Delta u 按 Delta 的 RMS 反归一化
5. **因果注意力 + T5相对位置编码**: 时间轴因果, 空间轴双向

**可操控物理表征 (Steerable Representations):**
- Walrus 学到了可操控的物理表征: 可以沿特定方向操控涡度(创建/移除旋转结构)、扩散(锐化/模糊界面)、时间速度等
- 这些表征在完全无关的物理系统间迁移 -- 从流体流动到化学反应

### 3.4 Poseidon -- PDE基础模型

**架构: 多尺度算子Transformer**

```
输入 PDE 快照 --> [多尺度编码] --> [Transformer + 时间条件LayerNorm] --> [多尺度解码] --> 预测
```

**核心创新:**
1. **时间条件Layer Norm**: 实现连续时间评估, 不限于离散时间步
2. **半群性质数据扩充**: 利用 S(t+s) = S(t) . S(s) 将长时间演化分解为短步序列, 大幅扩充训练数据
3. **极高样本效率**: Poseidon 仅需 **20个样本** 即可达到 FNO 用 **1024个样本** 的误差水平

**在15个下游任务中的14个上取得最佳性能, 包含大量预训练未见过的PDE类型.**

### 3.5 The Well 基准

**规模:** 15TB, 16个数据集

**覆盖域:**
- 生物系统
- 流体动力学 (可压缩/不可压缩 Navier-Stokes)
- 声学散射
- 磁流体力学
- 河外天体流体
- 超新星爆炸

**数据格式:**
- 均匀网格, 等时间间隔
- HDF5 格式, 自描述
- 形状: `(n_traj, n_steps, coord1, coord2, (coord3))`, fp32
- 区分标量/矢量/张量场
- 训练/测试/验证 = 80/10/10

**评估指标:**
- 逐场 VRMSE (Volume-averaged RMSE)
- 多步 rollout 误差
- 长期稳定性

**意义:** The Well 是物理AI领域的 "ImageNet", 为不同模型提供统一可比的评估框架.

**仓库:** [github.com/PolymathicAI/the_well](https://github.com/PolymathicAI/the_well)

---

## 四、关键论文列表

### 4.1 物理基础模型 (核心)

| # | 论文 | 作者 | 发表 | arXiv/DOI |
|---|------|------|------|-----------|
| 1 | Towards a Physics Foundation Model (GPhyT) | Wiesner et al. | 预印本 2025 | [2509.13805](https://arxiv.org/abs/2509.13805) |
| 2 | PhysiX: A Foundation Model for Physics Simulations | Nguyen, Koneru, Li, Grover | 预印本 2025 | [2506.17774](https://arxiv.org/abs/2506.17774) |
| 3 | Walrus: A Cross-Domain Foundation Model for Continuum Dynamics | Polymathic AI | 预印本 2025 | [2511.15684](https://arxiv.org/abs/2511.15684) |
| 4 | Poseidon: Efficient Foundation Models for PDEs | Herde et al. | NeurIPS 2024 | [2405.19101](https://arxiv.org/abs/2405.19101) |
| 5 | Multiple Physics Pretraining (MPP) | McCabe et al. | NeurIPS 2024 | [2310.02994](https://arxiv.org/abs/2310.02994) |
| 6 | DPOT: Auto-Regressive Denoising Operator Transformer | Hao et al. | ICML 2024 | [2403.03542](https://arxiv.org/abs/2403.03542) |
| 7 | Universal Physics Transformers (UPT) | Alkin et al. | NeurIPS 2024 | [2402.12365](https://arxiv.org/abs/2402.12365) |
| 8 | VICON: Vision In-Context Operator Networks | Cao et al. | 预印本 2024 | [2411.16063](https://arxiv.org/abs/2411.16063) |
| 9 | PDEformer: Towards a Foundation Model for PDEs | Ye et al. | ICLR 2024 WS | [2402.12652](https://arxiv.org/abs/2402.12652) |
| 10 | Towards Physics-Guided Foundation Models | - | 预印本 2025 | [2502.15013](https://arxiv.org/abs/2502.15013) |
| 11 | Large Physics Models (LPMs) | - | Eur. Phys. J. C 2025 | [Springer](https://link.springer.com/article/10.1140/epjc/s10052-025-14707-8) |
| 12 | CompNO: Compositional Neural Operators | - | Appl. Sci. 2026 | [MDPI](https://www.mdpi.com/2076-3417/16/2/972) |

### 4.2 天气/气候模型

| # | 论文 | 作者 | 发表 | arXiv/DOI |
|---|------|------|------|-----------|
| 13 | GraphCast: Learning Skillful Medium-Range Weather Forecasting | Lam et al. | Science 2023 | [10.1126/science.adi2336](https://www.science.org/doi/10.1126/science.adi2336) |
| 14 | Pangu-Weather: Accurate Medium-Range Forecasting with 3D NN | Bi et al. | Nature 2023 | Nature 619:533 |
| 15 | FourCastNet | Pathak et al. | 2022 | [NVlabs/FourCastNet](https://github.com/NVlabs/FourCastNet) |
| 16 | GenCast: Probabilistic Weather Forecasting | Price et al. | Nature 2024 | [Nature](https://www.nature.com/articles/s41586-024-08252-9) |
| 17 | ClimaX: A Foundation Model for Weather and Climate | Nguyen et al. | ICML 2023 | [2301.10343](https://arxiv.org/abs/2301.10343) |
| 18 | Aurora: A Foundation Model of the Atmosphere | Bodnar et al. | Nature 2025 | [Nature](https://www.nature.com/articles/s41586-025-09005-y) |

### 4.3 基准与基础工具

| # | 论文 | 作者 | 发表 | arXiv/DOI |
|---|------|------|------|-----------|
| 19 | The Well: Large-Scale Physics Simulations for ML | Ohana, McCabe et al. | NeurIPS 2024 D&B | [2412.00568](https://arxiv.org/abs/2412.00568) |
| 20 | Fourier Neural Operator (FNO) | Li et al. | ICLR 2021 | [2010.08895](https://arxiv.org/abs/2010.08895) |
| 21 | DeepONet | Lu et al. | Nature Machine Intelligence 2021 | - |

### 4.4 喷涂相关

| # | 论文 | 作者 | 发表 | 链接 |
|---|------|------|------|------|
| 22 | PINNs for Predicting Particle Properties in Plasma Spraying | - | J. Thermal Spray Tech. 2025 | [Springer](https://link.springer.com/article/10.1007/s11666-025-01965-x) |
| 23 | Cold Spray Nozzle Optimization via CFD + Neural Networks | - | J. Thermal Spray Tech. 2024 | [Springer](https://link.springer.com/article/10.1007/s11666-024-01716-4) |

---

## 五、与喷涂物理仿真的对接思路

### 5.1 喷涂过程的物理本质

喷涂涂装是一个典型的 **多物理场耦合** 问题:

```
                     [喷枪气流场]
                    (可压缩 Navier-Stokes)
                          |
            +-------------+-------------+
            |                           |
    [涂料液滴动力学]            [涂层沉积与成膜]
   (两相流/粒子追踪)          (自由表面流/固化)
            |                           |
    [热传递与蒸发]              [基底传热]
   (对流-扩散-相变)            (固体热传导)
```

涉及的PDE类型:
1. **可压缩/不可压缩 Navier-Stokes** -- 喷枪气流与涂料两相流
2. **对流-扩散方程** -- 热量传递、溶剂蒸发
3. **粒子追踪方程** -- 液滴飞行轨迹
4. **自由表面方程** -- 液膜铺展与流挂
5. **固化动力学** -- 涂层成膜过程

### 5.2 对接策略

#### 策略A: 直接微调预训练模型 (推荐起步方案)

```
[Walrus / GPhyT 预训练权重]
       |
       v
  [喷涂CFD仿真数据集] --> 微调 (Fine-tuning)
       |
       v
  [喷涂物理代理模型 (Surrogate)]
       |
       v
  [替代OpenFOAM, 加速1000x+]
       |
       v
  [接入 Rainbow DQN 路径规划]
```

**为什么可行:**
- Walrus 已经在可压缩/不可压缩 NS, 扩散, 对流等场景上预训练, 覆盖了喷涂的主要物理
- 20个样本 (Poseidon) 到几百个样本即可适配新物理, 大幅降低 CFD 数据需求
- Polymathic AI 的可操控表征 (涡度、扩散等) 与喷涂中的涡旋和液膜扩散高度相关

**具体步骤:**
1. 用 OpenFOAM / ANSYS Fluent 生成 500-2000 组不同喷涂参数的仿真数据
2. 将数据转换为 The Well 格式 (HDF5, 均匀网格)
3. 加载 Walrus 预训练权重, 微调 50-100 epochs
4. 验证: 对比微调模型 vs CFD 在新参数下的预测精度
5. 将代理模型嵌入 Rainbow DQN 环境, 替代在线CFD求解

#### 策略B: 用 PhysiX 的 Tokenizer 路线

```
[喷涂仿真视频/快照序列]
       |
       v
[Cosmos Tokenizer 微调] --> 离散Token
       |
       v
[自回归Transformer] --> 预测未来涂层分布
       |
       v
[精炼模块] --> 校正精度
```

**优势:** PhysiX 已证明自然视频知识可迁移到物理, 喷涂过程的视频 (RGB/IR) 可能直接用于预训练
**挑战:** 4.5B 参数模型计算资源需求大

#### 策略C: Poseidon 的小样本路线 (资源受限时)

```
[少量喷涂PDE参数空间采样 (20-50组)]
       |
       v
[Poseidon 预训练模型微调]
       |
       v
[PDE解算子] --> 给定参数直接预测解场
```

**优势:** 极少的训练数据, 适合资源受限场景
**局限:** 目前主要支持流体动力学 PDE, 多物理耦合需额外工作

### 5.3 与 RGB-IR 多模态检测的交叉

| 连接点 | 具体应用 |
|--------|----------|
| **物理表征作为特征** | 用 Walrus 学到的物理表征 (温度场、流场) 作为 RGB-IR 检测的额外输入通道 |
| **仿真-实测域自适应** | 用物理基础模型弥合仿真数据与真实传感器数据的域差距 |
| **反问题求解** | 给定 IR 热图, 用 PDEformer 反推涂层厚度分布 (PDE系数恢复) |
| **实时预测** | 在喷涂过程中, 用物理代理模型实时预测涂层形成, 与 IR 实测对比作为闭环反馈 |

### 5.4 技术路线图

```
Phase 1 (1-2个月): 数据准备与基线
  - 生成喷涂CFD数据集 (2D简化, 500组)
  - 实现 The Well 格式转换
  - 复现 Poseidon / MPP 在喷涂相关PDE上的效果

Phase 2 (2-3个月): 微调与代理模型
  - 微调 Walrus / GPhyT
  - 评估代理模型精度 vs CFD
  - 构建 Gym 环境, 接入 Rainbow DQN

Phase 3 (3-4个月): 多物理耦合与闭环
  - 扩展到热-流-固化多物理耦合
  - 融合 IR 热成像实测数据
  - 实现在线修正的喷涂路径规划
```

---

## 六、技术挑战与未来方向

### 6.1 当前主要挑战

| 挑战 | 描述 | 影响 |
|------|------|------|
| **数据稀缺** | 物理仿真数据远少于图像/文本 (The Well: 15TB vs ImageNet: ~1.2TB 但样本数量差距巨大) | PhysiX 通过视频迁移缓解 |
| **多尺度问题** | 同一模型需处理微米级液滴和米级喷涂区域 | GPhyT 承认这是主要障碍 |
| **多物理耦合** | 不同PDE之间的耦合 (流-热-固) 难以统一建模 | 目前模型主要处理单一类型PDE |
| **精度与可靠性** | 代理模型的误差是否在工程可接受范围内? | 长时间rollout误差累积 |
| **模糊化 (Blurring)** | 确定性模型倾向于预测平均值, 丢失精细结构 | GenCast用概率方法缓解 |
| **边界条件多样性** | 真实工程问题有复杂几何和边界条件 | UPT/GP-UPT 致力于解决 |
| **3D扩展** | 大多模型在2D上验证, 3D计算代价陡增 | Walrus的2D/3D统一是重要进步 |
| **可解释性** | 黑盒模型在工程应用中信任度低 | CompNO的模块化方法提供部分可解释性 |

### 6.2 未来方向

1. **统一物理Token化**: 发展像素/语音Token一样的通用物理Token, 使不同物理系统共享同一词表
2. **物理约束嵌入**: 将守恒律 (质量、能量、动量) 硬编码到模型架构中, 而非仅靠数据学习
3. **主动学习 + 基础模型**: 用基础模型指导CFD采样, 以最少仿真获取最多信息
4. **从观测数据直接学习**: 绕过CFD仿真, 直接从实验观测 (如IR/RGB视频) 学习物理
5. **可组合基础模型**: 如 CompNO 所示, 学习基本算子 (对流、扩散、反应) 的积木, 灵活组合
6. **实时在线适应**: 模型在部署时持续从新数据学习, 适应工况变化
7. **概率物理模型**: 不仅预测均值, 同时量化不确定性 (如 GenCast 的扩散方法)
8. **Sim-to-Real迁移**: 弥合仿真域与真实物理域的差距

---

## 七、推荐入门路线

### 7.1 快速上手 (1-2 周)

**第一步: 理解概念**
- 阅读综述: [Towards Physics-Guided Foundation Models](https://arxiv.org/abs/2502.15013)
- 阅读 GPhyT 论文了解全貌: [arXiv:2509.13805](https://arxiv.org/abs/2509.13805)

**第二步: 跑通基线**
```bash
# 安装 The Well 数据集工具
pip install the-well

# 下载一个小数据集 (如 incompressible NS)
python -m the_well.download --dataset active_matter

# 用 MPP 跑一个基线
git clone https://github.com/PolymathicAI/multiple_physics_pretraining
```

**第三步: 尝试预训练模型**
```bash
# Walrus (推荐, 最新最全)
git clone https://github.com/PolymathicAI/walrus
# 从 HuggingFace 下载预训练权重
# huggingface.co/polymathic-ai/walrus

# 或 Poseidon (小巧高效)
git clone https://github.com/camlab-ethz/poseidon
# 预训练模型和数据: huggingface.co/camlab-ethz
```

### 7.2 深入研究 (1-2 月)

1. **数据准备**: 用 OpenFOAM 生成 2D 简化喷涂仿真数据 (500组, 变化喷枪速度/距离/流量)
2. **格式转换**: 将 VTK/OpenFOAM 数据转为 HDF5 (The Well 格式)
3. **微调实验**: 分别用 Walrus / Poseidon / DPOT 微调, 对比效果
4. **代理模型验证**: 在测试集上评估 VRMSE, 绘制误差随rollout步数的增长曲线
5. **环境封装**: 将代理模型包装为 Gymnasium 环境, 对接 Rainbow DQN

### 7.3 推荐阅读顺序

```
入门级:
  1. The Well 论文 -- 了解数据全景
  2. MPP 论文 -- 理解跨物理预训练的核心思想
  3. Poseidon 论文 -- 理解PDE基础模型设计

进阶级:
  4. GPhyT 论文 -- 最完整的PFM设计与分析
  5. Walrus 论文 -- 最新最强的工程实现
  6. PhysiX 论文 -- 离散Tokenizer的新范式
  7. VICON 论文 -- 上下文学习在PDE中的应用

应用级:
  8. PINNs for Plasma Spraying 论文 -- 喷涂领域直接相关
  9. UPT/GP-UPT 论文 -- 工程几何上的可扩展性
  10. CompNO 论文 -- 模块化组合思路
```

### 7.4 计算资源需求参考

| 模型 | 推理 (单样本) | 微调 (500样本) | 预训练 | GPU需求 |
|------|-------------|---------------|--------|---------|
| Poseidon-T | ~100ms | ~2h | ~24h | 1x A100 |
| Poseidon-B | ~500ms | ~8h | ~72h | 1x A100 |
| MPP | ~200ms | ~4h | ~48h | 1x A100 |
| DPOT-S | ~150ms | ~3h | ~36h | 1x A100 |
| GPhyT-M | ~300ms | ~6h | ~100h | 4x A100 |
| Walrus (1.3B) | ~1s | ~24h | ~数百H100小时 | 4-8x A100/H100 |
| PhysiX (4.5B) | ~2s | ~48h+ | ~数千GPU小时 | 8x A100/H100 |

> 注: 以上为估算值, 实际取决于数据分辨率、序列长度等因素.

---

## 八、开源资源汇总

### 8.1 模型代码

| 资源 | 链接 | 说明 |
|------|------|------|
| GPhyT | [github.com/FloWsnr/General-Physics-Transformer](https://github.com/FloWsnr/General-Physics-Transformer) | 完整训练/推理代码 |
| PhysiX | [github.com/ArshKA/PhysiX](https://github.com/ArshKA/PhysiX) | 基于 NVIDIA Cosmos |
| Walrus | [github.com/PolymathicAI/walrus](https://github.com/PolymathicAI/walrus) | MIT 协议, 含预训练权重 |
| Poseidon | [github.com/camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon) | 含预训练权重 |
| MPP | [github.com/PolymathicAI/multiple_physics_pretraining](https://github.com/PolymathicAI/multiple_physics_pretraining) | 多物理预训练 |
| DPOT | [github.com/thu-ml/DPOT](https://github.com/thu-ml/DPOT) | 去噪预训练算子 |
| VICON | [github.com/Eydcao/VICON](https://github.com/Eydcao/VICON) | 视觉上下文算子 |
| GraphCast | [github.com/google-deepmind/graphcast](https://github.com/google-deepmind/graphcast) | 含 GenCast |
| FourCastNet | [github.com/NVlabs/FourCastNet](https://github.com/NVlabs/FourCastNet) | V1/V2/V3 |
| Aurora | [github.com/microsoft/aurora](https://github.com/microsoft/aurora) | 大气基础模型 |

### 8.2 数据集

| 资源 | 链接 | 规模 |
|------|------|------|
| The Well | [github.com/PolymathicAI/the_well](https://github.com/PolymathicAI/the_well) | 15TB, 16个数据集 |
| PDEBench | [github.com/pdebench/PDEBench](https://github.com/pdebench/PDEBench) | ~TB级, 多种PDE |
| ERA5 | [ECMWF CDS](https://cds.climate.copernicus.eu/) | PB级, 全球大气 |

### 8.3 相关论文列表 (持续更新)

| 资源 | 链接 | 说明 |
|------|------|------|
| Neural PDE Solver 论文合集 | [github.com/bitzhangcy/Neural-PDE-Solver](https://github.com/bitzhangcy/Neural-PDE-Solver) | 持续更新的综合列表 |
| Awesome Physical AI | [github.com/keon/awesome-physical-ai](https://github.com/keon/awesome-physical-ai) | 物理AI全景 |

---

## 九、总结与建议

### 对喷涂路径规划研究的核心建议:

1. **优先尝试 Walrus 微调路线**: 它覆盖最广泛的物理域, 开源最友好 (MIT), 有预训练权重, 且其可操控表征与喷涂中的涡旋/扩散物理高度相关.

2. **用 Poseidon 做快速验证**: 极高的样本效率 (20 samples) 意味着可以用很少的 CFD 数据快速验证概念.

3. **关注 PhysiX 的视频迁移范式**: 你的 RGB-IR 数据天然适合这条路线 -- 直接从观测视频学习物理, 无需中间CFD仿真.

4. **The Well 作为标准基准**: 在发表论文时, 在 The Well 基准上报告结果可以增加可比性和说服力.

5. **逐步构建喷涂专用基准**: 长期目标是构建类似 The Well 的喷涂仿真基准数据集, 推动领域发展.

---

> 本报告基于截至 2026 年 2 月的公开文献和资源编写. 物理基础模型领域发展极为迅速, 建议每月关注 arXiv cs.LG / physics.comp-ph 板块的最新论文.
