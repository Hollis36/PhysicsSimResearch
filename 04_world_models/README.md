# AI 世界模型 (World Models) 深度调研报告

> **面向场景**: 喷涂路径规划 (RL-based) + 多模态检测
> **调研日期**: 2026-02-04 (更新: **2026-06-17**)
> **模型知识截止**: 涵盖 2018-2026 年中关键进展

---

## 目录

0. [🔄 最新进展更新 (2026-02 → 2026-06)](#-最新进展更新-2026-02--2026-06)
1. [领域概述: 从 Schmidhuber 到当前世界模型热潮](#1-领域概述)
2. [世界模型 vs 传统仿真器对比](#2-世界模型-vs-传统仿真器对比)
3. [主要模型/平台对比表](#3-主要模型平台对比表)
4. [各模型/系统详细分析](#4-各模型系统详细分析)
5. [关键论文列表](#5-关键论文列表)
6. [Model-based RL 与世界模型结合 -- 对接喷涂路径规划](#6-model-based-rl-与世界模型结合----对接喷涂路径规划)
7. [技术路线建议](#7-技术路线建议)
8. [推荐入门资源](#8-推荐入门资源)

---

## 🔄 最新进展更新 (2026-02 → 2026-06)

> 本节于 **2026-06-17** 增补,记录自原调研 (2026-02-04) 以来约 4.5 个月的前沿进展。每条均附可验证来源;明确区分 **[已发布/已验证]** 与 **[仅预告/未证实]**。

### 0.1 重大新发布: NVIDIA Cosmos 3 (取代 Predict/Transfer/Reason 分体框架)

**[已验证]** **Cosmos 3** 于 **2026-05-31** 在 **GTC Taipei / COMPUTEX** 发布,官方定位为 "全球首个完全开放的 omnimodel" —— 可原生理解并生成 **文本 / 图像 / 视频 / 环境声音 / 动作 (action)**。

- **架构**: **Mixture-of-Transformers** —— 一个 "推理 Transformer (reasoning transformer)" + 一个 "专家生成 Transformer (expert generation transformer)" 配对,先理解交互再生成视频与**动作轨迹**。训练数据为 "横跨文本/图像/视频/声音/动作轨迹的数十亿样本"。
- **关键能力**: **原生输出动作轨迹** (关节角、夹爪位置等),使其同时充当 VLM + 世界/视频模型 + **world-action-model** 主干。这是相对原笔记 Cosmos 1/2 的根本性升级 —— 原 Predict / Transfer / Reason 三分体框架已被统一的 omnimodel 取代。
- **开放性**: 开放权重 (Hugging Face + GitHub),许可证 **OpenMDW 1.1** (Linux Foundation)。
- **已公开的机器人/工业采用方**: Agile Robots、Doosan Robotics、LG Electronics、Samsung Electronics、Skild AI(机器人);Li Auto(自动驾驶);Centific、Fogsphere、Linker Vision、Milestone、Yuan(视觉 AI)。
- **Cosmos Coalition**: 同期成立的开放世界模型联盟,成员含 Agile Robots、Black Forest Labs、Generalist、LTX、**Runway**、Skild AI。
- **对喷涂场景的意义**: 这是目前最可直接落地的平台级世界模型 —— 开放权重 + 原生动作输出 + 明确的合成数据/机器人策略定位,可作为喷涂工作单元的合成数据生成器或 world-action 主干。
- 来源: [NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai) · [NVIDIA Blog](https://blogs.nvidia.com/blog/cosmos-3-physical-ai-open-world-foundation-model/)

### 0.2 其他平台/产品更新

| 项目 | 状态 | 日期 | 要点 | 来源 |
|------|------|------|------|------|
| **Google DeepMind "Project Genie"** | [已发布] | 2026-01-29 | Genie 3 (~11B, 720p/24fps 实时交互) 从研究预览**产品化**,面向美国 AI Ultra 订阅者。**注: 不存在已确认的 "Genie 4"** | [blog.google](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/) |
| **World Labs (Fei-Fei Li) — World API** | [已发布] | 2026-01-21 | 以编程方式从文本/图像/视频生成可探索 3D 世界 (Marble 主干) | [worldlabs.ai](https://www.worldlabs.ai/blog) |
| **World Labs — Spark 2.0** | [已发布] | 2026-04-14 | 流式 3D Gaussian-Splat 渲染 (带 LOD),Web 端实时串流生成的 3D 世界 (渲染基建,非新世界模型) | [worldlabs.ai](https://www.worldlabs.ai/blog) |
| **World Labs 融资** | [已发布] | 2026-02-18 | 融资 **10 亿美元** (AMD、Autodesk、NVIDIA、Sea 等参投) | [Fast Company](https://www.fastcompany.com/91503667/world-labs-most-innovative-companies-2026) |
| **Yann LeCun / AMI Labs** | [已发布] | 2026-03-09 | LeCun 离开 Meta 后创立的 **AMI Labs** 完成 **10.3 亿美元**种子轮 (估值 ~35 亿美元),全力押注 JEPA 世界模型。JEPA 路线现已成为独立创业公司 | [TechCrunch](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/) |

**澄清/未证实项 (按真实性标注)**:
- **Dreamer 4**: 截至 2026-06-17 **仍无官方稳定发布或官方代码仓库**;论文仍为 arXiv:2509.24527 (2B 参数, "shortcut forcing")。社区有**非官方** PyTorch 实现 (`github.com/nicklashansen/dreamer4`) 与 `dreamer4` PyPI 包 —— 使用前需注意其非官方性质。
- **V-JEPA 2**: Feb–Jun 2026 内**未发现** Meta 发布 V-JEPA 后继版本。
- 未发现已确认的 "Genie 4" / "V-JEPA 3" / "GAIA-4" —— 不应引用这些名称。

### 0.3 新论文: 世界模型内做 RL (与喷涂 MBRL 直接相关)

| 论文 | 会议/状态 | 日期 | arXiv | 一句话 + 关键结果 |
|------|----------|------|-------|-------------------|
| **RISE: Self-Improving Robot Policy with Compositional World Model** | **RSS 2026** [已验证] | 2026-02-11 (v2 04-28) | [2602.11075](https://arxiv.org/abs/2602.11075) | **组合式世界模型** = 可控多视角动力学模型 + 独立的 "进度价值 (progress value)" 评估器,纯在想象中自我改进策略。接触密集操作上绝对提升: 砖块分拣 **+35%**、背包打包 **+45%**、关箱 **+35%**。"先预测再在想象中打分" 的结构与喷涂的膜厚奖励 MBRL 高度同构 |
| **World-Gymnast: Training Robots with RL in a World Model** | arXiv [已验证] | 2026-02-02 | [2602.02454](https://arxiv.org/abs/2602.02454) | 在**动作条件视频世界模型**中 roll-out + VLM 奖励来 RL 微调 VLA 策略;Bridge 机器人上比 SFT 最高快 **18×**、比软件仿真器最高快 **2×**。是 "在学习到的世界模型里做 RL 胜过软件仿真器" 的有力证据 |
| **World4RL: Diffusion World Models for Policy Refinement** | arXiv [已验证, 2026-03 修订] | 2509.19080 | [2509.19080](https://arxiv.org/abs/2509.19080) | 用扩散世界模型作高保真 "仿真",纯在想象中精修模仿学习初始化的操作策略 |
| **World Model for Robot Learning: A Comprehensive Survey** | 综述 [已验证] | 2026 (~03) | [项目页](https://ntumars.github.io/wm-robot-survey/) | 系统梳理 "世界模型即策略 vs 世界模型即仿真器",跟踪至 2026-03 |

> **空白提示 (诚实标注)**: 在 Feb–Jun 2026 内**未发现**专门将世界模型 / MBRL 应用于喷涂或膜厚控制的论文;喷涂方向的基线仍停留在 2026 年前 (PaintRL 等)。这对本研究是一个明确的开放机会 —— RISE / World-Gymnast 的 "可控动力学 + 独立奖励评估 + 想象中 RL" 范式是最值得迁移的模板。

---

## 1. 领域概述

### 1.1 世界模型的起源与发展脉络

**世界模型 (World Model)** 的核心思想是: 智能体在内部构建一个对环境的可学习表示, 通过该表示预测动作后果、进行规划和决策, 而非直接在真实环境中反复试错。

**关键里程碑:**

| 年代 | 事件 | 意义 |
|------|------|------|
| 1990s | Schmidhuber 提出 "通过预测环境来学习" | 世界模型最早的理论雏形 |
| 2015 | DeepMind DQN / AlphaGo | Model-free RL 的巅峰, 但样本效率极低 |
| 2018 | **Ha & Schmidhuber "World Models"** | 用 VAE + RNN + 进化策略, 首次在 "梦境" 中训练 agent; 定义了现代世界模型范式 |
| 2019 | **PlaNet** (Hafner et al.) | 提出 RSSM (Recurrent State-Space Model), 在潜空间中直接规划 |
| 2020 | **Dreamer** (Hafner et al.) | 在 RSSM 基础上引入 Actor-Critic, 用想象轨迹训练策略 |
| 2022 | **DayDreamer** | 首次将 Dreamer 部署到真实机器人 (四足、机械臂、轮式), 1 小时内学会行走 |
| 2023 | **DreamerV3** | 统一超参数, 150+ 任务通用; 首个从零在 Minecraft 中采集钻石的算法 |
| 2023 | **UniSim** (Yang et al.) | 用扩散模型生成条件未来帧, 向通用交互式仿真器迈进 |
| 2023 | **Wayve GAIA-1** | 9B 参数, 自回归 Transformer, 自动驾驶世界模型 |
| 2024 | **Sora** (OpenAI) | 视频生成爆发, 引发 "视频模型是否是世界模型" 的大讨论 |
| 2024 | **V-JEPA** (Meta/LeCun) | JEPA 架构在视频上的实例, 在抽象表示空间预测 |
| 2025.01 | **NVIDIA Cosmos** (CES 2025) | Physical AI 世界基础模型平台, 开源开放 |
| 2025.03 | **Cosmos Reason 1** (GTC 2025) | 物理常识推理 VLM, 用于具身决策 |
| 2025.03 | **Wayve GAIA-2** | 潜扩散世界模型, 多相机一致性 |
| 2025.08 | **Google DeepMind Genie 3** | 文本到 3D 交互环境, 24fps/720p 实时生成 |
| 2025.11 | **World Labs Marble** (Fei-Fei Li) | 空间智能世界模型, 可导出 3D 环境, 首个商业产品 |
| 2025.11 | **Yann LeCun 离开 Meta, 创立 AMI Labs** | 30 亿欧元估值, 全力押注 JEPA 世界模型 |
| 2025.12 | **Runway GWM-1** | 通用世界模型, Worlds/Robotics/Avatars 三个变体 |
| 2025.12 | **Wayve GAIA-3** | 15B 参数, 9 国数据, 聚焦安全评估 |
| 2026.01 | **Dreamer 4** | 可扩展世界模型 agent, 单 GPU 实时推理 |
| 2026.01 | **DeepMind Project Genie** | Genie 3 产品化 (AI Ultra 订阅) |
| 2026.02 | **World-Gymnast / RISE** | 世界模型内做 RL 用于机器人操作 (RSS 2026) |
| 2026.03 | **AMI Labs (LeCun) 成立** | 10.3 亿美元种子轮, JEPA 路线独立创业 |
| 2026.05 | **NVIDIA Cosmos 3** | 完全开放 omnimodel, 原生输出动作轨迹 (GTC Taipei) |

### 1.2 Gartner 2026 趋势: Physical AI

Gartner 将 **Physical AI** 列为 2026 年十大战略技术趋势之一。其定义为: 将智能注入物理世界, 驱动能感知、决策和行动的机器(机器人、无人机、智能设备)。

Gartner 将其归入 **"Synthesist" (综合者)** 主题 -- 结合专用模型、智能体和物理-数字系统创造新价值。这意味着世界模型不再只是学术概念, 而是即将进入产业落地阶段。

### 1.3 当前趋势总结

2025-2026 年世界模型领域呈现以下特征:

1. **从视频生成到交互仿真的跨越**: 不再只是生成好看的视频, 而是要求实时交互、物理一致性、因果推理
2. **从游戏/驾驶向通用制造场景扩展**: 机器人操控、工业仿真成为核心应用
3. **开源化加速**: NVIDIA Cosmos 全面开源, Dreamer 系列一直开源
4. **架构分化**: 自回归 Transformer vs 潜扩散模型 vs JEPA (抽象表示预测)
5. **与 RL 深度融合**: 世界模型 + 策略学习成为 Physical AI 的核心范式

---

## 2. 世界模型 vs 传统仿真器对比

| 维度 | 传统仿真器 (如 Isaac Sim, Gazebo, CARLA) | 学习型世界模型 (如 Dreamer, Cosmos) |
|------|------|------|
| **构建方式** | 手工定义物理引擎、场景、规则 | 从数据 (视频/交互) 自动学习 |
| **物理精度** | 高 (基于已知物理方程) | 中等 (逼近但不完全精确) |
| **泛化能力** | 弱 (仅限建模过的场景) | 强 (可迁移到新场景/任务) |
| **样本效率** | 需要大量仿真步数 | **10-100x 更高** (在想象中训练) |
| **非结构化环境** | 难以建模 (如流体、柔性体) | 自然建模 (从视频学习) |
| **构建成本** | 高 (需专业工程师手动构建) | 中 (需要训练数据和计算资源) |
| **Sim-to-Real Gap** | 存在, 需要 domain randomization 等技巧 | 存在, 但可通过真实数据微调缩小 |
| **实时性** | 通常可实时 | 取决于模型大小 (小模型可实时) |
| **可解释性** | 高 (物理规则透明) | 低 (黑盒) |
| **长时一致性** | 天然一致 | 目前仍是挑战 (累积误差) |
| **反事实推理** | 有限 (需手动修改场景) | 天然支持 (改变输入条件即可) |
| **边缘案例生成** | 手动设计 | 可自动生成罕见场景 |

### 核心洞察

**对于喷涂路径规划场景:**
- 传统仿真器适合 **精确的膜厚物理建模** (如射线追踪法计算涂层厚度)
- 世界模型适合 **策略探索和泛化** (在多种工件形状上快速探索路径策略)
- **最佳实践是混合方案**: 用传统仿真器提供精确的物理约束, 用世界模型加速策略搜索和泛化

---

## 3. 主要模型/平台对比表

| 模型/平台 | 开发者 | 架构 | 训练数据 | 应用领域 | 可用性 | 关键特点 |
|-----------|--------|------|----------|----------|--------|----------|
| **Cosmos Predict** | NVIDIA | Diffusion Transformer + 自回归 Transformer | 9000 万亿 token (20M 小时视频) | 机器人、自动驾驶 | 开源 (Apache 2) | 物理一致视频生成, Tokenizer 2048x 压缩 |
| **Cosmos Transfer** | NVIDIA | 多控制风格迁移 | 同上 | 仿真到真实迁移 | 开源 | 将 Isaac Sim 渲染转为真实感视频 |
| **Cosmos Reason 2** | NVIDIA | VLM (2B/8B) + CoT 推理 | 物理常识 + 具身推理数据 (SFT+RL) | 机器人规划、视觉 AI | 开源, HuggingFace | 理解时空物理, 输出轨迹坐标 |
| **Genie 3** | Google DeepMind | 多层记忆系统 + TPU v5 | 大规模视频 + 交互数据 | 游戏、Agent 训练 | AI Ultra 订阅 (美国) | 文本生成交互 3D 环境, 24fps/720p, 涌现一致性 |
| **GWM-1 Worlds** | Runway | 自回归 (基于 Gen-4.5) | 视频数据 | 游戏、创意、机器人 | API/SDK | 实时环境生成, 相机/动作控制 |
| **GWM-1 Robotics** | Runway | 自回归, 条件于机器人动作 | 机器人操作数据 | 机器人操控 | Python SDK (企业) | 反事实轨迹生成 |
| **Marble** | World Labs (Fei-Fei Li) | 多模态 3D 生成 | 图像/视频/3D 数据 | 3D 创作、仿真 | Freemium + 付费 | 可导出 Gaussian Splat/Mesh, Chisel 编辑器 |
| **JEPA / V-JEPA 2** | Meta -> AMI Labs (LeCun) | Joint Embedding Predictive Architecture | 100 万+小时互联网视频 | 机器人规划、通用 AI | 研究开源 | 抽象表示空间预测, 非生成式 |
| **DreamerV3** | Hafner et al. | RSSM (RNN + 随机/确定潜变量) | 在线交互 (每个环境单独) | 通用 RL (150+ 任务) | 完全开源 | 统一超参, Minecraft 钻石, 2025 Nature |
| **Dreamer 4** | Hafner et al. | Shortcut Forcing + Transformer | 无标签视频 + 少量动作数据 | 通用控制 | 开源 | 单 GPU 实时, 可扩展 |
| **DayDreamer** | Wu, Hafner et al. | Dreamer (RSSM) | 真实机器人在线交互 | 四足/机械臂/轮式机器人 | 开源 | 1 小时学会行走 |
| **GAIA-3** | Wayve | 潜扩散, 15B 参数 | 9 国真实驾驶数据 (10x GAIA-2) | 自动驾驶评估 | 商业 | World-on-rails 反事实评估 |
| **UniSim** | Yang et al. (Google) | 扩散模型 | 视频 + 动作 | 通用仿真 | 研究论文 | 条件未来帧生成 |

---

## 4. 各模型/系统详细分析

### 4.1 NVIDIA Cosmos -- Physical AI 世界基础模型

**论文**: [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575) (arXiv 2501.03575)

**架构体系**:
```
输入 (文本/图像/视频/传感器/动作)
    |
    v
Cosmos Tokenizer (CV8x8x8)
    |  - Haar 小波变换 (4x 降采样)
    |  - 因果时序卷积 + 注意力
    |  - 连续 (AE) 或离散 (FSQ) 量化
    |  - 总压缩比最高 2048x
    v
潜空间 Tokens
    |
    +---> [方案A] Diffusion Transformer
    |       - 3D Patchification
    |       - Self-Attention + Cross-Attention (文本)
    |       - AdaLN-LoRA (参数减少 36%)
    |       - QK-RMSNorm 稳定训练
    |
    +---> [方案B] Autoregressive Transformer
    |       - 逐 Token 预测
    |       - 适合实时交互
    v
Cosmos Tokenizer Decoder -> 视频输出
```

**训练策略**:
- 预训练数据: 9000 万亿 token, 2000 万小时视频
- NeMo Curator: CUDA 加速数据处理, 14 天处理 2000 万小时 (vs CPU 3 年+)
- 渐进式训练: 512p -> 720p, 逐步增加帧数
- 联合图像-视频训练: 因果设计使视频 tokenizer 也能处理单帧图像

**三大子模型**:
1. **Cosmos Predict**: 从文本/图像/视频/动作生成物理一致的视频预测
2. **Cosmos Transfer**: 将 Isaac Sim 等仿真器渲染转为高保真真实感视频
3. **Cosmos Reason (1 & 2)**: 物理推理 VLM, 支持 CoT 推理, 输出自然语言 + 轨迹坐标

**与喷涂场景的关联**:
- Cosmos Transfer 可将喷涂仿真器输出转为真实感视频用于检测模型训练
- Cosmos Predict 可生成多样化的喷涂场景用于数据增强
- Cosmos Reason 可用于喷涂质量的视觉推理和缺陷检测

**开源信息**: Apache 2 License, GitHub: [nvidia-cosmos](https://github.com/nvidia-cosmos), HuggingFace 可下载

---

### 4.2 Google DeepMind Genie 3 -- 交互式 3D 世界生成

**发布**: 2025 年 8 月

**官方博客**: [Genie 3: A new frontier for world models](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)

**架构特点**:
- 多层记忆系统: 短期记忆处理即时动作, 长期记忆保持世界稳定
- 运行于 Google TPU v5 基础设施
- 每帧生成时间 ~41ms, 实现 24fps/720p 实时交互
- **涌现一致性 (Emergent Consistency)**: 无需显式 3D 表示 (不依赖 NeRF/Gaussian Splatting), 一致性从训练中自然涌现

**关键能力**:
- 文本到交互世界: 输入文本描述, 生成可移动、可交互的 3D 环境
- 对象永久性: 环境修改持久保持
- 可提示世界事件: 文本指令改变天气、引入新对象等
- 物理仿真: 重力、碰撞、流体 (但不如专业物理引擎精确)

**与 SIMA Agent 集成**: DeepMind 已在 Genie 3 生成的世界中训练其通用 3D Agent SIMA

**可用性**: 已向美国 Google AI Ultra 订阅者开放, 计划扩展

---

### 4.3 Runway GWM-1 -- 通用世界模型

**发布**: 2025 年 12 月

**官方**: [Introducing Runway GWM-1](https://runwayml.com/research/introducing-runway-gwm-1)

**架构**: 基于 Gen-4.5 的自回归模型, 逐帧生成, 24fps/720p

**三个变体**:
1. **GWM Worlds**: 从静态场景生成无限可探索空间, 包含几何/光照/物理
2. **GWM Robotics**: 在机器人数据上训练, 预测条件于动作的视频轨迹, 支持反事实生成
3. **GWM Avatars**: 音频驱动的交互式虚拟人, 实时面部表情/唇同步

**技术路线差异**: GWM-1 学习 "像素级内部仿真" -- 从视频数据直接学习物理法则、光照、几何和因果关系, 而非建模高层符号。

**对喷涂场景的意义**: GWM Robotics 的反事实生成能力可用于探索不同喷涂轨迹的预期效果

---

### 4.4 World Labs Marble -- 空间智能世界模型 (Fei-Fei Li)

**发布**: 2025 年 11 月

**官方博客**: [Marble: A Multimodal World Model](https://www.worldlabs.ai/blog/marble-world-model)

**核心理念**: Fei-Fei Li 提出 **空间智能 (Spatial Intelligence)** 是 AI 的下一个前沿 -- 如果 LLM 教会机器读写, 空间智能将教会机器 "看" 和 "建造"

**关键能力**:
- 多模态输入: 文本、图像、视频、空间草图
- 生成空间一致的高保真持久 3D 世界
- **Chisel 编辑器**: 粗略 3D 布局 + 文本提示 -> AI 填充视觉细节 (类似 HTML+CSS 的结构/样式分离)
- 可导出: Gaussian Splats, 传统网格 (Mesh), 视频文件

**差异化优势**: 其他世界模型 (Genie 3, GWM Worlds) 是实时流式生成, Marble 生成的是**持久、可下载、可编辑的 3D 环境**, 更适合集成到仿真工具和渲染引擎中

**融资**: 2.3 亿美元 (2024), 估值持续增长

**对喷涂场景的潜在价值**: 可快速生成多样化工件的 3D 模型用于仿真训练

---

### 4.5 Yann LeCun / AMI Labs / JEPA

**公司**: Advanced Machine Intelligence (AMI) Labs (2025 年底成立)

**核心论文**:
- [I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) (Meta, 2023)
- [V-JEPA 2](https://arxiv.org/) (Meta, 2025)
- [LeJEPA](https://arxiv.org/) (Balestriero & LeCun, 2025)

**JEPA 架构哲学**:
```
LLM (大语言模型):
    输入: 离散 Token 序列
    目标: 预测下一个 Token
    局限: 无法理解物理世界

JEPA (联合嵌入预测架构):
    输入: 多模态数据 (视频/图像/传感器)
    目标: 在抽象表示空间预测未来状态变化
    优势: 学习世界的底层规则, 天然支持规划和执行
```

**关键区别**:
| 维度 | LLM | JEPA |
|------|-----|------|
| 输入 | 语言 Token | 多模态 (视频/图像/传感器) |
| 学习目标 | 预测下一个词 | 预测抽象状态变化 |
| 学习方式 | 离散序列建模 | 表示学习 + 因果建模 |
| 行动能力 | 无 | 天然支持规划和执行 |
| 生成能力 | 有 (文本/图像) | **非生成式** -- 只预测抽象表示 |

**V-JEPA 2 成果**: 在 100 万+小时互联网视频上预训练, 仅用少量机器人数据微调即可用于真实机械臂的规划任务

**LeCun 的核心观点**: LLM 是通往人类级智能的 "死胡同", JEPA 世界模型才是正确路径。但他承认, 这需要数年到十年才能看到成果。

**AMI Labs 估值**: 预发布估值 30 亿欧元, 寻求 5 亿欧元融资

---

### 4.6 Dreamer 系列 -- Model-based RL 的标杆

#### DreamerV3 架构详解

**论文**: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (Nature 2025)

```
┌─────────────────────────────────────────┐
│              World Model                │
│                                         │
│  Observation o_t                        │
│       |                                 │
│       v                                 │
│  [Encoder] ──> Representation Model     │
│       q(z_t | h_t, o_t)                │
│       |                                 │
│       v                                 │
│  RSSM:                                  │
│  ┌──────────────────────┐              │
│  │ Sequence Model       │              │
│  │ h_t = f(h_{t-1},     │              │
│  │        z_{t-1},       │              │
│  │        a_{t-1})       │              │
│  │ (GRU/RNN)            │              │
│  ├──────────────────────┤              │
│  │ Dynamics Model       │              │
│  │ z_hat_t ~ p(z|h_t)  │              │
│  │ (无需观测的预测)       │              │
│  └──────────────────────┘              │
│       |                                 │
│       v                                 │
│  Prediction Heads:                      │
│  - Observation Decoder                  │
│  - Reward Predictor                     │
│  - Continue Predictor                   │
│                                         │
│  潜变量 z_t: 32 个 one-hot 向量        │
│  (32 个 Categorical 分布, 比高斯更具    │
│   表达力)                               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│           Imagination Training          │
│                                         │
│  初始状态 z_0, h_0                      │
│       |                                 │
│       v                                 │
│  循环 T 步 (无需真实观测):              │
│    a_t ~ Actor(z_t, h_t)               │
│    h_{t+1} = f(h_t, z_t, a_t)          │
│    z_{t+1} ~ p(z|h_{t+1})              │
│    r_{t+1} = RewardHead(z_{t+1},h_{t+1})│
│       |                                 │
│       v                                 │
│  Actor: REINFORCE 优化                  │
│  Critic: lambda-returns 优化            │
│  (梯度不回传世界模型)                    │
└─────────────────────────────────────────┘
```

**DreamerV3 关键创新**:
- **Symlog 归一化**: 将奖励和值函数映射到对称对数空间, 统一不同任务的数值范围
- **百分位缩放**: 按百分位归一化 lambda-returns, 避免极端值影响
- **KL 均衡**: 用自由比特和均衡系数平衡编码器和先验
- **统一超参数**: 所有 150+ 任务使用完全相同的超参数

#### DayDreamer -- 真实机器人部署

**论文**: [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/abs/2206.14176)

**关键成果**:
- 四足机器人: 1 小时从零学会翻身、站立、行走; 10 分钟适应外部推力
- 机械臂: 直接从相机图像学会拾取放置, 接近人类水平
- 轮式机器人: 从相机图像学会导航到目标位置

#### Dreamer 4 (2025)

**论文**: [Dreamer 4: Scalable World Model Agents](https://arxiv.org/abs/2509.24527)

**关键突破**:
- **Shortcut Forcing**: 新的训练目标, 实现快速准确的推理
- **高效 Transformer 架构**: 单 GPU 实时交互推理
- **通用动作条件化**: 从少量动作数据学习, 大部分知识从无标签视频中提取

---

### 4.7 UniSim -- 通用交互式仿真器

**论文**: [UniSim: Learning Interactive Real-World Simulators](https://universal-simulator.github.io/) (Yang et al., 2023)

**方法**: 使用扩散模型, 条件于机器人动作生成未来帧, 构建通用仿真器

**意义**: 展示了从视频数据学习的世界模型可以替代传统仿真器用于机器人策略训练

---

### 4.8 Wayve GAIA 系列 -- 自动驾驶世界模型

| 版本 | 参数 | 架构 | 核心用途 |
|------|------|------|----------|
| GAIA-1 (2023) | 9B | 自回归 Transformer | 场景生成 |
| GAIA-2 (2025.03) | ~7.5B | 潜扩散模型 | 训练与验证 |
| GAIA-3 (2025.12) | 15B | 潜扩散模型 | 安全评估 |

**GAIA-3 关键创新**: "World-on-Rails" 方法 -- 保持场景中所有其他元素不变, 仅改变自车轨迹, 实现精确的反事实安全评估

---

## 5. 关键论文列表

### 5.1 奠基性论文

| # | 论文 | 作者 | 年份 | 关键贡献 |
|---|------|------|------|----------|
| 1 | **World Models** | Ha & Schmidhuber | 2018 | VAE+RNN+ES, 在 "梦" 中训练 agent |
| 2 | **Learning Latent Dynamics for Planning (PlaNet)** | Hafner et al. | 2019 | RSSM, 潜空间规划 |
| 3 | **Dream to Control (Dreamer)** | Hafner et al. | 2020 | RSSM + Actor-Critic 想象训练 |
| 4 | **DreamerV2** | Hafner et al. | 2021 | 离散潜变量, Atari 达到人类水平 |
| 5 | **DayDreamer** | Wu, Hafner et al. | 2022 | 首次真实机器人部署 |
| 6 | **DreamerV3** | Hafner et al. | 2023/2025 | 统一超参, 150+任务, Nature |

### 5.2 大规模世界基础模型 (2023-2026)

| # | 论文/系统 | 来源 | 年份 | 关键贡献 |
|---|-----------|------|------|----------|
| 7 | **GAIA-1** | Wayve | 2023 | 9B 参数自回归驾驶世界模型 |
| 8 | **UniSim** | Yang et al. | 2023 | 通用交互式扩散仿真器 |
| 9 | **I-JEPA** | LeCun et al. (Meta) | 2023 | 联合嵌入预测架构, 非生成式 |
| 10 | **Cosmos WFM** | NVIDIA | 2025.01 | Physical AI 平台, 开源, 9000T tokens |
| 11 | **Cosmos Reason 1** | NVIDIA | 2025.03 | 物理常识 VLM, 具身推理 |
| 12 | **GAIA-2** | Wayve | 2025.03 | 潜扩散, 多相机一致性 |
| 13 | **V-JEPA 2** | Meta | 2025 | 100 万小时视频预训练, 机器人微调 |
| 14 | **Genie 3** | DeepMind | 2025.08 | 文本到交互 3D, 涌现一致性 |
| 15 | **Cosmos Reason 2** | NVIDIA | 2025 | 2B/8B VLM, 轨迹坐标输出 |
| 16 | **Marble** | World Labs | 2025.11 | 空间智能, 可导出 3D |
| 17 | **GWM-1** | Runway | 2025.12 | 通用世界模型, 三变体 |
| 18 | **GAIA-3** | Wayve | 2025.12 | 15B, 安全评估 |
| 19 | **Dreamer 4** | Hafner et al. | 2025 | 可扩展, 单 GPU 实时 |

### 5.3 世界模型 + 机器人操控 (2024-2026)

| # | 论文 | 来源 | 年份 | 关键贡献 |
|---|------|------|------|----------|
| 20 | **DreMa: Dream to Manipulate** | - | 2024 | 组合世界模型, 物理可靠预测 |
| 21 | **PWM: Policy Learning with Multi-Task World Models** | ICLR | 2025 | 多任务策略学习 |
| 22 | **TD-MPC2** | ICLR | 2024 | 可扩展鲁棒世界模型, 连续控制 |
| 23 | **PointWorld** | - | 2026.01 | 3D 世界模型, 野外机器人操控 |
| 24 | **DC-MPC** | ICLR | 2025 | 离散码本世界模型, 连续控制 |
| 25 | **LS-Imagine** | ICLR (Oral) | 2025 | 开放世界 RL, 长短期想象 |

### 5.4 综述论文

| # | 论文 | 来源 | 年份 |
|---|------|------|------|
| 26 | **Understanding World or Predicting Future? A Comprehensive Survey of World Models** | ACM CSUR | 2025 |
| 27 | **A Step Toward World Models: A Survey on Robotic Manipulation** | arXiv | 2025.11 |
| 28 | **A Comprehensive Survey on World Models for Embodied AI** | arXiv | 2025 |

### 5.5 喷涂相关

| # | 论文 | 来源 | 年份 | 关键贡献 |
|---|------|------|------|----------|
| 29 | **PaintRL: Coverage Path Planning for Industrial Spray Painting with RL** | TransLearn | 2019 | PPO, UV 映射, PyBullet 射线追踪 |
| 30 | **Optimization of Robotic Spray Painting Trajectories Using ML** | Nature Sci. Rep. | 2025 | ML 优化喷涂轨迹, 膜厚均匀性 |

---

## 6. Model-based RL 与世界模型结合 -- 对接喷涂路径规划

### 6.1 喷涂路径规划的核心挑战

```
喷涂路径规划 = f(工件几何, 喷枪参数, 膜厚目标, 工艺约束)

核心目标:
1. 膜厚均匀性 (涂层厚度均匀分布)
2. 材料利用率 (减少过喷和浪费)
3. 路径效率 (最短时间完成)
4. 边缘覆盖 (复杂曲面无死角)

传统方法痛点:
- 复杂曲面工件的路径规划依赖人工经验
- CAD 模型与实际工件的偏差
- 喷涂参数 (流量/压力/距离/速度) 的耦合优化困难
- 新工件适配周期长
```

### 6.2 当前 SOTA: PaintRL 方法分析

PaintRL (TransLearn) 将喷涂路径规划建模为 MDP:
- **状态空间**: 喷枪位置 + 覆盖率
- **动作空间**: 4 个离散动作 (左/右/上/下)
- **奖励**: 基于覆盖率和膜厚均匀性
- **仿真**: PyBullet 射线追踪计算涂层厚度, UV 映射可视化
- **算法**: PPO

**局限性**: 离散动作空间过于简化, 未考虑 6-DOF 连续控制, 未建模喷涂物理 (雾化锥、飞溅等)

### 6.3 世界模型如何增强喷涂路径规划

#### 方案 A: Dreamer 式潜空间世界模型 (最推荐)

```
                 ┌──────────────────────────────────┐
                 │      Spray Coating World Model    │
                 │                                    │
 点云/深度图 ────>│  Encoder ──> RSSM ──> Decoder     │
 喷枪状态   ────>│    |           |          |        │
 工艺参数   ────>│    z_t     h_t, z_t   膜厚预测    │
                 │              |                     │
                 │         Dynamics Model             │
                 │         (预测下一状态)               │
                 └──────────────────────────────────┘
                              |
                    ┌─────────┴─────────┐
                    │ Imagination Loop  │
                    │                   │
                    │ 在潜空间中展开     │
                    │ 数百条轨迹        │
                    │ 评估膜厚均匀性    │
                    │ 选择最优路径      │
                    └───────────────────┘
                              |
                    Actor-Critic 策略优化
                              |
                    输出: 6-DOF 喷枪轨迹
```

**优势**:
- 样本效率高: 在想象中训练, 减少真实喷涂试验
- 连续动作空间: 自然支持 6-DOF 连续控制
- 可泛化: 一个世界模型可适应多种工件形状
- DreamerV3/Dreamer 4 的统一超参数设计减少了调参负担

#### 方案 B: Cosmos 式基础模型微调

```
Cosmos Predict (预训练)
    |
    v
在喷涂数据上微调 (视频 + 动作)
    |
    v
生成喷涂过程的视频预测
    |
    +---> Cosmos Reason: 分析膜厚分布
    |
    +---> Cosmos Transfer: 仿真到真实的视觉迁移
    |
    v
作为 RL 的环境模型 (替代传统仿真器)
```

**优势**:
- 利用大规模预训练的物理先验
- Cosmos Transfer 可缩小 Sim-to-Real Gap
- 生成多样化训练数据

**劣势**:
- 计算资源需求大 (至少需要 A100/H100)
- 微调需要大量喷涂视频数据
- 实时推理可能不足

#### 方案 C: 混合架构 (最实用)

```
传统喷涂仿真器 (精确膜厚物理)
    |
    +---> 生成训练数据
    |
    v
DreamerV3 世界模型
    |
    +---> 在潜空间中快速策略搜索
    |
    +---> 物理一致性约束 (来自仿真器)
    |
    v
策略输出 (6-DOF 轨迹)
    |
    +---> 仿真器验证
    +---> 真实环境微调
```

### 6.4 与多模态检测的结合

```
喷涂世界模型
    |
    +---> 预测膜厚分布 (作为奖励信号)
    |
    v
多模态检测系统
    |
    +---> 视觉: 表面缺陷检测 (流挂/橘皮/气泡)
    +---> 红外: 干膜厚度估计
    +---> 激光: 在线膜厚测量
    +---> 点云: 工件几何偏差检测
    |
    v
检测反馈 ──> 更新世界模型 ──> 优化喷涂路径

关键: 世界模型可以将检测结果作为校正信号,
      持续缩小预测与真实的差距 (在线学习)
```

---

## 7. 技术路线建议

### 7.1 短期路线 (0-6 个月): 验证可行性

```
目标: 在仿真中验证 Model-based RL 的喷涂路径规划

步骤:
1. 搭建喷涂仿真环境
   - 基于 PyBullet 或 Isaac Sim
   - 实现膜厚计算 (射线追踪 + 喷涂模型)
   - 支持 6-DOF 连续动作空间

2. 实现 DreamerV3
   - 直接使用开源代码: github.com/danijar/dreamerv3
   - 输入: 点云/深度图 + 喷枪状态
   - 输出: 6-DOF 喷枪动作
   - 奖励: 膜厚均匀性 + 覆盖率 + 效率

3. 基线对比
   - vs PaintRL (PPO + 离散动作)
   - vs SAC/TD3 (Model-free RL)
   - 评估指标: 膜厚方差, 覆盖率, 路径长度, 样本效率
```

**推荐工具栈**:
- 仿真: Isaac Sim / PyBullet
- 世界模型: DreamerV3 (JAX/PyTorch)
- 策略训练: 内置 Actor-Critic (DreamerV3 自带)
- 可视化: Weights & Biases

### 7.2 中期路线 (6-12 个月): 增强与迁移

```
目标: 提升泛化能力, 缩小 Sim-to-Real Gap

步骤:
1. 多工件泛化
   - 在多种形状的 CAD 模型上训练
   - 引入 domain randomization (颜色/材质/光照)
   - 训练条件化世界模型 (输入工件描述)

2. Cosmos Transfer 集成 (可选)
   - 将仿真渲染转为真实感图像
   - 增强检测模型的训练数据
   - 减少视觉域差距

3. 多模态检测反馈
   - 将检测结果作为额外观测输入世界模型
   - 实现在线模型更新 (元学习或持续学习)
   - 构建 "喷涂-检测-优化" 闭环

4. 升级到 Dreamer 4 (如已发布稳定版)
   - 利用无标签视频预训练提升世界模型质量
   - 单 GPU 实时推理用于在线规划
```

### 7.3 长期路线 (12-24 个月): 产业化

```
目标: 真实产线部署

步骤:
1. 真实机器人部署
   - 参考 DayDreamer 方法
   - 先在仿真中预训练, 再在真实环境中微调
   - 安全约束: 动作空间限制, 力矩限制

2. 大规模世界模型
   - 收集真实喷涂视频数据集
   - 微调 Cosmos Predict 或训练领域专用世界模型
   - 实现多工艺参数的联合优化

3. 产品化
   - 自适应路径规划: 根据检测反馈实时调整
   - 新工件快速适配: Few-shot 微调世界模型
   - 质量预测: 世界模型预测喷涂结果用于预审
```

### 7.4 关键技术选型建议

| 技术选择 | 推荐方案 | 理由 |
|----------|----------|------|
| 世界模型架构 | **DreamerV3 / Dreamer 4** | 开源成熟, 统一超参, 已有真实机器人部署经验 |
| 仿真环境 | **Isaac Sim** | NVIDIA 生态, 与 Cosmos 无缝集成 |
| 大模型辅助 | **Cosmos Reason** | 物理推理能力, 可用于质量评估 |
| 数据增强 | **Cosmos Transfer** | 缩小 Sim-to-Real 视觉差距 |
| 检测模型 | 现有多模态方案 + 世界模型预测校正 | 互补增强 |
| RL 算法备选 | **TD-MPC2** | 如果需要更强的模型预测控制 |

---

## 8. 推荐入门资源

### 8.1 必读论文 (按顺序)

1. **入门概念**: Ha & Schmidhuber, "World Models" (2018) -- [worldmodels.github.io](https://worldmodels.github.io/)
2. **核心架构**: Hafner et al., "DreamerV3" (2023) -- [arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
3. **真实机器人**: Wu et al., "DayDreamer" (2022) -- [arxiv.org/abs/2206.14176](https://arxiv.org/abs/2206.14176)
4. **工业应用**: NVIDIA, "Cosmos WFM" (2025) -- [arxiv.org/abs/2501.03575](https://arxiv.org/abs/2501.03575)
5. **综述**: Ding et al., "Understanding World or Predicting Future?" (2025) -- [arxiv.org/abs/2411.14499](https://arxiv.org/abs/2411.14499)
6. **喷涂 RL**: "PaintRL" -- [translearn.github.io/PaintRL](https://translearn.github.io/PaintRL/)

### 8.2 代码仓库

| 仓库 | 用途 | 链接 |
|------|------|------|
| DreamerV3 | Model-based RL 世界模型 | [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3) |
| Dreamer 4 | 最新可扩展世界模型 | 待发布 |
| NVIDIA Cosmos | Physical AI 平台 | [github.com/nvidia-cosmos](https://github.com/nvidia-cosmos) |
| Cosmos Tokenizer | 视频/图像 Tokenizer | [github.com/NVIDIA/Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) |
| Cosmos Reason 2 | 物理推理 VLM | [github.com/nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2) |
| DayDreamer | 真实机器人世界模型 | [github.com/danijar/daydreamer](https://github.com/danijar/daydreamer) |
| World Model 论文列表 | 综述配套 | [github.com/tsinghua-fib-lab/World-Model](https://github.com/tsinghua-fib-lab/World-Model) |
| Awesome World Model | 自动驾驶+机器人 | [github.com/LMD0311/Awesome-World-Model](https://github.com/LMD0311/Awesome-World-Model) |

### 8.3 学习路径 (2 周速成)

```
第 1-2 天: 概念建立
  - 阅读 Ha & Schmidhuber (2018) 全文
  - 运行 worldmodels.github.io 的交互 Demo
  - 理解 VAE + RNN + Controller 的三组件架构

第 3-4 天: 核心架构
  - 阅读 DreamerV3 论文
  - 理解 RSSM: 确定性 h_t + 随机性 z_t
  - 理解想象训练: 在潜空间展开轨迹训练 Actor-Critic
  - 克隆 DreamerV3 代码, 在 DMControl 上跑一个简单任务

第 5-6 天: 真实机器人
  - 阅读 DayDreamer 论文
  - 理解 Sim-to-Real 的关键问题
  - 了解传感器融合 (本体感受 + 视觉)

第 7-8 天: 大规模世界模型
  - 阅读 Cosmos 论文 (重点: Tokenizer + Diffusion Transformer)
  - 了解 Genie 3 和 GWM-1 的架构差异
  - 理解 JEPA 的非生成式路线

第 9-10 天: 喷涂应用
  - 阅读 PaintRL 论文
  - 阅读 Nature 2025 喷涂轨迹优化论文
  - 设计自己的喷涂 MDP (状态/动作/奖励)

第 11-14 天: 动手实践
  - 在 Isaac Sim / PyBullet 中搭建简单喷涂环境
  - 用 DreamerV3 训练世界模型 + 策略
  - 对比 Model-free baseline (PPO/SAC)
  - 记录实验结果, 制定后续计划
```

### 8.4 关键会议与研讨会

- **ICML 2025 Workshop**: "Building Physically Plausible World Models"
- **NeurIPS / ICLR**: 世界模型相关论文的主要发表场所
- **CoRL (Conference on Robot Learning)**: 机器人学习顶会
- **NVIDIA GTC**: Cosmos 最新进展发布

### 8.5 社区与博客

- [World Models Reading List (2025)](https://medium.com/@graison/world-models-reading-list-the-papers-you-actually-need-in-2025-882f02d758a9)
- [World Models: Five Competing Approaches (Themesis, 2026)](https://themesis.com/2026/01/07/world-models-five-competing-approaches/)
- [World Models and the Sparks of Little Robotics (a16z)](https://a16z.com/world-models-and-the-sparks-of-little-robotics/)
- [Fei-Fei Li: From Words to Worlds (Substack)](https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence)
- [NVIDIA Cosmos 技术博客](https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform/)

---

## 附录: 术语表

| 术语 | 含义 |
|------|------|
| **RSSM** | Recurrent State-Space Model, DreamerV3 的核心, 融合确定性和随机性潜变量 |
| **JEPA** | Joint Embedding Predictive Architecture, LeCun 提出的非生成式世界模型 |
| **WFM** | World Foundation Model, NVIDIA 对大规模预训练世界模型的称呼 |
| **VLM** | Vision-Language Model, 视觉语言模型 |
| **CoT** | Chain-of-Thought, 思维链推理 |
| **FSQ** | Finite Scalar Quantization, Cosmos Tokenizer 的离散量化方法 |
| **AdaLN-LoRA** | Adaptive Layer Norm + Low-Rank Adaptation, 参数高效的归一化层 |
| **Sim-to-Real** | 仿真到真实的迁移, 世界模型和 RL 部署的核心挑战 |
| **MDP** | Markov Decision Process, RL 问题的数学框架 |
| **Latent Rollout** | 潜空间展开, 在学到的潜空间中模拟未来轨迹 |
| **Domain Randomization** | 域随机化, 通过随机化仿真参数来提高迁移鲁棒性 |
| **Counterfactual Generation** | 反事实生成, 探索 "如果采取不同动作会怎样" |
| **Physical AI** | 物理 AI, Gartner 2026 趋势, 指能在物理世界中感知-决策-行动的 AI 系统 |

---

> **总结**: 世界模型正从学术研究快速走向产业应用。对于喷涂路径规划, 推荐以 **DreamerV3/Dreamer 4** 为核心世界模型架构, 结合传统喷涂仿真器的物理精度, 构建 "仿真器提供数据 + 世界模型加速策略搜索 + 多模态检测闭环校正" 的技术路线。NVIDIA Cosmos 生态可作为中长期的技术升级方向, 尤其在数据增强 (Cosmos Transfer) 和质量推理 (Cosmos Reason) 方面具有直接价值。
