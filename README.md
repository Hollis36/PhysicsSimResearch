# Physics Simulation Frontier Research

前沿物理仿真与智能感知技术调研资料库，涵盖6个核心研究方向。

> 原始调研: 2026-02-04 · **前沿更新: 2026-06-17** (每个方向新增一节 `🔄 最新进展更新 (2026-02 → 2026-06)`,内容均附可验证来源)

## 研究方向

| 目录 | 方向 | 简介 |
|------|------|------|
| `01_differentiable_physics/` | 可微物理仿真 | NVIDIA **Newton 1.0 GA**/Warp 1.14, **Genesis World 1.0**, DiffTaichi, Brax, MuJoCo MJX |
| `02_physics_foundation_models/` | 物理基础模型 | GPhyT, PhysiX, Walrus, Poseidon, The Well, **RealPDEBench**, **FourCastNet 3** |
| `03_gnn_particle_simulation/` | GNN粒子仿真 | GNS, NeuralMPM, **WorldParticle**, **NeuralDEM**, Dynami-CAL GraphNet |
| `04_world_models/` | 世界模型 | **NVIDIA Cosmos 3**, Genie, DreamerV3/4, JEPA/AMI Labs, RISE |
| `05_pinns/` | 物理信息神经网络 | DeepXDE, PhysicsNeMo v26.05, FNO, DeepONet, **KANO**, **Transolver-3**, PIKANs |
| `06_multimodal_fusion/` | 多模态融合 | BEVFusion, TransFusion, **4D Radar-Camera**, 不确定性/可靠性融合, MCWF扩展 |

## 🔄 2026-06 更新概览

> 2026-02 的调研冻结于该月;下表为 2026-06-17 增补的各方向"最有分量"的前沿变化。完整内容与来源见各目录 README 顶部的 `🔄 最新进展更新` 一节。

| 方向 | 本次最关键更新 | 对喷涂/检测项目的意义 |
|------|----------------|----------------------|
| 可微物理 | **Newton 已从 beta 转正为 GA (v1.3.0, GTC 2026, 252x/475x vs MJX)**;Warp→1.14;新增 Genesis World 1.0 | 完整机器人仿真后端已可用;iMPM/VBD+可微是最近的"沉积式"仿真基底 |
| 物理基础模型 | **Walrus 仿真→实验室湍流零样本迁移**;RealPDEBench (ICLR 2026 oral) 证实"仿真预训练助真实" | 直接支撑"少量 CFD 预训练 → IR/传感器微调"的喷涂代理路线 |
| GNN 粒子仿真 | **WorldParticle**:统一 Transformer 覆盖 6 类材料的拉格朗日仿真器 | 单模型跨流体/颗粒/固体;悬浮液流变 GNN 与涂料直接相关 |
| 世界模型 | **NVIDIA Cosmos 3** (完全开放 omnimodel,原生输出动作);RISE/World-Gymnast (世界模型内做 RL) | Cosmos 3 可作合成数据/world-action 主干;RISE 的"想象中 RL+膜厚式奖励"可直接迁移 |
| PINNs | **KANO** (ICLR 2026, 符号可解释算子);Transolver-3/GIST (工业级几何) | 更适合变系数 PDE 与大网格喷涂气流代理 |
| 多模态融合 | **LER-YOLO** (可靠性感知 MoE 路由);IIBalance (信息预算);4D 雷达+相机成热点 | 正是 MCWF 静态权重→自适应/不确定性权重的升级方向 |

> 说明: 各方向均**诚实标注了"空白"** —— 截至 2026 年中,仍缺少专门面向**喷涂/涂层**的可微分仿真、粒子级学习仿真、CFD 基础模型与"RGB-IR×喷涂 CFD"交叉工作,这些恰是本研究可切入的开放机会。

## 关联项目

- [MFARainbowNet](https://github.com/Hollis36/MFARainbowNet) - 基于深度强化学习的喷涂覆盖路径规划
- [DroneVehicle](https://github.com/Hollis36/DroneVehicle) - RGB-IR 多模态目标检测 (MCWF)

## 内容说明

每个研究方向均包含：
- 领域概述与发展脉络
- 核心方法详细对比
- 关键论文清单 (共计 190+ 篇; 2026-06 增补约 30 篇经核验的前沿新论文)
- 与现有项目的对接方案
- 开源代码资源汇总
- 推荐学习路线
