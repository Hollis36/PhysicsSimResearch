# 资源索引 (Resource Index)

本文档汇总了 PhysicsSimResearch 项目中提及的所有关键资源，便于快速查找。

---

## 📋 目录

- [开源代码库](#开源代码库)
- [数据集](#数据集)
- [预训练模型](#预训练模型)
- [论文分类索引](#论文分类索引)
- [学习资源](#学习资源)
- [工具和框架](#工具和框架)

---

## 💻 开源代码库

### 可微物理仿真

| 项目 | 机构 | Stars | 语言/框架 | 链接 |
|------|------|-------|-----------|------|
| NVIDIA Newton | NVIDIA + Google DeepMind | - | Python/Warp | [GitHub](https://github.com/newton-physics/newton) |
| NVIDIA Warp | NVIDIA | 3.8k+ | Python/CUDA | [GitHub](https://github.com/NVIDIA/warp) |
| Brax | Google | 2.2k+ | Python/JAX | [GitHub](https://github.com/google/brax) |
| MuJoCo MJX | DeepMind | - | Python/JAX | [GitHub](https://github.com/google-deepmind/mujoco) |
| DiffTaichi | MIT | 1.8k+ | Python/Taichi | [GitHub](https://github.com/taichi-dev/difftaichi) |
| gradSim | - | 500+ | Python/PyTorch | [GitHub](https://github.com/gradsim/gradsim) |
| PlasticineLab | MIT | 300+ | Python | [GitHub](https://github.com/hzaskywalker/PlasticineLab) |

### 物理基础模型

| 项目 | 机构 | 参数量 | 框架 | 链接 |
|------|------|--------|------|------|
| Walrus | Polymathic AI | 1.3B | PyTorch | [GitHub](https://github.com/PolymathicAI/walrus) |
| GPhyT | 独立研究 | 9.2M-796M | PyTorch | [GitHub](https://github.com/FloWsnr/General-Physics-Transformer) |
| PhysiX | UCLA | 4.5B | PyTorch | [GitHub](https://github.com/ArshKA/PhysiX) |
| Poseidon | ETH Zurich | Multi-scale | PyTorch | [GitHub](https://github.com/camlab-ethz/poseidon) |
| MPP | Polymathic AI | Multi-scale | PyTorch | [GitHub](https://github.com/PolymathicAI/multiple_physics_pretraining) |
| DPOT | 清华大学 | 7M-1B | PyTorch | [GitHub](https://github.com/thu-ml/DPOT) |
| VICON | UCLA | - | PyTorch | [GitHub](https://github.com/Eydcao/VICON) |

### GNN 粒子仿真

| 项目 | 机构 | 说明 | 框架 | 链接 |
|------|------|------|------|------|
| GNS | DeepMind | 通用粒子仿真器 | TensorFlow | [GitHub](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) |
| NeuralMPM | - | MPM启发的神经仿真 | PyTorch | [arXiv:2405.12963](https://arxiv.org/abs/2405.12963) |
| LagrangeBench | - | 拉格朗日流体基准 | JAX | [GitHub](https://github.com/tumaer/lagrangebench) |
| SEGNN | - | E(3)等变消息传递 | PyTorch | [GitHub](https://github.com/RobDHess/Steerable-E3-GNN) |
| PyTorch Geometric | - | GNN框架 | PyTorch | [GitHub](https://github.com/pyg-team/pytorch_geometric) |

### 世界模型

| 项目 | 机构 | 说明 | 框架 | 链接 |
|------|------|------|------|------|
| NVIDIA Cosmos | NVIDIA | Physical AI 平台 | PyTorch | [GitHub](https://github.com/NVIDIA/Cosmos) |
| DreamerV3 | DeepMind | 统一超参数世界模型 | TensorFlow | [GitHub](https://github.com/danijar/dreamerv3) |
| Genie 3 | Google DeepMind | 文本到3D交互环境 | - | [Blog](https://deepmind.google/discover/blog/genie-3/) |
| GraphCast | Google DeepMind | 天气预报 | JAX | [GitHub](https://github.com/google-deepmind/graphcast) |
| Aurora | Microsoft | 大气基础模型 | PyTorch | [GitHub](https://github.com/microsoft/aurora) |
| UniSim | - | 扩散模型仿真器 | PyTorch | [arXiv:2308.01253](https://arxiv.org/abs/2308.01253) |

### PINNs

| 项目 | 机构 | 说明 | 框架 | 链接 |
|------|------|------|------|------|
| DeepXDE | - | PINNs库 | TensorFlow/PyTorch/JAX | [GitHub](https://github.com/lululxvi/deepxde) |
| NVIDIA Modulus | NVIDIA | 物理ML平台 | PyTorch | [GitHub](https://github.com/NVIDIA/modulus) |
| PyDEns | - | 神经微分方程 | TensorFlow | [GitHub](https://github.com/analysiscenter/pydens) |
| NeuralOperator | - | FNO等算子实现 | PyTorch | [GitHub](https://github.com/neuraloperator/neuraloperator) |

### 多模态融合

| 项目 | 机构 | 说明 | 框架 | 链接 |
|------|------|------|------|------|
| BEVFusion | MIT | LiDAR-Camera融合 | PyTorch | [GitHub](https://github.com/mit-han-lab/bevfusion) |
| TransFusion | XMU | LiDAR-Camera融合 | PyTorch | [GitHub](https://github.com/XuyangBai/TransFusion) |
| DroneVehicle | - | RGB-IR MCWF | PyTorch | [GitHub](https://github.com/Hollis36/DroneVehicle) |
| MMDetection | OpenMMLab | 目标检测工具箱 | PyTorch | [GitHub](https://github.com/open-mmlab/mmdetection) |

---

## 📊 数据集

### 物理仿真数据集

| 数据集 | 规模 | 领域 | 格式 | 链接 |
|--------|------|------|------|------|
| **The Well** | 15TB, 16个数据集 | 多物理系统 | HDF5 | [GitHub](https://github.com/PolymathicAI/the_well) |
| PDEBench | TB级 | PDE求解 | HDF5 | [GitHub](https://github.com/pdebench/PDEBench) |
| ERA5 | PB级 | 全球大气 | GRIB/NetCDF | [ECMWF](https://cds.climate.copernicus.eu/) |
| FluidBench | - | 流体动力学 | - | 论文附带 |
| CLEVRER | - | 物理视频推理 | Video | [Website](http://clevrer.csail.mit.edu/) |

### 多模态数据集

| 数据集 | 模态 | 规模 | 任务 | 链接 |
|--------|------|------|------|------|
| nuScenes | Camera+LiDAR+Radar | 1000场景 | 自动驾驶 | [Website](https://www.nuscenes.org/) |
| KITTI | Camera+LiDAR | 200k+图像 | 3D检测 | [Website](http://www.cvlibs.net/datasets/kitti/) |
| Waymo Open | Camera+LiDAR | 1000场景 | 自动驾驶 | [Website](https://waymo.com/open/) |
| KAIST | RGB+Thermal | 95k+图像 | 目标检测 | [Website](https://soonminhwang.github.io/rgbt-ped-detection/) |
| LLVIP | RGB+Infrared | 30k+图像 | 目标检测 | [GitHub](https://github.com/bupt-ai-cz/LLVIP) |

---

## 🤖 预训练模型

### HuggingFace 模型

| 模型 | 组织 | 下载 |
|------|------|------|
| Walrus | polymathic-ai | [huggingface.co/polymathic-ai/walrus](https://huggingface.co/polymathic-ai/walrus) |
| Poseidon | camlab-ethz | [huggingface.co/camlab-ethz](https://huggingface.co/camlab-ethz) |
| GraphCast | google-deepmind | [huggingface.co/google/graphcast](https://huggingface.co/google/graphcast) |
| Aurora | microsoft | [huggingface.co/microsoft/aurora](https://huggingface.co/microsoft/aurora) |

### 其他来源

- **NVIDIA Cosmos**: [developer.nvidia.com/cosmos](https://developer.nvidia.com/cosmos)
- **DreamerV3**: [GitHub Releases](https://github.com/danijar/dreamerv3/releases)
- **PhysiX**: 论文附带链接

---

## 📚 论文分类索引

### 按会议分类

#### NeurIPS

- **2024:** MPP, DPOT, UPT, The Well
- **2023:** LagrangeBench
- **2022:** SEGNN, DMCF
- **2020:** GNS, FNO
- **2016:** Interaction Networks

#### ICML

- **2024:** NeuralMPM (→TMLR), DPOT
- **2023:** ClimaX, ICML 2023 Provable Dynamic Fusion
- **2021:** MeshGraphNets
- **2019:** DPI-Net

#### ICLR

- **2025:** SAPO, Rewarped
- **2024:** PDEformer (Workshop)
- **2022:** SEGNN, SHAC
- **2021:** FNO, MeshGraphNets

#### Nature/Science

- **2025:** Aurora (Nature)
- **2024:** GenCast (Nature)
- **2023:** Pangu-Weather (Nature), GraphCast (Science)
- **2021:** DeepONet (Nature Machine Intelligence)

### 按主题分类

#### 物理基础模型 (12篇核心论文)

1. GPhyT (2025) - [arXiv:2509.13805](https://arxiv.org/abs/2509.13805)
2. Walrus (2025) - [arXiv:2511.15684](https://arxiv.org/abs/2511.15684)
3. PhysiX (2025) - [arXiv:2506.17774](https://arxiv.org/abs/2506.17774)
4. Poseidon (2024) - [arXiv:2405.19101](https://arxiv.org/abs/2405.19101)
5. MPP (2024) - [arXiv:2310.02994](https://arxiv.org/abs/2310.02994)
6. DPOT (2024) - [arXiv:2403.03542](https://arxiv.org/abs/2403.03542)
7. UPT (2024) - [arXiv:2402.12365](https://arxiv.org/abs/2402.12365)
8. VICON (2024) - [arXiv:2411.16063](https://arxiv.org/abs/2411.16063)
9. PDEformer (2024) - [arXiv:2402.12652](https://arxiv.org/abs/2402.12652)
10. The Well (2024) - [arXiv:2412.00568](https://arxiv.org/abs/2412.00568)
11. FNO (2021) - [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
12. CompNO (2026) - [MDPI](https://www.mdpi.com/2076-3417/16/2/972)

#### 可微物理仿真 (主要综述)

- "A Review of Differentiable Simulators" (2024, IEEE Access)
- NVIDIA Newton 技术报告 (GTC 2025)
- DiffTaichi (2020, ICLR)
- gradSim (2021)

#### GNN 粒子仿真

- GNS (2020, ICML) - DeepMind
- NeuralMPM (2024, TMLR)
- LagrangeBench (2023, NeurIPS)
- SEGNN (2022, ICLR)
- MeshGraphNets (2021, ICLR)

#### 世界模型

- Genie 3 (2025) - Google DeepMind Blog
- NVIDIA Cosmos (2025) - GTC 2025
- DreamerV3 (2023)
- GraphCast (2023, Science)
- Aurora (2025, Nature)
- GenCast (2024, Nature)

#### PINNs

- PINNs 原始论文 (2019, JCP) - Raissi et al.
- DeepXDE (2021)
- FNO (2021, ICLR)
- DeepONet (2021, Nature MI)
- PINNs for Plasma Spraying (2025, J. Thermal Spray Tech.)

#### 多模态融合

- BEVFusion (2022, CVPR)
- TransFusion (2022, CVPR)
- FDSNet (2025)
- MCWF - DroneVehicle (自有项目)

---

## 🎓 学习资源

### 在线课程

- **Stanford CS 348C**: Physics-Based Animation
- **MIT 6.838**: Shape Analysis
- **DeepMind x UCL**: Deep Learning Lecture Series
- **Fast.ai**: Practical Deep Learning
- **Coursera**: Machine Learning Specialization (Andrew Ng)

### 书籍

- "Physics-Based Animation" - Erleben et al.
- "Computational Physics" - Giordano & Nakanishi
- "Deep Learning" - Goodfellow, Bengio, Courville
- "Reinforcement Learning: An Introduction" - Sutton & Barto

### 视频博主

- **Two Minute Papers**: AI论文速览
- **Yannic Kilcher**: 论文精读
- **3Blue1Brown**: 数学直觉可视化
- **Arxiv Insights**: 深度学习讲解

### 博客和网站

- **Distill.pub**: 交互式论文
- **Papers with Code**: 论文+代码
- **Towards Data Science**: 技术博客
- **arXiv**: 预印本论文

---

## 🛠️ 工具和框架

### 深度学习框架

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| **PyTorch** | 生态丰富，易用 | 通用深度学习 |
| **JAX** | 自动求导强大 | 可微物理仿真 |
| **TensorFlow** | 工业部署 | 生产环境 |

### 物理仿真引擎

| 引擎 | 类型 | 可微分 | 开源 |
|------|------|--------|------|
| **MuJoCo** | 刚体+接触 | 是 (MJX) | 是 |
| **Brax** | 刚体 | 是 | 是 |
| **Taichi** | 通用可编程 | 是 | 是 |
| **Warp** | GPU加速 | 是 | 是 |
| **PhysX** | 游戏物理 | 否 | 是 |
| **OpenFOAM** | CFD | 否 | 是 |

### 图神经网络

- **PyTorch Geometric (PyG)**
- **DGL (Deep Graph Library)**
- **Jraph (JAX)**

### 实验管理

- **Weights & Biases**: 实验追踪
- **MLflow**: 模型管理
- **TensorBoard**: 可视化
- **Neptune.ai**: 团队协作

### 数据处理

- **HDF5**: 大规模科学数据
- **Zarr**: 云原生数组存储
- **Xarray**: 多维标记数组
- **Pandas**: 表格数据

---

## 🔗 相关社区

### GitHub 组织

- [PolymathicAI](https://github.com/PolymathicAI) - The Well, Walrus
- [Google DeepMind](https://github.com/google-deepmind) - GNS, GraphCast
- [NVIDIA](https://github.com/NVIDIA) - Newton, Warp, Modulus
- [OpenMMLab](https://github.com/open-mmlab) - MMDetection

### 会议

- **NeurIPS**: Neural Information Processing Systems
- **ICML**: International Conference on Machine Learning
- **ICLR**: International Conference on Learning Representations
- **CVPR**: Computer Vision and Pattern Recognition
- **SIGGRAPH**: Computer Graphics

### 邮件列表

- Physics-ML Google Group
- JAX Discussions
- PyTorch Forums

---

## 📦 一键安装脚本

### 基础环境

```bash
# 创建虚拟环境
conda create -n physics-sim python=3.10
conda activate physics-sim

# 安装 PyTorch (根据 CUDA 版本调整)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装 JAX (根据 CUDA 版本调整)
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 通用科学计算
pip install numpy scipy matplotlib pandas jupyterlab
```

### 专项工具

```bash
# 可微物理
pip install brax mujoco taichi

# GNN
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv

# PINNs
pip install deepxde

# 实验管理
pip install wandb mlflow tensorboard

# 数据处理
pip install h5py zarr xarray
```

---

## 📧 更新与反馈

本索引持续更新中，如有遗漏或错误，欢迎提交 Issue 或 PR。

**最后更新:** 2026-02-17
