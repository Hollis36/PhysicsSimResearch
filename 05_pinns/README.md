# Physics-Informed Neural Networks (PINNs) 深度研究笔记

> 面向喷涂仿真与路径规划研究者的 PINNs 综合参考文档
> 最后更新: **2026-06-17** (原调研 2026-02)

---

## 目录

0. [🔄 最新进展更新 (2026-02 → 2026-06)](#-最新进展更新-2026-02--2026-06)
1. [PINNs 原理详解](#1-pinns-原理详解)
2. [与传统数值方法 (FEM/FDM/FVM) 对比](#2-与传统数值方法-femfdmfvm-对比)
3. [主要框架与工具对比](#3-主要框架与工具对比)
4. [算子学习 vs PINNs 对比](#4-算子学习-vs-pinns-对比)
5. [关键论文列表](#5-关键论文列表)
6. [PINNs 的局限性与解决方案](#6-pinns-的局限性与解决方案)
7. [最新进展 (2024-2026)](#7-最新进展-2024-2026)
8. [喷涂物理方程的 PINN 求解方案](#8-喷涂物理方程的-pinn-求解方案)
9. [代码示例框架](#9-代码示例框架)
10. [推荐入门路线](#10-推荐入门路线)

---

## 🔄 最新进展更新 (2026-02 → 2026-06)

> 本节于 **2026-06-17** 增补,记录自原调研 (2026-02) 以来的前沿进展。每条均附可验证来源;明确区分 **[已录用/已验证]**、**[预印本/评审中]** 与 **[需谨慎/已撤稿]**。

### 0.1 新方法/新论文 (PINNs, PIKANs)

| 方法 | 会议/状态 | 日期 | arXiv | 一句话 |
|------|----------|------|-------|--------|
| **Frozen-PINN** (无梯度下降的快速精确 PINN) | **ICLR 2026** [已录用] | 2026 (v1 2024-05) | [2405.20836](https://arxiv.org/abs/2405.20836) | 时空分离 PINN,用**随机特征代替梯度下降**,因果性由构造保证;9 个 PDE 基准 (极端对流/激波/高维) 精度与效率提升"数个数量级" |
| **Mesh Field Theory** (mesh 物理的端口哈密顿表述) | **ICML 2026** [已录用] | 2026 | (主 arXiv 未定位) | 证明满足局域性+置换等变+定向协变+能量耗散的 mesh 动力学**归约为端口哈密顿形式**,由 mesh 关联矩阵确定 (结构性理论结果) |
| **PEST** (Physics-Enhanced Swin Transformer, 3D 湍流) | 预印本 | 2026-02 | [2602.10150](https://arxiv.org/html/2602.10150) | 窗口注意力代理 + **频域自适应损失 + N-S 残差 + 无散度正则**,针对小尺度湍流结构 —— **CFD 相关** |
| **Lang-PINN** (从语言到 PINN) | 评审中 (ICLR 2026) | 2025-10 | [2510.05158](https://arxiv.org/pdf/2510.05158) | 多智能体 LLM 框架,从自然语言任务描述构建可训练 PINN (PDE 解析→架构→代码→精修) |

> PIKANs: 窗口内**未见**面向多物理 PDE 的重要*通用* PIKAN 新方法;2026 的 KAN-PINN 活动多为领域专用 (金融 RL、UAV 信道),超出范围。KAN 的相关进展体现在算子侧 (KANO,见 §0.3)。

### 0.2 框架版本更新 (老 → 新)

| 框架 | 原笔记 | 当前 (已验证) | 日期 | 来源 |
|------|--------|---------------|------|------|
| **DeepXDE** | v1.15.0 | **v1.15.0 (未变,无 2026 新版)** | 2024-12-05 | [GitHub](https://github.com/lululxvi/deepxde/releases) |
| **NVIDIA PhysicsNeMo** (`nvidia-physicsnemo`) | "2025 更名",无版本 | **v2.1.1** (语义版) / 框架 **v26.05** (日期版) | 2026-06-08 / 2026 | [PyPI](https://pypi.org/project/nvidia-physicsnemo/) · [release notes](https://docs.nvidia.com/physicsnemo/latest/release-notes/index.html) |
| **NeuralOperator** | v2.0.0 | **v2.0.0 (未变,仍为最新)** | 2024-10-22 | [GitHub](https://github.com/neuraloperator/neuraloperator/releases) |

> - PhysicsNeMo 现有**两套并行版本号** (PyPI 语义 `2.x` vs 日期 `25.xx/26.xx`),均为官方;日期版 v26.05 新增 Sym 内核、重构 CFD 模块、多数据集 mesh 训练、外流/underfill 流范例。
> - **DeepXDE 与 NeuralOperator 版本号自原笔记以来未变**;但 NeuralOperator v2.0.0 已内含原笔记表格外的架构: **Codano、OTNO、LocalNO、Tensor-GaLore、Mollified GNO、Fourier-Continuation 层** —— 建议补入 FNO 家族清单。

### 0.3 新算子学习架构

| 架构 | 会议/状态 | 日期 | arXiv | 一句话 |
|------|----------|------|-------|--------|
| **KANO** (Kolmogorov-Arnold Neural Operator) | **ICLR 2026** [已录用] | 2026-02 (v6) | [2509.16825](https://arxiv.org/abs/2509.16825) | 谱 + 空间双算子,带**符号可解释性**;修复 FNO 在变系数/位置相关 PDE 上的弱点 |
| **Transolver-3** (工业级几何的 Transformer 求解器) | 预印本 | 2026-02-04 | [2602.04940](https://arxiv.org/abs/2602.04940) | 通过更快 slice/deslice + 几何分区 + 随机子集训练扩展 PDE 求解器,处理 **>1.6 亿网格单元** (飞机/汽车设计) |
| **GIST** (Gauge-Invariant Spectral Transformer) | 预印本 (IBM) | 2026-04-20 | [2604.18491](https://arxiv.org/abs/2604.18491) | 图神经算子 + 谱化 mesh 连通嵌入;离散化不变、线性扩展;赛车 RANS 数据集 SOTA,交互式气动设计 |

### 0.4 新综述

- **[已验证] "Learning PDE Solvers with Physics and Data: A Unifying View of PINNs and Neural Operators"** (Dai et al., arXiv:[2601.14517](https://arxiv.org/abs/2601.14517), v2 2026-02-18):沿"学什么 / 如何嵌入物理 / 计算如何在实例间分布"三轴统一 PINN 与算子学习 —— 与本笔记的 PINN-vs-算子框架高度契合。
- **[需谨慎] "PINNs and Neural Operators for Parametric PDEs"** (arXiv:[2511.04576](https://arxiv.org/abs/2511.04576),v3 2026-01-30):**明确是 AI 生成/"人机协作"综述**,提交至 AI-Scientists track —— 引用时须加此免责声明,任何具体论断需另行核实。

### 0.5 与喷涂/CFD/传热相关

- **[需谨慎/已撤稿] PI-JEPA** (arXiv:[2604.01349](https://arxiv.org/abs/2604.01349), 2026-04-01):面向**算子分裂式耦合多物理**的无标签预训练 (压力/输运/反应子模块,Lie-Trotter 分解),概念上贴近喷涂的 N-S + 对流扩散 + 沉积栈。⚠️ **已于 v4 (2026-06-04) 撤稿** —— 仅作思路,勿引用其数字。
- **最贴近喷涂 CFD 的可迁移方向**: **GIST / Transolver-3** (§0.3,替代复杂/工业几何 CFD)、**PEST** (§0.1,含 N-S 残差 + 无散度约束的湍流代理,适合高 Re 喷涂气流)。
- **传热逆问题**: **HeatTransFormer** (arXiv:[2512.02618](https://arxiv.org/abs/2512.02618),2025-12):物理引导 Transformer (Laplace 激活、无掩码注意力) 求界面主导扩散逆问题,与"涂层-基材"热逆辨识直接同构 (⚠️ 日期在窗口前)。
- **空白提示 (诚实标注)**: 未发现专门针对喷涂/热喷涂的 2026 神经算子代理;喷涂路径规划仍以经典方法 (PSO + 沉积模型) 为主,原笔记的等离子喷涂 PINN 工作 (J. Therm. Spray Technol./ITSC) 仍是该细分 SOTA。算子分裂多物理 (PI-JEPA 思路)、PEST、Transolver-3/GIST 是最可迁移方向。

---

## 1. PINNs 原理详解

### 1.1 核心思想

PINNs (Physics-Informed Neural Networks) 由 **Raissi, Perdikaris & Karniadakis** 于 2019 年在 *Journal of Computational Physics* 上正式提出。其核心思想是：**用神经网络作为 PDE 解的函数逼近器，同时将物理定律（偏微分方程）嵌入到损失函数中作为约束**。

传统深度学习 = 纯数据驱动
PINNs = **数据驱动 + 物理驱动** 的混合范式

```
输入: (x, y, z, t) 时空坐标
  |
  v
┌─────────────────────────┐
│   全连接神经网络 (MLP)     │
│   u_nn(x,y,z,t; theta)  │
│                         │
│   隐藏层: tanh/sin 激活   │
│   一般 4-8 层, 每层 20-256 │
└────────────┬────────────┘
             |
             v
输出: u, v, w, p, T, ... (物理场量)
```

### 1.2 损失函数的三个组成部分

PINNs 的损失函数是其最关键的创新，包含三个（或更多）部分：

```
L_total = w_data * L_data + w_pde * L_pde + w_bc * L_bc + w_ic * L_ic
```

#### (1) 数据损失 L_data（观测数据拟合）

在有观测数据的点上，网络输出与实测值的均方误差：

```
L_data = (1/N_data) * sum_i || u_nn(x_i, t_i) - u_obs_i ||^2
```

- N_data: 观测数据点数量
- u_obs: 实验/传感器测量数据
- 可以是稀疏、噪声数据

#### (2) 物理损失 L_pde（PDE 残差约束）

在配点（collocation points）上计算 PDE 残差，要求其为零：

```
L_pde = (1/N_f) * sum_i || F[u_nn](x_i, t_i) ||^2
```

其中 F[u] 是 PDE 算子。以 Navier-Stokes 方程为例：

```
F_1 = du/dt + u*du/dx + v*du/dy + dp/dx - nu*(d2u/dx2 + d2u/dy2)
F_2 = dv/dt + u*dv/dx + v*dv/dy + dp/dy - nu*(d2v/dx2 + d2v/dy2)
F_3 = du/dx + dv/dy  (连续性方程)
```

关键技术：**自动微分 (Automatic Differentiation)**
- 神经网络对输入的导数可通过自动微分精确计算
- 不需要有限差分近似，无网格离散误差
- PyTorch: `torch.autograd.grad()`
- TensorFlow: `tf.GradientTape()`

#### (3) 边界/初始条件损失 L_bc + L_ic

```
L_bc = (1/N_bc) * sum_i || u_nn(x_i^bc, t_i) - g(x_i^bc, t_i) ||^2   [Dirichlet]
L_bc = (1/N_bc) * sum_i || du_nn/dn(x_i^bc, t_i) - h(x_i^bc, t_i) ||^2 [Neumann]
L_ic = (1/N_ic) * sum_i || u_nn(x_i, t_0) - u_0(x_i) ||^2
```

#### 损失权重的关键问题

各项损失的权重 w_data, w_pde, w_bc 的选择至关重要：
- 权重不当会导致训练失败（某一项主导优化过程）
- 解决方案：
  - **自适应权重** (NTK-based, gradient balancing)
  - **GradNorm** 方法动态调整
  - **对抗训练** (Competitive PINNs)

### 1.3 训练流程

```
1. 定义计算域 Omega 和边界 dOmega
2. 在域内随机采样配点 {x_f} (collocation points)
3. 在边界/初始条件上采样 {x_bc}, {x_ic}
4. (可选) 准备观测数据 {x_data, u_obs}
5. 构建神经网络 u_nn(x; theta)
6. 前向传播: 计算 u_nn, 通过自动微分计算导数
7. 计算 L_total = w_pde*L_pde + w_bc*L_bc + w_ic*L_ic + w_data*L_data
8. 反向传播: 更新网络参数 theta
9. 重复 6-8 直到收敛
   - 通常使用 Adam 预训练 + L-BFGS 精调
```

### 1.4 两类问题

| 类型 | 描述 | 已知量 | 未知量 |
|------|------|--------|--------|
| **正问题 (Forward)** | 已知方程和条件，求解 | PDE, BC, IC | 解 u(x,t) |
| **反问题 (Inverse)** | 已知部分观测，推断参数 | 部分 u_obs, PDE 形式 | PDE 参数 (如粘度 nu) |

PINNs 在反问题上的优势尤为显著 -- 传统方法需要专门的反演算法，而 PINNs 只需将未知参数作为可训练变量即可。

---

## 2. 与传统数值方法 (FEM/FDM/FVM) 对比

### 2.1 综合对比表

| 特性 | PINNs | FEM | FDM | FVM |
|------|-------|-----|-----|-----|
| **网格需求** | 无网格 (mesh-free) | 需要网格 | 需要结构化网格 | 需要网格 |
| **复杂几何适应性** | 好 (无网格) | 好 (非结构网格) | 差 (需结构网格) | 中等 |
| **高维问题** | 有优势 | 维度灾难 | 维度灾难 | 维度灾难 |
| **正问题精度** | 中等 (~0.1% 相对误差) | 高 | 中等 | 高 |
| **反问题能力** | 优秀 (天然支持) | 需专门算法 | 需专门算法 | 需专门算法 |
| **数据稀疏场景** | 优秀 | 不适用 | 不适用 | 不适用 |
| **计算成本 (小问题)** | 高 (训练开销) | 低 | 最低 | 低 |
| **计算成本 (大/复杂问题)** | 可能竞争 | 高 | 中等 | 高 |
| **实时推理** | 毫秒级 (训练后) | 分钟-小时 | 秒-分钟 | 分钟-小时 |
| **易实现性** | 中等 (框架支持) | 较难 | 最简单 | 中等 |
| **成熟度** | 新兴 (2019-) | 非常成熟 | 非常成熟 | 非常成熟 |
| **湍流处理** | 困难 | 成熟 (RANS/LES) | 有限 | 成熟 |
| **多物理场耦合** | 损失函数需精心设计 | 成熟框架 | 有限 | 成熟框架 |

### 2.2 PINNs 的核心优势场景

1. **反问题与参数识别** -- 从稀疏传感器数据反推物理参数（如热导率、粘度）
2. **数据同化** -- 融合物理模型和实验数据
3. **参数化代理模型** -- 训练后可实时预测不同参数下的结果
4. **高维问题** -- 避免网格方法的维度灾难
5. **不规则/动态域** -- 无需重新生成网格

### 2.3 PINNs 不适合的场景

1. **精度要求极高的标准正问题** -- FEM/FVM 仍然更精确可靠
2. **大规模湍流模拟** -- PINNs 目前难以处理宽频谱湍流
3. **需要严格误差界的工程认证** -- PINNs 缺乏传统方法的收敛性理论保证
4. **简单低维问题** -- 训练开销使其得不偿失

### 2.4 混合策略 (最有前景的方向)

```
传统求解器 (粗网格/低精度) --> 生成训练数据 --> PINNs 超分辨/插值
PINNs (快速近似解) --> 作为 Newton-Raphson 迭代初值 --> FEM 精调
FEM 离线计算 --> 训练 Neural Operator --> 在线实时推理
```

---

## 3. 主要框架与工具对比

### 3.1 框架总览

| 框架 | 开发者 | 后端 | 许可证 | Star数 | 核心特点 |
|------|--------|------|--------|--------|----------|
| **DeepXDE** | Lu Lu (Yale) | TF/PyTorch/JAX/Paddle | Apache 2.0 | ~2.8k | 最全面的 PINN 库，支持多种 PDE 类型 |
| **NVIDIA PhysicsNeMo** | NVIDIA | PyTorch | Apache 2.0 | ~2k | 工业级，GPU 优化，数字孪生 |
| **NeuralOperator** | NVIDIA/Caltech | PyTorch | BSD | ~1.5k | FNO/DeepONet 官方实现 |
| **SciANN** | E. Haghighat | TF/Keras | MIT | ~300 | 简洁 API，适合入门 |
| **NVIDIA PhysicsNeMo** | NVIDIA | PyTorch | Apache 2.0 | - | 工业 HPC 集成 |
| **JAX-PI** | - | JAX | MIT | ~200 | JAX 生态，高性能 |
| **PyDEns** | - | PyTorch | MIT | ~100 | 轻量级 |

### 3.2 DeepXDE 详解

**GitHub**: https://github.com/lululxvi/deepxde
**文档**: https://deepxde.readthedocs.io/
**论文**: Lu et al., "DeepXDE: A deep learning library for solving differential equations," *SIAM Review*, 63(1), 208-228, 2021.
**最新版本**: v1.15.0

#### 核心能力

- **PDE 类型**: ODE, PDE, IDE (积分微分方程), fPDE (分数阶 PDE)
- **问题类型**: 正问题, 反问题, 算子学习 (DeepONet)
- **几何支持**: Interval, Rectangle, Polygon, Disk, Cuboid, Sphere, CSG 布尔运算, 点云
- **边界条件**: Dirichlet, Neumann, Robin, Periodic, OperatorBC
- **网络架构**: FNN, ResNet, Multi-scale Fourier Feature Network
- **自动微分**: 反向模式, 前向模式, ZCS
- **后端**: TensorFlow 1.x/2.x, PyTorch, JAX, PaddlePaddle
- **特色算法**:
  - 残差自适应采样 (RAR)
  - 梯度增强 PINN (gPINN)
  - 硬约束 PINN (hPINN)
  - Multi-scale Fourier Features
  - DeepONet / POD-DeepONet / MIONet

#### 安装

```bash
pip install deepxde
# 选择后端
export DDE_BACKEND=pytorch  # 或 tensorflow, jax, paddle
```

### 3.3 NVIDIA PhysicsNeMo 详解

**前身**: NVIDIA SimNet (2020) -> NVIDIA Modulus (2021) -> NVIDIA PhysicsNeMo (2025)
**GitHub**: https://github.com/NVIDIA/physicsnemo
**论文**: arXiv:2012.07938 (SimNet)

#### 核心能力

- **面向工业级应用**: 数字孪生, CFD 加速, 参数化设计优化
- **多种模型**: PINNs, FNO, GNN, 扩散模型
- **GPU 优化**: 深度集成 NVIDIA GPU，支持多节点分布式训练
- **几何引擎**: 内置参数化几何 + Tessellated 几何
- **案例**:
  - Coanda 效应模拟 (CFD 替代)
  - 飞机结构疲劳裂纹预测 (500 架飞机，10 秒内)
  - 散热器设计优化

#### 安装

```bash
pip install nvidia-physicsnemo
# 或使用 Docker
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:<tag>
```

#### 迁移注意

```python
# 旧代码
import modulus
# 新代码
import physicsnemo
```

### 3.4 NeuralOperator 库

**GitHub**: https://github.com/neuraloperator/neuraloperator
**版本**: 2.0.0
**开发者**: NVIDIA + Caltech

#### 包含模型

| 模型 | 全称 | 特点 |
|------|------|------|
| FNO | Fourier Neural Operator | 频域全局卷积，分辨率不变 |
| TFNO | Tensor FNO | 张量分解降低内存 |
| SFNO | Spherical FNO | 球面问题 (气象) |
| GINO | Geometry-Informed NO | 复杂几何 |
| UQNO | Uncertainty Quantification NO | 不确定性量化 |
| RNO | Recurrent NO | 时间序列 |

```bash
pip install neuraloperator
```

---

## 4. 算子学习 vs PINNs 对比

### 4.1 核心区别

```
PINNs: 学习 一个 PDE 实例的解
       输入: (x, t) --> 输出: u(x, t)
       每组新的参数/条件都需要重新训练

算子学习: 学习从输入函数到输出函数的 映射算子
         输入: 初始条件/边界条件/参数场 f(x) --> 输出: 解 u(x, t)
         训练一次，可泛化到新的输入函数
```

### 4.2 详细对比

| 特性 | PINNs | FNO | DeepONet |
|------|-------|-----|----------|
| **学习目标** | 单一 PDE 的解 | 参数到解的映射算子 | 输入函数到输出函数的算子 |
| **训练数据** | 可以无数据 (纯物理) | 需要 (输入,输出) 数据对 | 需要 (输入函数, 输出函数) 对 |
| **泛化性** | 弱 (特定问题) | 强 (参数/分辨率泛化) | 强 (函数空间泛化) |
| **分辨率** | 固定 | 分辨率不变 | 灵活 (任意查询点) |
| **推理速度** | 毫秒级 | 毫秒级 | 毫秒级 |
| **物理保证** | 强 (内嵌 PDE) | 弱 (纯数据驱动) | 弱 (可加物理约束) |
| **训练成本** | 中等 | 高 (需要大量数据) | 高 (需要大量数据) |
| **复杂几何** | 好 | 差 (需规则网格做 FFT) | 好 (任意点查询) |
| **理论基础** | 万能逼近定理 | 万能算子逼近定理 | 算子万能逼近定理 (Chen & Chen, 1995) |

### 4.3 Fourier Neural Operator (FNO)

**论文**: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations," ICLR 2021
**核心架构**:

```
输入 v(x) --> Lift (P) --> [Fourier Layer] x L --> Projection (Q) --> 输出 u(x)

每个 Fourier Layer:
  v(x) --> FFT --> 截断高频 --> 可学习权重矩阵 R --> IFFT --> + 局部线性变换 W --> 激活 sigma
```

**关键特点**:
- 在频域学习全局特征，计算复杂度 O(N log N)
- 分辨率不变：在低分辨率训练，高分辨率推理（zero-shot super-resolution）
- 相比传统求解器加速可达 1000 倍
- **局限**: 频谱偏差（倾向学低频），需要规则网格

### 4.4 DeepONet

**论文**: Lu et al., "Learning nonlinear operators via DeepONet," *Nature Machine Intelligence*, 3, 218-229, 2021
**GitHub**: https://github.com/lululxvi/deeponet
**核心架构**:

```
Branch Net (编码输入函数):
  [u(x_1), u(x_2), ..., u(x_m)] --> DNN --> [b_1, b_2, ..., b_p]

Trunk Net (编码输出位置):
  (y) --> DNN --> [t_1, t_2, ..., t_p]

输出: G(u)(y) = sum_k b_k * t_k + bias
```

**两种变体**:
- **Stacked DeepONet**: p 个独立的 Branch Net + 1 个 Trunk Net (定理1)
- **Unstacked DeepONet**: 1 个 Branch Net + 1 个 Trunk Net，参数共享 (定理2)

**关键特点**:
- 可在任意点查询输出（不限于网格）
- Branch Net 可替换为 CNN/ResNet 等
- 可扩展为 POD-DeepONet (降阶基 + DeepONet)
- 支持复杂几何

### 4.5 喷涂仿真中的选择建议

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 求解单一喷涂工况的流场 | PINNs | 无需训练数据，直接嵌入 N-S 方程 |
| 参数扫描 (不同喷距、角度) | FNO / DeepONet | 训练一次，参数泛化 |
| 逆问题 (从涂层厚度反推工艺参数) | PINNs (反问题模式) | 天然支持，无需专门算法 |
| 实时路径规划中的涂层预测 | DeepONet | 毫秒推理，任意查询点 |
| 从 CFD 数据构建代理模型 | FNO | 利用已有仿真数据，分辨率泛化 |

---

## 5. 关键论文列表

### 5.1 奠基性工作

| 年份 | 论文 | 作者 | 贡献 |
|------|------|------|------|
| 2017 | Physics Informed Deep Learning (Part I & II) | Raissi et al. | PINNs 概念提出 (arXiv: 1711.10561, 1711.10566) |
| 2019 | Physics-informed neural networks: A deep learning framework for solving forward and inverse problems | Raissi, Perdikaris, Karniadakis | PINNs 正式论文 (*J. Comput. Phys.*, 378, 686-707) |
| 2019 | DeepONet: Learning nonlinear operators | Lu et al. | 深度算子网络 (arXiv: 1910.03193) |
| 2020 | NSFnets: Navier-Stokes Flow Nets | Jin et al. | PINNs 求解 NS 方程 (*J. Comput. Phys.*) |
| 2021 | Fourier Neural Operator for Parametric PDEs | Li et al. | FNO (ICLR 2021) |
| 2021 | DeepONet (Nature Machine Intelligence) | Lu et al. | DeepONet 正式版 (*Nat. Mach. Intell.*, 3, 218-229) |
| 2021 | DeepXDE: A deep learning library | Lu et al. | DeepXDE 工具论文 (*SIAM Review*, 63(1), 208-228) |
| 2021 | PINNs for Heat Transfer Problems | Cai et al. | 热传递问题 PINN (*J. Heat Transfer*, ASME) |

### 5.2 方法论改进

| 年份 | 论文 | 贡献 |
|------|------|------|
| 2020 | NVIDIA SimNet (arXiv: 2012.07938) | 工业级多物理场 PINN 框架 |
| 2021 | When and why PINNs fail to train | 训练失败的诊断与分析 (Wang et al.) |
| 2022 | Competitive Physics Informed Networks (arXiv: 2204.11144) | 对抗训练提升精度 4 个数量级 |
| 2022 | Respecting Causality for Training PINNs | Wang et al. 因果 PINN |
| 2022 | Scientific ML Through PINNs: Where We Are | Cuomo et al. 综合综述 (*J. Sci. Comput.*) |
| 2023 | An Expert's Guide to Training PINNs (arXiv: 2308.08468) | 训练技巧全面指南 |
| 2023 | Finite Basis PINNs (FBPINNs) | 域分解 PINN 解决多尺度问题 |

### 5.3 最新前沿 (2024-2026)

| 年份 | 论文 | 贡献 |
|------|------|------|
| 2024 | From PINNs to PIKANs (arXiv: 2410.13228) | KAN 替代 MLP 的 PINN 综述 |
| 2024 | PirateNets | 自适应残差连接的 PINN 架构 |
| 2024 | Adversarial Adaptive Sampling (ICLR 2024) | 最优传输理论自适应采样 |
| 2024 | TL-DPINN (IJCAI 2024) | 因果增强离散 PINN + 迁移学习 |
| 2024 | PINNs for Plasma Spraying | 等离子喷涂粒子属性预测 (ITSC 2024) |
| 2024 | Dual Cone Gradient Descent (NeurIPS 2024) | PINN 新优化方法 |
| 2025 | ConFIG (ICLR 2025) | 无冲突 PINN 训练 |
| 2025 | ANaGRAM (ICLR 2025) | 自然梯度高效 PINN 学习 |
| 2025 | Knowledge Distillation for PINNs | 知识蒸馏自动发现网络结构 (*Nat. Commun.*) |
| 2025 | NeuralOperator 2.0 Library Paper | 算子学习统一框架 |
| 2025 | Comprehensive Review: PINNs in Heat Transfer Multiphysics | 热传导主导多物理场综述 |
| 2025 | SPIKANs | 可分离 PIKAN 降低高维计算复杂度 |
| 2026 | PINNs as PDE Forward Solvers (综述, *Tsinghua Sci. Tech.*) | 面向问题的 PINN 综述 |
| 2026 | WHC-PINN (*Scientific Reports*) | 加权硬约束求解双曲方程 |

### 5.4 喷涂/涂层相关

| 年份 | 论文 | 贡献 |
|------|------|------|
| 2021 | ANN for Cold Spray Multi-layer Profile | 冷喷涂多层轮廓预测 (*J. Therm. Spray Technol.*) |
| 2023 | PINN-CNN Hierarchical NN for HVOF Coatings (Gui et al.) | PINN+CNN 分层预测粒子温度速度与涂层性能 |
| 2024 | PINNs for Plasma Spraying Particle Properties (ITSC 2024) | 首次将 PINNs 应用于 APS 粒子性能预测 |
| 2025 | ML for Tribological Property Prediction of Thermal Spray Coatings (综述) | 机器学习预测热喷涂涂层摩擦学性能综述 |
| 2021 | PINNs for Heat Conduction: Fireproof Coating Inverse Problems | 防火涂层热传导参数反演 |

---

## 6. PINNs 的局限性与解决方案

### 6.1 主要问题

#### 问题一：频谱偏差 (Spectral Bias)

**现象**: 神经网络倾向于先学习低频分量，高频分量收敛极慢
**影响**: 对多尺度问题（如喷涂中的小尺度涡结构）表现差

**解决方案**:
| 方案 | 原理 | 参考 |
|------|------|------|
| Multi-scale Fourier Features | 输入坐标映射到多尺度傅里叶特征 | Wang et al., 2021 |
| Random Fourier Features | 随机频率的正弦/余弦嵌入 | Tancik et al., 2020 |
| FRES | 动态傅里叶嵌入 + 多层残差校正 | 2025 |
| k-PINN | 波数域 (k-space) 中求解 | 2024 |
| 强边界条件 PINN | 延缓高频退化 | 2024 |

#### 问题二：损失函数失衡

**现象**: PDE 残差项与边界/数据项量级差异大，导致训练被某一项主导
**影响**: 网络忽略部分约束，解不满足全部物理条件

**解决方案**:
| 方案 | 原理 |
|------|------|
| NTK 自适应加权 | 基于 Neural Tangent Kernel 特征值调整权重 |
| GradNorm | 梯度范数归一化 |
| Learning Rate Annealing | 损失学习率退火 |
| Competitive PINNs | 对抗训练，判别器奖励发现 PINN 错误 |
| ConFIG (2025) | 无冲突梯度训练 |

#### 问题三：训练不稳定与收敛困难

**现象**: 损失震荡、不收敛或收敛到错误解
**影响**: 结果不可靠

**解决方案**:
| 方案 | 原理 |
|------|------|
| Adam + L-BFGS 组合 | Adam 粗调 + L-BFGS 精调 |
| 因果训练 (Causal Training) | 强制按时间顺序学习 |
| 域分解 (FBPINNs) | 多个小网络负责子域 |
| 课程学习 (Curriculum Learning) | 从简单到复杂逐步增加问题难度 |
| PirateNets | 自适应深度，浅层开始逐渐加深 |

#### 问题四：复杂 PDE / 刚性方程

**现象**: 强非线性、高 Reynolds 数、多物理场耦合时训练极其困难

**解决方案**:
| 方案 | 原理 |
|------|------|
| 分段网络 (Segregated Networks) | 不同物理量用不同网络 |
| 多阶段训练 (Multistage) | 分阶段求解，减少耦合难度 |
| 时间步进方案 (Runge-Kutta PINN) | 离散时间，逐步推进 |
| VS-PINN | 变量缩放处理刚性问题 |

#### 问题五：可扩展性

**现象**: 高维 (3D+时间) 需要大量配点，内存和计算成为瓶颈

**解决方案**:
| 方案 | 原理 |
|------|------|
| 小批量 (Mini-batch) 训练 | 每次迭代仅用部分配点 |
| 自适应采样 | 在残差大的区域加密采样 (RAR, AAS) |
| 可分离 PINN / SPIKANs | 维度分离降低计算量 |
| 分布式训练 | 多 GPU 并行 (PhysicsNeMo) |

### 6.2 诊断训练失败的检查清单

```
1. 检查损失曲线各分量是否平衡
2. 可视化 PDE 残差空间分布 -- 高残差区域是否欠采样
3. 检查解的物理合理性 (质量/能量守恒)
4. 尝试 Adam 5000 epochs + L-BFGS 精调
5. 调整损失权重（或使用自适应权重）
6. 增加配点数量或使用自适应采样
7. 检查激活函数 (推荐 tanh 或 sin，避免 ReLU)
8. 检查输入/输出归一化
9. 降低问题复杂度测试 (低 Re, 小域, 短时间)
10. 考虑域分解或时间步进策略
```

---

## 7. 最新进展 (2024-2026)

### 7.1 PIKANs -- KAN 替代 MLP

**核心思想**: 用 Kolmogorov-Arnold Network (KAN) 替代传统 MLP 作为 PINNs 的基础网络。KAN 在边上学习（B-spline/小波激活函数），而非在节点上用固定激活函数。

**优势**:
- 更少参数达到相同精度（99% 精度在多种 PDE 上）
- 更好的可解释性
- 多尺度/奇异性/非线性问题上优于 MLP
- 更快的收敛速度

**关键变体**:
| 变体 | 基函数 | 特点 |
|------|--------|------|
| PIKAN (B-spline) | B-样条 | 原始版本，最稳定 |
| WAV-KAN | 小波 | 多分辨率分析 |
| SPIKANs | 可分离 | 高维问题降低复杂度 |
| HWF-PIKAN | 混合小波-傅里叶 | 处理间断初值问题 |

### 7.2 自适应 PINN 方法

- **自适应采样**: RAR (残差自适应细化), AAS (对抗自适应采样, ICLR 2024)
- **自适应权重**: NTK-based, SA-PINNs (2-10x 更少 epoch)
- **自适应架构**: PirateNets (渐进式加深), 知识蒸馏自动结构发现

### 7.3 因果 PINNs

- 强制按时间因果顺序训练，避免"跳过"早期时间步
- TL-DPINN (IJCAI 2024): 离散时间 PINN + 迁移学习，稳定高效
- TCAS-PINN: 时间因果自适应采样

### 7.4 Competitive PINNs (CPINNs)

- 引入判别器网络，与 PINN 进行零和博弈
- 避免 PDE 离散化导致的大条件数问题
- Poisson 方程上精度比最佳 PINN 高 4 个数量级

### 7.5 其他值得关注的方向

| 方向 | 说明 |
|------|------|
| **Physics-Informed Diffusion Models** | 扩散模型 + 物理约束，生成式科学计算 |
| **GNN-based Physics** | 图神经网络处理非结构化网格 |
| **Foundation Models for Science** | 大规模预训练物理模型 |
| **Hybrid FEM-PINN** | 结合两种方法的优势 |
| **Multi-Fidelity Learning** | 融合不同精度等级的数据/模型 |

---

## 8. 喷涂物理方程的 PINN 求解方案

### 8.1 喷涂过程涉及的物理方程

喷涂仿真是典型的多物理场耦合问题，涉及以下方程组：

#### (1) 流场 -- Navier-Stokes 方程 (气流/喷雾)

```
连续性方程: div(rho * u) = 0
动量方程:   rho * (du/dt + (u . grad)u) = -grad(p) + mu * laplacian(u) + f
```

对于喷涂气流，需考虑：
- 可压缩性（高速气流 Ma > 0.3 时）
- 湍流模型（k-epsilon, SST 等 -- 可用 RANS 均值场方程）

#### (2) 传热 -- 能量方程

```
rho * c_p * (dT/dt + u . grad(T)) = k * laplacian(T) + Q_source
```

Q_source 包括：
- 喷涂粒子与基材间热交换
- 化学反应热 (热喷涂中涂层氧化等)
- 辐射换热

#### (3) 质量传输 -- 对流扩散方程

```
dC/dt + u . grad(C) = D * laplacian(C) + S
```

- C: 涂料浓度/液滴密度
- D: 扩散系数
- S: 源项 (喷嘴源)

#### (4) 涂层沉积 -- 涂层厚度模型

```
dh/dt = eta * n . J_droplet
```

- h: 涂层厚度
- eta: 沉积效率
- J_droplet: 液滴通量
- n: 表面法向量

### 8.2 PINN 求解方案设计

#### 方案 A: 分层求解 (推荐入门)

```
第一层 PINN: 求解气流场
  输入: (x, y, z)
  输出: (u, v, w, p)
  物理: 稳态 NS 方程 + 边界条件

第二层 PINN: 求解液滴/粒子传输
  输入: (x, y, z, t) + 来自第一层的流场
  输出: C(x,y,z,t) -- 涂料浓度
  物理: 对流扩散方程

第三层 PINN: 求解传热
  输入: (x, y, z, t) + 流场 + 浓度场
  输出: T(x,y,z,t)
  物理: 能量方程

后处理: 涂层厚度积分
  h(x,y) = integral_0^T eta * C * v_n dt
```

#### 方案 B: 统一多物理场 PINN (高级)

```python
# 单一网络同时输出所有物理场量
class SprayPINN(nn.Module):
    def __init__(self):
        self.shared = MLP([4, 128, 128, 128, 128], activation='tanh')
        self.head_flow = MLP([128, 64, 4])    # u, v, w, p
        self.head_temp = MLP([128, 64, 1])    # T
        self.head_conc = MLP([128, 64, 1])    # C

    def forward(self, x, y, z, t):
        features = self.shared(torch.cat([x,y,z,t], dim=-1))
        flow = self.head_flow(features)       # [u, v, w, p]
        T = self.head_temp(features)          # 温度
        C = self.head_conc(features)          # 浓度
        return flow, T, C

# 损失函数
L = (w1 * L_NS           # NS 方程残差
   + w2 * L_energy        # 能量方程残差
   + w3 * L_transport     # 传输方程残差
   + w4 * L_bc            # 边界条件
   + w5 * L_data)         # 观测数据 (传感器/CFD)
```

注意：多物理场统一训练中，各方程残差量级差异巨大，需要**自适应权重**。推荐使用分段网络 (Segregated Networks) 减轻优化难度。

#### 方案 C: Neural Operator 代理模型 (面向实时应用)

```
离线阶段:
  1. 用 CFD (如 Fluent/OpenFOAM) 生成大量仿真数据
     - 变化参数: 喷距、喷速、角度、流量
     - 输出: 涂层厚度分布 h(x, y; params)
  2. 训练 DeepONet / FNO
     - Branch: 编码工艺参数
     - Trunk: 编码空间坐标

在线阶段:
  给定新的工艺参数 --> 毫秒级预测涂层厚度分布
  --> 用于路径规划优化
```

### 8.3 喷涂 PINN 特殊考虑

| 挑战 | 应对策略 |
|------|----------|
| 多尺度 (喷嘴mm级 vs 涂层um级) | 多尺度 Fourier Features 或域分解 |
| 湍流 (高 Re) | RANS 均值场方程代替直接模拟 |
| 移动喷嘴/时变域 | 参考系变换或ALE方法嵌入损失 |
| 液滴碰撞/飞溅 | 简化模型 (经验公式) + 数据驱动 |
| 大量工况参数 | DeepONet/FNO 参数化代理模型 |
| 复杂工件几何 | 点云几何 + DeepXDE CSG 构造 |

### 8.4 与路径规划的集成

```
路径规划优化循环:
  1. 给定喷枪路径 P = {(x_i, y_i, z_i, theta_i, v_i)}
  2. PINN/Neural Operator 快速预测涂层厚度分布 h(x,y)
  3. 计算目标函数: J = || h(x,y) - h_target(x,y) ||^2 + lambda * PathCost(P)
  4. 梯度优化更新路径 P
  5. 重复 2-4 直到满足均匀性/效率要求

关键: 步骤 2 必须足够快 (< 100ms) -- 这是用 Neural Operator 而非 CFD 的核心理由
```

---

## 9. 代码示例框架

### 9.1 DeepXDE -- 热传导方程 (入门级)

```python
"""
用 DeepXDE 求解 2D 热传导方程:
  dT/dt = alpha * (d2T/dx2 + d2T/dy2)

模拟喷涂中基材的温度扩散
"""
import deepxde as dde
import numpy as np

# 物理参数
alpha = 1e-5  # 热扩散系数 (m^2/s)

# 定义 PDE
def heat_equation(x, T):
    """x = (x_coord, y_coord, t)"""
    dT_t = dde.grad.jacobian(T, x, i=0, j=2)      # dT/dt
    dT_xx = dde.grad.hessian(T, x, i=0, j=0)       # d2T/dx2
    dT_yy = dde.grad.hessian(T, x, i=1, j=1)       # d2T/dy2
    return dT_t - alpha * (dT_xx + dT_yy)

# 定义几何域和时间域
geom = dde.geometry.Rectangle([0, 0], [0.1, 0.1])    # 10cm x 10cm 基材
timedomain = dde.geometry.TimeDomain(0, 10)           # 0-10 秒
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 边界条件: 四周固定温度 300K
bc = dde.icbc.DirichletBC(
    geomtime, lambda x: 300, lambda _, on_boundary: on_boundary
)

# 初始条件: 均匀温度 300K
ic = dde.icbc.IC(
    geomtime, lambda x: 300, lambda _, on_initial: on_initial
)

# 喷涂热源 (高斯热源，模拟喷枪加热区域)
# 在 PDE 中加入源项
def heat_with_source(x, T):
    dT_t = dde.grad.jacobian(T, x, i=0, j=2)
    dT_xx = dde.grad.hessian(T, x, i=0, j=0)
    dT_yy = dde.grad.hessian(T, x, i=1, j=1)

    # 高斯热源: Q = Q0 * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    Q0 = 1e4   # 热源强度
    x0, y0 = 0.05, 0.05  # 热源中心
    sigma = 0.01
    Q = Q0 * np.exp(-((x_coord - x0)**2 + (y_coord - y0)**2) / (2*sigma**2))

    return dT_t - alpha * (dT_xx + dT_yy) - Q

# 构建数据
data = dde.data.TimePDE(
    geomtime,
    heat_with_source,
    [bc, ic],
    num_domain=5000,      # 域内配点
    num_boundary=200,      # 边界点
    num_initial=200,       # 初始条件点
    num_test=1000
)

# 构建网络
net = dde.nn.FNN(
    [3] + [64] * 4 + [1],   # 3 输入(x,y,t), 4 隐藏层每层 64, 1 输出(T)
    "tanh",
    "Glorot uniform"
)

# 编译和训练
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=10000)

# 切换到 L-BFGS 精调
model.compile("L-BFGS")
losshistory, train_state = model.train()

# 保存和可视化
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
```

### 9.2 PyTorch 原生实现 -- Navier-Stokes PINN (中级)

```python
"""
用 PyTorch 从零实现 2D 稳态 Navier-Stokes PINN
模拟喷涂气流场 (简化为方腔驱动流)
"""
import torch
import torch.nn as nn
import numpy as np

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN_NS(nn.Module):
    """2D 稳态 Navier-Stokes PINN"""

    def __init__(self, layers=[2, 128, 128, 128, 128, 3]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        # 初始化
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """输入: (x, y), 输出: (u, v, p)"""
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

    def compute_pde_residual(self, xy):
        """计算 NS 方程残差"""
        xy.requires_grad_(True)

        out = self.forward(xy)
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        # 一阶导数
        u_x = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        v_x = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(v),
                                   create_graph=True)[0]
        p_x = torch.autograd.grad(p, xy, grad_outputs=torch.ones_like(p),
                                   create_graph=True)[0]

        du_dx, du_dy = u_x[:, 0:1], u_x[:, 1:2]
        dv_dx, dv_dy = v_x[:, 0:1], v_x[:, 1:2]
        dp_dx, dp_dy = p_x[:, 0:1], p_x[:, 1:2]

        # 二阶导数
        du_dxx = torch.autograd.grad(du_dx, xy, grad_outputs=torch.ones_like(du_dx),
                                      create_graph=True)[0][:, 0:1]
        du_dyy = torch.autograd.grad(du_dy, xy, grad_outputs=torch.ones_like(du_dy),
                                      create_graph=True)[0][:, 1:2]
        dv_dxx = torch.autograd.grad(dv_dx, xy, grad_outputs=torch.ones_like(dv_dx),
                                      create_graph=True)[0][:, 0:1]
        dv_dyy = torch.autograd.grad(dv_dy, xy, grad_outputs=torch.ones_like(dv_dy),
                                      create_graph=True)[0][:, 1:2]

        nu = 0.01  # 运动粘度

        # NS 方程残差
        f_u = u * du_dx + v * du_dy + dp_dx - nu * (du_dxx + du_dyy)
        f_v = u * dv_dx + v * dv_dy + dp_dy - nu * (dv_dxx + dv_dyy)
        f_cont = du_dx + dv_dy  # 连续性方程

        return f_u, f_v, f_cont


def train_ns_pinn():
    """训练 NS PINN"""
    model = PINN_NS().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 采样点
    N_domain = 5000   # 域内配点
    N_bc = 500        # 每个边界

    for epoch in range(20000):
        optimizer.zero_grad()

        # 1. 域内配点 (随机采样)
        xy_domain = torch.rand(N_domain, 2, device=device, requires_grad=True)
        f_u, f_v, f_cont = model.compute_pde_residual(xy_domain)
        loss_pde = (torch.mean(f_u**2) + torch.mean(f_v**2)
                    + torch.mean(f_cont**2))

        # 2. 边界条件 (示例: 顶部 u=1, 其余三边 u=v=0)
        # 底部 y=0
        xy_bottom = torch.cat([torch.rand(N_bc, 1, device=device),
                                torch.zeros(N_bc, 1, device=device)], dim=1)
        out_bottom = model(xy_bottom)
        loss_bc = torch.mean(out_bottom[:, 0:1]**2 + out_bottom[:, 1:2]**2)

        # 顶部 y=1, u=1, v=0
        xy_top = torch.cat([torch.rand(N_bc, 1, device=device),
                            torch.ones(N_bc, 1, device=device)], dim=1)
        out_top = model(xy_top)
        loss_bc += torch.mean((out_top[:, 0:1] - 1.0)**2 + out_top[:, 1:2]**2)

        # 总损失
        loss = 10.0 * loss_pde + 100.0 * loss_bc

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6e}, "
                  f"PDE = {loss_pde.item():.6e}, BC = {loss_bc.item():.6e}")

    return model


if __name__ == "__main__":
    model = train_ns_pinn()
```

### 9.3 DeepONet -- 涂层厚度预测代理模型 (高级)

```python
"""
用 DeepXDE 的 DeepONet 学习:
  工艺参数 (喷距, 速度, 流量) --> 涂层厚度分布 h(x, y)

假设已有 CFD 仿真数据集
"""
import deepxde as dde
import numpy as np

# ========== 数据准备 ==========
# 假设有 N_samples 组仿真结果
# 每组: 工艺参数 (3维) -> 涂层厚度场 h(x,y) 在 M 个网格点上

N_samples = 200
M_sensors = 100    # 输入函数的传感器点数 (固定位置)
N_query = 50       # 查询点数

# 模拟数据 (实际应从 CFD 结果读取)
# 输入: 工艺参数编码为传感器值
X_branch = np.random.randn(N_samples, M_sensors).astype(np.float32)
# 输出位置: (x, y) 坐标
X_trunk = np.random.rand(N_query, 2).astype(np.float32)
# 输出: 涂层厚度值
y = np.random.rand(N_samples, N_query).astype(np.float32)

# 构建数据集
data = dde.data.Triple(
    X_train=(X_branch[:160], X_trunk),
    y_train=y[:160],
    X_test=(X_branch[160:], X_trunk),
    y_test=y[160:]
)

# ========== 网络架构 ==========
net = dde.nn.DeepONet(
    [M_sensors, 128, 128, 128],   # Branch net: 编码工艺参数
    [2, 128, 128, 128],           # Trunk net: 编码空间坐标 (x, y)
    "relu",
    "Glorot normal"
)

# ========== 训练 ==========
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(epochs=50000)

# ========== 推理 ==========
# 给定新的工艺参数，预测涂层厚度分布
new_params = np.random.randn(1, M_sensors).astype(np.float32)
# 在密集网格上查询
x_query = np.linspace(0, 1, 100)
y_query = np.linspace(0, 1, 100)
xx, yy = np.meshgrid(x_query, y_query)
query_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

# 预测 -- 毫秒级
h_predicted = model.predict((new_params, query_points))
# h_predicted 即为预测的涂层厚度分布
```

### 9.4 NVIDIA PhysicsNeMo -- 工业级多物理场 (生产级)

```python
"""
NVIDIA PhysicsNeMo 伪代码框架
用于工业级喷涂仿真 (需要 NVIDIA GPU)
"""
import physicsnemo
from physicsnemo.geometry import Box, Cylinder
from physicsnemo.models.fully_connected import FullyConnectedArch
from physicsnemo.domain import Domain
from physicsnemo.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.eq.pdes.advection_diffusion import AdvectionDiffusion

# 定义几何
spray_domain = Box(
    point_1=(-0.05, -0.05, 0.0),   # 5cm x 5cm x 20cm 喷涂区域
    point_2=(0.05, 0.05, 0.20)
)
nozzle = Cylinder(
    center=(0, 0, 0.20),
    radius=0.002,    # 2mm 喷嘴半径
    height=0.01
)

# 定义物理方程
ns_eq = NavierStokes(nu=1.5e-5, rho=1.225, dim=3)      # 空气流场
ad_eq = AdvectionDiffusion(D=1e-4, dim=3)                # 涂料传输

# 定义网络
flow_net = FullyConnectedArch(
    input_keys=["x", "y", "z"],
    output_keys=["u", "v", "w", "p"],
    layer_size=256,
    nr_layers=6
)
conc_net = FullyConnectedArch(
    input_keys=["x", "y", "z"],
    output_keys=["c"],
    layer_size=128,
    nr_layers=4
)

# 定义域和约束
domain = Domain()

# 内部点约束 (PDE 残差)
interior = PointwiseInteriorConstraint(
    nodes=[ns_eq, ad_eq],
    geometry=spray_domain,
    batch_size=4096,
    outvar={"momentum_x": 0, "momentum_y": 0, "momentum_z": 0,
            "continuity": 0, "advection_diffusion": 0}
)
domain.add_constraint(interior, "interior")

# 边界条件 (入口/出口/壁面)
# ... (省略详细边界条件设置)

# 求解器配置
slv = physicsnemo.Solver(cfg, domain)
slv.solve()
```

---

## 10. 推荐入门路线

### 10.1 学习路线图

```
第一阶段 (2-3 周): 理论基础
├── 学习 PDE 数值方法基础 (FDM/FEM 概念)
├── 学习 PyTorch / TensorFlow 自动微分
├── 阅读 Raissi 2019 原论文
├── 阅读 "An Expert's Guide to Training PINNs" (arXiv: 2308.08468)
└── 完成 DeepXDE 官方教程 (heat equation, diffusion equation)

第二阶段 (2-3 周): 动手实践
├── 用 DeepXDE 求解热传导方程 (1D -> 2D)
├── 用 DeepXDE 求解对流扩散方程
├── 尝试 Navier-Stokes (cavity flow)
├── 实现一个简单的反问题 (已知温度场反推热导率)
└── 学习损失权重调节和训练技巧

第三阶段 (3-4 周): 进阶技术
├── 学习 DeepONet / FNO 算子学习
├── 尝试 NVIDIA PhysicsNeMo (如有 GPU)
├── 实现多物理场耦合 (NS + 热方程)
├── 学习自适应采样和因果训练
└── 阅读 PIKANs 和最新方法

第四阶段 (持续): 喷涂应用
├── 定义喷涂简化物理模型
├── 从 2D 简化模型开始
├── 逐步增加物理复杂度
├── 用 CFD 数据训练 Neural Operator 代理模型
└── 集成到路径规划优化中
```

### 10.2 推荐学习资源

#### 必读论文 (按优先级)

1. Raissi et al., 2019 -- PINNs 原论文
2. Lu et al., 2021 -- DeepXDE 论文 (工具使用)
3. "An Expert's Guide to Training PINNs" (arXiv: 2308.08468) -- 实操技巧
4. "From PINNs to PIKANs" (arXiv: 2410.13228) -- 最新综述
5. Li et al., 2021 -- FNO 论文 (算子学习)
6. Lu et al., 2021 -- DeepONet 论文 (Nature MI)

#### 代码资源

| 资源 | 链接 | 说明 |
|------|------|------|
| DeepXDE 官方示例 | github.com/lululxvi/deepxde/examples | 最全面的 PINN 示例集 |
| DeepXDE 文档教程 | deepxde.readthedocs.io | 逐步引导 |
| Raissi PINNs 原始代码 | github.com/maziarraissi/PINNs | TensorFlow 1.x，概念参考 |
| NeuralOperator 教程 | github.com/neuraloperator/neuraloperator | FNO/DeepONet 示例 |
| PhysicsNeMo 示例 | github.com/NVIDIA/physicsnemo | 工业案例 |
| PINN Paper List | github.com/Event-AHU/PINN_Paper_List | 论文索引 |

#### 视频/课程

- George Karniadakis (Brown University) 的系列讲座
- Steve Brunton (UW) 的 YouTube 系列 -- 科学机器学习
- DeepXDE 官方 Webinar

### 10.3 硬件建议

| 阶段 | 硬件 | 说明 |
|------|------|------|
| 入门学习 | CPU (任意) | 1D/2D 小问题足够 |
| 进阶实践 | 单 GPU (RTX 3060+) | 2D/3D 问题需要 GPU |
| 生产应用 | 多 GPU (A100/H100) | PhysicsNeMo 大规模训练 |
| 替代方案 | Google Colab (免费 GPU) | 入门和小规模实验 |

### 10.4 快速上手检查清单

- [ ] 安装 Python 3.8+ 和 PyTorch
- [ ] `pip install deepxde` 并设置后端为 PyTorch
- [ ] 运行 DeepXDE 的 heat equation 示例
- [ ] 修改参数观察结果变化
- [ ] 尝试 2D Poisson 方程
- [ ] 尝试加入观测数据 (data-driven + physics-informed)
- [ ] 尝试一个反问题
- [ ] 阅读 "Expert's Guide" 论文
- [ ] 根据喷涂需求选择求解路线

---

## 附录: 术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| 配点 | Collocation Points | PDE 残差的采样点 |
| 自动微分 | Automatic Differentiation (AD) | 精确计算网络对输入的导数 |
| 频谱偏差 | Spectral Bias | 网络优先学低频，忽略高频 |
| 算子学习 | Operator Learning | 学习函数到函数的映射 |
| 代理模型 | Surrogate Model | 替代昂贵模拟的快速近似模型 |
| 域分解 | Domain Decomposition | 将大域分成小子域分别求解 |
| 因果训练 | Causal Training | 强制按时间顺序学习 |
| 硬约束 | Hard Constraints | 将边界条件精确嵌入网络结构 |
| 软约束 | Soft Constraints | 通过损失函数惩罚项近似满足 |

---

> **文档说明**: 本文档基于截至 2026 年 2 月的公开研究文献和工具信息编写。PINNs 领域发展极为迅速，建议定期关注 arXiv (cs.CE, cs.LG, physics.comp-ph) 的最新论文以及各框架的 GitHub 更新。
>
> **关键参考源**:
> - Raissi et al. PINNs 项目主页: https://maziarraissi.github.io/PINNs/
> - DeepXDE 文档: https://deepxde.readthedocs.io/
> - NVIDIA PhysicsNeMo: https://developer.nvidia.com/physicsnemo
> - NeuralOperator: https://github.com/neuraloperator/neuraloperator
> - PINN Paper List: https://github.com/Event-AHU/PINN_Paper_List
