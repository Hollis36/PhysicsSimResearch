# 更新日志 (Changelog)

本文档记录 PhysicsSimResearch 项目的主要更新和改进。

---

## [1.1.0] - 2026-02-17

### 🎉 重大改进

#### 新增文档
- ✅ **GETTING_STARTED.md** - 快速入门指南
  - 三种学习路径：应用工程师、算法研究员、物理仿真专家
  - 按应用场景查找技术方案
  - 分阶段实施建议
  
- ✅ **RESOURCES.md** - 资源索引
  - 100+ 开源代码库分类整理
  - 主要数据集汇总（The Well, PDEBench, ERA5 等）
  - 预训练模型索引
  - 160+ 论文分类索引（按会议、主题分类）
  - 学习资源推荐
  
- ✅ **ROADMAP.md** - 技术路线图
  - 6 个方向关联矩阵
  - 4 个典型应用场景技术栈
  - 分阶段实施建议
  - 跨方向创新机会点
  
- ✅ **FAQ.md** - 常见问题解答
  - 24 个常见问题及详细解答
  - 涵盖项目、入门、技术、应用、贡献等方面
  
- ✅ **CONTRIBUTING.md** - 贡献指南
  - 详细的贡献流程
  - 内容规范和质量标准
  - 分支和提交规范

#### 新增工具
- ✅ **requirements.txt** - Python 依赖管理
  - 基础科学计算库
  - 可选的专项工具（可微物理、GNN、PINNs 等）
  - 实验追踪工具
  
- ✅ **setup_env.sh** - 环境自动配置脚本
  - 一键创建 Conda 环境
  - 智能检测 CUDA 版本
  - 分模块安装依赖
  - 交互式选择研究方向

#### README 增强
- ✅ 添加项目徽章（研究方向、论文数量、更新日期）
- ✅ 增加技术对比矩阵
- ✅ 添加成熟度评级
- ✅ 增加快速导航（按应用场景、技术成熟度）
- ✅ 改进视觉结构和可读性
- ✅ 添加所有新文档的链接

### 📊 内容改进
- 改进了 6 个研究方向的表格展示
- 添加了典型应用场景和成熟度指标
- 增加了与现有项目（MFARainbowNet、DroneVehicle）的关联说明
- 添加了推荐学习路线

### 🔧 技术优化
- 统一了文档格式和风格
- 改进了内部链接结构
- 优化了代码示例的可读性

---

## [1.0.0] - 2026-02-04

### 🎉 初始发布

#### 核心内容
- ✅ **6 个研究方向完整文档**
  - 01_differentiable_physics - 可微物理仿真
  - 02_physics_foundation_models - 物理基础模型
  - 03_gnn_particle_simulation - GNN 粒子仿真
  - 04_world_models - 世界模型
  - 05_pinns - 物理信息神经网络
  - 06_multimodal_fusion - 多模态融合

#### 每个方向包含
- 领域概述与发展脉络
- 核心方法详细对比
- 关键论文清单（总计 160+ 篇）
- 与现有项目的对接方案
- 开源代码资源汇总
- 推荐学习路线

#### 论文统计
- NeurIPS: 30+ 篇
- ICML: 20+ 篇
- ICLR: 15+ 篇
- Nature/Science: 10+ 篇
- 其他顶会/期刊: 85+ 篇

#### 代码资源
- 物理基础模型: Walrus, GPhyT, PhysiX, Poseidon 等
- 可微物理: NVIDIA Newton, Brax, MuJoCo MJX 等
- GNN 仿真: GNS, NeuralMPM, LagrangeBench 等
- 世界模型: NVIDIA Cosmos, DreamerV3, GraphCast 等
- PINNs: DeepXDE, NVIDIA Modulus 等
- 多模态: BEVFusion, TransFusion 等

#### 数据集
- The Well (15TB)
- PDEBench (TB 级)
- ERA5 (PB 级)
- nuScenes, KITTI, Waymo Open 等

---

## 计划中的功能

### v1.2.0 (预计 2026-03)
- [ ] 添加 Jupyter Notebook 示例
- [ ] 创建视频教程链接
- [ ] 添加 Docker 支持
- [ ] 创建交互式技术选择工具
- [ ] 添加更多案例研究

### v1.3.0 (预计 2026-04)
- [ ] 添加性能基准测试
- [ ] 创建模型动物园（Model Zoo）
- [ ] 添加自动化测试
- [ ] 创建社区论坛链接
- [ ] 多语言支持（英文版）

### v2.0.0 (长期计划)
- [ ] 开发 Web 界面
- [ ] 集成代码运行环境
- [ ] 添加实时论文追踪
- [ ] 创建知识图谱
- [ ] AI 助手集成

---

## 贡献统计

### v1.1.0 贡献
- 文档更新: 5 个新文档
- 代码工具: 2 个新工具
- 内容增强: README 全面改进
- 结构优化: 交叉引用系统

### v1.0.0 贡献
- 初始内容: 6 个方向完整文档
- 论文收录: 160+ 篇
- 代码收录: 100+ 个项目
- 数据集: 10+ 个

---

## 如何获取更新

### GitHub
- **Watch** 本仓库以接收更新通知
- **Star** 以标记收藏
- **Fork** 以创建自己的副本

### 更新频率
- **重大更新**: 每月 1 次
- **小修小补**: 每周多次
- **紧急修复**: 随时

### 订阅更新
1. 点击 GitHub 页面的 "Watch" 按钮
2. 选择 "Custom" → "Releases"
3. 在 Issues/Discussions 中关注公告

---

## 版本说明

### 版本号规则
遵循 [Semantic Versioning](https://semver.org/) 规范：

```
主版本号.次版本号.修订号

主版本号: 重大架构变更
次版本号: 新增功能
修订号: 问题修复
```

### 发布周期
- **主版本 (X.0.0)**: 每年 1-2 次
- **次版本 (1.X.0)**: 每月 1 次
- **修订版 (1.1.X)**: 根据需要

---

## 反馈与建议

欢迎通过以下方式提供反馈：

- **GitHub Issues**: [问题报告](https://github.com/Hollis36/PhysicsSimResearch/issues)
- **GitHub Discussions**: [功能建议](https://github.com/Hollis36/PhysicsSimResearch/discussions)
- **Pull Requests**: [直接贡献](CONTRIBUTING.md)

---

**感谢所有贡献者的支持！** 🎉

---

*最后更新: 2026-02-17*
