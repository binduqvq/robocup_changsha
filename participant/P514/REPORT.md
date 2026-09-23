# P514 方案报告（REPORT.md）

## 1. 方案概述

最终提交是**纯规则策略**（`method_type: rule`）：每台机器人独立、只用本机器人局部观测与回合内历史，
逐决策步输出 `float32[2]` 归一化驱动力。核心由三部分组成：

1. **解析式追捕（bang-bang）**：对可见目标，直接算出朝目标推进的最优驱动方向，不使用任何学习模型。
2. **视野外定向搜索**：视野内没有可见目标时，用 `agent_index` 决定一个确定性的搜索方向。
   因为编号在回合内稳定且对每台机器人可见，无需通信即可让各机器人朝不同方向分散。
3. **可见邻居排斥避碰**：对观测到的邻近队友施加排斥项，降低机器人重叠带来的碰撞惩罚。

全部逻辑位于 `policies/rule.py`（策略实现）与 `entry.py`（协议适配器）。
只依赖官方评测环境已提供的 `numpy`，无模型产物（`artifacts/` 为空），无训练入口。

## 2. 为什么规则策略能在本题达到上界（关键推导）

官方物理更新为 `v' = (1-damping)·v + (drive_force/mass)·dt·u`，代码中位置按 `pos += vel*dt` 推进。
本任务冻结参数 damping=0.25、dt=0.1、drive_force/mass=1.0，记 `k=0.75`、`c=0.1`。

从**静止**出发、动作恒为满舵时，第 i 步的速度增量为 `c·k^{i-1}` 的累加，于是 n 步累计位移：

```
S(n) = dt · Σ_{i=1..n} c·(1-k^i)/(1-k)
S(10) ≈ 0.208
```

即 **10 步内机器人整场最多只能移动约 0.208（场地半宽为 1.0）**。
机器人速度上限 1.0 对应 40 步的位移量，在该回合长度内**不是紧约束**；真正的限制是驱动力与阻尼。

覆盖判定为"中心距 ≤ 目标覆盖半径 0.15"。因此从初始位置出发能够覆盖到的目标，
必须基本落在 `0.208 + 0.15 ≈ 0.36` 以内；而感知半径是 0.6。
**"看得见"与"够得着"不是一回事，覆盖率主要由初始布局的几何关系决定。**

动作空间是盒约束 `u ∈ [-1,1]^2`，在上述线性动力学下可达集是以当前位置为中心的同心球，
故单步最优控制是 **bang-bang**：要使下一步位移最朝向目标，只需令
`u ∝ (目标相对位置 − k·v·dt)`。这就是策略第 1 部分不需要学习的根本原因。

## 3. 关键对照实验（全部为本地实测）

公开套件为 `configs/public-suite-v1.yaml`（basic/uniform 2 例 + cooperation/crossing 2 例，
`policy_repeats=2`，共 8 回合）。**注意**：评测器对同一 case 的两次 repeat 使用**同一个
`scenario_seed`**，所以 8 个回合只有 4 个独立场景，公开分数方差很大。

为此另建"留出场景"评测：使用公开套件的同构任务配置 + 自选 `scenario_seed`
（`tools/sweep.py`，每组 32 个场景 × 2 个**不同**种子 = 每组 64 个独立场景，共 128 个）。

| 编号 | 方案 | 公开套件 score | 留出场景 score |
|---|---|---|---|
| E000 | 模板 SB3 PPO（公开包示例权重） | 66.67 | — |
| E003 | 均匀随机动作（P901 口径） | 133.33 | 58.02 |
| **E006（提交）** | **解析规则策略（仅局部观测）** | **183.33** | **144.64** |
| E004 | oracle：注入训练专用全局状态 + 同款解析控制 | — | **144.32** |

**最重要的结论（E004）**：把训练专用全局状态（所有机器人与目标的精确位置、速度）直接注入策略后，
留出场景成绩 144.32，与只用局部观测的规则策略 144.64 **在噪声范围内相同**。
这说明在 10 步 × 该动力学的约束下，**信息不是瓶颈，物理可达性才是瓶颈**；
规则策略在"够得着的目标"这一层面已经达到全信息上界，因此进一步引入通信、
全局分配或更复杂的信息利用，预期收益极低。

## 4. 失败与负结果（如实记录）

- **E002（60 组网格）**：`avoid_gain ∈ {0,0.5,0.9,1.5,2.5}` × `search_scale ∈ {0.5,1,2}`
  × `goal_select ∈ {nearest,index}` × `reach_aware ∈ {0,1}`。
  60 组只落在两个分数上（129.333 / 124.333）。`avoid_gain` 从 0 到 2.5 **完全不改变结果**，
  平均碰撞率仅 0.0033 → 碰撞项在该回合长度下几乎不生效，避碰不是主要矛盾。
- **E005（24 组网格）**：`search_mode × fan_span × keep_track × goal_select`，仍无显著差异。
- **度量陷阱**：不同留出种子集之间分数可差约 ±7 分。最初在 40 个留出场景上得到 129.33，
  后在 128 个独立场景上得到 144.64，差异来自种子集而非策略改动。
  因此本项目所有"提升"结论都以固定的大种子集复测为准；公开套件的 183.33
  明显高于留出均值，属公开场景偏有利，**不应视为期望成绩**。
- **API 踩坑**：`build_policy(BuildContext)` 拿不到任务参数（N/M/T、物理量在
  `reset(EpisodeContext)` 才下发）。最初在 `build_policy` 内读 `context.task`，
  评测器把 8 个回合全部记为 initialize 阶段 `runtime_error`，整份提交 0 分。
- **静态审计踩坑**：自建开发工具 `tools/rollout.py` 曾使用
  `importlib.util.spec_from_file_location` 动态载入策略模块，触发官方静态审计硬拒绝项
  `UNREGISTERED_DYNAMIC_LOAD`。已改为包内静态导入。

## 5. 复现方法

评测环境（仓库根目录）：

```sh
.venv/Scripts/python.exe scripts/check_submission.py --submission participant/P514 --output outputs/P514/E006-rule-official/check
.venv/Scripts/python.exe scripts/evaluate_one.py --submission participant/P514 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P514/E006-rule-official/eval
```

留出场景对照与 oracle 标定（开发期工具，不参与推理）：

```sh
.venv/Scripts/python.exe participant/P514/tools/rollout.py --policy rule          # 4 个公开场景
.venv/Scripts/python.exe participant/P514/tools/sweep.py --policy rule --n 32 --repeats 2
.venv/Scripts/python.exe participant/P514/tools/sweep.py --policy random_ref --n 32 --repeats 2
.venv/Scripts/python.exe participant/P514/tools/sweep.py --policy probe --n 32 --repeats 2   # oracle 上界
```

实测环境：Python 3.12.14，numpy 2.5.3、pettingzoo 1.27.0、mpe2 1.1.1、scipy 1.18.1。

## 6. 结果与局限

**结果（本地公开测试，`local_preview`，非组织方核验）**

- 最终提交 E006：`status=ok`、`performance_score=183.33`、8/8 回合成功、0 错误。
- 分组：basic `mean_j=0.0833`（`ci95_j=[0.0333,0.1333]`）、
  cooperation `mean_j=0.2833`（`ci95_j=[0.0,0.5667]`）。
- 单步推理耗时均值约 0.02~0.13 ms，远低于 `act_ms=1000` 限制。

**局限**

1. 本机为 Windows，平台不提供 `setitimer`/`SIGALRM`，评测退化为 `local_preview`，
   阶段超时未生效；正式成绩以组织方 Linux 容器为准。
2. 公开套件只有 4 个独立场景，`performance_score` 方差大；留出场景估计（144.64）
   更接近期望表现。二者都不是核验成绩。
3. 由于可达性是硬瓶颈，本方案基本已触及该任务参数下的表现上限；
   在容量更大（N、M 增大）或回合更长（T 增大）的设定下，结论是否仍然成立**未经实验**，
   不应外推。
4. 规模泛化（4v5、5v7 等）未做实验，`REPORT.md` 不声称其表现。

## 7. 与"学习路线"的关系

公开包模板 PPO（E000，66.67）低于官方学习基线 P903（116.67），
原因是模板只用了 `basic-0`（uniform 布局）一个用例训练，**完全不含 crossing 布局**，
而协作组正是 crossing。本次提交选择规则路线，不是因为学习不可行，
而是因为第 2、3 节的证据表明该任务在当前参数下的瓶颈是物理可达性而非信息/协同，
规则策略已达全信息上界。若后续把回合长度或容量调大（使可达性不再是硬约束），
学习路线的价值需要重新评估。
