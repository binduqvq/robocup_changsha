# 第三方依赖与许可声明（THIRD_PARTY.md）

## 最终提交的推理依赖

最终提交是**纯规则策略**，不含学习模型，`artifacts/` 为空。

- 推理只使用官方评测环境已经提供的 **NumPy**（评测依赖锁 `requirements-official.lock` 内），
  未引入任何额外第三方源码、模型权重或数据文件。
- `requirements-infer.lock` 中不登记额外依赖。
- 无外部下载、无动态代码载入、无不安全反序列化调用。

结论：**最终提交不含第三方代码或模型，无需署名与许可证传递。**

## 开发期（不进入提交运行路径）使用过的第三方库

以下库仅用于本地对照实验与诊断，**不在推理路径上**，其代码也未复制进本目录：

| 库 | 用途 | 许可证 | 来源 |
|---|---|---|---|
| Stable-Baselines3 | 参考公开包模板 PPO 基线（E000） | MIT | https://github.com/DLR-RM/stable-baselines3 |
| SuperSuit | 模板训练栈的向量化封装 | Apache-2.0 | https://github.com/Farama-Foundation/SuperSuit |
| PyTorch | 模板训练栈（仅训练环境 `.venv-train`） | BSD-3-Clause | https://github.com/pytorch/pytorch |
| MPE2 | 官方任务环境依赖（官方提供） | MIT | https://github.com/Farama-Foundation/MPE2 |
| PettingZoo | 官方并行环境接口（官方提供） | MIT | https://github.com/Farama-Foundation/PettingZoo |

说明：这些库由公开包模板与本仓库依赖锁引入；本参赛者未复制其源码，
也未修改其实现。最终提交不依赖它们。

## 算法调研资料（仅参考论文思想，未复制代码或模型）

| 资料 | 本项目中的用途 | 来源 |
|---|---|---|
| Dynamic multi-agent assignment via discrete optimal transport | 动力学感知的目标分配思路；精确局部匹配消融 E036 | https://arxiv.org/abs/1910.10748 |
| Toward multi-target self-organizing pursuit in a partially observable Markov game | 将无通信追捕拆为搜索、分配、单目标追捕 | https://arxiv.org/abs/2206.12330 |
| The Hybrid Reciprocal Velocity Obstacle | 相对位置/速度的预测式避碰思想 | https://doi.org/10.1109/TRO.2011.2120810 |
| The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games | MAPPO/CTDE 对照路线 | https://arxiv.org/abs/2103.01955 |
| Multi-Agent Reinforcement Learning is a Sequence Modeling Problem | MAT 作为长回合协同的后续候选 | https://arxiv.org/abs/2205.14953 |
| PettingZoo MPE Simple Spread | 相似的多智能体覆盖、协作与碰撞任务；用于确认问题结构 | https://pettingzoo.farama.org/1.22.3/environments/mpe/simple_spread/ |
| Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments | MADDPG 与 MPE 学习基线背景；用于定位规则/学习对照 | https://arxiv.org/abs/1706.02275 |
| Scalable Distributed Planning for Multi-Robot Multi-Target Tracking | 滚动时域、分布式目标分配与跟踪的调研依据 | https://arxiv.org/abs/2107.08550 |
| Distributed Assignment with Limited Communication | 有限通信条件下分布式分配的边界与后续方向 | https://arxiv.org/abs/1812.11172 |
| Multi-Robot Target Tracking with Limited Field of View | 有限视野下短时目标记忆/轨迹外推的调研依据；E041 消融 | https://www.roboticsproceedings.org/rss19/p109.pdf |
| Multi-agent coverage control with transient behavior | 将短回合瞬态性能而非仅稳态覆盖作为优化目标 | https://proceedings.mlr.press/v242/zhang24e.html |

这些资料只影响算法设计与实验选择；本目录没有复制其源码、伪代码、模型权重或数据集，
因此不会给最终推理路径增加第三方许可或运行依赖。

## 本目录内自研代码

`entry.py`、`policies/rule.py`、`policies/random_ref.py`、`policies/probe.py`、
`tools/*.py` 均由本参赛者编写，按 `LICENSE`（MIT）授权。

其中 `tools/` 与 `policies/probe.py`、`policies/random_ref.py` 是**开发期对照实验工具**，
不参与正式推理；`probe.py` 会读取训练专用全局状态，仅用于测量任务的物理上界，
**不是可部署策略**，`entry.py` 不会导入它。

工具清单与用途：

| 工具 | 用途 |
|---|---|
| `tools/rollout.py` | 按公开 `scenario_seed` 复现 4 个公开场景，与评测器同口径 |
| `tools/sweep.py` | 留出场景评测（防公开套件过拟合） |
| `tools/generalize.py` | **参数泛化台**：N×M×T×layout 与随机基线对照 |
| `tools/sanity_check.py` | 指标量级核查（下结论前先验尺子） |
| `tools/upper_bound.py` | 规则策略 vs 全信息 oracle 对照 |
| `tools/bootstrap_ci.py` | 公开成绩的 bootstrap 置信区间 |
| `tools/summarize_eval.py` | 评测输出目录逐回合汇总 |
| `tools/export_learned.py` | 把 SB3 PPO 权重导出为纯 NumPy `policy.npz` 并做一致性检查 |
| `tools/compare_three.py` | 规则 / 学习 / 随机 三方案同口径对照 |
| `tools/final_metrics.py` | 随机化轴逐格综合指标（策略 vs 随机，32 种子/格） |

> 说明：曾有一个 `tools/baseline_template.py`（用于把官方模板基线放到同一批场景上量化），
> 因使用 `importlib.util.spec_from_file_location` 命中硬拒绝项
> `UNREGISTERED_DYNAMIC_LOAD`，已**删除**；其测量结果已记录在 `LOG.md` 第 26 节。
> 模板本身仍在 `participant/_template/` 与 git 历史中，可随时复现该对照。


其中 `tools/rollout.py` 早期版本曾使用 `importlib.util.spec_from_file_location`
动态载入策略模块，触发官方静态审计硬拒绝项 `UNREGISTERED_DYNAMIC_LOAD`；
现已改为包内静态导入，预检 `rejections=0`。提交目录内**不含**任何动态载入调用。

## 学习路线相关文件（实验对照，不参与最终推理）

最终提交是纯规则策略，`artifacts/` 为空，推理路径不导入以下文件；
它们用于复现"为什么不用学习"这一对照实验：

| 文件 | 用途 | 是否在推理路径 |
|---|---|---|
| `train.py` | 场景混合的 SB3 PPO 训练（`.venv-train` 内运行，含 torch） | 否 |
| `policies/learned.py` | 学习策略的纯 NumPy 推理实现（**无 torch、无 pickle**） | 否（`entry.py` 不导入） |
| `tools/export_learned.py` | 训练权重 → `policy.npz` 导出与一致性校验 | 否 |

**注意（实测的静态审计结果）**：`train.py` 与 `tools/export_learned.py` 会在训练侧使用
PyTorch，因此源码审计命中 `torch` / `torch.no_grad` 符号。按 `coverage_bench/audit.py`，
`torch` 属于 `SCAN_SYMBOLS`（**仅供人工审核，不参与拒绝判断**），
只有 `torch.load` / `pickle.load` 等反序列化调用才是硬拒绝项（`_PICKLE_DESERIALIZE_CALLS`）——
本项目**未使用任何反序列化调用**。实测预检结果：
`rejections = []`，`scan_hits` 仅 2 条（均来自 `tools/export_learned.py` 的 torch 调用）。

学习策略的模型产物不放在 `artifacts/`，因为预检会拒绝"清单未登记的产物文件"
（实测报错 `提交校验失败: 发现未在清单中登记的产物文件: artifacts/policy.npz`）；
它保存在 `outputs/` 下的训练目录中，属于实验材料而非评分产物。



