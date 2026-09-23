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

## 本目录内自研代码

`entry.py`、`policies/rule.py`、`policies/random_ref.py`、`policies/probe.py`、
`tools/*.py` 均由本参赛者编写，按 `LICENSE`（MIT）授权。

其中 `tools/` 与 `policies/probe.py`、`policies/random_ref.py` 是**开发期对照实验工具**，
不参与正式推理；`probe.py` 会读取训练专用全局状态，仅用于测量任务的物理上界，
**不是可部署策略**，`entry.py` 不会导入它。
