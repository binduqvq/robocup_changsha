"""P514 规则策略的评测入口：解析式追捕 + 定向搜索 + 邻居避碰。

生命周期要点（官方协议）：
- `build_policy(BuildContext)` 在**回合之前**调用，只能拿到协议规格、artifacts 目录、
  设备、资源限制和构造随机源；**拿不到任务参数**（N/M/T、物理参数）。
- 任务参数在 `reset(EpisodeContext)` 时才下发，且 `reset` 每回合都会调用。
因此策略实例必须在 reset 里根据 EpisodeContext 构建/重建内部状态，构造期不做任务相关推导。
忘记这一点的后果：在 build_policy 里读 context.task 会直接抛 AttributeError，
被评测器记为 initialize 阶段的 runtime_error，整份提交拿不到任何分数。

纯规则策略按规程第 16 节无需训练入口，也不依赖任何模型产物（artifacts 为空）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from policies.rule import AnalyticTracker, build_tracker  # noqa: E402

# 提交默认参数：
# - coordinated 用“自己 + 可见队友”做无通信就近分配；
# - intercept 用相邻观测估计目标速度，在含当前机器人动量的有限时域可达集上
#   求最早截获点。两套不相交的 128 场景上分别提升 +0.68 / +1.20 分，且
#   12 个规模/布局格仍全部优于随机（见 LOG.md 第七轮）。
SUBMISSION_OVERRIDES = {
    "goal_select": "coordinated",
    "intercept": 1,
    "intercept_weight": 1.0,
    "intercept_max_lead": 10,
    "vel_ema": 1.0,
}


class RulePolicy:
    """策略协议适配器：reset 时按本回合公开参数构建追踪器。"""

    def __init__(self, context):
        self._tracker: AnalyticTracker | None = None

    def reset(self, context):
        self._tracker = build_tracker(context, SUBMISSION_OVERRIDES)

    def act(self, observation):
        if self._tracker is None:  # 兜底：未 reset 不应发生，但保证始终返回合法动作
            return np.zeros(2, dtype=np.float32)
        return np.asarray(self._tracker.act(observation), dtype=np.float32)

    def close(self):
        self._tracker = None


def build_policy(context):
    return RulePolicy(context)
