"""P514 标定用随机策略（仅用于对照实验，不参与正式提交）。

作用：在留出场景上给出"随机动作"的分数下界，用来判断规则/学习策略的增益是否真实。
动作在每个决策步独立均匀采样自 [-1,1]^2，等价于评测里 P901 随机基线的口径。
"""
from __future__ import annotations

import numpy as np


class RandomPolicy:
    """每步均匀随机动作；使用 policy_seed 保证可复现。"""

    def __init__(self, policy_seed=0, **kwargs):
        self._seed = int(policy_seed)
        self._rng = np.random.default_rng(self._seed)

    def reset(self, context=None):
        seed = self._seed
        if context is not None and getattr(context, "policy_seed", None) is not None:
            seed = int(context.policy_seed)
            self._seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, observation):
        return self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    def close(self):
        pass


def build_policy_for_agent(context, artifact_dir=None, params=None):
    return RandomPolicy(policy_seed=getattr(context, "policy_seed", 0))
