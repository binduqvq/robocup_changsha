"""P514 学习型策略的 NumPy 推理实现（无 torch、无 pickle，符合静态审计）。

评测环境不安装 PyTorch / Stable-Baselines3，因此训练得到 MLP 权重后必须以
纯 NumPy 前向的方式加载。本模块与 `participant/_template/entry.py` 的导出惯例一致：
`artifacts/policy.npz` 内含 W1/b1/W2/b2/W3/b3 六个权重数组与四个 VecNormalize 统计量
（obs_mean / obs_var / obs_eps / obs_clip），前向顺序为
**先归一化再 clip**（与 SB3 `VecNormalize.normalize_obs` 一致），再两层 tanh 隐层。

确定性动作取高斯策略的均值（`deterministic=True` 的等价实现），因此本推理与
SB3 `model.predict(obs, deterministic=True)` 应当逐位一致——`tools/export_learned.py`
会用同一批观测断言这一点（max|Δa| ≤ 1e-5）。

注意：本文件只做推理，不参与训练；训练入口见 `train.py`。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from coverage_bench.protocol import get_protocol_spec
from coverage_bench.spaces import flatten_observation

_PROTOCOL_SPEC = get_protocol_spec()


class NumpyMLPPolicy:
    """SB3 PPO 导出权重的 NumPy 前向推理。"""

    def __init__(self, artifact_dir: Path):
        data = np.load(Path(artifact_dir) / "policy.npz")
        self._w1, self._b1 = data["W1"], data["b1"]
        self._w2, self._b2 = data["W2"], data["b2"]
        self._w3, self._b3 = data["W3"], data["b3"]
        self._mean = data["obs_mean"]
        self._var = data["obs_var"]
        self._eps = float(data["obs_eps"])
        self._clip = float(data["obs_clip"])

    def reset(self, context=None):
        # MLP 无回合内记忆；本方法存在只是为满足策略协议
        pass

    def act(self, observation) -> np.ndarray:
        x = flatten_observation(observation, _PROTOCOL_SPEC).astype(np.float64)
        # 与 VecNormalize.normalize_obs 同序：先归一化再 clip
        x = np.clip((x - self._mean) / np.sqrt(self._var + self._eps), -self._clip, self._clip)
        h = np.tanh(x @ self._w1 + self._b1)
        h = np.tanh(h @ self._w2 + self._b2)
        action = h @ self._w3 + self._b3
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def close(self):
        pass


def build_policy(context):
    return NumpyMLPPolicy(context.artifact_dir)


def build_policy_for_agent(context, artifact_dir=None, params=None):
    """供本地评测工具统一调用的适配器（与规则策略同签名）。"""
    if artifact_dir is None:
        raise ValueError("学习型策略需要 artifact_dir")
    return NumpyMLPPolicy(artifact_dir)
