"""P514 规则策略 v1：解析式追捕 + 视野外定向搜索 + 邻居避碰。

全部依据来自公开协议与 EpisodeContext 的公开 task 参数，不读取任何隐藏信息。

动力学推导（关键，决定了"追捕"不需要学习）：
    真实更新   v_{t+1} = (1-damping) * v_t + (drive_force / mass) * dt * u_t
    单步位移   pos_{t+1} = pos_t + v_{t+1} * dt
    本任务冻结参数：damping=0.25、dt=0.1、drive_force/mass=1.0 ⇒ k=0.75、c=0.1。
    从静止出发、动作恒为满舵时，n 步累计位移
        S(n) = dt * Σ_{i=1..n} c * (1 - k^i) / (1 - k)  =  0.01 * Σ_{i=1..n} (1-0.75^i)/0.25
    S(10) ≈ 0.208：**10 步整场最多只能移动约 0.21（场地半宽 1.0）**。
    速度上限 1.0 在该回合长度内不是紧约束，驱动力/阻尼才是。
    盒约束 u ∈ [-1,1]^2 下可达集是同心球，故单步最优控制是 bang-bang：
    最大化"下一步位移在目标方向上的投影"⇒ u ∝ (目标相对位置 − k·v·dt)。

    目标相对位置定义为 目标 − 自身，因此朝目标推进 = 让自身位移沿该向量。

可调参数集中在 RuleParams，便于对照实验与复现。
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class RuleParams:
    """策略超参数（可由 EpisodeContext 的公开 task 参数填充，也可由实验覆盖）。"""

    agent_index: int = 0
    num_agents: int = 3
    num_targets: int = 3
    horizon: int = 10
    damping: float = 0.25
    dt: float = 0.1
    drive_step: float = 0.1          # c = drive_force/mass * dt
    coverage_radius: float = 0.15
    sense_radius: float = 0.6
    robot_radius: float = 0.05
    avoid_radius: float = 0.16
    avoid_gain: float = 0.9
    avoid_shrink: float = 0.0
    search_mode: str = "index"       # index | fan | last_seen
    fan_span: float = 3.141592653589793
    search_scale: float = 1.0
    goal_select: str = "nearest"     # nearest | index
    keep_track: int = 1

    @property
    def k(self) -> float:
        return 1.0 - float(self.damping)

    @property
    def c(self) -> float:
        return float(self.drive_step)

    @classmethod
    def from_context(cls, context) -> "RuleParams":
        """从 EpisodeContext 的公开 task 参数填充默认值（reset 阶段可用）。"""
        task = context.task
        return cls(
            agent_index=int(context.agent_index),
            num_agents=int(context.num_agents),
            num_targets=int(context.num_targets),
            horizon=int(context.horizon),
            damping=float(task.damping),
            dt=float(task.dt),
            drive_step=float(task.drive_force) / float(task.robot_mass) * float(task.dt),
            coverage_radius=float(task.target_radius),
            sense_radius=float(task.sense_radius),
            robot_radius=float(task.robot_radius),
        )

    def with_overrides(self, overrides: dict | None) -> "RuleParams":
        """只允许覆盖已知字段，避免实验脚本写入拼错的键而静默失效。"""
        if not overrides:
            return self
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"未知的策略参数: {sorted(unknown)}")
        return replace(self, **overrides)


class AnalyticTracker:
    """解析追捕策略：每机器人一个实例，无共享可变状态。"""

    def __init__(self, params: RuleParams):
        self.p = params
        self.last_seen = np.zeros((params.num_targets, 3), dtype=np.float64)
        self.have_seen = np.zeros(params.num_targets, dtype=bool)
        self._radius_table = self._build_radius_table(params.horizon + 2)

    # ------------------------------------------------------------------
    def _build_radius_table(self, max_steps: int):
        """R[t] = 从静止出发、t 步内可达到的位移半径（与场地同尺度）。"""
        dt, k, c = self.p.dt, self.p.k, self.p.c
        table = [0.0]
        cum = 0.0
        c_i = 0.0
        for _ in range(max_steps):
            c_i = c_i * k + c          # 第 i 步的速度增量
            cum += c_i * dt            # 第 i 步的位移
            table.append(cum)
        return table

    def _reachable(self, steps) -> float:
        steps = max(0, min(int(steps), len(self._radius_table) - 1))
        return self._radius_table[steps]

    # ------------------------------------------------------------------
    def reset(self, context=None):
        self.last_seen[:] = 0.0
        self.have_seen[:] = False

    def _search_goal(self) -> np.ndarray:
        n = max(1, self.p.num_agents)
        if self.p.search_mode == "fan":
            span = float(self.p.fan_span)
            angle = 0.0 if n == 1 else -span / 2.0 + span * (self.p.agent_index / (n - 1))
        else:  # index：按编号把 2π 均分，无需通信即可分散
            angle = 2.0 * np.pi * (self.p.agent_index / n)
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64) * self.p.search_scale

    def act(self, obs):
        p = self.p
        st = np.asarray(obs["self_state"], dtype=np.float64)
        self_vel = st[2:4]
        steps_left = max(1, p.horizon - int(obs["step_index"]))

        # 观测按固定容量 A=B=8 给出，实际实体数由 exists 掩码表达
        targets = np.asarray(obs["targets"], dtype=np.float64)[: p.num_targets]
        t_vis = np.asarray(obs["target_visible"], dtype=bool)[: p.num_targets]
        peers = np.asarray(obs["peers"], dtype=np.float64)[: p.num_agents]
        p_vis = np.asarray(obs["peer_visible"], dtype=bool)[: p.num_agents]

        goal = None
        if t_vis.any():
            self.last_seen[t_vis] = targets[t_vis]
            self.have_seen[t_vis] = True
            idx = np.flatnonzero(t_vis)
            dist = np.linalg.norm(targets[idx, :2], axis=1)
            order = idx[np.argsort(dist)]
            if p.keep_track:
                # 优先追"剩余步数内真的够得着"的目标；一个都够不着时才退回最近者
                reach = self._reachable(steps_left) + p.coverage_radius
                ok = order[dist[np.argsort(dist)] <= reach]
                if len(ok):
                    order = ok
            if p.goal_select == "index":
                mine = p.agent_index % max(1, p.num_targets)
                goal = targets[mine, :2] if t_vis[mine] else targets[order[0], :2]
            else:
                goal = targets[order[0], :2]

        # 视野内无目标：优先朝最后目击点，否则按编号方向搜索
        if goal is None and p.search_mode == "last_seen" and self.have_seen.any():
            cand = self.last_seen[self.have_seen]
            goal = cand[int(np.argmin(np.linalg.norm(cand, axis=1)))][:2]
        if goal is None:
            goal = self._search_goal()

        # ---- 解析式追击方向：u ∝ goal − k·v·dt ----
        drive = goal - p.k * self_vel * p.dt

        # ---- 邻居排斥势场（只使用可见邻居，符合局部观测约束）----
        avoid_r = p.avoid_radius
        if p.avoid_shrink > 0.0:
            avoid_r *= max(0.0, min(1.0, steps_left / max(1, p.horizon)))
        for i in range(len(p_vis)):
            if not p_vis[i]:
                continue
            rel = peers[i, :2]
            d = float(np.linalg.norm(rel))
            if d < _EPS:
                continue
            if d < avoid_r:
                w = p.avoid_gain * (avoid_r - d) / avoid_r
                drive = drive - w * (rel / d)

        n = float(np.linalg.norm(drive))
        u = drive / n if n > _EPS else np.zeros(2, dtype=np.float64)
        return np.clip(u, -1.0, 1.0).astype(np.float32)

    def close(self):
        pass


def build_tracker(context, overrides: dict | None = None) -> AnalyticTracker:
    """按 EpisodeContext 构建追踪器（供 entry 与实验工具共用）。"""
    return AnalyticTracker(RuleParams.from_context(context).with_overrides(overrides))


# 兼容旧的实验工具调用名
def build_policy_for_agent(context, artifact_dir=None, params=None):
    return build_tracker(context, params)
