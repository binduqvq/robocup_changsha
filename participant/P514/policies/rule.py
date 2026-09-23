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
    search_angle_offset: float = 0.0   # 搜索基准角偏移（index 模式）：所有机器人一起旋转
    search_angle_offset_1: float = 0.0  # 仅 agent_index==1 的额外偏移（用来打破均匀分布）
    goal_select: str = "nearest"     # nearest | index
    keep_track: int = 1
    # ---- 拦截（预估目标速度，瞄准未来位置而非当前位置）----
    intercept: int = 0               # 1=启用速度估计与拦截瞄准
    vel_ema: float = 1.0             # 速度估计的 EMA 系数（1.0=只用最近两次观测）
    intercept_max_lead: int = 6      # 最多向前预判多少步
    intercept_weight: float = 1.0    # 1.0=完全瞄准预判点；<1 时与当前位置混合

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
        # 拦截所需的逐目标观测历史（绝对位置、观测步、估计速度）
        self._obs_pos_abs = np.zeros((params.num_targets, 2), dtype=np.float64)
        self._obs_step = np.full(params.num_targets, -1, dtype=np.int64)
        self._vel = np.zeros((params.num_targets, 2), dtype=np.float64)
        self._radius_table = self._build_radius_table(params.horizon + 2)

    def _predicted_rel(self, j: int, at_step: int, self_pos: np.ndarray) -> np.ndarray:
        """目标 j 在 at_step 时刻相对**当前**自身位置的预测位置。

        观测给的是"目标 − 自身"的相对量，而自身也在移动，因此不能直接对
        相对量做外推：必须先还原成绝对坐标再预测。
            target_abs(t_obs) = self_abs(t_obs) + rel_obs
            target_abs(t)     = target_abs(t_obs) + v̂·(t − t_obs)
            rel_pred(t)       = target_abs(t) − self_abs(t)
        目标在 T=10 内转向间隔 5~10 步，几个步长内的匀速外推是合理近似。
        """
        if not self.have_seen[j] or self._obs_step[j] < 0:
            return self._obs_pos_abs[j] - self_pos
        age = at_step - self._obs_step[j]
        if age <= 0:
            return self._obs_pos_abs[j] - self_pos
        pred_abs = self._obs_pos_abs[j] + self._vel[j] * age
        # 目标被限制在 ±(L − r_target) 内反射，夹紧避免外推到场地外
        limit = 1.0 - self.p.coverage_radius
        pred_abs = np.clip(pred_abs, -limit, limit)
        return pred_abs - self_pos

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
        self._obs_pos_abs[:] = 0.0
        self._obs_step[:] = -1
        self._vel[:] = 0.0

    def _search_goal(self) -> np.ndarray:
        n = max(1, self.p.num_agents)
        if self.p.search_mode == "fan":
            span = float(self.p.fan_span)
            angle = 0.0 if n == 1 else -span / 2.0 + span * (self.p.agent_index / (n - 1))
            angle += self.p.search_angle_offset
        else:  # index：按编号把 2π 均分，无需通信即可分散
            angle = 2.0 * np.pi * (self.p.agent_index / n) + self.p.search_angle_offset
            if self.p.agent_index == 1:
                angle += self.p.search_angle_offset_1
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64) * self.p.search_scale

    def act(self, obs):
        p = self.p
        st = np.asarray(obs["self_state"], dtype=np.float64)
        self_pos = st[:2]
        self_vel = st[2:4]
        step_idx = int(obs["step_index"])
        steps_left = max(1, p.horizon - step_idx)

        # 观测按固定容量 A=B=8 给出，实际实体数由 exists 掩码表达
        targets = np.asarray(obs["targets"], dtype=np.float64)[: p.num_targets]
        t_vis = np.asarray(obs["target_visible"], dtype=bool)[: p.num_targets]
        peers = np.asarray(obs["peers"], dtype=np.float64)[: p.num_agents]
        p_vis = np.asarray(obs["peer_visible"], dtype=bool)[: p.num_agents]

        # ---- 目标速度估计：用两次"相邻步"观测做有限差分（目标不提供速度字段）----
        if p.intercept and t_vis.any():
            for j in np.flatnonzero(t_vis):
                gap = step_idx - self._obs_step[j]
                if self._obs_step[j] >= 0 and gap == 1:
                    v = (targets[j, :2] - self._obs_pos_abs[j] + self_pos) / p.dt  # 场地尺度/步
                    self._vel[j] = p.vel_ema * v + (1.0 - p.vel_ema) * self._vel[j]
                elif gap > 1:
                    # 间隔超过 1 步：期间可能发生转向，不能把旧差分当速度用
                    self._vel[j] *= 0.0
                self._obs_pos_abs[j] = targets[j, :2] + self_pos
                self._obs_step[j] = step_idx
                self.have_seen[j] = True

        goal = None
        if t_vis.any():
            idx = np.flatnonzero(t_vis)
            dist = np.linalg.norm(targets[idx, :2], axis=1)
            order = idx[np.argsort(dist)]

            if p.intercept:
                # 拦截：分别在每个目标上求"最小可达步数 t"，取 t 最小者，
                # 并瞄准该目标在 t 步后的预测位置（目标在 T=10 内近似弹道运动）。
                lead_cap = min(int(p.intercept_max_lead), steps_left)
                best = None
                for k, j in enumerate(order):
                    base = targets[j, :2]
                    t_min = None
                    if float(np.linalg.norm(base)) <= self._reachable(steps_left) + p.coverage_radius:
                        t_min = 0
                    else:
                        for t in range(1, lead_cap + 1):
                            pt = self._predicted_rel(j, step_idx + t, self_pos)
                            if float(np.linalg.norm(pt)) <= self._reachable(steps_left - t) + p.coverage_radius:
                                t_min = t
                                break
                    if t_min is None:
                        continue
                    cand_pt = self._predicted_rel(j, step_idx + t_min, self_pos)
                    if best is None or t_min < best[0]:
                        best = (t_min, cand_pt)
                if best is not None:
                    goal = (1.0 - p.intercept_weight) * targets[order[0], :2] + p.intercept_weight * best[1]
            if goal is None:
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
