"""P514 规则策略 v1：解析式追捕 + 视野外定向搜索 + 邻居避碰。

全部依据来自公开协议与 EpisodeContext 的公开 task 参数，不读取任何隐藏信息。

动力学推导（关键，决定了"追捕"不需要学习）：
    真实更新   v_{t+1} = (1-damping) * v_t + (drive_force / mass) * dt * u_t
    单步位移   pos_{t+1} = pos_t + v_{t+1} * dt
    本任务冻结参数：damping=0.25、dt=0.1、drive_force/mass=1.0 ⇒ k=0.75、c=0.1。
    从静止出发、动作恒为满舵时，第 i 步结束后的速度 v_i = c·(1-k^i)/(1-k)，
    而位置更新用的是**上一步**的速度，故 n 步累计位移
        S(n) = dt · Σ_{i=1..n} v_{i-1} = dt · c · Σ_{i=1..n} (1 - k^(i-1)) / (1-k)
    S(10) ≈ 0.249：**10 步整场最多只能移动约 0.25（场地半宽 1.0）**。
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
    avoid_radius: float = 0.20
    avoid_gain: float = 0.9
    avoid_shrink: float = 0.0
    search_mode: str = "index"       # index | fan | last_seen | center | bounce | unvisited
    fan_span: float = 3.141592653589793
    search_scale: float = 1.0
    search_angle_offset: float = 0.0   # 搜索基准角偏移（index 模式）：所有机器人一起旋转
    search_angle_offset_1: float = 0.0  # 仅 agent_index==1 的额外偏移（用来打破均匀分布）
    center_weight: float = 1.0         # center 模式下朝场地中心的权重
    goal_select: str = "nearest"     # nearest | index | coordinated | cover
    keep_track: int = 1
    # ---- 覆盖时长最优的目标选择（第五轮新增）----
    # 动机：离线"恒定动作上界"在 T=10 上比本规则高约 41%（32 场景 5.97 vs 4.22），
    # 且最优动作里大量是 [0,0]（**原地不动**）—— 说明奖励是**逐步累加**的，
    # 而"奔向最近目标"只优化了"到达"，没优化"停留多久"。
    # 覆盖半径 0.15 的圆内，目标自己会走动，因此"让目标走进来"常常优于"追过去"。
    # cover 模式按"最早可覆盖时刻 t_start + ω·持续步数"给每个可见目标打分并取最优。
    cover_hold_weight: float = 0.5
    cover_horizon: int = 24
    thrust_mode: str = "full"        # full | arrive（按剩余可达位移缩放推力，近处不冲过头）
    reach_mode: str = "exact"        # exact（修正后的真实可达半径）| legacy（旧的有缺陷实现）
    # ---- 搜索阶段的"持续探索"（第五轮新增，修复长回合顶墙退化）----
    # 旧行为：视野内无目标时按 agent_index 取固定方向**满舵**，机器人在 T 步内
    # 必然开到墙上并永久停住（实测 T=256 时 2 号机第 65 步到角落，之后 191 步静止）。
    # bounce：到达边界前主动转向，使搜索轨迹在整个场地内遍历（反射式巡逻）。
    # unvisited：优先朝"本地未访问网格"前进，网格全部访问后回退到 bounce。
    search_bounce_margin: float = 0.25   # 距边界多少范围内开始计算转向
    search_bounce_look: float = 0.35     # 前瞻距离
    explore_cell: float = 0.3            # 未访问网格边长
    explore_radius: int = 3              # 只维护“自身周围 ±radius 格”的访问位图（内存有界）
    # ---- 拦截（预估目标速度，瞄准未来位置而非当前位置）----
    intercept: int = 0               # 1=启用速度估计与拦截瞄准
    vel_ema: float = 1.0             # 速度估计的 EMA 系数（1.0=只用最近两次观测）
    intercept_max_lead: int = 6      # 最多向前预判多少步
    intercept_weight: float = 1.0    # 1.0=完全瞄准预判点；<1 时与当前位置混合
    # ---- 交叉感知避碰（第六轮新增）----
    # 动机：长回合下把搜索模式从"固定方向"改成"反射巡逻"后覆盖率上升，
    # 但**碰撞惩罚同步上升**，净 J 反而下降（T=256: 0.6118 -> 0.6006）。
    # 原有避碰是"距离小于 avoid_radius 就施加径向排斥"，它是**反应式**的：
    # 等两台机器人已经贴到一起才开始推，且推力与前进力互相抵消（都是单位向量），
    # 结果既没完全避开、又偏了方向。
    # 本项改为**预测式侧向让行**：用相对位置与相对速度算出"最近接近距离 d_min"
    # 与"多久后接近 t_cpa"，若预判会侵入 contact_radius 则沿垂直方向让行，
    # 让行方向固定（按相对速度的固定侧），因此双方不会互相镜像翻转而锁死。
    # 只使用可见队友的相对位置/相对速度，不需要通信，符合局部观测权限。
    avoid_vel: int = 1               # 1=启用预测式侧向让行
    contact_radius: float = 0.115    # 预判侵入阈值（机器人直径 0.10 + 余量）
    avoid_horizon: float = 3.5       # 预判时间窗（步）
    avoid_vel_gain: float = 0.6      # 让行强度（0.6 是两套种子集上的稳健取值，见 LOG §40）

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
        # 持续探索状态（反射巡逻方向 + 本地已访问网格）
        self._explore_dir = np.zeros(2, dtype=np.float64)
        self._visited: set[tuple[int, int]] = set()
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
            # 尚无该目标的任何观测：退回"自身相对原点"（首步且目标不可见时的中性假设），
            # 绝不能用截断到 0 的缓冲值（那会让上层误以为目标在场地中心）。
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
        """R[n] = 从静止出发、n 步内可达到的位移半径（与场地同尺度）。

        推导：速度与位置对控制都是线性的，且单步位移 = v·dt，
        故 `max_u v(p)·p̂ = ‖v(p)‖·max_u(u·p̂) = ‖v(p)‖`，
        即"满舵朝一个固定方向"就是让 n 步位移最大的最优控制，
        R[n] = dt·Σ_{i=1..n} v_{i-1}，其中 v_i = c·Σ_{j=0..i-1} k^j。
        例：R[10] = 0.249010（由 diagnose.py 的真实动力学推演逐位核对）。

        ⚠️ 历史缺陷（第五轮修正）：旧实现把"速度累加"当成了位移表，
        少了 dt 因子，因此旧 R[n] 恰好是真实值的 1/dt = 10 倍速度尺度 ——
        等价于"多算约一步"（旧 R[10]=0.2868 > 真实 0.2490）。
        这会让可达筛选把 0.40~0.44 区间的目标误判为够得着。
        保留 `reach_mode="legacy"` 只为复核历史数字，默认不用。
        """
        dt, k, c = self.p.dt, self.p.k, self.p.c
        if self.p.reach_mode == "legacy":
            table = [0.0]
            acc = 0.0
            vv = 0.0
            for _ in range(max_steps):
                vv = vv * k + c
                acc += vv
                table.append(acc)
            return table
        table = [0.0]
        cum = 0.0
        v_prev = 0.0                      # 位置更新用的是上一步速度
        last_v = 0.0
        for _ in range(max_steps):
            cum += v_prev * dt            # 第 i 步的位移（用的是 v_{i-1}）
            table.append(cum)
            last_v = last_v * k + c
            v_prev = last_v
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
        # 规程 §14：每局必须清除历史观测。探索状态同属回合内记忆，必须一并清空，
        # 且不同机器人之间不共享（每台机器人一个实例）。
        self._explore_dir[:] = 0.0
        self._visited.clear()

    def _local_assignment(self, targets, t_vis, peers, p_vis):
        """无通信的确定性"就近分配"：把可见目标分配给"自己 + 可见队友"。

        每台机器人只用自己看得到的对手表与目标表做同一套贪心匹配，
        在信息一致时得到相同的解，因此不需要通信通道也能减少撞车。
        只使用可见队友与可见目标，完全符合局部观测权限。

        返回：分配给"自己"的目标下标；未分配则返回 None。
        """
        # 局中人：自己（下标 0）+ 可见队友（相对位置已在 peers 里）
        poses = [np.zeros(2, dtype=np.float64)]
        owner = [self.p.agent_index]
        for i in range(len(p_vis)):
            if p_vis[i]:
                poses.append(peers[i, :2])
                owner.append(i)  # 观测里的槽位号即队友编号
        poses = np.asarray(poses, dtype=np.float64)

        tgt_idx = np.flatnonzero(t_vis)
        if len(tgt_idx) == 0:
            return None
        tgt_pos = targets[tgt_idx, :2]

        dmat = np.linalg.norm(poses[:, None, :] - tgt_pos[None, :, :], axis=2)
        order = np.dstack(np.unravel_index(np.argsort(dmat, axis=None), dmat.shape))[0]
        taken_agents: set[int] = set()
        taken_targets: set[int] = set()
        assign: dict[int, int] = {}
        for ai, ti in order:
            if ai in taken_agents or ti in taken_targets:
                continue
            assign[int(ai)] = int(ti)
            taken_agents.add(int(ai))
            taken_targets.add(int(ti))

        mine = assign.get(0)
        return None if mine is None else int(tgt_idx[mine])

    # ---- 覆盖时长评分（goal_select="cover"）----------------------------
    def _cover_score(self, j: int, step_idx: int, steps_left: int, self_pos: np.ndarray):
        """估计对目标 j 的"最早可覆盖时刻 + 持续步数"，返回可比分数。

        用直线匀速外推测目标未来位置（目标速度由相邻观测差分得到；无速度信息时
        假定静止），对每个未来时刻 t 判断：
            ‖目标预测位置 − 自身当前位置‖ ≤ R(steps_left − t) + coverage_radius
        取最小的 t 作为 t_start，并统计从 t_start 起连续可覆盖的步数作为持续步数。
        当 t_start == 0 且持续步数为 0 时（目标太远且不可达），返回 None。
        """
        p = self.p
        horizon = min(int(p.cover_horizon), max(0, steps_left))
        first = None
        hold = 0
        for t in range(0, horizon + 1):
            pred = self._predicted_rel(j, step_idx + t, self_pos)
            reach_t = self._reachable(steps_left - t) if t > 0 else 0.0
            if float(np.linalg.norm(pred)) <= reach_t + p.coverage_radius:
                if first is None:
                    first = t
                if t == first + hold:
                    hold += 1
            elif first is not None:
                break
        if first is None:
            return None
        return float(first) - float(p.cover_hold_weight) * float(hold)

    def _search_goal(self, self_pos: np.ndarray | None = None) -> np.ndarray:
        n = max(1, self.p.num_agents)
        if self.p.search_mode in ("bounce", "unvisited") and self_pos is not None:
            if self.p.search_mode == "unvisited":
                g = self._explore_goal(self_pos)
                if g is not None:
                    return g * self.p.search_scale
            return self._bounce_dir(self_pos) * self.p.search_scale
        if self.p.search_mode == "center" and self_pos is not None:
            # 位置感知搜索：目标按 uniform(±(L−r)) 布设，靠近场地中心的**面密度更高**，
            # 而感知半径内到边界的面积远小于中心。因此原地不动/固定方向都不如朝中心走。
            # 用"朝圆心 + 沿编号切向分量"兼顾密度与分散（纯朝中心会让多机挤到一起）。
            to_center = -np.asarray(self_pos, dtype=np.float64)
            nc = float(np.linalg.norm(to_center))
            if nc > _EPS:
                radial = to_center / nc
            else:
                radial = np.zeros(2, dtype=np.float64)
            angle = 2.0 * np.pi * (self.p.agent_index / n)
            tangential = np.array([-radial[1], radial[0]], dtype=np.float64)
            w = float(np.clip(self.p.center_weight, 0.0, 1.0))
            goal = w * radial + (1.0 - w) * tangential
            if float(np.linalg.norm(goal)) < _EPS:
                goal = tangential
            return goal * self.p.search_scale
        if self.p.search_mode == "fan":
            span = float(self.p.fan_span)
            angle = 0.0 if n == 1 else -span / 2.0 + span * (self.p.agent_index / (n - 1))
        else:  # index：按编号把 2π 均分，无需通信即可分散
            angle = 2.0 * np.pi * (self.p.agent_index / n)
        angle += self.p.search_angle_offset
        if self.p.agent_index == 1:
            angle += self.p.search_angle_offset_1
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64) * self.p.search_scale

    # ---- 持续探索（长回合修复）----------------------------------------
    def _bounce_dir(self, self_pos: np.ndarray) -> np.ndarray:
        """反射式巡逻方向：保持当前朝向，若按前瞻距离会越界则把该分量反向。

        旧实现的固定方向在长回合下必然把机器人停在墙上（见 RuleParams 注释），
        本函数让搜索轨迹在整个场地内持续遍历，不依赖任何额外观测。
        """
        v = np.asarray(self._explore_dir, dtype=np.float64)
        if float(np.linalg.norm(v)) < _EPS:
            angle = 2.0 * np.pi * (self.p.agent_index / max(1, self.p.num_agents))
            v = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        look = float(self.p.search_bounce_look)
        limit = 1.0 - self.p.robot_radius
        proj = np.asarray(self_pos, dtype=np.float64) + v * look
        for axis in (0, 1):
            if proj[axis] > limit:
                v[axis] = -abs(v[axis])
            elif proj[axis] < -limit:
                v[axis] = abs(v[axis])
        nrm = float(np.linalg.norm(v))
        if nrm < _EPS:
            v = np.array([1.0, 0.0], dtype=np.float64)
            nrm = 1.0
        v = v / nrm
        self._explore_dir = v.copy()
        return v

    def _explore_goal(self, self_pos: np.ndarray):
        """找"自身周围最近的一个未访问网格中心"，朝它前进；全访问过则返回 None。"""
        p = self.p
        cell = max(1e-3, float(p.explore_cell))
        rad = max(1, int(p.explore_radius))
        c0 = np.floor(np.asarray(self_pos, dtype=np.float64) / cell).astype(np.int64)
        best = None
        best_d2 = np.inf
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                gx, gy = int(c0[0]) + dx, int(c0[1]) + dy
                if (gx, gy) in self._visited:
                    continue
                center = np.array([(gx + 0.5) * cell, (gy + 0.5) * cell], dtype=np.float64)
                if float(np.abs(center).max()) > 1.0:
                    self._visited.add((gx, gy))   # 场外格子直接标记，避免反复选中
                    continue
                d2 = float(np.dot(center - self_pos, center - self_pos))
                if d2 < best_d2:
                    best_d2, best = d2, center
        if best is None:
            return None
        return best - np.asarray(self_pos, dtype=np.float64)

    def act(self, obs):
        p = self.p
        st = np.asarray(obs["self_state"], dtype=np.float64)
        self_pos = st[:2]
        self_vel = st[2:4]
        step_idx = int(obs["step_index"])
        steps_left = max(1, p.horizon - step_idx)

        # 记录本地访问网格（只在 unvisited 模式需要；容量有上界，避免长回合无界增长）
        if p.search_mode == "unvisited" and len(self._visited) < 4096:
            cell = max(1e-3, float(p.explore_cell))
            self._visited.add(
                (int(np.floor(self_pos[0] / cell)), int(np.floor(self_pos[1] / cell)))
            )

        # 观测按固定容量 A=B=8 给出，实际实体数由 exists 掩码表达
        targets = np.asarray(obs["targets"], dtype=np.float64)[: p.num_targets]
        t_vis = np.asarray(obs["target_visible"], dtype=bool)[: p.num_targets]
        peers = np.asarray(obs["peers"], dtype=np.float64)[: p.num_agents]
        p_vis = np.asarray(obs["peer_visible"], dtype=bool)[: p.num_agents]

        # ---- 目标观测记录（无条件进行）----
        # 必须**无条件**存储：`_cover_score`（goal_select="cover"）与 `search_mode="last_seen"`
        # 都依赖绝对位置与速度估计，旧实现把它们锁在 `if p.intercept` 里面，
        # 导致这两个功能在没有开启拦截时静默失效（_predicted_rel 返回垃圾 → 退化。
        # 这个坑真实发生过：cover 与 nearest 在 128 个场景上逐位一致）。
        if t_vis.any():
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
                self.last_seen[j, :2] = targets[j, :2]
                self.last_seen[j, 2] = targets[j, 2]

        goal = None
        if t_vis.any():
            idx = np.flatnonzero(t_vis)
            dist = np.linalg.norm(targets[idx, :2], axis=1)
            order = idx[np.argsort(dist)]

            # 协同选择：用"自己 + 可见队友"的就近分配决定自己负责哪个目标
            assigned = None
            if p.goal_select == "coordinated":
                assigned = self._local_assignment(targets, t_vis, peers, p_vis)
                if assigned is not None:
                    order = np.array(
                        [assigned] + [j for j in order if j != assigned], dtype=np.int64
                    )

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
                if p.goal_select == "cover":
                    # 覆盖时长最优：按"最早可覆盖时刻 + ω·持续步数"取分数最小者。
                    # 分数 < 0 表示当前已在覆盖半径内（立刻有收益）→ 直接奔向它，
                    # 解析式追击会自动把速度调到"贴着目标走"，等价于原地驻留跟随。
                    best = None
                    for j in order:
                        sc = self._cover_score(int(j), step_idx, steps_left, self_pos)
                        if sc is None:
                            continue
                        if best is None or sc < best[0]:
                            best = (sc, int(j))
                    goal = targets[best[1], :2] if best is not None else targets[order[0], :2]
                elif p.goal_select == "index":
                    mine = p.agent_index % max(1, p.num_targets)
                    goal = targets[mine, :2] if t_vis[mine] else targets[order[0], :2]
                else:
                    goal = targets[order[0], :2]

        # 视野内无目标：优先朝最后目击点，否则按编号方向搜索
        if goal is None and p.search_mode == "last_seen" and self.have_seen.any():
            cand = self.last_seen[self.have_seen]
            goal = cand[int(np.argmin(np.linalg.norm(cand, axis=1)))][:2]
        if goal is None:
            goal = self._search_goal(self_pos)

        # ---- 解析式追击方向：u ∝ goal − k·v·dt ----
        drive = goal - p.k * self_vel * p.dt

        # ---- 邻居避碰（只使用可见邻居，符合局部观测约束）----
        # 1) 预测式侧向让行（第六轮）：先按相对速度预判"会不会撞上"，会则提前侧移。
        #    这一步放在径向排斥之前，因为它给出的是**方向性**修正，
        #    而径向排斥在近距离会与前进力互相抵消。
        if p.avoid_vel:
            vg = float(p.avoid_vel_gain)
            n_vis = max(1, int(np.sum(p_vis)))
            # 拥挤时衰减：可见队友越多，单次让行幅度越小。
            # 用 1/n_vis 而不是 1/√n_vis —— 大规模格子（6v3/8v8）里
            # 侧向修正会累积成整体偏航，线性衰减能把这部分代价压掉。
            crowd = 1.0 / float(n_vis)
            for i in range(len(p_vis)):
                if not p_vis[i]:
                    continue
                rel = peers[i, :2]
                if float(np.linalg.norm(rel)) < _EPS:
                    continue
                # 观测给的是"队友相对自身"的位置与速度，故（自身速度 − 队友速度）= −dv
                dvel = -np.asarray(peers[i, 2:4], dtype=np.float64)
                rv = float(np.linalg.norm(dvel))
                if rv < _EPS:
                    continue
                t_cpa = -float(np.dot(rel, dvel)) / (rv * rv)
                if t_cpa <= 0.0 or t_cpa > p.avoid_horizon:
                    continue                      # 正在远离或还太远，不必让
                d_min = float(np.linalg.norm(rel + dvel * t_cpa))
                if d_min >= p.contact_radius:
                    continue                      # 预判不会侵入，保持航线
                # 让行方向取"相对速度的固定侧"法向（双方算法一致 → 不会镜像翻转）
                e_perp = np.array([-dvel[1], dvel[0]], dtype=np.float64) / rv
                urgency = (p.contact_radius - d_min) / p.contact_radius
                # 采用**叠加**而非旋转：旋转保持模长但会把净前进方向整体偏转，
                # 在密集场景里造成绕路；叠加只是用一部分推力做侧移，
                # 实测在 12 格规模轴上的平均代价更小（+0.0001 vs −0.0011）。
                drive = drive + vg * crowd * urgency * e_perp * float(np.linalg.norm(drive))

        # 2) 径向排斥（原有机制）：贴得很近时的兜底推离
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
        if p.thrust_mode == "arrive":
            # 到达式推力：剩余可达位移 R(steps_left) 大于目标距离时不必满舵，
            # 否则会在覆盖半径内"冲过头"再折返，浪费掉本来可以持续覆盖的步数。
            # 目标距离按"到覆盖半径边缘"计（进半径即算覆盖）。
            gap = max(0.0, float(np.linalg.norm(goal)) - p.coverage_radius)
            reach = self._reachable(steps_left)
            if reach > _EPS and gap < reach:
                u = u * float(np.clip(gap / reach, 0.0, 1.0))
        return np.clip(u, -1.0, 1.0).astype(np.float32)

    def close(self):
        pass


def build_tracker(context, overrides: dict | None = None) -> AnalyticTracker:
    """按 EpisodeContext 构建追踪器（供 entry 与实验工具共用）。"""
    return AnalyticTracker(RuleParams.from_context(context).with_overrides(overrides))


# 兼容旧的实验工具调用名
def build_policy_for_agent(context, artifact_dir=None, params=None):
    return build_tracker(context, params)
