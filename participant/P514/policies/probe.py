"""P514 标定/上界探针（仅开发期使用，绝不进入提交）。

两个对照：
- random_ref：均匀随机动作，给出"无信息"下界。
- oracle_nearest：使用训练专用全局状态 env.state()（正式策略禁止访问），
  按"最近目标 + 解析 bang-bang"驱动。它不是合法策略，只用来回答一个诊断问题：
  在同样的 10 步/动力学下，**如果信息无限**，覆盖还能提高多少？
  即"差距里有多少是信息/协同问题，多少是物理可达性问题"。
"""
from __future__ import annotations

import numpy as np

POS_SCALE = 1.0
VEL_SCALE = 1.0


class _Base:
    def reset(self, context=None):
        self.ctx = context
        self.step_index = 0

    def close(self):
        pass


class RandomRef(_Base):
    def __init__(self, policy_seed=0, **kwargs):
        self._seed = int(policy_seed)
        self._rng = np.random.default_rng(self._seed)

    def reset(self, context=None):
        super().reset(context)
        seed = getattr(context, "policy_seed", None)
        self._rng = np.random.default_rng(int(seed) if seed is not None else self._seed)

    def act(self, observation):
        return self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


class _GlobalAware(_Base):
    """读全局状态的诊断用基类：主进程在每个 act 前注入 global_state。"""

    def __init__(self, num_agents=3, num_targets=3, horizon=10, damping=0.25, dt=0.1, drive_step=0.1, **kwargs):
        self.n = int(num_agents)
        self.m = int(num_targets)
        self.horizon = int(horizon)
        self.k = 1.0 - float(damping)
        self.dt = float(dt)
        self.c = float(drive_step)
        self.global_state = None
        self.agent_index = 0

    def reset(self, context=None):
        super().reset(context)
        if context is not None:
            self.agent_index = int(context.agent_index)
            self.n = int(context.num_agents)
            self.m = int(context.num_targets)
            self.horizon = int(context.horizon)

    def _parse(self):
        A, B = 8, 8
        s = np.asarray(self.global_state, dtype=np.float64)
        robots = s[0 : 5 * A].reshape(A, 5)
        robot_exists = s[5 * A : 6 * A].astype(bool)
        targets = s[6 * A : 6 * A + 5 * B].reshape(B, 5)
        target_exists = s[6 * A + 5 * B : 6 * A + 6 * B].astype(bool)
        return robots, robot_exists, targets, target_exists

    def _bang_bang(self, self_pos, self_vel, goal_abs_pos):
        goal_rel = goal_abs_pos - self_pos
        drive = goal_rel - self.k * self_vel * self.dt
        n = float(np.linalg.norm(drive))
        return (drive / n if n > 1e-12 else np.zeros(2)).astype(np.float32)

    def act(self, observation):
        raise NotImplementedError


class OracleNearest(_GlobalAware):
    """诊断上界：每个机器人奔向最近的存在目标。"""

    def act(self, observation):
        robots, robot_exists, targets, target_exists = self._parse()
        me = self.agent_index
        self_pos = robots[me, :2]
        self_vel = robots[me, 2:4]
        cand = targets[target_exists, :2]
        if len(cand) == 0:
            return np.zeros(2, dtype=np.float32)
        d = np.linalg.norm(cand - self_pos, axis=1)
        return self._bang_bang(self_pos, self_vel, cand[int(np.argmin(d))])


class OracleOptimalAssignment(_GlobalAware):
    """诊断上界：全局贪心分配（最近优先去重），再各自 bang-bang 过去。"""

    def act(self, observation):
        robots, robot_exists, targets, target_exists = self._parse()
        n = int(robot_exists.sum())
        tgt = targets[target_exists, :2]
        rob = robots[robot_exists, :2]
        if len(tgt) == 0:
            return np.zeros(2, dtype=np.float32)
        # 距离矩阵，按"谁离得最近"依次配对
        dmat = np.linalg.norm(rob[:, None, :] - tgt[None, :, :], axis=2)
        assign = np.full(n, -1, dtype=int)
        used_t = set()
        order = np.dstack(np.unravel_index(np.argsort(dmat, axis=None), dmat.shape))[0]
        for ri, ti in order:
            if assign[ri] == -1 and ti not in used_t:
                assign[ri] = ti
                used_t.add(ti)
        me = self.agent_index
        goal = tgt[assign[me]] if assign[me] >= 0 else tgt[int(np.argmin(dmat[me]))]
        return self._bang_bang(rob[me], robots[me, 2:4], goal)


def build_policy_for_agent(context, artifact_dir=None, params=None):
    task = context.task
    kw = dict(
        num_agents=context.num_agents,
        num_targets=context.num_targets,
        horizon=context.horizon,
        damping=task.damping,
        dt=task.dt,
        drive_step=task.drive_force / task.robot_mass * task.dt,
    )
    kw.update(params or {})
    name = (params or {}).get("_class", "OracleNearest")
    return {"OracleNearest": OracleNearest, "OracleOptimalAssignment": OracleOptimalAssignment}[name](**kw)


class RandomPolicy(RandomRef):
    pass
