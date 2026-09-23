"""P514 标定/上界探针（仅开发期使用，绝不进入提交）。

两个对照：
- random_ref：均匀随机动作，给出"无信息"下界。
- oracle_nearest：使用训练专用全局状态 env.state()（正式策略禁止访问），
  按"最近目标 + 解析 bang-bang"驱动。它不是合法策略，只用来回答一个诊断问题：
  在同样的 10 步/动力学下，**如果信息无限**，覆盖还能提高多少？
  即"差距里有多少是信息/协同问题，多少是物理可达性问题"。
"""
from __future__ import annotations

import itertools

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


_ORACLES = {
    "OracleNearest": OracleNearest,
    "OracleOptimalAssignment": OracleOptimalAssignment,
}


class OracleGreedySearch(_GlobalAware):
    """最强诊断上界：每步用**真实物理**枚举所有机器人的动作组合，选出下一步
    期望瞬时奖励最高的组合，然后各机器人执行自己那一分量。

    这是"同一物理 + 同一回合长度下的最好情况"的上界估计，比 OracleNearest 强得多：
    - 它直接优化官方指标（含碰撞项），而不是"奔向最近目标"这种代理目标。
    - 它使用全局状态，是正式策略**不允许**的信息权限。
    因为所有机器人在同一状态上做同一枚举，它们在"该组合"上天然一致，无需通信。
    动作集按网格离散（默认每轴 3 档 → 9 个候选 × N 台机器人）。
    只用于回答"还差多少是物理限制"，绝不进入提交。
    """

    def __init__(self, grid=None, collision_weight=0.2, robot_radius=0.05, **kwargs):
        super().__init__(**kwargs)
        levels = np.array(grid if grid is not None else [-1.0, 0.0, 1.0], dtype=np.float64)
        self._levels = levels
        self._cw = float(collision_weight)
        self._robot_r = float(robot_radius)
        self._best_actions = None
        self._cache_key = None

    def reset(self, context=None):
        super().reset(context)
        self._best_actions = None
        self._cache_key = None

    @staticmethod
    def _matching_count(robots, targets, radius):
        """小规模最大匹配（暴力回溯；N=M=3 时足够快）。"""
        n, m = len(robots), len(targets)
        adj = [
            [j for j in range(m) if float(np.linalg.norm(robots[i] - targets[j])) <= radius]
            for i in range(n)
        ]

        def dfs(i, used):
            if i == n:
                return 0
            res = dfs(i + 1, used)  # 机器人 i 不覆盖任何目标
            for j in adj[i]:
                if j not in used:
                    used.add(j)
                    res = max(res, 1 + dfs(i + 1, used))
                    used.discard(j)
            return res

        return dfs(0, set())

    def _plan(self, rob_pos, rob_vel, tgt_pos, tgt_r):
        """枚举所有动作组合，返回使"下一步官方瞬时奖励"最大的组合。"""
        levels = self._levels
        n, m = len(rob_pos), len(tgt_pos)
        candidates = list(itertools.product(levels, repeat=2))
        best_score, best_actions = -np.inf, None
        limit = 1.0 - self._robot_r
        for combo in itertools.product(range(len(candidates)), repeat=n):
            act = np.array([candidates[c] for c in combo], dtype=np.float64)
            new_vel = self.k * rob_vel + self.c * act
            new_pos = np.clip(rob_pos + new_vel * self.dt, -limit, limit)
            matched = self._matching_count(new_pos, tgt_pos, tgt_r)
            cover = matched / m if m else 0.0
            coll_agents = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if float(np.linalg.norm(new_pos[i] - new_pos[j])) < 2.0 * self._robot_r:
                        coll_agents.add(i)
                        coll_agents.add(j)
            coll_rate = (len(coll_agents) / n) if n else 0.0
            score = cover - self._cw * coll_rate
            if score > best_score:
                best_score, best_actions = score, act
        return best_actions

    def act(self, observation):
        robots, robot_exists, targets, target_exists = self._parse()
        # 每个决策步所有 agent 共用同一局面：用局面签名做缓存，避免重复枚举
        key = float(np.sum(robots)) + float(np.sum(targets)) * 3.0
        if self._cache_key is None or abs(key - self._cache_key) > 0.0:
            rob_pos = robots[robot_exists, :2]
            rob_vel = robots[robot_exists, 2:4]
            tgt_pos = targets[target_exists, :2]
            tgt_r = float(targets[target_exists, 4].mean()) if target_exists.any() else 0.15
            self._best_actions = self._plan(rob_pos, rob_vel, tgt_pos, tgt_r)
            self._cache_key = key
        return np.asarray(self._best_actions[self.agent_index], dtype=np.float32)


_ORACLES["OracleGreedySearch"] = OracleGreedySearch


def build_policy_for_agent(context, artifact_dir=None, params=None):
    """按 context 构建诊断探针。

    params 里可用 "_class" 选择 oracle 类型；其余键作为构造参数传入。
    注意：本模块只用于开发期标定，entry.py 不会导入它。
    """
    params = dict(params or {})
    name = params.pop("_class", "OracleNearest")
    task = context.task
    kw = dict(
        num_agents=context.num_agents,
        num_targets=context.num_targets,
        horizon=context.horizon,
        damping=task.damping,
        dt=task.dt,
        drive_step=task.drive_force / task.robot_mass * task.dt,
    )
    if name == "OracleGreedySearch":
        kw["robot_radius"] = float(task.robot_radius)
        kw["collision_weight"] = 0.2
    kw.update(params)
    cls = _ORACLES[name]
    return cls(**kw)


class RandomPolicy(RandomRef):
    pass
