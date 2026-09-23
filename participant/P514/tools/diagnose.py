"""P514 诊断工具：量化"规则策略离真正的多步最优还有多远"（开发期，不参与推理）。

动机（第五轮迭代）：前四轮把结论写成"信息不是瓶颈、物理可达性是瓶颈、规则已达上界"。
但那个"上界"用的是 OracleNearest —— 它本身也是**短视**的（E007c 已证明瞬时贪心在本题是错的）。
因此严格来说，我们从未建立过**真正的**上界，只证明了"规则 ≈ 另一个短视启发式"。
本工具用三类更强的对照来补上这个缺口：

R0  规则策略（提交版本）在官方环境里的真实回放。
R1  **离线最优上界**：用记录下来的真实机器人轨迹还原出"完美未来目标轨迹"
    （目标运动与机器人动作完全解耦，见下），再对每台机器人做
    "离线恒定动作"枚举：r = argmax_r Σ_t cov_match(pos_t(r), targets_t)。
    这是**开环、全信息、逐机器人无约束**的上界，因此 ≥ 任何在线策略。
R2  **可达性剥离**：把目标分成"整场都够不着"与"至少够得着一次"，
    报告规则在"够得着"那部分上的**取用率** —— 直接区分
    "物理没给机会"与"给了机会但没抓住"。

复现级仿真器（sim_*）：按官方物理与目标运动逐位复算，并由 main() 断言与真实轨迹一致；
只有断言通过，才能用它在"未真正跑过的动作序列"上做推演。

用法：
    .venv/Scripts/python.exe participant/P514/tools/diagnose.py --n 12 --repeats 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_P514 = _REPO_ROOT / "participant" / "P514"
for _p in (str(_REPO_ROOT), str(_P514)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coverage_bench.envs.factory import make_training_env  # noqa: E402

from tools.rollout import POLICIES, build_agent_policy  # noqa: E402
from tools.sweep import make_heldout_cases  # noqa: E402

_ORACLE_ASSUMED_SPEED = 0.2  # 仅用于 first_possible() 的速度上界近似


# ----------------------------------------------------------------------
# 复现级仿真器
# ----------------------------------------------------------------------
def sim_reflect(pos, vel, dt, low, high):
    """与 coverage_bench.envs.motion.reflect_coordinate 等价的向量化复算。"""
    pos = float(pos)
    vel = float(vel)
    remaining = float(dt)
    for _ in range(8):
        if vel == 0.0:
            return pos, 0.0
        wall = high if vel > 0 else low
        t_hit = (wall - pos) / vel
        if t_hit > remaining:
            return pos + vel * remaining, vel
        pos = wall
        vel = -vel
        remaining -= t_hit
    return pos, vel


def sim_robots(rob_pos, rob_vel, cmds, k, c, dt, max_speed, radius, half_extent):
    """复算一步机器人动力学（内置欧拉分支，与 envs/physics.advance_robots 一致）。"""
    p_force = np.array(cmds, dtype=np.float64, copy=True)  # drive_force = 1.0
    n = len(rob_pos)
    for i in range(n):
        for j in range(i + 1, n):
            delta = rob_pos[i] - rob_pos[j]
            d = float(np.linalg.norm(delta))
            dist_min = 2.0 * radius
            contact_force, contact_margin = 100.0, 0.001
            if d == 0.0:
                normal = np.array([1.0, 0.0])
                pen = np.logaddexp(0.0, dist_min / contact_margin) * contact_margin
                f = contact_force * normal * pen
            else:
                pen = np.logaddexp(0.0, -(d - dist_min) / contact_margin) * contact_margin
                f = contact_force * (delta / d) * pen
            p_force[i] = p_force[i] + f
            p_force[j] = p_force[j] - f

    new_pos = np.empty_like(rob_pos)
    new_vel = np.empty_like(rob_vel)
    limit = half_extent - radius
    for i in range(n):
        pos = rob_pos[i] + rob_vel[i] * dt
        vel = k * rob_vel[i] + p_force[i] * dt
        speed = float(np.linalg.norm(vel))
        if speed > max_speed:
            vel = vel * (max_speed / speed)
        for axis in (0, 1):
            if pos[axis] > limit:
                pos[axis] = limit
                if vel[axis] > 0.0:
                    vel[axis] = 0.0
            elif pos[axis] < -limit:
                pos[axis] = -limit
                if vel[axis] < 0.0:
                    vel[axis] = 0.0
        new_pos[i] = pos
        new_vel[i] = vel
    return new_pos, new_vel


def sim_targets(tgt_pos, tgt_vel, step_index, next_turn_plan, dt, low, high):
    """复算一步目标运动：转向计划为**预生成**表（不消耗 RNG）。"""
    m = len(tgt_pos)
    out_p = np.empty_like(tgt_pos)
    out_v = np.empty_like(tgt_vel)
    for j in range(m):
        vx, vy = tgt_vel[j]
        if step_index in next_turn_plan[j]:
            vx, vy = next_turn_plan[j][step_index]
        px, vx = sim_reflect(tgt_pos[j, 0], vx, dt, low, high)
        py, vy = sim_reflect(tgt_pos[j, 1], vy, dt, low, high)
        out_p[j] = (px, py)
        out_v[j] = (vx, vy)
    return out_p, out_v


# ----------------------------------------------------------------------
# 匹配（与 scipy maximum_bipartite_matching 等价的小规模暴力回溯）
# ----------------------------------------------------------------------
def matched_count(rob_pos, tgt_pos, radius, i_from=0, i_to=None):
    n = len(rob_pos) if i_to is None else i_to
    m = len(tgt_pos)
    adj = []
    for i in range(i_from, n):
        adj.append([j for j in range(m) if float(np.linalg.norm(rob_pos[i] - tgt_pos[j])) <= radius])

    def dfs(i, used):
        if i == len(adj):
            return 0
        best = dfs(i + 1, used)
        for j in adj[i]:
            if j not in used:
                used.add(j)
                best = max(best, 1 + dfs(i + 1, used))
                used.discard(j)
        return best

    return dfs(0, set())


# ----------------------------------------------------------------------
# 回放：收集真实轨迹
# ----------------------------------------------------------------------
def rollout_and_record(policy_module, case, params=None, policy_seed=1000):
    cfg = case.task_config
    pub = cfg.public
    env = make_training_env(cfg)
    observations, _ = env.reset(seed=case.scenario_seed)
    agent_ids = list(env.agents)
    policies = []
    for i in range(cfg.num_agents):
        pol, _ = build_agent_policy(policy_module, case, i, policy_seed=policy_seed + i, params=params)
        policies.append(pol)

    rec = {
        "rob_pos": [], "rob_vel": [], "tgt_pos": [], "tgt_vel": [],
        "step_index": [], "actions": [], "cov": [], "act_obs": [],
    }
    snap = env.unwrapped._current_snapshot
    rec["rob_pos"].append(np.array(snap.robot_positions))
    rec["rob_vel"].append(np.array(snap.robot_velocities))
    rec["tgt_pos"].append(np.array(snap.target_positions))
    rec["tgt_vel"].append(np.array(snap.target_velocities))
    rec["step_index"].append(int(snap.step_index))

    total_return = 0.0
    for _ in range(cfg.horizon):
        actions = {
            aid: np.asarray(policies[i].act(observations[aid]), dtype=np.float32)
            for i, aid in enumerate(agent_ids)
        }
        rec["actions"].append(np.array([actions[a] for a in agent_ids], dtype=np.float64))
        observations, rewards, _t, _tr, _i = env.step(actions)
        total_return += float(rewards[agent_ids[0]])
        snap = env.unwrapped._current_snapshot
        rec["rob_pos"].append(np.array(snap.robot_positions))
        rec["rob_vel"].append(np.array(snap.robot_velocities))
        rec["tgt_pos"].append(np.array(snap.target_positions))
        rec["tgt_vel"].append(np.array(snap.target_velocities))
        rec["step_index"].append(int(snap.step_index))

    for p in policies:
        p.close()
    env.close()
    rec["return_sum"] = total_return
    rec["mean_j"] = total_return / cfg.horizon
    rec["case_id"] = case.case_id
    rec["group_id"] = case.group_id
    rec["num_agents"] = cfg.num_agents
    rec["num_targets"] = cfg.num_targets
    rec["horizon"] = cfg.horizon
    rec["radius"] = float(pub.target_radius)
    return rec


def verify_simulator(rec):
    """断言复现级仿真器能逐位重放真实轨迹（含碰撞斥力）。"""
    cfg_k = 1.0 - 0.25
    c = 0.1
    rp = np.array(rec["rob_pos"][0])
    rv = np.array(rec["rob_vel"][0])
    max_dp = 0.0
    for t in range(rec["horizon"]):
        rp, rv = sim_robots(rp, rv, rec["actions"][t], cfg_k, c, 0.1, 1.0, 0.05, 1.0)
        max_dp = max(max_dp, float(np.max(np.abs(rp - rec["rob_pos"][t + 1]))))
        max_dp = max(max_dp, float(np.max(np.abs(rv - rec["rob_vel"][t + 1]))))
    return max_dp


def targets_plan_from_record(rec):
    """从真实轨迹提取"完美未来目标轨迹"以及转向时刻表。

    目标运动只依赖自身的 RNG 与步号，与机器人动作**完全解耦**，因此真实轨迹里的
    目标位置序列对任何机器人动作序列都成立 —— 这是 R1 能成立的关键前提。
    """
    n_steps = rec["horizon"] + 1
    plan = []
    for j in range(rec["num_targets"]):
        turns = {}
        for t in range(rec["horizon"]):
            v0 = rec["tgt_vel"][t][j]
            v1 = rec["tgt_vel"][t + 1][j]
            if not np.allclose(v0, v1):
                turns[t] = (float(rec["tgt_vel"][t][j, 0]), float(rec["tgt_vel"][t][j, 1]))
        plan.append(turns)
    future = np.array(rec["tgt_pos"])  # (n_steps, M, 2)
    return future, plan


# ----------------------------------------------------------------------
# R1：离线最优上界（恒定动作枚举，逐机器人独立）
# ----------------------------------------------------------------------
def offline_best_per_robot(rec, n_headings=36, mags=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """**精确**的逐机器人恒定动作最优（委托给 offline_optimum，保持两个工具口径一致）。

    历史注意：本函数最初用"逐步最大匹配 + 坐标上升"求解，那会卡在局部最优，
    得到的只是**下界**（32 场景 4.45），曾被误读成"上界只比规则高 1.4%"。
    精确枚举（含幅度轴）在同样场景上给出 5.97。此处直接复用精确实现，
    避免两个工具给出互相矛盾的数字。
    """
    from tools.offline_optimum import constant_action_optimum

    joint, pick, cmds = constant_action_optimum(rec, n_dirs=n_headings, mags=mags)
    return float(joint), pick, cmds


def realized_matches(rec):
    """规则策略实际拿到的总覆盖对数（Σ_t matched）。"""
    tot = 0
    radius = rec["radius"]
    for t in range(1, rec["horizon"] + 1):
        tot += matched_count(np.array(rec["rob_pos"][t]), np.array(rec["tgt_pos"][t]), radius)
    return tot


def exposure_stats(rec):
    """逐步暴露统计：每步有多少个目标至少被一台机器人"够得到"。

    返回 (Σ_t coverable_pairs, Σ_t 至少对一台机器人 open 的 (步,目标) 对)。
    用来回答："损失是没人去得了，还是去了但没覆盖到？"
    """
    radius = rec["radius"]
    T = rec["horizon"]
    tot_pairs = 0
    tot_open = 0
    for t in range(1, T + 1):
        rp = np.array(rec["rob_pos"][t])
        tp = np.array(rec["tgt_pos"][t])
        d = np.linalg.norm(rp[:, None, :] - tp[None, :, :], axis=-1)
        within = d <= radius
        tot_pairs += int(within.sum())
        tot_open += int(within.any(axis=0).sum())
    return tot_pairs, tot_open


# ----------------------------------------------------------------------
# R2：可达性剥离
# ----------------------------------------------------------------------
def reachability_split(rec, final_only=False, n_headings=36):
    """把 (步, 目标) 对分成"够得着"与"够不着"，并统计规则在够得着部分的取用率。

    够得着的判定用**最宽松**的口径：该机器人在**已完成的真实轨迹**里
    任一时刻曾进入覆盖半径。这样"取用率 < 100%"就只能是策略选择问题，
    因为位置已经证明是到得了的。
    注：这是"事后"判定，只用于归因，不是可达性预测。
    """
    radius = rec["radius"]
    T = rec["horizon"]
    n, m = rec["num_agents"], rec["num_targets"]
    ever = np.zeros((n, m), dtype=bool)
    hit = np.zeros((n, m, T + 1), dtype=bool)
    for t in range(1, T + 1):
        rp = np.array(rec["rob_pos"][t])
        tp = np.array(rec["tgt_pos"][t])
        d = np.linalg.norm(rp[:, None, :] - tp[None, :, :], axis=-1)
        hit[:, :, t] = d <= radius
        ever |= hit[:, :, t]
    return ever, hit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12, help="每组留出场景数")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=555000)
    parser.add_argument("--policy", default="entry")
    parser.add_argument("--headings", type=int, default=36)
    parser.add_argument("--max-scenes", type=int, default=0, help="0=全部")
    parser.add_argument("--dump", type=Path, default=None, help="逐场景 CSV 输出路径")
    args = parser.parse_args()

    cases = make_heldout_cases(args.seed_offset, args.n, args.repeats)
    if args.max_scenes:
        cases = cases[: args.max_scenes]
    module = POLICIES[args.policy]
    print(f"场景数 {len(cases)}  策略={args.policy}  恒定动作候选方向={args.headings}")

    sim_err = 0.0
    rows = []
    for idx, case in enumerate(cases):
        rec = rollout_and_record(module, case)
        sim_err = max(sim_err, verify_simulator(rec))

        real = realized_matches(rec)
        best, choice, cands = offline_best_per_robot(rec, n_headings=args.headings)
        ever, hit = reachability_split(rec)
        n_pairs, n_open = exposure_stats(rec)

        n, m, T = rec["num_agents"], rec["num_targets"], rec["horizon"]
        # 有多少 (机,目标) 对整场都够不着
        unreachable_pairs = int((~ever).sum())
        # 规则实际完成的对数（按"是否曾覆盖"计，粗略）
        done_pairs = int(ever.sum())
        rows.append(
            dict(
                case_id=rec["case_id"], group=rec["group_id"], real=real,
                ub=best, T=T, n=n, m=m,
                unreachable_pairs=unreachable_pairs, done_pairs=done_pairs,
                total_pairs=n * m, mean_j=rec["mean_j"],
                within_pairs=n_pairs, open_targets=n_open,
            )
        )
        if idx < 5 or (idx + 1) % 25 == 0:
            print(
                f"  [{idx+1:>3d}/{len(cases)}] {rec['case_id']:>16s} "
                f"matched={real:>3d}  恒定动作上界={best:>3.0f}  "
                f"够不着(机,目标)对={unreachable_pairs:>2d}/{n*m}  J={rec['mean_j']:+.4f}"
            )

    print("-" * 78)
    real = np.array([r["real"] for r in rows], dtype=np.float64)
    ub = np.array([r["ub"] for r in rows], dtype=np.float64)
    print(f"仿真器重放最大偏差 = {sim_err:.3e}  (必须 < 1e-9)")
    print(f"规则策略  总覆盖对数 均值 = {real.mean():.4f} / 场景")
    print(f"恒定动作上界 均值         = {ub.mean():.4f} / 场景")
    print(f"上界/规则 比              = {ub.mean() / max(real.mean(), 1e-9):.4f}")
    print(f"平均每个(机,目标)对够不着 = {np.mean([r['unreachable_pairs'] for r in rows]):.2f} / {rows[0]['total_pairs']}")
    w = np.array([r["within_pairs"] for r in rows], dtype=np.float64)
    op = np.array([r["open_targets"] for r in rows], dtype=np.float64)
    print(f"平均 (步,机,目标) 三元组落在覆盖半径内 = {w.mean():.3f} / 场景 (理论上限 {rows[0]['T']*rows[0]['total_pairs']})")
    print(f"平均 (步,目标) 至少被一台机器人覆盖到 = {op.mean():.3f} / 场景 (即'可同时被匹配'的目标数) ")
    print(f"平均 J                    = {np.mean([r['mean_j'] for r in rows]):+.4f}")
    print(f"上界对应 J                = {ub.sum() / (len(rows) * rows[0]['T']):+.4f}")
    gains = ub - real
    print(f"有正增益的场景数          = {int((gains > 0.5).sum())} / {len(rows)}")
    print(f"最大单场景增益            = {gains.max():.1f} 对")

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        keys = ["case_id", "group", "real", "ub", "T", "n", "m",
                "unreachable_pairs", "done_pairs", "total_pairs", "mean_j"]
        lines = [",".join(keys)]
        for r in rows:
            lines.append(",".join(str(r[k]) for k in keys))
        args.dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"逐场景结果已写入 {args.dump}")


if __name__ == "__main__":
    main()
