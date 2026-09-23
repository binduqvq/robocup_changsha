"""P514 真·离线最优上界（束搜索，开发期使用，不参与推理）。

与 `diagnose.py` 的"恒定动作上界"相比，本工具允许**逐机器人、逐时刻改变动作**，
在完美未来目标轨迹（由真实回放导出，见 diagnose 的说明）下对联合动作序列做束搜索。
得到的是本题在给定初始场景与目标轨迹下 **开环全信息最优的近似上界**，因此比"恒定动作"强。

用途：把 T=10/3v3 这一格的剩余空间彻底定死 —— 如果束搜索最优也只是略高于规则策略，
就能断定"正式计分设定下已无可用空间"，并把迭代重心转向泛化轴（T / N / M）。

用法：
    .venv/Scripts/python.exe participant/P514/tools/offline_optimum.py --n 8 --repeats 2 --width 16
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_P514 = _REPO_ROOT / "participant" / "P514"
for _p in (str(_REPO_ROOT), str(_P514)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.diagnose import (  # noqa: E402
    realized_matches,
    rollout_and_record,
    sim_robots,
    verify_simulator,
)
from tools.rollout import POLICIES  # noqa: E402
from tools.sweep import make_heldout_cases  # noqa: E402


def matched_fast(rob_pos, tgt_pos, radius):
    """小规模最大匹配的位掩码实现（N=M=3，比递归+set 快一个量级）。

    与 coverage_bench.metrics 的 scipy maximum_bipartite_matching 在 N=M=3 上等价；
    当匹配数为 3 时，König 定理保证存在覆盖全部 3 台机器人的匹配，故省略该分支。
    """
    m = len(tgt_pos)
    adj = []
    for p in rob_pos:
        mask = 0
        for j in range(m):
            dx = p[0] - tgt_pos[j][0]
            dy = p[1] - tgt_pos[j][1]
            if dx * dx + dy * dy <= radius * radius:
                mask |= 1 << j
        adj.append(mask)
    n = len(adj)

    def go(i, used):
        if i == n:
            return 0
        best = go(i + 1, used)
        avail = adj[i] & ~used
        while avail:
            b = avail & -avail
            v = 1 + go(i + 1, used | b)
            if v > best:
                best = v
            avail ^= b
        return best

    return go(0, 0)


def make_command_set(n_dirs: int = 12, mags=(0.0, 0.5, 1.0)):
    """候选动作集合：含零动作与多个幅度/方向（连续动作空间的离散近似）。"""
    cmds = [np.zeros(2, dtype=np.float64)]
    dirs = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    for mag in mags:
        if mag == 0.0:
            continue
        for th in dirs:
            cmds.append(mag * np.array([np.cos(th), np.sin(th)], dtype=np.float64))
    return np.array(cmds, dtype=np.float64)


def sim_step_vec(pos, vel, acts, k, c, dt, radius, half_extent):
    """**向量化**复算一步机器人动力学，与 diagnose.sim_robots 逐位等价。

    关键：官方 `advance_robots` 是"位置先于速度"的欧拉积分，
    位置用的是 `vel_old + 本步速度增量`（本仓库 physics.py 第 100 行），即
        pos' = pos + (vel_old + (drive/mass)·dt·u) · dt
        vel' = (1-damping)·vel_old + (drive/mass)·dt·u
    因此第 1 步**就有**位移（初速为 0 时 pos 位移 = c·u·dt）。若误写成
    `pos + vel_old·dt`，第一步位移恒为 0，整个规划器会从根上失效。

    机器人间弹性接触斥力在本任务参数下只在间距 < 0.1 时显著；开环规划忽略它
    会**高估**可用位移（推挤只会减少位移），对"上界"用途是安全方向。

    参数形状：(K, n, 2)。返回同形状的 (new_pos, new_vel)。
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    acts = np.asarray(acts, dtype=np.float64)

    K, n, _ = pos.shape
    new_pos = np.empty((K, n, 2), dtype=np.float64)
    new_vel = np.empty((K, n, 2), dtype=np.float64)
    for i in range(n):
        pi = pos[:, i, :]
        vi = vel[:, i, :]
        ui = acts[:, i, :]
        acc = ui * dt                                     # drive_force/mass=1
        nvi = k * vi + acc                                # v' = (1-damping)·v + 力·dt
        speed = np.linalg.norm(nvi, axis=-1, keepdims=True)
        over = speed > 1.0
        nvi = np.where(over, nvi * (1.0 / np.maximum(speed, 1e-12)), nvi)
        npi = pi + (vi + acc) * dt
        limit = half_extent - radius
        hi = npi > limit
        lo = npi < -limit
        npi = np.clip(npi, -limit, limit)
        nvi = np.where(hi & (nvi > 0.0), 0.0, nvi)
        nvi = np.where(lo & (nvi < 0.0), 0.0, nvi)
        new_pos[:, i, :] = npi
        new_vel[:, i, :] = nvi
    return new_pos, new_vel


def matched_vec(rob_pos, tgt_pos, radius):
    """对 (K, n, 2) 的机器人位置批量算最大匹配数，返回 (K,) 整数数组。

    n ≤ 3 时用位掩码枚举（最多 2^3 种分配），全部 K 个候选一起算。
    """
    K, n, _ = rob_pos.shape
    m = len(tgt_pos)
    diff = rob_pos[:, :, None, :] - tgt_pos[None, None, :, :]
    d2 = (diff * diff).sum(axis=-1)                   # (K, n, m)
    adj = d2 <= radius * radius
    masks = np.zeros((K, n), dtype=np.int64)
    for j in range(m):
        masks |= (adj[:, :, j].astype(np.int64) << j)

    best = np.zeros(K, dtype=np.int64)
    full = (1 << m) - 1

    def rec(i, used, count):
        if i == n:
            return count
        # 不分配
        b = rec(i + 1, used, count)
        av = masks[:, i] & ~used
        # 逐个可分配目标（n ≤ 3，最多 3 支）
        for j in range(m):
            sel = (av >> j) & 1
            if sel.any():
                sub = rec(i + 1, np.where(sel.astype(bool), used | (1 << j), used), count + 1)
                b = np.maximum(b, np.where(sel.astype(bool), sub, 0))
        return b

    res = rec(0, np.zeros(K, dtype=np.int64), np.zeros(K, dtype=np.int64))
    _ = full
    return np.minimum(res, m)


def constant_action_optimum(rec, n_dirs: int = 36, mags=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """恒定动作最优（逐机器人独立可分解，因此本函数是**精确**的，不是采样近似）。

    对每台机器人独立枚举"整场恒定动作"，取最大化自身覆盖时刻数的那个；
    再用真实的最大匹配把各机的独立最优收紧为联合下界（同时给出上界）。
    返回 (联合匹配下界, Σ_i 单机最优覆盖时刻数之和) 两个尺度：
      - 单机最优之和是**不可达的乐观值**（同一目标会被多机重复计数），
        只用来说明"总共有多少机会"。
      - 联合匹配值是把这些动作真正放在一起跑之后测到的匹配数，是可比较的量。
    """
    k, c, dt = 0.75, 0.1, 0.1
    radius = rec["radius"]
    n = rec["num_agents"]
    T = rec["horizon"]
    future = np.array(rec["tgt_pos"])
    cmds = make_command_set(n_dirs, mags)

    # 逐机器人：每个候选动作的整场轨迹
    best_cmd = []
    for i in range(n):
        p = np.array(rec["rob_pos"][0][i])[None, None, :]
        v = np.array(rec["rob_vel"][0][i])[None, None, :]
        acts = cmds[:, None, :]
        pos = np.broadcast_to(p, (len(cmds), 1, 2)).copy()
        vel = np.broadcast_to(v, (len(cmds), 1, 2)).copy()
        traj = np.empty((len(cmds), T + 1, 1, 2))
        traj[:, 0] = pos
        for t in range(1, T + 1):
            pos, vel = sim_step_vec(pos, vel, acts, k, c, dt, 0.05, 1.0)
            traj[:, t] = pos
        # 每个候选覆盖的时刻数
        posj = traj[:, :, 0, :]                       # (C, T+1, 2)
        d = np.linalg.norm(posj[:, :, None, :] - future[None, :, :, :], axis=-1)
        cover = (d <= radius)[:, 1:, :]               # (C, T, M)
        best_cmd.append(int(np.argmax(cover.reshape(len(cmds), -1).sum(axis=1))))

    # 联合评估：所有机器人执行各自的恒定动作
    pos = np.array(rec["rob_pos"][0], dtype=np.float64)[None]
    vel = np.array(rec["rob_vel"][0], dtype=np.float64)[None]
    acts = np.array([cmds[best_cmd]], dtype=np.float64)   # (1, n, 2)
    joint = 0
    for t in range(1, T + 1):
        pos, vel = sim_step_vec(np.broadcast_to(pos, (1, n, 2)).copy(),
                                np.broadcast_to(vel, (1, n, 2)).copy(),
                                acts, k, c, dt, 0.05, 1.0)
        joint += int(matched_vec(pos, future[t], radius)[0])
    return float(joint), best_cmd, cmds


def beam_optimum(rec, width: int = 16, n_dirs: int = 12, mags=(0.0, 0.5, 1.0), return_hist: bool = False):
    """束搜索求开环最优，返回 (最优 Σ_t matched, 规则实际 Σ_t matched)。

    同时算"恒定动作最优"作为对照：束搜索的空间包含恒定动作，因此
    beam 结果应当 ≥ 恒定动作结果；若小于，说明束宽不足或离散过粗（工具自身的限）。
    """
    k, c, dt = 0.75, 0.1, 0.1
    n = rec["num_agents"]
    m = rec["num_targets"]
    T = rec["horizon"]
    radius = rec["radius"]
    half = 1.0
    future = np.array(rec["tgt_pos"])  # (T+1, M, 2)

    cmds = make_command_set(n_dirs, mags)      # (C, 2)
    n_c = len(cmds)
    joints = np.array(list(itertools.product(range(n_c), repeat=n)), dtype=np.int64)  # (J, n)
    joint_acts = cmds[joints]                                                          # (J, n, 2)

    pos0 = np.array(rec["rob_pos"][0], dtype=np.float64)
    vel0 = np.array(rec["rob_vel"][0], dtype=np.float64)

    def step_all(pos, vel):
        """pos/vel: (n, 2) → (J, n, 2)。"""
        p = np.broadcast_to(pos[None, :, :], (len(joints), n, 2))
        v = np.broadcast_to(vel[None, :, :], (len(joints), n, 2))
        return sim_step_vec(p, v, joint_acts, k, c, dt, 0.05, half)

    np1, nv1 = step_all(pos0, vel0)
    s1 = matched_vec(np1, future[1], radius)
    keep = np.argsort(-s1)[: max(width, 1)]
    nodes = [(np1[ji], nv1[ji], float(s1[ji]) / m) for ji in keep]

    for t in range(2, T + 1):
        all_p, all_v, all_s = [], [], []
        base = []
        for pos, vel, acc in nodes:
            np_, nv_ = step_all(pos, vel)
            all_p.append(np_)
            all_v.append(nv_)
            base.append(acc)
        # 逐个节点打分（K=J，节点数 = width）
        for bi in range(len(nodes)):
            s_ = matched_vec(all_p[bi], future[t], radius)
            all_s.append(s_)
        cands = []
        for bi in range(len(nodes)):
            for ji in range(len(joints)):
                cands.append((all_p[bi][ji], all_v[bi][ji], base[bi] + float(all_s[bi][ji]) / m))
        cands.sort(key=lambda x: -x[2])
        nodes = cands[: max(width, 1)]

    best = max(x[2] for x in nodes) * m if nodes else 0.0
    const_joint, const_cmds, _cmds = constant_action_optimum(rec)
    best = max(best, const_joint)
    if return_hist:
        return float(best), float(realized_matches(rec)), float(const_joint)
    return float(best), float(realized_matches(rec))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="每组留出场景数")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=555000)
    parser.add_argument("--policy", default="entry")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dirs", type=int, default=12)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args()

    cases = make_heldout_cases(args.seed_offset, args.n, args.repeats)
    if args.max_scenes:
        cases = cases[: args.max_scenes]
    module = POLICIES[args.policy]
    n_cmd = args.dirs * 2 + 1
    print(
        f"场景数 {len(cases)}  束宽={args.width}  候选动作数/机={n_cmd}  "
        f"联合动作数/步={n_cmd ** 3}"
    )

    sim_err = 0.0
    rows = []
    for idx, case in enumerate(cases):
        rec = rollout_and_record(module, case)
        sim_err = max(sim_err, verify_simulator(rec))
        opt, real, const = beam_optimum(
            rec, width=args.width, n_dirs=args.dirs, return_hist=True
        )
        beam_only = opt if opt > const + 1e-9 else const
        rows.append(dict(case_id=rec["case_id"], opt=opt, real=real, const=const,
                         T=rec["horizon"], m=rec["num_targets"]))
        print(
            f"  [{idx+1:>3d}/{len(cases)}] {rec['case_id']:>16s} "
            f"束搜索={beam_only:>5.1f}  恒定动作={const:>5.1f}  规则={real:>5.1f}  "
            f"差={max(opt, const)-real:+.1f}"
        )

    opt = np.array([r["opt"] for r in rows])
    const = np.array([r["const"] for r in rows])
    real = np.array([r["real"] for r in rows])
    print("-" * 78)
    print(f"仿真器重放最大偏差 = {sim_err:.3e}  (必须 < 1e-9)")
    print(f"束搜索最优 总匹配对数/场景 = {opt.mean():.4f}  (含恒定动作对照)")
    print(f"恒定动作最优 总匹配对数/场景 = {const.mean():.4f}")
    print(f"规则策略   总匹配对数/场景 = {real.mean():.4f}")
    print(f"最优/规则 = {opt.mean() / max(real.mean(), 1e-9):.4f}")
    print(f"上界对应 J = {opt.sum() / (len(rows) * rows[0]['T']):+.4f}   "
          f"规则 J = {real.sum() / (len(rows) * rows[0]['T']):+.4f}")
    print(f"有正增益场景数 = {int((opt - real > 0.5).sum())} / {len(rows)}")
    print(f"束搜索超过恒定动作的场景数 = {int((opt - const > 0.5).sum())} / {len(rows)}"
          f"  （若为 0，说明束搜索没有任何增益，上界实际由恒定动作给出）")

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        lines = ["case_id,opt,real,gap"]
        for r in rows:
            lines.append(f"{r['case_id']},{r['opt']:.1f},{r['real']:.1f},{r['opt']-r['real']:.1f}")
        args.dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"逐场景结果已写入 {args.dump}")


if __name__ == "__main__":
    main()
