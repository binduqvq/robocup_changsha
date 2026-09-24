"""P514 策略协议自检：在多种边界条件下验证动作合法性与状态隔离。

为什么需要它：策略一旦返回非法动作（形状/类型/非有限/越界）或跨回合残留状态，
评测器会记为 `protocol_error` / `runtime_error` 并让**整份提交拿不到分数**。
官方预检只做静态审计与结构校验，不执行策略，因此这里主动把"运行期契约"
在本地逐条打成断言，作为交付前的最后一道自检。

用法：
    .venv/Scripts/python.exe participant/P514/tools/selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_P514 = _REPO_ROOT / "participant" / "P514"
for _p in (str(_REPO_ROOT), str(_P514)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.protocol import (  # noqa: E402
    BuildContext,
    EpisodeContext,
    ResourceLimits,
    get_protocol_spec,
)
from coverage_bench.runtime import _public_task_params  # noqa: E402
from coverage_bench.suites import ScenarioCase  # noqa: E402
from coverage_bench.validation import validate_action  # noqa: E402

import entry  # noqa: E402
from policies import rule  # noqa: E402
from tools import generalize as g  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_ctx(case, i, seed=7):
    cfg = case.task_config
    return EpisodeContext(
        agent_index=i,
        num_agents=cfg.num_agents,
        num_targets=cfg.num_targets,
        horizon=cfg.horizon,
        task=_public_task_params(case),
        policy_seed=seed + i,
    )


def policy_for(case, i, seed=7):
    ctx = make_ctx(case, i, seed)
    bctx = BuildContext(
        spec=get_protocol_spec(),
        artifact_dir=_P514 / "artifacts",
        device="cpu",
        limits=ResourceLimits(),
        rng=np.random.default_rng(seed + i),
    )
    pol = entry.build_policy(bctx)
    pol.reset(ctx)
    return pol, ctx


def run_case_actions(case, seed=7):
    """跑一个完整回合，返回每个动作与动作校验异常。"""
    cfg = case.task_config
    env = make_training_env(cfg)
    obs, _ = env.reset(seed=case.scenario_seed)
    ids = list(env.agents)
    pols = [policy_for(case, i, seed)[0] for i in range(cfg.num_agents)]
    acts = []
    try:
        for _ in range(cfg.horizon):
            d = {}
            for i, aid in enumerate(ids):
                a = pols[i].act(obs[aid])
                validate_action(a)
                acts.append(np.asarray(a))
                d[aid] = a
            obs, _r, _t, _tr, _i = env.step(d)
    finally:
        for p in pols:
            p.close()
        env.close()
    return acts


def main():
    base = g._base_config()

    print("=== 1. 动作协议：形状/类型/有限性/取值范围 ===")
    for label, (n, m, t, lay) in {
        "3v3 T=10 uniform": (3, 3, 10, "uniform"),
        "3v3 T=10 crossing": (3, 3, 10, "crossing"),
        "3v3 T=10 clustered": (3, 3, 10, "clustered"),
        "1v1 T=10": (1, 1, 10, "uniform"),
        "8v8 T=10": (8, 8, 10, "uniform"),
        "2v5 T=3": (2, 5, 3, "uniform"),
        "3v3 T=1": (3, 3, 1, "uniform"),
        "3v3 T=50": (3, 3, 50, "uniform"),
        "3v3 T=256": (3, 3, 256, "uniform"),
    }.items():
        case = ScenarioCase("st", "st", g.make_config(base, n, m, t, lay), 12345)
        acts = run_case_actions(case)
        n_expected = n * t
        ok_len = len(acts) == n_expected
        shapes = {a.shape for a in acts}
        dtypes = {a.dtype for a in acts}
        finite = all(np.all(np.isfinite(a)) for a in acts)
        inrange = all(np.all(a >= -1.0) and np.all(a <= 1.0) for a in acts)
        check(
            f"{label}: 动作数/形状/类型/有限/范围",
            ok_len and shapes == {(2,)} and dtypes == {np.dtype("float32")} and finite and inrange,
            f"n={len(acts)}/{n_expected} shapes={shapes} dtypes={dtypes} finite={finite} inrange={inrange}",
        )

    print("=== 2. 未 reset 直接 act 必须仍返回合法动作（兜底） ===")
    bctx = BuildContext(
        spec=get_protocol_spec(),
        artifact_dir=_P514 / "artifacts",
        device="cpu",
        limits=ResourceLimits(),
        rng=np.random.default_rng(0),
    )
    p = entry.build_policy(bctx)
    case = ScenarioCase("st", "st", g.make_config(base, 3, 3, 10, "uniform"), 999)
    env = make_training_env(case.task_config)
    obs, _ = env.reset(seed=999)
    aid = env.agents[0]
    try:
        a = p.act(obs[aid])
        validate_action(a)
        check("未 reset 的兜底动作合法", True, f"shape={np.asarray(a).shape}")
    except Exception as exc:  # noqa: BLE001
        check("未 reset 的兜底动作合法", False, repr(exc))
    finally:
        env.close()

    print("=== 3. 回合状态隔离：reset 后不得残留上一回合状态 ===")
    case = ScenarioCase("st", "st", g.make_config(base, 3, 3, 10, "uniform"), 555)
    pol, ctx = policy_for(case, 0)
    tracker = pol._tracker
    # 人为污染内部状态
    tracker.have_seen[:] = True
    tracker.last_seen[:] = 9.9
    tracker._obs_step[:] = 3
    tracker._vel[:] = 1.0
    tracker._last_goal_target = 2
    pol.reset(ctx)
    t2 = pol._tracker
    clean = (
        not t2.have_seen.any()
        and not np.any(t2.last_seen != 0.0)
        and np.all(t2._obs_step == -1)
        and not np.any(t2._vel != 0.0)
        and t2._last_goal_target == -1
    )
    check("reset 清空目标观测、速度与最后追踪编号", clean)

    # 第五轮新增的探索状态同属回合内记忆，必须一并清空（规程 §14）
    pol2, ctx2 = policy_for(case, 0)
    tr2 = pol2._tracker
    tr2._visited.update({(1, 1), (2, 2), (-3, 4)})
    tr2._explore_dir[:] = np.array([0.5, -0.5])
    pol2.reset(ctx2)
    t3 = pol2._tracker
    check(
        "reset 清空 _visited/_explore_dir（探索状态）",
        len(t3._visited) == 0 and not np.any(t3._explore_dir != 0.0),
        f"visited={len(t3._visited)} dir={t3._explore_dir}",
    )

    print("=== 4. 每个 agent 实例互不共享可变状态 ===")
    p0, _ = policy_for(case, 0)
    p1, _ = policy_for(case, 1)
    check("两个实例的追踪器是不同对象", p0._tracker is not p1._tracker)
    check("两个实例的缓冲数组不共享内存",
          not np.shares_memory(p0._tracker.last_seen, p1._tracker.last_seen))

    print("=== 5. 确定性：同一观测重复调用结果一致 ===")
    pol, _ = policy_for(case, 0)
    env = make_training_env(case.task_config)
    obs, _ = env.reset(seed=555)
    aid = env.agents[0]
    try:
        o = obs[aid]
        a1 = np.asarray(pol.act(o))
        a2 = np.asarray(pol.act(o))
        check("重复调用同观测动作一致", np.array_equal(a1, a2), f"{a1} vs {a2}")
    finally:
        env.close()

    print("=== 6. 极小/退化配置不崩 ===")
    for label, (n, m, t) in {"1v1 T=1": (1, 1, 1), "1v8 T=2": (1, 8, 2), "8v1 T=2": (8, 1, 2)}.items():
        try:
            case = ScenarioCase("st", "st", g.make_config(base, n, m, t, "uniform"), 31337)
            acts = run_case_actions(case)
            ok = len(acts) == n * t and all(np.all(np.isfinite(a)) for a in acts)
            check(f"{label} 正常完成", ok, f"actions={len(acts)}")
        except Exception as exc:  # noqa: BLE001
            check(f"{label} 正常完成", False, repr(exc))

    print("=== 7. 目标速度外推的单位一致性 ===")
    unit_case = ScenarioCase("unit", "unit", g.make_config(base, 3, 3, 10, "uniform"), 2026)
    pol, _ = policy_for(unit_case, 0)
    tr = pol._tracker
    tr.have_seen[0] = True
    tr._obs_pos_abs[0] = np.array([0.2, -0.1])
    tr._obs_step[0] = 2
    tr._vel[0] = np.array([0.2, -0.1])  # 场地尺度/秒
    pred = tr._predicted_rel(0, at_step=5, self_pos=np.array([0.05, 0.05]))
    expected = np.array([0.21, -0.18])  # 三步 = 0.3 秒
    check(
        "速度乘 elapsed_steps*dt 后再外推",
        np.allclose(pred, expected, atol=1e-12),
        f"pred={pred} expected={expected}",
    )

    print("=== 8. 盒约束控制与可达几何 ===")
    obs_box = {
        "self_state": np.array([0.0, 0.0, 0.0, 0.0, 0.05], dtype=np.float32),
        "targets": np.array([[0.4, 0.2, 0.15]], dtype=np.float32),
        "target_visible": np.array([True]),
        "peers": np.zeros((1, 5), dtype=np.float32),
        "peer_visible": np.array([False]),
        "step_index": np.int64(0),
    }
    box_tracker = rule.AnalyticTracker(rule.RuleParams(
        num_agents=1, num_targets=1, horizon=10, action_norm="box",
        action_box_scale=0.0, reach_box=1,
    ))
    box_action = box_tracker.act(obs_box)
    check(
        "短回合逐轴饱和会同时用满两轴",
        np.array_equal(box_action, np.array([1.0, 1.0], dtype=np.float32)),
        f"action={box_action}",
    )
    r10 = box_tracker._reachable(10)
    corner_residual = np.array([r10 + 0.1, r10 + 0.1])
    check(
        "方盒可达集与覆盖圆的相交判定",
        box_tracker._can_cover(corner_residual, 10),
        f"R10={r10:.6f} residual={corner_residual}",
    )
    adaptive_tracker = rule.AnalyticTracker(rule.RuleParams(
        num_agents=1, num_targets=1, horizon=50, action_norm="adaptive",
        action_box_scale=0.0, box_horizon_max=10, reach_box=1,
    ))
    adaptive_action = adaptive_tracker.act(obs_box)
    check(
        "adaptive 在长回合回退为方向保持 L∞ 控制",
        np.allclose(adaptive_action, np.array([1.0, 0.5], dtype=np.float32)),
        f"action={adaptive_action}",
    )

    print("=== 9. 搜索模式（bounce / unvisited）在长回合下动作合法 ===")
    for mode in ("index", "bounce", "unvisited"):
        for (n, m, t) in ((3, 3, 10), (3, 3, 60), (1, 3, 30), (5, 5, 30)):
            try:
                cfg = g.make_config(base, n, m, t, "uniform")
                c = ScenarioCase("st", "st", cfg, 4242)
                env = make_training_env(cfg)
                obs, _ = env.reset(seed=4242)
                ids = list(env.agents)
                pols = []
                for i in range(n):
                    ctx = make_ctx(c, i)
                    p = rule.build_policy_for_agent(ctx, None, {"search_mode": mode})
                    p.reset(ctx)
                    pols.append(p)
                bad = 0
                for _ in range(t):
                    d = {}
                    for i, aid in enumerate(ids):
                        a = pols[i].act(obs[aid])
                        try:
                            validate_action(a)
                        except Exception:  # noqa: BLE001
                            bad += 1
                        d[aid] = a
                    obs, _r, _tt, _tr, _i = env.step(d)
                check(
                    f"{mode} {n}v{m} T={t}: 全部动作合法",
                    bad == 0,
                    f"非法动作数={bad}",
                )
            except Exception as exc:  # noqa: BLE001
                check(f"{mode} {n}v{m} T={t}: 全部动作合法", False, repr(exc))
            finally:
                try:
                    for q in pols:
                        q.close()
                    env.close()
                except Exception:  # noqa: BLE001
                    pass

    print()
    if FAILURES:
        print(f"自检失败 {len(FAILURES)} 项: {FAILURES}")
        raise SystemExit(1)
    print("全部自检通过")


if __name__ == "__main__":
    main()
