"""本地回合回放工具（P514，仅开发期使用，不参与推理）。

用官方环境按公开套件的 scenario_seed 复现 4 个场景，逐步驱动任意策略，
输出与评测器同口径的逐回合指标，并可直接汇总成 performance_score。

用法：
    .venv/Scripts/python.exe participant/P514/tools/rollout.py --policy rule
    .venv/Scripts/python.exe participant/P514/tools/rollout.py --policy entry --repeat 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_P514 = _REPO_ROOT / "participant" / "P514"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_P514) not in sys.path:
    sys.path.insert(0, str(_P514))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.metrics import compute_step_metrics  # noqa: E402
from coverage_bench.protocol import (  # noqa: E402
    BuildContext,
    EpisodeContext,
    ResourceLimits,
    get_protocol_spec,
)
from coverage_bench.runtime import _public_task_params  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402

import entry  # noqa: E402  P514 根目录下，与正式评测生命周期一致
from policies import probe, random_ref, rule  # noqa: E402

PUBLIC_SUITE = _REPO_ROOT / "configs" / "public-suite-v1.yaml"

# 策略名 → 模块。使用包内静态导入而非按路径动态载入：
# 官方静态审计（coverage_bench/audit.py）把 importlib 动态载入列为硬拒绝项，
# 因此提交目录内不允许出现 spec_from_file_location 之类的调用。
# "entry" 是**正式提交入口**，走 BuildContext→reset(EpisodeContext) 完整生命周期，
# 结论应以它为准；rule 是同一实现但不带 SUBMISSION_OVERRIDES 的直接构建路径。
POLICIES = {
    "entry": entry,
    "rule": rule,
    "random_ref": random_ref,
    "probe": probe,
}


def make_episode_context(case, agent_index, policy_seed):
    cfg = case.task_config
    return EpisodeContext(
        agent_index=agent_index,
        num_agents=cfg.num_agents,
        num_targets=cfg.num_targets,
        horizon=cfg.horizon,
        task=_public_task_params(case),
        policy_seed=policy_seed,
    )


DEFAULT_ARTIFACT_DIR = _P514 / "artifacts"


def build_agent_policy(policy_module, case, agent_index, policy_seed, params=None, artifact_dir=None):
    art = Path(artifact_dir) if artifact_dir is not None else DEFAULT_ARTIFACT_DIR
    ctx = make_episode_context(case, agent_index, policy_seed)
    if hasattr(policy_module, "build_policy_for_agent"):
        policy = policy_module.build_policy_for_agent(ctx, art, params)
    else:  # entry.py 风格：build_policy(BuildContext)
        from coverage_bench.protocol import BuildContext, ResourceLimits, get_protocol_spec

        build_ctx = BuildContext(
            spec=get_protocol_spec(),
            artifact_dir=art,
            device="cpu",
            limits=ResourceLimits(),
            rng=np.random.default_rng(policy_seed),
        )
        policy = policy_module.build_policy(build_ctx)
    policy.reset(ctx)
    return policy, ctx


def rollout_case(policy_module, case, repeat_index=0, verbose=False, params=None):
    cfg = case.task_config
    env = make_training_env(cfg)
    observations, _ = env.reset(seed=case.scenario_seed)
    agent_ids = list(env.agents)

    policies = []
    for i in range(cfg.num_agents):
        policy, _ctx = build_agent_policy(
            policy_module, case, i, policy_seed=1595635815038561411 + i, params=params
        )
        policies.append(policy)

    total_return = 0.0
    coverages, collisions, full_covs = [], [], []
    trace = []
    for step in range(cfg.horizon):
        actions = {
            agent_id: np.asarray(policies[i].act(observations[agent_id]), dtype=np.float32)
            for i, agent_id in enumerate(agent_ids)
        }
        observations, rewards, _t, _tr, _i = env.step(actions)
        total_return += float(rewards[agent_ids[0]])
        m = compute_step_metrics(env.unwrapped._current_snapshot)
        coverages.append(m.coverage_rate)
        collisions.append(m.collision_rate)
        full_covs.append(1.0 if m.full_coverage else 0.0)
        if verbose:
            snap = env.unwrapped._current_snapshot
            trace.append(
                (
                    step,
                    m.matched_targets,
                    round(m.coverage_rate, 3),
                    round(m.collision_rate, 3),
                    np.round(snap.robot_positions, 3).tolist(),
                    np.round(snap.target_positions, 3).tolist(),
                )
            )
    for p in policies:
        p.close()
    env.close()

    result = {
        "case_id": case.case_id,
        "group_id": case.group_id,
        "repeat_index": repeat_index,
        "return_sum": total_return,
        "mean_j": total_return / cfg.horizon,
        "mean_coverage_rate": float(np.mean(coverages)),
        "mean_collision_rate": float(np.mean(collisions)),
        "full_coverage_fraction": float(np.mean(full_covs)),
    }
    return result, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="rule", help="rule | entry | 任意 policies/<name>.py")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--case", default=None)
    args = parser.parse_args()

    if args.policy not in POLICIES:
        raise SystemExit(f"未知策略: {args.policy}（可用: {', '.join(POLICIES)}）")
    module = POLICIES[args.policy]

    suite = load_suite(PUBLIC_SUITE)
    rows = []
    for group in suite.groups:
        for case in group.cases:
            if args.case and case.case_id != args.case:
                continue
            for rep in range(args.repeat):
                result, trace = rollout_case(module, case, rep, args.verbose)
                rows.append(result)
                print(
                    f"{result['group_id']:>12s} {result['case_id']:>8s} rep{rep} "
                    f"R={result['return_sum']:+.4f} J={result['mean_j']:+.4f} "
                    f"cov={result['mean_coverage_rate']:.3f} col={result['mean_collision_rate']:.3f} "
                    f"full={result['full_coverage_fraction']:.3f}"
                )
                if args.verbose:
                    for t in trace:
                        print(f"   step{t[0]} matched={t[1]} cov={t[2]} col={t[3]}")
                        print(f"      robots {t[4]}")
                        print(f"      targets {t[5]}")

    group_means: dict[str, list[float]] = {}
    for r in rows:
        group_means.setdefault(r["group_id"], []).append(r["mean_j"])
    print("-" * 78)
    score = 0.0
    for gid, vals in group_means.items():
        score += 1000.0 * 0.5 * float(np.mean(vals))
        print(f"group {gid:>12s} mean_j={np.mean(vals):+.4f} (n={len(vals)})")
    print(f"performance_score = {score:.4f}")


if __name__ == "__main__":
    main()
