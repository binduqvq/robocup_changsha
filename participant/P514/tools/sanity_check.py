"""数值核查：随机动作与规则策略在 3v3/T=10 下的覆盖率与碰撞率量级校核。

目的：在得出结论前先确认测量本身没有量级错误（例如碰撞惩罚主导导致 J 为负），
以及规则策略的覆盖率是否与"可达性几何"的粗略估计一致。

用法：
    .venv/Scripts/python.exe participant/P514/tools/sanity_check.py
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
from coverage_bench.metrics import compute_step_metrics  # noqa: E402
from coverage_bench.protocol import EpisodeContext  # noqa: E402
from coverage_bench.runtime import _public_task_params  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402

from policies import random_ref, rule  # noqa: E402

PUBLIC_SUITE = _REPO_ROOT / "configs" / "public-suite-v1.yaml"


def rollout(policy_module, case, seed, params=None):
    cfg = case.task_config
    env = make_training_env(cfg)
    observations, _ = env.reset(seed=seed)
    agent_ids = list(env.agents)
    policies = []
    for i in range(cfg.num_agents):
        ctx = EpisodeContext(
            agent_index=i,
            num_agents=cfg.num_agents,
            num_targets=cfg.num_targets,
            horizon=cfg.horizon,
            task=_public_task_params(case),
            policy_seed=seed + i,
        )
        if hasattr(policy_module, "build_policy_for_agent"):
            p = policy_module.build_policy_for_agent(ctx, _P514 / "artifacts", params)
        else:
            from coverage_bench.protocol import BuildContext, ResourceLimits, get_protocol_spec

            bctx = BuildContext(
                spec=get_protocol_spec(),
                artifact_dir=_P514 / "artifacts",
                device="cpu",
                limits=ResourceLimits(),
                rng=np.random.default_rng(seed + i),
            )
            p = policy_module.build_policy(bctx)
        p.reset(ctx)
        policies.append(p)

    matched_hist, cov_hist, col_hist = [], [], []
    total = 0.0
    for _ in range(cfg.horizon):
        actions = {
            aid: np.asarray(policies[i].act(observations[aid]), dtype=np.float32)
            for i, aid in enumerate(agent_ids)
        }
        observations, rewards, _t, _tr, _i = env.step(actions)
        total += float(rewards[agent_ids[0]])
        m = compute_step_metrics(env.unwrapped._current_snapshot)
        matched_hist.append(m.matched_targets)
        cov_hist.append(m.coverage_rate)
        col_hist.append(m.collision_rate)
    for p in policies:
        p.close()
    env.close()
    return {
        "matched": np.array(matched_hist),
        "cov": float(np.mean(cov_hist)),
        "col": float(np.mean(col_hist)),
        "J": total / cfg.horizon,
    }


def main():
    suite = load_suite(PUBLIC_SUITE)
    case = suite.groups[0].cases[0]
    n_seeds = 40

    for name, module in (("random", random_ref), ("rule", rule)):
        runs = [rollout(module, case, 900000 + 7919 * k) for k in range(n_seeds)]
        matched = np.concatenate([r["matched"] for r in runs])
        cov = float(np.mean([r["cov"] for r in runs]))
        col = float(np.mean([r["col"] for r in runs]))
        J = float(np.mean([r["J"] for r in runs]))
        print(
            f"{name:>7s}: J={J:+.4f}  平均覆盖率={cov:.3f}  平均碰撞率={col:.3f}  "
            f"匹配数分布={dict(zip(*np.unique(matched, return_counts=True)))}"
        )
        print(
            f"         每步覆盖率均值={matched.mean()/3:.4f}  "
            f"分解: C-K*0.2 = {matched.mean()/3:.4f} - {col:.4f}*0.2 = "
            f"{matched.mean()/3 - 0.2*col:+.4f}"
        )
        print(
            f"         有覆盖的步占比={(matched > 0).mean():.3f}  "
            f"至少覆盖2个目标的步占比={(matched >= 2).mean():.3f}"
        )


if __name__ == "__main__":
    main()
