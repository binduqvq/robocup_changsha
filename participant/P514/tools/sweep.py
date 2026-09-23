"""P514 参数扫描 / 泛化评估工具（开发期使用，不参与推理）。

为什么需要它：公开套件只有 4 个场景（basic 2 + coop 2），直接在上面调参极易过拟合。
本工具用同构的任务配置 + 自选的 scenario_seed 生成"留出场景"，在小规模上与公开套件
同口径打分（score = 1000 * Σ 0.5 * mean_j），用于判断改动是否真的更泛化。

用法：
    .venv/Scripts/python.exe participant/P514/tools/sweep.py --suite heldout --n 24
"""
from __future__ import annotations

import argparse
import itertools
import json
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
from coverage_bench.suites import ScenarioCase, load_suite  # noqa: E402
from tools.rollout import POLICIES, PUBLIC_SUITE, build_agent_policy  # noqa: E402

_LAYOUT_BY_GROUP = {"basic": "uniform", "cooperation": "crossing"}


def make_heldout_cases(seed_offset: int, per_group: int, repeats: int = 2):
    """用公开套件的任务配置 + 自选种子生成留出场景（布局由组决定）。

    repeats > 1 时同一"场景编号"会得到 repeats 个**互不相同**的 scenario_seed，
    这样做才与评测器 repeat 的语义一致（评测器里 repeat 用的是同一个
    scenario_seed，因此公开套件的 2 次重复其实不是独立样本；本工具刻意用
    不同种子，让方差估计不被低估）。
    """
    suite = load_suite(PUBLIC_SUITE)
    cases = []
    for gi, group in enumerate(suite.groups):
        tc = group.cases[0].task_config
        expected = _LAYOUT_BY_GROUP[group.group_id]
        assert tc.scenario.layout_kind == expected, (tc.scenario.layout_kind, expected)
        for k in range(per_group):
            for rep in range(repeats):
                seed = seed_offset + 1000003 * gi + 7919 * k + 104729 * rep
                cases.append(
                    ScenarioCase(
                        case_id=f"{group.group_id}-ho{k}r{rep}",
                        group_id=group.group_id,
                        task_config=tc,
                        scenario_seed=seed,
                    )
                )
    return cases


def evaluate_cases(policy_module, cases, params=None, verbose=False, artifact_dir=None):
    rows = []
    for case in cases:
        cfg = case.task_config
        env = make_training_env(cfg)
        observations, _ = env.reset(seed=case.scenario_seed)
        agent_ids = list(env.agents)
        policies = []
        for i in range(cfg.num_agents):
            policy, _ = build_agent_policy(
                policy_module, case, i, policy_seed=1000 + i, params=params,
                artifact_dir=artifact_dir,
            )
            policies.append(policy)

        def inject_state():
            # 仅诊断用：把训练专用全局状态注入"全局感知"探针策略
            if any(hasattr(p, "global_state") for p in policies):
                st = env.state()
                for p in policies:
                    if hasattr(p, "global_state"):
                        p.global_state = st

        inject_state()
        total_return = 0.0
        covs, cols = [], []
        for _step in range(cfg.horizon):
            actions = {
                aid: np.asarray(policies[i].act(observations[aid]), dtype=np.float32)
                for i, aid in enumerate(agent_ids)
            }
            observations, rewards, _t, _tr, _i = env.step(actions)
            inject_state()
            total_return += float(rewards[agent_ids[0]])
            m = compute_step_metrics(env.unwrapped._current_snapshot)
            covs.append(m.coverage_rate)
            cols.append(m.collision_rate)
        for p in policies:
            p.close()
        env.close()
        rows.append(
            {
                "case_id": case.case_id,
                "group_id": case.group_id,
                "return_sum": total_return,
                "mean_j": total_return / cfg.horizon,
                "mean_coverage_rate": float(np.mean(covs)),
                "mean_collision_rate": float(np.mean(cols)),
            }
        )
        if verbose:
            print(
                f"   {case.case_id:>14s} R={total_return:+.4f} cov={np.mean(covs):.3f} "
                f"col={np.mean(cols):.3f} seed={case.scenario_seed}"
            )
    return rows


def score_of(rows):
    per_group = {}
    for r in rows:
        per_group.setdefault(r["group_id"], []).append(r["mean_j"])
    score = sum(1000.0 * 0.5 * float(np.mean(v)) for v in per_group.values())
    detail = {g: float(np.mean(v)) for g, v in per_group.items()}
    cov = {g: float(np.mean([r["mean_coverage_rate"] for r in rows if r["group_id"] == g])) for g in per_group}
    col = {g: float(np.mean([r["mean_collision_rate"] for r in rows if r["group_id"] == g])) for g in per_group}
    return score, detail, cov, col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="rule")
    parser.add_argument("--seed-offset", type=int, default=555000)
    parser.add_argument("--n", type=int, default=25, help="每组留出场景数")
    parser.add_argument("--repeats", type=int, default=2, help="每个留出场景的独立种子数")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--grid",
        default=None,
        help='JSON 网格，例如 \'{"avoid_gain":[0.0,0.9]}\'；留空只跑默认参数',
    )
    parser.add_argument("--grid-file", type=Path, default=None, help="从文件读 JSON 网格（避免 shell 引号问题）")
    args = parser.parse_args()

    if args.policy not in POLICIES:
        raise SystemExit(f"未知策略: {args.policy}（可用: {', '.join(POLICIES)}）")
    module = POLICIES[args.policy]
    cases = make_heldout_cases(args.seed_offset, args.n, args.repeats)
    print(
        f"留出场景: {len(cases)} 个 (每组 {args.n}×{args.repeats} 个独立种子)  "
        f"策略={args.policy}  种子起点={args.seed_offset}"
    )

    if args.grid_file is not None:
        grid = json.loads(args.grid_file.read_text(encoding="utf-8"))
    elif args.grid is not None:
        grid = json.loads(args.grid)
    else:
        grid = None

    if grid is None:
        rows = evaluate_cases(module, cases, None, args.verbose)
        score, detail, cov, col = score_of(rows)
        print("-" * 78)
        for g in detail:
            print(f"  {g:>12s} mean_j={detail[g]:+.4f} cov={cov[g]:.3f} col={col[g]:.3f}")
        print(f"HELDOUT performance_score = {score:.4f}")
        return

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    print(f"网格组合数: {len(combos)}")
    best = None
    for i, params in enumerate(combos):
        rows = evaluate_cases(module, cases, params)
        score, detail, cov, col = score_of(rows)
        line = f"[{i + 1:>3d}/{len(combos)}] score={score:8.3f}  " + " ".join(
            f"{g}={detail[g]:+.4f}" for g in detail
        ) + f"  col={np.mean(list(col.values())):.4f}  {params}"
        print(line)
        if best is None or score > best[0]:
            best = (score, params, detail)
    print("-" * 78)
    print(f"BEST score={best[0]:.4f} params={best[1]} detail={best[2]}")


if __name__ == "__main__":
    main()
