"""P514 参数泛化台（开发期使用，不参与推理）。

为什么需要它：公开套件只覆盖 N=M=3、T=10、layout∈{uniform,crossing} 这一个角落，
而官方合法范围是 N、M ∈ [1,8]、T ∈ [1,256]、layout ∈ {uniform,crossing,clustered}
（见 coverage_bench/config.py 的 _validate_numeric_bounds）。
正式核验使用未公开且与公开测试不重合的种子，因此"换规模/换回合同样能跑、且优于随机"
比在公开 4 个场景上刷分重要得多。

本工具在同一批场景上同时跑策略与随机基线，输出"相对随机的增益"，用来判断
策略是真的有效，还是仅仅复制了任务本身的可达性概率。

用法：
    .venv/Scripts/python.exe participant/P514/tools/generalize.py                 # 全部轴
    .venv/Scripts/python.exe participant/P514/tools/generalize.py --axis horizon
    .venv/Scripts/python.exe participant/P514/tools/generalize.py --axis params --per-cell 8
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
from coverage_bench.metrics import compute_step_metrics  # noqa: E402
from coverage_bench.protocol import EpisodeContext  # noqa: E402
from coverage_bench.runtime import _public_task_params  # noqa: E402
from coverage_bench.suites import ScenarioCase  # noqa: E402

from policies import probe, random_ref, rule  # noqa: E402

PUBLIC_SUITE = _REPO_ROOT / "configs" / "public-suite-v1.yaml"

_PROBE_PARAMS = {"_class": "OracleNearest"}


def _base_config():
    from coverage_bench.suites import load_suite

    suite = load_suite(PUBLIC_SUITE)
    return suite.groups[0].cases[0].task_config


def make_config(base, num_agents=None, num_targets=None, horizon=None, layout=None, **pub_overrides):
    """在公开套件任务配置上做局部覆盖，得到同构但不同规模的配置。"""
    cfg = base
    if num_agents is not None or num_targets is not None or horizon is not None:
        cfg = cfg.model_copy(
            update={
                k: v
                for k, v in (
                    ("num_agents", num_agents),
                    ("num_targets", num_targets),
                    ("horizon", horizon),
                )
                if v is not None
            }
        )
    if layout is not None or pub_overrides:
        pub = cfg.public.model_copy(update=pub_overrides) if pub_overrides else cfg.public
        sc = cfg.scenario.model_copy(update={"layout_kind": layout}) if layout else cfg.scenario
        cfg = cfg.model_copy(update={"public": pub, "scenario": sc})
    return cfg


def run_case(policy_module, case, params=None, seed=1234, global_state=False):
    """跑一个场景，返回 (mean_j, mean_coverage, mean_collision, steps)。

    global_state=True 时按需把训练专用全局状态注入诊断探针（只用于测上界）。
    """
    cfg = case.task_config
    env = make_training_env(cfg)
    observations, _ = env.reset(seed=case.scenario_seed)
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

    def inject():
        if global_state and any(hasattr(p, "global_state") for p in policies):
            st = env.state()
            for p in policies:
                if hasattr(p, "global_state"):
                    p.global_state = st

    inject()
    total = 0.0
    covs, cols = [], []
    steps_done = 0
    for _ in range(cfg.horizon):
        actions = {}
        for i, aid in enumerate(agent_ids):
            actions[aid] = np.asarray(policies[i].act(observations[aid]), dtype=np.float32)
        observations, rewards, _t, _tr, _i = env.step(actions)
        inject()
        total += float(rewards[agent_ids[0]])
        m = compute_step_metrics(env.unwrapped._current_snapshot)
        covs.append(m.coverage_rate)
        cols.append(m.collision_rate)
        steps_done += 1
    for p in policies:
        p.close()
    env.close()
    return total / steps_done, float(np.mean(covs)), float(np.mean(cols)), steps_done


def compare(cases, per_cell, params=None, seed_base=555000, with_oracle=False):
    """在同一批场景上比较策略与随机基线（可选再跑全信息 oracle），返回逐格结果。"""
    rows = []
    for case in cases:
        rule_vals, rand_vals, orac_vals = [], [], []
        for k in range(per_cell):
            seed = seed_base + 104729 * k
            case_k = ScenarioCase(
                case_id=case.case_id,
                group_id=case.group_id,
                task_config=case.task_config,
                scenario_seed=seed,
            )
            rj, rc, rcol, _ = run_case(rule, case_k, params, seed=seed)
            nj, nc, ncol, _ = run_case(random_ref, case_k, None, seed=seed)
            rule_vals.append((rj, rc, rcol))
            rand_vals.append((nj, nc, ncol))
            if with_oracle:
                oj, _, _, _ = run_case(probe, case_k, _PROBE_PARAMS, seed=seed, global_state=True)
                orac_vals.append(oj)
        rj = float(np.mean([v[0] for v in rule_vals]))
        nj = float(np.mean([v[0] for v in rand_vals]))
        rc = float(np.mean([v[1] for v in rule_vals]))
        rcol = float(np.mean([v[2] for v in rule_vals]))
        rows.append(
            {
                "case_id": case.case_id,
                "n": case.task_config.num_agents,
                "m": case.task_config.num_targets,
                "t": case.task_config.horizon,
                "layout": case.task_config.scenario.layout_kind,
                "rule_j": rj,
                "rand_j": nj,
                "gain": rj - nj,
                "oracle_j": float(np.mean(orac_vals)) if orac_vals else float("nan"),
                "rule_cov": rc,
                "rule_col": rcol,
                "steps": case.task_config.horizon,
            }
        )
    return rows


def report(rows, title, with_oracle=False):
    print(f"\n=== {title} ===")
    head = (f"{'case':>18s} {'N':>2s} {'M':>2s} {'T':>4s} {'layout':>10s} "
            f"{'rule J':>9s} {'rand J':>9s} {'gain':>9s}")
    if with_oracle:
        head += f" {'oracle J':>9s} {'gap':>9s}"
    print(head + f" {'cov':>7s} {'col':>7s}")
    for r in rows:
        line = (f"{r['case_id']:>18s} {r['n']:>2d} {r['m']:>2d} {r['t']:>4d} {r['layout']:>10s} "
                f"{r['rule_j']:>+9.4f} {r['rand_j']:>+9.4f} {r['gain']:>+9.4f}")
        if with_oracle:
            gap = r["oracle_j"] - r["rule_j"]
            line += f" {r['oracle_j']:>+9.4f} {gap:>+9.4f}"
        line += f" {r['rule_cov']:>7.3f} {r['rule_col']:>7.3f}"
        print(line)
    gains = [r["gain"] for r in rows]
    print(f"  -> 平均增益 {np.mean(gains):+.4f}；策略劣于随机的格子数 "
          f"{sum(1 for g in gains if g < 0)}/{len(gains)}")
    if with_oracle:
        gaps = [r["oracle_j"] - r["rule_j"] for r in rows]
        print(f"  -> 与全信息 oracle 的平均差距 {np.mean(gaps):+.4f}"
              f"（正数 = 还有空间，负数 = 规则更好）")


def _cases(combos, tag):
    return [
        ScenarioCase(
            case_id=cid,
            group_id=tag,
            task_config=make_config(_BASE, n, m, t, lay, **extra),
            scenario_seed=1,
        )
        for (cid, n, m, t, lay, extra) in combos
    ]


def axis_params_cases():
    combos = [
        ("N3M3-unif", 3, 3, 10, "uniform", {}),
        ("N3M3-cros", 3, 3, 10, "crossing", {}),
        ("N3M3-clus", 3, 3, 10, "clustered", {}),
        ("N1M1-unif", 1, 1, 10, "uniform", {}),
        ("N2M3-unif", 2, 3, 10, "uniform", {}),
        ("N4M4-unif", 4, 4, 10, "uniform", {}),
        ("N4M5-cros", 4, 5, 10, "crossing", {}),
        ("N5M7-unif", 5, 7, 10, "uniform", {}),
        ("N8M8-unif", 8, 8, 10, "uniform", {}),
        ("N3M6-unif", 3, 6, 10, "uniform", {}),
        ("N6M3-unif", 6, 3, 10, "uniform", {}),
        ("N8M3-cros", 8, 3, 10, "crossing", {}),
    ]
    return _cases(combos, "params")


def axis_horizon_cases():
    combos = [(f"T{t}", 3, 3, t, "uniform", {}) for t in (5, 10, 20, 50, 100, 256)]
    return _cases(combos, "horizon")


def axis_sensing_cases():
    combos = []
    for sr in (0.35, 0.6, 1.2, 3.0):
        combos.append((f"sense{sr}", 3, 3, 10, "uniform", {"sense_radius": sr}))
    for tms in (0.05, 0.2, 0.5):
        combos.append((f"tspd{tms}", 3, 3, 10, "uniform", {"target_max_speed": tms}))
    return _cases(combos, "sensing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="all", choices=["all", "params", "horizon", "sensing"])
    ap.add_argument("--per-cell", type=int, default=6, help="每格独立种子数")
    ap.add_argument("--oracle", action="store_true", help="同时跑全信息 oracle 作为上界对照")
    args = ap.parse_args()

    global _BASE
    _BASE = _base_config()

    axes = {
        "params": (axis_params_cases, "参数泛化（N×M×layout）"),
        "horizon": (axis_horizon_cases, "回合长度泛化（T）"),
        "sensing": (axis_sensing_cases, "感知/目标速度泛化"),
    }
    chosen = list(axes) if args.axis == "all" else [args.axis]
    for key in chosen:
        fn, title = axes[key]
        rows = compare(fn(), args.per_cell, with_oracle=args.oracle)
        report(rows, title, with_oracle=args.oracle)


_BASE = None


if __name__ == "__main__":
    main()
