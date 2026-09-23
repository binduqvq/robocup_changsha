"""三方案同口径对照：规则策略 vs 学习策略 vs 随机基线。

在同一批场景（公开套件 4 个场景 + 留出场景）上依次评测三种策略，
输出统一口径的 performance_score（= 1000·Σ w_k·mean_j，w=0.5）与覆盖率/碰撞率，
用于判断"学习路线是否值得替换规则路线"。

学习策略走 `policies/learned.py`（纯 NumPy 推理，加载 `artifacts/policy.npz`），
与正式评测的推理路径同类。

用法：
    .venv/Scripts/python.exe participant/P514/tools/compare_three.py --n 32 --repeats 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_P514 = _REPO_ROOT / "participant" / "P514"
for _p in (str(_REPO_ROOT), str(_P514)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coverage_bench.suites import load_suite  # noqa: E402

from policies import learned, random_ref, rule  # noqa: E402
from tools import sweep  # noqa: E402
from tools.rollout import PUBLIC_SUITE  # noqa: E402

WEIGHTS = {"basic": 0.5, "cooperation": 0.5}


def score_of(rows):
    """把逐场景结果聚合成与评测器一致的 performance_score（1000·Σ w_k·mean_j）。"""
    per_group: dict[str, list[float]] = {}
    for r in rows:
        per_group.setdefault(r["group_id"], []).append(r["mean_j"])
    total = sum(1000.0 * WEIGHTS[g] * float(np.mean(v)) for g, v in per_group.items())
    return total, {g: float(np.mean(v)) for g, v in per_group.items()}


def public_cases():
    suite = load_suite(PUBLIC_SUITE)
    return [case for group in suite.groups for case in group.cases]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="留出场景每组场景数")
    ap.add_argument("--repeats", type=int, default=2, help="每个留出场景的独立种子数")
    ap.add_argument("--seed-offset", type=int, default=555000)
    ap.add_argument(
        "--learned-artifact-dirs",
        type=Path,
        nargs="+",
        default=[Path("outputs/P514/train-1M"), Path("outputs/P514/train-10M")],
        help="学习策略 policy.npz 所在目录（不属于提交产物，故不放 artifacts/）",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/P514/three_way.json"))
    args = ap.parse_args()

    combos = [("rule(submitted)", rule, None, None)]
    for d in args.learned_artifact_dirs:
        combos.append((f"learned:{Path(d).name}", learned, None, Path(d)))
    combos.append(("random", random_ref, None, None))

    results: dict[str, dict] = {}

    print("=== 公开套件（4 个独立场景；repeat 复用同一 scenario_seed，与评测器一致）===")
    cases_pub = public_cases()
    for name, module, params, art in combos:
        rows = sweep.evaluate_cases(module, cases_pub, params, artifact_dir=art)
        s, pg = score_of(rows)
        results.setdefault(name, {})["public"] = {"score": s, "per_group": pg}
        print(f"  {name:>16s} score={s:8.3f}  " + " ".join(f"{g}={v:+.4f}" for g, v in pg.items()))

    print(f"\n=== 留出场景（每组 {args.n}×{args.repeats} 独立种子，seed={args.seed_offset}）===")
    cases_ho = sweep.make_heldout_cases(args.seed_offset, args.n, args.repeats)
    for name, module, params, art in combos:
        t0 = time.time()
        rows = sweep.evaluate_cases(module, cases_ho, params, artifact_dir=art)
        s, pg = score_of(rows)
        cov = float(np.mean([r["mean_coverage_rate"] for r in rows]))
        col = float(np.mean([r["mean_collision_rate"] for r in rows]))
        results[name]["heldout"] = {"score": s, "per_group": pg, "cov": cov, "col": col}
        print(
            f"  {name:>16s} score={s:8.3f}  "
            + " ".join(f"{g}={v:+.4f}" for g, v in pg.items())
            + f"  cov={cov:.3f} col={col:.3f}  [{time.time() - t0:.1f}s]"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
