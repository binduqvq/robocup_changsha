"""一次性诊断：比较"最强一步前瞻全局搜索上界"与最终规则策略（开发期工具）。

用法：
    .venv/Scripts/python.exe participant/P514/tools/upper_bound.py --n 16 --levels -1,0,1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_P514 = _REPO_ROOT / "participant" / "P514"
for _p in (str(_REPO_ROOT), str(_P514)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from policies import probe, rule  # noqa: E402
from tools import sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="每组场景数")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--levels", default="-1,0,1", help="每个动作轴的离散档位")
    ap.add_argument("--seed-offset", type=int, default=555000)
    args = ap.parse_args()

    levels = [float(x) for x in args.levels.split(",")]
    cases = sweep.make_heldout_cases(args.seed_offset, args.n, args.repeats)
    print(f"对比场景: {len(cases)} 个（每组 {args.n}×{args.repeats}）")

    for name, module, params in (
        ("rule (最终提交)", rule, None),
        ("oracle-greedy-1step (全局信息上界)", probe, {"_class": "OracleGreedySearch", "grid": levels}),
    ):
        t0 = time.time()
        rows = sweep.evaluate_cases(module, cases, params)
        score, detail, cov, col = sweep.score_of(rows)
        print(
            f"{name:>38s}  score={score:8.3f}  "
            + " ".join(f"{g}={v:+.4f}" for g, v in detail.items())
            + f"  cov={sum(cov.values())/len(cov):.3f} col={sum(col.values())/len(col):.3f}"
            + f"  [{time.time()-t0:.1f}s]"
        )


if __name__ == "__main__":
    main()
