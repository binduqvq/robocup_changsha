"""对公开套件成绩做 bootstrap 置信区间（开发期工具）。

公开套件只有 4 个独立场景（同一 case 的两次 repeat 复用同一 scenario_seed），
因此 performance_score 的点估计方差很大。本工具用**按场景**的 bootstrap
给出总分的 95% 区间，避免把点估计当成稳定结论。

用法：
    .venv/Scripts/python.exe participant/P514/tools/bootstrap_ci.py --eval outputs/P514/final/eval
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WEIGHTS = {"basic": 0.5, "cooperation": 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("outputs/P514/final/eval"))
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=81273)
    args = ap.parse_args()

    rows = list(csv.DictReader((args.eval / "episodes.csv").open(encoding="utf-8")))
    result = json.loads((args.eval / "result.json").read_text(encoding="utf-8"))

    # 按 (group, case) 聚合：重复回合取均值，得到"每个场景一个样本"
    per_case: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] != "ok":
            continue
        per_case[r["group_id"]][r["case_id"]].append(float(r["mean_j"]))

    case_means = {
        g: {c: float(np.mean(v)) for c, v in cases.items()} for g, cases in per_case.items()
    }
    print("每个场景的 mean_j（重复回合已取均值）：")
    for g, cases in case_means.items():
        print(f"  [{g}] " + "  ".join(f"{c}={v:+.4f}" for c, v in sorted(cases.items())))

    point = float(result.get("performance_score"))
    print(f"\n点估计 performance_score = {point:.4f}（n_scenes={sum(len(v) for v in case_means.values())}）")

    rng = np.random.default_rng(args.seed)
    scores = np.empty(args.resamples, dtype=np.float64)
    groups = sorted(case_means)
    for i in range(args.resamples):
        total = 0.0
        for g in groups:
            vals = np.array(list(case_means[g].values()), dtype=np.float64)
            draw = rng.integers(0, len(vals), size=len(vals))
            total += 1000.0 * WEIGHTS[g] * float(np.mean(vals[draw]))
        scores[i] = total

    lo, hi = np.percentile(scores, [2.5, 97.5])
    print(f"bootstrap 95% 区间: [{lo:.2f}, {hi:.2f}]  (宽度 {hi - lo:.2f})")
    print(f"bootstrap 中位数     : {np.median(scores):.2f}")
    print(
        "\n说明：区间宽度反映公开套件的场景数太少（仅 4 个独立场景），"
        "不代表策略真实波动；留出场景估计更可靠。"
    )


if __name__ == "__main__":
    main()
