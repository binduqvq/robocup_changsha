"""最终版本的统一指标汇总（开发期工具）。

把"随机化轴"（N×M×layout 12 格）上规则策略与随机基线的深度分开测量，
用于量化"策略相对随机到底强多少"，而不是只看总分。

用法：
    .venv/Scripts/python.exe participant/P514/tools/final_metrics.py
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

from policies import random_ref, rule  # noqa: E402
from tools import generalize as g  # noqa: E402

CELLS = [
    ("N3M3-uniform", 3, 3, "uniform"),
    ("N3M3-crossing", 3, 3, "crossing"),
    ("N3M3-clustered", 3, 3, "clustered"),
    ("N1M1", 1, 1, "uniform"),
    ("N2M3", 2, 3, "uniform"),
    ("N4M4", 4, 4, "uniform"),
    ("N4M5-crossing", 4, 5, "crossing"),
    ("N5M7", 5, 7, "uniform"),
    ("N8M8", 8, 8, "uniform"),
    ("N3M6", 3, 6, "uniform"),
    ("N6M3", 6, 3, "uniform"),
    ("N8M3-crossing", 8, 3, "crossing"),
]


def main():
    base = g._base_config()
    per_cell = 32
    print(f"随机化轴：{len(CELLS)} 格 × {per_cell} 独立种子")
    print(f"{'cell':>16s} {'rule J':>9s} {'rand J':>9s} {'gain':>9s} {'ratio':>7s} {'cov':>7s} {'col':>7s}")
    rows = []
    for cid, n, m, lay in CELLS:
        case = g.ScenarioCase(cid, "x", g.make_config(base, n, m, 10, lay), 1)
        rj, rc, rcol = [], [], []
        nj = []
        for k in range(per_cell):
            seed = 555000 + 104729 * k
            ck = g.ScenarioCase(cid, "x", case.task_config, seed)
            r = g.run_case(rule, ck, {"goal_select": "coordinated"}, seed=seed)
            nres = g.run_case(random_ref, ck, None, seed=seed)
            rj.append(r[0]); rc.append(r[1]); rcol.append(r[2]); nj.append(nres[0])
        rj_m, nj_m = float(np.mean(rj)), float(np.mean(nj))
        rows.append((cid, rj_m, nj_m, rc, rcol))
        ratio = (rj_m / nj_m) if abs(nj_m) > 1e-6 else float("nan")
        print(f"{cid:>16s} {rj_m:>+9.4f} {nj_m:>+9.4f} {rj_m - nj_m:>+9.4f} "
              f"{ratio:>7.2f} {np.mean(rc):>7.3f} {np.mean(rcol):>7.3f}")

    gains = [r[1] - r[2] for r in rows]
    covs = [np.mean(r[3]) for r in rows]
    cols = [np.mean(r[4]) for r in rows]
    print("-" * 72)
    print(f"平均增益 {np.mean(gains):+.4f}；策略劣于随机的格数 "
          f"{sum(1 for x in gains if x < 0)}/{len(gains)}")
    print(f"平均覆盖率 {np.mean(covs):.4f}；平均碰撞率 {np.mean(cols):.4f}")
    # 相对提升（不含随机基线接近 0 的格子）
    rel = [
        (r[1] - r[2]) / r[2]
        for r in rows
        if abs(r[2]) > 0.01
    ]
    print(f"相对随机的中位提升 {np.median(rel) * 100:.1f}%（{len(rel)} 个有效格）")


if __name__ == "__main__":
    main()
