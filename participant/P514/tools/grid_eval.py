"""P514 参数网格评测（开发期使用，不参与推理）。

为什么需要它：`tools/sweep.py --policy entry --grid ...` 会被 `entry.SUBMISSION_OVERRIDES`
覆盖，实验参数静默失效（这个坑真实发生过）。本脚本直接调用评测函数并**显式传入**参数网格，
并且会先做一次"参数确实生效"的自检（对比两套预期不同的参数是否给出不同分数）。

用法：
    .venv/Scripts/python.exe participant/P514/tools/grid_eval.py --config participant/P514/configs/grid-cover.json
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

import entry  # noqa: E402
from tools.sweep import evaluate_cases, make_heldout_cases, score_of  # noqa: E402


def run_grid(cases, combos, label):
    print(f"\n=== {label}（场景 {len(cases)} 个）===")
    best = None
    for p in combos:
        rows = evaluate_cases(entry, cases, dict(p))
        s, d, c, col = score_of(rows)
        cmean = float(np.mean(list(col.values())))
        print(
            f"  score={s:8.3f}  basic={d['basic']:+.4f}  coop={d['cooperation']:+.4f}  "
            f"col={cmean:.4f}  {p}"
        )
        if best is None or s > best[0]:
            best = (s, p)
    print(f"  -> BEST score={best[0]:.4f} params={best[1]}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed-offset", type=int, default=555000)
    ap.add_argument("--allow-neutral", action="store_true",
                    help="允许某些键不生效（默认若发现某键完全无影响会警告）")
    args = ap.parse_args()

    grid = json.loads(args.config.read_text(encoding="utf-8"))
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    cases = make_heldout_cases(args.seed_offset, args.n, args.repeats)

    # 生效性自检：对每个键分别扫描它是否真的改变结果
    print("=== 参数生效性自检（防止 SUBMISSION_OVERRIDES 静默吞掉实验参数）===")
    for k in keys:
        vals = grid[k]
        if len(vals) < 2:
            continue
        sub = [dict(zip(keys, v)) for v in itertools.product(
            *[vals if kk == k else [grid[kk][0]] for kk in keys])]
        seen = set()
        for params in sub:
            rows = evaluate_cases(entry, cases, dict(params))
            s, _d, _c, _col = score_of(rows)
            seen.add(round(s, 6))
        ok = len(seen) > 1
        print(f"  {'OK ' if ok else 'WARN'} 键 {k}: {len(seen)} 个不同分数 "
              f"（{'参数生效' if ok else '该键对所有取值给出相同结果，需人工确认'}）")
        if not ok and not args.allow_neutral:
            print("       WARN: 若预期该键应有影响，请检查参数是否被覆盖。")

    run_grid(cases, combos, f"{args.config.name}  seed_base={args.seed_offset}")


if __name__ == "__main__":
    main()
