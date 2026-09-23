"""汇总一个评测输出目录的关键指标（开发期使用）。

用法：
    .venv/Scripts/python.exe participant/P514/tools/summarize_eval.py outputs/P514/E006-rule-official/eval
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: summarize_eval.py <评测输出目录>")
    out = Path(sys.argv[1])
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))

    print(f"status            : {result.get('status')}")
    print(f"performance_score : {result.get('performance_score')}")
    print(f"provenance        : {result.get('provenance')}")
    print(f"participant_id    : {result.get('participant_id')}")
    print(f"suite_id          : {result.get('suite_id')}")
    print(
        "versions          : "
        f"protocol={result.get('protocol_version')} "
        f"task={result.get('task_version')} "
        f"score={result.get('score_version')}"
    )
    for gid, m in (result.get("group_metrics") or {}).items():
        print(
            f"  [{gid}] episodes={m.get('num_episodes')} mean_j={m.get('mean_j'):+.4f} "
            f"cov={m.get('mean_coverage_rate'):.3f} col={m.get('mean_collision_rate'):.3f} "
            f"ci95_j={m.get('ci95_j')} act_ms_mean={m.get('act_ms_mean'):.3f}"
        )
    errs = result.get("errors") or []
    print(f"errors            : {len(errs)}")
    for e in errs[:5]:
        print(f"   {e.get('code')} {e.get('stage')} {e.get('case_id')} {e.get('message')}")

    ep_path = out / "episodes.csv"
    if ep_path.is_file():
        rows = list(csv.DictReader(ep_path.open(encoding="utf-8")))
        print(f"\nepisodes ({len(rows)}):")
        for r in rows:
            act_cnt = max(1, int(r["act_count"]))
            print(
                f"  {r['group_id']:>12s} {r['case_id']:>8s} rep{r['repeat_index']} "
                f"status={r['status']:<4s} steps={r['steps_completed']:>2s} "
                f"R={float(r['return_sum']):+.4f} cov={float(r['mean_coverage_rate']):.3f} "
                f"col={float(r['mean_collision_rate']):.3f} "
                f"act_ms_mean={float(r['act_ms_sum']) / act_cnt:.3f}"
            )
        print(f"\nall status ok  : {all(r['status'] == 'ok' for r in rows)}")
        print(f"all steps == T : {all(int(r['steps_completed']) == int(r['horizon']) for r in rows)}")


if __name__ == "__main__":
    main()
