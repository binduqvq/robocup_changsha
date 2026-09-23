"""模板基线在泛化轴上的测量（开发期工具）。

首轮只测了模板 PPO 在公开 4 个场景上的 66.67 分，没有测它在留出场景与
参数网格上的表现。要公平说明"提升"，必须把起点也放到同一把尺子上量。

模板代码/权重位于 `participant/_template/`（官方模板，本目录只读取不修改）。

用法：
    .venv/Scripts/python.exe participant/P514/tools/baseline_template.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_P514 = _REPO_ROOT / "participant" / "P514"
_TEMPLATE = _REPO_ROOT / "participant" / "_template"
for _p in (str(_REPO_ROOT), str(_P514)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from policies import random_ref, rule  # noqa: E402
from tools import generalize as g  # noqa: E402
from tools import sweep  # noqa: E402


def load_template_module():
    """加载官方模板 entry.py（刻意不改动模板目录）。"""
    spec = importlib.util.spec_from_file_location("_tmpl_entry", _TEMPLATE / "entry.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tmpl_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


class _TemplateAdapter:
    """把模板 build_policy(BuildContext) 适配成工具要求的 build_policy_for_agent 签名。"""

    def __init__(self, module):
        self._module = module

    def build_policy_for_agent(self, context, artifact_dir=None, params=None):
        """始终加载模板自带的 artifacts（调用方传进来的目录与本基线的权重无关）。"""
        from coverage_bench.protocol import BuildContext, ResourceLimits, get_protocol_spec

        ctx = BuildContext(
            spec=get_protocol_spec(),
            artifact_dir=_TEMPLATE / "artifacts",
            device="cpu",
            limits=ResourceLimits(),
            rng=np.random.default_rng(context.policy_seed),
        )
        return self._module.build_policy(ctx)


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
    tmpl = _TemplateAdapter(load_template_module())
    base = g._base_config()
    per_cell = 32

    # 1) 公开套件 4 个场景
    from coverage_bench.suites import load_suite

    suite = load_suite(g.PUBLIC_SUITE)
    pub = [c for grp in suite.groups for c in grp.cases]
    rows = sweep.evaluate_cases(tmpl, pub)
    s, pg, _cov, _col = sweep.score_of(rows)
    print(f"模板 PPO 公开套件 score = {s:.3f}  " + " ".join(f"{k}={v:+.4f}" for k, v in pg.items()))

    # 2) 留出场景 128 个
    ho = sweep.make_heldout_cases(555000, 32, 2)
    rows = sweep.evaluate_cases(tmpl, ho)
    s, pg, cov, _col = sweep.score_of(rows)
    print(f"模板 PPO 留出场景(128) score = {s:.3f}  " + " ".join(f"{k}={v:+.4f}" for k, v in pg.items())
          + f"  cov={sum(cov.values()) / len(cov):.3f}")

    # 3) 随机化轴 12 格
    print()
    print(f"{'cell':>16s} {'tmpl J':>9s} {'rule J':>9s} {'rand J':>9s}")
    t_vals, r_vals, n_vals = [], [], []
    for cid, n, m, lay in CELLS:
        case = g.ScenarioCase(cid, "x", g.make_config(base, n, m, 10, lay), 1)
        tv, rv, nv = [], [], []
        for k in range(per_cell):
            seed = 555000 + 104729 * k
            ck = g.ScenarioCase(cid, "x", case.task_config, seed)
            tv.append(g.run_case(tmpl, ck, None, seed=seed)[0])
            rv.append(g.run_case(rule, ck, {"goal_select": "coordinated"}, seed=seed)[0])
            nv.append(g.run_case(random_ref, ck, None, seed=seed)[0])
        t_vals.append(np.mean(tv)); r_vals.append(np.mean(rv)); n_vals.append(np.mean(nv))
        print(f"{cid:>16s} {np.mean(tv):>+9.4f} {np.mean(rv):>+9.4f} {np.mean(nv):>+9.4f}")

    print("-" * 56)
    print(f"12 格平均: 模板 {np.mean(t_vals):+.4f}  最终 {np.mean(r_vals):+.4f}  随机 {np.mean(n_vals):+.4f}")
    wins = sum(1 for a, b in zip(r_vals, t_vals) if a > b)
    print(f"最终版本优于模板的格数: {wins}/{len(CELLS)}")


if __name__ == "__main__":
    main()
