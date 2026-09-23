"""把训练得到的 SB3 PPO 权重导出为纯 NumPy 的 artifacts/policy.npz。

为什么需要它：评测环境没有 PyTorch，且提交源码禁止 `torch.load`/`pickle.load`
（静态审计硬拒绝）。因此训练侧必须显式把 actor 网络的权重和 VecNormalize
统计量抽成 npz，再由 `policies/learned.py` 用 NumPy 前向推理。

本脚本会做**导出一致性检查**：在同一批真实环境观测上比较
`model.predict(obs, deterministic=True)` 与 NumPy 前向的动作，要求 max|Δa| ≤ 1e-5。

用法（仓库根目录，训练环境）：
    .venv-train/Scripts/python.exe participant/P514/tools/export_learned.py \
        --run-dir outputs/P514/train-1M --out participant/P514/artifacts/policy.npz
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
from coverage_bench.protocol import get_protocol_spec  # noqa: E402
from coverage_bench.spaces import flatten_observation  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402

PUBLIC_SUITE = _REPO_ROOT / "configs" / "public-suite-v1.yaml"


def extract_mlp_weights(model):
    """从 SB3 PPO 的 actor 中抽出两层 tanh 隐层 + 线性输出头的权重。"""
    sd = model.policy.state_dict()
    return {
        "W1": sd["mlp_extractor.policy_net.0.weight"].cpu().numpy().T.copy(),
        "b1": sd["mlp_extractor.policy_net.0.bias"].cpu().numpy().copy(),
        "W2": sd["mlp_extractor.policy_net.2.weight"].cpu().numpy().T.copy(),
        "b2": sd["mlp_extractor.policy_net.2.bias"].cpu().numpy().copy(),
        "W3": sd["action_net.weight"].cpu().numpy().T.copy(),
        "b3": sd["action_net.bias"].cpu().numpy().copy(),
    }


def collect_raw_observations(n_obs: int, seed: int = 424242):
    """在真实环境上收集未归一化的 104 维观测（与训练同源的 flatten）。"""
    suite = load_suite(PUBLIC_SUITE)
    spec = get_protocol_spec()
    raws = []
    rng = np.random.default_rng(seed)
    for group in suite.groups:
        cfg = group.cases[0].task_config
        env = make_training_env(cfg)
        obs, _ = env.reset(seed=seed)
        agent_ids = list(env.agents)
        while len(raws) < n_obs:
            for aid in agent_ids:
                raws.append(flatten_observation(obs[aid], spec).astype(np.float64))
            actions = {aid: rng.uniform(-1, 1, size=2).astype(np.float32) for aid in agent_ids}
            obs, _r, _t, _tr, _i = env.step(actions)
            if not env.agents:
                obs, _ = env.reset(seed=seed + 1)
                agent_ids = list(env.agents)
        env.close()
        if len(raws) >= n_obs:
            break
    return np.stack(raws[:n_obs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="train.py 的输出目录")
    ap.add_argument("--out", type=Path, default=_P514 / "artifacts" / "policy.npz")
    ap.add_argument("--n-obs", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(str(args.run_dir / "model.zip"), device="cpu")
    stats = np.load(args.run_dir / "vecnormalize-stats.npz")
    weights = extract_mlp_weights(model)

    obs_mean = np.asarray(stats["obs_mean"], dtype=np.float64)
    obs_var = np.asarray(stats["obs_var"], dtype=np.float64)
    obs_eps = float(stats["obs_eps"])
    obs_clip = float(stats["obs_clip"])

    raw = collect_raw_observations(args.n_obs)
    x = np.clip((raw - obs_mean) / np.sqrt(obs_var + obs_eps), -obs_clip, obs_clip)

    import torch

    with torch.no_grad():
        sb3_actions, _ = model.predict(x, deterministic=True)
    sb3_actions = np.asarray(sb3_actions, dtype=np.float64)

    h = np.tanh(x @ weights["W1"] + weights["b1"])
    h = np.tanh(h @ weights["W2"] + weights["b2"])
    np_actions = np.clip(h @ weights["W3"] + weights["b3"], -1.0, 1.0)

    max_diff = float(np.max(np.abs(sb3_actions - np_actions)))
    print(f"样本数={len(x)}  max|Δa|={max_diff:.3e}  容差={args.tol:.1e}")
    if max_diff > args.tol:
        raise SystemExit(f"导出一致性检查失败: max|Δa|={max_diff:.3e} > {args.tol:.1e}")
    print("导出一致性检查通过")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        obs_mean=obs_mean, obs_var=obs_var,
        obs_eps=np.array(obs_eps), obs_clip=np.array(obs_clip),
        **weights,
    )
    print(f"已写入 {args.out}  字节数={args.out.stat().st_size}")


if __name__ == "__main__":
    main()
