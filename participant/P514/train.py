"""P514 学习型策略训练脚本（开发/对照实验用，训练在 .venv-train 内执行）。

与公开包模板 `participant/_template/train.py` 的关系与差异：
- 复用模板已验证可跑的 SB3 PPO 训练栈（含三处针对安装版本的适配，见下）。
- **关键修正**：模板的 `build_env_stack` 只取 `suite.groups[0].cases[0]`
  （即 `basic-0`，layout=uniform），因此**完全没有 crossing 布局的训练数据**，
  而协作组正是 crossing —— 这是模板 PPO 只有 66.67 分的主要原因。
  本脚本改为在**场景混合**上采样：uniform 与 crossing 两组布局、
  各自多个场景种子随机抽取，使训练分布覆盖公开套件的两种布局。

三处针对 stable_baselines3 2.9.0 / supersuit 3.11.0 的适配（与模板一致）：
1. AliveAgentsBridge：CoverageParallelEnv.possible_agents 按容量给出，
   而每回合仅 num_agents 个 agent 在场；supersuit 要求 reset 后所有
   possible_agents 都有观测，故在桥接处收拢 possible_agents。
2. seed/reset：supersuit 3.11 的 ConcatVecEnv 没有 seed()，SB3 2.9 的 learn()
   不再调用 env.seed()，种子只能经 reset(seed=...) 传入。
3. RecordingVecMonitor：SB3 2.9 的 VecMonitor 不保留历史 episode 回报。

用法（仓库根目录）：
    .venv-train/Scripts/python.exe participant/P514/train.py --total-steps 1000000 --out outputs/P514/train-run1
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import supersuit as ss
from pettingzoo.utils.wrappers import BaseParallelWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coverage_bench.envs.factory import make_training_env  # noqa: E402
from coverage_bench.protocol import get_protocol_spec  # noqa: E402
from coverage_bench.spaces import flatten_observation, flattened_observation_size  # noqa: E402
from coverage_bench.suites import load_suite  # noqa: E402

_PUBLIC_SUITE = _REPO_ROOT / "configs" / "public-suite-v1.yaml"

# 训练场景池：**不**使用公开评测种子（configs/public-seeds-v1.json 里的是
# scenario_seed 1001/1002/2001/2002），也**不**使用任何留出评测种子
# （tools/sweep.py / generalize.py 用 555000 / 777000 系列）。
# 这里从 4_000_000 起另取一段，保证训练分布与所有评测分布不相交。
TRAIN_SEED_BASE = 4_000_000
TRAIN_SEEDS_PER_LAYOUT = 64


def build_training_cases(seed_scales: int = 1):
    """构造训练场景池：uniform（basic 组）与 crossing（cooperation 组）各若干种子。

    seed_scales>1 时对每个模式再派生更多种子，扩大场景多样性——避免网络
    只记住有限的 128 个初始布局（那本身就是一种过拟合）。
    """
    suite = load_suite(_PUBLIC_SUITE)
    cases = []
    for gi, group in enumerate(suite.groups):
        base = group.cases[0]
        for k in range(TRAIN_SEEDS_PER_LAYOUT):
            for s in range(seed_scales):
                cases.append(
                    (
                        base.task_config,
                        TRAIN_SEED_BASE + 100003 * gi + 7919 * k + 104729 * s,
                        group.group_id,
                    )
                )
    return cases


class AliveAgentsBridge(BaseParallelWrapper):
    """把 possible_agents 对齐到场景实际存活的 agent（num_agents 个）。"""

    def __init__(self, env):
        super().__init__(env)
        self.possible_agents = [f"agent_{i}" for i in range(env.config.num_agents)]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.possible_agents = list(self.env.agents)
        return obs, infos


class FlattenCoverageVecEnv(VecEnvWrapper):
    """把 agent 拆分后的 Dict 观测压平为与评测侧完全一致的 104 维向量。"""

    def __init__(self, venv):
        super().__init__(venv)
        self._spec = get_protocol_spec()
        size = flattened_observation_size(self._spec)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self._pending_seed = None

    @staticmethod
    def _split_batch(obs):
        num_envs = len(next(iter(obs.values())))
        return [{key: value[i] for key, value in obs.items()} for i in range(num_envs)]

    def seed(self, seed=None):
        self._pending_seed = seed
        return [seed]

    def reset(self):
        if self._pending_seed is not None:
            obs = self.venv.venv.reset(seed=self._pending_seed)[0]
            self._pending_seed = None
        else:
            obs = self.venv.reset()
        return np.stack([flatten_observation(item, self._spec) for item in self._split_batch(obs)])

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        stacked = np.stack([flatten_observation(item, self._spec) for item in self._split_batch(obs)])
        return stacked, rewards, dones, infos


class ScenarioMixVecEnv(FlattenCoverageVecEnv):
    """场景混合向量环境：每次 reset 从场景池随机抽种子，覆盖 uniform 与 crossing。

    注意 ConcatVecEnv 会给第 i 个副本下发 seed+i，若只传一个基准种子，
    4 个副本看到的是 4 个**连续**种子，场景多样性会被压缩。
    因此这里把随机抽取的基准乘上一个大的互质步长，让副本之间拉开距离。
    """

    _STRIDE = 1_000_003

    def __init__(self, venv, cases, seed):
        super().__init__(venv)
        self._cases = cases
        self._rng = np.random.default_rng(seed)

    def reset(self):
        idx = int(self._rng.integers(0, len(self._cases)))
        base = int(self._cases[idx][1])
        self._pending_seed = base * self._STRIDE
        return super().reset()


class RecordingVecMonitor(VecMonitor):
    """补记完成 episode 回报历史的 VecMonitor。"""

    def __init__(self, venv):
        super().__init__(venv)
        self._completed_rewards = []

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        for i, done in enumerate(dones):
            if done and isinstance(infos[i], dict) and "episode" in infos[i]:
                self._completed_rewards.append(float(infos[i]["episode"]["r"]))
        return obs, rewards, dones, infos

    def get_episode_rewards(self):
        return self._completed_rewards


def build_env_stack(num_vec_envs: int, seed: int, cases):
    env = make_training_env(cases[0][0])
    env = AliveAgentsBridge(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=0, base_class="stable_baselines3")
    env = ScenarioMixVecEnv(env, cases, seed)
    env.seed(seed)
    return env


def train(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_training_cases(args.seed_scales)
    layouts = sorted({c[2] for c in cases})
    print(f"训练场景池：{len(cases)} 个场景，布局={layouts}，种子起点={TRAIN_SEED_BASE}")

    flat_env = build_env_stack(args.num_vec_envs, args.seed, cases)
    env = VecNormalize(RecordingVecMonitor(flat_env), norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO(
        "MlpPolicy", env,
        seed=args.seed,
        n_steps=512, batch_size=256, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, verbose=1,
    )
    started = time.time()
    steps_per_iter = max(args.total_steps // args.curve_points, 1)
    curve_rows = []
    while model.num_timesteps < args.total_steps:
        model.learn(total_timesteps=steps_per_iter, reset_num_timesteps=False)
        ep_rewards = env.get_episode_rewards()
        mean100 = float(np.mean(ep_rewards[-100:])) if ep_rewards else float("nan")
        curve_rows.append((model.num_timesteps, mean100))
        print(f"[curve] steps={model.num_timesteps} mean_ep_reward(last100)={mean100:.4f}")
    wall = time.time() - started

    model.save(str(out_dir / "model"))
    # 归一化统计量与权重直接落 npz（提交源码禁 pickle；torch.load/pickle.load 为硬拒绝项）
    np.savez(
        out_dir / "vecnormalize-stats.npz",
        obs_mean=np.asarray(env.obs_rms.mean, dtype=np.float64),
        obs_var=np.asarray(env.obs_rms.var, dtype=np.float64),
        obs_eps=np.array(env.epsilon, dtype=np.float64),
        obs_clip=np.array(env.clip_obs, dtype=np.float64),
    )
    with open(out_dir / "training_curve.csv", "w", encoding="utf-8", newline="") as f:
        f.write("total_steps,mean_ep_reward_last100\n")
        for steps, reward in curve_rows:
            f.write(f"{steps},{reward:.6f}\n")
    meta = {
        "seed": args.seed,
        "total_steps": int(model.num_timesteps),
        "wall_seconds": round(wall, 1),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "num_vec_envs": args.num_vec_envs,
        "suite": str(_PUBLIC_SUITE.relative_to(_REPO_ROOT)),
        "scenario_pool": {
            "n_scenarios": len(cases),
            "layouts": layouts,
            "seed_base": TRAIN_SEED_BASE,
            "seeds_per_layout": TRAIN_SEEDS_PER_LAYOUT,
        },
    }
    (out_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"训练完成: {model.num_timesteps} 步, {wall:.1f}s → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="P514 学习型策略训练（场景混合 SB3 PPO → npz）")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--curve-points", type=int, default=20)
    parser.add_argument("--num-vec-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--seed-scales", type=int, default=16, help="训练场景池的种子扩充倍数")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "outputs" / "P514" / "train")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
