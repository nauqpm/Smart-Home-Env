"""
Hybrid IL + PPO Training for SmartHomeEnv
- NO tqdm
- Progress printed as percentage
- Warm-start PPO from BC (if available)
- VecNormalize for stability
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
from smart_home_env import SmartHomeEnv


# ===============================
# Gymnasium compatibility wrapper
# ===============================
class GymCompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs if isinstance(obs, tuple) else (obs, {})

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            return out
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, done, False, info
        raise ValueError("Bad step() format")


# ===============================
# Episode reward logger
# ===============================
class EpisodeRewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None:
            self.current_reward += float(rewards[0])

        if dones is not None and dones[0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0

        return True


# ===============================
# Progress callback (PRINT %, no tqdm)
# ===============================
class ProgressPrintCallback(BaseCallback):
    def __init__(self, total_timesteps, print_every=5000):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.print_every = int(print_every)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            percent = 100.0 * self.num_timesteps / self.total_timesteps
            print(
                f"⏳ Progress: {percent:6.2f}% "
                f"({self.num_timesteps}/{self.total_timesteps})"
            )
        return True


# ===============================
# Load IL → PPO warm start
# ===============================
def load_il_weights_into_ppo(model: PPO, bc_path="bc_policy.pt"):
    if not os.path.exists(bc_path):
        print("⚠ bc_policy.pt not found — training PPO from scratch.")
        return model

    print(f"🔵 Loading IL weights from {bc_path}")
    ckpt = torch.load(bc_path, map_location="cpu")
    bc_state = ckpt.get("model_state_dict", ckpt)

    ppo_state = model.policy.state_dict()
    copied = 0

    for k in ppo_state:
        if k in bc_state and ppo_state[k].shape == bc_state[k].shape:
            ppo_state[k] = bc_state[k]
            copied += 1

    model.policy.load_state_dict(ppo_state, strict=False)
    print(f"✅ Warm-start done — copied tensors: {copied}")
    return model


# ===============================
# Environment factory
# ===============================
def make_env():
    def _init():
        T = 24
        price = 0.1 + 0.2 * np.random.rand(T)
        pv = np.clip(
            1.4 * np.sin(np.linspace(0, np.pi, T)) +
            0.2 * np.random.randn(T),
            0, None
        )

        config = {
            "critical": [0.3] * T,
            "adjustable": [
                {"P_min": 0.1, "P_max": 1.5, "P_com": 1.2, "alpha": 0.06},
                {"P_min": 0.0, "P_max": 1.2, "P_com": 1.0, "alpha": 0.12},
            ],
            "shiftable_su": [
                {"rate": 0.5, "L": 2, "t_s": 6, "t_f": 20},
                {"rate": 0.6, "L": 1, "t_s": 8, "t_f": 22},
            ],
            "shiftable_si": [
                {"rate": 1.0, "E": 4.0, "t_s": 0, "t_f": 23},
            ],
            "beta": 0.5,
            "battery": {
                "capacity_kwh": 6.0,
                "soc0": 0.5,
                "soc_min": 0.1,
                "soc_max": 0.9,
            },
            "reward_mode": "advanced",
        }

        env = SmartHomeEnv(price, pv, config)
        env = GymCompatWrapper(env)
        env = Monitor(env)
        return env

    return _init


# ===============================
# MAIN TRAINING
# ===============================
def main():
    T = 24
    episodes_to_train = 10000
    total_timesteps = T * episodes_to_train

    print(f"\nHybrid IL + PPO")
    print(f"Training for {episodes_to_train} episodes "
          f"({total_timesteps} timesteps)\n")

    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    reward_logger = EpisodeRewardLogger()
    progress = ProgressPrintCallback(
        total_timesteps=total_timesteps,
        print_every=5000
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        verbose=0
    )

    model = load_il_weights_into_ppo(model, "bc_policy.pt")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, progress]
    )

    model.save("ppo_hybrid_smart_home")
    np.save(
        "ppo_hybrid_rewards.npy",
        np.array(reward_logger.episode_rewards)
    )

    print("\n✅ Training finished.")
    print("Saved model: ppo_hybrid_smart_home")
    print("Saved rewards: ppo_hybrid_rewards.npy")

    # ===== Plot rewards =====
    rewards = reward_logger.episode_rewards
    if rewards:
        plt.figure(figsize=(9, 4.5))
        plt.plot(rewards, label="Episode reward")

        window = min(200, max(1, len(rewards) // 20))
        if window > 1:
            ma = np.convolve(
                rewards,
                np.ones(window) / window,
                mode="valid"
            )
            plt.plot(
                range(window - 1, window - 1 + len(ma)),
                ma,
                label=f"MA (window={window})",
                linewidth=2,
            )

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Hybrid PPO (BC warm-start) — Episode rewards")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ No episode rewards logged.")

    vec_env.close()


if __name__ == "__main__":
    main()
