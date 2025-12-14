"""
Train PPO on SmartHomeEnv
- No tqdm
- Progress printed as percentage
- Episode reward logging
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
from smart_home_env import SmartHomeEnv


# ===============================
# Gymnasium compatibility wrapper
# ===============================
class GymCompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            return obs
        return obs, {}

    def step(self, action):
        out = self.env.step(action)
        # SB3 expects: obs, reward, terminated, truncated, info
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
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.current_reward += float(rewards[0])

        if dones[0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0

        return True


# ===============================
# Progress printer (NO tqdm)
# ===============================
class ProgressPrintCallback(BaseCallback):
    def __init__(self, total_timesteps, print_every=5000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            percent = 100.0 * self.num_timesteps / self.total_timesteps
            print(
                f"⏳ Progress: {percent:6.2f}% "
                f"({self.num_timesteps}/{self.total_timesteps})"
            )
        return True


# ===============================
# Environment factory
# ===============================
def make_env():
    def _init():
        T = 24

        price = 0.1 + 0.2 * np.random.rand(T)
        pv = np.clip(
            1.4 * np.sin(np.linspace(0, np.pi, T))
            + 0.2 * np.random.randn(T),
            0,
            None,
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
                {"rate": 1.0, "E": 4.0, "t_s": 0, "t_f": 23}
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
# Main training function
# ===============================
def main():
    T = 24
    episodes_to_train = 10000
    total_timesteps = T * episodes_to_train

    print(
        f"\nTraining PPO for {episodes_to_train} episodes "
        f"({total_timesteps} timesteps)\n"
    )

    vec_env = DummyVecEnv([make_env()])

    reward_logger = EpisodeRewardLogger()
    progress = ProgressPrintCallback(
        total_timesteps=total_timesteps,
        print_every=5000
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        tensorboard_log="./ppo_tb/",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, progress],
    )

    model.save("ppo_smart_home")
    np.save("ppo_baseline_rewards.npy", reward_logger.episode_rewards)

    # ===== Plot rewards =====
    if reward_logger.episode_rewards:
        plt.figure(figsize=(8, 5))
        plt.plot(reward_logger.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("PPO Reward per Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ No episode rewards logged — check env termination logic.")


if __name__ == "__main__":
    main()
