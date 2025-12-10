"""
Train PPO on SmartHomeEnv with proper episode logging and progress bar
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        # ensure format: (obs, info)
        if isinstance(obs, tuple):
            return obs
        return obs, {}

    def step(self, action):
        out = self.env.step(action)
        # SB3 expects: obs, reward, done, truncated, info
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
        self.current_reward = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.current_reward += float(rewards[0])

        if dones[0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0

        return True


# ===============================
# Progress bar
# ===============================
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training PPO", ncols=90)

    def _on_step(self):
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


# ===============================
# Environment factory
# ===============================
def make_env():
    def _init():
        # random price each episode
        T = 24
        price = 0.1 + 0.2 * np.random.rand(T)
        pv = np.clip(1.4 * np.sin(np.linspace(0, np.pi, T)) +
                     0.2 * np.random.randn(T), 0, None)

        config = {
            'critical': [0.3] * T,
            'adjustable': [
                {'P_min': 0.1, 'P_max': 1.5, 'P_com': 1.2, 'alpha': 0.06},
                {'P_min': 0.0, 'P_max': 1.2, 'P_com': 1.0, 'alpha': 0.12}
            ],
            'shiftable_su': [
                {'rate': 0.5, 'L': 2, 't_s': 6, 't_f': 20},
                {'rate': 0.6, 'L': 1, 't_s': 8, 't_f': 22}
            ],
            'shiftable_si': [
                {'rate': 1.0, 'E': 4.0, 't_s': 0, 't_f': 23}
            ],
            'beta': 0.5,
            'battery': {'capacity_kwh': 6.0, 'soc0': 0.5, 'soc_min': 0.1, 'soc_max': 0.9},
            "reward_mode": "advanced"
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

    print(f"\nTraining PPO for {episodes_to_train} episodes ({total_timesteps} timesteps)...")

    vec_env = DummyVecEnv([make_env()])

    reward_logger = EpisodeRewardLogger()
    progress = ProgressCallback(total_timesteps)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        tensorboard_log="./ppo_tb/"
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, progress]
    )

    model.save("ppo_smart_home")
    np.save("ppo_baseline_rewards.npy", reward_logger.episode_rewards)

    # plot
    if len(reward_logger.episode_rewards) > 0:
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

