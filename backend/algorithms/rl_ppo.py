"""
Train PPO on SmartHomeEnv
- No tqdm
- Progress printed as percentage
- Episode reward logging
- Device-Specific Action Space (7 dimensions)
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

        # Let environment generate its own price and PV profiles
        price = None
        pv = None

        config = {
            'time_step_hours': 1.0,
            'sim_start': '2025-01-01',
            'sim_steps': T,
            'sim_freq': '1h',
            'battery': {
                'capacity_kwh': 10.0,
                'soc_init': 0.5,
                'soc_min': 0.1,
                'soc_max': 0.9,
                'p_charge_max_kw': 3.0,
                'p_discharge_max_kw': 3.0,
                'eta_ch': 0.95,
                'eta_dis': 0.95
            },
            'pv_config': {
                'latitude': 10.762622,
                'longitude': 106.660172,
                'tz': 'Asia/Ho_Chi_Minh',
                'surface_tilt': 10.0,
                'surface_azimuth': 180.0,
                'module_parameters': {'pdc0': 3.0}
            },
            'behavior': {
                'residents': [],
                'must_run_base': 0.15
            }
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

    print(f"\nTraining PPO with Device-Specific Control")
    print(f"Action Space: 7 dimensions (Battery + 3 ACs + EV + WM + DW)")
    print(f"Training for {episodes_to_train} episodes "
          f"({total_timesteps} timesteps)\n")

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
        plt.title("PPO (Device-Specific) Reward per Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ No episode rewards logged — check env termination logic.")


if __name__ == "__main__":
    main()
