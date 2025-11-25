"""
Train PPO on SmartHomeEnv with progress bar (tqdm)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from smart_home_env import SmartHomeEnv

# ------------ Compatibility wrapper ------------
import gymnasium as gym

class GymCompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            return out
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, done, False, info
        raise ValueError("Unexpected step output format")

# ------------ Episode Reward Logger ------------
class EpisodeRewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos:
            for info in infos:
                if info and 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
        return True

# ------------ Progress Bar Callback ------------
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training PPO", ncols=90)

    def _on_step(self):
        # mỗi lần _on_step SB3 gọi là 1 environment step
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()

# ------------ Make Env Factory ------------
def make_env(price_profile, pv_profile, config, forecast_horizon=3):
    def _init():
        env = SmartHomeEnv(price_profile, pv_profile, config, forecast_horizon=forecast_horizon)
        env = GymCompatWrapper(env)
        env = Monitor(env)
        return env
    return _init

# ------------ Main ------------
def main():
    T = 24
    np.random.seed(0)
    price = 0.1 + 0.2 * np.random.rand(T)
    pv = np.clip(1.5 * np.sin(np.linspace(0, np.pi, T)) + 0.2 * np.random.randn(T), 0, None)

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
        'battery': {'soc0': 0.5, 'soc_min': 0.1, 'soc_max': 0.9},
        "reward_mode": "advanced"
    }

    n_envs = 1
    vec_env = DummyVecEnv([make_env(price, pv, config)])

    # Training params
    episodes_to_train = 5000
    total_timesteps = T * episodes_to_train

    print(f"Training PPO for {episodes_to_train} episodes (~{total_timesteps} timesteps)...")

    reward_logger = EpisodeRewardLogger()
    progress_bar = ProgressCallback(total_timesteps)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        tensorboard_log="./ppo_smart_home_tb/"
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, progress_bar]
    )

    model.save("ppo_smart_home")

    # Plot rewards
    if reward_logger.episode_rewards:
        plt.figure(figsize=(8, 5))
        plt.plot(reward_logger.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Episode reward")
        plt.title("Reward per Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No rewards logged. Check Monitor wrapper.")

if __name__ == "__main__":
    main()
