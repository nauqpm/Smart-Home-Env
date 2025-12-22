"""
Quick retrain PPO and Hybrid models to match new observation space (13 dims)
- Reduced epochs (1000) for quick testing
- Increase to 10000+ for production
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
from smart_home_env import SmartHomeEnv


class GymCompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            return obs
        return obs, {}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            return out
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, done, False, info
        raise ValueError("Bad step() format")


class ProgressPrintCallback(BaseCallback):
    def __init__(self, total_timesteps, print_every=2000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            percent = 100.0 * self.num_timesteps / self.total_timesteps
            print(f"⏳ Progress: {percent:6.2f}% ({self.num_timesteps}/{self.total_timesteps})")
        return True


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


def make_env():
    def _init():
        config = {
            'time_step_hours': 1.0,
            'sim_start': '2025-01-01',
            'sim_steps': 24,
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
            'behavior': {'residents': [], 'must_run_base': 0.15}
        }
        env = SmartHomeEnv(None, None, config)
        env = GymCompatWrapper(env)
        env = Monitor(env)
        return env
    return _init


def train_ppo(episodes=1000):
    """Train PPO model"""
    T = 24
    total_timesteps = T * episodes

    print(f"\n{'='*60}")
    print(f"🤖 Training PPO Model")
    print(f"   Observation Space: 13 dimensions (NEW with room temps)")
    print(f"   Action Space: 7 dimensions")
    print(f"   Episodes: {episodes} ({total_timesteps} timesteps)")
    print(f"{'='*60}\n")

    vec_env = DummyVecEnv([make_env()])

    reward_logger = EpisodeRewardLogger()
    progress = ProgressPrintCallback(total_timesteps)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, progress],
    )

    model.save("ppo_smart_home")
    np.save("ppo_baseline_rewards.npy", reward_logger.episode_rewards)
    print(f"\n✅ PPO model saved to ppo_smart_home.zip")
    
    return reward_logger.episode_rewards


def train_hybrid(episodes=1000):
    """Train Hybrid PPO model"""
    T = 24
    total_timesteps = T * episodes

    print(f"\n{'='*60}")
    print(f"🧠 Training Hybrid PPO Model")
    print(f"   Observation Space: 13 dimensions (NEW with room temps)")
    print(f"   Action Space: 7 dimensions")
    print(f"   Episodes: {episodes} ({total_timesteps} timesteps)")
    print(f"{'='*60}\n")

    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    reward_logger = EpisodeRewardLogger()
    progress = ProgressPrintCallback(total_timesteps)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=0,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, progress],
    )

    model.save("ppo_hybrid_smart_home")
    np.save("ppo_hybrid_rewards.npy", reward_logger.episode_rewards)
    print(f"\n✅ Hybrid model saved to ppo_hybrid_smart_home.zip")
    
    return reward_logger.episode_rewards


def main():
    # Full training (10000 episodes) - takes ~20-30 minutes
    EPISODES = 10000  # Production quality training

    print("\n" + "="*60)
    print("🚀 RETRAINING MODELS FOR NEW OBSERVATION SPACE (13 dims)")
    print("="*60)
    
    # Check observation space
    env = SmartHomeEnv(None, None, {"sim_steps": 24})
    obs, _ = env.reset()
    print(f"\n📐 Current observation space: {env.observation_space.shape}")
    print(f"📐 Sample observation shape: {obs.shape}")
    
    # Train PPO
    ppo_rewards = train_ppo(episodes=EPISODES)
    
    # Train Hybrid
    hybrid_rewards = train_hybrid(episodes=EPISODES)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ppo_rewards)
    plt.title("PPO Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(hybrid_rewards)
    plt.title("Hybrid PPO Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_rewards.png")
    print(f"\n📊 Training plot saved to training_rewards.png")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("   - ppo_smart_home.zip")
    print("   - ppo_hybrid_smart_home.zip")
    print("="*60)
    print("\nNow restart the backend server: python ws_server.py")


if __name__ == "__main__":
    main()
