"""
Complete Retrain Script for PPO and Hybrid Models
- Uses expert_utils for centralized logic
- Forced BC Retraining with LARGER network [256, 256]
- Tuned PPO hyperparameters to prevent Catastrophic Forgetting
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
from smart_home_env import SmartHomeEnv
from device_config import ROOM_OCCUPANCY_HOURS, EV_CONFIG, DEVICE_CONFIG, THERMAL_CONSTANTS
from expert_utils import expert_heuristic_action, is_room_occupied

# =====================================================
# Constants
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ACTION_DIM = 7
OBS_DIM = 13

# Network size - UPGRADED from 64 to 256
HIDDEN_SIZE = 256

# =====================================================
# BC (Behavior Cloning) Training Section
# =====================================================
class BCPolicyPPOCompat(nn.Module):
    """
    BC Policy with LARGER Network [256, 256] to match PPO policy_kwargs
    """
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.mlp_extractor = nn.ModuleDict({
            'policy_net': nn.Sequential(
                nn.Linear(obs_dim, HIDDEN_SIZE),    # 256
                nn.Tanh(),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), # 256
                nn.Tanh(),
            )
        })
        self.action_net = nn.Linear(HIDDEN_SIZE, action_dim)
        
    def forward(self, x):
        features = self.mlp_extractor['policy_net'](x)
        return self.action_net(features)

def collect_expert_data(n_episodes=50):
    """Collect expert demonstrations using Centralized Expert Logic"""
    obs_list, act_list = [], []
    price = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)
    pv = np.zeros(24)

    config = {'sim_steps': 24}

    print(f"  Collecting {n_episodes} expert episodes...")
    for ep in range(n_episodes):
        day = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365))
        config["sim_start"] = day.strftime("%Y-%m-%d")

        env = SmartHomeEnv(price, pv, config)
        obs, _ = env.reset()

        for t in range(env.sim_steps):
            hour = t % 24
            expert_action = expert_heuristic_action(obs, hour, price)
            
            obs_list.append(obs.astype(np.float32))
            act_list.append(expert_action)

            obs, _, done, _, _ = env.step(expert_action)
            if done:
                break
                
    return np.stack(obs_list), np.stack(act_list)

def train_bc(X, Y, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicyPPOCompat(obs_dim=X.shape[1], action_dim=Y.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)

    print(f"  Training BC Policy [256x256]: {X.shape[0]} samples, {epochs} epochs")

    for e in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        opt.step()

    save_dict = {"model_state_dict": model.state_dict()}
    torch.save(save_dict, "bc_policy.pt")
    print(f"  ✅ Saved FRESH bc_policy.pt (256x256 network)")

def run_bc_training(episodes=100, epochs=150):
    print(f"\n{'='*60}")
    print(f"Phase 1: Behavior Cloning (LARGE NET [256,256])")
    print(f"{'='*60}")
    X, Y = collect_expert_data(n_episodes=episodes)
    train_bc(X, Y, epochs=epochs)

# =====================================================
# BC Warm-Start Utility
# =====================================================
def load_il_weights_into_ppo(model, path="bc_policy.pt"):
    if not os.path.exists(path):
        print(f"⚠ {path} not found → PPO from scratch (no warm-start)")
        return model

    ckpt = torch.load(path, map_location="cpu")
    bc_state = ckpt.get("model_state_dict", ckpt)
    ppo_state = model.policy.state_dict()
    
    copied = 0
    for k in ppo_state:
        if k in bc_state and ppo_state[k].shape == bc_state[k].shape:
            ppo_state[k] = bc_state[k]
            copied += 1

    model.policy.load_state_dict(ppo_state, strict=False)
    print(f"✅ Warm-start PPO [256x256] from BC ({copied} tensors loaded)")
    return model

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

class ProgressPrintCallback(BaseCallback):
    def __init__(self, total_timesteps, print_every=2000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            percent = 100.0 * self.num_timesteps / self.total_timesteps
            print(f"⏳ Progress: {percent:6.2f}%")
        return True

# =====================================================
# Training Functions
# =====================================================
def make_env():
    def _init():
        config = {'time_step_hours': 1.0, 'sim_steps': 24}
        env = SmartHomeEnv(None, None, config)
        env = GymCompatWrapper(env)
        env = Monitor(env)
        return env
    return _init

def train_ppo(episodes=1000):
    T = 24
    total_timesteps = T * episodes
    print(f"\n{'='*60}\n🤖 Training PPO Model (Baseline, LARGE NET [256,256])\n{'='*60}")
    
    vec_env = DummyVecEnv([make_env()])
    
    # UPGRADED: Use 256x256 network
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=0,
        policy_kwargs=policy_kwargs,  # LARGE NET
    )
    # PPO Baseline = Random Init (No warm-start)
    
    logger = EpisodeRewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=[logger, ProgressPrintCallback(total_timesteps)])
    
    model.save("ppo_smart_home")
    np.save("ppo_baseline_rewards.npy", logger.episode_rewards)
    return logger.episode_rewards

def train_hybrid(episodes=1000):
    T = 24
    total_timesteps = T * episodes
    print(f"\n{'='*60}\n🧠 Training Hybrid PPO Model (LARGE NET [256,256] + BC Init)\n{'='*60}")
    print("Strategy: Low LR to preserve Expert Knowledge (Fine-tuning)")

    vec_env = DummyVecEnv([make_env()])
    
    # UPGRADED: Use 256x256 network to match BC
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=5e-5,    # Low LR for fine-tuning
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,        # Conservative updates
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,  # LARGE NET - MUST MATCH BC
        verbose=0,
    )

    # Load FRESH BC weights (now 256x256)
    model = load_il_weights_into_ppo(model, path="bc_policy.pt")

    logger = EpisodeRewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=[logger, ProgressPrintCallback(total_timesteps)])

    model.save("ppo_hybrid_smart_home")
    np.save("ppo_hybrid_rewards.npy", logger.episode_rewards)
    return logger.episode_rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--bc-episodes", type=int, default=2000)  # More data for larger net
    parser.add_argument("--bc-epochs", type=int, default=500)
    args = parser.parse_args()

    # ALWAYS RUN BC Phase to ensure file is fresh
    run_bc_training(episodes=args.bc_episodes, epochs=args.bc_epochs)
    
    train_ppo(episodes=args.episodes)
    train_hybrid(episodes=args.episodes)
    
    print("\nTRAINING COMPLETE! Models saved (256x256 network).")

if __name__ == "__main__":
    main()
