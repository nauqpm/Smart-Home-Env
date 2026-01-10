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

import sys
import gymnasium as gym

# Add parent directory (backend) to sys.path to resolve 'simulation' and 'algorithms'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Updated imports for new structure
from simulation.smart_home_env import SmartHomeEnv
from simulation.device_config import ROOM_OCCUPANCY_HOURS, EV_CONFIG, DEVICE_CONFIG, THERMAL_CONSTANTS
from algorithms.expert_utils import expert_heuristic_action, is_room_occupied
from algorithms.hybrid_training_env import HybridTrainingEnvWrapper
# Reuse LBWO logic from train_il_bc
from training.train_il_bc import collect_expert_data, train_bc, BCPolicyPPOCompat

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
# Logic moved to training/train_il_bc.py to support LBWO
# functions collect_expert_data() and train_bc() are imported above

def run_bc_training(episodes=100, epochs=150):
    print(f"\n{'='*60}")
    print(f"Phase 1: Behavior Cloning (LARGE NET [256,256])")
    print(f"{'='*60}")
    print(f"{'='*60}")
    # Pass default config for simulation parameters
    default_cfg = {'sim_steps': 24} 
    X, Y = collect_expert_data(default_cfg, n_episodes=episodes)
    train_bc(X, Y, epochs=epochs)

# =====================================================
# BC Warm-Start Utility
# =====================================================
def load_il_weights_into_ppo(model, path="models/bc_policy.pt"):
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
    """Standard environment for PPO Baseline training."""
    def _init():
        config = {'time_step_hours': 1.0, 'sim_steps': 24}
        env = SmartHomeEnv(None, None, config)
        env = GymCompatWrapper(env)
        env = Monitor(env)
        return env
    return _init

def make_env_hybrid():
    """Environment with Expert Rules for Hybrid training.
    
    This wrapper applies SAME expert rules as hybrid_wrapper.py during training,
    so the PPO policy learns to optimize within expert constraints.
    This fixes the training-inference mismatch issue.
    """
    def _init():
        config = {'time_step_hours': 1.0, 'sim_steps': 24}
        env = SmartHomeEnv(None, None, config)
        env = HybridTrainingEnvWrapper(env)  # Apply expert rules during training!
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
    
    model.save("models/ppo_smart_home")
    np.save("models/ppo_baseline_rewards.npy", logger.episode_rewards)
    return logger.episode_rewards

def train_hybrid(episodes=1000):
    T = 24
    total_timesteps = T * episodes
    print(f"\n{'='*60}\n🧠 Training Hybrid PPO Model (LARGE NET [256,256] + BC Init)\n{'='*60}")
    print("Strategy: Training WITH Expert Rules (HybridTrainingEnvWrapper)")
    print("         + Low LR to preserve Expert Knowledge (Fine-tuning)")

    # KEY CHANGE: Use make_env_hybrid() which includes expert rule wrapper
    vec_env = DummyVecEnv([make_env_hybrid()])
    
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
    model = load_il_weights_into_ppo(model, path="models/bc_policy.pt")

    logger = EpisodeRewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=[logger, ProgressPrintCallback(total_timesteps)])

    model.save("models/ppo_hybrid_smart_home")
    np.save("models/ppo_hybrid_rewards.npy", logger.episode_rewards)
    return logger.episode_rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--bc-episodes", type=int, default=2000)  # More data for larger net
    parser.add_argument("--bc-epochs", type=int, default=500)
    args = parser.parse_args()

    # CLEAR OLD MODELS FIRST - ensure fresh training
    import shutil
    import glob
    models_dir = "models"
    if os.path.exists(models_dir):
        files = glob.glob(os.path.join(models_dir, "*"))
        for f in files:
            try:
                os.remove(f)
                print(f"🗑️ Deleted: {f}")
            except Exception as e:
                print(f"⚠️ Could not delete {f}: {e}")
    os.makedirs(models_dir, exist_ok=True)
    print(f"\n✅ Cleared models folder. Starting fresh training...\n")

    # ALWAYS RUN BC Phase to ensure file is fresh
    run_bc_training(episodes=args.bc_episodes, epochs=args.bc_epochs)
    
    train_ppo(episodes=args.episodes)
    train_hybrid(episodes=args.episodes)
    
    print("\nTRAINING COMPLETE! Models saved (256x256 network).")

if __name__ == "__main__":
    main()
