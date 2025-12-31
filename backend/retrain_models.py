"""
Complete Retrain Script for PPO and Hybrid Models
- Includes BC (Behavior Cloning) training for Hybrid warm-start
- Reduced epochs (1000) for quick testing
- Increase to 10000+ for production
- Hybrid uses BC warm-start from bc_policy.pt
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
from smart_home_env import SmartHomeEnv
from device_config import ROOM_OCCUPANCY_HOURS, EV_CONFIG, DEVICE_CONFIG, THERMAL_CONSTANTS

# =====================================================
# Constants
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ACTION_DIM = 7
OBS_DIM = 13


# =====================================================
# BC (Behavior Cloning) Training Section
# =====================================================
class BCPolicyPPOCompat(nn.Module):
    """
    BC Policy with PPO-compatible layer names.
    Uses EXACT same key names as SB3's MlpPolicy for warm-start.
    """
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        super().__init__()
        
        self.mlp_extractor = nn.ModuleDict({
            'policy_net': nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )
        })
        
        self.action_net = nn.Linear(64, action_dim)
        
    def forward(self, x):
        features = self.mlp_extractor['policy_net'](x)
        return self.action_net(features)


def expert_heuristic_action(obs, hour, price):
    """Generate expert action for all 7 dimensions."""
    action = np.zeros(7, dtype=np.float32)
    
    soc = obs[0]
    n_home = obs[6]
    room_temps = obs[8:11]
    ev_soc = obs[12]
    
    # Battery: Charge when cheap, discharge when expensive
    if price[hour] < 0.12:
        action[0] = 0.8 if soc < 0.8 else 0.0
    elif price[hour] > 0.20:
        action[0] = -0.8 if soc > 0.3 else 0.0
    else:
        action[0] = 0.0
    
    # ACs: Turn on when occupied and hot
    comfort_temp = THERMAL_CONSTANTS["comfort_temp"]
    for idx, room in enumerate(["living", "master", "bed2"]):
        if is_room_occupied(room, hour) and n_home > 0:
            temp_diff = room_temps[idx] - comfort_temp
            if temp_diff > 2:
                action[1 + idx] = min(1.0, temp_diff / 5)
            elif temp_diff < -2:
                action[1 + idx] = -0.5
            else:
                action[1 + idx] = 0.3
        else:
            action[1 + idx] = -1.0
    
    # EV: Charge during off-peak or when deadline approaching
    deadline = EV_CONFIG["deadline_hour"]
    target_soc = EV_CONFIG["min_target_soc"]
    hours_left = (deadline - hour) % 24
    hours_left = max(1, hours_left)
    
    if (hour >= 22 or hour < 4) and ev_soc < 0.9:
        action[4] = 1.0
    elif ev_soc < target_soc and hours_left < 6:
        action[4] = 1.0
    elif ev_soc < 0.5 and price[hour] < 0.15:
        action[4] = 0.7
    else:
        action[4] = -1.0
    
    # WM/DW: Run during off-peak hours
    if 0 <= hour < 6 or 22 <= hour < 24:
        action[5] = 1.0
        action[6] = 1.0
    else:
        action[5] = -1.0
        action[6] = -1.0
    
    return np.clip(action, -1, 1)


def collect_expert_data(n_episodes=50):
    """Collect expert demonstrations for all 7 actions"""
    obs_list, act_list = [], []
    price = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)
    pv = np.zeros(24)

    config = {'sim_steps': 24}

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

        if (ep + 1) % 10 == 0:
            print(f"  Collected episode {ep + 1}/{n_episodes}")

    return np.stack(obs_list), np.stack(act_list)


def train_bc(X, Y, epochs=100):
    """Train BC policy with PPO-compatible architecture"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicyPPOCompat(obs_dim=X.shape[1], action_dim=Y.shape[1]).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)

    print(f"  Training BC Policy: {X.shape[0]} samples, {epochs} epochs")

    for e in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        opt.step()

        if (e + 1) % 50 == 0:
            print(f"    Epoch {e+1}/{epochs} | Loss {loss.item():.4f}")

    save_dict = {"model_state_dict": model.state_dict()}
    torch.save(save_dict, "bc_policy.pt")
    print(f"  Saved bc_policy.pt")


def run_bc_training(episodes=100, epochs=150):
    """Run complete BC training pipeline"""
    print(f"\n{'='*60}")
    print(f"Phase 1: Behavior Cloning (IL) Training")
    print(f"{'='*60}")
    
    X, Y = collect_expert_data(n_episodes=episodes)
    print(f"  Dataset: X={X.shape}, Y={Y.shape}")
    
    train_bc(X, Y, epochs=epochs)
    print(f"  BC training complete!")


# =====================================================
# BC Warm-Start: Load Imitation Learning weights into PPO
# =====================================================
def load_il_weights_into_ppo(model, path="bc_policy.pt"):
    """
    Load pre-trained Behavior Cloning weights into PPO model.
    This gives Hybrid a "head start" with expert knowledge.
    """
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
    print(f"✅ Warm-start PPO from BC ({copied} tensors loaded from {path})")
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


# =====================================================
# Rule-Based Environment Wrapper for Hybrid Training
# =====================================================
from device_config import ROOM_OCCUPANCY_HOURS, EV_CONFIG, DEVICE_CONFIG

def is_room_occupied(room, hour):
    """Check if room is occupied at given hour"""
    for start, end in ROOM_OCCUPANCY_HOURS.get(room, []):
        if start <= end and start <= hour < end:
            return True
        if start > end and (hour >= start or hour < end):
            return True
    return False


class RuleBasedEnvWrapper(gym.Wrapper):
    """
    Wrapper that applies Hybrid rules to actions BEFORE stepping the environment.
    This allows PPO to learn in an environment where rules are already applied,
    matching inference behavior exactly.
    
    Action order: [battery, ac_living, ac_master, ac_bed2, ev, wm, dw]
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        
    def reset(self, **kwargs):
        self.step_count = 0
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            return obs
        return obs, {}
    
    def step(self, action):
        # Apply rules to action before stepping
        modified_action = self._apply_rules(action)
        
        # Step with modified action
        out = self.env.step(modified_action)
        self.step_count += 1
        
        if len(out) == 5:
            return out
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, done, False, info
        raise ValueError("Bad step() format")
    
    def _apply_rules(self, action):
        """Apply Hybrid rules to action"""
        a = np.array(action, dtype=np.float32).flatten().copy()
        
        # Get current state from environment
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        
        hour = self.step_count % 24
        soc = getattr(env, 'soc', 0.5)
        ev_soc = getattr(env, 'ev_soc', 0.5)
        
        # Get n_home from load schedules
        n_home = 0
        if hasattr(env, 'load_schedules') and self.step_count < len(env.load_schedules):
            n_home = env.load_schedules[self.step_count].get('n_home', 0)
        
        # -------- Rule 1: EV off-peak charging --------
        if (hour >= 22 or hour < 4) and ev_soc < 0.9:
            a[4] = 1.0
        
        # -------- Rule 2: EV deadline enforcement --------
        deadline = EV_CONFIG["deadline_hour"]
        hours_left = (deadline - hour) % 24
        hours_left = max(1, hours_left)
        
        capacity = DEVICE_CONFIG["shiftable"]["ev"]["capacity"]
        pmax = DEVICE_CONFIG["shiftable"]["ev"]["power_max"]
        need = (EV_CONFIG["min_target_soc"] - ev_soc) * capacity
        
        if need > 0 and need > pmax * hours_left * 0.8:
            a[4] = 1.0
        
        # -------- Rule 3: Battery safety --------
        if soc < 0.15:
            a[0] = max(0.0, a[0])
        
        # -------- Rule 4: AC off if no one home --------
        if n_home == 0:
            a[1:4] = -1.0
        else:
            for idx, room in zip([1, 2, 3], ["living", "master", "bed2"]):
                if not is_room_occupied(room, hour):
                    a[idx] = min(a[idx], -0.3)
        
        return np.clip(a, -1, 1)


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
    """Create standard environment for PPO training"""
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


def make_hybrid_env():
    """Create environment with rules for Hybrid training"""
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
        env = RuleBasedEnvWrapper(env)  # <-- Apply rules during training!
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
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Higher entropy for more exploration
        vf_coef=0.5,
    )
    
    # BC warm-start for PPO too - helps avoid starting from random policy
    model = load_il_weights_into_ppo(model, path="bc_policy.pt")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, progress],
    )

    model.save("ppo_smart_home")
    np.save("ppo_baseline_rewards.npy", reward_logger.episode_rewards)
    print(f"\n PPO model saved to ppo_smart_home.zip")
    
    return reward_logger.episode_rewards


def train_hybrid(episodes=1000):
    """Train Hybrid PPO model with rules applied during training"""
    T = 24
    total_timesteps = T * episodes

    print(f"\n{'='*60}")
    print(f"🧠 Training Hybrid PPO Model (with RULES)")
    print(f"   Observation Space: 13 dimensions (NEW with room temps)")
    print(f"   Action Space: 7 dimensions")
    print(f"   Episodes: {episodes} ({total_timesteps} timesteps)")
    print(f"   Training Mode: Rules applied DURING training")
    print(f"   BC Warm-Start: bc_policy.pt")
    print(f"{'='*60}\n")

    # Use make_hybrid_env which has RuleBasedEnvWrapper!
    vec_env = DummyVecEnv([make_hybrid_env()])
    # Note: No VecNormalize to match inference environment

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
        ent_coef=0.05,  # Higher entropy for more exploration
        vf_coef=0.5,
        verbose=0,
    )

    # ===== BC WARM-START: Load pre-trained weights =====
    model = load_il_weights_into_ppo(model, path="bc_policy.pt")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, progress],
    )

    model.save("ppo_hybrid_smart_home")
    np.save("ppo_hybrid_rewards.npy", reward_logger.episode_rewards)
    print(f"\n✅ Hybrid model saved to ppo_hybrid_smart_home.zip")
    
    return reward_logger.episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Complete retraining script for Smart Home RL models")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes for PPO training")
    parser.add_argument("--bc-episodes", type=int, default=100, help="Number of episodes for BC data collection")
    parser.add_argument("--bc-epochs", type=int, default=150, help="Number of epochs for BC training")
    parser.add_argument("--skip-bc", action="store_true", help="Skip BC training if bc_policy.pt exists")
    args = parser.parse_args()

    EPISODES = args.episodes

    print("\n" + "="*60)
    print("COMPLETE RETRAINING PIPELINE")
    print("  Phase 1: Behavior Cloning (BC) for Hybrid warm-start")
    print("  Phase 2: PPO Training")
    print("  Phase 3: Hybrid PPO Training")
    print("="*60)
    
    # Check observation space
    env = SmartHomeEnv(None, None, {"sim_steps": 24})
    obs, _ = env.reset()
    print(f"\nCurrent observation space: {env.observation_space.shape}")
    print(f"Sample observation shape: {obs.shape}")
    
    # Phase 1: BC Training
    if args.skip_bc and os.path.exists("bc_policy.pt"):
        print(f"\n[Phase 1] Skipping BC training - bc_policy.pt already exists")
    else:
        run_bc_training(episodes=args.bc_episodes, epochs=args.bc_epochs)
    
    # Phase 2: Train PPO
    print(f"\n{'='*60}")
    print(f"Phase 2: PPO Training ({EPISODES} episodes)")
    print(f"{'='*60}")
    ppo_rewards = train_ppo(episodes=EPISODES)
    
    # Phase 3: Train Hybrid
    print(f"\n{'='*60}")
    print(f"Phase 3: Hybrid PPO Training ({EPISODES} episodes)")
    print(f"{'='*60}")
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
    print(f"\nTraining plot saved to training_rewards.png")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("   - bc_policy.pt (BC warm-start weights)")
    print("   - ppo_smart_home.zip")
    print("   - ppo_hybrid_smart_home.zip")
    print("="*60)
    print("\nNow restart the backend server: python ws_server.py")


if __name__ == "__main__":
    main()
