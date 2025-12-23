"""
Quick retrain PPO and Hybrid models to match new observation space (13 dims)
- Reduced epochs (1000) for quick testing
- Increase to 10000+ for production
- Hybrid uses BC warm-start from bc_policy.pt
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
        ent_coef=0.01,
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
