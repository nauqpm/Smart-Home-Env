"""
Hybrid IL + PPO Training for SmartHomeEnv
- NO tqdm
- Progress printed as percentage
- Warm-start PPO from BC (if available)
- VecNormalize for stability
- HybridAgentWrapper for rule-based overrides
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
from device_config import (
    DEVICE_CONFIG, ACTION_INDICES, ROOM_OCCUPANCY_HOURS, EV_CONFIG
)


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
# Hybrid Agent Wrapper
# Rule-based constraints over PPO actions
# ===============================
class HybridAgentWrapper:
    """
    Wrapper that applies rule-based constraints and overrides on top of PPO actions.
    This combines the learning capability of PPO with domain knowledge rules.
    """
    
    def __init__(self, ppo_model, device_config=None):
        self.ppo_model = ppo_model
        self.config = device_config or DEVICE_CONFIG
    
    def predict(self, obs, env_state=None, deterministic=True):
        """
        Get action from PPO and apply rule-based overrides.
        
        Args:
            obs: Observation from environment
            env_state: Dictionary with current environment state (hour, soc, ev_soc, wm_remaining, etc.)
            deterministic: Whether to use deterministic policy
            
        Returns:
            Modified action array
        """
        # 1. Get raw action from PPO
        rl_action, _states = self.ppo_model.predict(obs, deterministic=deterministic)
        rl_action = np.array(rl_action, dtype=np.float32).flatten().copy()
        
        if env_state is None:
            return rl_action, _states
        
        hour = env_state.get('hour', 12)
        
        # 2. APPLY RULE-BASED OVERRIDES
        
        # --- Rule 1: EV Charging - Off-peak optimization ---
        # If off-peak hours (22h - 4h) and EV not full -> Force charge
        is_offpeak = (22 <= hour or hour < 4)
        ev_soc = env_state.get('ev_soc', 0.5)
        
        if is_offpeak and ev_soc < 0.9:
            rl_action[ACTION_INDICES['ev']] = 1.0  # Force max charging
        
        # --- Rule 2: EV Deadline ---
        # If approaching deadline and not enough charge -> Force charge
        ev_deadline = EV_CONFIG['deadline_hour']
        if hour < ev_deadline:
            hours_to_deadline = ev_deadline - hour
        else:
            hours_to_deadline = 24 - hour + ev_deadline
        
        ev_capacity = DEVICE_CONFIG['shiftable']['ev']['capacity']
        ev_power_max = DEVICE_CONFIG['shiftable']['ev']['power_max']
        energy_needed = (EV_CONFIG['min_target_soc'] - ev_soc) * ev_capacity
        max_possible_energy = ev_power_max * hours_to_deadline
        
        if energy_needed > 0 and energy_needed > max_possible_energy * 0.8:
            rl_action[ACTION_INDICES['ev']] = 1.0  # Force max charging (urgent)
        
        # --- Rule 3: Washing Machine Deadline ---
        wm_remaining = env_state.get('wm_remaining', 0)
        wm_deadline = env_state.get('wm_deadline', 22)
        
        if wm_remaining > 0:
            hours_to_wm_deadline = wm_deadline - hour if hour < wm_deadline else 0
            if hours_to_wm_deadline <= wm_remaining:
                rl_action[ACTION_INDICES['wm']] = 1.0  # Force ON (must complete)
        
        # --- Rule 4: Dishwasher Deadline ---
        dw_remaining = env_state.get('dw_remaining', 0)
        dw_deadline = env_state.get('dw_deadline', 23)
        
        if dw_remaining > 0:
            hours_to_dw_deadline = dw_deadline - hour if hour < dw_deadline else 0
            if hours_to_dw_deadline <= dw_remaining:
                rl_action[ACTION_INDICES['dw']] = 1.0  # Force ON
        
        # --- Rule 5: Battery Safety ---
        # If battery SOC is too low, prevent discharge
        soc = env_state.get('soc', 0.5)
        if soc < 0.15:
            # Clamp battery action to prevent discharge (keep non-negative)
            rl_action[ACTION_INDICES['battery']] = max(0.0, rl_action[ACTION_INDICES['battery']])
        
        # --- Rule 6: Off-peak Battery Charging ---
        # Encourage charging during off-peak hours
        price_tier = env_state.get('price_tier', 3)
        if price_tier == 1 and soc < 0.8:  # Tier 1 = cheapest
            rl_action[ACTION_INDICES['battery']] = max(0.5, rl_action[ACTION_INDICES['battery']])
        
        # --- Rule 7: AC Efficiency - Turn off when unoccupied ---
        # If no one is in a room, reduce/turn off AC
        n_home = env_state.get('n_home', 0)
        if n_home == 0:
            # No one home - reduce all ACs significantly
            rl_action[ACTION_INDICES['ac_living']] = min(-0.5, rl_action[ACTION_INDICES['ac_living']])
            rl_action[ACTION_INDICES['ac_master']] = min(-0.5, rl_action[ACTION_INDICES['ac_master']])
            rl_action[ACTION_INDICES['ac_bed2']] = min(-0.5, rl_action[ACTION_INDICES['ac_bed2']])
        else:
            # Check individual room occupancy
            for room, ac_key in [('living', 'ac_living'), ('master', 'ac_master'), ('bed2', 'ac_bed2')]:
                if not self._is_room_occupied(room, hour):
                    # Reduce AC for unoccupied rooms
                    rl_action[ACTION_INDICES[ac_key]] = min(-0.3, rl_action[ACTION_INDICES[ac_key]])
        
        return rl_action, _states
    
    def _is_room_occupied(self, room: str, hour: int) -> bool:
        """Helper to check if room is occupied based on schedule"""
        ranges = ROOM_OCCUPANCY_HOURS.get(room, [])
        for (start, end) in ranges:
            if start <= end:
                if start <= hour < end:
                    return True
            else:
                if hour >= start or hour < end:
                    return True
        return False


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
# MAIN TRAINING
# ===============================
def main():
    T = 24
    episodes_to_train = 10000
    total_timesteps = T * episodes_to_train

    print(f"\nHybrid IL + PPO with Device-Specific Control")
    print(f"Action Space: 7 dimensions (Battery + 3 ACs + EV + WM + DW)")
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
        plt.title("Hybrid PPO (Device-Specific) — Episode rewards")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ No episode rewards logged.")

    vec_env.close()


if __name__ == "__main__":
    main()
