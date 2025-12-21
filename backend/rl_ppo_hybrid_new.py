"""
Hybrid IL + PPO Training for SmartHomeEnv
- Progress printed as percentage (NO tqdm)
- Reward plot kept
- Warm-start PPO from bc_policy.pt
- Compatible with SmartHomeEnv (7 actions)
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
from device_config import ROOM_OCCUPANCY_HOURS, EV_CONFIG, DEVICE_CONFIG


# ======================================================
# Gym compatibility
# ======================================================
class GymCompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs if isinstance(obs, tuple) else (obs, {})

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info


# ======================================================
# Hybrid Agent Wrapper (FIXED for SmartHomeEnv)
# ======================================================
class HybridAgentWrapper:
    """
    PPO + rule-based override
    Action order (FIXED):
    [0] battery, [1] ac_living, [2] ac_master, [3] ac_bed2,
    [4] ev, [5] wm, [6] dw
    """

    def __init__(self, ppo_model):
        self.model = ppo_model

    def predict(self, obs, env_state=None, deterministic=True):
        action, states = self.model.predict(obs, deterministic=deterministic)
        a = np.array(action, dtype=np.float32).flatten()

        if env_state is None:
            return np.clip(a, -1, 1), states

        hour = env_state.get("hour", 12)
        soc = env_state.get("soc", 0.5)
        ev_soc = env_state.get("ev_soc", 0.5)
        n_home = env_state.get("n_home", 0)

        # -------- EV off-peak charging --------
        if (hour >= 22 or hour < 4) and ev_soc < 0.9:
            a[4] = 1.0

        # -------- EV deadline --------
        deadline = EV_CONFIG["deadline_hour"]
        hours_left = (deadline - hour) % 24
        hours_left = max(1, hours_left)

        capacity = DEVICE_CONFIG["shiftable"]["ev"]["capacity"]
        pmax = DEVICE_CONFIG["shiftable"]["ev"]["power_max"]
        need = (EV_CONFIG["min_target_soc"] - ev_soc) * capacity

        if need > 0 and need > pmax * hours_left * 0.8:
            a[4] = 1.0

        # -------- Battery safety --------
        if soc < 0.15:
            a[0] = max(0.0, a[0])

        # -------- AC off if no one home --------
        if n_home == 0:
            a[1:4] = -1.0
        else:
            for idx, room in zip([1, 2, 3], ["living", "master", "bed2"]):
                if not self._is_room_occupied(room, hour):
                    a[idx] = min(a[idx], -0.3)

        return np.clip(a, -1, 1), states

    def _is_room_occupied(self, room, hour):
        for start, end in ROOM_OCCUPANCY_HOURS.get(room, []):
            if start <= end and start <= hour < end:
                return True
            if start > end and (hour >= start or hour < end):
                return True
        return False


# ======================================================
# Reward logger
# ======================================================
class EpisodeRewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True


# ======================================================
# Progress printer (UNCHANGED)
# ======================================================
class ProgressPrintCallback(BaseCallback):
    def __init__(self, total_timesteps, print_every=5000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            pct = 100 * self.num_timesteps / self.total_timesteps
            print(f"⏳ Progress: {pct:6.2f}% ({self.num_timesteps}/{self.total_timesteps})")
        return True


# ======================================================
# Load BC → PPO
# ======================================================
def load_il_weights_into_ppo(model, path="bc_policy.pt"):
    if not os.path.exists(path):
        print("⚠ bc_policy.pt not found → PPO from scratch")
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
    print(f"✅ Warm-start PPO from BC ({copied} tensors)")
    return model


# ======================================================
# Env factory
# ======================================================
def make_env():
    def _init():
        env = SmartHomeEnv(None, None, {"sim_steps": 24})
        env = GymCompatWrapper(env)
        env = Monitor(env)
        return env

    return _init


# ======================================================
# MAIN
# ======================================================
def main():
    episodes = 10000
    T = 24
    total_steps = episodes * T

    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    reward_logger = EpisodeRewardLogger()
    progress = ProgressPrintCallback(total_steps)

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

    model = load_il_weights_into_ppo(model)

    model.learn(total_timesteps=total_steps, callback=[reward_logger, progress])
    model.save("ppo_hybrid_smart_home")
    np.save("ppo_hybrid_rewards.npy", reward_logger.episode_rewards)

    # -------- Plot (UNCHANGED) --------
    rewards = reward_logger.episode_rewards
    plt.figure(figsize=(9, 4.5))
    plt.plot(rewards, label="Episode reward")
    if len(rewards) > 50:
        ma = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        plt.plot(range(99, 99 + len(ma)), ma, label="MA(100)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
