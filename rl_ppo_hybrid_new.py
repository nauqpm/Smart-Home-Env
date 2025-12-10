"""
Hybrid IL + PPO Training for SmartHomeEnv (optimized)
- Warm-start PPO policy from bc_policy.pt (if available)
- Use VecNormalize for stable training
- Tuned PPO hyperparameters for faster convergence
- Save rewards (.npy) and plot after training
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is not None:
            self.current_reward += float(rewards[0])
        if dones is not None and dones[0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True


# ===============================
# Progress callback (silent pbar)
# ===============================
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.pbar = None
        self.prev_steps = 0

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training PPO",
            ncols=90,
            mininterval=0.3,
            smoothing=0.1,
            leave=True
        )

    # SB3 bắt buộc phải có
    def _on_step(self):
        return True

    def _on_rollout_end(self):
        # Update thanh tiến trình sau mỗi rollout
        new_steps = self.num_timesteps - self.prev_steps
        if self.pbar is not None:
            self.pbar.update(max(1, new_steps))
        self.prev_steps = self.num_timesteps

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

# ===============================
# Load IL -> PPO warm start (robust)
# ===============================
def load_il_weights_into_ppo(model: PPO, bc_path="bc_policy.pt"):
    """
    Try to warm-start parts of PPO policy using BC checkpoint.
    - Accepts: bc_checkpoint = {'model_state_dict': ... , ...} OR an OrderedDict state_dict
    - Maps matching tensor names where possible; loads with strict=False.
    """
    if not os.path.exists(bc_path):
        print("⚠ Không tìm thấy bc_policy.pt — PPO sẽ train từ đầu.")
        return model

    print(f"🔵 Loading IL weights from: {bc_path}")
    ckpt = torch.load(bc_path, map_location="cpu")

    # extract state_dict if wrapped
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        bc_state = ckpt["model_state_dict"]
    else:
        bc_state = ckpt

    ppo_state = model.policy.state_dict()
    copy_count = 0

    # Try to copy any identical keys first
    for k in list(ppo_state.keys()):
        if k in bc_state and ppo_state[k].shape == bc_state[k].shape:
            ppo_state[k] = bc_state[k]
            copy_count += 1

    # attempt some common name mappings if not direct (best-effort)
    # mapping guessed from typical BC->SB3 names (may or may not match)
    name_map_candidates = [
        ("net.0.weight", "mlp_extractor.policy_net.0.weight"),
        ("net.0.bias", "mlp_extractor.policy_net.0.bias"),
        ("net.2.weight", "mlp_extractor.policy_net.2.weight"),
        ("net.2.bias", "mlp_extractor.policy_net.2.bias"),
        ("action_head.weight", "action_net.weight"),
        ("action_head.bias", "action_net.bias"),
    ]
    for bc_k, ppo_k in name_map_candidates:
        if bc_k in bc_state and ppo_k in ppo_state and ppo_state[ppo_k].shape == bc_state[bc_k].shape:
            ppo_state[ppo_k] = bc_state[bc_k]
            copy_count += 1

    # load updated state (non-strict so missing keys allowed)
    model.policy.load_state_dict(ppo_state, strict=False)
    print(f"✅ Warm-start applied — tensors copied: {copy_count}")
    return model


# ===============================
# Environment factory (kept as your original)
# ===============================
def make_env():
    def _init():
        T = 24
        price = 0.1 + 0.2 * np.random.rand(T)
        pv = np.clip(
            1.4 * np.sin(np.linspace(0, np.pi, T)) +
            0.2 * np.random.randn(T), 0, None
        )

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
# MAIN TRAINING (optimized)
# ===============================
def main():
    T = 24
    episodes_to_train = 10000
    total_timesteps = T * episodes_to_train

    print(f"\n🔥 Hybrid IL + PPO — target: {episodes_to_train} episodes ({total_timesteps} timesteps)")

    # create vectorized env and normalize observations (stabilizes training)
    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)  # keep rewards raw
    reward_logger = EpisodeRewardLogger()
    progress = ProgressCallback(total_timesteps)

    # Tuned PPO hyperparameters to improve sample efficiency / convergence
    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,           # larger rollout helps stable advantage estimates
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log="./ppo_tb/"
    )

    model = PPO(**ppo_kwargs)

    # warm-start weights if BC available (best-effort safe mapping)
    model = load_il_weights_into_ppo(model, "bc_policy.pt")

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, progress]
    )

    # Save model and reward logs
    save_name = "ppo_hybrid_smart_home"
    model.save(save_name)
    np.save("ppo_hybrid_new_rewards.npy", np.array(reward_logger.episode_rewards))
    print(f"Saved model: {save_name} , saved rewards -> ppo_hybrid_rewards.npy")

    # Plot (episode rewards)
    rewards = reward_logger.episode_rewards
    if len(rewards) > 0:
        plt.figure(figsize=(9, 4.5))
        plt.plot(rewards, label="Episode reward")
        # plot moving average for smoothing
        window = min(200, max(1, len(rewards)//20))
        if window > 1:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, window-1+len(ma)), ma, label=f"MA (window={window})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Hybrid PPO (with BC warm-start) - Episode rewards")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ No episode rewards recorded — check env/termination logic.")

    # Close vec_env
    vec_env.close()


if __name__ == "__main__":
    main()
