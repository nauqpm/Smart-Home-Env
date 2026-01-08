
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from smart_home_env import SmartHomeEnv
import gymnasium as gym
import numpy as np

class GymCompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs if isinstance(obs, tuple) else (obs, {})
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

def make_env():
    env = SmartHomeEnv(None, None, {"sim_steps": 24})
    env = GymCompatWrapper(env)
    return env

def load_il_weights_into_ppo(model, path="bc_policy.pt"):
    ckpt = torch.load(path)
    bc_state = ckpt.get("model_state_dict", ckpt)
    ppo_state = model.policy.state_dict()
    for k in ppo_state:
        if k in bc_state and ppo_state[k].shape == bc_state[k].shape:
            ppo_state[k] = bc_state[k]
    model.policy.load_state_dict(ppo_state, strict=False)
    return model

env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]), verbose=1)
model = load_il_weights_into_ppo(model, "bc_policy.pt")

# Verify In-Memory
obs = np.zeros(13)
obs[1] = 3.2
obs[7] = 25.0
act_mem, _ = model.predict(obs, deterministic=True)
print(f"In-Memory Prediction (PV=3.2): {act_mem[0]}")

model.save("ppo_hybrid_smart_home")
print("Saved ppo_hybrid_smart_home.zip with pure BC weights.")

# Verify Reload
m2 = PPO.load("ppo_hybrid_smart_home")
act_disk, _ = m2.predict(obs, deterministic=True)
print(f"Reloaded Prediction (PV=3.2): {act_disk[0]}")
