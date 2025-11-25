"""
Train PPO on SmartHomeEnv (user-provided env).
Usage:
    python train_ppo_smart_home.py
Requirements:
    - stable-baselines3
    - gymnasium (or gym)
    - numpy, matplotlib
    - pulp (optional, env handles fallback)
Make sure smart_home_env.py and human_behavior.py are in the same folder or in PYTHONPATH.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Import user env (adjust module name/path if needed)
# from smart_home_env import SmartHomeEnv  # nếu file tên khác hãy chỉnh lại
from smart_home_env import SmartHomeEnv

# ------------ Compatibility wrapper ------------
# Vì env của bạn trong file có vẻ trả về kiểu cũ (reset -> obs only, step -> 4-tuple),
# ta tạo wrapper để chuyển về chuẩn Gymnasium 5-tuple (obs, reward, terminated, truncated, info)
import gymnasium as gym

class GymCompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # Nếu env trả về (obs, info) (Gymnasium kiểu mới), giữ nguyên
        if isinstance(out, tuple) and len(out) == 2:
            return out
        # Nếu env trả về obs (kiểu cũ), convert
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        # Nếu env trả về 5-tuple (gymnasium), giữ nguyên
        if isinstance(out, tuple) and len(out) == 5:
            return out
        # Nếu env trả về 4-tuple (obs, reward, done, info) -> chuyển thành 5-tuple
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            # Không có truncated distinction, đặt truncated=False
            return obs, reward, done, False, info
        raise ValueError("Unexpected step return length: {}".format(len(out)))

# ------------ Callback ghi reward per episode ------------
class EpisodeRewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Trong vectored env, infos là 1 list mỗi step
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        for info in infos:
            # Monitor wrapper đặt key 'episode' khi episode kết thúc
            if info is None:
                continue
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                if self.verbose > 0:
                    print("Episode done, reward:", info['episode']['r'])
        return True

# ------------ Helper: make env factory ------------
def make_env(price_profile, pv_profile, config, forecast_horizon=3):
    def _init():
        env = SmartHomeEnv(price_profile, pv_profile, config, forecast_horizon=forecast_horizon)
        # Thêm GymCompat wrapper để tương thích return values
        env = GymCompatWrapper(env)
        # Monitor để ghi stats episode (lưu vào info['episode'])
        env = Monitor(env)
        return env
    return _init

# ------------ Main training logic ------------
def main():
    # Demo small dataset, bạn nên thay bằng price/pv thật cho training nghiêm túc
    T = 24
    np.random.seed(0)
    price = 0.1 + 0.2 * np.random.rand(T)
    pv = np.clip(1.5 * np.sin(np.linspace(0, 3.14, T)) + 0.2*np.random.randn(T), 0, None)

    config = {
        'critical': [0.3]*T,
        'adjustable': [
            {'P_min':0.1, 'P_max':1.5, 'P_com':1.2, 'alpha':0.06},
            {'P_min':0.0, 'P_max':1.2, 'P_com':1.0, 'alpha':0.12}
        ],
        'shiftable_su': [ {'rate':0.5, 'L':2, 't_s':6, 't_f':20}, {'rate':0.6, 'L':1, 't_s':8, 't_f':22} ],
        'shiftable_si': [ {'rate':1.0, 'E':4.0, 't_s':0, 't_f':23} ],
        'beta': 0.5,
        'battery': {'soc0':0.5, 'soc_min':0.1, 'soc_max':0.9},
        "reward_mode": "advanced"
    }

    # Tạo vector env (DummyVecEnv)
    n_envs = 1
    env_fns = [make_env(price, pv, config) for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Callback logger
    cb = EpisodeRewardLogger(verbose=1)

    # PPO model
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_smart_home_tb/")

    # Train: chỉnh total_timesteps theo nhu cầu (ví dụ 24 steps * 500 episodes = 12000)
    episodes_to_train = 5000
    total_timesteps = T * episodes_to_train
    print(f"Training PPO for {episodes_to_train} episodes (~{total_timesteps} timesteps)...")
    model.learn(total_timesteps=total_timesteps, callback=cb)

    # Lưu model
    model.save("ppo_smart_home")

    # Vẽ reward theo episode (nếu có)
    if len(cb.episode_rewards) == 0:
        print("Không thu được reward nào từ callback. Kiểm tra Monitor wrapper.")
    else:
        plt.figure(figsize=(8,5))
        plt.plot(cb.episode_rewards, marker='o', linestyle='-')
        plt.xlabel("Episode")
        plt.ylabel("Episode reward")
        plt.title("Reward per Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
