import numpy as np
import matplotlib.pyplot as plt

# Load reward logs
r_base = np.load("ppo_baseline_rewards.npy")
r_hybrid_new = np.load("ppo_hybrid_rewards.npy")

# Smoothing (moving average)
def smooth(x, k=50):
    return np.convolve(x, np.ones(k)/k, mode='valid')

plt.figure(figsize=(10,5))

plt.plot(smooth(r_base), label="PPO Baseline", alpha=0.8)
plt.plot(smooth(r_hybrid_new), label="PPO Hybrid (IL + PPO)", alpha=0.8)

plt.title("So sánh đường hội tụ PPO Baseline vs PPO Hybrid")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
