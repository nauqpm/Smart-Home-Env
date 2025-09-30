from smart_home_env import SmartHomeEnv
import numpy as np
import matplotlib.pyplot as plt

# số bước (24 giờ)
T = 24

# tạo dữ liệu test
price = 0.1 + 0.2 * np.random.rand(T)  # giá điện giả định
pv = np.clip(1.5 * np.sin(np.linspace(0, 3.14, T)) + 0.2*np.random.randn(T), 0, None)  # PV giả định

# cấu hình hệ thống
cfg = {
    "critical": [0.3]*T,
    "adjustable": [
        {"P_min": 0.1, "P_max": 1.5, "P_com": 1.2, "alpha": 0.06},
        {"P_min": 0.0, "P_max": 1.2, "P_com": 1.0, "alpha": 0.12}
    ],
    "shiftable_su": [
        {"rate": 0.5, "L": 2, "t_s": 6, "t_f": 20},
        {"rate": 0.6, "L": 1, "t_s": 8, "t_f": 22}
    ],
    "shiftable_si": [
        {"rate": 1.0, "E": 4.0, "t_s": 0, "t_f": 23}
    ],
    "beta": 0.5,
    "battery": {"soc0": 0.5, "soc_min": 0.1, "soc_max": 0.9}
}

# khởi tạo môi trường
env = SmartHomeEnv(price, pv, cfg)

obs = env.reset()
done = False
step = 0
total_reward = 0.0

# log dữ liệu
rewards = []
soc_hist = []
pv_hist = []
load_hist = []

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward

    rewards.append(reward)
    soc_hist.append(env.SOC)
    pv_hist.append(info["P_pv"])
    load_hist.append(info["P_load"])

    step += 1

print("===== Episode kết thúc =====")
print(f"Tổng reward: {total_reward:.3f}")

# Vẽ biểu đồ
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# PV vs Load
axs[0].plot(range(T), pv_hist, label="PV (kW)")
axs[0].plot(range(T), load_hist, label="Load (kW)")
axs[0].set_ylabel("Power (kW)")
axs[0].set_title("PV vs Load")
axs[0].legend()

# SOC
axs[1].plot(range(T), soc_hist, color="orange")
axs[1].set_ylabel("SOC")
axs[1].set_title("Battery SOC")

# Reward
axs[2].bar(range(T), rewards, color="green")
axs[2].set_ylabel("Reward")
axs[2].set_xlabel("Timestep")
axs[2].set_title("Reward per timestep")

plt.tight_layout()
plt.show()
