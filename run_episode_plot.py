import numpy as np
import matplotlib.pyplot as plt
from smart_home_env import SmartHomeEnv

# số bước (24 giờ)
T = 24

# tạo dữ liệu test
np.random.seed(42)
price = 0.1 + 0.2 * np.random.rand(T)  # giá điện giả định
pv = np.clip(5.0 * np.sin(np.linspace(0, 3.14, T)) + 0.3*np.random.randn(T), 0, None)  # PV mạnh

# cấu hình hệ thống
cfg = {
    "critical": [0.3] * T,
    "adjustable": [
        {"P_min": 0.1, "P_max": 1.5, "P_com": 1.2, "alpha": 0.06},
        {"P_min": 0.0, "P_max": 1.2, "P_com": 1.0, "alpha": 0.12}
    ],
    "shiftable_su": [
        {"rate": 0.5, "L": 3, "t_s": 6, "t_f": 20},
        {"rate": 0.6, "L": 2, "t_s": 8, "t_f": 22}
    ],
    "shiftable_si": [{"rate": 1.0, "E": 4.0, "t_s": 0, "t_f": 23}],
    # Thêm C_bat vào cấu hình pin
    "battery": {"soc0": 0.5, "soc_min": 0.1, "soc_max": 0.9, "C_bat": 13.0}
}

# khởi tạo môi trường
env = SmartHomeEnv(price, pv, cfg, forecast_horizon=3)
obs, info = env.reset()
done = False
total_reward = 0.0

# log dữ liệu
rewards = []
soc_hist = []
pv_hist = []
load_hist = []
grid_hist = []

# chạy 1 episode
while not done:
    action = env.action_space.sample()  # Hành động ngẫu nhiên
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

    rewards.append(reward);
    soc_hist.append(info["SOC"])
    pv_hist.append(info["P_pv"]);
    load_hist.append(info["P_load"])
    grid_hist.append(info["P_grid"])

print("===== Episode kết thúc =====")
print(f"Tổng reward: {total_reward:.3f}")
print(f"Tổng chi phí: {env.total_cost:.3f}")

# Vẽ biểu đồ
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# 1. PV vs Load
axs[0].plot(range(T), pv_hist, label="PV (kW)")
axs[0].plot(range(T), load_hist, label="Load (kW)")
axs[0].set_ylabel("Power (kW)")
axs[0].set_title("PV vs Load")
axs[0].legend()

# 2. Battery SOC
axs[1].plot(range(T), soc_hist, color="orange")
axs[1].set_ylabel("SOC")
axs[1].set_title("Battery SOC")

# 3. Grid Power
axs[2].plot(range(T), grid_hist, color="red")
axs[2].set_ylabel("P_grid (kW)")
axs[2].set_title("Power from/to Grid")

# 4. Reward
axs[3].bar(range(T), rewards, color="green")
axs[3].set_ylabel("Reward")
axs[3].set_xlabel("Timestep")
axs[3].set_title("Reward per timestep")

plt.tight_layout()
plt.show()
